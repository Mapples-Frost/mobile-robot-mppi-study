"""Validated dynamics bridge for scalar MPPI rollouts.

The historical planner keeps its original ``dynamics_model=None`` path. This
module is imported lazily only when an experiment explicitly supplies a
prediction model, so the legacy planner does not acquire a PyTorch dependency.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.dynamics.combined_dynamics import CombinedDynamics
from src.dynamics.integrators import integrate_step
from src.dynamics.interfaces import (
    validate_angle_indices,
    validate_derivative,
    validate_model_dimensions,
    validate_state_control,
    validate_time,
    validate_vector,
)
from src.dynamics.nominal_unicycle import NominalUnicycle
from src.dynamics.residual.icode_residual import ICODEResidual
from src.dynamics.residual.mlp_residual import MLPResidual
from src.dynamics.residual.oracle_residual import OracleResidual
from src.learning.checkpointing import (
    create_model_from_checkpoint,
    load_checkpoint,
    normalizers_from_checkpoint,
)
from src.learning.normalization import NormalizerBundle


_PREDICTION_MODES = (
    "nominal",
    "oracle_residual",
    "mlp_residual",
    "icode_residual",
)


def _positive_dt(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("dt must be a positive finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("dt must be a positive finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError("dt must be finite")
    if result <= 0.0:
        raise ValueError("dt must be greater than zero")
    return result


def _integration_method(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("integration_method must be a string")
    result = value.strip().lower()
    if result not in ("euler", "rk4"):
        raise ValueError("integration_method must be 'euler' or 'rk4'")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("{} must be boolean".format(name))
    return bool(value)


def _call_model_step(
    model: Any,
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    method: str,
    time: float,
) -> np.ndarray:
    """Advance one interval, preferring a stateful plant's step contract."""

    operation = getattr(model, "step", None)
    if not callable(operation):
        return integrate_step(
            model,
            state,
            control,
            dt,
            method=method,
            time=time,
        )

    kwargs: Dict[str, Any] = {}
    try:
        parameters = inspect.signature(operation).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs or "method" in parameters:
        kwargs["method"] = method
    if accepts_kwargs or "time" in parameters:
        kwargs["time"] = time
    return operation(state, control, dt, **kwargs)


class MppiDynamicsAdapter:
    """Adapt a validated continuous dynamics model to MPPI list rollouts.

    ``clone_per_rollout=True`` gives every candidate an independent copy of
    mutable plant history. A model-provided ``clone()`` is preferred; a model
    explicitly marked ``is_stateful`` or ``has_control_delay`` is otherwise
    copied with ``deepcopy``. Stateless models are safely reused. Each control
    interval calls the
    model's plant-level ``step`` exactly once when it exists, so a delay queue
    is never advanced once per RK4 stage.
    """

    def __init__(
        self,
        dynamics: Any,
        integration_method: str = "rk4",
        clone_per_rollout: bool = True,
        start_time: float = 0.0,
    ) -> None:
        state_dim, control_dim, angle_indices = validate_model_dimensions(dynamics)
        numeric_start_time = validate_time(start_time)
        self.dynamics = dynamics
        self.model = dynamics
        self.integration_method = _integration_method(integration_method)
        self.clone_per_rollout = _bool(clone_per_rollout, "clone_per_rollout")
        self.start_time = 0.0 if numeric_start_time is None else numeric_start_time
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.angle_indices: Tuple[int, ...] = angle_indices

    def _model_for_rollout(self) -> Any:
        if not self.clone_per_rollout:
            return self.dynamics
        clone = getattr(self.dynamics, "clone", None)
        explicitly_stateful = bool(
            getattr(self.dynamics, "is_stateful", False)
            or getattr(self.dynamics, "has_control_delay", False)
        )
        if callable(clone):
            model = clone()
        elif explicitly_stateful:
            model = copy.deepcopy(self.dynamics)
        else:
            # Nominal and learned CombinedDynamics instances are stateless.
            # Reusing them avoids copying a complete Torch network for every
            # MPPI sample while stateful/delayed models remain isolated above.
            return self.dynamics
        state_dim, control_dim, angle_indices = validate_model_dimensions(model)
        if (
            state_dim != self.state_dim
            or control_dim != self.control_dim
            or angle_indices != self.angle_indices
        ):
            raise ValueError("cloned dynamics dimensions do not match the source model")
        return model

    def rollout(
        self,
        start_state: Any,
        control_sequence: Sequence[Any],
        dt: float,
    ) -> list:
        """Return a finite list of state tuples, including the start state."""

        numeric_dt = _positive_dt(dt)
        model = self._model_for_rollout()
        current_state = validate_vector(start_state, self.state_dim, "start_state")
        trajectory = [tuple(float(value) for value in current_state)]
        current_time = self.start_time
        try:
            controls = iter(control_sequence)
        except TypeError as exc:
            raise TypeError("control_sequence must be iterable") from exc

        for control in controls:
            _, control_array = validate_state_control(model, current_state, control)
            next_state = _call_model_step(
                model,
                current_state,
                control_array,
                numeric_dt,
                self.integration_method,
                current_time,
            )
            current_state = validate_vector(next_state, self.state_dim, "next_state")
            trajectory.append(tuple(float(value) for value in current_state))
            current_time += numeric_dt
        return trajectory

    __call__ = rollout

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Delegate the dynamics protocol for unified factory consumers."""

        state_array, control_array = validate_state_control(self, state, control)
        numeric_time = validate_time(time)
        return validate_derivative(
            self.dynamics.derivative(
                state_array,
                control_array,
                time=numeric_time,
            ),
            self.state_dim,
        )


def _model_registry() -> Mapping[str, Any]:
    """Return the complete allow-list used for checkpoint construction."""

    return {
        "MLPResidual": MLPResidual,
        "MLPResidualModel": MLPResidual,
        "src.dynamics.residual.mlp_residual.MLPResidual": MLPResidual,
        "ICODEResidual": ICODEResidual,
        "ICodeResidual": ICODEResidual,
        "ControlAffineResidual": ICODEResidual,
        "src.dynamics.residual.icode_residual.ICODEResidual": ICODEResidual,
    }


class LearnedResidualDynamics:
    """Expose a normalized PyTorch residual through the NumPy dynamics API."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalizers: NormalizerBundle,
        state_encoder: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        expose_icode_components: bool = False,
    ) -> None:
        if not isinstance(model, (MLPResidual, ICODEResidual)):
            raise TypeError("model must be an explicitly supported MLPResidual or ICODEResidual")
        if not isinstance(normalizers, NormalizerBundle):
            raise TypeError("normalizers must be a NormalizerBundle")
        self.expose_icode_components = _bool(
            expose_icode_components, "expose_icode_components"
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.model.eval()
        self.normalizers = normalizers
        self.state_encoder = (
            normalizers.state_encoder if state_encoder is None else state_encoder
        )
        self.state_dim = int(model.state_dim)
        self.control_dim = int(model.control_dim)
        config = dict(normalizers.encoder_config)
        default_angles = (2,) if self.state_dim == 3 else ()
        self.angle_indices = validate_angle_indices(
            tuple(config.get("angle_indices", default_angles)),
            self.state_dim,
        )
        if normalizers.state.feature_dim != int(model.state_feature_dim):
            raise ValueError("state normalizer dimension does not match learned model")
        if normalizers.control.feature_dim != self.control_dim:
            raise ValueError("control normalizer dimension does not match learned model")
        if normalizers.residual.feature_dim != self.state_dim:
            raise ValueError("residual normalizer dimension does not match learned model")
        self.model_type = str(model.MODEL_TYPE)
        self.checkpoint: Optional[Mapping[str, Any]] = None
        self.checkpoint_path: Optional[Path] = None
        self.last_icode_components: Optional[Dict[str, np.ndarray]] = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
        expose_icode_components: bool = False,
        expected_model_type: Optional[str] = None,
    ) -> "LearnedResidualDynamics":
        """Safely restore a known residual class, encoder, and statistics."""

        checkpoint = load_checkpoint(
            path,
            map_location="cpu",
            strict=strict,
            weights_only=True,
        )
        model = create_model_from_checkpoint(
            checkpoint,
            registry=_model_registry(),
            strict=strict,
            allow_import=False,
        )
        model_type = getattr(model, "MODEL_TYPE", None)
        if expected_model_type is not None and model_type != expected_model_type:
            raise ValueError(
                "checkpoint contains {!r}, expected {!r}".format(
                    model_type, expected_model_type
                )
            )
        normalizers = normalizers_from_checkpoint(checkpoint)
        instance = cls(
            model=model,
            normalizers=normalizers,
            device=device,
            expose_icode_components=expose_icode_components,
        )
        instance.checkpoint = checkpoint
        instance.checkpoint_path = Path(path).resolve()
        return instance

    def _encoded_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_encoder is None:
            encoded = state
        else:
            operation = getattr(self.state_encoder, "encode", None)
            if operation is None:
                operation = self.state_encoder if callable(self.state_encoder) else None
            if operation is None:
                raise TypeError("state_encoder must be callable or provide encode()")
            encoded = operation(state)
        return validate_vector(
            encoded,
            int(self.model.state_feature_dim),
            "encoded state",
        )

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _normalized_inputs(
        self, state: Any, control: Any
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        state_array, control_array = validate_state_control(self, state, control)
        encoded = self._encoded_state(state_array)
        normalized_state = self.normalizers.state.transform(encoded)
        normalized_control = self.normalizers.control.transform(control_array)
        dtype = self._model_dtype()
        state_tensor = torch.as_tensor(
            normalized_state, dtype=dtype, device=self.device
        )
        control_tensor = torch.as_tensor(
            normalized_control, dtype=dtype, device=self.device
        )
        return state_array, control_array, state_tensor, control_tensor

    @staticmethod
    def _numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("{} must be a torch.Tensor".format(name))
        array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError("{} contains NaN or Inf".format(name))
        return np.asarray(array, dtype=np.float64)

    def _physical_residual(
        self,
        state: Any,
        control: Any,
        include_components: bool,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        _, control_array, state_tensor, control_tensor = self._normalized_inputs(
            state, control
        )
        with torch.no_grad():
            if include_components:
                if not isinstance(self.model, ICODEResidual):
                    raise TypeError("physical components are available only for ICODEResidual")
                output = self.model(
                    state_tensor, control_tensor, return_components=True
                )
                assert isinstance(output, dict)
                normalized_residual = output["residual"]
            else:
                output = self.model(state_tensor, control_tensor)
                if isinstance(output, dict):
                    output = output.get("residual")
                normalized_residual = output
        normalized = self._numpy(normalized_residual, "normalized residual")
        residual = validate_derivative(
            self.normalizers.residual.inverse_transform(normalized),
            self.state_dim,
        )
        if not include_components:
            return residual, None

        assert isinstance(output, dict)
        drift_normalized = self._numpy(output["drift"], "normalized drift")
        gain_normalized = self._numpy(output["gain"], "normalized gain")
        residual_scale = self.normalizers.residual.scale
        residual_mean = self.normalizers.residual.mean
        control_mean = self.normalizers.control.mean
        control_scale = self.normalizers.control.scale
        gain = residual_scale[:, None] * gain_normalized / control_scale[None, :]
        # Express the debug decomposition in raw control coordinates:
        #   residual(u) = drift_physical + gain_physical @ u.
        # The network consumes (u - mean_u) / scale_u, so its offset must be
        # folded into the physical drift rather than into the contribution.
        drift = (
            drift_normalized * residual_scale
            + residual_mean
            - np.matmul(gain, control_mean)
        )
        control_contribution = np.matmul(gain, control_array)
        components = {
            "residual": residual.copy(),
            "drift": validate_derivative(drift, self.state_dim),
            "gain": np.asarray(gain, dtype=np.float64),
            "control_contribution": validate_derivative(
                control_contribution, self.state_dim
            ),
            "control": control_array.copy(),
            "normalized_control": self._numpy(control_tensor, "normalized control"),
        }
        for name, value in components.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    "physical ICODE component {} contains NaN or Inf".format(name)
                )
        if components["gain"].shape != (self.state_dim, self.control_dim):
            raise ValueError("physical ICODE gain has an invalid shape")
        if not np.allclose(
            np.matmul(components["gain"], components["control"]),
            components["control_contribution"],
            rtol=1.0e-7,
            atol=1.0e-9,
        ):
            raise ValueError("physical ICODE gain/control contribution is inconsistent")
        if not np.allclose(
            components["drift"] + components["control_contribution"],
            components["residual"],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError("physical ICODE components do not reconstruct the residual")
        return residual, components

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        validate_time(time)
        residual, components = self._physical_residual(
            state,
            control,
            include_components=self.expose_icode_components,
        )
        self.last_icode_components = components
        return residual

    def icode_physical_components(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Return physical drift/gain/control terms for ICODE debugging."""

        validate_time(time)
        _, components = self._physical_residual(
            state, control, include_components=True
        )
        assert components is not None
        self.last_icode_components = components
        return {name: value.copy() for name, value in components.items()}

    derivative_components = icode_physical_components


def build_prediction_dynamics(
    mode: str = "nominal",
    checkpoint_path: Optional[Union[str, Path]] = None,
    true_dynamics: Optional[Any] = None,
    nominal_dynamics: Optional[Any] = None,
    integration_method: str = "rk4",
    clone_per_rollout: bool = True,
    start_time: float = 0.0,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
    expose_icode_components: bool = False,
) -> MppiDynamicsAdapter:
    """Build one unified MPPI adapter for the four prediction ablations.

    ``oracle_residual`` is deliberately an upper-bound/interface ablation: it
    requires the caller to pass the true dynamics explicitly. Learned modes
    never receive the true plant and restore only allow-listed model classes.
    """

    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    normalized_mode = mode.strip().lower()
    if normalized_mode not in _PREDICTION_MODES:
        raise ValueError(
            "unknown prediction mode {!r}; expected one of {}".format(
                mode, _PREDICTION_MODES
            )
        )
    nominal = NominalUnicycle() if nominal_dynamics is None else nominal_dynamics
    validate_model_dimensions(nominal)

    if normalized_mode == "nominal":
        if checkpoint_path is not None:
            raise ValueError("nominal mode does not accept a checkpoint")
        if true_dynamics is not None:
            raise ValueError("nominal mode does not accept true_dynamics")
        dynamics = nominal
    elif normalized_mode == "oracle_residual":
        if checkpoint_path is not None:
            raise ValueError("oracle_residual mode does not accept a checkpoint")
        if true_dynamics is None:
            raise ValueError(
                "oracle_residual is an explicit ablation and requires true_dynamics"
            )
        warnings.warn(
            "oracle_residual exposes the true dynamics to the planner; ablation only",
            RuntimeWarning,
            stacklevel=2,
        )
        dynamics = CombinedDynamics(
            nominal, OracleResidual(true_dynamics, nominal)
        )
    else:
        if true_dynamics is not None:
            raise ValueError("learned prediction modes must not receive true_dynamics")
        if checkpoint_path is None:
            raise ValueError("{} mode requires checkpoint_path".format(normalized_mode))
        residual = LearnedResidualDynamics.from_checkpoint(
            checkpoint_path,
            device=device,
            strict=strict,
            expose_icode_components=expose_icode_components,
            expected_model_type=normalized_mode,
        )
        dynamics = CombinedDynamics(nominal, residual)

    adapter = MppiDynamicsAdapter(
        dynamics=dynamics,
        integration_method=integration_method,
        clone_per_rollout=clone_per_rollout,
        start_time=start_time,
    )
    adapter.prediction_mode = normalized_mode
    adapter.is_oracle_ablation = normalized_mode == "oracle_residual"
    return adapter


__all__ = [
    "LearnedResidualDynamics",
    "MppiDynamicsAdapter",
    "build_prediction_dynamics",
]
