"""Differentiable residual-dynamics prediction and training losses.

The learned models operate in normalized feature coordinates, while the
nominal dynamics, integration, and state losses operate in physical unicycle
coordinates.  Keeping those conversions here gives training and rollout code
one implementation of the boundary between the two spaces.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .normalization import NormalizerBundle


NormalizerSource = Union[NormalizerBundle, Mapping[str, Any], "TorchNormalizerBundle"]
DerivativeFunction = Callable[[Tensor, Tensor], Tensor]


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point():
        raise TypeError("normalizer dtype must be floating point")
    return dtype


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a finite nonnegative scalar".format(name))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and nonnegative".format(name))
    return result


def _normalizer_section(state: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = state.get(name)
    if not isinstance(section, Mapping):
        raise ValueError("normalizer state is missing the '{}' section".format(name))
    if "mean" not in section or "scale" not in section:
        raise ValueError("normalizer '{}' must contain mean and scale".format(name))
    return section


def _vector_tensor(
    values: Any,
    name: str,
    device: Optional[Union[str, torch.device]],
    dtype: torch.dtype,
) -> Tensor:
    try:
        tensor = torch.as_tensor(values, device=device, dtype=dtype).detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("{} must be a numeric feature vector".format(name)) from exc
    if tensor.ndim != 1 or tensor.numel() <= 0:
        raise ValueError("{} must be a nonempty feature vector".format(name))
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("{} contains NaN or Inf".format(name))
    return tensor


class TorchNormalizerBundle(nn.Module):
    """Torch buffers reconstructed from a train-fitted ``NormalizerBundle``.

    ``source`` may be the NumPy bundle itself, its JSON-safe ``state_dict``, or
    the flat state of another :class:`TorchNormalizerBundle`.  Statistics are
    registered as buffers, so ``.to(device)`` keeps them with a model without
    ever making them trainable parameters.
    """

    def __init__(
        self,
        source: NormalizerSource,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dtype = _floating_dtype(dtype)

        if isinstance(source, TorchNormalizerBundle):
            encoder_config = dict(source.encoder_config)
            vectors = {
                "state_mean": source.state_mean,
                "state_scale": source.state_scale,
                "control_mean": source.control_mean,
                "control_scale": source.control_scale,
                "residual_mean": source.residual_mean,
                "residual_scale": source.residual_scale,
            }
        else:
            if isinstance(source, NormalizerBundle):
                state = source.state_dict()
            elif isinstance(source, Mapping):
                state = source
            else:
                raise TypeError(
                    "source must be a NormalizerBundle or normalizer state mapping"
                )

            if all(
                key in state
                for key in (
                    "state_mean",
                    "state_scale",
                    "control_mean",
                    "control_scale",
                    "residual_mean",
                    "residual_scale",
                )
            ):
                extra = state.get("_extra_state", {})
                if not isinstance(extra, Mapping):
                    extra = {}
                config_value = state.get(
                    "encoder_config", extra.get("encoder_config", {"mode": "raw"})
                )
                encoder_config = dict(config_value)
                vectors = {
                    key: state[key]
                    for key in (
                        "state_mean",
                        "state_scale",
                        "control_mean",
                        "control_scale",
                        "residual_mean",
                        "residual_scale",
                    )
                }
            else:
                state_section = _normalizer_section(state, "state")
                control_section = _normalizer_section(state, "control")
                residual_section = _normalizer_section(state, "residual")
                config_value = state.get("encoder_config", {"mode": "raw"})
                if not isinstance(config_value, Mapping):
                    raise TypeError("encoder_config must be a mapping")
                encoder_config = dict(config_value)
                vectors = {
                    "state_mean": state_section["mean"],
                    "state_scale": state_section["scale"],
                    "control_mean": control_section["mean"],
                    "control_scale": control_section["scale"],
                    "residual_mean": residual_section["mean"],
                    "residual_scale": residual_section["scale"],
                }

        if not isinstance(encoder_config, Mapping):
            raise TypeError("encoder_config must be a mapping")
        self.encoder_config: Dict[str, Any] = dict(encoder_config)
        for name, values in vectors.items():
            self.register_buffer(
                name, _vector_tensor(values, name, device=device, dtype=dtype)
            )

        for mean_name, scale_name in (
            ("state_mean", "state_scale"),
            ("control_mean", "control_scale"),
            ("residual_mean", "residual_scale"),
        ):
            mean = getattr(self, mean_name)
            scale = getattr(self, scale_name)
            if mean.shape != scale.shape:
                raise ValueError("{} and {} shapes must match".format(mean_name, scale_name))
            if not bool((scale > 0.0).all().item()):
                raise ValueError("{} must be strictly positive".format(scale_name))

        configured_output_dim = self.encoder_config.get("output_dim")
        if configured_output_dim is not None and int(configured_output_dim) != self.state_dim:
            raise ValueError(
                "encoder output_dim {} does not match state normalizer dimension {}".format(
                    configured_output_dim, self.state_dim
                )
            )

    @classmethod
    def from_bundle(
        cls,
        bundle: NormalizerBundle,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TorchNormalizerBundle":
        return cls(bundle, device=device, dtype=dtype)

    from_normalizer_bundle = from_bundle

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TorchNormalizerBundle":
        return cls(state, device=device, dtype=dtype)

    @classmethod
    def identity(
        cls,
        state_feature_dim: int = 3,
        control_dim: int = 2,
        residual_dim: int = 3,
        encoder_config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TorchNormalizerBundle":
        """Build an identity bundle for analytic checks and controlled ablations."""

        dimensions = (state_feature_dim, control_dim, residual_dim)
        if any(isinstance(value, bool) or int(value) <= 0 for value in dimensions):
            raise ValueError("identity normalizer dimensions must be positive")
        config = dict(encoder_config or {"mode": "raw", "output_dim": state_feature_dim})
        state = {
            "encoder_config": config,
            "state": {
                "mean": [0.0] * int(state_feature_dim),
                "scale": [1.0] * int(state_feature_dim),
            },
            "control": {
                "mean": [0.0] * int(control_dim),
                "scale": [1.0] * int(control_dim),
            },
            "residual": {
                "mean": [0.0] * int(residual_dim),
                "scale": [1.0] * int(residual_dim),
            },
        }
        return cls(state, device=device, dtype=dtype)

    @property
    def state_dim(self) -> int:
        """Encoded state-feature dimension (not necessarily physical state size)."""

        return int(self.state_mean.numel())

    @property
    def state_feature_dim(self) -> int:
        return self.state_dim

    @property
    def control_dim(self) -> int:
        return int(self.control_mean.numel())

    @property
    def residual_dim(self) -> int:
        return int(self.residual_mean.numel())

    @property
    def device(self) -> torch.device:
        return self.state_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.state_mean.dtype

    def get_extra_state(self) -> Dict[str, Any]:
        return {"encoder_config": dict(self.encoder_config)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        config = state.get("encoder_config", {"mode": "raw"})
        if not isinstance(config, Mapping):
            raise TypeError("encoder_config must be a mapping")
        self.encoder_config = dict(config)

    @staticmethod
    def _stats_like(mean: Tensor, scale: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            mean.to(device=values.device, dtype=values.dtype),
            scale.to(device=values.device, dtype=values.dtype),
        )

    def encode_state(self, states: Tensor) -> Tensor:
        return encode_state_torch(states, self.encoder_config)

    def normalize_state(self, states: Tensor, pre_encoded: bool = False) -> Tensor:
        encoded = states if pre_encoded else self.encode_state(states)
        mean, scale = self._stats_like(self.state_mean, self.state_scale, encoded)
        if encoded.shape[-1] != mean.numel():
            raise ValueError(
                "encoded states last dimension must be {}, got {}".format(
                    mean.numel(), encoded.shape[-1]
                )
            )
        return (encoded - mean) / scale

    transform_state = normalize_state

    def normalize_control(self, controls: Tensor) -> Tensor:
        if not isinstance(controls, Tensor) or controls.ndim < 1:
            raise TypeError("controls must be a torch.Tensor with a feature dimension")
        mean, scale = self._stats_like(self.control_mean, self.control_scale, controls)
        if controls.shape[-1] != mean.numel():
            raise ValueError(
                "controls last dimension must be {}, got {}".format(
                    mean.numel(), controls.shape[-1]
                )
            )
        return (controls - mean) / scale

    transform_control = normalize_control

    def normalize_residual(self, residuals: Tensor) -> Tensor:
        if not isinstance(residuals, Tensor) or residuals.ndim < 1:
            raise TypeError("residuals must be a torch.Tensor with a feature dimension")
        mean, scale = self._stats_like(self.residual_mean, self.residual_scale, residuals)
        if residuals.shape[-1] != mean.numel():
            raise ValueError(
                "residuals last dimension must be {}, got {}".format(
                    mean.numel(), residuals.shape[-1]
                )
            )
        return (residuals - mean) / scale

    transform_residual = normalize_residual

    def inverse_residual(self, normalized_residuals: Tensor) -> Tensor:
        if not isinstance(normalized_residuals, Tensor) or normalized_residuals.ndim < 1:
            raise TypeError(
                "normalized_residuals must be a torch.Tensor with a feature dimension"
            )
        mean, scale = self._stats_like(
            self.residual_mean, self.residual_scale, normalized_residuals
        )
        if normalized_residuals.shape[-1] != mean.numel():
            raise ValueError(
                "normalized residual last dimension must be {}, got {}".format(
                    mean.numel(), normalized_residuals.shape[-1]
                )
            )
        return normalized_residuals * scale + mean

    inverse_transform_residual = inverse_residual


def _encoder_configuration(
    encoder_config: Optional[Any],
    mode: Optional[str],
    state_dim: int,
    angle_indices: Optional[Sequence[int]],
) -> Tuple[str, int, Tuple[int, ...]]:
    if encoder_config is None:
        config: Mapping[str, Any] = {}
    elif isinstance(encoder_config, str):
        config = {"mode": encoder_config}
    elif isinstance(encoder_config, Mapping):
        config = encoder_config
    elif hasattr(encoder_config, "to_config"):
        config = encoder_config.to_config()
    elif isinstance(encoder_config, TorchNormalizerBundle):
        config = encoder_config.encoder_config
    else:
        raise TypeError("encoder_config must be a mode, mapping, or StateEncoder")

    selected_mode = mode if mode is not None else config.get("mode", "raw")
    if not isinstance(selected_mode, str):
        raise TypeError("state encoding mode must be a string")
    selected_mode = selected_mode.strip().lower()
    if selected_mode in ("pre_encoded", "external"):
        selected_mode = "raw"
    if selected_mode not in ("raw", "sincos"):
        raise ValueError("unsupported state encoding {!r}".format(selected_mode))

    configured_state_dim = config.get("state_dim", state_dim)
    if isinstance(configured_state_dim, bool) or int(configured_state_dim) <= 0:
        raise ValueError("state_dim must be positive")
    physical_dim = int(configured_state_dim)
    if physical_dim != state_dim:
        raise ValueError(
            "states last dimension must be {}, got {}".format(physical_dim, state_dim)
        )

    configured_angles = angle_indices
    if configured_angles is None:
        configured_angles = config.get("angle_indices")
    if configured_angles is None:
        configured_angles = (2,) if physical_dim == 3 and selected_mode == "sincos" else ()
    if isinstance(configured_angles, (str, bytes)):
        raise TypeError("angle_indices must be a sequence of integers")
    result = []
    for value in configured_angles:
        if isinstance(value, bool) or int(value) != value:
            raise TypeError("angle_indices must contain integers")
        index = int(value)
        if index < 0 or index >= physical_dim or index in result:
            raise ValueError("angle_indices contains an invalid or duplicate index")
        result.append(index)
    return selected_mode, physical_dim, tuple(result)


def encode_state_torch(
    states: Tensor,
    encoder_config: Optional[Any] = None,
    mode: Optional[str] = None,
    angle_indices: Optional[Sequence[int]] = None,
) -> Tensor:
    """Differentiably encode raw states using ``raw`` or periodic ``sincos``.

    Leading dimensions are preserved.  In ``sincos`` mode, each configured
    angular component is replaced in place by ``sin(angle), cos(angle)``.
    """

    if not isinstance(states, Tensor):
        raise TypeError("states must be a torch.Tensor")
    if states.ndim < 1 or not states.is_floating_point():
        raise ValueError("states must be a floating tensor with a feature dimension")
    selected_mode, state_dim, angles = _encoder_configuration(
        encoder_config, mode, int(states.shape[-1]), angle_indices
    )
    if selected_mode == "raw":
        return states

    angle_set = set(angles)
    features = []
    for index in range(state_dim):
        component = states[..., index : index + 1]
        if index in angle_set:
            features.extend((torch.sin(component), torch.cos(component)))
        else:
            features.append(component)
    return torch.cat(features, dim=-1)


def torch_nominal_unicycle_derivative(state: Tensor, control: Tensor) -> Tensor:
    """Return ``[v cos(theta), v sin(theta), omega]`` with batch preservation."""

    if not isinstance(state, Tensor) or not isinstance(control, Tensor):
        raise TypeError("state and control must be torch tensors")
    if state.ndim < 1 or control.ndim != state.ndim:
        raise ValueError("state and control must have matching batched/unbatched ranks")
    if state.shape[:-1] != control.shape[:-1]:
        raise ValueError("state and control leading dimensions must match")
    if state.shape[-1] != 3 or control.shape[-1] != 2:
        raise ValueError("unicycle state/control dimensions must be 3 and 2")
    if state.device != control.device or state.dtype != control.dtype:
        raise ValueError("state and control must share device and dtype")
    theta = state[..., 2]
    velocity = control[..., 0]
    yaw_rate = control[..., 1]
    return torch.stack(
        (velocity * torch.cos(theta), velocity * torch.sin(theta), yaw_rate), dim=-1
    )


def _torch_normalizers(normalizers: NormalizerSource, reference: Tensor) -> TorchNormalizerBundle:
    if isinstance(normalizers, TorchNormalizerBundle):
        return normalizers
    return TorchNormalizerBundle(normalizers, device=reference.device, dtype=reference.dtype)


def predict_physical_residual(
    model: nn.Module,
    state: Tensor,
    control: Tensor,
    normalizers: NormalizerSource,
) -> Tensor:
    """Run a normalized residual model and return raw derivative coordinates."""

    bundle = _torch_normalizers(normalizers, state)
    normalized_state = bundle.normalize_state(state)
    normalized_control = bundle.normalize_control(control)
    prediction = model(normalized_state, normalized_control)
    if isinstance(prediction, Mapping):
        prediction = prediction.get("residual")
    if not isinstance(prediction, Tensor):
        raise TypeError("residual model must return a tensor")
    if prediction.shape[:-1] != state.shape[:-1]:
        raise ValueError("residual prediction batch shape must match state")
    if prediction.shape[-1] != bundle.residual_dim:
        raise ValueError(
            "residual prediction last dimension must be {}, got {}".format(
                bundle.residual_dim, prediction.shape[-1]
            )
        )
    return bundle.inverse_residual(prediction)


def combined_unicycle_derivative(
    model: nn.Module,
    state: Tensor,
    control: Tensor,
    normalizers: NormalizerSource,
) -> Tensor:
    """Add the learned physical residual to protected nominal dynamics."""

    return torch_nominal_unicycle_derivative(state, control) + predict_physical_residual(
        model, state, control, normalizers
    )


# Explicit shorter aliases used by trainer and evaluation code.
combined_derivative = combined_unicycle_derivative
torch_combined_derivative = combined_unicycle_derivative


def wrap_state_angles_torch(
    state: Tensor, angle_indices: Sequence[int] = (2,)
) -> Tensor:
    """Wrap selected components to ``[-pi, pi]`` without breaking gradients."""

    if not isinstance(state, Tensor) or state.ndim < 1:
        raise TypeError("state must be a torch tensor with a feature dimension")
    indices = set()
    for value in angle_indices:
        if isinstance(value, bool) or int(value) != value:
            raise TypeError("angle_indices must contain integers")
        index = int(value)
        if index < 0 or index >= state.shape[-1] or index in indices:
            raise ValueError("angle_indices contains an invalid or duplicate index")
        indices.add(index)
    if not indices:
        return state
    return torch.stack(
        [
            torch.atan2(torch.sin(state[..., index]), torch.cos(state[..., index]))
            if index in indices
            else state[..., index]
            for index in range(state.shape[-1])
        ],
        dim=-1,
    )


def _dt_multiplier(dt: Any, state: Tensor) -> Tensor:
    if isinstance(dt, bool):
        raise TypeError("dt must be a positive scalar or batch tensor")
    dt_tensor = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    if not bool(torch.isfinite(dt_tensor).all().item()):
        raise ValueError("dt contains NaN or Inf")
    if not bool((dt_tensor > 0.0).all().item()):
        raise ValueError("dt must be strictly positive")
    leading_shape = tuple(state.shape[:-1])
    if dt_tensor.ndim == 0:
        return dt_tensor
    if dt_tensor.numel() == 1:
        return dt_tensor.reshape(())
    if tuple(dt_tensor.shape) == leading_shape:
        return dt_tensor.unsqueeze(-1)
    if tuple(dt_tensor.shape) == leading_shape + (1,):
        return dt_tensor
    raise ValueError(
        "dt shape {} must be scalar or match state batch shape {}".format(
            tuple(dt_tensor.shape), leading_shape
        )
    )


def _evaluate_derivative(
    derivative: DerivativeFunction, state: Tensor, control: Tensor
) -> Tensor:
    result = derivative(state, control)
    if not isinstance(result, Tensor):
        raise TypeError("derivative function must return a torch.Tensor")
    if result.shape != state.shape:
        raise ValueError(
            "derivative shape {} must match state shape {}".format(
                tuple(result.shape), tuple(state.shape)
            )
        )
    return result


def torch_integrate_step(
    derivative: DerivativeFunction,
    state: Tensor,
    control: Tensor,
    dt: Any,
    method: str = "rk4",
    angle_indices: Sequence[int] = (2,),
    wrap_angles: bool = True,
) -> Tensor:
    """Differentiably integrate one zero-order-held control interval.

    ``dt`` may be scalar or match the state's leading batch dimensions.  RK4
    evaluates all four stages with the exact same control (zero-order hold),
    and angles are wrapped only after the complete step.
    """

    if not isinstance(state, Tensor) or not isinstance(control, Tensor):
        raise TypeError("state and control must be torch tensors")
    if state.ndim < 1 or control.ndim != state.ndim:
        raise ValueError("state and control must have matching ranks")
    if state.shape[:-1] != control.shape[:-1]:
        raise ValueError("state and control batch shapes must match")
    if not isinstance(method, str):
        raise TypeError("integration method must be a string")
    selected_method = method.strip().lower()
    if selected_method not in ("euler", "rk4"):
        raise ValueError("integration method must be 'euler' or 'rk4'")
    if not isinstance(wrap_angles, bool):
        raise TypeError("wrap_angles must be boolean")
    step = _dt_multiplier(dt, state)

    k1 = _evaluate_derivative(derivative, state, control)
    if selected_method == "euler":
        next_state = state + step * k1
    else:
        half_step = 0.5 * step
        k2 = _evaluate_derivative(derivative, state + half_step * k1, control)
        k3 = _evaluate_derivative(derivative, state + half_step * k2, control)
        k4 = _evaluate_derivative(derivative, state + step * k3, control)
        next_state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if wrap_angles:
        return wrap_state_angles_torch(next_state, angle_indices)
    return next_state


def wrapped_state_error(
    prediction: Tensor,
    target: Tensor,
    angle_indices: Sequence[int] = (2,),
) -> Tensor:
    """Return prediction-minus-target error with periodic components wrapped."""

    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    error = prediction - target
    return wrap_state_angles_torch(error, angle_indices)


def _feature_mse(error: Tensor, feature_weights: Optional[Any] = None) -> Tensor:
    squared = error.square()
    if feature_weights is None:
        return squared.mean(dim=-1)
    weights = torch.as_tensor(feature_weights, device=error.device, dtype=error.dtype)
    if weights.ndim != 1 or weights.shape[0] != error.shape[-1]:
        raise ValueError("feature weights must match the final error dimension")
    if not bool(torch.isfinite(weights).all().item()) or not bool((weights >= 0).all().item()):
        raise ValueError("feature weights must be finite and nonnegative")
    denominator = weights.sum()
    if not bool((denominator > 0).item()):
        raise ValueError("at least one feature weight must be positive")
    return (squared * weights).sum(dim=-1) / denominator


def residual_derivative_loss(
    model: nn.Module,
    state: Tensor,
    control: Tensor,
    residual_target: Tensor,
    normalizers: NormalizerSource,
    residual_weights: Optional[Any] = None,
) -> Tensor:
    """Supervise the physical residual derivative target."""

    prediction = predict_physical_residual(model, state, control, normalizers)
    if prediction.shape != residual_target.shape:
        raise ValueError("residual prediction and target shapes must match")
    return _feature_mse(prediction - residual_target, residual_weights).mean()


derivative_loss = residual_derivative_loss


def one_step_prediction_loss(
    model: nn.Module,
    state: Tensor,
    control: Tensor,
    dt: Any,
    target_next_state: Tensor,
    normalizers: NormalizerSource,
    method: str = "rk4",
    state_weights: Optional[Any] = None,
    angle_indices: Sequence[int] = (2,),
    return_prediction: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Combined nominal-plus-residual one-step state prediction loss."""

    derivative = lambda current, held_control: combined_unicycle_derivative(
        model, current, held_control, normalizers
    )
    prediction = torch_integrate_step(
        derivative,
        state,
        control,
        dt,
        method=method,
        angle_indices=angle_indices,
    )
    error = wrapped_state_error(prediction, target_next_state, angle_indices)
    loss = _feature_mse(error, state_weights).mean()
    if return_prediction:
        return loss, prediction
    return loss


one_step_loss = one_step_prediction_loss


def _rollout_dt(
    dt: Any, step_index: int, horizon: int, initial_state: Tensor
) -> Any:
    dt_tensor = torch.as_tensor(dt, device=initial_state.device, dtype=initial_state.dtype)
    if dt_tensor.ndim == 0 or dt_tensor.numel() == 1:
        return dt_tensor
    if initial_state.ndim == 1:
        if tuple(dt_tensor.shape) == (horizon,):
            return dt_tensor[step_index]
        if tuple(dt_tensor.shape) == (horizon, 1):
            return dt_tensor[step_index]
    else:
        batch = initial_state.shape[0]
        if tuple(dt_tensor.shape) == (batch, horizon):
            return dt_tensor[:, step_index]
        if tuple(dt_tensor.shape) == (batch, horizon, 1):
            return dt_tensor[:, step_index, :]
        if tuple(dt_tensor.shape) == (horizon,):
            return dt_tensor[step_index]
        if tuple(dt_tensor.shape) in ((batch,), (batch, 1)):
            return dt_tensor
    raise ValueError("rollout dt must be scalar, per-step, or per-batch/per-step")


def rollout_combined_dynamics(
    model: nn.Module,
    initial_state: Tensor,
    controls: Tensor,
    dt: Any,
    normalizers: NormalizerSource,
    method: str = "rk4",
    angle_indices: Sequence[int] = (2,),
) -> Tensor:
    """Roll out combined dynamics and include the initial state in the result."""

    if not isinstance(initial_state, Tensor) or not isinstance(controls, Tensor):
        raise TypeError("initial_state and controls must be torch tensors")
    if initial_state.ndim not in (1, 2):
        raise ValueError("initial_state must be unbatched or batched")
    if controls.ndim != initial_state.ndim + 1:
        raise ValueError("controls must add one horizon dimension to initial_state")
    if initial_state.ndim == 2 and controls.shape[0] != initial_state.shape[0]:
        raise ValueError("initial_state and controls batch sizes must match")
    if controls.shape[-2] <= 0:
        raise ValueError("rollout horizon must be positive")

    horizon = int(controls.shape[-2])
    derivative = lambda current, held_control: combined_unicycle_derivative(
        model, current, held_control, normalizers
    )
    current = initial_state
    trajectory = [current]
    for step_index in range(horizon):
        step_control = controls[..., step_index, :]
        step_dt = _rollout_dt(dt, step_index, horizon, initial_state)
        current = torch_integrate_step(
            derivative,
            current,
            step_control,
            step_dt,
            method=method,
            angle_indices=angle_indices,
        )
        trajectory.append(current)
    return torch.stack(trajectory, dim=-2)


def multistep_rollout_loss(
    model: nn.Module,
    initial_state: Tensor,
    controls: Tensor,
    dt: Any,
    target_states: Tensor,
    normalizers: NormalizerSource,
    method: str = "rk4",
    state_weights: Optional[Any] = None,
    horizon_weights: Optional[Any] = None,
    angle_indices: Sequence[int] = (2,),
) -> Tuple[Tensor, Tensor]:
    """Return weighted H-step MSE and the trajectory including its initial state.

    ``target_states`` may contain exactly the H predicted states or H+1 states
    including the initial state.  No target is silently truncated beyond this
    explicitly supported convention.
    """

    trajectory = rollout_combined_dynamics(
        model,
        initial_state,
        controls,
        dt,
        normalizers,
        method=method,
        angle_indices=angle_indices,
    )
    horizon = controls.shape[-2]
    if target_states.ndim != trajectory.ndim:
        raise ValueError("target_states rank must match rollout trajectory rank")
    if target_states.shape[:-2] != trajectory.shape[:-2] or target_states.shape[-1] != trajectory.shape[-1]:
        raise ValueError("target_states batch and state dimensions must match trajectory")
    if target_states.shape[-2] == horizon:
        prediction = trajectory[..., 1:, :]
    elif target_states.shape[-2] == horizon + 1:
        prediction = trajectory
    else:
        raise ValueError("target_states horizon must be H or H+1")

    error = wrapped_state_error(prediction, target_states, angle_indices)
    per_step = _feature_mse(error, state_weights)
    if horizon_weights is None:
        loss = per_step.mean()
    else:
        weights = torch.as_tensor(
            horizon_weights, device=per_step.device, dtype=per_step.dtype
        )
        expected_steps = int(per_step.shape[-1])
        if weights.ndim != 1 or weights.shape[0] != expected_steps:
            raise ValueError("horizon_weights must match the target horizon")
        if not bool(torch.isfinite(weights).all().item()) or not bool((weights >= 0).all().item()):
            raise ValueError("horizon_weights must be finite and nonnegative")
        denominator = weights.sum()
        if not bool((denominator > 0).item()):
            raise ValueError("at least one horizon weight must be positive")
        loss = ((per_step * weights).sum(dim=-1) / denominator).mean()
    return loss, trajectory


rollout_loss = multistep_rollout_loss


def parameter_regularization(model: nn.Module) -> Tensor:
    """Mean squared parameter value, or a device-local zero for parameterless models."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return torch.zeros((), dtype=torch.get_default_dtype())
    numerator = sum(parameter.square().sum() for parameter in parameters)
    count = sum(parameter.numel() for parameter in parameters)
    return numerator / float(count)


def _looks_like_normalizers(value: Any) -> bool:
    if isinstance(value, (NormalizerBundle, TorchNormalizerBundle)):
        return True
    return isinstance(value, Mapping) and (
        "state" in value or "state_mean" in value
    )


class CompositeResidualLoss(nn.Module):
    """Weighted derivative, one-step, rollout, and parameter objectives.

    Construction supports both ``CompositeResidualLoss(model, normalizers)``
    followed by ``criterion(batch)`` and ``CompositeResidualLoss(normalizers)``
    followed by ``criterion(model, batch)``.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        normalizers: Optional[NormalizerSource] = None,
        lambda_residual: float = 1.0,
        lambda_one_step: float = 1.0,
        lambda_multistep: float = 1.0,
        lambda_reg: float = 0.0,
        method: str = "rk4",
        state_weights: Optional[Any] = None,
        residual_weights: Optional[Any] = None,
        horizon_weights: Optional[Any] = None,
        angle_indices: Sequence[int] = (2,),
        lambda_regularization: Optional[float] = None,
    ) -> None:
        super().__init__()
        if normalizers is None and _looks_like_normalizers(model):
            normalizers = model  # type: ignore[assignment]
            model = None
        if normalizers is None:
            raise ValueError("normalizers are required")
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        device = None
        dtype = torch.float32
        if model is not None:
            first_parameter = next(model.parameters(), None)
            if first_parameter is not None:
                device = first_parameter.device
                dtype = first_parameter.dtype
        self.model = model
        self.normalizers = (
            normalizers
            if isinstance(normalizers, TorchNormalizerBundle)
            else TorchNormalizerBundle(normalizers, device=device, dtype=dtype)
        )
        self.lambda_residual = _finite_nonnegative(lambda_residual, "lambda_residual")
        self.lambda_one_step = _finite_nonnegative(lambda_one_step, "lambda_one_step")
        self.lambda_multistep = _finite_nonnegative(
            lambda_multistep, "lambda_multistep"
        )
        if lambda_regularization is not None:
            lambda_reg = lambda_regularization
        self.lambda_reg = _finite_nonnegative(lambda_reg, "lambda_reg")
        if not isinstance(method, str) or method.strip().lower() not in ("euler", "rk4"):
            raise ValueError("method must be 'euler' or 'rk4'")
        self.method = method.strip().lower()
        self.angle_indices = tuple(int(index) for index in angle_indices)
        self.register_buffer(
            "state_weights",
            None if state_weights is None else torch.as_tensor(state_weights, dtype=dtype),
        )
        self.register_buffer(
            "residual_weights",
            None
            if residual_weights is None
            else torch.as_tensor(residual_weights, dtype=dtype),
        )
        self.register_buffer(
            "horizon_weights",
            None
            if horizon_weights is None
            else torch.as_tensor(horizon_weights, dtype=dtype),
        )

    @staticmethod
    def _required(batch: Mapping[str, Tensor], keys: Sequence[str], name: str) -> None:
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError("{} batch is missing {}".format(name, ", ".join(missing)))

    @staticmethod
    def _zero(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.new_zeros(())
        for value in batch.values():
            if isinstance(value, Tensor) and value.is_floating_point():
                return value.new_zeros(())
        return torch.zeros((), dtype=torch.get_default_dtype())

    def forward(self, *args: Any) -> Dict[str, Tensor]:
        if len(args) == 1:
            model = self.model
            batch = args[0]
        elif len(args) == 2:
            model, batch = args
        else:
            raise TypeError("forward expects batch or (model, batch)")
        if model is None:
            raise ValueError("a model must be supplied at construction or forward time")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")

        transition_keys = (
            "state_t",
            "control_t",
            "residual_target",
            "state_t_plus_1",
            "dt",
        )
        has_transition = all(key in batch for key in transition_keys)
        if not has_transition and (self.lambda_residual > 0.0 or self.lambda_one_step > 0.0):
            self._required(batch, transition_keys, "transition")

        zero = self._zero(model, batch)
        residual = zero
        one_step = zero
        one_step_prediction = batch.get("state_t_plus_1", zero)
        if has_transition:
            residual = residual_derivative_loss(
                model,
                batch["state_t"],
                batch["control_t"],
                batch["residual_target"],
                self.normalizers,
                self.residual_weights,
            )
            one_step_result = one_step_prediction_loss(
                model,
                batch["state_t"],
                batch["control_t"],
                batch["dt"],
                batch["state_t_plus_1"],
                self.normalizers,
                method=self.method,
                state_weights=self.state_weights,
                angle_indices=self.angle_indices,
                return_prediction=True,
            )
            one_step, one_step_prediction = one_step_result

        rollout_keys = (
            "rollout_initial_state",
            "rollout_controls",
            "rollout_dt",
            "rollout_target_states",
        )
        has_rollout = all(key in batch for key in rollout_keys)
        if has_rollout:
            rollout_initial = batch["rollout_initial_state"]
            # Some collators preserve the rollout keys with a zero-sized
            # leading window dimension.  Treat that exactly like omitted keys
            # so model implementations need not accept empty batches.
            has_rollout = not (
                isinstance(rollout_initial, Tensor)
                and rollout_initial.ndim == 2
                and rollout_initial.shape[0] == 0
            )
        # A valid episode split can contain no contiguous H-step windows (for
        # example, when every episode is shorter than H).  In that case the
        # trainer omits rollout fields and this component is a stable zero.
        multistep = zero
        trajectory = batch.get("rollout_initial_state", zero)
        if has_rollout:
            multistep, trajectory = multistep_rollout_loss(
                model,
                batch["rollout_initial_state"],
                batch["rollout_controls"],
                batch["rollout_dt"],
                batch["rollout_target_states"],
                self.normalizers,
                method=self.method,
                state_weights=self.state_weights,
                horizon_weights=self.horizon_weights,
                angle_indices=self.angle_indices,
            )

        regularization = parameter_regularization(model).to(
            device=zero.device, dtype=zero.dtype
        )
        total = (
            self.lambda_residual * residual
            + self.lambda_one_step * one_step
            + self.lambda_multistep * multistep
            + self.lambda_reg * regularization
        )
        return {
            "loss": total,
            "total": total,
            "total_loss": total,
            "residual": residual,
            "residual_loss": residual,
            "derivative_loss": residual,
            "one_step": one_step,
            "one_step_loss": one_step,
            "multistep": multistep,
            "multistep_loss": multistep,
            "rollout_loss": multistep,
            "regularization": regularization,
            "regularization_loss": regularization,
            "residual_rmse": torch.sqrt(torch.clamp(residual, min=0.0)),
            "one_step_rmse": torch.sqrt(torch.clamp(one_step, min=0.0)),
            "rollout_rmse": torch.sqrt(torch.clamp(multistep, min=0.0)),
            "one_step_prediction": one_step_prediction,
            "trajectory": trajectory,
            "rollout_trajectory": trajectory,
        }


def compute_loss(
    model: nn.Module,
    batch: Mapping[str, Tensor],
    normalizers: NormalizerSource,
    **kwargs: Any
) -> Dict[str, Tensor]:
    """Functional entry point for :class:`CompositeResidualLoss`."""

    return CompositeResidualLoss(
        model=model, normalizers=normalizers, **kwargs
    )(batch)


__all__ = [
    "CompositeResidualLoss",
    "TorchNormalizerBundle",
    "combined_derivative",
    "combined_unicycle_derivative",
    "compute_loss",
    "derivative_loss",
    "encode_state_torch",
    "multistep_rollout_loss",
    "one_step_loss",
    "one_step_prediction_loss",
    "parameter_regularization",
    "predict_physical_residual",
    "residual_derivative_loss",
    "rollout_combined_dynamics",
    "rollout_loss",
    "torch_combined_derivative",
    "torch_integrate_step",
    "torch_nominal_unicycle_derivative",
    "wrap_state_angles_torch",
    "wrapped_state_error",
]
