import math
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _activation(name):
    choices = {
        "softplus": nn.Softplus,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    key = str(name).lower()
    if key not in choices:
        raise ValueError("unsupported activation: %s" % name)
    return choices[key]


def encode_state(state, angle_indices):
    angle_set = set(int(index) for index in angle_indices)
    values = []
    for index in range(state.shape[-1]):
        component = state[..., index:index + 1]
        if index in angle_set:
            values.extend((torch.sin(component), torch.cos(component)))
        else:
            values.append(component)
    return torch.cat(values, dim=-1)


def _mlp(input_dim, output_dim, hidden_sizes, activation, final_scale):
    layers = []
    previous = int(input_dim)
    activation_class = _activation(activation)
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, int(width)), activation_class()))
        previous = int(width)
    final = nn.Linear(previous, int(output_dim))
    nn.init.uniform_(final.weight, -float(final_scale), float(final_scale))
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ResidualNetwork(nn.Module):
    """MLP or control-affine ICODE residual with embedded normalization."""

    def __init__(self, state_dim, control_dim, config, statistics):
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.model_type = str(config.get("type", "icode_residual"))
        self.angle_indices = tuple(int(v) for v in config.get("angle_indices", (2,)))
        self.feature_dim = self.state_dim + len(self.angle_indices)
        hidden = tuple(int(v) for v in config.get("hidden_sizes", (128, 128)))
        activation = str(config.get("activation", "softplus"))
        scale = float(config.get("final_layer_scale", 1e-3))
        if self.model_type == "icode_residual":
            self.drift = _mlp(self.feature_dim, self.state_dim, hidden, activation, scale)
            self.gain = _mlp(
                self.feature_dim, self.state_dim * self.control_dim, hidden, activation, scale
            )
            self.direct = None
        elif self.model_type == "mlp_residual":
            self.direct = _mlp(
                self.feature_dim + self.control_dim, self.state_dim, hidden, activation, scale
            )
            self.drift = None
            self.gain = None
        else:
            raise ValueError("unknown residual model type: %s" % self.model_type)
        output_mask = np.asarray(
            config.get("residual_output_mask", np.ones(self.state_dim)),
            dtype=np.float32,
        ).reshape(-1)
        if output_mask.shape != (self.state_dim,):
            raise ValueError("residual_output_mask must match state dimension")
        if (
            not np.isfinite(output_mask).all()
            or np.any(output_mask < 0.0)
            or np.any(output_mask > 1.0)
            or not np.any(output_mask > 0.0)
        ):
            raise ValueError(
                "residual_output_mask entries must be finite in [0,1] and retain a channel"
            )
        self.register_buffer(
            "residual_output_mask", torch.as_tensor(output_mask), persistent=False
        )
        for name in ("feature_mean", "feature_scale", "control_mean", "control_scale", "residual_mean", "residual_scale"):
            value = torch.as_tensor(statistics[name], dtype=torch.float32)
            self.register_buffer(name, value)
        self.model_config = dict(config)
        self.check_finite = True

    def components(self, state, control):
        feature = encode_state(state, self.angle_indices)
        feature_normalized = (feature - self.feature_mean) / self.feature_scale
        control_normalized = (control - self.control_mean) / self.control_scale
        if self.model_type == "icode_residual":
            drift_normalized = self.drift(feature_normalized)
            gain_normalized = self.gain(feature_normalized).reshape(
                state.shape[:-1] + (self.state_dim, self.control_dim)
            )
            control_contribution = torch.matmul(
                gain_normalized, control_normalized.unsqueeze(-1)
            ).squeeze(-1)
            normalized = drift_normalized + control_contribution
            gain_physical = (
                self.residual_scale[..., :, None]
                * gain_normalized
                / self.control_scale[..., None, :]
            )
            drift_physical = self.residual_mean + self.residual_scale * (
                drift_normalized
                - torch.matmul(
                    gain_normalized,
                    (self.control_mean / self.control_scale).unsqueeze(-1),
                ).squeeze(-1)
            )
            control_contribution_physical = torch.matmul(
                gain_physical, control.unsqueeze(-1)
            ).squeeze(-1)
        else:
            normalized = self.direct(torch.cat((feature_normalized, control_normalized), dim=-1))
            drift_normalized = normalized
            gain_normalized = torch.zeros(
                state.shape[:-1] + (self.state_dim, self.control_dim),
                dtype=state.dtype, device=state.device,
            )
            control_contribution = torch.zeros_like(normalized)
            gain_physical = torch.zeros_like(gain_normalized)
            drift_physical = self.residual_mean + self.residual_scale * normalized
            control_contribution_physical = torch.zeros_like(normalized)
        residual_unmasked = self.residual_mean + self.residual_scale * normalized
        residual = residual_unmasked * self.residual_output_mask
        if self.check_finite and not torch.isfinite(residual).all():
            raise FloatingPointError("residual network produced NaN or Inf")
        return residual, {
            "drift_normalized": drift_normalized,
            "gain_normalized": gain_normalized,
            "control_contribution_normalized": control_contribution,
            "drift_physical": drift_physical,
            "gain_physical": gain_physical,
            "control_contribution_physical": control_contribution_physical,
            "residual_unmasked": residual_unmasked,
            "residual_output_mask": self.residual_output_mask,
        }

    def forward(self, state, control):
        return self.components(state, control)[0]

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint_config(self):
        return {
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "model": dict(self.model_config),
            "statistics": {
                name: getattr(self, name).detach().cpu().tolist()
                for name in ("feature_mean", "feature_scale", "control_mean", "control_scale", "residual_mean", "residual_scale")
            },
        }


def load_platform_checkpoint(path, device="cpu"):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Torch < 2.0 compatibility; checkpoints remain project-owned.
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "mobile_robot_mppi_residual_v1":
        raise ValueError("not a refactored-platform residual checkpoint")
    config = payload["model_config"]
    model = ResidualNetwork(
        config["state_dim"], config["control_dim"], config["model"], config["statistics"]
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


class PlatformResidualDynamics:
    def __init__(self, model, device="cpu", use_torchscript=False):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.state_dim = int(model.state_dim)
        self.control_dim = int(model.control_dim)
        self.use_torchscript = bool(use_torchscript)
        self.inference_model = self.model
        if self.use_torchscript:
            example_state = torch.zeros((1, self.state_dim), device=self.device)
            example_control = torch.zeros((1, self.control_dim), device=self.device)
            # The external NumPy boundary below retains the runtime finite
            # check; disabling the Python conditional only while tracing avoids
            # hard-coding a tensor-to-bool branch into the graph.
            self.model.check_finite = False
            try:
                traced = torch.jit.trace(
                    self.model, (example_state, example_control), check_trace=False
                )
                self.inference_model = torch.jit.optimize_for_inference(traced)
            finally:
                self.model.check_finite = True

    @classmethod
    def from_checkpoint(cls, path, device="cpu", use_torchscript=False):
        model, _ = load_platform_checkpoint(path, device)
        return cls(model, device, use_torchscript=use_torchscript)

    def derivative(self, state, control, time=None):
        del time
        state_value = np.asarray(state, dtype=np.float32)
        control_value = np.asarray(control, dtype=np.float32)
        unbatched = state_value.ndim == 1
        if unbatched:
            if (
                state_value.shape != (self.state_dim,)
                or control_value.shape != (self.control_dim,)
            ):
                raise ValueError("single residual input dimensions do not match checkpoint")
            state_value = state_value[None, :]
            control_value = control_value[None, :]
        elif (
            state_value.ndim < 2
            or state_value.shape[-1] != self.state_dim
            or control_value.shape != state_value.shape[:-1] + (self.control_dim,)
        ):
            raise ValueError("batched residual inputs must have aligned leading dimensions")
        with torch.inference_mode():
            state_tensor = torch.as_tensor(state_value, device=self.device)
            control_tensor = torch.as_tensor(control_value, device=self.device)
            output = self.inference_model(state_tensor, control_tensor)
        result = output.detach().cpu().numpy().astype(np.float64, copy=False)
        if unbatched:
            result = result[0]
        if not np.isfinite(result).all():
            raise FloatingPointError("compiled residual inference produced NaN or Inf")
        return result


class PlatformResidualEnsemble:
    """Mean residual prediction with auditable epistemic diagnostics.

    The ensemble never scales its residual output from uncertainty.  It keeps
    model prediction and authority allocation separate: MPPI always receives
    the ensemble mean, while higher-level sampling logic may consume
    disagreement, training-support confidence, and completed-transition
    innovation.  This avoids silently falling back to a potentially worse
    nominal model when uncertainty rises.
    """

    def __init__(
        self,
        members,
        state_scales=None,
        disagreement_scales=None,
        innovation_scales=None,
        support_soft_z=3.0,
        support_hard_z=7.0,
        innovation_decay=0.9,
        member_paths=None,
    ):
        self.members = tuple(members)
        if len(self.members) < 2:
            raise ValueError("residual ensemble requires at least two members")
        self.state_dim = int(self.members[0].state_dim)
        self.control_dim = int(self.members[0].control_dim)
        if any(
            int(member.state_dim) != self.state_dim
            or int(member.control_dim) != self.control_dim
            for member in self.members
        ):
            raise ValueError("residual ensemble member dimensions differ")
        self.model = getattr(self.members[0], "model", None)
        legacy_scales = (
            np.ones(self.state_dim, dtype=np.float64)
            if state_scales is None
            else state_scales
        )
        self.disagreement_scales = self._validated_scales(
            (
                legacy_scales
                if disagreement_scales is None
                else disagreement_scales
            ),
            "disagreement_scales",
        )
        self.innovation_scales = self._validated_scales(
            (
                legacy_scales
                if innovation_scales is None
                else innovation_scales
            ),
            "innovation_scales",
        )
        # Backwards-compatible diagnostic alias. New configurations must use
        # the dimensionally explicit scales above.
        self.state_scales = self.innovation_scales
        self.support_soft_z = float(support_soft_z)
        self.support_hard_z = float(support_hard_z)
        if (
            not math.isfinite(self.support_soft_z)
            or not math.isfinite(self.support_hard_z)
            or self.support_soft_z < 0.0
            or self.support_hard_z <= self.support_soft_z
        ):
            raise ValueError(
                "residual ensemble requires 0 <= support_soft_z < support_hard_z"
            )
        self.innovation_decay = float(innovation_decay)
        if not 0.0 <= self.innovation_decay < 1.0:
            raise ValueError("innovation_decay must lie in [0,1)")
        if member_paths is None:
            self.member_paths = tuple(None for _ in self.members)
        else:
            self.member_paths = tuple(str(value) for value in member_paths)
            if len(self.member_paths) != len(self.members):
                raise ValueError("member_paths must align with ensemble members")
        self.innovation_error_ema = 0.0
        self.innovation_samples = 0
        self.last_innovation_error = 0.0
        self.innovation_error_vector_ema = np.zeros(
            self.state_dim, dtype=np.float64
        )
        self.last_innovation_error_vector = np.zeros(
            self.state_dim, dtype=np.float64
        )

    def _validated_scales(self, values, name):
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if (
            result.shape != (self.state_dim,)
            or not np.isfinite(result).all()
            or np.any(result <= 0.0)
        ):
            raise ValueError(
                "residual ensemble %s must be finite and positive" % name
            )
        return result

    def reset(self):
        self.innovation_error_ema = 0.0
        self.innovation_samples = 0
        self.last_innovation_error = 0.0
        self.innovation_error_vector_ema.fill(0.0)
        self.last_innovation_error_vector.fill(0.0)

    def member_derivatives(self, state, control, time=None):
        values = np.stack(
            [
                np.asarray(
                    member.derivative(state, control, time),
                    dtype=np.float64,
                )
                for member in self.members
            ],
            axis=0,
        )
        if values.shape[-1] != self.state_dim or not np.isfinite(values).all():
            raise FloatingPointError(
                "residual ensemble member inference is invalid"
            )
        return values

    def derivative(self, state, control, time=None):
        return np.mean(
            self.member_derivatives(state, control, time), axis=0
        )

    def ungated_derivative(self, state, control, time=None):
        return self.derivative(state, control, time)

    def disagreement(self, state, control, time=None):
        """Return normalized RMS ensemble standard deviation."""

        standard_deviation = np.std(
            self.member_derivatives(state, control, time), axis=0
        )
        normalized = standard_deviation / self.disagreement_scales
        score = np.sqrt(np.mean(normalized ** 2, axis=-1))
        if not np.isfinite(score).all():
            raise FloatingPointError(
                "residual ensemble disagreement is not finite"
            )
        return score

    @staticmethod
    def _features(model, state):
        angle_indices = set(
            int(value) for value in getattr(model, "angle_indices", ())
        )
        values = []
        for index in range(state.shape[-1]):
            component = state[..., index:index + 1]
            if index in angle_indices:
                values.extend((np.sin(component), np.cos(component)))
            else:
                values.append(component)
        return np.concatenate(values, axis=-1)

    def _member_support_confidence(self, member, state, control):
        evaluator = getattr(member, "support_confidence", None)
        if callable(evaluator):
            return np.asarray(evaluator(state, control), dtype=np.float64)
        model = getattr(member, "model", None)
        if model is None:
            raise TypeError(
                "ensemble member lacks model normalization statistics"
            )
        features = self._features(model, state)
        feature_mean = model.feature_mean.detach().cpu().numpy()
        feature_scale = model.feature_scale.detach().cpu().numpy()
        control_mean = model.control_mean.detach().cpu().numpy()
        control_scale = model.control_scale.detach().cpu().numpy()
        feature_z = np.max(
            np.abs((features - feature_mean) / feature_scale), axis=-1
        )
        control_z = np.max(
            np.abs((control - control_mean) / control_scale), axis=-1
        )
        maximum_z = np.maximum(feature_z, control_z)
        confidence = np.clip(
            (self.support_hard_z - maximum_z)
            / (self.support_hard_z - self.support_soft_z),
            0.0,
            1.0,
        )
        return np.where(maximum_z <= self.support_soft_z, 1.0, confidence)

    def support_confidence(self, state, control):
        state_value = np.asarray(state, dtype=np.float64)
        control_value = np.asarray(control, dtype=np.float64)
        if (
            state_value.shape[-1] != self.state_dim
            or control_value.shape
            != state_value.shape[:-1] + (self.control_dim,)
        ):
            raise ValueError(
                "ensemble support inputs have incompatible dimensions"
            )
        confidence = np.min(
            np.stack(
                [
                    self._member_support_confidence(
                        member, state_value, control_value
                    )
                    for member in self.members
                ],
                axis=0,
            ),
            axis=0,
        )
        if not np.isfinite(confidence).all():
            raise FloatingPointError(
                "residual ensemble support confidence is not finite"
            )
        return confidence

    def observe_prediction_errors(self, nominal_error, residual_error):
        """Update innovation using only the completed transition.

        ``nominal_error`` is accepted to preserve the common reliability hook,
        but does not affect ensemble authority.  Runtime confidence describes
        how accurately the ensemble predicted the observed transition, not
        whether it happened to beat nominal on that one transition.
        """

        del nominal_error
        error = np.asarray(residual_error, dtype=np.float64).reshape(-1)
        if error.shape != (self.state_dim,) or not np.isfinite(error).all():
            raise ValueError(
                "ensemble innovation error must match the state dimension"
            )
        value = float(np.sqrt(np.mean(
            (error / self.innovation_scales) ** 2
        )))
        if self.innovation_samples == 0:
            self.innovation_error_ema = value
            self.innovation_error_vector_ema = error.copy()
        else:
            self.innovation_error_ema = (
                self.innovation_decay * self.innovation_error_ema
                + (1.0 - self.innovation_decay) * value
            )
            self.innovation_error_vector_ema = (
                self.innovation_decay * self.innovation_error_vector_ema
                + (1.0 - self.innovation_decay) * error
            )
        self.last_innovation_error = value
        self.last_innovation_error_vector = error.copy()
        self.innovation_samples += 1
        return self.innovation_error_ema

    def diagnostics(self):
        return {
            "residual_ensemble_enabled": True,
            "residual_ensemble_members": len(self.members),
            "residual_ensemble_innovation_samples": int(
                self.innovation_samples
            ),
            "residual_ensemble_innovation_error_ema": float(
                self.innovation_error_ema
            ),
            "residual_ensemble_last_innovation_error": float(
                self.last_innovation_error
            ),
            "residual_ensemble_innovation_error_vector_ema": (
                self.innovation_error_vector_ema.copy()
            ),
            "residual_ensemble_last_innovation_error_vector": (
                self.last_innovation_error_vector.copy()
            ),
        }


class NormalizedSupportGatedResidualDynamics:
    """Fail toward nominal dynamics outside the residual training support.

    Confidence is one inside ``soft_z``, decreases linearly, and is exactly
    zero at and beyond ``hard_z``.  It uses only checkpoint normalization
    statistics; simulator truth and task outcome never enter the gate.
    """

    def __init__(self, residual, soft_z=3.0, hard_z=5.0):
        self.residual = residual
        self.model = getattr(residual, "model", None)
        if self.model is None:
            raise TypeError("support gating requires an embedded residual model")
        self.state_dim = int(residual.state_dim)
        self.control_dim = int(residual.control_dim)
        self.soft_z = float(soft_z)
        self.hard_z = float(hard_z)
        if (
            not math.isfinite(self.soft_z)
            or not math.isfinite(self.hard_z)
            or self.soft_z < 0.0
            or self.hard_z <= self.soft_z
        ):
            raise ValueError("support gate requires 0 <= soft_z < hard_z")

    def _features(self, state):
        angle_indices = set(int(value) for value in self.model.angle_indices)
        values = []
        for index in range(self.state_dim):
            component = state[..., index:index + 1]
            if index in angle_indices:
                values.extend((np.sin(component), np.cos(component)))
            else:
                values.append(component)
        return np.concatenate(values, axis=-1)

    def confidence(self, state, control):
        state_value = np.asarray(state, dtype=np.float64)
        control_value = np.asarray(control, dtype=np.float64)
        if state_value.shape[-1] != self.state_dim:
            raise ValueError("support-gate state dimension mismatch")
        if control_value.shape[-1] != self.control_dim:
            raise ValueError("support-gate control dimension mismatch")
        features = self._features(state_value)
        feature_mean = self.model.feature_mean.detach().cpu().numpy()
        feature_scale = self.model.feature_scale.detach().cpu().numpy()
        control_mean = self.model.control_mean.detach().cpu().numpy()
        control_scale = self.model.control_scale.detach().cpu().numpy()
        feature_z = np.max(np.abs((features - feature_mean) / feature_scale), axis=-1)
        control_z = np.max(np.abs((control_value - control_mean) / control_scale), axis=-1)
        maximum_z = np.maximum(feature_z, control_z)
        confidence = np.clip(
            (self.hard_z - maximum_z) / (self.hard_z - self.soft_z), 0.0, 1.0
        )
        confidence = np.where(maximum_z <= self.soft_z, 1.0, confidence)
        if not np.isfinite(confidence).all():
            raise FloatingPointError("support confidence produced NaN or Inf")
        return confidence

    def derivative(self, state, control, time=None):
        value = np.asarray(
            self.residual.derivative(state, control, time), dtype=np.float64
        )
        confidence = np.asarray(self.confidence(state, control), dtype=np.float64)
        return value * confidence[..., None]


class ResidualComponentMaskedDynamics:
    """Apply a documented structural mask to residual derivative channels.

    For a dynamic unicycle, pose kinematics can remain exact while the learned
    model corrects only the uncertain velocity and yaw-rate dynamics.  The mask
    is supplied by configuration rather than inferred from simulator truth.
    """

    def __init__(self, residual, mask):
        self.residual = residual
        self.model = getattr(residual, "model", None)
        self.state_dim = int(residual.state_dim)
        self.control_dim = int(residual.control_dim)
        self.mask = np.asarray(mask, dtype=np.float64).reshape(-1)
        if self.mask.shape != (self.state_dim,):
            raise ValueError("residual component mask must match state dimension")
        if not np.isfinite(self.mask).all() or np.any(self.mask < 0.0) or np.any(self.mask > 1.0):
            raise ValueError("residual component mask entries must be finite in [0,1]")
        if not np.any(self.mask > 0.0):
            raise ValueError("residual component mask must retain at least one channel")

    def derivative(self, state, control, time=None):
        value = np.asarray(
            self.residual.derivative(state, control, time), dtype=np.float64
        )
        return value * self.mask


class InnovationGatedResidualDynamics:
    """Scale a learned residual using only completed transition evidence.

    The gate receives paired one-step prediction errors after the next state has
    already been observed.  It therefore cannot inspect future plant state when
    choosing the residual scale for the current MPPI plan.  Evidence is the
    bounded relative reduction in normalized squared error, accumulated with an
    exponentially weighted lower confidence bound.  Cold start fails closed to
    the nominal model.
    """

    def __init__(
        self,
        residual,
        state_indices,
        state_scales,
        forgetting_factor=0.95,
        minimum_samples=8,
        confidence_z=1.0,
        off_threshold=0.0,
        on_threshold=0.15,
        rise_rate=0.25,
        fall_rate=0.5,
        context_value=None,
        context_off_threshold=0.5,
        context_on_threshold=0.8,
    ):
        self.residual = residual
        self.model = getattr(residual, "model", None)
        self.state_dim = int(residual.state_dim)
        self.control_dim = int(residual.control_dim)
        self.state_indices = np.asarray(state_indices, dtype=np.int64).reshape(-1)
        self.state_scales = np.asarray(state_scales, dtype=np.float64).reshape(-1)
        if not self.state_indices.size or self.state_scales.shape != self.state_indices.shape:
            raise ValueError("reliability state indices and scales must be nonempty and aligned")
        if np.any(self.state_indices < 0) or np.any(self.state_indices >= self.state_dim):
            raise ValueError("reliability state index is outside the state dimension")
        if len(set(self.state_indices.tolist())) != int(self.state_indices.size):
            raise ValueError("reliability state indices must be unique")
        if not np.isfinite(self.state_scales).all() or np.any(self.state_scales <= 0.0):
            raise ValueError("reliability state scales must be finite and positive")
        self.forgetting_factor = float(forgetting_factor)
        self.minimum_samples = int(minimum_samples)
        self.confidence_z = float(confidence_z)
        self.off_threshold = float(off_threshold)
        self.on_threshold = float(on_threshold)
        self.rise_rate = float(rise_rate)
        self.fall_rate = float(fall_rate)
        self.context_value = (
            None if context_value is None else float(context_value)
        )
        self.context_off_threshold = float(context_off_threshold)
        self.context_on_threshold = float(context_on_threshold)
        numeric = (
            self.forgetting_factor, self.confidence_z, self.off_threshold,
            self.on_threshold, self.rise_rate, self.fall_rate,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("reliability gate parameters must be finite")
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0,1]")
        if self.minimum_samples <= 0 or self.confidence_z < 0.0:
            raise ValueError("minimum_samples must be positive and confidence_z nonnegative")
        if self.on_threshold <= self.off_threshold:
            raise ValueError("on_threshold must exceed off_threshold")
        if not 0.0 < self.rise_rate <= 1.0 or not 0.0 < self.fall_rate <= 1.0:
            raise ValueError("reliability rise/fall rates must be in (0,1]")
        if self.context_value is None:
            self.context_alpha = 1.0
        else:
            context_values = (
                self.context_value, self.context_off_threshold,
                self.context_on_threshold,
            )
            if (
                not np.isfinite(context_values).all()
                or self.context_value < 0.0
                or self.context_on_threshold <= self.context_off_threshold
            ):
                raise ValueError(
                    "context value must be nonnegative and finite with on > off"
                )
            self.context_alpha = float(np.clip(
                (self.context_value - self.context_off_threshold)
                / (self.context_on_threshold - self.context_off_threshold),
                0.0, 1.0,
            ))
        self.reset()

    def reset(self):
        self.alpha = 0.0
        self.evidence_alpha = 0.0
        self.samples = 0
        self._weight = 0.0
        self._weight_squared = 0.0
        self._mean = 0.0
        self._second_moment = 0.0
        self._lcb = -1.0
        self._last_nominal_error = 0.0
        self._last_residual_error = 0.0
        self._last_relative_improvement = 0.0

    def _normalized_error(self, error):
        values = np.asarray(error, dtype=np.float64).reshape(-1)
        if values.shape != (self.state_dim,) or not np.isfinite(values).all():
            raise ValueError("reliability prediction error must be one finite state vector")
        selected = values[self.state_indices] / self.state_scales
        return float(np.mean(selected ** 2))

    def observe_prediction_errors(self, nominal_error, residual_error):
        nominal = self._normalized_error(nominal_error)
        residual = self._normalized_error(residual_error)
        relative = (nominal - residual) / max(nominal + residual, 1e-12)
        relative = float(np.clip(relative, -1.0, 1.0))
        decay = self.forgetting_factor
        previous_weight = self._weight
        self._weight = decay * previous_weight + 1.0
        self._weight_squared = decay ** 2 * self._weight_squared + 1.0
        self._mean = (
            decay * previous_weight * self._mean + relative
        ) / self._weight
        self._second_moment = (
            decay * previous_weight * self._second_moment + relative ** 2
        ) / self._weight
        self.samples += 1
        variance = max(self._second_moment - self._mean ** 2, 0.0)
        effective_samples = self._weight ** 2 / max(self._weight_squared, 1e-12)
        standard_error = np.sqrt(variance / max(effective_samples, 1.0))
        self._lcb = float(self._mean - self.confidence_z * standard_error)
        if self.samples < self.minimum_samples:
            target = 0.0
        else:
            target = float(np.clip(
                (self._lcb - self.off_threshold)
                / (self.on_threshold - self.off_threshold), 0.0, 1.0
            ))
        rate = self.rise_rate if target > self.evidence_alpha else self.fall_rate
        self.evidence_alpha = float(np.clip(
            self.evidence_alpha + rate * (target - self.evidence_alpha),
            0.0, 1.0,
        ))
        self.alpha = float(self.evidence_alpha * self.context_alpha)
        self._last_nominal_error = nominal
        self._last_residual_error = residual
        self._last_relative_improvement = relative
        return self.alpha

    def ungated_derivative(self, state, control, time=None):
        return np.asarray(
            self.residual.derivative(state, control, time), dtype=np.float64
        )

    def derivative(self, state, control, time=None):
        return self.alpha * self.ungated_derivative(state, control, time)

    def confidence(self, state, control):
        support = getattr(self.residual, "confidence", None)
        if callable(support):
            return support(state, control)
        state_value = np.asarray(state)
        return np.ones(state_value.shape[:-1], dtype=np.float64)

    def diagnostics(self):
        return {
            "residual_reliability_enabled": True,
            "residual_reliability_alpha": float(self.alpha),
            "residual_reliability_evidence_alpha": float(self.evidence_alpha),
            "residual_reliability_context_alpha": float(self.context_alpha),
            "residual_reliability_context_value": (
                0.0 if self.context_value is None else float(self.context_value)
            ),
            "residual_reliability_samples": int(self.samples),
            "residual_reliability_mean_improvement": float(self._mean),
            "residual_reliability_lcb": float(self._lcb),
            "residual_reliability_last_relative_improvement": float(
                self._last_relative_improvement
            ),
            "residual_reliability_nominal_error": float(self._last_nominal_error),
            "residual_reliability_residual_error": float(self._last_residual_error),
        }
