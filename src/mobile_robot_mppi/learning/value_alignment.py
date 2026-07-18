"""Critic-informed fine-tuning objectives for continuous-time residual models.

The SAC Actor and twin critics are frozen. Gradients are allowed only from the
critic value at a predicted physical state back through the residual rollout.
This makes the residual model control-aware without silently updating the RL
policy or treating simulator truth as an online controller input.
"""

import copy
import math
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.observation import ObservationEncoderConfig
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig

from .trainer import integrate, state_mse


def _finite_nonnegative(value, name):
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be finite and non-negative" % name)
    return result


class FrozenDirectSACValue(nn.Module):
    """Differentiable state-to-value view of a frozen direct-control SAC model."""

    def __init__(
        self,
        agent,
        encoder_config,
        normalizer_state,
        critic_source="target",
        support_soft_z=3.0,
        support_hard_z=7.0,
        disagreement_scale=1.0,
    ):
        super().__init__()
        if bool(getattr(agent, "is_correction_policy", False)):
            raise ValueError("value alignment requires a direct-control Actor")
        config = (
            encoder_config
            if isinstance(encoder_config, ObservationEncoderConfig)
            else ObservationEncoderConfig.from_mapping(encoder_config)
        )
        config.validate()
        if int(config.history_frames) != 1:
            raise ValueError("value alignment currently requires history_frames=1")
        self.encoder_config = config
        self.observation_dim = int(agent.observation_dim)
        self.action_dim = int(agent.action_dim)
        self.actor = agent.actor
        source = str(critic_source)
        if source == "target":
            self.critic1 = agent.target_critic1
            self.critic2 = agent.target_critic2
        elif source == "online":
            self.critic1 = agent.critic1
            self.critic2 = agent.critic2
        else:
            raise ValueError("critic_source must be online or target")
        self.critic_source = source
        self.actor.requires_grad_(False)
        self.critic1.requires_grad_(False)
        self.critic2.requires_grad_(False)
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

        count = int(normalizer_state["count"])
        mean = np.asarray(normalizer_state["mean"], dtype=np.float32)
        m2 = np.asarray(normalizer_state["m2"], dtype=np.float64)
        min_std = float(normalizer_state.get("min_std", 0.05))
        if mean.shape != (self.observation_dim,) or m2.shape != mean.shape:
            raise ValueError("SAC normalizer shape does not match observation dimension")
        if count < 0 or not np.isfinite(mean).all() or not np.isfinite(m2).all():
            raise ValueError("SAC normalizer statistics are invalid")
        if count < 2:
            std = np.ones_like(mean)
        else:
            std = np.maximum(np.sqrt(np.maximum(m2, 0.0) / float(count - 1)), min_std)
        self.register_buffer("normalizer_mean", torch.as_tensor(mean))
        self.register_buffer("normalizer_std", torch.as_tensor(std, dtype=torch.float32))
        self.normalizer_clip = float(normalizer_state.get("clip", 10.0))
        if not math.isfinite(self.normalizer_clip) or self.normalizer_clip <= 0.0:
            raise ValueError("SAC normalizer clip must be positive")

        self.support_soft_z = float(support_soft_z)
        self.support_hard_z = float(support_hard_z)
        self.disagreement_scale = float(disagreement_scale)
        if (
            not 0.0 <= self.support_soft_z < self.support_hard_z
            or not math.isfinite(self.support_hard_z)
            or not math.isfinite(self.disagreement_scale)
            or self.disagreement_scale <= 0.0
        ):
            raise ValueError("invalid critic confidence calibration")

    @classmethod
    def from_checkpoint(
        cls,
        path,
        device="cpu",
        critic_source="target",
        **kwargs
    ):
        payload = load_sac_checkpoint(path, map_location=device)
        if str(payload.get("action_mode", "")) != "direct_control":
            raise ValueError("value alignment requires a direct_control checkpoint")
        state = payload["agent"]
        agent = SACAgent(
            state["observation_dim"],
            state["action_dim"],
            SACConfig.from_mapping(state["config"]),
            device=device,
        )
        agent.load_state_dict(state, load_optimizers=False)
        return cls(
            agent,
            payload["encoder_config"],
            payload["normalizer"],
            critic_source=critic_source,
            **kwargs
        )

    def train(self, mode=True):
        # The residual model may enter training mode, but the value model must
        # remain deterministic and frozen.
        super().train(False)
        return self

    def _normalize(self, raw_observation):
        if raw_observation.shape[-1] != self.observation_dim:
            raise ValueError("raw critic observation dimension mismatch")
        mean = self.normalizer_mean.to(
            device=raw_observation.device, dtype=raw_observation.dtype
        )
        std = self.normalizer_std.to(
            device=raw_observation.device, dtype=raw_observation.dtype
        )
        z = (raw_observation - mean) / std
        return torch.clamp(z, -self.normalizer_clip, self.normalizer_clip), z

    def _critic_values(self, normalized_observation):
        action = self.actor.mean_action(normalized_observation)
        q1_distribution = self.critic1(normalized_observation, action)
        q2_distribution = self.critic2(normalized_observation, action)
        q1 = q1_distribution.mean(dim=-1)
        q2 = q2_distribution.mean(dim=-1)
        return torch.minimum(q1, q2), torch.abs(q1 - q2)

    def value_from_raw(self, raw_observation):
        normalized, _ = self._normalize(raw_observation)
        value, _ = self._critic_values(normalized)
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("frozen SAC value produced NaN or Inf")
        return value

    def confidence_from_raw(self, raw_observation):
        normalized, unbounded_z = self._normalize(raw_observation)
        _, disagreement = self._critic_values(normalized)
        support_z = torch.amax(torch.abs(unbounded_z), dim=-1)
        support = torch.clamp(
            (self.support_hard_z - support_z)
            / (self.support_hard_z - self.support_soft_z),
            0.0,
            1.0,
        )
        support = torch.where(
            support_z <= self.support_soft_z,
            torch.ones_like(support),
            support,
        )
        critic_agreement = torch.exp(-disagreement / self.disagreement_scale)
        confidence = support * critic_agreement
        if not bool(torch.isfinite(confidence).all().item()):
            raise FloatingPointError("critic confidence produced NaN or Inf")
        return confidence

    def raw_observation_for_state(
        self,
        state,
        raw_template,
        target_position,
    ):
        """Replace only physical-state features in a recorded critic context."""

        if state.shape[:-1] != raw_template.shape[:-1]:
            raise ValueError("state and raw template leading dimensions must match")
        if target_position.shape != state.shape[:-1] + (2,):
            raise ValueError("target position must match the state batch")
        if state.shape[-1] not in (3, 5):
            raise ValueError("value alignment supports 3D or 5D unicycle states")
        if raw_template.shape[-1] != self.observation_dim:
            raise ValueError("raw template observation dimension mismatch")
        theta = state[..., 2]
        x = state[..., 0]
        y = state[..., 1]
        dx = target_position[..., 0] - x
        dy = target_position[..., 1] - y
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        body_dx = cosine * dx + sine * dy
        body_dy = -sine * dx + cosine * dy
        distance_scale = float(self.encoder_config.goal_distance_scale)
        if state.shape[-1] == 5:
            velocity = state[..., 3]
            yaw_rate = state[..., 4]
        else:
            # Three-state residual models have no differentiable measured
            # velocity state. Preserve those two recorded context features.
            velocity = (
                raw_template[..., 4]
                * float(self.encoder_config.velocity_scale)
            )
            yaw_rate = (
                raw_template[..., 5]
                * float(self.encoder_config.yaw_rate_scale)
            )
        physical = [
            torch.clamp(body_dx / distance_scale, -1.0, 1.0),
            torch.clamp(body_dy / distance_scale, -1.0, 1.0),
            torch.clamp(
                torch.sqrt(dx.square() + dy.square()) / distance_scale,
                0.0,
                1.0,
            ),
            torch.atan2(body_dy, body_dx) / math.pi,
            torch.clamp(
                velocity / float(self.encoder_config.velocity_scale),
                -2.0,
                2.0,
            ),
            torch.clamp(
                yaw_rate / float(self.encoder_config.yaw_rate_scale),
                -2.0,
                2.0,
            ),
            sine,
            cosine,
        ]
        prefix = torch.stack(physical, dim=-1)
        consumed = 8
        if self.encoder_config.include_absolute_pose:
            absolute = torch.stack(
                (x / distance_scale, y / distance_scale), dim=-1
            )
            prefix = torch.cat((prefix, absolute), dim=-1)
            consumed += 2
        return torch.cat((prefix, raw_template[..., consumed:]), dim=-1)

    def value_from_state(self, state, raw_template, target_position):
        raw = self.raw_observation_for_state(
            state, raw_template, target_position
        )
        return self.value_from_raw(raw)


class ValueAlignedResidualObjective(nn.Module):
    """Base residual supervision plus critic alignment and checkpoint anchoring."""

    def __init__(
        self,
        anchor_model,
        value_model,
        dynamics_config,
        derivative_weight=0.25,
        one_step_weight=1.0,
        multistep_weight=4.0,
        value_weight=0.05,
        anchor_weight=1.0,
        value_scale=1.0,
        competence_off_value=None,
        competence_on_value=None,
        state_weights=None,
        horizon_weights=None,
    ):
        super().__init__()
        self.anchor_model = copy.deepcopy(anchor_model)
        self.anchor_model.requires_grad_(False)
        self.anchor_model.eval()
        self.value_model = value_model
        self.value_model.requires_grad_(False)
        self.value_model.eval()
        self.dynamics_config = copy.deepcopy(dict(dynamics_config))
        self.derivative_weight = _finite_nonnegative(
            derivative_weight, "derivative_weight"
        )
        self.one_step_weight = _finite_nonnegative(
            one_step_weight, "one_step_weight"
        )
        self.multistep_weight = _finite_nonnegative(
            multistep_weight, "multistep_weight"
        )
        self.value_weight = _finite_nonnegative(value_weight, "value_weight")
        self.anchor_weight = _finite_nonnegative(anchor_weight, "anchor_weight")
        self.value_scale = float(value_scale)
        if not math.isfinite(self.value_scale) or self.value_scale <= 0.0:
            raise ValueError("value_scale must be finite and positive")
        if competence_off_value is None and competence_on_value is None:
            self.competence_off_value = None
            self.competence_on_value = None
        elif competence_off_value is None or competence_on_value is None:
            raise ValueError(
                "critic competence requires both off and on values"
            )
        else:
            self.competence_off_value = float(competence_off_value)
            self.competence_on_value = float(competence_on_value)
            if (
                not math.isfinite(self.competence_off_value)
                or not math.isfinite(self.competence_on_value)
                or self.competence_on_value <= self.competence_off_value
            ):
                raise ValueError(
                    "critic competence requires finite on > off"
                )
        self.state_weights = (
            None
            if state_weights is None
            else tuple(float(value) for value in state_weights)
        )
        self.horizon_weights = (
            None
            if horizon_weights is None
            else tuple(float(value) for value in horizon_weights)
        )

    def train(self, mode=True):
        super().train(mode)
        self.anchor_model.eval()
        self.value_model.eval()
        return self

    def forward(self, model, batch):
        required = (
            "initial_state",
            "states_t",
            "controls",
            "dt",
            "target_states",
            "residual_targets",
            "raw_observations",
            "target_positions",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError("value-alignment batch is missing %s" % ", ".join(missing))
        states_t = batch["states_t"]
        controls = batch["controls"]
        dt = batch["dt"]
        target_states = batch["target_states"]
        horizon = int(controls.shape[-2])
        if horizon <= 0 or states_t.shape != target_states.shape:
            raise ValueError("value-alignment trajectory shapes are invalid")
        if controls.shape[:-1] != states_t.shape[:-1]:
            raise ValueError("controls and states must share batch/horizon dimensions")
        if batch["raw_observations"].shape[:-1] != states_t.shape[:-1]:
            raise ValueError("critic contexts must share batch/horizon dimensions")

        flat_state = states_t.reshape(-1, states_t.shape[-1])
        flat_control = controls.reshape(-1, controls.shape[-1])
        flat_target_residual = batch["residual_targets"].reshape(
            -1, states_t.shape[-1]
        )
        predicted_residual = model(flat_state, flat_control)
        mask = model.residual_output_mask.reshape(1, -1)
        derivative_squared = (predicted_residual - flat_target_residual).square()
        derivative = torch.sum(derivative_squared * mask) / (
            derivative_squared.shape[0] * torch.sum(mask)
        )
        with torch.no_grad():
            anchored = self.anchor_model(flat_state, flat_control)
        anchor_squared = (predicted_residual - anchored).square()
        anchor = torch.sum(anchor_squared * mask) / (
            anchor_squared.shape[0] * torch.sum(mask)
        )

        flat_dt = dt.reshape(-1, 1)
        predicted_next = integrate(
            model,
            flat_state,
            flat_control,
            flat_dt,
            self.dynamics_config,
            self.dynamics_config.get("integrator", "rk4"),
        )
        one_step = state_mse(
            predicted_next,
            target_states.reshape(-1, target_states.shape[-1]),
            self.state_weights,
        )

        if self.horizon_weights is None:
            step_weights = torch.ones(
                horizon, device=states_t.device, dtype=states_t.dtype
            )
        else:
            step_weights = torch.as_tensor(
                self.horizon_weights,
                device=states_t.device,
                dtype=states_t.dtype,
            )
            if step_weights.shape != (horizon,):
                raise ValueError("horizon_weights must match batch horizon")
        step_weights = step_weights / torch.sum(step_weights)

        current = batch["initial_state"]
        rollout_losses = []
        predicted_trajectory = []
        for offset in range(horizon):
            current = integrate(
                model,
                current,
                controls[:, offset],
                dt[:, offset].reshape(-1, 1),
                self.dynamics_config,
                self.dynamics_config.get("integrator", "rk4"),
            )
            predicted_trajectory.append(current)
            rollout_losses.append(
                state_mse(
                    current,
                    target_states[:, offset],
                    self.state_weights,
                )
            )
        trajectory = torch.stack(predicted_trajectory, dim=1)
        multistep = torch.sum(
            torch.stack(rollout_losses) * step_weights
        )

        raw_true = batch["raw_observations"]
        target_positions = batch["target_positions"]
        with torch.no_grad():
            true_values = self.value_model.value_from_raw(raw_true)
            confidence = self.value_model.confidence_from_raw(raw_true)
            if self.competence_off_value is None:
                competence = torch.ones_like(true_values)
            else:
                competence = torch.clamp(
                    (
                        true_values - self.competence_off_value
                    ) / (
                        self.competence_on_value
                        - self.competence_off_value
                    ),
                    0.0,
                    1.0,
                )
                # Smoothstep retains exact zero/one authority outside the
                # calibrated band without introducing a hard discontinuity.
                competence = competence.square() * (
                    3.0 - 2.0 * competence
                )
            confidence = confidence * competence
        predicted_values = self.value_model.value_from_state(
            trajectory, raw_true, target_positions
        )
        normalized_value_error = (
            predicted_values - true_values
        ) / self.value_scale
        per_step_value = (
            confidence * normalized_value_error.square()
        ).sum(dim=0) / torch.clamp(confidence.sum(dim=0), min=1e-6)
        value = torch.sum(per_step_value * step_weights)
        value_rmse = torch.sqrt(torch.mean(
            (predicted_values - true_values).square()
        ))

        total = (
            self.derivative_weight * derivative
            + self.one_step_weight * one_step
            + self.multistep_weight * multistep
            + self.value_weight * value
            + self.anchor_weight * anchor
        )
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("value-aligned residual loss is non-finite")
        return {
            "total": total,
            "derivative": derivative,
            "one_step": one_step,
            "multistep": multistep,
            "value": value,
            "value_rmse": value_rmse,
            "anchor": anchor,
            "confidence_mean": confidence.mean(),
            "competence_mean": competence.mean(),
            "predicted_trajectory": trajectory,
        }


__all__ = [
    "FrozenDirectSACValue",
    "ValueAlignedResidualObjective",
]
