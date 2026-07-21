"""Torch-free sampling priors, including the stable future RL policy port."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import RobotObservation


@dataclass(frozen=True)
class ProposalDistribution:
    """One auditable candidate source for mixture-based MPPI optimizers."""

    label: str
    mean: np.ndarray
    covariance: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PriorOutput:
    mean: np.ndarray
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    proposals: Tuple[ProposalDistribution, ...] = ()


class ZeroPrior:
    def propose(self, observation, reference, horizon, action_spec):
        del observation, reference
        return PriorOutput(
            mean=np.zeros((int(horizon), action_spec.dimension), dtype=np.float64),
            metadata={"type": "zero"},
        )


class PreviousSequencePrior:
    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def set_previous(self, sequence):
        self.previous = np.asarray(sequence, dtype=np.float64).copy()

    def propose(self, observation, reference, horizon, action_spec):
        del observation, reference
        shape = (int(horizon), action_spec.dimension)
        if self.previous is None or self.previous.shape != shape:
            mean = np.zeros(shape, dtype=np.float64)
        else:
            mean = np.empty_like(self.previous)
            mean[:-1] = self.previous[1:]
            mean[-1] = self.previous[-1]
        return PriorOutput(mean=mean, metadata={"type": "previous_sequence"})


class GoalWarmStartPrior:
    def __init__(
        self,
        v_gain=0.8,
        yaw_gain=1.2,
        translation_heading_gate_rad=None,
        translation_heading_gate_terminal_only=False,
        terminal_max_speed=None,
        path_rollout_enabled=False,
        rollout_dt=0.1,
    ):
        self.v_gain = float(v_gain)
        self.yaw_gain = float(yaw_gain)
        self.translation_heading_gate_rad = (
            None
            if translation_heading_gate_rad is None
            else float(translation_heading_gate_rad)
        )
        if (
            self.translation_heading_gate_rad is not None
            and not 0.0 < self.translation_heading_gate_rad <= np.pi
        ):
            raise ValueError("translation heading gate must be in (0, pi]")
        self.translation_heading_gate_terminal_only = bool(
            translation_heading_gate_terminal_only
        )
        self.terminal_max_speed = (
            None if terminal_max_speed is None else float(terminal_max_speed)
        )
        if self.terminal_max_speed is not None and (
            not np.isfinite(self.terminal_max_speed)
            or self.terminal_max_speed <= 0.0
        ):
            raise ValueError("terminal_max_speed must be finite and positive")
        self.path_rollout_enabled = bool(path_rollout_enabled)
        self.rollout_dt = float(rollout_dt)
        if not np.isfinite(self.rollout_dt) or self.rollout_dt <= 0.0:
            raise ValueError("warm-start rollout dt must be finite and positive")

    def _command(self, x, y, theta, target, action_spec):
        dx = target.pose.x - float(x)
        dy = target.pose.y - float(y)
        distance = float(np.hypot(dx, dy))
        desired = float(np.arctan2(dy, dx))
        yaw_error = float(np.arctan2(
            np.sin(desired - float(theta)),
            np.cos(desired - float(theta)),
        ))
        action = np.zeros(action_spec.dimension, dtype=np.float64)
        alignment = 0.0
        gate_active = False
        terminal_speed_cap_active = False
        if "v_cmd" in action_spec.names:
            alignment = max(0.0, float(np.cos(yaw_error)))
            gate_active = (
                self.translation_heading_gate_rad is not None
                and (
                    not self.translation_heading_gate_terminal_only
                    or target.phase in ("terminal_approach", "terminal")
                )
            )
            if gate_active:
                gate_cosine = float(np.cos(self.translation_heading_gate_rad))
                if abs(yaw_error) >= self.translation_heading_gate_rad:
                    alignment = 0.0
                else:
                    alignment = max(
                        0.0,
                        (float(np.cos(yaw_error)) - gate_cosine)
                        / max(1.0 - gate_cosine, 1e-12),
                    )
            v_value = self.v_gain * distance * alignment
            terminal_speed_cap_active = bool(
                self.terminal_max_speed is not None
                and target.phase in ("terminal_approach", "terminal")
            )
            if terminal_speed_cap_active:
                v_value = min(v_value, self.terminal_max_speed)
            action[action_spec.index("v_cmd")] = v_value
        if "omega_cmd" in action_spec.names:
            action[action_spec.index("omega_cmd")] = self.yaw_gain * yaw_error
        return (
            action_spec.clip(action),
            yaw_error,
            alignment,
            gate_active,
            terminal_speed_cap_active,
        )

    def propose(self, observation: RobotObservation, reference, horizon, action_spec):
        state = observation.pose.as_array()
        target = reference.target_at(observation.timestamp, state)
        action, yaw_error, alignment, gate_active, speed_cap = self._command(
            observation.pose.x,
            observation.pose.y,
            observation.pose.theta,
            target,
            action_spec,
        )
        mean = np.repeat(action[None, :], int(horizon), axis=0)
        supports_path_rollout = all(callable(getattr(reference, name, None)) for name in (
            "preview_target_at", "project",
        )) and hasattr(reference, "progress")
        if self.path_rollout_enabled and supports_path_rollout:
            mean = np.zeros(
                (int(horizon), action_spec.dimension), dtype=np.float64
            )
            virtual = np.asarray((
                observation.pose.x,
                observation.pose.y,
                observation.pose.theta,
            ), dtype=np.float64)
            progress_floor = float(reference.progress)
            previous = np.zeros(action_spec.dimension, dtype=np.float64)
            if "v_cmd" in action_spec.names:
                previous[action_spec.index("v_cmd")] = observation.twist.v
            if "omega_cmd" in action_spec.names:
                previous[action_spec.index("omega_cmd")] = observation.twist.omega
            for step in range(int(horizon)):
                step_target = reference.preview_target_at(
                    observation.timestamp + step * self.rollout_dt,
                    virtual,
                    progress_floor=progress_floor,
                )
                command = self._command(
                    virtual[0], virtual[1], virtual[2], step_target, action_spec
                )[0]
                command = action_spec.clip(
                    command, previous=previous, dt=self.rollout_dt
                )
                mean[step] = command
                v_value = (
                    command[action_spec.index("v_cmd")]
                    if "v_cmd" in action_spec.names else 0.0
                )
                omega_value = (
                    command[action_spec.index("omega_cmd")]
                    if "omega_cmd" in action_spec.names else 0.0
                )
                midpoint_theta = virtual[2] + 0.5 * self.rollout_dt * omega_value
                virtual[0] += self.rollout_dt * v_value * np.cos(midpoint_theta)
                virtual[1] += self.rollout_dt * v_value * np.sin(midpoint_theta)
                virtual[2] = np.arctan2(
                    np.sin(virtual[2] + self.rollout_dt * omega_value),
                    np.cos(virtual[2] + self.rollout_dt * omega_value),
                )
                progress_floor = reference.project(
                    virtual[:2], minimum_progress=progress_floor
                ).progress
                previous = command
        return PriorOutput(
            mean=mean,
            metadata={
                "type": (
                    "path_rollout_warm_start"
                    if self.path_rollout_enabled and supports_path_rollout
                    else "goal_warm_start"
                ),
                "yaw_error": yaw_error,
                "translation_alignment": alignment,
                "translation_heading_gate_rad": self.translation_heading_gate_rad,
                "translation_heading_gate_active": gate_active,
                "target_is_terminal": target.is_terminal,
                "target_phase": target.phase,
                "terminal_max_speed": self.terminal_max_speed,
                "terminal_speed_cap_active": speed_cap,
                "path_rollout_enabled": bool(
                    self.path_rollout_enabled and supports_path_rollout
                ),
                "path_rollout_dt": self.rollout_dt,
            },
        )


class HybridBaselinePrior:
    """Non-learning control for hybrid mixture and elite-update effects.

    Both named proposal ports contain the same conventional warm start. This
    isolates any gain caused by the optimizer from a gain caused by the learned
    RL proposal itself.
    """

    def __init__(self, baseline_prior=None):
        self.baseline_prior = baseline_prior or GoalWarmStartPrior()

    def reset(self):
        reset = getattr(self.baseline_prior, "reset", None)
        if callable(reset):
            reset()

    def propose(self, observation, reference, horizon, action_spec):
        baseline = self.baseline_prior.propose(
            observation, reference, horizon, action_spec
        )
        metadata = dict(baseline.metadata)
        metadata["type"] = "hybrid_baseline_control"
        return PriorOutput(
            baseline.mean,
            baseline.covariance,
            metadata,
            proposals=(
                ProposalDistribution(
                    "rl", baseline.mean, baseline.covariance
                ),
                ProposalDistribution(
                    "base", baseline.mean, baseline.covariance
                ),
            ),
        )


class FixedCovariancePrior:
    """Apply fixed MPPI sampling scales without changing a baseline mean.

    ``scale`` multiplies standard deviation, not variance.  This prior is the
    non-learning comparator for adaptive-covariance policies.
    """

    def __init__(self, baseline_prior, noise_sigma, scale):
        self.baseline_prior = baseline_prior
        sigma = np.asarray(noise_sigma, dtype=np.float64).reshape(-1)
        value = np.asarray(scale, dtype=np.float64).reshape(-1)
        if sigma.size == 0 or value.shape != sigma.shape:
            raise ValueError("fixed covariance scale must match noise_sigma")
        if (
            not np.isfinite(sigma).all()
            or not np.isfinite(value).all()
            or np.any(sigma <= 0.0)
            or np.any(value <= 0.0)
        ):
            raise ValueError("fixed covariance sigma and scale must be positive")
        self.noise_sigma = sigma
        self.scale = value

    def reset(self):
        reset = getattr(self.baseline_prior, "reset", None)
        if callable(reset):
            reset()

    def propose(self, observation, reference, horizon, action_spec):
        baseline = self.baseline_prior.propose(
            observation, reference, horizon, action_spec
        )
        if self.scale.shape != (action_spec.dimension,):
            raise ValueError("fixed covariance does not match the action space")
        standard_deviation = self.noise_sigma * self.scale
        covariance = np.diag(np.square(standard_deviation))
        metadata = dict(baseline.metadata)
        metadata.update({
            "type": "fixed_covariance",
            "covariance_scale": self.scale.tolist(),
            "covariance_standard_deviation": standard_deviation.tolist(),
        })
        return PriorOutput(
            baseline.mean,
            covariance,
            metadata,
            proposals=(
                ProposalDistribution("rl", baseline.mean, covariance),
                ProposalDistribution(
                    "base", baseline.mean, baseline.covariance
                ),
            ),
        )


class RLPolicyPrior:
    """Adapter for an RL policy without importing a policy framework.

    The callable receives observation/reference/horizon/action_spec and may
    return a PriorOutput, a mean sequence, or ``(mean, covariance)``.
    Safety arbitration remains downstream of MPPI and cannot be bypassed.
    """

    def __init__(self, policy: Callable[..., Any], policy_id="rl_policy"):
        self.policy = policy
        self.policy_id = str(policy_id)

    def propose(self, observation, reference, horizon, action_spec):
        output = self.policy(observation, reference, int(horizon), action_spec)
        if isinstance(output, PriorOutput):
            return output
        covariance = None
        mean = output
        if isinstance(output, tuple) and len(output) == 2:
            mean, covariance = output
        mean = np.asarray(mean, dtype=np.float64)
        expected = (int(horizon), action_spec.dimension)
        if mean.shape != expected or not np.isfinite(mean).all():
            raise ValueError("RL prior mean must be finite with shape %r" % (expected,))
        mean = np.clip(mean, action_spec.lower, action_spec.upper)
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
        return PriorOutput(mean, covariance, {"type": "rl", "policy_id": self.policy_id})
