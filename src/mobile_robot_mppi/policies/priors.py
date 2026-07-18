"""Torch-free sampling priors, including the stable future RL policy port."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import RobotObservation


@dataclass(frozen=True)
class PriorOutput:
    mean: np.ndarray
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def propose(self, observation: RobotObservation, reference, horizon, action_spec):
        state = observation.pose.as_array()
        target = reference.target_at(observation.timestamp, state)
        dx = target.pose.x - observation.pose.x
        dy = target.pose.y - observation.pose.y
        distance = float(np.hypot(dx, dy))
        desired = float(np.arctan2(dy, dx))
        yaw_error = float(np.arctan2(
            np.sin(desired - observation.pose.theta),
            np.cos(desired - observation.pose.theta),
        ))
        action = np.zeros(action_spec.dimension, dtype=np.float64)
        alignment = 0.0
        gate_active = False
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
                    # Smoothly recover translation after an in-place turn;
                    # discontinuous on/off motion excites the actuator loop.
                    alignment = max(
                        0.0,
                        (float(np.cos(yaw_error)) - gate_cosine)
                        / max(1.0 - gate_cosine, 1e-12),
                    )
            v_value = self.v_gain * distance * alignment
            if (
                self.terminal_max_speed is not None
                and target.phase in ("terminal_approach", "terminal")
            ):
                v_value = min(v_value, self.terminal_max_speed)
            action[action_spec.index("v_cmd")] = v_value
        if "omega_cmd" in action_spec.names:
            action[action_spec.index("omega_cmd")] = self.yaw_gain * yaw_error
        action = action_spec.clip(action)
        return PriorOutput(
            mean=np.repeat(action[None, :], int(horizon), axis=0),
            metadata={
                "type": "goal_warm_start",
                "yaw_error": yaw_error,
                "translation_alignment": alignment,
                "translation_heading_gate_rad": self.translation_heading_gate_rad,
                "translation_heading_gate_active": gate_active,
                "target_is_terminal": target.is_terminal,
                "target_phase": target.phase,
                "terminal_max_speed": self.terminal_max_speed,
                "terminal_speed_cap_active": bool(
                    self.terminal_max_speed is not None
                    and target.phase in ("terminal_approach", "terminal")
                ),
            },
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
        return PriorOutput(baseline.mean, covariance, metadata)


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
