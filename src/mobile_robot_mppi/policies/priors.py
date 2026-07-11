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
    def __init__(self, v_gain=0.8, yaw_gain=1.2):
        self.v_gain = float(v_gain)
        self.yaw_gain = float(yaw_gain)

    def propose(self, observation: RobotObservation, reference, horizon, action_spec):
        state = observation.pose.as_array()
        target = reference.target_at(observation.timestamp, state).pose
        dx = target.x - observation.pose.x
        dy = target.y - observation.pose.y
        distance = float(np.hypot(dx, dy))
        desired = float(np.arctan2(dy, dx))
        yaw_error = float(np.arctan2(
            np.sin(desired - observation.pose.theta),
            np.cos(desired - observation.pose.theta),
        ))
        action = np.zeros(action_spec.dimension, dtype=np.float64)
        if "v_cmd" in action_spec.names:
            action[action_spec.index("v_cmd")] = self.v_gain * distance * max(0.0, np.cos(yaw_error))
        if "omega_cmd" in action_spec.names:
            action[action_spec.index("omega_cmd")] = self.yaw_gain * yaw_error
        action = action_spec.clip(action)
        return PriorOutput(
            mean=np.repeat(action[None, :], int(horizon), axis=0),
            metadata={"type": "goal_warm_start", "yaw_error": yaw_error},
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
