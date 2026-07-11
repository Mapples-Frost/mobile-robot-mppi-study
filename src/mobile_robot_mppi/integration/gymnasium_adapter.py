"""Optional Gymnasium-compatible facade without a hard Gymnasium dependency."""

import math

import numpy as np

from mobile_robot_mppi.core.types import ControlCommand


class GymnasiumAdapter:
    def __init__(self, components, config):
        self.components = components
        self.config = config
        self.observation = None
        self.steps = 0

    @staticmethod
    def observation_vector(observation):
        return np.asarray((
            observation.pose.x, observation.pose.y, observation.pose.theta,
            observation.twist.v, observation.twist.omega,
        ), dtype=np.float32)

    def reset(self, seed=None, options=None):
        options = options or {}
        experiment = self.config["experiment"]
        initial = np.asarray(options.get("initial_state", experiment.get("initial_state", (0, 0, 0))), dtype=np.float64)
        truth = self.components["plant"].reset(int(seed or experiment.get("seed", 0)), initial)
        self.observation = self.components["sensors"].reset(truth, int(seed or experiment.get("seed", 0)))
        self.steps = 0
        return self.observation_vector(self.observation), {"ground_truth": truth}

    def step(self, action):
        if self.observation is None:
            raise RuntimeError("reset must be called before step")
        perceived = self.components["perception"].process(self.observation)
        proposed = ControlCommand(np.asarray(action, dtype=np.float64), self.observation.timestamp, "rl_policy")
        decision = self.components["safety"].arbitrate(proposed, perceived.guard)
        result = self.components["plant"].step(
            decision.executed_control, float(self.config["experiment"]["control_dt"])
        )
        self.observation = self.components["sensors"].observe(result.ground_truth)
        self.steps += 1
        state = self.observation.pose.as_array()
        target = self.components["reference"].target_at(self.observation.timestamp, state)
        distance = math.hypot(target.pose.x - state[0], target.pose.y - state[1])
        reward = -distance - 0.01 * float(np.dot(decision.executed_control.values, decision.executed_control.values))
        if result.ground_truth.collision:
            reward -= 100.0
        terminated = bool(distance <= target.position_tolerance or result.ground_truth.collision)
        truncated = bool(self.steps >= int(self.config["experiment"]["max_steps"]))
        info = {
            "ground_truth": result.ground_truth,
            "proposed_action": proposed.values.copy(),
            "executed_action": decision.executed_control.values.copy(),
            "safety_override": decision.overridden,
            "safety_reason": decision.reason,
        }
        return self.observation_vector(self.observation), float(reward), terminated, truncated, info

    def close(self):
        self.components["plant"].close()
