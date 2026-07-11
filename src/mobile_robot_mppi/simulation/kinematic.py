"""Deterministic compatibility plant for legacy numerical regression."""

import math
from typing import Mapping, Sequence

import numpy as np

from mobile_robot_mppi.core.types import ControlCommand, GroundTruth, PlantStep, Pose2D, Twist2D


class KinematicPlant:
    def __init__(self, config: Mapping[str, object]):
        self.config = dict(config)
        self.dynamic = bool(config.get("dynamic", False))
        self.velocity_time_constant = float(config.get("velocity_time_constant", 0.18))
        self.yaw_time_constant = float(config.get("yaw_time_constant", 0.12))
        self.bounds = tuple(float(v) for v in config.get("bounds", (-2.0, 4.0, -2.0, 4.0)))
        self.obstacles = tuple(tuple(float(v) for v in item) for item in config.get("obstacles", ()))
        self.robot_radius = float(config.get("robot_radius", 0.25))
        self.velocity_gain = float(config.get("velocity_gain", 1.0))
        self.yaw_gain = float(config.get("yaw_gain", 1.0))
        self.yaw_bias = float(config.get("yaw_bias", 0.0))
        self._state = np.zeros(5 if self.dynamic else 3, dtype=np.float64)
        self._time = 0.0
        self._last_control = ControlCommand(np.zeros(2), 0.0, "reset")

    @staticmethod
    def _wrap(value):
        return math.atan2(math.sin(value), math.cos(value))

    def reset(self, seed: int, initial_state: np.ndarray) -> GroundTruth:
        del seed
        value = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        expected = self._state.size
        if value.size == 3 and expected == 5:
            value = np.concatenate((value, np.zeros(2, dtype=np.float64)))
        if value.shape != (expected,) or not np.isfinite(value).all():
            raise ValueError("initial state must have %d finite elements" % expected)
        self._state[:] = value
        self._time = 0.0
        self._last_control = ControlCommand(np.zeros(2), 0.0, "reset")
        return self.ground_truth()

    def _clearance(self):
        if not self.obstacles:
            return float("inf")
        x_value, y_value = self._state[:2]
        return min(
            math.hypot(x_value - obstacle[0], y_value - obstacle[1])
            - self.robot_radius - (obstacle[2] if len(obstacle) >= 3 else 0.08)
            for obstacle in self.obstacles
        )

    def ground_truth(self) -> GroundTruth:
        theta = float(self._state[2])
        if self.dynamic:
            v_value, omega = float(self._state[3]), float(self._state[4])
        else:
            v_value, omega = self._last_control.v, self._last_control.omega
        clearance = self._clearance()
        return GroundTruth(
            timestamp=self._time,
            pose=Pose2D(float(self._state[0]), float(self._state[1]), theta),
            twist=Twist2D(v_value, omega),
            collision=bool(clearance <= 0.0),
            minimum_clearance=float(clearance),
            metadata={"backend": "legacy_kinematic"},
        )

    def step(self, command: ControlCommand, dt: float) -> PlantStep:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        theta = float(self._state[2])
        if self.dynamic:
            v_value = float(self._state[3])
            omega = float(self._state[4])
            self._state[0] += v_value * math.cos(theta) * dt
            self._state[1] += v_value * math.sin(theta) * dt
            self._state[2] = self._wrap(theta + omega * dt)
            self._state[3] += dt * (command.v - v_value) / self.velocity_time_constant
            self._state[4] += dt * (command.omega - omega) / self.yaw_time_constant
        else:
            v_value = self.velocity_gain * command.v
            omega = self.yaw_gain * command.omega + self.yaw_bias
            self._state[0] += v_value * math.cos(theta) * dt
            self._state[1] += v_value * math.sin(theta) * dt
            self._state[2] = self._wrap(theta + omega * dt)
        self._time += float(dt)
        self._last_control = command
        return PlantStep(self.ground_truth(), command, float(dt))

    def close(self):
        return None
