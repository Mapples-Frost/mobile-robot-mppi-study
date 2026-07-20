"""Privileged offline polyline teacher for direct-control Actor datasets.

The route is used only while collecting demonstrations. Student shards hold
normal environment observations and normalized ``(v, omega)`` actions; route
geometry and target diagnostics remain in the audit sidecar.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .parameterization import PriorParameterizationConfig
from .scripted_subgoal import ScriptedPolylineSubgoal, ScriptedSubgoalConfig


@dataclass(frozen=True)
class ScriptedDirectControlConfig:
    lookahead_distance: float = 0.70
    cruise_speed: float = 0.28
    yaw_gain: float = 1.8
    heading_stop_angle: float = 1.35
    terminal_slow_distance: float = 0.45

    def validate(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("scripted direct-control values must be finite")
        if self.lookahead_distance <= 0.0 or self.cruise_speed <= 0.0:
            raise ValueError("lookahead and cruise speed must be positive")
        if self.yaw_gain <= 0.0 or not 0.0 < self.heading_stop_angle <= np.pi:
            raise ValueError("yaw gain/heading stop angle are invalid")
        if self.terminal_slow_distance <= 0.0:
            raise ValueError("terminal slow distance must be positive")


class ScriptedPolylineDirectControl:
    """Map a privileged reference polyline to normalized direct controls."""

    def __init__(self, points, action_spec, config=None):
        self.config = config or ScriptedDirectControlConfig()
        if not isinstance(self.config, ScriptedDirectControlConfig):
            self.config = ScriptedDirectControlConfig(**dict(self.config))
        self.config.validate()
        self.lower = np.asarray(action_spec.lower, dtype=np.float64).reshape(-1)
        self.upper = np.asarray(action_spec.upper, dtype=np.float64).reshape(-1)
        if (
            self.lower.shape != (2,)
            or self.upper.shape != (2,)
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.upper <= self.lower)
        ):
            raise ValueError("direct-control teacher requires a finite 2-D action spec")
        if self.lower[0] < -1e-9:
            raise ValueError("direct-control teacher currently requires non-negative v")
        tracker_prior = PriorParameterizationConfig(
            kind="local_subgoal", num_knots=2, learn_covariance=False
        )
        self.tracker = ScriptedPolylineSubgoal(
            points,
            tracker_prior,
            ScriptedSubgoalConfig(
                lookahead_distance=self.config.lookahead_distance
            ),
        )

    def reset(self):
        self.tracker.reset()

    def _normalize(self, physical):
        center = 0.5 * (self.lower + self.upper)
        half_range = 0.5 * (self.upper - self.lower)
        return np.clip((physical - center) / half_range, -1.0, 1.0)

    def action(self, pose):
        _, diagnostic = self.tracker.action(pose)
        bearing = float(diagnostic["target_bearing"])
        heading_scale = float(np.clip(
            1.0 - abs(bearing) / self.config.heading_stop_angle, 0.0, 1.0
        ))
        remaining = max(
            0.0,
            float(diagnostic["route_length"])
            - float(diagnostic["route_progress"]),
        )
        terminal_scale = float(np.clip(
            remaining / self.config.terminal_slow_distance, 0.0, 1.0
        ))
        desired_v = self.config.cruise_speed * heading_scale * terminal_scale
        desired_omega = self.config.yaw_gain * bearing
        physical = np.clip(
            np.asarray((desired_v, desired_omega), dtype=np.float64),
            self.lower,
            self.upper,
        )
        action = self._normalize(physical).astype(np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise FloatingPointError("direct-control teacher produced invalid action")
        result = dict(diagnostic)
        result.update({
            "desired_v": float(physical[0]),
            "desired_omega": float(physical[1]),
            "heading_speed_scale": heading_scale,
            "terminal_speed_scale": terminal_scale,
        })
        return action, result
