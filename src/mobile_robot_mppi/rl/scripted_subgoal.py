"""Privileged local-subgoal policy for interface upper-bound diagnostics.

This module is deliberately not a deployable planner.  It follows an offline
polyline to answer a causal engineering question: can the exact local-subgoal
parameterization used by SAC guide MPPI when the proposed subgoal is good?
The polyline must never be presented as a learned-policy input or a production
obstacle source.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .parameterization import (
    PriorParameterizationConfig,
    encode_local_subgoal_action,
)


@dataclass(frozen=True)
class ScriptedSubgoalConfig:
    lookahead_distance: float = 0.70
    projection_backtrack: float = 0.20
    maximum_projection_advance: float = 1.50

    def validate(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("scripted-subgoal values must be finite")
        if self.lookahead_distance <= 0.0:
            raise ValueError("scripted-subgoal lookahead must be positive")
        if self.projection_backtrack < 0.0:
            raise ValueError("scripted-subgoal projection backtrack cannot be negative")
        if self.maximum_projection_advance <= 0.0:
            raise ValueError("scripted-subgoal projection advance must be positive")


class ScriptedPolylineSubgoal:
    """Convert progress along a privileged polyline into normalized RL action.

    Projection progress is monotonic and locally bounded so a route that passes
    near itself cannot silently jump to a later branch.  The output has exactly
    the same two normalized values consumed by ``PriorParameterization``.
    """

    def __init__(self, points, prior_config, config=None):
        self.points = np.asarray(points, dtype=np.float64)
        if (
            self.points.ndim != 2
            or self.points.shape[0] < 2
            or self.points.shape[1] != 2
            or not np.isfinite(self.points).all()
        ):
            raise ValueError("scripted-subgoal polyline must be finite [N, 2]")
        self.prior_config = (
            prior_config
            if isinstance(prior_config, PriorParameterizationConfig)
            else PriorParameterizationConfig.from_mapping(prior_config)
        )
        self.prior_config.validate()
        if self.prior_config.kind != "local_subgoal":
            raise ValueError("scripted-subgoal diagnostic requires local_subgoal prior")
        if self.prior_config.learn_covariance:
            raise ValueError("scripted-subgoal diagnostic does not choose covariance")
        self.config = config or ScriptedSubgoalConfig()
        if not isinstance(self.config, ScriptedSubgoalConfig):
            self.config = ScriptedSubgoalConfig(**dict(self.config))
        self.config.validate()
        self.segments = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(self.segments, axis=1)
        if np.any(self.segment_lengths <= 1e-9):
            raise ValueError("scripted-subgoal polyline contains a zero-length segment")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.total_length = float(self.cumulative[-1])
        self.progress = 0.0

    def reset(self):
        self.progress = 0.0

    @staticmethod
    def _wrap(angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _project(self, position):
        lower = max(0.0, self.progress - self.config.projection_backtrack)
        upper = min(
            self.total_length,
            self.progress + self.config.maximum_projection_advance,
        )
        best_distance = float("inf")
        best_progress = self.progress
        for index, (start, vector, length) in enumerate(
            zip(self.points[:-1], self.segments, self.segment_lengths)
        ):
            segment_start = float(self.cumulative[index])
            segment_end = float(self.cumulative[index + 1])
            if segment_end < lower or segment_start > upper:
                continue
            raw_fraction = float(
                np.dot(position - start, vector) / max(length * length, 1e-12)
            )
            minimum_fraction = max(0.0, (lower - segment_start) / length)
            maximum_fraction = min(1.0, (upper - segment_start) / length)
            fraction = float(np.clip(
                raw_fraction, minimum_fraction, maximum_fraction
            ))
            projected = start + fraction * vector
            distance = float(np.linalg.norm(position - projected))
            candidate_progress = segment_start + fraction * length
            if distance < best_distance:
                best_distance = distance
                best_progress = candidate_progress
        self.progress = max(self.progress, float(best_progress))
        return best_distance

    def _point_at(self, progress):
        progress = float(np.clip(progress, 0.0, self.total_length))
        index = min(
            int(np.searchsorted(self.cumulative, progress, side="right") - 1),
            len(self.segments) - 1,
        )
        offset = progress - float(self.cumulative[index])
        fraction = offset / float(self.segment_lengths[index])
        return self.points[index] + fraction * self.segments[index]

    def action(self, pose):
        position = np.asarray((float(pose.x), float(pose.y)), dtype=np.float64)
        cross_track_error = self._project(position)
        target_progress = min(
            self.total_length,
            self.progress + self.config.lookahead_distance,
        )
        target = self._point_at(target_progress)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        world_bearing = float(np.arctan2(delta[1], delta[0]))
        body_bearing = self._wrap(world_bearing - float(pose.theta))
        action = encode_local_subgoal_action(
            self.prior_config, distance, body_bearing
        )
        if action.shape != (2,) or not np.isfinite(action).all():
            raise FloatingPointError("scripted-subgoal policy produced invalid action")
        metadata = {
            "route_progress": float(self.progress),
            "target_progress": float(target_progress),
            "route_length": self.total_length,
            "cross_track_error": float(cross_track_error),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "target_distance": distance,
            "target_bearing": body_bearing,
        }
        return action, metadata
