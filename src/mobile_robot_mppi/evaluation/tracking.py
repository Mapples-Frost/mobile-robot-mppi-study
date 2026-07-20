"""Auditable constrained-path geometry and event telemetry.

The helpers in this module are evaluation-side: obstacle ground truth is used
only to label pass/recovery events and is never exposed to MPPI.  Online path
geometry comes from the same :class:`PolylineReference` used by the planner.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from mobile_robot_mppi.core.references import PolylineReference


def wrap_angle(value: float) -> float:
    """Return an angle in ``[-pi, pi]`` without a branch-cut subtraction."""

    value = float(value)
    if not np.isfinite(value):
        raise ValueError("angle must be finite")
    return float(np.arctan2(np.sin(value), np.cos(value)))


@dataclass(frozen=True)
class FootprintCorridor:
    """Symmetric path corridor with a circular robot footprint."""

    half_width: float
    footprint_radius: float
    half_width_profile: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.half_width, self.footprint_radius), dtype=np.float64
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("corridor dimensions must be positive and finite")
        if self.footprint_radius >= self.half_width:
            raise ValueError("footprint must fit inside the path corridor")

        profile = tuple(
            (float(item[0]), float(item[1]))
            for item in self.half_width_profile
        )
        if profile:
            values = np.asarray(profile, dtype=np.float64)
            if (
                values.ndim != 2
                or values.shape[1] != 2
                or not np.isfinite(values).all()
                or values[0, 0] != 0.0
                or values[-1, 0] != 1.0
                or np.any(np.diff(values[:, 0]) <= 0.0)
                or np.any(values[:, 1] <= self.footprint_radius)
            ):
                raise ValueError(
                    "corridor profile must span [0,1] with valid half widths"
                )
        object.__setattr__(self, "half_width_profile", profile)

    def half_width_at(self, path_progress_ratio: Optional[float] = None) -> float:
        if not self.half_width_profile or path_progress_ratio is None:
            return float(self.half_width)
        progress = float(path_progress_ratio)
        if not np.isfinite(progress):
            raise ValueError("path progress ratio must be finite")
        values = np.asarray(self.half_width_profile, dtype=np.float64)
        return float(np.interp(np.clip(progress, 0.0, 1.0), values[:, 0], values[:, 1]))

    def margins(
        self,
        signed_cross_track_error: float,
        path_progress_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Return footprint clearance to left and right boundaries.

        Positive cross-track error is to the path's left, so it decreases the
        left margin and increases the right margin.
        """

        error = float(signed_cross_track_error)
        if not np.isfinite(error):
            raise ValueError("signed cross-track error must be finite")
        half_width = self.half_width_at(path_progress_ratio)
        left = half_width - self.footprint_radius - error
        right = half_width - self.footprint_radius + error
        return float(left), float(right)


@dataclass(frozen=True)
class TrackingSample:
    path_arc_length: float
    path_progress_ratio: float
    signed_cross_track_error: float
    tangent_heading_error: float
    local_curvature: float
    remaining_path_length: float
    left_boundary_margin: float
    right_boundary_margin: float
    minimum_footprint_boundary_margin: float
    boundary_violation: bool
    branch_id: int = -1
    lap_progress: float = 0.0
    center_crossing_count: int = 0
    obstacle_pass_event: bool = False
    obstacle_pass_index: int = -1
    recovery_event: bool = False
    recovery_time: float = 0.0
    recovery_distance: float = 0.0

    def to_dict(self) -> Mapping[str, object]:
        return dict(self.__dict__)


def tracking_sample(
    reference: PolylineReference,
    state: Sequence[float],
    corridor: FootprintCorridor,
    minimum_progress: Optional[float] = None,
) -> TrackingSample:
    """Compute one side-effect-free constrained-path observation."""

    value = np.asarray(state, dtype=np.float64).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        raise ValueError("tracking state must contain finite x, y and heading")
    projection = reference.project(value[:2], minimum_progress=minimum_progress)
    left, right = corridor.margins(
        projection.signed_cross_track_error,
        projection.normalized_progress,
    )
    minimum = min(left, right)
    return TrackingSample(
        path_arc_length=float(projection.progress),
        path_progress_ratio=float(projection.normalized_progress),
        signed_cross_track_error=float(projection.signed_cross_track_error),
        tangent_heading_error=wrap_angle(
            float(value[2]) - projection.tangent_heading
        ),
        local_curvature=float(projection.curvature),
        remaining_path_length=float(projection.remaining),
        left_boundary_margin=left,
        right_boundary_margin=right,
        minimum_footprint_boundary_margin=float(minimum),
        boundary_violation=bool(minimum < 0.0),
        branch_id=int(projection.segment_index),
        lap_progress=float(projection.normalized_progress),
    )


class TrackingEventMonitor:
    """Stateful obstacle-pass and centreline-recovery event labeller."""

    def __init__(
        self,
        reference: PolylineReference,
        corridor: FootprintCorridor,
        obstacle_positions: Iterable[Sequence[float]] = (),
        pass_distance: float = 0.0,
        recovery_error: float = 0.12,
        center_crossing: Optional[Sequence[float]] = None,
        center_crossing_radius: float = 0.5,
    ):
        self.reference = reference
        self.corridor = corridor
        self.pass_distance = float(pass_distance)
        self.recovery_error = float(recovery_error)
        self.center_crossing_radius = float(center_crossing_radius)
        self.center_crossing = None
        if center_crossing is not None:
            value = np.asarray(center_crossing, dtype=np.float64).reshape(-1)
            if value.size < 2 or not np.isfinite(value[:2]).all():
                raise ValueError("center crossing must contain finite x/y")
            self.center_crossing = value[:2].copy()
        if (
            not np.isfinite(self.pass_distance)
            or self.pass_distance < 0.0
            or not np.isfinite(self.recovery_error)
            or self.recovery_error <= 0.0
            or not np.isfinite(self.center_crossing_radius)
            or self.center_crossing_radius <= 0.0
        ):
            raise ValueError("event thresholds must be finite and non-negative")
        positions = []
        for value in obstacle_positions:
            point = np.asarray(value, dtype=np.float64).reshape(-1)
            if point.size < 2 or not np.isfinite(point[:2]).all():
                raise ValueError("obstacle positions must contain finite x/y")
            projection = reference.project(point[:2])
            positions.append(float(projection.progress))
        self.obstacle_progress = tuple(sorted(positions))
        self.reset()

    def reset(self) -> None:
        self.progress = 0.0
        self.next_obstacle = 0
        self.recovery_start_time = None
        self.recovery_start_progress = None
        self.center_crossing_count = 0
        self._inside_center_crossing = False

    def update(self, state: Sequence[float], timestamp: float) -> TrackingSample:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("tracking timestamp must be finite and non-negative")
        sample = tracking_sample(
            self.reference, state, self.corridor, minimum_progress=self.progress
        )
        self.progress = max(self.progress, sample.path_arc_length)
        if self.center_crossing is not None:
            position = np.asarray(state, dtype=np.float64).reshape(-1)[:2]
            inside = bool(
                np.linalg.norm(position - self.center_crossing)
                <= self.center_crossing_radius
            )
            if inside and not self._inside_center_crossing:
                self.center_crossing_count += 1
            self._inside_center_crossing = inside
        passed = False
        passed_index = -1
        if self.next_obstacle < len(self.obstacle_progress):
            threshold = (
                self.obstacle_progress[self.next_obstacle] + self.pass_distance
            )
            if self.progress >= threshold:
                passed = True
                passed_index = self.next_obstacle
                self.next_obstacle += 1
                self.recovery_start_time = timestamp
                self.recovery_start_progress = self.progress

        recovered = False
        recovery_time = 0.0
        recovery_distance = 0.0
        if (
            self.recovery_start_time is not None
            and abs(sample.signed_cross_track_error) <= self.recovery_error
        ):
            recovered = True
            recovery_time = timestamp - float(self.recovery_start_time)
            recovery_distance = self.progress - float(self.recovery_start_progress)
            self.recovery_start_time = None
            self.recovery_start_progress = None
        elif self.recovery_start_time is not None:
            recovery_time = timestamp - float(self.recovery_start_time)
            recovery_distance = self.progress - float(self.recovery_start_progress)

        return TrackingSample(
            **{
                **sample.to_dict(),
                "obstacle_pass_event": passed,
                "obstacle_pass_index": passed_index,
                "recovery_event": recovered,
                "recovery_time": float(recovery_time),
                "recovery_distance": float(recovery_distance),
                "center_crossing_count": int(self.center_crossing_count),
            }
        )


def obstacle_positions_from_scene(
    obstacles: Iterable[Mapping[str, object]],
) -> Tuple[Tuple[float, float], ...]:
    """Extract static obstacle centres for offline event labelling."""

    positions = []
    for obstacle in obstacles:
        if bool(obstacle.get("dynamic", False)):
            continue
        position = obstacle.get("position")
        if position is None:
            continue
        value = np.asarray(position, dtype=np.float64).reshape(-1)
        if value.size >= 2 and np.isfinite(value[:2]).all():
            positions.append((float(value[0]), float(value[1])))
    return tuple(positions)
