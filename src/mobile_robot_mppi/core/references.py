"""Variable point, pose, waypoint, and time-trajectory references."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Pose2D


@dataclass(frozen=True)
class ReferenceTarget:
    pose: Pose2D
    position_tolerance: float = 0.2
    heading_tolerance: Optional[float] = None
    reference_id: str = "reference"
    is_terminal: bool = True
    phase: str = "terminal"


@dataclass(frozen=True)
class PointGoal:
    x: float
    y: float
    position_tolerance: float = 0.2
    reference_id: str = "point_goal"

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del time
        theta = float(state[2]) if np.asarray(state).size >= 3 else 0.0
        return ReferenceTarget(
            Pose2D(float(self.x), float(self.y), theta),
            position_tolerance=float(self.position_tolerance),
            reference_id=self.reference_id,
        )


@dataclass(frozen=True)
class PoseGoal:
    x: float
    y: float
    theta: float
    position_tolerance: float = 0.2
    heading_tolerance: float = 0.15
    reference_id: str = "pose_goal"

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del time, state
        return ReferenceTarget(
            Pose2D(float(self.x), float(self.y), float(self.theta)),
            position_tolerance=float(self.position_tolerance),
            heading_tolerance=float(self.heading_tolerance),
            reference_id=self.reference_id,
        )


class WaypointReference:
    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        tolerance: float = 0.2,
        waypoint_tolerance: Optional[float] = None,
        terminal_approach_count: int = 1,
    ):
        if not waypoints:
            raise ValueError("at least one waypoint is required")
        self.waypoints = tuple(tuple(float(v) for v in point) for point in waypoints)
        self.tolerance = float(tolerance)
        self.waypoint_tolerance = float(
            tolerance if waypoint_tolerance is None else waypoint_tolerance
        )
        if self.tolerance <= 0.0 or self.waypoint_tolerance <= 0.0:
            raise ValueError("waypoint tolerances must be positive")
        self.terminal_approach_count = int(terminal_approach_count)
        if not 1 <= self.terminal_approach_count <= len(self.waypoints):
            raise ValueError("terminal_approach_count must be within waypoint count")
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del time
        state_value = np.asarray(state, dtype=np.float64)
        while self.index < len(self.waypoints) - 1:
            point = self.waypoints[self.index]
            if (
                np.linalg.norm(state_value[:2] - np.asarray(point[:2]))
                > self.waypoint_tolerance
            ):
                break
            self.index += 1
        point = self.waypoints[self.index]
        theta = point[2] if len(point) >= 3 else float(state_value[2])
        target_tolerance = (
            self.tolerance
            if self.index == len(self.waypoints) - 1
            else self.waypoint_tolerance
        )
        final_index = len(self.waypoints) - 1
        approach_start = max(0, len(self.waypoints) - self.terminal_approach_count)
        phase = (
            "terminal"
            if self.index == final_index
            else "terminal_approach"
            if self.index >= approach_start
            else "tracking"
        )
        return ReferenceTarget(
            Pose2D(point[0], point[1], theta),
            position_tolerance=target_tolerance,
            reference_id="waypoint_%d" % self.index,
            is_terminal=self.index == final_index,
            phase=phase,
        )


class PolylineReference:
    """Monotonic progress reference with a metric lookahead target.

    Unlike radial waypoint switching, projection onto the route cannot leave
    the controller chasing a point that the physical plant has already passed.
    The route remains a task reference; it does not expose obstacles to MPPI.
    """

    def __init__(
        self,
        points: Sequence[Sequence[float]],
        tolerance: float = 0.2,
        lookahead_distance: float = 0.45,
        terminal_approach_distance: float = 0.9,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[0] < 2 or self.points.shape[1] < 2:
            raise ValueError("polyline points must have shape [N,2+] with N >= 2")
        if not np.isfinite(self.points).all():
            raise ValueError("polyline points must be finite")
        self.points = self.points[:, :2].copy()
        self.tolerance = float(tolerance)
        self.lookahead_distance = float(lookahead_distance)
        self.terminal_approach_distance = float(terminal_approach_distance)
        if min(self.tolerance, self.lookahead_distance, self.terminal_approach_distance) <= 0.0:
            raise ValueError("polyline tolerances and distances must be positive")
        self.segment_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        if np.any(self.segment_lengths <= 1e-9):
            raise ValueError("polyline cannot contain duplicate consecutive points")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.total_length = float(self.cumulative[-1])
        self.progress = 0.0

    def reset(self) -> None:
        self.progress = 0.0

    def _project_progress(self, position: np.ndarray) -> float:
        starts = self.points[:-1]
        vectors = np.diff(self.points, axis=0)
        fractions = np.sum((position[None, :] - starts) * vectors, axis=1)
        fractions /= self.segment_lengths ** 2
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(projections - position[None, :], axis=1)
        candidate_progress = self.cumulative[:-1] + fractions * self.segment_lengths
        # Never jump to an earlier branch of a route that passes near itself.
        admissible = candidate_progress >= self.progress - self.lookahead_distance
        if not np.any(admissible):
            return self.progress
        masked = np.where(admissible, distances, np.inf)
        candidate = float(candidate_progress[int(np.argmin(masked))])
        return max(self.progress, candidate)

    def _point_at_progress(self, progress: float):
        value = float(np.clip(progress, 0.0, self.total_length))
        index = min(
            int(np.searchsorted(self.cumulative, value, side="right") - 1),
            len(self.segment_lengths) - 1,
        )
        fraction = (value - self.cumulative[index]) / self.segment_lengths[index]
        point = self.points[index] + fraction * (self.points[index + 1] - self.points[index])
        tangent = self.points[index + 1] - self.points[index]
        theta = float(np.arctan2(tangent[1], tangent[0]))
        return point, theta

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del time
        state_value = np.asarray(state, dtype=np.float64)
        if state_value.size < 2 or not np.isfinite(state_value[:2]).all():
            raise ValueError("polyline reference state must contain finite x/y")
        self.progress = self._project_progress(state_value[:2])
        target_progress = min(self.total_length, self.progress + self.lookahead_distance)
        point, theta = self._point_at_progress(target_progress)
        remaining = self.total_length - self.progress
        is_terminal = bool(target_progress >= self.total_length - 1e-9)
        phase = (
            "terminal"
            if is_terminal
            else "terminal_approach"
            if remaining <= self.terminal_approach_distance
            else "tracking"
        )
        return ReferenceTarget(
            Pose2D(float(point[0]), float(point[1]), theta),
            position_tolerance=self.tolerance if is_terminal else 0.0,
            reference_id="polyline_%.3f" % target_progress,
            is_terminal=is_terminal,
            phase=phase,
        )


class TimeTrajectoryReference:
    def __init__(self, times: Sequence[float], poses: Sequence[Sequence[float]], tolerance=0.2):
        self.times = np.asarray(times, dtype=np.float64)
        self.poses = np.asarray(poses, dtype=np.float64)
        if self.times.ndim != 1 or self.poses.shape != (self.times.size, 3):
            raise ValueError("trajectory times/poses must have shapes [N] and [N,3]")
        if self.times.size < 2 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")
        self.tolerance = float(tolerance)

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del state
        t = float(np.clip(time, self.times[0], self.times[-1]))
        x = float(np.interp(t, self.times, self.poses[:, 0]))
        y = float(np.interp(t, self.times, self.poses[:, 1]))
        sin_theta = float(np.interp(t, self.times, np.sin(self.poses[:, 2])))
        cos_theta = float(np.interp(t, self.times, np.cos(self.poses[:, 2])))
        return ReferenceTarget(
            Pose2D(x, y, float(np.arctan2(sin_theta, cos_theta))),
            position_tolerance=self.tolerance,
            reference_id="time_trajectory",
            is_terminal=bool(t >= self.times[-1]),
            phase="terminal" if t >= self.times[-1] else "tracking",
        )


def reference_from_config(config: Mapping[str, object]):
    kind = str(config.get("type", "point_goal"))
    tolerance = float(config.get("position_tolerance", 0.2))
    if kind == "point_goal":
        position = config.get("position", (3.0, 3.0))
        return PointGoal(position[0], position[1], tolerance)
    if kind == "pose_goal":
        pose = config["pose"]
        return PoseGoal(
            pose[0], pose[1], pose[2], tolerance,
            float(config.get("heading_tolerance", 0.15)),
        )
    if kind == "waypoints":
        return WaypointReference(
            config["waypoints"], tolerance,
            waypoint_tolerance=config.get("waypoint_tolerance"),
            terminal_approach_count=int(config.get("terminal_approach_count", 1)),
        )
    if kind == "polyline":
        return PolylineReference(
            config["points"],
            tolerance,
            lookahead_distance=float(config.get("lookahead_distance", 0.45)),
            terminal_approach_distance=float(
                config.get("terminal_approach_distance", 0.9)
            ),
        )
    if kind == "time_trajectory":
        return TimeTrajectoryReference(config["times"], config["poses"], tolerance)
    raise ValueError("unknown reference type: %s" % kind)
