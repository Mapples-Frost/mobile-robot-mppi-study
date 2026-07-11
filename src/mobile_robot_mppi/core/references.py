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
    def __init__(self, waypoints: Sequence[Sequence[float]], tolerance: float = 0.2):
        if not waypoints:
            raise ValueError("at least one waypoint is required")
        self.waypoints = tuple(tuple(float(v) for v in point) for point in waypoints)
        self.tolerance = float(tolerance)
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del time
        state_value = np.asarray(state, dtype=np.float64)
        while self.index < len(self.waypoints) - 1:
            point = self.waypoints[self.index]
            if np.linalg.norm(state_value[:2] - np.asarray(point[:2])) > self.tolerance:
                break
            self.index += 1
        point = self.waypoints[self.index]
        theta = point[2] if len(point) >= 3 else float(state_value[2])
        return ReferenceTarget(
            Pose2D(point[0], point[1], theta),
            position_tolerance=self.tolerance,
            reference_id="waypoint_%d" % self.index,
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
        return WaypointReference(config["waypoints"], tolerance)
    if kind == "time_trajectory":
        return TimeTrajectoryReference(config["times"], config["poses"], tolerance)
    raise ValueError("unknown reference type: %s" % kind)
