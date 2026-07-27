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
class PolylineProjection:
    """Read-only geometric projection of a pose onto a polyline.

    ``signed_cross_track_error`` is positive on the left side of the local
    path tangent and negative on the right.  Curvature is a finite-difference
    estimate at the active segment; it is an observable path descriptor, not
    a vehicle-state estimate.
    """

    point: Tuple[float, float]
    progress: float
    remaining: float
    normalized_progress: float
    segment_index: int
    tangent_heading: float
    cross_track_error: float
    signed_cross_track_error: float
    curvature: float


@dataclass(frozen=True)
class PolylineProjectionBatch:
    """Vectorized read-only projections with one leading batch dimension."""

    point: np.ndarray
    progress: np.ndarray
    remaining: np.ndarray
    normalized_progress: np.ndarray
    segment_index: np.ndarray
    tangent_heading: np.ndarray
    cross_track_error: np.ndarray
    signed_cross_track_error: np.ndarray
    curvature: np.ndarray


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
        projection_backtrack_distance: Optional[float] = None,
        projection_forward_distance: Optional[float] = None,
        corridor_half_width: Optional[float] = None,
        footprint_radius: Optional[float] = None,
        corridor_half_width_profile: Sequence[Sequence[float]] = (),
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
        self.projection_backtrack_distance = float(
            self.lookahead_distance
            if projection_backtrack_distance is None
            else projection_backtrack_distance
        )
        self.projection_forward_distance = (
            float("inf")
            if projection_forward_distance is None
            else float(projection_forward_distance)
        )
        self.corridor_half_width = (
            None if corridor_half_width is None else float(corridor_half_width)
        )
        self.footprint_radius = (
            None if footprint_radius is None else float(footprint_radius)
        )
        self.corridor_half_width_profile = tuple(
            (float(item[0]), float(item[1]))
            for item in corridor_half_width_profile
        )
        if min(self.tolerance, self.lookahead_distance, self.terminal_approach_distance) <= 0.0:
            raise ValueError("polyline tolerances and distances must be positive")
        if (
            not np.isfinite(self.projection_backtrack_distance)
            or self.projection_backtrack_distance < 0.0
            or self.projection_forward_distance <= 0.0
            or np.isnan(self.projection_forward_distance)
        ):
            raise ValueError("polyline projection window must be positive")
        if (self.corridor_half_width is None) != (self.footprint_radius is None):
            raise ValueError(
                "polyline corridor width and footprint radius must be configured together"
            )
        if self.corridor_half_width is not None and (
            not np.isfinite(self.corridor_half_width)
            or not np.isfinite(self.footprint_radius)
            or self.corridor_half_width <= 0.0
            or self.footprint_radius < 0.0
            or self.footprint_radius >= self.corridor_half_width
        ):
            raise ValueError("polyline corridor must contain the configured footprint")
        if self.corridor_half_width_profile:
            profile = np.asarray(self.corridor_half_width_profile, dtype=np.float64)
            if (
                self.corridor_half_width is None
                or profile.ndim != 2
                or profile.shape[1] != 2
                or not np.isfinite(profile).all()
                or profile[0, 0] != 0.0
                or profile[-1, 0] != 1.0
                or np.any(np.diff(profile[:, 0]) <= 0.0)
                or np.any(profile[:, 1] <= self.footprint_radius)
            ):
                raise ValueError(
                    "corridor width profile must span [0,1] with feasible widths"
                )
        self.segment_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        if np.any(self.segment_lengths <= 1e-9):
            raise ValueError("polyline cannot contain duplicate consecutive points")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.total_length = float(self.cumulative[-1])
        self.progress = 0.0
        self._revision = 0

    @property
    def revision(self) -> int:
        """Monotonic geometry revision for route-relative observers."""

        return int(self._revision)

    def reset(self) -> None:
        self.progress = 0.0

    def replace_points(self, points: Sequence[Sequence[float]]) -> None:
        """Replace the soft route after a standard static-only replan.

        The goal and all tracking semantics stay unchanged. Progress restarts
        at zero because the refreshed route begins at the current robot pose.
        """

        replacement = PolylineReference(
            points,
            tolerance=self.tolerance,
            lookahead_distance=self.lookahead_distance,
            terminal_approach_distance=self.terminal_approach_distance,
            projection_backtrack_distance=self.projection_backtrack_distance,
            projection_forward_distance=self.projection_forward_distance,
            corridor_half_width=self.corridor_half_width,
            footprint_radius=self.footprint_radius,
            corridor_half_width_profile=self.corridor_half_width_profile,
        )
        self.points = replacement.points
        self.segment_lengths = replacement.segment_lengths
        self.cumulative = replacement.cumulative
        self.total_length = replacement.total_length
        self.progress = 0.0
        self._revision += 1

    def project(
        self,
        position: np.ndarray,
        minimum_progress: Optional[float] = None,
    ) -> PolylineProjection:
        """Project a position without mutating online reference progress.

        ``minimum_progress`` supplies the online progress floor when the path
        crosses itself or a hypothetical rollout falls behind the robot.  The
        method is intentionally side-effect free so batched policy and MPPI
        rollouts cannot influence one another through candidate order.
        """

        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("polyline projection requires one finite x/y pair")
        floor = None if minimum_progress is None else float(minimum_progress)
        if floor is not None and (
            not np.isfinite(floor) or not 0.0 <= floor <= self.total_length
        ):
            raise ValueError("minimum polyline progress is outside the route")
        starts = self.points[:-1]
        vectors = np.diff(self.points, axis=0)
        fractions = np.sum((position[None, :] - starts) * vectors, axis=1)
        fractions /= self.segment_lengths ** 2
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(projections - position[None, :], axis=1)
        candidate_progress = self.cumulative[:-1] + fractions * self.segment_lengths
        # Never jump to an earlier branch of a route that passes near itself.
        admissible = np.ones(candidate_progress.shape, dtype=bool)
        if floor is not None:
            admissible = (
                candidate_progress
                >= floor - self.projection_backtrack_distance
            ) & (
                candidate_progress
                <= floor + self.projection_forward_distance
            )
        if not np.any(admissible):
            progress = floor
        else:
            masked = np.where(admissible, distances, np.inf)
            progress = float(candidate_progress[int(np.argmin(masked))])
            if floor is not None:
                progress = max(floor, progress)
        point, theta, index = self._geometry_at_progress(progress)
        tangent = np.asarray((np.cos(theta), np.sin(theta)), dtype=np.float64)
        displacement = position - point
        signed_error = float(
            tangent[0] * displacement[1] - tangent[1] * displacement[0]
        )
        cross_track = float(np.linalg.norm(displacement))
        curvature = self._curvature_at_segment(index)
        return PolylineProjection(
            point=(float(point[0]), float(point[1])),
            progress=float(progress),
            remaining=float(self.total_length - progress),
            normalized_progress=float(progress / self.total_length),
            segment_index=int(index),
            tangent_heading=float(theta),
            cross_track_error=cross_track,
            signed_cross_track_error=signed_error,
            curvature=float(curvature),
        )

    def project_batch(
        self,
        positions: np.ndarray,
        minimum_progress: Optional[np.ndarray] = None,
    ) -> PolylineProjectionBatch:
        """Project finite ``[B,2]`` positions without Python candidate loops.

        ``minimum_progress`` may be one scalar or one value per position.  The
        projection window and monotonic progress semantics are identical to
        :meth:`project`; this method only vectorizes the independent geometry.
        """

        values = np.asarray(positions, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] <= 0
            or values.shape[1] != 2
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "polyline batch projection requires finite [B,2] positions"
            )
        floors = None
        if minimum_progress is not None:
            supplied = np.asarray(minimum_progress, dtype=np.float64)
            if supplied.ndim == 0:
                floors = np.full(values.shape[0], float(supplied))
            else:
                floors = supplied.reshape(-1)
            if (
                floors.shape != (values.shape[0],)
                or not np.isfinite(floors).all()
                or np.any(floors < 0.0)
                or np.any(floors > self.total_length)
            ):
                raise ValueError(
                    "minimum polyline progress batch is outside the route"
                )

        starts = self.points[:-1]
        vectors = np.diff(self.points, axis=0)
        displacement = values[:, None, :] - starts[None, :, :]
        fractions = np.sum(
            displacement * vectors[None, :, :], axis=-1
        ) / (self.segment_lengths[None, :] ** 2)
        fractions = np.clip(fractions, 0.0, 1.0)
        candidate_points = (
            starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
        )
        distances = np.linalg.norm(
            candidate_points - values[:, None, :], axis=-1
        )
        candidate_progress = (
            self.cumulative[None, :-1]
            + fractions * self.segment_lengths[None, :]
        )
        admissible = np.ones(candidate_progress.shape, dtype=bool)
        if floors is not None:
            admissible = (
                candidate_progress
                >= floors[:, None] - self.projection_backtrack_distance
            ) & (
                candidate_progress
                <= floors[:, None] + self.projection_forward_distance
            )
        has_admissible = np.any(admissible, axis=1)
        selected = np.argmin(
            np.where(admissible, distances, np.inf), axis=1
        )
        rows = np.arange(values.shape[0])
        progress = candidate_progress[rows, selected]
        if floors is not None:
            progress = np.where(has_admissible, progress, floors)
            progress = np.maximum(floors, progress)

        progress = np.clip(progress, 0.0, self.total_length)
        indices = np.searchsorted(
            self.cumulative, progress, side="right"
        ) - 1
        indices = np.clip(indices, 0, len(self.segment_lengths) - 1)
        segment_fractions = (
            (progress - self.cumulative[indices])
            / self.segment_lengths[indices]
        )
        active_vectors = self.points[indices + 1] - self.points[indices]
        points = (
            self.points[indices]
            + segment_fractions[:, None] * active_vectors
        )
        headings = np.arctan2(active_vectors[:, 1], active_vectors[:, 0])
        point_displacement = values - points
        signed_error = (
            np.cos(headings) * point_displacement[:, 1]
            - np.sin(headings) * point_displacement[:, 0]
        )
        if len(self.segment_lengths) <= 1:
            curvature = np.zeros(values.shape[0], dtype=np.float64)
        else:
            all_headings = np.arctan2(vectors[:, 1], vectors[:, 0])
            left = np.clip(indices, 0, len(all_headings) - 2)
            delta = np.arctan2(
                np.sin(all_headings[left + 1] - all_headings[left]),
                np.cos(all_headings[left + 1] - all_headings[left]),
            )
            scale = 0.5 * (
                self.segment_lengths[left]
                + self.segment_lengths[left + 1]
            )
            curvature = delta / scale
        result = PolylineProjectionBatch(
            point=points,
            progress=progress,
            remaining=self.total_length - progress,
            normalized_progress=progress / self.total_length,
            segment_index=indices.astype(np.int64, copy=False),
            tangent_heading=headings,
            cross_track_error=np.linalg.norm(point_displacement, axis=1),
            signed_cross_track_error=signed_error,
            curvature=curvature,
        )
        if not all(
            np.isfinite(np.asarray(field)).all()
            for field in (
                result.point,
                result.progress,
                result.remaining,
                result.normalized_progress,
                result.tangent_heading,
                result.cross_track_error,
                result.signed_cross_track_error,
                result.curvature,
            )
        ):
            raise FloatingPointError(
                "polyline batch projection produced NaN or Inf"
            )
        return result

    def _project_progress(self, position: np.ndarray) -> float:
        return self.project(position, minimum_progress=self.progress).progress

    def _geometry_at_progress(self, progress: float):
        value = float(np.clip(progress, 0.0, self.total_length))
        index = min(
            int(np.searchsorted(self.cumulative, value, side="right") - 1),
            len(self.segment_lengths) - 1,
        )
        fraction = (value - self.cumulative[index]) / self.segment_lengths[index]
        point = self.points[index] + fraction * (self.points[index + 1] - self.points[index])
        tangent = self.points[index + 1] - self.points[index]
        theta = float(np.arctan2(tangent[1], tangent[0]))
        return point, theta, index

    def _curvature_at_segment(self, index: int) -> float:
        headings = np.arctan2(
            np.diff(self.points, axis=0)[:, 1],
            np.diff(self.points, axis=0)[:, 0],
        )
        if headings.size <= 1:
            return 0.0
        left = int(np.clip(index, 0, headings.size - 2))
        delta = float(np.arctan2(
            np.sin(headings[left + 1] - headings[left]),
            np.cos(headings[left + 1] - headings[left]),
        ))
        scale = 0.5 * (
            self.segment_lengths[left] + self.segment_lengths[left + 1]
        )
        return delta / float(scale)

    def _point_at_progress(self, progress: float):
        point, theta, _ = self._geometry_at_progress(progress)
        return point, theta

    def poses_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return finite ``[x, y, tangent]`` poses at route arc lengths.

        The operation is side-effect free and clips arc lengths to the route.
        It is the shared task representation used by MPPI path-preview costs
        and learned-policy preview features; it never exposes obstacle truth.
        """

        values = np.asarray(progress_values, dtype=np.float64)
        if values.ndim == 0 or not np.isfinite(values).all():
            raise ValueError("polyline progress samples must be a finite array")
        flat = np.clip(values.reshape(-1), 0.0, self.total_length)
        indices = np.searchsorted(self.cumulative, flat, side="right") - 1
        indices = np.clip(indices, 0, len(self.segment_lengths) - 1)
        fractions = (
            (flat - self.cumulative[indices]) / self.segment_lengths[indices]
        )
        vectors = self.points[indices + 1] - self.points[indices]
        points = self.points[indices] + fractions[:, None] * vectors
        headings = np.arctan2(vectors[:, 1], vectors[:, 0])
        result = np.column_stack((points, headings)).reshape(values.shape + (3,))
        if not np.isfinite(result).all():
            raise FloatingPointError("polyline pose preview produced NaN or Inf")
        return result

    def target_poses_at_progress(
        self, progress_values: np.ndarray
    ) -> np.ndarray:
        """Return lookahead target poses for projected route progress values."""

        progress = np.asarray(progress_values, dtype=np.float64)
        if progress.ndim == 0 or not np.isfinite(progress).all():
            raise ValueError(
                "polyline target progress samples must be a finite array"
            )
        if np.any(progress < 0.0) or np.any(progress > self.total_length):
            raise ValueError("polyline target progress is outside the route")
        return self.poses_at_progress(progress + self.lookahead_distance)

    def preview_poses(
        self,
        distances: np.ndarray,
        progress_floor: Optional[float] = None,
    ) -> np.ndarray:
        """Sample route poses at non-negative distances beyond live progress."""

        offsets = np.asarray(distances, dtype=np.float64)
        if offsets.ndim == 0 or not np.isfinite(offsets).all() or np.any(offsets < 0.0):
            raise ValueError("polyline preview distances must be a non-negative array")
        floor = self.progress if progress_floor is None else float(progress_floor)
        if not np.isfinite(floor) or not 0.0 <= floor <= self.total_length:
            raise ValueError("polyline preview progress floor is outside the route")
        return self.poses_at_progress(floor + offsets)

    def preview_corridor_half_widths(
        self,
        distances: np.ndarray,
        progress_floor: Optional[float] = None,
    ) -> np.ndarray:
        """Return corridor half widths aligned with path-preview samples."""

        if self.corridor_half_width is None:
            raise ValueError("polyline does not define a tracking corridor")
        offsets = np.asarray(distances, dtype=np.float64)
        if offsets.ndim == 0 or not np.isfinite(offsets).all() or np.any(offsets < 0.0):
            raise ValueError("corridor preview distances must be a non-negative array")
        floor = self.progress if progress_floor is None else float(progress_floor)
        if not np.isfinite(floor) or not 0.0 <= floor <= self.total_length:
            raise ValueError("corridor preview progress floor is outside the route")
        ratios = np.clip((floor + offsets) / self.total_length, 0.0, 1.0)
        if not self.corridor_half_width_profile:
            return np.full(ratios.shape, self.corridor_half_width, dtype=np.float64)
        profile = np.asarray(self.corridor_half_width_profile, dtype=np.float64)
        return np.interp(ratios, profile[:, 0], profile[:, 1])

    def _target_from_progress(self, progress: float) -> ReferenceTarget:
        target_progress = min(self.total_length, progress + self.lookahead_distance)
        point, theta = self._point_at_progress(target_progress)
        remaining = self.total_length - progress
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

    def preview_target_at(
        self,
        time: float,
        state: np.ndarray,
        progress_floor: Optional[float] = None,
    ) -> ReferenceTarget:
        """Return a hypothetical target without advancing the live route."""

        del time
        state_value = np.asarray(state, dtype=np.float64)
        if state_value.size < 2 or not np.isfinite(state_value[:2]).all():
            raise ValueError("polyline reference state must contain finite x/y")
        floor = self.progress if progress_floor is None else float(progress_floor)
        progress = self.project(
            state_value[:2], minimum_progress=floor
        ).progress
        return self._target_from_progress(progress)

    def target_at(self, time: float, state: np.ndarray) -> ReferenceTarget:
        del time
        state_value = np.asarray(state, dtype=np.float64)
        if state_value.size < 2 or not np.isfinite(state_value[:2]).all():
            raise ValueError("polyline reference state must contain finite x/y")
        self.progress = self._project_progress(state_value[:2])
        return self._target_from_progress(self.progress)


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
            projection_backtrack_distance=config.get(
                "projection_backtrack_distance"
            ),
            projection_forward_distance=config.get(
                "projection_forward_distance"
            ),
            corridor_half_width=config.get("corridor_half_width"),
            footprint_radius=config.get("footprint_radius"),
            corridor_half_width_profile=config.get(
                "corridor_half_width_profile", ()
            ),
        )
    if kind == "time_trajectory":
        return TimeTrajectoryReference(config["times"], config["poses"], tolerance)
    raise ValueError("unknown reference type: %s" % kind)
