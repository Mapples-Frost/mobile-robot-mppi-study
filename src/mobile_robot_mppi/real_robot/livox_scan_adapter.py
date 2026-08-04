"""Convert timestamped Livox point packets into the planner's 2-D scan.

The adapter owns only geometry and validity filtering.  It deliberately does
not classify points as static or dynamic: the raw scan must remain available
to the geometric safety path, while a separate causal filter supplies the
dynamic-only copy used by IMM/CA-IMM.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mobile_robot_mppi.core.types import LaserScan


@dataclass(frozen=True)
class LivoxPointCloudFrame:
    """One accumulated Mid-360 frame in the native lidar coordinate frame."""

    timestamp_ns: int
    points: np.ndarray
    reflectivity: np.ndarray
    tags: np.ndarray
    point_timestamps_ns: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        reflectivity = np.asarray(self.reflectivity, dtype=np.uint8).reshape(-1)
        tags = np.asarray(self.tags, dtype=np.uint8).reshape(-1)
        point_timestamps_ns = (
            None
            if self.point_timestamps_ns is None
            else np.asarray(self.point_timestamps_ns, dtype=np.int64).reshape(-1)
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Livox points must have shape (N, 3)")
        if points.shape[0] != reflectivity.size or points.shape[0] != tags.size:
            raise ValueError("Livox point, reflectivity and tag counts must match")
        if int(self.timestamp_ns) < 0:
            raise ValueError("Livox timestamp_ns must be non-negative")
        if (
            point_timestamps_ns is not None
            and point_timestamps_ns.size != points.shape[0]
        ):
            raise ValueError(
                "Livox point timestamp count must match point count"
            )
        if point_timestamps_ns is not None:
            if np.any(point_timestamps_ns < 0):
                raise ValueError("Livox point timestamps must be non-negative")
            if np.any(point_timestamps_ns > int(self.timestamp_ns)):
                raise ValueError(
                    "Livox point timestamps cannot exceed frame timestamp"
                )
        points = np.ascontiguousarray(points)
        reflectivity = np.ascontiguousarray(reflectivity)
        tags = np.ascontiguousarray(tags)
        points.setflags(write=False)
        reflectivity.setflags(write=False)
        tags.setflags(write=False)
        if point_timestamps_ns is not None:
            point_timestamps_ns = np.ascontiguousarray(point_timestamps_ns)
            point_timestamps_ns.setflags(write=False)
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "reflectivity", reflectivity)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "point_timestamps_ns", point_timestamps_ns)


@dataclass(frozen=True)
class LivoxScanAdapterConfig:
    """Calibratable geometry for a Mid-360 mounted on SCOUT MINI.

    ``lidar_to_base`` is a row-major homogeneous transform from Livox XYZ to
    the ROS-style base frame (+x forward, +y left, +z up).  Deployment must
    replace the identity transform with the measured sensor extrinsic before
    wheels-down testing.
    """

    lidar_to_base: Tuple[float, ...] = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    angle_min_rad: float = -np.pi
    angle_max_rad: float = np.pi
    beam_count: int = 720
    range_min_m: float = 0.15
    range_max_m: float = 12.0
    minimum_z_m: float = -0.15
    maximum_z_m: float = 1.80
    minimum_reflectivity: int = 0
    self_min_x_m: float = -0.45
    self_max_x_m: float = 0.45
    self_min_y_m: float = -0.40
    self_max_y_m: float = 0.40
    self_min_z_m: float = -0.30
    self_max_z_m: float = 0.80
    frame_id: str = "livox_base_scan"

    def __post_init__(self) -> None:
        transform = np.asarray(self.lidar_to_base, dtype=np.float64)
        if transform.size != 16 or not np.isfinite(transform).all():
            raise ValueError("lidar_to_base must contain 16 finite values")
        transform = transform.reshape(4, 4)
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ValueError("lidar_to_base must be a homogeneous transform")
        if int(self.beam_count) < 16:
            raise ValueError("beam_count must be at least 16")
        finite = np.asarray(
            (
                self.angle_min_rad,
                self.angle_max_rad,
                self.range_min_m,
                self.range_max_m,
                self.minimum_z_m,
                self.maximum_z_m,
                self.self_min_x_m,
                self.self_max_x_m,
                self.self_min_y_m,
                self.self_max_y_m,
                self.self_min_z_m,
                self.self_max_z_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(finite).all():
            raise ValueError("Livox scan geometry must be finite")
        if not self.angle_max_rad > self.angle_min_rad:
            raise ValueError("angle_max_rad must exceed angle_min_rad")
        if not 0.0 < self.range_min_m < self.range_max_m:
            raise ValueError("invalid Livox scan range interval")
        if not self.maximum_z_m > self.minimum_z_m:
            raise ValueError("invalid Livox height interval")
        if not 0 <= int(self.minimum_reflectivity) <= 255:
            raise ValueError("minimum_reflectivity must be a uint8 value")


@dataclass(frozen=True)
class LivoxScanDiagnostics:
    input_points: int
    finite_nonzero_points: int
    height_points: int
    nonself_points: int
    range_points: int
    populated_beams: int


class LivoxScanAdapter:
    """Vectorized, dependency-light point cloud to LaserScan adapter."""

    def __init__(self, config: Optional[LivoxScanAdapterConfig] = None):
        self.config = config or LivoxScanAdapterConfig()
        self._transform = np.asarray(
            self.config.lidar_to_base, dtype=np.float64
        ).reshape(4, 4)
        self._rotation = self._transform[:3, :3].astype(np.float32)
        self._translation = self._transform[:3, 3].astype(np.float32)
        self._angle_span = float(
            self.config.angle_max_rad - self.config.angle_min_rad
        )
        # LaserScan uses the angle of the first beam and a constant increment.
        # A full 360-degree scan must not duplicate the -pi/+pi endpoint.
        self._angle_increment = self._angle_span / int(self.config.beam_count)

    def convert(self, frame: LivoxPointCloudFrame):
        """Return ``(LaserScan, diagnostics)`` without modifying raw input."""

        points = np.asarray(frame.points, dtype=np.float32)
        input_count = int(points.shape[0])
        finite_nonzero = np.isfinite(points).all(axis=1) & np.any(
            points != 0.0, axis=1
        )
        finite_nonzero &= (
            np.asarray(frame.reflectivity, dtype=np.uint8)
            >= int(self.config.minimum_reflectivity)
        )
        base = points[finite_nonzero] @ self._rotation.T + self._translation
        finite_count = int(base.shape[0])

        height_mask = (
            (base[:, 2] >= float(self.config.minimum_z_m))
            & (base[:, 2] <= float(self.config.maximum_z_m))
        )
        base = base[height_mask]
        height_count = int(base.shape[0])

        in_self = (
            (base[:, 0] >= float(self.config.self_min_x_m))
            & (base[:, 0] <= float(self.config.self_max_x_m))
            & (base[:, 1] >= float(self.config.self_min_y_m))
            & (base[:, 1] <= float(self.config.self_max_y_m))
            & (base[:, 2] >= float(self.config.self_min_z_m))
            & (base[:, 2] <= float(self.config.self_max_z_m))
        )
        base = base[~in_self]
        nonself_count = int(base.shape[0])

        planar_ranges = np.hypot(base[:, 0], base[:, 1])
        angles = np.arctan2(base[:, 1], base[:, 0])
        valid = (
            (planar_ranges >= float(self.config.range_min_m))
            & (planar_ranges <= float(self.config.range_max_m))
            & (angles >= float(self.config.angle_min_rad))
            & (angles < float(self.config.angle_max_rad))
        )
        planar_ranges = planar_ranges[valid].astype(np.float64, copy=False)
        angles = angles[valid]
        range_count = int(planar_ranges.size)

        beam_count = int(self.config.beam_count)
        ranges = np.full(
            beam_count, float(self.config.range_max_m), dtype=np.float64
        )
        obstacle_ranges = np.full(beam_count, np.inf, dtype=np.float64)
        if planar_ranges.size:
            indices = np.floor(
                (angles - float(self.config.angle_min_rad))
                / self._angle_increment
            ).astype(np.int64)
            indices = np.clip(indices, 0, beam_count - 1)
            np.minimum.at(obstacle_ranges, indices, planar_ranges)
            occupied = np.isfinite(obstacle_ranges)
            ranges[occupied] = obstacle_ranges[occupied]
        populated = int(np.isfinite(obstacle_ranges).sum())

        scan = LaserScan(
            ranges=ranges,
            obstacle_ranges=obstacle_ranges,
            angle_min=float(self.config.angle_min_rad),
            angle_increment=float(self._angle_increment),
            range_min=float(self.config.range_min_m),
            range_max=float(self.config.range_max_m),
            timestamp=1.0e-9 * int(frame.timestamp_ns),
            frame_id=str(self.config.frame_id),
        )
        diagnostics = LivoxScanDiagnostics(
            input_points=input_count,
            finite_nonzero_points=finite_count,
            height_points=height_count,
            nonself_points=nonself_count,
            range_points=range_count,
            populated_beams=populated,
        )
        return scan, diagnostics


__all__ = [
    "LivoxPointCloudFrame",
    "LivoxScanAdapter",
    "LivoxScanAdapterConfig",
    "LivoxScanDiagnostics",
]
