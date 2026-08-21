"""Calibrated-background rejection for the real-robot dynamic tracker.

This module intentionally filters only the observation passed to the dynamic
tracker.  The unmodified LaserScan still reaches the local obstacle layer and
the geometric guard, so mapped walls remain physical collision obstacles.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution

from complex_scene_tracker import MotionBootstrapMultiObstacleTracker


MAP_CONTRACT = "real_robot_static_background_map_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation(theta):
    cosine = math.cos(float(theta))
    sine = math.sin(float(theta))
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


class CalibratedStaticBackgroundFilter:
    """Reject scan endpoints that coincide with a measured static surface."""

    def __init__(
        self,
        map_path,
        *,
        endpoint_gate_m=0.10,
        cluster_center_gate_m=0.12,
        minimum_alignment_fraction=0.30,
        alignment_gate_m=0.15,
        minimum_voxel_hits=3,
    ):
        self.map_path = Path(map_path).resolve()
        if not self.map_path.is_file():
            raise FileNotFoundError(str(self.map_path))
        self.metadata_path = self.map_path.with_name(
            "static_background_map.metadata.json"
        )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract") != MAP_CONTRACT:
            raise ValueError("unsupported static background map contract")
        payload = np.load(self.map_path)
        points = np.asarray(payload["points_xy"], dtype=np.float64)
        hit_counts = np.asarray(payload["hit_counts"], dtype=np.int64)
        retained = hit_counts >= int(minimum_voxel_hits)
        points = points[retained]
        if points.shape[0] < 100:
            raise ValueError("static background map has too few retained voxels")
        self._calibration_points = points
        self.metadata = metadata
        self.map_sha256 = _sha256(self.map_path)
        self.metadata_sha256 = _sha256(self.metadata_path)
        self.endpoint_gate_m = float(endpoint_gate_m)
        self.cluster_center_gate_m = float(cluster_center_gate_m)
        self.minimum_alignment_fraction = float(minimum_alignment_fraction)
        self.alignment_gate_m = float(alignment_gate_m)
        self.minimum_voxel_hits = int(minimum_voxel_hits)
        self.sensor_forward_offset_m = float(
            metadata["sensor_forward_offset_m"]
        )
        if min(
            self.endpoint_gate_m,
            self.cluster_center_gate_m,
            self.alignment_gate_m,
        ) <= 0.0:
            raise ValueError("static background gates must be positive")
        self.run_origin_odom = None
        self.run_pose_in_calibration = None
        self.odometry_pose_in_calibration = None
        self.relocalization = None
        self.points_run_xy = None
        self.tree = None
        self.alignment = None
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.last_input_valid_beams = 0
        self.last_masked_beams = 0
        self.last_retained_beams = 0
        self.last_endpoint_match_fraction = 0.0

    def set_run_origin(self, run_origin_odom):
        run_origin = np.asarray(run_origin_odom, dtype=np.float64).reshape(3)
        calibration_origin = np.asarray(
            self.metadata["origin_odom"], dtype=np.float64
        ).reshape(3)
        global_points = (
            self._calibration_points @ _rotation(calibration_origin[2]).T
            + calibration_origin[:2]
        )
        self.points_run_xy = (
            global_points - run_origin[:2]
        ) @ _rotation(run_origin[2])
        self.tree = cKDTree(self.points_run_xy)
        self.run_origin_odom = tuple(float(value) for value in run_origin)

    def _odometry_pose_in_calibration_frame(self, run_origin_odom):
        run_origin = np.asarray(run_origin_odom, dtype=np.float64).reshape(3)
        calibration_origin = np.asarray(
            self.metadata["origin_odom"], dtype=np.float64
        ).reshape(3)
        translation = (
            run_origin[:2] - calibration_origin[:2]
        ) @ _rotation(calibration_origin[2])
        heading = (
            run_origin[2] - calibration_origin[2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        return np.asarray(
            (translation[0], translation[1], heading), dtype=np.float64
        )

    def relocalize(self, observation, run_origin_odom):
        """Align the current stationary scan to the calibration map.

        Wheel odometry supplies only a translation prior.  Heading is searched
        globally because real tests showed a 1.01 rad odometry discontinuity
        while the robot remained in the calibrated laboratory region.
        """

        indices, endpoints = self._scan_endpoints(observation)
        if indices.size < 80:
            raise RuntimeError("too few live lidar returns for relocalization")
        odometry_pose = self._odometry_pose_in_calibration_frame(
            run_origin_odom
        )
        calibration_tree = cKDTree(self._calibration_points)

        def objective(candidate):
            pose = np.asarray(candidate, dtype=np.float64)
            transformed = (
                endpoints @ _rotation(pose[2]).T + pose[:2]
            )
            distances, _ = calibration_tree.query(transformed, k=1)
            clipped = np.minimum(distances, 0.75)
            retained = np.sort(clipped)[: max(int(0.70 * clipped.size), 1)]
            translation_regularizer = 0.005 * float(
                np.linalg.norm(pose[:2] - odometry_pose[:2])
            )
            return float(np.mean(retained)) + translation_regularizer

        bounds = (
            (odometry_pose[0] - 1.0, odometry_pose[0] + 1.0),
            (odometry_pose[1] - 1.0, odometry_pose[1] + 1.0),
            (-math.pi, math.pi),
        )
        solution = differential_evolution(
            objective,
            bounds,
            seed=20260731,
            popsize=10,
            maxiter=60,
            tol=1.0e-6,
            polish=True,
            workers=1,
        )
        pose = np.asarray(solution.x, dtype=np.float64)
        transformed = endpoints @ _rotation(pose[2]).T + pose[:2]
        distances, _ = calibration_tree.query(transformed, k=1)
        matched_fraction = float(np.mean(distances <= self.alignment_gate_m))
        median_distance = float(np.median(distances))
        accepted = (
            matched_fraction >= 0.50 and median_distance <= 0.12
        )
        heading_correction = (
            pose[2] - odometry_pose[2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        self.relocalization = {
            "accepted": bool(accepted),
            "objective": float(solution.fun),
            "valid_beams": int(indices.size),
            "matched_fraction": matched_fraction,
            "median_distance_m": median_distance,
            "pose_in_calibration": pose.tolist(),
            "odometry_pose_in_calibration": odometry_pose.tolist(),
            "translation_correction_m": float(
                np.linalg.norm(pose[:2] - odometry_pose[:2])
            ),
            "heading_correction_rad": float(heading_correction),
        }
        if not accepted:
            raise RuntimeError(
                "static background relocalization Gate failed: "
                "matched=%.3f median=%.3f m"
                % (matched_fraction, median_distance)
            )
        self.points_run_xy = (
            self._calibration_points - pose[:2]
        ) @ _rotation(pose[2])
        self.tree = cKDTree(self.points_run_xy)
        self.run_origin_odom = tuple(
            float(value) for value in np.asarray(run_origin_odom).reshape(3)
        )
        self.run_pose_in_calibration = tuple(float(value) for value in pose)
        self.odometry_pose_in_calibration = tuple(
            float(value) for value in odometry_pose
        )
        return dict(self.relocalization)

    def _scan_endpoints(self, observation):
        scan = observation.scan
        if scan is None:
            return np.empty((0,), dtype=np.int64), np.empty((0, 2))
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(scan.range_min))
            & (ranges < float(scan.range_max) - 1.0e-12)
        )
        indices = np.flatnonzero(valid)
        if not indices.size:
            return indices, np.empty((0, 2), dtype=np.float64)
        theta = float(observation.pose.theta)
        sensor_offset = self.sensor_forward_offset_m
        sensor = np.asarray(
            (
                observation.pose.x + sensor_offset * math.cos(theta),
                observation.pose.y + sensor_offset * math.sin(theta),
            ),
            dtype=np.float64,
        )
        angles = (
            theta
            + float(scan.angle_min)
            + indices.astype(np.float64) * float(scan.angle_increment)
        )
        endpoints = sensor[None, :] + ranges[indices, None] * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        return indices, endpoints

    def validate_alignment(self, observation):
        if self.tree is None:
            raise RuntimeError("run origin must be set before alignment")
        indices, endpoints = self._scan_endpoints(observation)
        if indices.size < 30:
            raise RuntimeError("too few live lidar returns for map alignment")
        distances, _ = self.tree.query(endpoints, k=1)
        matched_fraction = float(np.mean(distances <= self.alignment_gate_m))
        median_distance = float(np.median(distances))
        accepted = matched_fraction >= self.minimum_alignment_fraction
        self.alignment = {
            "accepted": bool(accepted),
            "valid_beams": int(indices.size),
            "matched_fraction": matched_fraction,
            "median_distance_m": median_distance,
            "gate_m": self.alignment_gate_m,
            "minimum_fraction": self.minimum_alignment_fraction,
        }
        if not accepted:
            raise RuntimeError(
                "static background alignment Gate failed: %.3f < %.3f "
                "(median nearest-map distance %.3f m)"
                % (
                    matched_fraction,
                    self.minimum_alignment_fraction,
                    median_distance,
                )
            )
        return dict(self.alignment)

    def filter_observation(self, observation):
        if self.tree is None or self.alignment is None:
            raise RuntimeError("static background filter is not aligned")
        indices, endpoints = self._scan_endpoints(observation)
        ranges = np.asarray(observation.scan.ranges, dtype=np.float64).copy()
        mapped = np.zeros((indices.size,), dtype=bool)
        if indices.size:
            distances, _ = self.tree.query(endpoints, k=1)
            mapped = distances <= self.endpoint_gate_m
            ranges[indices[mapped]] = float(observation.scan.range_max)
            self.last_input_valid_beams = int(indices.size)
            self.last_masked_beams = int(np.sum(mapped))
            self.last_retained_beams = int(indices.size - np.sum(mapped))
            self.last_endpoint_match_fraction = float(np.mean(mapped))
        else:
            self.reset_diagnostics()
        obstacle_ranges = observation.scan.obstacle_ranges
        if obstacle_ranges is not None:
            obstacle_ranges = np.asarray(obstacle_ranges, dtype=np.float64).copy()
            obstacle_ranges[indices[mapped]] = float(observation.scan.range_max)
        filtered_scan = replace(
            observation.scan,
            ranges=ranges,
            obstacle_ranges=obstacle_ranges,
        )
        return replace(observation, scan=filtered_scan)

    def reject_cluster_centers(self, centers):
        values = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        if not values.size:
            return np.zeros((0,), dtype=bool)
        distances, _ = self.tree.query(values, k=1)
        return distances <= self.cluster_center_gate_m

    def diagnostics(self):
        alignment = self.alignment or {}
        return {
            "static_background_filter_enabled": True,
            "static_background_map_path": str(self.map_path),
            "static_background_map_sha256": self.map_sha256,
            "static_background_metadata_sha256": self.metadata_sha256,
            "static_background_minimum_voxel_hits": self.minimum_voxel_hits,
            "static_background_retained_voxels": int(
                self._calibration_points.shape[0]
            ),
            "static_background_alignment_accepted": bool(
                alignment.get("accepted", False)
            ),
            "static_background_alignment_matched_fraction": float(
                alignment.get("matched_fraction", 0.0)
            ),
            "static_background_alignment_median_distance_m": float(
                alignment.get("median_distance_m", float("inf"))
            ),
            "static_background_input_valid_beams": self.last_input_valid_beams,
            "static_background_masked_beams": self.last_masked_beams,
            "static_background_retained_beams": self.last_retained_beams,
            "static_background_endpoint_match_fraction": (
                self.last_endpoint_match_fraction
            ),
            "static_background_relocalization_accepted": bool(
                (self.relocalization or {}).get("accepted", False)
            ),
            "static_background_relocalization_matched_fraction": float(
                (self.relocalization or {}).get("matched_fraction", 0.0)
            ),
            "static_background_relocalization_median_distance_m": float(
                (self.relocalization or {}).get(
                    "median_distance_m", float("inf")
                )
            ),
            "static_background_relocalization_translation_correction_m": (
                float(
                    (self.relocalization or {}).get(
                        "translation_correction_m", float("inf")
                    )
                )
            ),
            "static_background_relocalization_heading_correction_rad": (
                float(
                    (self.relocalization or {}).get(
                        "heading_correction_rad", float("inf")
                    )
                )
            ),
        }


class BackgroundFilteredMotionBootstrapTracker(
    MotionBootstrapMultiObstacleTracker
):
    """Motion-bootstrap tracker with a tracker-only calibrated map filter."""

    def __init__(self, trackers, background_filter, **bootstrap_options):
        self.background_filter = background_filter
        self._background_rejected_cluster_count = 0
        super().__init__(trackers, **bootstrap_options)

    @classmethod
    def from_existing(cls, tracker, background_filter):
        if not isinstance(tracker, MotionBootstrapMultiObstacleTracker):
            raise TypeError("background filter requires motion-bootstrap tracker")
        return cls(
            tracker.trackers,
            background_filter,
            # Synchronized human traverses repeatedly reached two persistent
            # intervals but changed direction/shape before a third.  Across
            # hundreds of empty calibrated-room frames the maximum streak was
            # zero.  Keep all motion/shape/map gates and shorten only this
            # map-filtered confirmation latency (about 0.7 s at runtime).
            required_motion_intervals=2,
            minimum_speed_mps=tracker.minimum_speed_mps,
            maximum_speed_mps=tracker.maximum_speed_mps,
            maximum_support_beams=tracker.maximum_support_beams,
            maximum_range_m=tracker.maximum_range_m,
            candidate_gate_m=tracker.candidate_gate_m,
            minimum_total_displacement_m=(
                tracker.minimum_total_displacement_m
            ),
            minimum_direction_cosine=tracker.minimum_direction_cosine,
        )

    def reset(self):
        super().reset()
        self._background_rejected_cluster_count = 0
        if hasattr(self, "background_filter"):
            self.background_filter.reset_diagnostics()

    def scan_clusters(self, observation):
        filtered = self.background_filter.filter_observation(observation)
        clusters = tuple(super().scan_clusters(filtered))
        if not clusters:
            self._background_rejected_cluster_count = 0
            return clusters
        centers = np.asarray(
            [cluster.center for cluster in clusters], dtype=np.float64
        )
        rejected = self.background_filter.reject_cluster_centers(centers)
        self._background_rejected_cluster_count = int(np.sum(rejected))
        return tuple(
            cluster for cluster, drop in zip(clusters, rejected) if not drop
        )

    def update(self, observation):
        result = super().update(observation)
        diagnostics = dict(result.diagnostics)
        diagnostics.update(self.background_filter.diagnostics())
        diagnostics["static_background_rejected_cluster_count"] = (
            self._background_rejected_cluster_count
        )
        diagnostics["static_background_tracker_only"] = True
        return replace(result, diagnostics=diagnostics)


__all__ = [
    "BackgroundFilteredMotionBootstrapTracker",
    "CalibratedStaticBackgroundFilter",
]
