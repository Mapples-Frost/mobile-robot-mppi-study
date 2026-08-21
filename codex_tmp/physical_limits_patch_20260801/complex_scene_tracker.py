"""Real-lidar bootstrap adapter for the complex-scene multi-target tracker.

The formal MuJoCo complex-scene stack has an exact static map, so it can
remove wall returns before multi-target association.  The real laboratory
does not yet have a calibrated static map.  Feeding all clusters directly to
the formal three-slot tracker would therefore fill the slots with walls.

This adapter keeps the audited multi-target IMM, assignment, motion
confirmation, dropout handling, and retirement logic.  It changes only which
clusters may initialize an empty slot: a compact cluster must exhibit
persistent motion in the odometry frame.  Once a slot is initialized, normal
causal association continues, including while a person pauses briefly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mobile_robot_mppi.obstacles.multi_online_tracking import (
    MultiObstacleChangeAwareTracker,
)


def _tracker_observation(observation):
    """Select the isolated tracker's deskewed scan when available.

    The original scan on ``observation`` remains authoritative for scan_guard
    and the local geometric obstacle layer.  Only this dynamic-tracker copy is
    replaced.
    """

    compensated = observation.auxiliary.get(
        "motion_compensated_tracker_scan"
    )
    if compensated is None:
        return observation
    return replace(
        observation,
        timestamp=float(compensated.timestamp),
        scan=compensated,
    )


class MotionBootstrapMultiObstacleTracker(
    MultiObstacleChangeAwareTracker
):
    """Three-slot tracker whose empty slots initialize from observed motion."""

    def __init__(
        self,
        trackers,
        *,
        required_motion_intervals=4,
        minimum_speed_mps=0.08,
        maximum_speed_mps=1.80,
        maximum_support_beams=35,
        vehicle_minimum_support_beams=12,
        vehicle_maximum_support_beams=80,
        vehicle_candidate_gate_m=0.65,
        vehicle_maximum_speed_mps=2.50,
        maximum_range_m=4.00,
        candidate_gate_m=0.35,
        minimum_total_displacement_m=0.10,
        minimum_direction_cosine=0.50,
        provisional_motion_intervals=1,
        provisional_minimum_displacement_m=0.025,
        provisional_minimum_closing_speed_mps=0.10,
        provisional_horizon_s=3.00,
        provisional_closest_approach_m=0.75,
    ):
        self.required_motion_intervals = int(
            required_motion_intervals
        )
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_support_beams = int(maximum_support_beams)
        self.vehicle_minimum_support_beams = int(
            vehicle_minimum_support_beams
        )
        self.vehicle_maximum_support_beams = int(
            vehicle_maximum_support_beams
        )
        self.vehicle_candidate_gate_m = float(vehicle_candidate_gate_m)
        self.vehicle_maximum_speed_mps = float(
            vehicle_maximum_speed_mps
        )
        self.maximum_range_m = float(maximum_range_m)
        self.candidate_gate_m = float(candidate_gate_m)
        self.minimum_total_displacement_m = float(
            minimum_total_displacement_m
        )
        self.minimum_direction_cosine = float(
            minimum_direction_cosine
        )
        self.provisional_motion_intervals = int(
            provisional_motion_intervals
        )
        self.provisional_minimum_displacement_m = float(
            provisional_minimum_displacement_m
        )
        self.provisional_minimum_closing_speed_mps = float(
            provisional_minimum_closing_speed_mps
        )
        self.provisional_horizon_s = float(provisional_horizon_s)
        self.provisional_closest_approach_m = float(
            provisional_closest_approach_m
        )
        if self.required_motion_intervals < 2:
            raise ValueError(
                "complex-scene bootstrap requires at least 2 intervals"
            )
        if not (
            0.0 < self.minimum_speed_mps < self.maximum_speed_mps
        ):
            raise ValueError("invalid complex-scene bootstrap speed interval")
        if self.maximum_support_beams <= 0:
            raise ValueError("maximum cluster support must be positive")
        if not (
            0 < self.vehicle_minimum_support_beams
            <= self.vehicle_maximum_support_beams
        ):
            raise ValueError("invalid vehicle support interval")
        if (
            self.vehicle_candidate_gate_m < self.candidate_gate_m
            or self.vehicle_maximum_speed_mps < self.maximum_speed_mps
        ):
            raise ValueError("vehicle motion gates must include compact gates")
        if self.maximum_range_m <= 0.0 or self.candidate_gate_m <= 0.0:
            raise ValueError("complex-scene bootstrap gates must be positive")
        if self.minimum_total_displacement_m <= 0.0:
            raise ValueError(
                "minimum total displacement must be positive"
            )
        if not -1.0 <= self.minimum_direction_cosine <= 1.0:
            raise ValueError("invalid direction cosine threshold")
        if self.provisional_motion_intervals < 1:
            raise ValueError("provisional motion intervals must be positive")
        if self.provisional_minimum_displacement_m <= 0.0:
            raise ValueError("provisional displacement must be positive")
        if self.provisional_minimum_closing_speed_mps <= 0.0:
            raise ValueError("provisional closing speed must be positive")
        if (
            self.provisional_horizon_s <= 0.0
            or self.provisional_closest_approach_m <= 0.0
        ):
            raise ValueError("provisional collision gates must be positive")
        super().__init__(trackers)

    @classmethod
    def from_existing(cls, tracker):
        if not isinstance(tracker, MultiObstacleChangeAwareTracker):
            raise TypeError(
                "complex-scene bootstrap requires the multi-target tracker"
            )
        return cls(tracker.trackers)

    def reset(self):
        super().reset()
        self._bootstrap_previous_clusters = ()
        self._bootstrap_previous_timestamp = None
        self._bootstrap_candidates = ()
        self._bootstrap_raw_cluster_count = 0
        self._bootstrap_retained_cluster_count = 0
        self._bootstrap_confirmed_cluster_count = 0
        self._bootstrap_maximum_streak = 0
        self._bootstrap_raw_support_range = (0, 0)
        self._bootstrap_raw_distance_range_m = (None, None)
        self._bootstrap_rejection_counts = {}
        self._bootstrap_provisional_threats = ()

    def _tracked_cluster_indices(self, clusters, timestamp):
        pairs = []
        for track_index, tracker in enumerate(self.trackers):
            reference = tracker._association_reference(timestamp)
            if reference is None:
                continue
            for cluster_index, cluster in enumerate(clusters):
                distance = float(
                    np.linalg.norm(cluster.center - reference)
                )
                if distance <= self.config.association_gate_m:
                    pairs.append((distance, track_index, cluster_index))
        selected_tracks = set()
        selected_clusters = set()
        for _, track_index, cluster_index in sorted(pairs):
            if (
                track_index in selected_tracks
                or cluster_index in selected_clusters
            ):
                continue
            selected_tracks.add(track_index)
            selected_clusters.add(cluster_index)
        return selected_clusters

    def _motion_candidates(
        self, clusters, timestamp, excluded_indices=()
    ):
        rejection_counts = {
            "excluded_tracked": 0,
            "support_too_large": 0,
            "range_too_large": 0,
            "candidate_gate": 0,
            "speed_gate": 0,
            "direction_gate": 0,
            "support_ratio_gate": 0,
            "accepted": 0,
        }
        self._bootstrap_rejection_counts = rejection_counts
        excluded = set(excluded_indices)
        if (
            self._bootstrap_previous_timestamp is None
            or not self._bootstrap_previous_clusters
        ):
            return ()
        dt = float(timestamp) - float(
            self._bootstrap_previous_timestamp
        )
        if dt <= 1.0e-6:
            return ()
        previous_centers = np.asarray(
            [
                cluster.center
                for cluster in self._bootstrap_previous_clusters
            ],
            dtype=np.float64,
        )
        previous_candidates = tuple(self._bootstrap_candidates)
        result = []
        for cluster_index, cluster in enumerate(clusters):
            if cluster_index in excluded:
                rejection_counts["excluded_tracked"] += 1
                continue
            vehicle_shape = bool(
                cluster.support_beams
                >= self.vehicle_minimum_support_beams
            )
            if cluster.support_beams > self.vehicle_maximum_support_beams:
                rejection_counts["support_too_large"] += 1
                continue
            if cluster.minimum_range_m > self.maximum_range_m:
                rejection_counts["range_too_large"] += 1
                continue
            streak = 1
            origin = None
            previous_velocity = None
            if previous_candidates:
                candidate_distances = np.asarray(
                    [
                        np.linalg.norm(
                            cluster.center - candidate["center"]
                        )
                        for candidate in previous_candidates
                    ],
                    dtype=np.float64,
                )
                candidate_index = int(np.argmin(candidate_distances))
                if (
                    float(candidate_distances[candidate_index])
                    <= (
                        self.vehicle_candidate_gate_m
                        if vehicle_shape
                        else self.candidate_gate_m
                    )
                ):
                    previous = previous_candidates[candidate_index]
                    displacement_vector = (
                        cluster.center - previous["center"]
                    )
                    displacement = float(
                        np.linalg.norm(displacement_vector)
                    )
                    speed = displacement / dt
                    if not (
                        self.minimum_speed_mps
                        <= speed
                        <= (
                            self.vehicle_maximum_speed_mps
                            if vehicle_shape
                            else self.maximum_speed_mps
                        )
                    ):
                        rejection_counts["speed_gate"] += 1
                        continue
                    previous_velocity = np.asarray(
                        previous["velocity_mps"], dtype=np.float64
                    )
                    velocity = displacement_vector / dt
                    denominator = float(
                        np.linalg.norm(previous_velocity)
                        * np.linalg.norm(velocity)
                    )
                    direction_cosine = (
                        -1.0
                        if denominator <= 1.0e-12
                        else float(
                            np.dot(previous_velocity, velocity)
                            / denominator
                        )
                    )
                    if (
                        direction_cosine
                        < self.minimum_direction_cosine
                    ):
                        rejection_counts["direction_gate"] += 1
                        continue
                    previous_support = max(
                        int(previous["support_beams"]), 1
                    )
                    support_ratio = (
                        float(cluster.support_beams)
                        / float(previous_support)
                    )
                    support_ratio_low = 0.35 if vehicle_shape else 0.50
                    support_ratio_high = 2.85 if vehicle_shape else 2.00
                    if not support_ratio_low <= support_ratio <= support_ratio_high:
                        rejection_counts["support_ratio_gate"] += 1
                        continue
                    streak = int(previous["streak"]) + 1
                    origin = np.asarray(
                        previous["origin"], dtype=np.float64
                    )
                else:
                    previous = None
            else:
                previous = None

            if previous is None:
                distances = np.linalg.norm(
                    previous_centers - cluster.center[None, :], axis=1
                )
                previous_index = int(np.argmin(distances))
                displacement_vector = (
                    cluster.center
                    - previous_centers[previous_index]
                )
                displacement = float(
                    np.linalg.norm(displacement_vector)
                )
                candidate_gate = (
                    self.vehicle_candidate_gate_m
                    if vehicle_shape
                    else self.candidate_gate_m
                )
                if displacement > candidate_gate:
                    rejection_counts["candidate_gate"] += 1
                    continue
                speed = displacement / dt
                if not (
                    self.minimum_speed_mps
                    <= speed
                    <= (
                        self.vehicle_maximum_speed_mps
                        if vehicle_shape
                        else self.maximum_speed_mps
                    )
                ):
                    rejection_counts["speed_gate"] += 1
                    continue
                velocity = displacement_vector / dt
                origin = previous_centers[previous_index].copy()

            total_displacement = float(
                np.linalg.norm(cluster.center - origin)
            )
            result.append(
                {
                    "cluster_index": int(cluster_index),
                    "center": cluster.center.copy(),
                    "streak": int(streak),
                    "speed_mps": float(speed),
                    "velocity_mps": np.asarray(
                        velocity, dtype=np.float64
                    ).copy(),
                    "origin": np.asarray(
                        origin, dtype=np.float64
                    ).copy(),
                    "total_displacement_m": total_displacement,
                    "support_beams": int(cluster.support_beams),
                    "vehicle_shape": vehicle_shape,
                }
            )
            rejection_counts["accepted"] += 1
        return tuple(result)

    def _provisional_threat(self, candidate, observation):
        """Return a causal closest-approach diagnostic for one candidate.

        Candidate velocity and cluster centers are already in the odometry
        frame.  Subtracting the robot's world velocity makes the gate useful
        while the base is moving without reintroducing scan-frame ego motion.
        """

        if (
            int(candidate["streak"]) < self.provisional_motion_intervals
            or float(candidate["total_displacement_m"])
            < self.provisional_minimum_displacement_m
        ):
            return None
        center = np.asarray(candidate["center"], dtype=np.float64)
        robot_position = np.asarray(
            (observation.pose.x, observation.pose.y), dtype=np.float64
        )
        relative_position = center - robot_position
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1.0e-9:
            return None
        heading = float(observation.pose.theta)
        robot_velocity = float(observation.twist.v) * np.asarray(
            (np.cos(heading), np.sin(heading)), dtype=np.float64
        )
        relative_velocity = (
            np.asarray(candidate["velocity_mps"], dtype=np.float64)
            - robot_velocity
        )
        closing_speed = -float(
            np.dot(relative_position, relative_velocity) / distance
        )
        relative_speed_sq = float(
            np.dot(relative_velocity, relative_velocity)
        )
        if relative_speed_sq <= 1.0e-9:
            return None
        closest_time = -float(
            np.dot(relative_position, relative_velocity)
        ) / relative_speed_sq
        closest_time = min(
            max(closest_time, 0.0), self.provisional_horizon_s
        )
        closest_vector = (
            relative_position + closest_time * relative_velocity
        )
        closest_distance = float(np.linalg.norm(closest_vector))
        if (
            closing_speed < self.provisional_minimum_closing_speed_mps
            or closest_time <= 0.0
            or closest_time > self.provisional_horizon_s
            or closest_distance > self.provisional_closest_approach_m
        ):
            return None
        left_axis = np.asarray(
            (-np.sin(heading), np.cos(heading)), dtype=np.float64
        )
        return {
            "cluster_index": int(candidate["cluster_index"]),
            "distance_m": distance,
            "closing_speed_mps": closing_speed,
            "closest_approach_time_s": closest_time,
            "closest_approach_distance_m": closest_distance,
            "lateral_m": float(np.dot(relative_position, left_axis)),
            "streak": int(candidate["streak"]),
            "speed_mps": float(candidate["speed_mps"]),
            "total_displacement_m": float(
                candidate["total_displacement_m"]
            ),
            "support_beams": int(candidate["support_beams"]),
            "dynamic_shape": (
                "wide_vehicle"
                if bool(candidate.get("vehicle_shape", False))
                else "compact"
            ),
        }

    def scan_clusters(self, observation):
        observation = _tracker_observation(observation)
        raw_clusters = tuple(
            self.trackers[0].scan_clusters(observation)
        )
        supports = [int(item.support_beams) for item in raw_clusters]
        distances = [float(item.minimum_range_m) for item in raw_clusters]
        self._bootstrap_raw_support_range = (
            (min(supports), max(supports)) if supports else (0, 0)
        )
        self._bootstrap_raw_distance_range_m = (
            (min(distances), max(distances))
            if distances
            else (None, None)
        )
        timestamp = float(observation.timestamp)
        tracked_indices = self._tracked_cluster_indices(
            raw_clusters, timestamp
        )
        motion_candidates = self._motion_candidates(
            raw_clusters,
            timestamp,
            excluded_indices=tracked_indices,
        )
        provisional_threats = tuple(
            threat
            for threat in (
                self._provisional_threat(candidate, observation)
                for candidate in motion_candidates
            )
            if threat is not None
        )
        provisional_indices = {
            int(threat["cluster_index"])
            for threat in provisional_threats
        }
        confirmed_indices = {
            int(candidate["cluster_index"])
            for candidate in motion_candidates
            if (
                (
                    bool(candidate.get("vehicle_shape", False))
                    and (
                        (
                            int(candidate["streak"]) >= 2
                            and float(candidate["total_displacement_m"])
                            >= 0.18
                        )
                        or (
                            int(candidate["streak"]) >= 3
                            and float(candidate["total_displacement_m"])
                            >= 0.14
                        )
                    )
                )
                or (
                    int(candidate["streak"])
                    >= self.required_motion_intervals
                    and float(candidate["total_displacement_m"])
                    >= self.minimum_total_displacement_m
                )
            )
        }
        retained_indices = (
            tracked_indices | confirmed_indices | provisional_indices
        )
        retained = tuple(
            cluster
            for index, cluster in enumerate(raw_clusters)
            if index in retained_indices
        )

        self._bootstrap_previous_clusters = raw_clusters
        self._bootstrap_previous_timestamp = timestamp
        self._bootstrap_candidates = motion_candidates
        self._bootstrap_raw_cluster_count = len(raw_clusters)
        self._bootstrap_retained_cluster_count = len(retained)
        self._bootstrap_confirmed_cluster_count = len(
            confirmed_indices
        )
        self._bootstrap_maximum_streak = max(
            (
                int(candidate["streak"])
                for candidate in motion_candidates
            ),
            default=0,
        )
        self._bootstrap_provisional_threats = provisional_threats
        return retained

    def update(self, observation):
        compensated = _tracker_observation(observation)
        result = super().update(compensated)
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "real_lidar_motion_bootstrap_enabled": True,
                "real_lidar_bootstrap_required_motion_intervals": (
                    self.required_motion_intervals
                ),
                "ego_motion_compensated_tracker_scan": bool(
                    compensated is not observation
                ),
                "real_lidar_raw_cluster_count": (
                    self._bootstrap_raw_cluster_count
                ),
                "real_lidar_retained_cluster_count": (
                    self._bootstrap_retained_cluster_count
                ),
                "real_lidar_confirmed_cluster_count": (
                    self._bootstrap_confirmed_cluster_count
                ),
                "real_lidar_bootstrap_maximum_streak": (
                    self._bootstrap_maximum_streak
                ),
                "real_lidar_raw_cluster_support_range": (
                    self._bootstrap_raw_support_range
                ),
                "real_lidar_raw_cluster_distance_range_m": (
                    self._bootstrap_raw_distance_range_m
                ),
                "real_lidar_bootstrap_rejection_counts": dict(
                    self._bootstrap_rejection_counts
                ),
                "provisional_collision_candidate_count": len(
                    self._bootstrap_provisional_threats
                ),
                "provisional_collision_candidates": tuple(
                    dict(item)
                    for item in self._bootstrap_provisional_threats
                ),
                "provisional_collision_minimum_ttc_s": min(
                    (
                        float(item["closest_approach_time_s"])
                        for item in self._bootstrap_provisional_threats
                    ),
                    default=None,
                ),
            }
        )
        return replace(result, diagnostics=diagnostics)


__all__ = ["MotionBootstrapMultiObstacleTracker"]
