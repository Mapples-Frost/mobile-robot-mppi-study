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
        vehicle_minimum_support_beams=6,
        vehicle_maximum_support_beams=80,
        vehicle_minimum_extent_m=0.25,
        vehicle_maximum_extent_m=1.10,
        vehicle_candidate_gate_m=0.45,
        vehicle_maximum_speed_mps=1.80,
        maximum_range_m=4.00,
        candidate_gate_m=0.35,
        minimum_total_displacement_m=0.10,
        minimum_direction_cosine=0.50,
        provisional_motion_intervals=1,
        provisional_minimum_displacement_m=0.060,
        provisional_minimum_closing_speed_mps=0.10,
        provisional_horizon_s=3.00,
        provisional_closest_approach_m=1.00,
        temporal_flow_threat_preemption_enabled=False,
        temporal_flow_threat_maximum_ttc_s=2.00,
        temporal_flow_threat_minimum_support_beams=3,
        temporal_flow_threat_angle_tolerance_deg=25.0,
        temporal_flow_threat_range_tolerance_m=0.75,
        temporal_flow_threat_hold_cycles=6,
        temporal_flow_threat_preemption_allow_active_reset=True,
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
        self.vehicle_minimum_extent_m = float(vehicle_minimum_extent_m)
        self.vehicle_maximum_extent_m = float(vehicle_maximum_extent_m)
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
        self.temporal_flow_threat_preemption_enabled = bool(
            temporal_flow_threat_preemption_enabled
        )
        self.temporal_flow_threat_maximum_ttc_s = float(
            temporal_flow_threat_maximum_ttc_s
        )
        self.temporal_flow_threat_minimum_support_beams = int(
            temporal_flow_threat_minimum_support_beams
        )
        self.temporal_flow_threat_angle_tolerance_rad = float(np.deg2rad(
            temporal_flow_threat_angle_tolerance_deg
        ))
        self.temporal_flow_threat_range_tolerance_m = float(
            temporal_flow_threat_range_tolerance_m
        )
        self.temporal_flow_threat_hold_cycles = int(
            temporal_flow_threat_hold_cycles
        )
        # Resetting an initialized IMM to make room for a scan-flow threat can
        # swap identities when a person's legs/body fragment between beams.
        # Keep the historical behavior available for offline ablations, while
        # allowing the physical profile to protect active tracks.
        self.temporal_flow_threat_preemption_allow_active_reset = bool(
            temporal_flow_threat_preemption_allow_active_reset
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
        if not (
            0.0 < self.vehicle_minimum_extent_m
            < self.vehicle_maximum_extent_m
        ):
            raise ValueError("invalid vehicle extent interval")
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
        if (
            self.temporal_flow_threat_maximum_ttc_s <= 0.0
            or self.temporal_flow_threat_minimum_support_beams <= 0
            or not 0.0 < self.temporal_flow_threat_angle_tolerance_rad <= np.pi
            or self.temporal_flow_threat_range_tolerance_m <= 0.0
            or self.temporal_flow_threat_hold_cycles <= 0
        ):
            raise ValueError("temporal-flow threat gates must be positive")
        super().__init__(trackers)

    @classmethod
    def from_existing(cls, tracker, **kwargs):
        if not isinstance(tracker, MultiObstacleChangeAwareTracker):
            raise TypeError(
                "complex-scene bootstrap requires the multi-target tracker"
            )
        # Explicit real-robot profiles may tune only the empty-slot bootstrap.
        # With no overrides this remains the historical constructor path.
        return cls(tracker.trackers, **kwargs)

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
        self._bootstrap_cluster_extents_m = ()
        self._retained_cluster_extents_m = ()
        self._temporal_flow_threat_retained_indices = set()
        self._temporal_flow_threat_track_indices = set()
        self._temporal_flow_threat_hold = [0] * self.maximum_tracks
        self._temporal_flow_preempted_track_indices = ()
        self._temporal_flow_preempted_raw_cluster_indices = ()
        self._temporal_flow_threat_matches = ()

    @staticmethod
    def _cluster_extent_m(cluster, scan):
        indices = np.asarray(cluster.beam_indices, dtype=np.int64)
        if indices.size < 2:
            return 0.0
        ranges = np.asarray(
            scan.obstacle_ranges
            if scan.obstacle_ranges is not None
            else scan.ranges,
            dtype=np.float64,
        )[indices]
        angles = float(scan.angle_min) + indices * float(
            scan.angle_increment
        )
        valid = np.isfinite(ranges)
        if int(valid.sum()) < 2:
            return 0.0
        points = np.column_stack((
            ranges[valid] * np.cos(angles[valid]),
            ranges[valid] * np.sin(angles[valid]),
        ))
        span = np.ptp(points, axis=0)
        return float(np.linalg.norm(span))

    def _vehicle_shape(self, cluster, cluster_index):
        extents = getattr(self, "_bootstrap_cluster_extents_m", ())
        extent = (
            extents[cluster_index]
            if cluster_index < len(extents)
            else None
        )
        support_valid = bool(
            self.vehicle_minimum_support_beams
            <= int(cluster.support_beams)
            <= self.vehicle_maximum_support_beams
        )
        if extent is None:
            return support_valid, None
        return bool(
            support_valid
            and self.vehicle_minimum_extent_m
            <= float(extent)
            <= self.vehicle_maximum_extent_m
        ), float(extent)

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
            vehicle_shape, vehicle_extent_m = self._vehicle_shape(
                cluster, cluster_index
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
                    "minimum_range_m": float(cluster.minimum_range_m),
                    "vehicle_shape": vehicle_shape,
                    "vehicle_extent_m": vehicle_extent_m,
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
            "vehicle_extent_m": candidate.get("vehicle_extent_m"),
            "dynamic_shape": (
                "wide_vehicle"
                if bool(candidate.get("vehicle_shape", False))
                else "compact"
            ),
        }

    def _temporal_flow_matches(self, candidates, observation):
        """Cross-check odometry-frame motion against one scan-flow threat."""

        if not self.temporal_flow_threat_preemption_enabled:
            return ()
        flow = dict(
            observation.auxiliary.get("temporal_scan_flow", {}) or {}
        )
        if (
            not bool(flow.get("valid", False))
            or float(flow.get("ttc_s", float("inf")))
            > self.temporal_flow_threat_maximum_ttc_s
            or int(flow.get("support_beams", 0))
            < self.temporal_flow_threat_minimum_support_beams
        ):
            return ()
        center_angle = float(flow.get("center_angle_rad", 0.0))
        clearance = float(flow.get("clearance_m", float("inf")))
        if not np.isfinite(clearance):
            return ()
        robot = np.asarray(
            (observation.pose.x, observation.pose.y), dtype=np.float64
        )
        heading = float(observation.pose.theta)
        matches = []
        for value in candidates:
            candidate = dict(value)
            relative = np.asarray(
                candidate["center"], dtype=np.float64
            ) - robot
            bearing = float(
                np.arctan2(relative[1], relative[0]) - heading
            )
            bearing = float(np.arctan2(np.sin(bearing), np.cos(bearing)))
            angle_error = float(np.arctan2(
                np.sin(center_angle - bearing),
                np.cos(center_angle - bearing),
            ))
            range_error = abs(
                float(candidate["minimum_range_m"]) - clearance
            )
            if (
                abs(angle_error)
                <= self.temporal_flow_threat_angle_tolerance_rad
                and range_error
                <= self.temporal_flow_threat_range_tolerance_m
            ):
                candidate.update({
                    "temporal_flow_angle_error_rad": angle_error,
                    "temporal_flow_range_error_m": range_error,
                    "temporal_flow_ttc_s": float(flow["ttc_s"]),
                    "temporal_flow_support_beams": int(
                        flow["support_beams"]
                    ),
                })
                matches.append(candidate)
        return tuple(sorted(
            matches,
            key=lambda item: (
                float(item["temporal_flow_ttc_s"]),
                abs(float(item["temporal_flow_angle_error_rad"])),
                float(item["temporal_flow_range_error_m"]),
            ),
        ))

    def _preempt_background_track(
        self, raw_clusters, threat, observation, timestamp
    ):
        """Release one far background IMM slot for a corroborated threat."""

        if not getattr(
            self,
            "temporal_flow_threat_preemption_allow_active_reset",
            True,
        ):
            return None, None

        if any(
            tracker.predictor.state is None for tracker in self.trackers
        ):
            return None, None
        threat_center = np.asarray(threat["center"], dtype=np.float64)
        robot = np.asarray(
            (observation.pose.x, observation.pose.y), dtype=np.float64
        )
        choices = []
        for index, tracker in enumerate(self.trackers):
            if self._temporal_flow_threat_hold[index] > 0:
                continue
            reference = tracker._association_reference(timestamp)
            if reference is None:
                continue
            threat_distance = float(
                np.linalg.norm(np.asarray(reference) - threat_center)
            )
            if threat_distance <= self.config.association_gate_m:
                continue
            choices.append((
                -float(np.linalg.norm(np.asarray(reference) - robot)),
                index,
                np.asarray(reference, dtype=np.float64),
            ))
        if not choices:
            return None, None
        _, victim, reference = sorted(choices)[0]
        background_cluster = None
        distances = [
            float(np.linalg.norm(cluster.center - reference))
            for cluster in raw_clusters
        ]
        if distances:
            nearest = int(np.argmin(distances))
            if distances[nearest] <= self.config.association_gate_m:
                background_cluster = nearest
        self.trackers[victim].reset()
        self._temporal_flow_threat_hold[victim] = 0
        return int(victim), background_cluster

    def _assign(self, clusters, timestamp):
        """Assign flow threats before ordinary uninitialized background."""

        self._temporal_flow_threat_hold = [
            max(int(value) - 1, 0)
            for value in self._temporal_flow_threat_hold
        ]
        assignments = {}
        available_tracks = set(range(self.maximum_tracks))
        available_clusters = set(range(len(clusters)))
        pairs = []
        for track_index, tracker in enumerate(self.trackers):
            reference = tracker._association_reference(timestamp)
            if reference is None:
                continue
            for cluster_index, cluster in enumerate(clusters):
                distance = float(np.linalg.norm(cluster.center - reference))
                if distance <= self.config.association_gate_m:
                    pairs.append((distance, track_index, cluster_index))
        for distance, track_index, cluster_index in sorted(pairs):
            if (
                track_index not in available_tracks
                or cluster_index not in available_clusters
            ):
                continue
            assignments[track_index] = (cluster_index, distance)
            available_tracks.remove(track_index)
            available_clusters.remove(cluster_index)

        uninitialized = [
            index for index in sorted(available_tracks)
            if self.trackers[index].predictor.state is None
        ]
        remaining_clusters = sorted(
            available_clusters,
            key=lambda index: (
                0
                if index in self._temporal_flow_threat_retained_indices
                else 1,
                -clusters[index].support_beams,
                clusters[index].minimum_range_m,
                index,
            ),
        )
        for track_index, cluster_index in zip(
            uninitialized, remaining_clusters
        ):
            assignments[track_index] = (cluster_index, None)

        self._temporal_flow_threat_track_indices = {
            track_index
            for track_index, (cluster_index, _) in assignments.items()
            if cluster_index in self._temporal_flow_threat_retained_indices
        }
        for track_index in self._temporal_flow_threat_track_indices:
            self._temporal_flow_threat_hold[track_index] = (
                self.temporal_flow_threat_hold_cycles
            )
        return assignments

    def scan_clusters(self, observation):
        observation = _tracker_observation(observation)
        raw_clusters = tuple(
            self.trackers[0].scan_clusters(observation)
        )
        self._bootstrap_cluster_extents_m = tuple(
            self._cluster_extent_m(cluster, observation.scan)
            for cluster in raw_clusters
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
        flow_matches = self._temporal_flow_matches(
            motion_candidates, observation
        )
        preempted_track = None
        preempted_cluster = None
        if flow_matches:
            preempted_track, preempted_cluster = (
                self._preempt_background_track(
                    raw_clusters, flow_matches[0], observation, timestamp
                )
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
            tracked_indices
            | confirmed_indices
            | provisional_indices
            | {
                int(candidate["cluster_index"])
                for candidate in flow_matches
            }
        )
        if preempted_cluster is not None:
            retained_indices.discard(int(preempted_cluster))
        retained = tuple(
            cluster
            for index, cluster in enumerate(raw_clusters)
            if index in retained_indices
        )
        self._retained_cluster_extents_m = tuple(
            self._bootstrap_cluster_extents_m[index]
            for index in range(len(raw_clusters))
            if index in retained_indices
        )
        retained_raw_indices = tuple(
            index
            for index in range(len(raw_clusters))
            if index in retained_indices
        )
        flow_raw_indices = {
            int(candidate["cluster_index"]) for candidate in flow_matches
        }
        self._temporal_flow_threat_retained_indices = {
            retained_index
            for retained_index, raw_index in enumerate(retained_raw_indices)
            if raw_index in flow_raw_indices
        }
        self._temporal_flow_preempted_track_indices = (
            () if preempted_track is None else (int(preempted_track),)
        )
        self._temporal_flow_preempted_raw_cluster_indices = (
            () if preempted_cluster is None else (int(preempted_cluster),)
        )
        self._temporal_flow_threat_matches = tuple(
            dict(item) for item in flow_matches
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
        tracks = [dict(item) for item in diagnostics.get("tracks", ())]
        for track in tracks:
            assigned = track.get("assigned_cluster_index")
            extent = (
                self._retained_cluster_extents_m[int(assigned)]
                if assigned is not None
                and 0 <= int(assigned) < len(self._retained_cluster_extents_m)
                else None
            )
            track["vehicle_extent_m"] = extent
            track["vehicle_geometry_confirmed"] = bool(
                extent is not None
                and self.vehicle_minimum_extent_m
                <= float(extent)
                <= self.vehicle_maximum_extent_m
                and int(track.get("selected_support_beams") or 0)
                >= self.vehicle_minimum_support_beams
            )
            track_index = int(track.get("track_index", -1))
            track["temporal_flow_threat_matched"] = bool(
                track_index in self._temporal_flow_threat_track_indices
            )
            track["temporal_flow_threat_hold_cycles"] = int(
                self._temporal_flow_threat_hold[track_index]
                if 0 <= track_index < self.maximum_tracks
                else 0
            )
        diagnostics.update(
            {
                "tracks": tuple(tracks),
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
                "temporal_flow_threat_preemption_enabled": bool(
                    self.temporal_flow_threat_preemption_enabled
                ),
                "temporal_flow_threat_preemption_allow_active_reset": bool(
                    getattr(
                        self,
                        "temporal_flow_threat_preemption_allow_active_reset",
                        True,
                    )
                ),
                "temporal_flow_threat_match_count": len(
                    self._temporal_flow_threat_matches
                ),
                "temporal_flow_threat_matches": tuple(
                    dict(item) for item in self._temporal_flow_threat_matches
                ),
                "temporal_flow_preempted_track_indices": (
                    self._temporal_flow_preempted_track_indices
                ),
                "temporal_flow_preempted_raw_cluster_indices": (
                    self._temporal_flow_preempted_raw_cluster_indices
                ),
                "temporal_flow_threat_track_indices": tuple(sorted(
                    index
                    for index, hold in enumerate(
                        self._temporal_flow_threat_hold
                    )
                    if hold > 0
                )),
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
