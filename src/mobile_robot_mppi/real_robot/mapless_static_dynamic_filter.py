"""Map-free causal static/dynamic classification for real lidar tracks."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from mobile_robot_mppi.real_robot.motion_bootstrap_tracker import (
    MotionBootstrapMultiObstacleTracker,
)


class MaplessStaticDynamicFilter:
    """Expose forecasts only for smooth motion in the local odometry frame.

    The wrapped tracker already consumes ego-motion-compensated scans, so a
    static object should remain at one odometry-frame position while the base
    moves.  Classification uses only causal track measurements from the live
    run and never consumes a pre-recorded map.
    """

    def __init__(
        self,
        tracker,
        history_size=8,
        minimum_samples=3,
        minimum_duration_s=0.50,
        static_speed_mps=0.10,
        dynamic_speed_mps=0.20,
        dynamic_displacement_m=0.18,
        minimum_direction_coherence=0.80,
        maximum_fit_residual_m=0.06,
        maximum_step_m=0.20,
        maximum_association_distance_m=0.45,
        maximum_history_gap_s=0.70,
        fast_minimum_duration_s=0.27,
        fast_minimum_speed_mps=0.55,
        fast_minimum_displacement_m=0.18,
        fast_minimum_direction_coherence=0.90,
        fast_maximum_fit_residual_m=0.04,
        fast_maximum_step_m=0.22,
        fast_minimum_tracker_speed_mps=0.30,
        persistent_minimum_tracker_speed_mps=0.15,
        minimum_dynamic_support_beams=3,
        vehicle_minimum_support_beams=6,
        vehicle_minimum_samples=3,
        vehicle_minimum_duration_s=0.20,
        vehicle_minimum_speed_mps=0.20,
        vehicle_minimum_displacement_m=0.07,
        vehicle_minimum_tracker_speed_mps=0.12,
        vehicle_minimum_direction_coherence=0.75,
        vehicle_maximum_fit_residual_m=0.08,
        vehicle_maximum_step_m=0.45,
        vehicle_maximum_extent_relative_span=0.35,
        allow_compact_dynamic=False,
        vehicle_shape_hold_cycles=4,
        vehicle_provisional_minimum_streak=2,
        vehicle_provisional_minimum_displacement_m=0.06,
        vehicle_provisional_minimum_closing_speed_mps=0.12,
        vehicle_provisional_maximum_ttc_s=3.00,
        vehicle_provisional_maximum_closest_approach_m=0.75,
        dynamic_memory_ttl_s=0.90,
        dynamic_handoff_gate_m=0.35,
        dynamic_hold_cycles=2,
        static_track_eviction_cycles=0,
        allow_temporal_flow_provisional=False,
        allow_recent_vehicle_fragment_dynamic=False,
        recent_vehicle_minimum_direction_coherence=0.45,
        dynamic_classification_temporal_corroboration_enabled=False,
        dynamic_classification_temporal_corroboration_maximum_ttc_s=6.0,
        dynamic_classification_temporal_corroboration_hold_cycles=12,
        dynamic_classification_temporal_corroboration_minimum_support_beams=3,
        dynamic_classification_collision_course_bypass_enabled=False,
        allow_collision_course_provisional=False,
    ):
        if not isinstance(tracker, MotionBootstrapMultiObstacleTracker):
            raise TypeError("mapless filter requires motion-bootstrap tracker")
        self.tracker = tracker
        self.config = tracker.config
        self.maximum_tracks = int(tracker.maximum_tracks)
        self.history_size = int(history_size)
        self.minimum_samples = int(minimum_samples)
        self.minimum_duration_s = float(minimum_duration_s)
        self.static_speed_mps = float(static_speed_mps)
        self.dynamic_speed_mps = float(dynamic_speed_mps)
        self.dynamic_displacement_m = float(dynamic_displacement_m)
        self.minimum_direction_coherence = float(
            minimum_direction_coherence
        )
        self.maximum_fit_residual_m = float(maximum_fit_residual_m)
        self.maximum_step_m = float(maximum_step_m)
        self.maximum_association_distance_m = float(
            maximum_association_distance_m
        )
        self.maximum_history_gap_s = float(maximum_history_gap_s)
        self.fast_minimum_duration_s = float(
            fast_minimum_duration_s
        )
        self.fast_minimum_speed_mps = float(fast_minimum_speed_mps)
        self.fast_minimum_displacement_m = float(
            fast_minimum_displacement_m
        )
        self.fast_minimum_direction_coherence = float(
            fast_minimum_direction_coherence
        )
        self.fast_maximum_fit_residual_m = float(
            fast_maximum_fit_residual_m
        )
        self.fast_maximum_step_m = float(fast_maximum_step_m)
        self.fast_minimum_tracker_speed_mps = float(
            fast_minimum_tracker_speed_mps
        )
        self.persistent_minimum_tracker_speed_mps = float(
            persistent_minimum_tracker_speed_mps
        )
        self.minimum_dynamic_support_beams = int(
            minimum_dynamic_support_beams
        )
        self.vehicle_minimum_support_beams = int(
            vehicle_minimum_support_beams
        )
        self.vehicle_minimum_samples = int(vehicle_minimum_samples)
        self.vehicle_minimum_duration_s = float(
            vehicle_minimum_duration_s
        )
        self.vehicle_minimum_speed_mps = float(
            vehicle_minimum_speed_mps
        )
        self.vehicle_minimum_displacement_m = float(
            vehicle_minimum_displacement_m
        )
        self.vehicle_minimum_tracker_speed_mps = float(
            vehicle_minimum_tracker_speed_mps
        )
        self.vehicle_minimum_direction_coherence = float(
            vehicle_minimum_direction_coherence
        )
        self.vehicle_maximum_fit_residual_m = float(
            vehicle_maximum_fit_residual_m
        )
        self.vehicle_maximum_step_m = float(vehicle_maximum_step_m)
        self.vehicle_maximum_extent_relative_span = float(
            vehicle_maximum_extent_relative_span
        )
        self.allow_compact_dynamic = bool(allow_compact_dynamic)
        self.vehicle_shape_hold_cycles = int(vehicle_shape_hold_cycles)
        self.vehicle_provisional_minimum_streak = int(
            vehicle_provisional_minimum_streak
        )
        self.vehicle_provisional_minimum_displacement_m = float(
            vehicle_provisional_minimum_displacement_m
        )
        self.vehicle_provisional_minimum_closing_speed_mps = float(
            vehicle_provisional_minimum_closing_speed_mps
        )
        self.vehicle_provisional_maximum_ttc_s = float(
            vehicle_provisional_maximum_ttc_s
        )
        self.vehicle_provisional_maximum_closest_approach_m = float(
            vehicle_provisional_maximum_closest_approach_m
        )
        self.dynamic_memory_ttl_s = float(dynamic_memory_ttl_s)
        self.dynamic_handoff_gate_m = float(dynamic_handoff_gate_m)
        self.dynamic_hold_cycles = int(dynamic_hold_cycles)
        self.static_track_eviction_cycles = int(
            static_track_eviction_cycles
        )
        self.allow_temporal_flow_provisional = bool(
            allow_temporal_flow_provisional
        )
        self.allow_recent_vehicle_fragment_dynamic = bool(
            allow_recent_vehicle_fragment_dynamic
        )
        self.recent_vehicle_minimum_direction_coherence = float(
            recent_vehicle_minimum_direction_coherence
        )
        self.dynamic_classification_temporal_corroboration_enabled = bool(
            dynamic_classification_temporal_corroboration_enabled
        )
        self.dynamic_classification_temporal_corroboration_maximum_ttc_s = float(
            dynamic_classification_temporal_corroboration_maximum_ttc_s
        )
        self.dynamic_classification_temporal_corroboration_hold_cycles = int(
            dynamic_classification_temporal_corroboration_hold_cycles
        )
        self.dynamic_classification_temporal_corroboration_minimum_support_beams = int(
            dynamic_classification_temporal_corroboration_minimum_support_beams
        )
        self.dynamic_classification_collision_course_bypass_enabled = bool(
            dynamic_classification_collision_course_bypass_enabled
        )
        # A strict collision-course certificate is allowed to retain an
        # already-valid forecast before the slower semantic dynamic label is
        # complete.  It is deliberately separate from classification: the
        # certificate requires coherent motion, support, TTC and closest
        # approach gates in _strong_motion_collision_course().
        self.allow_collision_course_provisional = bool(
            allow_collision_course_provisional
        )
        if self.dynamic_hold_cycles < 0:
            raise ValueError("dynamic hold cycles cannot be negative")
        if not 0.0 <= self.recent_vehicle_minimum_direction_coherence <= 1.0:
            raise ValueError(
                "recent vehicle direction coherence must be in [0, 1]"
            )
        if self.static_track_eviction_cycles < 0:
            raise ValueError("static track eviction cycles cannot be negative")
        if (
            self.dynamic_classification_temporal_corroboration_maximum_ttc_s <= 0.0
            or self.dynamic_classification_temporal_corroboration_hold_cycles < 1
            or self.dynamic_classification_temporal_corroboration_minimum_support_beams < 1
        ):
            raise ValueError(
                "dynamic temporal-corroboration gates must be positive"
            )
        self.reset_filter_state()

    @classmethod
    def from_existing(cls, tracker, **kwargs):
        # Keyword overrides are used only by explicitly selected real-robot
        # deployment profiles.  With no overrides the historical constructor
        # and classification thresholds remain exactly unchanged.
        return cls(tracker, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tracker, name)

    def reset_filter_state(self):
        self.histories = tuple(
            deque(maxlen=self.history_size)
            for _ in range(self.maximum_tracks)
        )
        self.labels = ["unknown"] * self.maximum_tracks
        self.dynamic_hold = [0] * self.maximum_tracks
        self.static_track_streak = [0] * self.maximum_tracks
        self.vehicle_shape_hold = [0] * self.maximum_tracks
        self.last_track_updates = [None] * self.maximum_tracks
        self.dynamic_memories = []
        self.next_dynamic_memory_id = 0
        self.dynamic_classification_temporal_corroboration_remaining = 0
        self.dynamic_classification_temporal_corroboration_current = False
        self.dynamic_classification_temporal_corroboration_primed = False

    def reset(self):
        self.tracker.reset()
        self.reset_filter_state()

    def prime_dynamic_classification_temporal_corroboration(self):
        """Exercise the gated forecast path during disarmed synthetic warmup."""
        if self.dynamic_classification_temporal_corroboration_enabled:
            self.dynamic_classification_temporal_corroboration_remaining = (
                self.dynamic_classification_temporal_corroboration_hold_cycles
            )
            # Synthetic warm-start geometry has no TemporalScanFlow-to-cluster
            # association because it is deliberately injected after that
            # stage.  Permit the next disarmed update only; reset() clears this
            # state before the Pi connection opens, so physical observations
            # still require the production track-specific match.
            self.dynamic_classification_temporal_corroboration_primed = True

    def _strong_motion_collision_course(self, evidence, track, observation):
        """Certify an early coherent track without radial scan-flow evidence.

        Consecutive-beam range flow is deliberately conservative and becomes
        observable late for a lateral crossing.  The odometry-frame tracker,
        however, can already contain a smooth physical trajectory.  Admit
        only the strict intersection of strong motion evidence and a
        robot-relative closest-approach calculation.  This is not a generic
        motion bypass: distant/background motion, incoherent cluster hopping,
        rear motion and trajectories that miss the chassis remain gated by
        temporal flow.
        """

        if not bool(getattr(
            self,
            "dynamic_classification_collision_course_bypass_enabled",
            False,
        )):
            return False, {}
        if not bool(track.get("associated", False)):
            return False, {}
        try:
            position = np.asarray(
                (track["measurement_x"], track["measurement_y"]),
                dtype=np.float64,
            )
            track_velocity = np.asarray(
                (
                    evidence["fitted_velocity_x_mps"],
                    evidence["fitted_velocity_y_mps"],
                ),
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            return False, {}
        if not np.isfinite(position).all() or not np.isfinite(
            track_velocity
        ).all():
            return False, {}

        pose = observation.pose
        yaw = float(pose.theta)
        relative_position = position - np.asarray(
            (float(pose.x), float(pose.y)), dtype=np.float64
        )
        current_range = float(np.linalg.norm(relative_position))
        bearing = float(np.arctan2(
            np.sin(
                np.arctan2(relative_position[1], relative_position[0]) - yaw
            ),
            np.cos(
                np.arctan2(relative_position[1], relative_position[0]) - yaw
            ),
        ))
        robot_velocity = float(observation.twist.v) * np.asarray(
            (np.cos(yaw), np.sin(yaw)), dtype=np.float64
        )
        relative_velocity = track_velocity - robot_velocity
        relative_speed_squared = float(relative_velocity @ relative_velocity)
        closest_time = (
            -float(relative_position @ relative_velocity)
            / relative_speed_squared
            if relative_speed_squared > 1.0e-4
            else float("inf")
        )
        closest_position = (
            relative_position + closest_time * relative_velocity
            if np.isfinite(closest_time)
            else relative_position
        )
        closest_distance = float(np.linalg.norm(closest_position))
        diagnostics = {
            "current_range_m": current_range,
            "bearing_rad": bearing,
            "closest_approach_time_s": closest_time,
            "closest_approach_distance_m": closest_distance,
        }
        certified = bool(
            int(evidence.get("sample_count", 0)) >= 6
            and float(evidence.get("duration_s", 0.0)) >= 0.50
            and float(evidence.get("fitted_speed_mps", 0.0)) >= 0.25
            and float(evidence.get("net_displacement_m", 0.0)) >= 0.22
            and float(evidence.get("direction_coherence", 0.0)) >= 0.70
            and float(evidence.get("fit_residual_m", float("inf"))) <= 0.05
            and float(evidence.get("maximum_step_m", float("inf"))) <= 0.18
            and int(track.get("selected_support_beams", 0) or 0) >= 6
            and current_range <= 3.50
            and abs(bearing) <= 0.5 * np.pi
            and 0.0 < closest_time <= 4.0
            and closest_distance <= 1.20
        )
        return certified, diagnostics

    def _evidence(self, history):
        values = np.asarray(history, dtype=np.float64)
        result = {
            "sample_count": int(values.shape[0]),
            "duration_s": 0.0,
            "net_displacement_m": 0.0,
            "path_length_m": 0.0,
            "direction_coherence": 0.0,
            "fitted_speed_mps": 0.0,
            "fitted_velocity_x_mps": 0.0,
            "fitted_velocity_y_mps": 0.0,
            "fit_residual_m": 0.0,
            "maximum_step_m": 0.0,
            "vehicle_extent_sample_count": 0,
            "vehicle_extent_median_m": None,
            "vehicle_extent_relative_span": None,
            "vehicle_extent_stable": True,
            "dynamic_mode": None,
        }
        if values.ndim == 2 and values.shape[1] >= 4:
            extents = values[:, 3]
            extents = extents[np.isfinite(extents) & (extents > 0.0)]
            if extents.size:
                median_extent = float(np.median(extents))
                relative_span = float(
                    (np.max(extents) - np.min(extents))
                    / max(median_extent, 0.10)
                )
                result.update(
                    {
                        "vehicle_extent_sample_count": int(extents.size),
                        "vehicle_extent_median_m": median_extent,
                        "vehicle_extent_relative_span": relative_span,
                        "vehicle_extent_stable": bool(
                            extents.size < 2
                            or relative_span
                            <= self.vehicle_maximum_extent_relative_span
                        ),
                    }
                )
        if values.shape[0] < 2:
            return result
        times = values[:, 0]
        points = values[:, 1:3]
        duration = float(times[-1] - times[0])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        net = float(np.linalg.norm(points[-1] - points[0]))
        path_length = float(np.sum(steps))
        centered = times - float(np.mean(times))
        denominator = float(np.sum(centered ** 2))
        velocity = (
            np.zeros((2,), dtype=np.float64)
            if denominator <= 1.0e-12
            else np.sum(
                centered[:, None] * points, axis=0
            ) / denominator
        )
        intercept = np.mean(points, axis=0)
        fitted = intercept[None, :] + centered[:, None] * velocity[None, :]
        residual = float(
            np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1)))
        )
        result.update(
            {
                "duration_s": duration,
                "net_displacement_m": net,
                "path_length_m": path_length,
                "direction_coherence": (
                    0.0 if path_length <= 1.0e-9 else net / path_length
                ),
                "fitted_speed_mps": float(np.linalg.norm(velocity)),
                "fitted_velocity_x_mps": float(velocity[0]),
                "fitted_velocity_y_mps": float(velocity[1]),
                "fit_residual_m": residual,
                "maximum_step_m": float(np.max(steps)),
            }
        )
        return result

    def _classify(self, evidence, diagnostics):
        persistent_evidence = bool(
            evidence["sample_count"] >= self.minimum_samples
            and evidence["duration_s"] >= self.minimum_duration_s
        )
        if (
            persistent_evidence
            and
            evidence["fitted_speed_mps"] <= self.static_speed_mps
            and evidence["net_displacement_m"] <= 0.08
            and evidence["fit_residual_m"] <= self.maximum_fit_residual_m
        ):
            return "static"
        stable_association = bool(
            not diagnostics.get("change_triggered", False)
            and not diagnostics.get("recovery_active", False)
            and not diagnostics.get("dropout_guard_triggered", False)
            and float(
                diagnostics.get("association_distance_m") or 0.0
            )
            <= self.maximum_association_distance_m
        )
        fast_dynamic = bool(
            evidence["sample_count"] >= 2
            and evidence["duration_s"] >= self.fast_minimum_duration_s
            and evidence["fitted_speed_mps"]
            >= self.fast_minimum_speed_mps
            and evidence["net_displacement_m"]
            >= self.fast_minimum_displacement_m
            and evidence["direction_coherence"]
            >= self.fast_minimum_direction_coherence
            and evidence["fit_residual_m"]
            <= self.fast_maximum_fit_residual_m
            and evidence["maximum_step_m"] <= self.fast_maximum_step_m
            and float(diagnostics.get("measurement_speed_mps") or 0.0)
            >= self.fast_minimum_tracker_speed_mps
        )
        persistent_dynamic = bool(
            persistent_evidence
            and evidence["fitted_speed_mps"] >= self.dynamic_speed_mps
            and evidence["net_displacement_m"]
            >= self.dynamic_displacement_m
            and evidence["direction_coherence"]
            >= self.minimum_direction_coherence
            and evidence["fit_residual_m"]
            <= self.maximum_fit_residual_m
            and evidence["maximum_step_m"] <= self.maximum_step_m
            and float(diagnostics.get("measurement_speed_mps") or 0.0)
            >= self.persistent_minimum_tracker_speed_mps
        )
        vehicle_dynamic = bool(
            evidence["sample_count"] >= self.vehicle_minimum_samples
            and evidence["vehicle_extent_stable"]
            and evidence["duration_s"] >= self.vehicle_minimum_duration_s
            and evidence["fitted_speed_mps"]
            >= self.vehicle_minimum_speed_mps
            and evidence["net_displacement_m"]
            >= self.vehicle_minimum_displacement_m
            and evidence["direction_coherence"]
            >= self.vehicle_minimum_direction_coherence
            and evidence["fit_residual_m"]
            <= self.vehicle_maximum_fit_residual_m
            and evidence["maximum_step_m"]
            <= self.vehicle_maximum_step_m
            and float(diagnostics.get("measurement_speed_mps") or 0.0)
            >= self.vehicle_minimum_tracker_speed_mps
            and (
                int(diagnostics.get("selected_support_beams") or 0)
                >= self.vehicle_minimum_support_beams
                or bool(
                    diagnostics.get(
                        "mapless_vehicle_shape_recent", False
                    )
                )
            )
        )
        vehicle_shape_recent = bool(
            diagnostics.get("mapless_vehicle_shape_recent", False)
        )
        # A walking person seen by the Mid-360 often alternates between a
        # wide multi-beam body cluster and one- or two-beam leg fragments.
        # Requiring a stable extent in that sequence delayed the 20260804
        # physical crossing until TTC had already fallen to 1.21 s.  Admit
        # the fragment only after a recent wide observation and a smooth,
        # persistent odometry-frame translation; the generic profile remains
        # unchanged unless this explicitly selected real-person gate is on.
        recent_vehicle_fragment_dynamic = bool(
            self.allow_recent_vehicle_fragment_dynamic
            and vehicle_shape_recent
            and persistent_evidence
            and evidence["fitted_speed_mps"] >= self.dynamic_speed_mps
            and evidence["net_displacement_m"]
            >= self.dynamic_displacement_m
            and evidence["direction_coherence"]
            >= self.recent_vehicle_minimum_direction_coherence
            and evidence["fit_residual_m"]
            <= self.maximum_fit_residual_m
            and evidence["maximum_step_m"] <= self.maximum_step_m
            and float(diagnostics.get("measurement_speed_mps") or 0.0)
            >= self.persistent_minimum_tracker_speed_mps
        )
        dynamic_support = bool(
            int(diagnostics.get("selected_support_beams") or 0)
            >= self.minimum_dynamic_support_beams
            or (
                recent_vehicle_fragment_dynamic
                and int(diagnostics.get("selected_support_beams") or 0) >= 1
            )
        )
        if (
            stable_association
            and dynamic_support
            and (
                not self.dynamic_classification_temporal_corroboration_enabled
                or bool(diagnostics.get(
                    "mapless_temporal_flow_corroborated", False
                ))
                or bool(diagnostics.get(
                    "mapless_strong_motion_collision_course", False
                ))
            )
            and (
                vehicle_dynamic
                or recent_vehicle_fragment_dynamic
                or (
                    self.allow_compact_dynamic
                    and (fast_dynamic or persistent_dynamic)
                )
            )
        ):
            evidence["dynamic_mode"] = (
                "wide_vehicle"
                if (
                    vehicle_dynamic
                    or recent_vehicle_fragment_dynamic
                    or vehicle_shape_recent
                )
                else "compact"
            )
            return "dynamic"
        return "unknown"

    def _vetted_vehicle_provisional(self, candidates):
        result = []
        for value in tuple(candidates or ()):
            candidate = dict(value)
            if (
                candidate.get("dynamic_shape") == "wide_vehicle"
                and int(candidate.get("streak") or 0)
                >= self.vehicle_provisional_minimum_streak
                and float(candidate.get("total_displacement_m") or 0.0)
                >= self.vehicle_provisional_minimum_displacement_m
                and float(candidate.get("closing_speed_mps") or 0.0)
                >= self.vehicle_provisional_minimum_closing_speed_mps
                and 0.0
                < float(candidate.get("closest_approach_time_s") or 0.0)
                <= self.vehicle_provisional_maximum_ttc_s
                and float(
                    candidate.get(
                        "closest_approach_distance_m", float("inf")
                    )
                )
                <= self.vehicle_provisional_maximum_closest_approach_m
            ):
                candidate["early_wide_vehicle"] = True
                result.append(candidate)
        return tuple(result)

    def update(self, observation):
        synthetic_prime_current = bool(
            self.dynamic_classification_temporal_corroboration_primed
        )
        self.dynamic_classification_temporal_corroboration_primed = False
        flow = dict(
            getattr(observation, "auxiliary", {}).get(
                "temporal_scan_flow", {}
            )
            or {}
        )
        flow_ttc_s = float(
            flow.get("ttc_s", float("inf")) or float("inf")
        )
        self.dynamic_classification_temporal_corroboration_current = bool(
            self.dynamic_classification_temporal_corroboration_enabled
            and flow.get("valid", False)
            and np.isfinite(flow_ttc_s)
            and flow_ttc_s
            <= self.dynamic_classification_temporal_corroboration_maximum_ttc_s
            and int(flow.get("support_beams", 0) or 0)
            >= self.dynamic_classification_temporal_corroboration_minimum_support_beams
        )
        if self.dynamic_classification_temporal_corroboration_current:
            self.dynamic_classification_temporal_corroboration_remaining = (
                self.dynamic_classification_temporal_corroboration_hold_cycles
            )
        elif self.dynamic_classification_temporal_corroboration_remaining > 0:
            self.dynamic_classification_temporal_corroboration_remaining -= 1
        result = self.tracker.update(observation)
        diagnostics = dict(result.diagnostics)
        track_values = [
            dict(item) for item in diagnostics.get("tracks", ())
        ]
        timestamp = float(observation.timestamp)
        self.dynamic_memories = [
            item
            for item in self.dynamic_memories
            if float(item["expires_at"]) >= timestamp
        ]
        claimed_memory_ids = set()
        static_evicted_track_indices = []
        evidence_values = []
        for index in range(self.maximum_tracks):
            track = (
                track_values[index]
                if index < len(track_values)
                else {}
            )
            history = self.histories[index]
            update_count = track.get("update_count")
            previous_update = self.last_track_updates[index]
            reset_identity = bool(
                previous_update is not None
                and update_count is not None
                and int(update_count) <= int(previous_update)
            )
            association_distance = track.get("association_distance_m")
            preserve_vehicle_shape = bool(
                self.vehicle_shape_hold[index] > 0
                and association_distance is not None
                and float(association_distance) <= 0.65
            )
            if (
                association_distance is not None
                and float(association_distance)
                > self.maximum_association_distance_m
            ):
                reset_identity = True
            if (
                history
                and timestamp - float(history[-1][0])
                > self.maximum_history_gap_s
            ):
                reset_identity = True
            if reset_identity:
                history.clear()
                self.labels[index] = "unknown"
                self.dynamic_hold[index] = 0
                if not preserve_vehicle_shape:
                    self.vehicle_shape_hold[index] = 0
            self.last_track_updates[index] = update_count
            x_value = track.get("measurement_x")
            y_value = track.get("measurement_y")
            extent_value = track.get("vehicle_extent_m")
            if (
                bool(track.get("associated", False))
                and x_value is not None
                and y_value is not None
            ):
                history.append(
                    (
                        timestamp,
                        float(x_value),
                        float(y_value),
                        (
                            float(extent_value)
                            if extent_value is not None
                            else float("nan")
                        ),
                    )
                )
            support_beams = int(
                track.get("selected_support_beams") or 0
            )
            geometry_confirmed = track.get("vehicle_geometry_confirmed")
            vehicle_shape_observed = bool(
                geometry_confirmed
                if geometry_confirmed is not None
                else support_beams >= self.vehicle_minimum_support_beams
            )
            if bool(track.get("associated", False)) and vehicle_shape_observed:
                self.vehicle_shape_hold[index] = (
                    self.vehicle_shape_hold_cycles
                )
            elif self.vehicle_shape_hold[index] > 0:
                self.vehicle_shape_hold[index] -= 1
            track["mapless_vehicle_shape_recent"] = bool(
                self.vehicle_shape_hold[index] > 0
            )
            # A scan-flow alert belongs only to the cluster/track that the
            # bootstrap tracker spatially matched.  The old global hold
            # broadcast one distant Livox range-flow fluctuation to every IMM
            # track, so unrelated background fragments could all become
            # dynamic and fight for the escape direction.  Keep the global
            # hold as the temporal validity window, but intersect it with the
            # track-specific match/hold maintained by MotionBootstrapTracker.
            track_temporal_flow_hold = int(
                track.get("temporal_flow_threat_hold_cycles") or 0
            )
            track["mapless_temporal_flow_corroborated"] = bool(
                self.dynamic_classification_temporal_corroboration_remaining > 0
                and (
                    track_temporal_flow_hold > 0
                    or synthetic_prime_current
                )
            )
            evidence = self._evidence(history)
            (
                strong_collision_course,
                strong_collision_course_diagnostics,
            ) = self._strong_motion_collision_course(
                evidence, track, observation
            )
            track["mapless_strong_motion_collision_course"] = bool(
                strong_collision_course
            )
            track[
                "mapless_strong_motion_collision_course_diagnostics"
            ] = dict(strong_collision_course_diagnostics)
            candidate_label = self._classify(evidence, track)
            if (
                candidate_label == "static"
                and bool(track.get("associated", False))
            ):
                self.static_track_streak[index] += 1
            else:
                self.static_track_streak[index] = 0
            evict_static_track = bool(
                self.static_track_eviction_cycles > 0
                and self.static_track_streak[index]
                >= self.static_track_eviction_cycles
            )
            inherited_memory = None
            inherited_distance = None
            position = (
                None
                if x_value is None or y_value is None
                else np.asarray(
                    (float(x_value), float(y_value)), dtype=np.float64
                )
            )
            stable_handoff = bool(
                position is not None
                and bool(track.get("associated", False))
                and bool(track.get("forecast_valid", False))
                and candidate_label != "static"
                and not track.get("change_triggered", False)
                and not track.get("recovery_active", False)
                and not track.get("dropout_guard_triggered", False)
                and float(association_distance or 0.0)
                <= self.maximum_association_distance_m
            )
            if stable_handoff:
                for memory in self.dynamic_memories:
                    if int(memory["id"]) in claimed_memory_ids:
                        continue
                    elapsed = max(
                        timestamp - float(memory["timestamp"]), 0.0
                    )
                    predicted = (
                        np.asarray(memory["position"], dtype=np.float64)
                        + elapsed
                        * np.asarray(memory["velocity"], dtype=np.float64)
                    )
                    distance = float(np.linalg.norm(position - predicted))
                    if (
                        inherited_distance is None
                        or distance < inherited_distance
                    ):
                        inherited_memory = memory
                        inherited_distance = distance
                if (
                    inherited_distance is not None
                    and inherited_distance > self.dynamic_handoff_gate_m
                ):
                    inherited_memory = None
                    inherited_distance = None
            inherited_dynamic = bool(
                candidate_label == "unknown"
                and inherited_memory is not None
            )
            if candidate_label == "dynamic" or inherited_dynamic:
                self.labels[index] = "dynamic"
                self.dynamic_hold[index] = self.dynamic_hold_cycles
                if inherited_memory is None:
                    inherited_memory = {
                        "id": self.next_dynamic_memory_id,
                        "timestamp": timestamp,
                        "position": position.copy(),
                        "velocity": np.asarray(
                            (
                                evidence["fitted_velocity_x_mps"],
                                evidence["fitted_velocity_y_mps"],
                            ),
                            dtype=np.float64,
                        ),
                        "expires_at": timestamp + self.dynamic_memory_ttl_s,
                        "mode": evidence.get("dynamic_mode") or "compact",
                    }
                    self.next_dynamic_memory_id += 1
                    self.dynamic_memories.append(inherited_memory)
                else:
                    elapsed = max(
                        timestamp - float(inherited_memory["timestamp"]),
                        1.0e-6,
                    )
                    observed_velocity = (
                        position
                        - np.asarray(
                            inherited_memory["position"], dtype=np.float64
                        )
                    ) / elapsed
                    if candidate_label == "dynamic":
                        updated_velocity = np.asarray(
                            (
                                evidence["fitted_velocity_x_mps"],
                                evidence["fitted_velocity_y_mps"],
                            ),
                            dtype=np.float64,
                        )
                    else:
                        updated_velocity = (
                            0.70
                            * np.asarray(
                                inherited_memory["velocity"],
                                dtype=np.float64,
                            )
                            + 0.30 * observed_velocity
                        )
                    inherited_memory.update(
                        {
                            "timestamp": timestamp,
                            "position": position.copy(),
                            "velocity": updated_velocity,
                            "expires_at": (
                                timestamp + self.dynamic_memory_ttl_s
                            ),
                        }
                    )
                claimed_memory_ids.add(int(inherited_memory["id"]))
            elif candidate_label == "static":
                self.labels[index] = "static"
                self.dynamic_hold[index] = 0
            elif (
                self.labels[index] == "dynamic"
                and self.dynamic_hold[index] > 0
                # CA-IMM keeps a causal forecast alive across short lidar
                # cluster dropouts. Requiring a fresh association here threw
                # that valid forecast away and made Full alternate between
                # forecast/no-forecast planners on fragmented human-leg scans.
                # The low-level forecast validity/staleness gate remains the
                # fail-closed authority.
                and bool(track.get("forecast_valid", False))
                and not reset_identity
            ):
                self.dynamic_hold[index] -= 1
            else:
                self.labels[index] = "unknown"
                self.dynamic_hold[index] = 0
            track["mapless_classification"] = self.labels[index]
            track["mapless_motion_evidence"] = dict(evidence)
            track["mapless_dynamic_hold_cycles"] = int(
                self.dynamic_hold[index]
            )
            track["mapless_dynamic_inherited"] = inherited_dynamic
            track["mapless_dynamic_handoff_distance_m"] = (
                inherited_distance
            )
            track["mapless_dynamic_memory_id"] = (
                None
                if self.labels[index] != "dynamic"
                or inherited_memory is None
                else int(inherited_memory["id"])
            )
            track["mapless_dynamic_mode"] = (
                None
                if self.labels[index] != "dynamic"
                else (
                    evidence.get("dynamic_mode")
                    or (
                        inherited_memory.get("mode")
                        if inherited_memory is not None
                        else (
                            "wide_vehicle"
                            if (
                                not self.allow_compact_dynamic
                                or track.get(
                                    "mapless_vehicle_shape_recent", False
                                )
                            )
                            else "compact"
                        )
                    )
                )
            )
            track["mapless_static_track_streak"] = int(
                self.static_track_streak[index]
            )
            track["mapless_static_track_evicted"] = evict_static_track
            if evict_static_track:
                static_evicted_track_indices.append(index)
            if index < len(track_values):
                track_values[index] = track
            evidence_values.append(evidence)

        # Confirmed background geometry must not occupy all finite IMM slots
        # forever in map-free deployment.  This is disabled by default and is
        # evaluated only after the conservative static classifier has agreed
        # for several associated updates.
        for index in static_evicted_track_indices:
            self.tracker.trackers[index].reset()
            self.histories[index].clear()
            self.labels[index] = "unknown"
            self.dynamic_hold[index] = 0
            self.vehicle_shape_hold[index] = 0
            self.static_track_streak[index] = 0
            self.last_track_updates[index] = None

        forecasts = tuple(result.forecast or ())
        forecast_indices = tuple(
            int(value)
            for value in diagnostics.get(
                "forecast_track_indices", range(len(forecasts))
            )
        )
        provisional_flow_indices = tuple(
            index
            for index, track in enumerate(track_values)
            if (
                self.allow_temporal_flow_provisional
                and bool(track.get("associated", False))
                and bool(track.get("forecast_valid", False))
                and int(
                    track.get("temporal_flow_threat_hold_cycles") or 0
                ) > 0
            )
        )
        collision_course_indices = tuple(
            index
            for index, track in enumerate(track_values)
            if (
                self.allow_collision_course_provisional
                and bool(track.get("associated", False))
                and bool(track.get("forecast_valid", False))
                and bool(track.get(
                    "mapless_strong_motion_collision_course", False
                ))
                and self.labels[index] != "static"
            )
        )
        provisional_indices = tuple(dict.fromkeys(
            (*provisional_flow_indices, *collision_course_indices)
        ))
        provisional_flow_index_set = set(provisional_indices)
        for index in provisional_indices:
            track_values[index][
                "mapless_temporal_flow_provisional"
            ] = True
            track_values[index][
                "mapless_collision_course_provisional"
            ] = bool(index in collision_course_indices)
        retained_pairs = tuple(
            (track_index, forecast)
            for track_index, forecast in zip(
                forecast_indices, forecasts
            )
            if (
                0 <= track_index < self.maximum_tracks
                and (
                    self.labels[track_index] == "dynamic"
                    or track_index in provisional_flow_index_set
                )
            )
        )
        retained_indices = tuple(item[0] for item in retained_pairs)
        retained_forecasts = tuple(item[1] for item in retained_pairs)
        nearest_track_index = diagnostics.get("nearest_track_index")
        vehicle_provisional = self._vetted_vehicle_provisional(
            diagnostics.get("provisional_collision_candidates", ())
        )
        diagnostics.update(
            {
                "tracks": track_values,
                "forecast_track_indices": retained_indices,
                "valid_forecast_count": len(retained_forecasts),
                "forecast_valid": bool(retained_forecasts),
                "nearest_forecast_index": (
                    retained_indices.index(int(nearest_track_index))
                    if nearest_track_index is not None
                    and int(nearest_track_index) in retained_indices
                    else None
                ),
                "motion_confirmed": bool(retained_forecasts),
                "mapless_dynamic_classification_temporal_corroboration_enabled": bool(
                    self.dynamic_classification_temporal_corroboration_enabled
                ),
                "mapless_dynamic_classification_temporal_corroboration_current": bool(
                    self.dynamic_classification_temporal_corroboration_current
                ),
                "mapless_dynamic_classification_temporal_corroboration_remaining": int(
                    self.dynamic_classification_temporal_corroboration_remaining
                ),
                "mapless_dynamic_classification_collision_course_bypass_enabled": bool(
                    getattr(
                        self,
                        "dynamic_classification_collision_course_bypass_enabled",
                        False,
                    )
                ),
                "mapless_strong_motion_collision_course_track_indices": tuple(
                    index
                    for index, track in enumerate(track_values)
                    if bool(track.get(
                        "mapless_strong_motion_collision_course", False
                    ))
                ),
                "mapless_temporal_flow_provisional_track_indices": (
                    provisional_flow_indices
                ),
                "mapless_collision_course_provisional_track_indices": (
                    collision_course_indices
                ),
                "forecast_unavailable_reason": (
                    "available"
                    if retained_forecasts
                    else "mapless_motion_not_consistent"
                ),
                "provisional_collision_candidate_count": len(
                    vehicle_provisional
                ),
                "provisional_collision_candidates": vehicle_provisional,
                "provisional_collision_minimum_ttc_s": (
                    min(
                        float(item["closest_approach_time_s"])
                        for item in vehicle_provisional
                    )
                    if vehicle_provisional
                    else None
                ),
                "mapless_static_dynamic_filter_enabled": True,
                "mapless_dynamic_track_indices": tuple(
                    index
                    for index, label in enumerate(self.labels)
                    if label == "dynamic"
                ),
                "mapless_static_track_indices": tuple(
                    index
                    for index, label in enumerate(self.labels)
                    if label == "static"
                ),
                "mapless_unknown_track_indices": tuple(
                    index
                    for index, label in enumerate(self.labels)
                    if label == "unknown"
                ),
                "mapless_uses_prebuilt_map": False,
                "mapless_dynamic_memory_count": len(
                    self.dynamic_memories
                ),
                "mapless_static_evicted_track_indices": tuple(
                    static_evicted_track_indices
                ),
                "mapless_dynamic_handoff_enabled": True,
                "mapless_vehicle_only_dynamic": bool(
                    not self.allow_compact_dynamic
                ),
            }
        )
        return replace(
            result,
            forecast=(retained_forecasts or None),
            diagnostics=diagnostics,
        )


__all__ = ["MaplessStaticDynamicFilter"]
