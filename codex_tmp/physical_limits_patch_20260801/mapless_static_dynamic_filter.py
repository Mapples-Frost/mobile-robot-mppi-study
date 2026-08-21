"""Map-free causal static/dynamic classification for real lidar tracks."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from complex_scene_tracker import MotionBootstrapMultiObstacleTracker


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
        maximum_association_distance_m=0.30,
        fast_minimum_duration_s=0.27,
        fast_minimum_speed_mps=0.55,
        fast_minimum_displacement_m=0.18,
        fast_minimum_direction_coherence=0.90,
        fast_maximum_fit_residual_m=0.04,
        fast_maximum_step_m=0.22,
        fast_minimum_tracker_speed_mps=0.30,
        persistent_minimum_tracker_speed_mps=0.15,
        minimum_dynamic_support_beams=3,
        vehicle_minimum_support_beams=12,
        vehicle_minimum_duration_s=0.27,
        vehicle_minimum_speed_mps=0.45,
        vehicle_minimum_displacement_m=0.18,
        vehicle_minimum_tracker_speed_mps=0.25,
        vehicle_minimum_direction_coherence=0.92,
        vehicle_maximum_fit_residual_m=0.03,
        vehicle_maximum_step_m=0.60,
        allow_compact_dynamic=False,
        vehicle_shape_hold_cycles=4,
        vehicle_provisional_minimum_streak=2,
        vehicle_provisional_minimum_displacement_m=0.10,
        vehicle_provisional_minimum_closing_speed_mps=0.15,
        vehicle_provisional_maximum_ttc_s=3.00,
        vehicle_provisional_maximum_closest_approach_m=0.90,
        dynamic_memory_ttl_s=0.90,
        dynamic_handoff_gate_m=0.35,
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
        self.reset_filter_state()

    @classmethod
    def from_existing(cls, tracker):
        return cls(tracker)

    def __getattr__(self, name):
        return getattr(self.tracker, name)

    def reset_filter_state(self):
        self.histories = tuple(
            deque(maxlen=self.history_size)
            for _ in range(self.maximum_tracks)
        )
        self.labels = ["unknown"] * self.maximum_tracks
        self.dynamic_hold = [0] * self.maximum_tracks
        self.vehicle_shape_hold = [0] * self.maximum_tracks
        self.last_track_updates = [None] * self.maximum_tracks
        self.dynamic_memories = []
        self.next_dynamic_memory_id = 0

    def reset(self):
        self.tracker.reset()
        self.reset_filter_state()

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
            "dynamic_mode": None,
        }
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
            evidence["sample_count"] >= 2
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
        if (
            stable_association
            and int(diagnostics.get("selected_support_beams") or 0)
            >= self.minimum_dynamic_support_beams
            and (
                vehicle_dynamic
                or (
                    self.allow_compact_dynamic
                    and (fast_dynamic or persistent_dynamic)
                )
            )
        ):
            evidence["dynamic_mode"] = (
                "wide_vehicle"
                if vehicle_dynamic or vehicle_shape_recent
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
            if history and timestamp - float(history[-1][0]) > 0.35:
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
            if (
                bool(track.get("associated", False))
                and x_value is not None
                and y_value is not None
            ):
                history.append(
                    (timestamp, float(x_value), float(y_value))
                )
            support_beams = int(
                track.get("selected_support_beams") or 0
            )
            if (
                bool(track.get("associated", False))
                and support_beams >= self.vehicle_minimum_support_beams
            ):
                self.vehicle_shape_hold[index] = (
                    self.vehicle_shape_hold_cycles
                )
            elif self.vehicle_shape_hold[index] > 0:
                self.vehicle_shape_hold[index] -= 1
            track["mapless_vehicle_shape_recent"] = bool(
                self.vehicle_shape_hold[index] > 0
            )
            evidence = self._evidence(history)
            candidate_label = self._classify(evidence, track)
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
                self.dynamic_hold[index] = 2
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
                and bool(track.get("associated", False))
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
                        else "compact"
                    )
                )
            )
            if index < len(track_values):
                track_values[index] = track
            evidence_values.append(evidence)

        forecasts = tuple(result.forecast or ())
        forecast_indices = tuple(
            int(value)
            for value in diagnostics.get(
                "forecast_track_indices", range(len(forecasts))
            )
        )
        retained_pairs = tuple(
            (track_index, forecast)
            for track_index, forecast in zip(
                forecast_indices, forecasts
            )
            if (
                0 <= track_index < self.maximum_tracks
                and self.labels[track_index] == "dynamic"
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
