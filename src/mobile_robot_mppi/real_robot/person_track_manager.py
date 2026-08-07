"""Causal person-level identity and forecast qualification.

The lidar tracker owns short-lived cluster/leg slots.  Those slots are useful
measurements, but they are not identities: a person's two legs can alternate
between slots and a fragmented scan can temporarily produce no preferred slot.
This module adds a small, deterministic identity layer above the tracker.  It
does not generate chassis commands and it never changes the static/hard-safety
filters.  Its only outputs are causal identity, continuity and forecast
qualification facts for the planner and audit stream.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


_QUALIFICATIONS = ("VALID", "PROVISIONAL", "STALE", "INVALID")


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _unit(value: Sequence[float]) -> Optional[np.ndarray]:
    vector = np.asarray(value, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 1.0e-9:
        return None
    return vector / norm


@dataclass
class _PersonState:
    person_id: int
    timestamp_s: float
    position: np.ndarray
    velocity: np.ndarray
    member_track_indices: Tuple[int, ...]
    age_s: float = 0.0
    missed_cycles: int = 0
    last_qualification: str = "INVALID"
    last_forecast_timestamp_s: float = float("nan")


class PersonTrackManager:
    """Associate low-level tracks into causal person identities.

    Association is greedy over predicted position residuals.  The bounded
    track count and deterministic ordering make this suitable for the 10 Hz
    real-robot loop while remaining easy to replay.  A person identity is
    retained through short slot swaps/dropouts, but stale identity is never
    presented as fresh forecast evidence.
    """

    def __init__(self, config: Mapping[str, Any] = None):
        values = dict(config or {})
        self.enabled = bool(values.get("enabled", True))
        self.maximum_gap_s = float(values.get("maximum_gap_s", 0.65))
        self.association_gate_m = float(
            values.get("association_gate_m", 0.85)
        )
        self.fusion_position_m = float(
            values.get("fusion_position_m", 0.58)
        )
        self.fusion_velocity_difference_mps = float(
            values.get("fusion_velocity_difference_mps", 0.80)
        )
        self.minimum_support_beams = int(
            values.get("minimum_support_beams", 3)
        )
        self.maximum_forecast_age_s = float(
            values.get("maximum_forecast_age_s", 0.45)
        )
        self.stale_hold_cycles = int(values.get("stale_hold_cycles", 4))
        self.minimum_motion_speed_mps = float(
            values.get("minimum_motion_speed_mps", 0.10)
        )
        self.reset()

    def reset(self) -> None:
        self._states: Dict[int, _PersonState] = {}
        self._next_id = 0
        self._last_timestamp_s: Optional[float] = None

    @staticmethod
    def _track_position(track: Mapping[str, Any]) -> Optional[np.ndarray]:
        use_body_centroid = bool(
            track.get("point_cloud_human_candidate", False)
        )
        x = _finite(
            track.get("point_cloud_human_centroid_x")
            if use_body_centroid
            else track.get("measurement_x")
        )
        y = _finite(
            track.get("point_cloud_human_centroid_y")
            if use_body_centroid
            else track.get("measurement_y")
        )
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return np.asarray((x, y), dtype=np.float64)

    @staticmethod
    def _track_velocity(track: Mapping[str, Any]) -> np.ndarray:
        velocity = np.asarray((
            _finite(track.get("measurement_velocity_x_mps"), 0.0),
            _finite(track.get("measurement_velocity_y_mps"), 0.0),
        ), dtype=np.float64)
        return velocity if np.isfinite(velocity).all() else np.zeros(2)

    def _eligible(self, track: Mapping[str, Any]) -> bool:
        if not bool(track.get("associated", False)):
            return False
        if str(track.get("mapless_classification", "unknown")) == "static":
            return False
        return bool(
            track.get("forecast_valid", False)
            or track.get("mapless_strong_motion_collision_course", False)
            or track.get("mapless_temporal_flow_corroborated", False)
            or track.get("temporal_flow_threat_matched", False)
        )

    def _group_tracks(self, tracks: Sequence[Mapping[str, Any]]):
        records = []
        for ordinal, raw in enumerate(tracks or ()):
            track = dict(raw or {})
            position = self._track_position(track)
            if position is None or not self._eligible(track):
                continue
            velocity = self._track_velocity(track)
            records.append((
                int(track.get("track_index", ordinal)),
                position,
                velocity,
                track,
            ))
        groups = []
        for record in sorted(records, key=lambda item: item[0]):
            _, position, velocity, _ = record
            compatible = []
            for index, group in enumerate(groups):
                group_position = np.mean(
                    np.asarray([item[1] for item in group]), axis=0
                )
                group_velocity = np.mean(
                    np.asarray([item[2] for item in group]), axis=0
                )
                if (
                    float(np.linalg.norm(position - group_position))
                    <= self.fusion_position_m
                    and float(np.linalg.norm(velocity - group_velocity))
                    <= self.fusion_velocity_difference_mps
                ):
                    compatible.append(index)
            if compatible:
                groups[compatible[0]].append(record)
            else:
                groups.append([record])
        return groups

    def _associate_groups(self, groups, timestamp_s: float):
        candidates = []
        for group_index, group in enumerate(groups):
            position = np.mean(
                np.asarray([item[1] for item in group]), axis=0
            )
            velocity = np.mean(
                np.asarray([item[2] for item in group]), axis=0
            )
            for person_id, state in self._states.items():
                dt_s = max(0.0, timestamp_s - state.timestamp_s)
                if dt_s > self.maximum_gap_s:
                    continue
                predicted = state.position + state.velocity * dt_s
                residual = float(np.linalg.norm(position - predicted))
                velocity_residual = float(
                    np.linalg.norm(velocity - state.velocity)
                )
                gate = self.association_gate_m + min(
                    0.35, 0.6 * dt_s
                )
                if residual <= gate and velocity_residual <= 1.5:
                    candidates.append((
                        residual + 0.15 * velocity_residual,
                        group_index,
                        person_id,
                        residual,
                    ))
        assignments = {}
        used_groups = set()
        used_people = set()
        for _, group_index, person_id, residual in sorted(candidates):
            if group_index in used_groups or person_id in used_people:
                continue
            assignments[group_index] = (person_id, residual)
            used_groups.add(group_index)
            used_people.add(person_id)
        for group_index in range(len(groups)):
            if group_index not in assignments:
                assignments[group_index] = (self._next_id, None)
                self._next_id += 1
        return assignments

    def _qualification(
        self,
        group,
        forecast_indices: set,
        timestamp_s: float,
        state: _PersonState,
    ) -> str:
        tracks = [item[3] for item in group]
        member_indices = {
            int(track.get("track_index", -1)) for track in tracks
        }
        has_forecast = bool(member_indices & forecast_indices)
        support = max(
            int(track.get("selected_support_beams", 0) or 0)
            for track in tracks
        )
        motion = max(
            _finite(track.get("measurement_speed_mps"), 0.0)
            for track in tracks
        )
        dynamic_label = any(
            str(track.get("mapless_classification", "unknown")) == "dynamic"
            for track in tracks
        )
        corroborated = any(
            bool(track.get("mapless_temporal_flow_corroborated", False))
            or bool(track.get("mapless_strong_motion_collision_course", False))
            or bool(track.get("mapless_person_provisional", False))
            or bool(track.get("point_cloud_human_candidate", False))
            for track in tracks
        )
        point_cloud_human = any(
            bool(track.get("point_cloud_human_candidate", False))
            for track in tracks
        )
        if (
            has_forecast
            and (
                support >= self.minimum_support_beams
                or point_cloud_human
            )
            and (dynamic_label or corroborated)
            and motion >= self.minimum_motion_speed_mps
        ):
            return "VALID"
        if has_forecast and (corroborated or dynamic_label):
            return "PROVISIONAL"
        age = timestamp_s - state.last_forecast_timestamp_s
        if (
            state.last_qualification in {"VALID", "PROVISIONAL", "STALE"}
            and math.isfinite(age)
            and age <= self.maximum_forecast_age_s
            and state.missed_cycles <= self.stale_hold_cycles
        ):
            return "STALE"
        return "INVALID"

    def update(
        self,
        tracker_diagnostics: Mapping[str, Any],
        forecasts: Sequence[Any] = (),
        timestamp_s: float = 0.0,
        pose: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Dict[str, Any]:
        """Annotate tracker diagnostics and return person-level facts."""

        diagnostics = dict(tracker_diagnostics or {})
        if not self.enabled:
            diagnostics["person_tracking_enabled"] = False
            diagnostics["person_forecast_qualification"] = "INVALID"
            return diagnostics
        timestamp_s = float(timestamp_s)
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("person tracker timestamps must increase strictly")
        self._last_timestamp_s = timestamp_s
        tracks = [dict(item or {}) for item in diagnostics.get("tracks", ())]
        groups = self._group_tracks(tracks)
        assignments = self._associate_groups(groups, timestamp_s)
        # Prefer the mapless qualification set.  It contains ordinary
        # dynamic tracks plus the strict person-compatible provisional gate;
        # raw low-level IMM indices are retained only for audit and must not
        # silently become person forecasts.
        forecast_source = "mapless_qualified"
        qualified_indices = diagnostics.get(
            "person_forecast_candidate_track_indices",
            diagnostics.get("forecast_track_indices", ()),
        )
        if "person_forecast_candidate_track_indices" not in diagnostics:
            forecast_source = "legacy_mapless_retained"
        forecast_indices = {int(value) for value in qualified_indices}
        active_people = []
        for group_index, group in enumerate(groups):
            person_id, residual = assignments[group_index]
            position = np.mean(np.asarray([item[1] for item in group]), axis=0)
            velocity = np.mean(np.asarray([item[2] for item in group]), axis=0)
            member_indices = tuple(sorted(int(item[0]) for item in group))
            previous = self._states.get(person_id)
            if previous is None:
                previous = _PersonState(
                    person_id=person_id,
                    timestamp_s=timestamp_s,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    member_track_indices=member_indices,
                )
            else:
                previous.age_s = max(0.0, timestamp_s - previous.timestamp_s)
                previous.missed_cycles = 0
                previous.position = position.copy()
                previous.velocity = velocity.copy()
                previous.timestamp_s = timestamp_s
                previous.member_track_indices = member_indices
            qualification = self._qualification(
                group, forecast_indices, timestamp_s, previous
            )
            if qualification in {"VALID", "PROVISIONAL"}:
                previous.last_forecast_timestamp_s = timestamp_s
            previous.last_qualification = qualification
            self._states[person_id] = previous
            for track in tracks:
                track_index = int(track.get("track_index", -1))
                if track_index in member_indices:
                    track.update({
                        "person_id": int(person_id),
                        "person_member_track_indices": member_indices,
                        "person_association_residual_m": (
                            None if residual is None else float(residual)
                        ),
                        "person_track_age_s": float(previous.age_s),
                        "person_track_continuous": bool(
                            residual is not None and residual <= self.association_gate_m
                        ),
                        "person_forecast_qualification": qualification,
                    })
            active_people.append({
                "person_id": int(person_id),
                "position_x": float(position[0]),
                "position_y": float(position[1]),
                "velocity_x_mps": float(velocity[0]),
                "velocity_y_mps": float(velocity[1]),
                "member_track_indices": member_indices,
                "forecast_qualification": qualification,
                "forecast_track_indices": tuple(
                    sorted(set(member_indices) & forecast_indices)
                ),
                "association_residual_m": (
                    None if residual is None else float(residual)
                ),
            })
        # Advance missing identities through a short, audited coasting state.
        # This preserves the same person ID when legs/body fragments disappear
        # for one or two scans, but does not publish a forecast or command from
        # the extrapolated state.
        active_ids = {item["person_id"] for item in active_people}
        coasting_people = []
        for person_id, state in list(self._states.items()):
            if person_id in active_ids:
                continue
            state.missed_cycles += 1
            state.age_s = max(0.0, timestamp_s - state.timestamp_s)
            if (
                state.age_s > self.maximum_gap_s
                or state.missed_cycles > self.stale_hold_cycles
            ):
                del self._states[person_id]
                continue
            forecast_age_s = timestamp_s - state.last_forecast_timestamp_s
            if (
                state.last_qualification
                in {"VALID", "PROVISIONAL", "STALE"}
                and math.isfinite(forecast_age_s)
                and forecast_age_s <= self.maximum_forecast_age_s
            ):
                predicted = state.position + state.velocity * state.age_s
                state.last_qualification = "STALE"
                coasting_people.append({
                    "person_id": int(person_id),
                    "position_x": float(predicted[0]),
                    "position_y": float(predicted[1]),
                    "velocity_x_mps": float(state.velocity[0]),
                    "velocity_y_mps": float(state.velocity[1]),
                    "member_track_indices": (),
                    "forecast_qualification": "STALE",
                    "forecast_track_indices": (),
                    "association_residual_m": None,
                    "coasting": True,
                    "coasting_age_s": float(state.age_s),
                })
        pose_values = np.asarray(pose, dtype=np.float64).reshape(3)
        selected = None
        # An identity with an unqualified forecast remains in the audit list,
        # but it must not become the geometry source for ScanGuard or the
        # planner.  Otherwise a low-level unknown/background slot could move
        # the dynamic bearing even though its forecast was rejected.
        selectable_people = [
            item
            for item in active_people
            if item["forecast_qualification"] != "INVALID"
        ]
        if selectable_people:
            def rank(item):
                relative = np.asarray((
                    item["position_x"] - pose_values[0],
                    item["position_y"] - pose_values[1],
                ))
                distance = float(np.linalg.norm(relative))
                longitudinal = float(
                    relative[0] * math.cos(pose_values[2])
                    + relative[1] * math.sin(pose_values[2])
                )
                quality_rank = {
                    "VALID": 0,
                    "PROVISIONAL": 1,
                    "STALE": 2,
                    "INVALID": 3,
                }.get(item["forecast_qualification"], 4)
                return (quality_rank, 0 if longitudinal >= -0.45 else 1, distance)
            selected = min(selectable_people, key=rank)
        person_records = tuple(active_people + coasting_people)
        diagnostics.update({
            "tracks": tuple(tracks),
            "person_tracking_enabled": True,
            "person_track_count": len(self._states),
            "person_active_count": len(active_people),
            "person_coasting_count": len(coasting_people),
            "person_coasting_track_ids": tuple(
                item["person_id"] for item in coasting_people
            ),
            "person_tracks": person_records,
            "person_track_ids": tuple(
                item["person_id"] for item in person_records
            ),
            "person_selected_id": None if selected is None else selected["person_id"],
            "person_selected_member_track_indices": (
                () if selected is None else selected["member_track_indices"]
            ),
            "person_selected_position_x": (
                None if selected is None else float(selected["position_x"])
            ),
            "person_selected_position_y": (
                None if selected is None else float(selected["position_y"])
            ),
            "person_selected_velocity_x_mps": (
                None if selected is None else float(selected["velocity_x_mps"])
            ),
            "person_selected_velocity_y_mps": (
                None if selected is None else float(selected["velocity_y_mps"])
            ),
            "person_forecast_qualification": (
                "INVALID" if selected is None else selected["forecast_qualification"]
            ),
            "person_forecast_source": forecast_source,
            "person_forecast_candidate_track_indices": tuple(
                sorted(forecast_indices)
            ),
            "person_low_level_forecast_track_indices": tuple(
                sorted(
                    int(value)
                    for value in diagnostics.get(
                        "low_level_forecast_track_indices", ()
                    )
                )
            ),
            "person_forecast_qualification_states": tuple(
                (item["person_id"], item["forecast_qualification"])
                for item in person_records
            ),
            "person_forecast_valid": bool(
                selected is not None
                and selected["forecast_qualification"] == "VALID"
            ),
            "person_forecast_provisional": bool(
                selected is not None
                and selected["forecast_qualification"] == "PROVISIONAL"
            ),
            "person_forecast_stale": bool(
                selected is not None
                and selected["forecast_qualification"] == "STALE"
            ),
            "person_identity_continuity": bool(
                selected is not None
                and selected["association_residual_m"] is not None
            ),
            "person_identity_rejection_reason": (
                "coasting_only"
                if selected is None and coasting_people
                else "no_eligible_track"
                if selected is None
                else "none"
            ),
        })
        return diagnostics


__all__ = ["PersonTrackManager"]
