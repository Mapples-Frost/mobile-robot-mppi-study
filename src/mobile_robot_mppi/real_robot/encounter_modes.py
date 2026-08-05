"""Semantic recognition for structured human/robot encounters.

The manager owns *semantic* interaction state (crossing, frontal approach,
and goal-line rejoin) but deliberately has no command-writing API.  A separate
authority module can consume its diagnostics behind an explicit deployment
switch, while the default configuration remains read-only shadow mode.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


class EncounterMode(Enum):
    """Observable behaviour classes plus controller lifecycle states."""

    IDLE = "idle"
    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    RECEDING = "receding"
    STRAIGHT_CROSSING = "straight_crossing"
    OBLIQUE_CROSSING = "oblique_crossing"
    FRONTAL_APPROACH = "frontal_approach"
    REJOIN = "rejoin"


class PassageStrategy(Enum):
    """Locked passage intent proposed by the semantic manager."""

    NONE = "none"
    FRONT_PASS = "front_pass"
    BEHIND_PASS = "behind_pass"
    LEFT_BYPASS = "left_bypass"
    RIGHT_BYPASS = "right_bypass"


_ACTIVE_MODES = frozenset({
    EncounterMode.STRAIGHT_CROSSING,
    EncounterMode.OBLIQUE_CROSSING,
    EncounterMode.FRONTAL_APPROACH,
})


@dataclass(frozen=True)
class EncounterModeConfig:
    """Initial geometry thresholds; live data will calibrate these values."""

    enabled: bool = True
    shadow_only: bool = True
    straight_crossing_maximum_deg: float = 20.0
    oblique_crossing_maximum_deg: float = 55.0
    minimum_motion_speed_mps: float = 0.12
    minimum_approach_speed_mps: float = 0.08
    ordinary_confirmation_cycles: int = 3
    abrupt_confirmation_cycles: int = 2
    change_score_threshold: float = 0.55
    direction_change_reference_deg: float = 70.0
    track_jump_minimum_m: float = 0.75
    track_jump_speed_scale: float = 3.0
    maximum_track_gap_s: float = 0.50
    nominal_ego_speed_mps: float = 0.40
    cpa_horizon_s: float = 5.0
    forecast_behavior_lookahead_s: float = 0.60
    engagement_distance_m: float = 5.5
    engagement_cpa_distance_m: float = 1.35
    rear_engagement_allowance_m: float = 0.45
    crossing_line_tolerance_m: float = 0.12
    frontal_clearance_m: float = 0.65
    rejoin_heading_tolerance_deg: float = 12.0
    rejoin_cross_track_tolerance_m: float = 0.25
    rejoin_clear_cycles: int = 4
    lost_track_grace_cycles: int = 3
    front_pass_time_margin_s: float = 0.70
    robot_intersection_clearance_m: float = 0.75
    static_clearance_probe_m: float = 1.20
    static_clearance_half_width_m: float = 0.45
    crossing_detour_clearance_m: float = 0.80
    oblique_detour_clearance_m: float = 0.90
    frontal_detour_clearance_m: float = 0.90
    passage_forward_clearance_m: float = 0.65
    rejoin_lookahead_m: float = 1.10

    def __post_init__(self) -> None:
        if not (0.0 < self.straight_crossing_maximum_deg
                < self.oblique_crossing_maximum_deg < 90.0):
            raise ValueError("encounter angle thresholds must increase to 90 deg")
        if self.minimum_motion_speed_mps <= 0.0:
            raise ValueError("minimum motion speed must be positive")
        if self.ordinary_confirmation_cycles < 1:
            raise ValueError("ordinary confirmation cycles must be positive")
        if self.abrupt_confirmation_cycles < 1:
            raise ValueError("abrupt confirmation cycles must be positive")
        if self.forecast_behavior_lookahead_s <= 0.0:
            raise ValueError("forecast behaviour lookahead must be positive")
        if min(
            self.crossing_detour_clearance_m,
            self.oblique_detour_clearance_m,
            self.frontal_detour_clearance_m,
            self.passage_forward_clearance_m,
            self.rejoin_lookahead_m,
        ) <= 0.0:
            raise ValueError("encounter detour and rejoin clearances must be positive")


@dataclass
class _TrackSample:
    index: int
    timestamp_s: float
    position: np.ndarray
    velocity: np.ndarray
    semantic_velocity: np.ndarray
    velocity_source: str
    speed_mps: float
    innovation_nis: float
    change_triggered: bool
    support_beams: int
    forecast_valid: bool
    associated: bool
    forecast_positions: Optional[np.ndarray]
    forecast_times_s: Optional[np.ndarray]
    forecast_position_std_m: Optional[np.ndarray]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _unit(vector: Sequence[float]) -> Optional[np.ndarray]:
    value = np.asarray(vector, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1.0e-9:
        return None
    return value / norm


def _signed_angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(math.atan2(
        first[0] * second[1] - first[1] * second[0],
        float(np.dot(first, second)),
    ))


class EncounterModeManager:
    """Recognize and latch a single causally continuous human encounter.

    ``update`` only returns diagnostics.  It never writes chassis commands;
    shadow or active authority is selected by the deployment composition.
    """

    def __init__(self, config: EncounterModeConfig = EncounterModeConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._selected_track_index = None  # type: Optional[int]
        self._last_track = None  # type: Optional[_TrackSample]
        self._candidate = EncounterMode.UNKNOWN
        self._candidate_streak = 0
        self._candidate_fast_path = False
        self._confirmed_behavior = EncounterMode.UNKNOWN
        self._phase = EncounterMode.IDLE
        self._strategy = PassageStrategy.NONE
        self._entry_goal_origin = None  # type: Optional[np.ndarray]
        self._entry_goal_direction = None  # type: Optional[np.ndarray]
        self._entry_human_line_offset = None  # type: Optional[float]
        self._last_human_line_offset = None  # type: Optional[float]
        self._entry_human_position = None  # type: Optional[np.ndarray]
        self._entry_human_direction = None  # type: Optional[np.ndarray]
        self._entry_robot_human_line_offset = None  # type: Optional[float]
        self._locked_steering_side = 0
        self._lost_track_cycles = 0
        self._rejoin_clear_streak = 0
        self._strategy_timing = {
            "human_time_to_intersection_s": float("inf"),
            "robot_clear_time_s": float("inf"),
            "front_space_clear": None,
            "front_pass_feasible": False,
        }
        self._temporary_waypoint = None  # type: Optional[np.ndarray]

    def _track_records(
        self,
        timestamp_s: float,
        diagnostics: Mapping[str, Any],
        forecasts: Sequence[Any] = (),
    ) -> Tuple[_TrackSample, ...]:
        dynamic = {
            int(value)
            for value in diagnostics.get("mapless_dynamic_track_indices", ())
        }
        forecast = {
            int(value)
            for value in diagnostics.get("forecast_track_indices", ())
        }
        temporal = {
            int(value)
            for value in diagnostics.get(
                "temporal_flow_threat_track_indices", ()
            )
        }
        eligible = dynamic | forecast | temporal
        forecast_values = tuple(forecasts or ())
        forecast_indices = tuple(
            int(value)
            for value in diagnostics.get("forecast_track_indices", ())
        )
        forecast_by_track = {
            track_index: forecast_values[ordinal]
            for ordinal, track_index in enumerate(forecast_indices)
            if ordinal < len(forecast_values)
        }
        records = []
        for ordinal, raw in enumerate(diagnostics.get("tracks", ()) or ()):
            track = dict(raw or {})
            index = int(track.get("track_index", ordinal))
            associated = bool(track.get("associated", False))
            forecast_valid = bool(track.get("forecast_valid", False))
            motion_confirmed = bool(track.get("motion_confirmed", False))
            classification = str(
                track.get("mapless_classification", "unknown")
            )
            if index not in eligible and not (
                classification == "dynamic"
                or (associated and forecast_valid and motion_confirmed)
            ):
                continue
            position = np.asarray((
                _finite_float(track.get("measurement_x")),
                _finite_float(track.get("measurement_y")),
            ), dtype=np.float64)
            velocity = np.asarray((
                _finite_float(track.get("measurement_velocity_x_mps")),
                _finite_float(track.get("measurement_velocity_y_mps")),
            ), dtype=np.float64)
            if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                continue
            raw_forecast = forecast_by_track.get(index)
            # The single-target tracker predates forecast_track_indices.  Its
            # sole forecast is still unambiguous, so retain compatibility
            # without guessing when several tracks/forecasts are present.
            if (
                raw_forecast is None
                and len(forecast_values) == 1
                and len(tuple(diagnostics.get("tracks", ()) or ())) == 1
            ):
                raw_forecast = forecast_values[0]
            forecast_evidence = self._forecast_evidence(
                position, raw_forecast
            )
            semantic_velocity = (
                forecast_evidence[0]
                if forecast_evidence is not None
                else velocity
            )
            records.append(_TrackSample(
                index=index,
                timestamp_s=float(timestamp_s),
                position=position,
                velocity=velocity,
                semantic_velocity=semantic_velocity,
                velocity_source=(
                    "gaussian_mixture_forecast"
                    if forecast_evidence is not None
                    else "tracker_velocity"
                ),
                speed_mps=float(np.linalg.norm(semantic_velocity)),
                innovation_nis=_finite_float(track.get("innovation_nis"), 0.0),
                change_triggered=bool(track.get("change_triggered", False)),
                support_beams=int(track.get("selected_support_beams", 0) or 0),
                forecast_valid=forecast_valid,
                associated=associated,
                forecast_positions=(
                    None if forecast_evidence is None
                    else forecast_evidence[1]
                ),
                forecast_times_s=(
                    None if forecast_evidence is None
                    else forecast_evidence[2]
                ),
                forecast_position_std_m=(
                    None if forecast_evidence is None
                    else forecast_evidence[3]
                ),
            ))
        return tuple(records)

    def _forecast_evidence(
        self,
        position: np.ndarray,
        forecast: Any,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Reduce one Gaussian mixture to causal semantic path evidence.

        MPPI continues to consume the complete mixture.  The encounter
        classifier only needs its probability-weighted mean path and total
        positional uncertainty; malformed or absent payloads fall back to the
        tracker's fitted velocity without affecting control availability.
        """

        if forecast is None:
            return None
        try:
            means = np.asarray(
                forecast.component_means, dtype=np.float64
            )
            covariances = np.asarray(
                forecast.component_covariances, dtype=np.float64
            )
            weights = np.asarray(
                forecast.component_weights, dtype=np.float64
            )
            dt_s = float(forecast.dt)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            means.ndim != 3
            or means.shape[-1] != 2
            or covariances.shape != means.shape[:2] + (2, 2)
            or weights.shape != means.shape[:2]
            or means.shape[0] < 1
            or not math.isfinite(dt_s)
            or dt_s <= 0.0
            or not np.isfinite(means).all()
            or not np.isfinite(covariances).all()
            or not np.isfinite(weights).all()
        ):
            return None
        row_sums = np.sum(weights, axis=1)
        if np.any(weights < 0.0) or np.any(row_sums <= 1.0e-12):
            return None
        normalized = weights / row_sums[:, None]
        expected = np.sum(means * normalized[..., None], axis=1)
        deltas = means - expected[:, None, :]
        total_covariance = np.sum(
            normalized[..., None, None]
            * (
                covariances
                + deltas[..., :, None] * deltas[..., None, :]
            ),
            axis=1,
        )
        eigenvalues = np.linalg.eigvalsh(total_covariance)
        position_std = np.sqrt(np.maximum(eigenvalues[:, -1], 0.0))
        times = dt_s * np.arange(1, expected.shape[0] + 1, dtype=np.float64)
        lookahead_index = int(np.searchsorted(
            times, self.config.forecast_behavior_lookahead_s, side="left"
        ))
        lookahead_index = min(lookahead_index, len(times) - 1)
        semantic_velocity = (
            expected[lookahead_index] - position
        ) / times[lookahead_index]
        if not np.isfinite(semantic_velocity).all():
            return None
        return semantic_velocity, expected, times, position_std

    def _forecast_geometry(
        self,
        track: _TrackSample,
        robot_position: np.ndarray,
        goal_direction: np.ndarray,
        ego_speed_mps: float,
    ) -> Dict[str, float]:
        """Project forecast mean/covariance onto goal-line interaction facts."""

        positions = track.forecast_positions
        times = track.forecast_times_s
        std_values = track.forecast_position_std_m
        if positions is None or times is None or std_values is None:
            return {
                "forecast_available": False,
                "forecast_horizon_s": 0.0,
                "forecast_position_std_m": float("nan"),
                "forecast_goal_line_crossing": False,
                "forecast_crossing_time_s": float("inf"),
                "forecast_crossing_position_m": float("inf"),
                "forecast_cpa": False,
                "forecast_t_cpa_s": float("inf"),
                "forecast_d_cpa_mean_m": float("inf"),
                "forecast_d_cpa_conservative_m": float("inf"),
            }
        lateral_direction = np.asarray(
            (-goal_direction[1], goal_direction[0]), dtype=np.float64
        )
        path = np.vstack((track.position[None, :], positions))
        path_times = np.concatenate(([0.0], times))
        lateral = (path - robot_position[None, :]) @ lateral_direction
        crossing_time = float("inf")
        crossing_position = float("inf")
        for index in range(1, len(path)):
            before = float(lateral[index - 1])
            after = float(lateral[index])
            if (
                abs(after) > self.config.crossing_line_tolerance_m
                and before * after > 0.0
            ):
                continue
            denominator = abs(before) + abs(after)
            fraction = (
                1.0 if denominator <= 1.0e-12
                else float(np.clip(abs(before) / denominator, 0.0, 1.0))
            )
            point = path[index - 1] + fraction * (
                path[index] - path[index - 1]
            )
            crossing_time = float(
                path_times[index - 1]
                + fraction * (path_times[index] - path_times[index - 1])
            )
            crossing_position = float(np.dot(
                point - robot_position, goal_direction
            ))
            break

        valid = times <= self.config.cpa_horizon_s + 1.0e-12
        if np.any(valid):
            robot_path = (
                robot_position[None, :]
                + ego_speed_mps * times[valid, None] * goal_direction[None, :]
            )
            mean_distances = np.linalg.norm(
                positions[valid] - robot_path, axis=1
            )
            conservative = np.maximum(
                0.0, mean_distances - std_values[valid]
            )
            selected = int(np.argmin(conservative))
            valid_times = times[valid]
            valid_std = std_values[valid]
            t_cpa = float(valid_times[selected])
            d_cpa_mean = float(mean_distances[selected])
            d_cpa_conservative = float(conservative[selected])
            position_std = float(valid_std[selected])
        else:
            t_cpa = float("inf")
            d_cpa_mean = float("inf")
            d_cpa_conservative = float("inf")
            position_std = float(std_values[-1])
        return {
            "forecast_available": True,
            "forecast_horizon_s": float(times[-1]),
            "forecast_position_std_m": position_std,
            "forecast_goal_line_crossing": math.isfinite(crossing_time),
            "forecast_crossing_time_s": crossing_time,
            "forecast_crossing_position_m": crossing_position,
            "forecast_cpa": math.isfinite(t_cpa),
            "forecast_t_cpa_s": t_cpa,
            "forecast_d_cpa_mean_m": d_cpa_mean,
            "forecast_d_cpa_conservative_m": d_cpa_conservative,
        }

    def _kinematics(
        self,
        track: _TrackSample,
        robot_position: np.ndarray,
        goal_direction: np.ndarray,
        robot_speed_mps: float,
    ) -> Dict[str, float]:
        lateral_direction = np.asarray(
            (-goal_direction[1], goal_direction[0]), dtype=np.float64
        )
        relative_position = track.position - robot_position
        distance = float(np.linalg.norm(relative_position))
        longitudinal = float(np.dot(relative_position, goal_direction))
        lateral = float(np.dot(relative_position, lateral_direction))
        behavior_velocity = track.semantic_velocity
        human_longitudinal = float(np.dot(
            behavior_velocity, goal_direction
        ))
        human_lateral = float(np.dot(
            behavior_velocity, lateral_direction
        ))
        radial_velocity = (
            0.0 if distance <= 1.0e-9
            else float(np.dot(
                behavior_velocity, relative_position / distance
            ))
        )
        ego_speed = max(float(robot_speed_mps), self.config.nominal_ego_speed_mps)
        relative_velocity = behavior_velocity - ego_speed * goal_direction
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        if relative_speed_sq <= 1.0e-9:
            t_cpa = float("inf")
            d_cpa = distance
        else:
            raw_t_cpa = -float(np.dot(relative_position, relative_velocity)) / (
                relative_speed_sq
            )
            if raw_t_cpa < 0.0 or raw_t_cpa > self.config.cpa_horizon_s:
                t_cpa = float("inf")
                d_cpa = distance
            else:
                t_cpa = raw_t_cpa
                d_cpa = float(np.linalg.norm(
                    relative_position + t_cpa * relative_velocity
                ))
        forecast_geometry = self._forecast_geometry(
            track, robot_position, goal_direction, ego_speed
        )
        if forecast_geometry["forecast_cpa"]:
            t_cpa = float(forecast_geometry["forecast_t_cpa_s"])
            d_cpa = float(
                forecast_geometry["forecast_d_cpa_conservative_m"]
            )
        if forecast_geometry["forecast_goal_line_crossing"]:
            crossing_time = float(
                forecast_geometry["forecast_crossing_time_s"]
            )
            crossing_position = float(
                forecast_geometry["forecast_crossing_position_m"]
            )
            forecast_crossing_used = True
        elif abs(human_lateral) > 1.0e-9:
            crossing_time = -lateral / human_lateral
            crossing_position = (
                longitudinal + human_longitudinal * crossing_time
                if crossing_time > 0.0 else float("inf")
            )
            forecast_crossing_used = False
        else:
            crossing_time = float("inf")
            crossing_position = float("inf")
            forecast_crossing_used = False
        approach_angle = math.degrees(math.atan2(
            max(0.0, -human_longitudinal),
            max(1.0e-9, abs(human_lateral)),
        ))
        return {
            "distance_m": distance,
            "longitudinal_m": longitudinal,
            "lateral_m": lateral,
            "human_longitudinal_mps": human_longitudinal,
            "human_lateral_mps": human_lateral,
            "radial_velocity_mps": radial_velocity,
            "approach_angle_deg": approach_angle,
            "t_cpa_s": t_cpa,
            "d_cpa_m": d_cpa,
            "goal_line_crossing_time_s": crossing_time,
            "goal_line_crossing_position_m": crossing_position,
            "forecast_goal_line_crossing_used": forecast_crossing_used,
            **forecast_geometry,
        }

    def _select_track(
        self,
        tracks: Iterable[_TrackSample],
        robot_position: np.ndarray,
        goal_direction: np.ndarray,
        robot_speed_mps: float,
    ) -> Tuple[Optional[_TrackSample], Dict[str, float]]:
        ranked = []
        for track in tracks:
            values = self._kinematics(
                track, robot_position, goal_direction, robot_speed_mps
            )
            ahead = values["longitudinal_m"] >= -self.config.rear_engagement_allowance_m
            collision_course = bool(
                math.isfinite(values["t_cpa_s"])
                and values["d_cpa_m"] <= self.config.engagement_cpa_distance_m
            )
            in_range = values["distance_m"] <= self.config.engagement_distance_m
            rank = (
                0 if collision_course and ahead and in_range else 1,
                0 if ahead else 1,
                values["t_cpa_s"] if math.isfinite(values["t_cpa_s"]) else 99.0,
                values["d_cpa_m"],
                values["distance_m"],
                0 if track.index == self._selected_track_index else 1,
            )
            ranked.append((rank, track, values))
        if not ranked:
            return None, {}
        _, track, values = min(ranked, key=lambda item: item[0])
        return track, values

    def _classify(
        self,
        track: Optional[_TrackSample],
        values: Mapping[str, float],
    ) -> Tuple[EncounterMode, float, Dict[str, float]]:
        if track is None:
            return EncounterMode.UNKNOWN, 0.0, {
                mode.value: 0.0 for mode in (
                    EncounterMode.STRAIGHT_CROSSING,
                    EncounterMode.OBLIQUE_CROSSING,
                    EncounterMode.FRONTAL_APPROACH,
                    EncounterMode.RECEDING,
                    EncounterMode.STATIONARY,
                    EncounterMode.UNKNOWN,
                )
            }
        speed = float(track.speed_mps)
        radial = float(values["radial_velocity_mps"])
        angle = float(values["approach_angle_deg"])
        if speed < self.config.minimum_motion_speed_mps:
            mode = EncounterMode.STATIONARY
            confidence = float(np.clip(
                1.0 - speed / self.config.minimum_motion_speed_mps, 0.0, 1.0
            ))
        elif radial >= self.config.minimum_approach_speed_mps:
            mode = EncounterMode.RECEDING
            confidence = float(np.clip(
                radial / max(speed, 1.0e-9), 0.0, 1.0
            ))
        elif angle <= self.config.straight_crossing_maximum_deg:
            mode = EncounterMode.STRAIGHT_CROSSING
            confidence = 1.0 - 0.5 * angle / (
                self.config.straight_crossing_maximum_deg
            )
        elif angle < self.config.oblique_crossing_maximum_deg:
            mode = EncounterMode.OBLIQUE_CROSSING
            midpoint = 0.5 * (
                self.config.straight_crossing_maximum_deg
                + self.config.oblique_crossing_maximum_deg
            )
            half_width = 0.5 * (
                self.config.oblique_crossing_maximum_deg
                - self.config.straight_crossing_maximum_deg
            )
            confidence = 1.0 - 0.35 * abs(angle - midpoint) / half_width
        else:
            mode = EncounterMode.FRONTAL_APPROACH
            confidence = 0.65 + 0.35 * (
                angle - self.config.oblique_crossing_maximum_deg
            ) / (90.0 - self.config.oblique_crossing_maximum_deg)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        # Probabilities are deliberately interpretable soft memberships, not a
        # trained network.  They make threshold behaviour auditable in logs.
        widths = {
            EncounterMode.STRAIGHT_CROSSING: 18.0,
            EncounterMode.OBLIQUE_CROSSING: 19.0,
            EncounterMode.FRONTAL_APPROACH: 24.0,
        }
        centres = {
            EncounterMode.STRAIGHT_CROSSING: 0.0,
            EncounterMode.OBLIQUE_CROSSING: 0.5 * (
                self.config.straight_crossing_maximum_deg
                + self.config.oblique_crossing_maximum_deg
            ),
            EncounterMode.FRONTAL_APPROACH: 90.0,
        }
        scores = {
            candidate.value: math.exp(-0.5 * (
                (angle - centres[candidate]) / widths[candidate]
            ) ** 2)
            for candidate in centres
        }
        scores[EncounterMode.RECEDING.value] = max(0.0, radial) / max(
            speed, 1.0e-9
        )
        scores[EncounterMode.STATIONARY.value] = max(
            0.0, 1.0 - speed / self.config.minimum_motion_speed_mps
        )
        scores[EncounterMode.UNKNOWN.value] = 0.05
        total = max(sum(scores.values()), 1.0e-12)
        probabilities = {
            key: float(value / total) for key, value in scores.items()
        }
        return mode, confidence, probabilities

    def _continuity_and_change(
        self,
        track: Optional[_TrackSample],
        candidate: EncounterMode,
    ) -> Tuple[bool, bool, float, str, float]:
        previous = self._last_track
        if track is None or previous is None:
            return False, False, 0.0, "no_track_history", 0.0
        if track.index != previous.index:
            return False, False, 0.0, "track_index_changed", 0.0
        dt_s = track.timestamp_s - previous.timestamp_s
        if dt_s <= 0.0 or dt_s > self.config.maximum_track_gap_s:
            return False, False, 0.0, "track_time_gap", 0.0
        predicted = previous.position + previous.velocity * dt_s
        residual = float(np.linalg.norm(track.position - predicted))
        jump_limit = max(
            self.config.track_jump_minimum_m,
            self.config.track_jump_speed_scale
            * max(previous.speed_mps, track.speed_mps)
            * dt_s,
        )
        if residual > jump_limit:
            return False, False, 0.0, "position_jump", residual
        previous_direction = _unit(previous.semantic_velocity)
        current_direction = _unit(track.semantic_velocity)
        direction_change_deg = 0.0
        if previous_direction is not None and current_direction is not None:
            direction_change_deg = abs(math.degrees(
                _signed_angle(previous_direction, current_direction)
            ))
        direction_score = min(
            0.40,
            0.40 * direction_change_deg
            / self.config.direction_change_reference_deg,
        )
        nis_score = min(0.25, 0.25 * max(0.0, track.innovation_nis - 4.0) / 8.0)
        ca_score = 0.35 if track.change_triggered else 0.0
        class_score = 0.15 if (
            self._confirmed_behavior not in (
                EncounterMode.UNKNOWN, EncounterMode.STATIONARY,
                EncounterMode.RECEDING,
            )
            and candidate != self._confirmed_behavior
        ) else 0.0
        score = min(1.0, direction_score + nis_score + ca_score + class_score)
        detected = bool(score >= self.config.change_score_threshold)
        reasons = []
        if track.change_triggered:
            reasons.append("ca_imm")
        if nis_score > 0.0:
            reasons.append("innovation")
        if direction_score > 0.0:
            reasons.append(
                "forecast_direction"
                if track.velocity_source == "gaussian_mixture_forecast"
                else "velocity_direction"
            )
        if class_score > 0.0:
            reasons.append("mode_distribution")
        return True, detected, score, "+".join(reasons) or "continuous", residual

    def _front_space_clear(
        self,
        track: _TrackSample,
        local_obstacles: Sequence[Sequence[float]],
    ) -> bool:
        direction = _unit(track.semantic_velocity)
        if direction is None:
            return False
        lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        for raw in local_obstacles or ():
            try:
                point = np.asarray(raw[:2], dtype=np.float64)
            except (TypeError, ValueError, IndexError):
                continue
            if not np.isfinite(point).all():
                continue
            relative = point - track.position
            forward = float(np.dot(relative, direction))
            side = abs(float(np.dot(relative, lateral)))
            if (
                0.35 < forward <= self.config.static_clearance_probe_m
                and side <= self.config.static_clearance_half_width_m
            ):
                return False
        return True

    def _choose_strategy(
        self,
        mode: EncounterMode,
        track: _TrackSample,
        values: Mapping[str, float],
        robot_position: np.ndarray,
        goal_direction: np.ndarray,
        robot_speed_mps: float,
        local_obstacles: Sequence[Sequence[float]],
    ) -> Tuple[PassageStrategy, int, Dict[str, Any]]:
        human_lateral = float(values["human_lateral_mps"])
        if mode in (
            EncounterMode.STRAIGHT_CROSSING,
            EncounterMode.OBLIQUE_CROSSING,
        ):
            t_human = float(values["goal_line_crossing_time_s"])
            if t_human <= 0.0:
                t_human = float("inf")
            intersection_longitudinal = float(
                values["goal_line_crossing_position_m"]
            )
            ego_speed = max(float(robot_speed_mps), self.config.nominal_ego_speed_mps)
            t_robot_clear = (
                max(0.0, intersection_longitudinal)
                + self.config.robot_intersection_clearance_m
            ) / ego_speed
            front_space_clear = self._front_space_clear(track, local_obstacles)
            front_pass_feasible = bool(
                math.isfinite(t_human)
                and front_space_clear
                and t_robot_clear + self.config.front_pass_time_margin_s < t_human
            )
            if front_pass_feasible:
                strategy = PassageStrategy.FRONT_PASS
                side = 1 if human_lateral > 0.0 else -1
            else:
                strategy = PassageStrategy.BEHIND_PASS
                side = -1 if human_lateral > 0.0 else 1
            return strategy, side, {
                "human_time_to_intersection_s": t_human,
                "robot_clear_time_s": t_robot_clear,
                "front_space_clear": front_space_clear,
                "front_pass_feasible": front_pass_feasible,
            }
        # For a frontal approach, compare the complete robot-to-bypass
        # segments against measured local geometry.  The tracked human is
        # excluded from this *static-space* comparison because its risk is
        # already represented by the encounter itself.
        lateral_direction = np.asarray(
            (-goal_direction[1], goal_direction[0]), dtype=np.float64
        )
        bypass_forward = track.position + (
            self.config.passage_forward_clearance_m * goal_direction
        )

        def segment_clearance(side_value: int) -> float:
            endpoint = (
                bypass_forward
                + float(side_value)
                * self.config.frontal_detour_clearance_m
                * lateral_direction
            )
            segment = endpoint - robot_position
            length_sq = float(np.dot(segment, segment))
            clearance = float("inf")
            for raw in local_obstacles or ():
                try:
                    point = np.asarray(raw[:2], dtype=np.float64)
                except (TypeError, ValueError, IndexError):
                    continue
                if not np.isfinite(point).all():
                    continue
                if float(np.linalg.norm(point - track.position)) <= 0.60:
                    continue
                fraction = (
                    0.0 if length_sq <= 1.0e-9
                    else float(np.clip(
                        np.dot(point - robot_position, segment) / length_sq,
                        0.0,
                        1.0,
                    ))
                )
                distance = float(np.linalg.norm(
                    point - (robot_position + fraction * segment)
                ))
                clearance = min(clearance, distance)
            return clearance

        left_clearance = segment_clearance(1)
        right_clearance = segment_clearance(-1)
        human_side = float(values["lateral_m"])
        if abs(left_clearance - right_clearance) >= 0.10:
            side = 1 if left_clearance > right_clearance else -1
        else:
            # Equal space falls away from an off-centre human; a centred
            # encounter uses deterministic left and remains locked.
            side = -1 if human_side > 0.10 else 1
        return (
            PassageStrategy.LEFT_BYPASS if side > 0 else PassageStrategy.RIGHT_BYPASS,
            side,
            {
                "human_time_to_intersection_s": float("inf"),
                "robot_clear_time_s": float("inf"),
                "front_space_clear": None,
                "front_pass_feasible": False,
                "left_bypass_clearance_m": left_clearance,
                "right_bypass_clearance_m": right_clearance,
            },
        )

    def _enter_active(
        self,
        mode: EncounterMode,
        track: _TrackSample,
        values: Mapping[str, float],
        robot_position: np.ndarray,
        goal_direction: np.ndarray,
        robot_speed_mps: float,
        local_obstacles: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        self._phase = mode
        self._entry_goal_origin = robot_position.copy()
        self._entry_goal_direction = goal_direction.copy()
        lateral_direction = np.asarray(
            (-goal_direction[1], goal_direction[0]), dtype=np.float64
        )
        offset = float(np.dot(
            track.position - self._entry_goal_origin, lateral_direction
        ))
        self._entry_human_line_offset = offset
        self._last_human_line_offset = offset
        self._entry_human_position = track.position.copy()
        self._entry_human_direction = _unit(track.semantic_velocity)
        if self._entry_human_direction is not None:
            self._entry_robot_human_line_offset = float(
                self._entry_human_direction[0]
                * (robot_position[1] - track.position[1])
                - self._entry_human_direction[1]
                * (robot_position[0] - track.position[0])
            )
        else:
            self._entry_robot_human_line_offset = None
        strategy, side, timing = self._choose_strategy(
            mode,
            track,
            values,
            robot_position,
            goal_direction,
            robot_speed_mps,
            local_obstacles,
        )
        self._strategy = strategy
        self._locked_steering_side = int(side)
        self._strategy_timing = dict(timing)
        if mode == EncounterMode.FRONTAL_APPROACH:
            forward = max(
                0.60,
                float(values.get("longitudinal_m", 0.0))
                + self.config.passage_forward_clearance_m,
            )
            lateral_clearance = self.config.frontal_detour_clearance_m
        else:
            crossing_time = float(timing["human_time_to_intersection_s"])
            intersection_longitudinal = (
                float(values.get("longitudinal_m", 0.0))
                + float(values.get("human_longitudinal_mps", 0.0))
                * crossing_time
                if math.isfinite(crossing_time)
                else float(values.get("longitudinal_m", 0.0))
            )
            forward = max(
                0.60,
                intersection_longitudinal
                + self.config.passage_forward_clearance_m,
            )
            lateral_clearance = (
                self.config.crossing_detour_clearance_m
                if mode == EncounterMode.STRAIGHT_CROSSING
                else self.config.oblique_detour_clearance_m
            )
        lateral_direction = np.asarray((
            -goal_direction[1], goal_direction[0]
        ), dtype=np.float64)
        waypoint = (
            self._entry_goal_origin
            + forward * goal_direction
            + float(side) * lateral_clearance * lateral_direction
        )
        self._temporary_waypoint = waypoint
        self._rejoin_clear_streak = 0
        return timing

    def _set_rejoin_waypoint(self, robot_position: np.ndarray) -> None:
        if self._entry_goal_origin is None or self._entry_goal_direction is None:
            self._temporary_waypoint = None
            return
        progress = float(np.dot(
            robot_position - self._entry_goal_origin,
            self._entry_goal_direction,
        ))
        self._temporary_waypoint = (
            self._entry_goal_origin
            + (progress + self.config.rejoin_lookahead_m)
            * self._entry_goal_direction
        )

    def _crossing_complete(self, track: _TrackSample) -> bool:
        if self._entry_goal_origin is None or self._entry_goal_direction is None:
            return False
        lateral_direction = np.asarray((
            -self._entry_goal_direction[1], self._entry_goal_direction[0]
        ))
        current = float(np.dot(
            track.position - self._entry_goal_origin, lateral_direction
        ))
        previous = self._last_human_line_offset
        self._last_human_line_offset = current
        if previous is None:
            return False
        tolerance = self.config.crossing_line_tolerance_m
        had_separation = abs(previous) > tolerance
        reached_line = abs(current) <= tolerance
        crossed_sign = previous * current < 0.0
        return bool(had_separation and (reached_line or crossed_sign))

    def _frontal_complete(
        self, track: _TrackSample, robot_position: np.ndarray
    ) -> bool:
        if self._entry_human_position is None or self._entry_goal_direction is None:
            return False
        lateral_direction = np.asarray((
            -self._entry_goal_direction[1], self._entry_goal_direction[0]
        ))
        robot_lateral = float(np.dot(
            robot_position - self._entry_human_position, lateral_direction
        ))
        human_ahead = float(np.dot(
            track.position - robot_position, self._entry_goal_direction
        ))
        correct_side = robot_lateral * self._locked_steering_side > 0.0
        return bool(
            correct_side
            and abs(robot_lateral) >= self.config.frontal_clearance_m
            and human_ahead <= self.config.robot_intersection_clearance_m
        )

    def _update_rejoin(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        dangerous: bool,
    ) -> Tuple[float, float]:
        if self._entry_goal_origin is None or self._entry_goal_direction is None:
            self._phase = EncounterMode.IDLE
            return 0.0, 0.0
        lateral_direction = np.asarray((
            -self._entry_goal_direction[1], self._entry_goal_direction[0]
        ))
        cross_track = float(np.dot(
            robot_position - self._entry_goal_origin, lateral_direction
        ))
        goal_heading = math.atan2(
            self._entry_goal_direction[1], self._entry_goal_direction[0]
        )
        heading_error = math.atan2(
            math.sin(goal_heading - robot_yaw), math.cos(goal_heading - robot_yaw)
        )
        clear = bool(
            abs(cross_track) <= self.config.rejoin_cross_track_tolerance_m
            and abs(math.degrees(heading_error))
            <= self.config.rejoin_heading_tolerance_deg
            and not dangerous
        )
        self._rejoin_clear_streak = (
            self._rejoin_clear_streak + 1 if clear else 0
        )
        if self._rejoin_clear_streak >= self.config.rejoin_clear_cycles:
            self._phase = EncounterMode.IDLE
            self._strategy = PassageStrategy.NONE
            self._locked_steering_side = 0
            self._entry_goal_origin = None
            self._entry_goal_direction = None
            self._temporary_waypoint = None
        return cross_track, heading_error

    def update(
        self,
        *,
        timestamp_s: float,
        pose: Sequence[float],
        goal: Sequence[float],
        robot_speed_mps: float,
        tracker_diagnostics: Mapping[str, Any],
        forecasts: Sequence[Any] = (),
        local_obstacles: Sequence[Sequence[float]] = (),
    ) -> Dict[str, Any]:
        """Advance the semantic state machine and return JSON-safe facts."""

        if not self.config.enabled:
            return {
                "encounter_enabled": False,
                "encounter_control_enabled": False,
                "encounter_phase": EncounterMode.IDLE.value,
            }
        pose_values = np.asarray(pose, dtype=np.float64).reshape(3)
        goal_values = np.asarray(goal, dtype=np.float64).reshape(2)
        if not np.isfinite(pose_values).all() or not np.isfinite(goal_values).all():
            raise ValueError("encounter pose and goal must be finite")
        robot_position = pose_values[:2]
        goal_direction = _unit(goal_values - robot_position)
        if goal_direction is None:
            goal_direction = np.asarray((math.cos(pose_values[2]), math.sin(pose_values[2])))
        tracks = self._track_records(
            float(timestamp_s), tracker_diagnostics, forecasts
        )
        selected, values = self._select_track(
            tracks, robot_position, goal_direction, float(robot_speed_mps)
        )
        candidate, confidence, probabilities = self._classify(selected, values)
        continuous, changed, change_score, change_reason, residual = (
            self._continuity_and_change(selected, candidate)
        )
        selected_index = None if selected is None else selected.index
        track_switched = bool(
            self._selected_track_index is not None
            and selected_index is not None
            and selected_index != self._selected_track_index
        )
        if candidate == self._candidate and (
            continuous or self._candidate_streak == 0
        ):
            self._candidate_streak += 1
            self._candidate_fast_path = bool(
                self._candidate_fast_path or changed
            )
        else:
            self._candidate = candidate
            self._candidate_streak = 1
            self._candidate_fast_path = bool(changed)
        required = (
            self.config.abrupt_confirmation_cycles
            if self._candidate_fast_path
            else self.config.ordinary_confirmation_cycles
        )
        confirmed_this_cycle = bool(self._candidate_streak >= required)
        previous_confirmed = self._confirmed_behavior
        if confirmed_this_cycle:
            self._confirmed_behavior = candidate

        dangerous = bool(
            selected is not None
            and values.get("distance_m", float("inf"))
            <= self.config.engagement_distance_m
            and (
                values.get("d_cpa_m", float("inf"))
                <= self.config.engagement_cpa_distance_m
                or values.get("distance_m", float("inf"))
                <= self.config.engagement_cpa_distance_m
            )
            and values.get("longitudinal_m", -float("inf"))
            >= -self.config.rear_engagement_allowance_m
        )
        crossing_time = float(values.get(
            "goal_line_crossing_time_s", float("inf")
        ))
        crossing_position = float(values.get(
            "goal_line_crossing_position_m", float("inf")
        ))
        crossing_relevant = bool(
            candidate in (
                EncounterMode.STRAIGHT_CROSSING,
                EncounterMode.OBLIQUE_CROSSING,
            )
            and 0.0 < crossing_time <= self.config.cpa_horizon_s
            and -self.config.rear_engagement_allowance_m <= crossing_position
            <= self.config.engagement_distance_m
        )
        frontal_relevant = bool(
            candidate == EncounterMode.FRONTAL_APPROACH
            and values.get("longitudinal_m", -float("inf"))
            >= -self.config.rear_engagement_allowance_m
            and values.get("distance_m", float("inf"))
            <= self.config.engagement_distance_m
            and -values.get("human_longitudinal_mps", 0.0)
            >= self.config.minimum_approach_speed_mps
        )
        interaction_relevant = bool(
            selected is not None
            and (dangerous or crossing_relevant or frontal_relevant)
        )
        timing = dict(self._strategy_timing)
        line_crossed = False
        phase_before = self._phase
        verified_transition = bool(
            self._confirmed_behavior in _ACTIVE_MODES
            and selected is not None
            and interaction_relevant
        )
        if verified_transition and (
            self._phase == EncounterMode.IDLE
            or (
                self._confirmed_behavior != previous_confirmed
                and self._candidate_fast_path
                and continuous
                and self._phase != EncounterMode.REJOIN
            )
        ):
            timing = self._enter_active(
                self._confirmed_behavior,
                selected,
                values,
                robot_position,
                goal_direction,
                float(robot_speed_mps),
                local_obstacles,
            )
        if selected is None:
            self._lost_track_cycles += 1
        else:
            self._lost_track_cycles = 0
        if self._phase in (
            EncounterMode.STRAIGHT_CROSSING,
            EncounterMode.OBLIQUE_CROSSING,
        ) and selected is not None:
            line_crossed = self._crossing_complete(selected)
        elif self._phase == EncounterMode.FRONTAL_APPROACH and selected is not None:
            line_crossed = self._frontal_complete(selected, robot_position)
        if self._phase in _ACTIVE_MODES and (
            line_crossed
            or (
                self._confirmed_behavior == EncounterMode.RECEDING
                and not dangerous
            )
            or self._lost_track_cycles > self.config.lost_track_grace_cycles
        ):
            self._phase = EncounterMode.REJOIN
            self._rejoin_clear_streak = 0
            self._set_rejoin_waypoint(robot_position)
        cross_track = 0.0
        heading_error = 0.0
        if self._phase == EncounterMode.REJOIN:
            cross_track, heading_error = self._update_rejoin(
                robot_position, float(pose_values[2]), dangerous
            )

        active_or_rejoin = bool(
            self._phase in _ACTIVE_MODES or self._phase == EncounterMode.REJOIN
        )
        diagnostics = {
            "encounter_enabled": True,
            "encounter_shadow_only": bool(self.config.shadow_only),
            "encounter_control_enabled": bool(not self.config.shadow_only),
            "encounter_candidate_mode": candidate.value,
            "encounter_confirmed_mode": self._confirmed_behavior.value,
            "encounter_phase": self._phase.value,
            "encounter_phase_before": phase_before.value,
            "encounter_phase_changed": self._phase != phase_before,
            "encounter_candidate_streak": int(self._candidate_streak),
            "encounter_candidate_fast_path": bool(self._candidate_fast_path),
            "encounter_confirmation_cycles_required": int(required),
            "encounter_confidence": float(confidence),
            "encounter_mode_probabilities": probabilities,
            "encounter_track_index": selected_index,
            "encounter_track_continuous": bool(continuous),
            "encounter_track_switched": track_switched,
            "encounter_track_residual_m": float(residual),
            "encounter_change_detected": bool(changed),
            "encounter_change_score": float(change_score),
            "encounter_change_reason": str(change_reason),
            "encounter_strategy": self._strategy.value,
            "encounter_locked_steering_side": int(self._locked_steering_side),
            "encounter_line_crossed": bool(line_crossed),
            "encounter_dangerous": bool(dangerous),
            "encounter_interaction_relevant": interaction_relevant,
            "encounter_predicted_goal_line_crossing_time_s": float(
                crossing_time
            ),
            "encounter_predicted_goal_line_crossing_position_m": float(
                crossing_position
            ),
            "encounter_rear_pass_inhibited_shadow": active_or_rejoin,
            "encounter_forward_passage_inhibited_shadow": active_or_rejoin,
            "encounter_dynamic_escape_direction_owned_shadow": active_or_rejoin,
            "encounter_rear_pass_inhibited": bool(
                active_or_rejoin and not self.config.shadow_only
            ),
            "encounter_forward_passage_inhibited": bool(
                active_or_rejoin and not self.config.shadow_only
            ),
            "encounter_dynamic_escape_inhibited": bool(
                active_or_rejoin and not self.config.shadow_only
            ),
            "encounter_rejoin_cross_track_m": float(cross_track),
            "encounter_rejoin_heading_error_rad": float(heading_error),
            "encounter_rejoin_clear_streak": int(self._rejoin_clear_streak),
            "encounter_lost_track_cycles": int(self._lost_track_cycles),
            "encounter_approach_angle_deg": _finite_float(
                values.get("approach_angle_deg")
            ),
            "encounter_human_longitudinal_velocity_mps": _finite_float(
                values.get("human_longitudinal_mps")
            ),
            "encounter_human_lateral_velocity_mps": _finite_float(
                values.get("human_lateral_mps")
            ),
            "encounter_human_longitudinal_position_m": _finite_float(
                values.get("longitudinal_m")
            ),
            "encounter_human_lateral_position_m": _finite_float(
                values.get("lateral_m")
            ),
            "encounter_t_cpa_s": _finite_float(
                values.get("t_cpa_s"), float("inf")
            ),
            "encounter_d_cpa_m": _finite_float(
                values.get("d_cpa_m"), float("inf")
            ),
            "encounter_distance_m": _finite_float(
                values.get("distance_m"), float("inf")
            ),
            "encounter_velocity_source": (
                "none" if selected is None else selected.velocity_source
            ),
            "encounter_forecast_available": bool(
                values.get("forecast_available", False)
            ),
            "encounter_forecast_horizon_s": _finite_float(
                values.get("forecast_horizon_s"), 0.0
            ),
            "encounter_forecast_position_std_m": _finite_float(
                values.get("forecast_position_std_m")
            ),
            "encounter_forecast_goal_line_crossing_used": bool(
                values.get("forecast_goal_line_crossing_used", False)
            ),
            "encounter_forecast_cpa_used": bool(
                values.get("forecast_cpa", False)
            ),
            "encounter_forecast_d_cpa_mean_m": _finite_float(
                values.get("forecast_d_cpa_mean_m"), float("inf")
            ),
            "encounter_forecast_d_cpa_conservative_m": _finite_float(
                values.get("forecast_d_cpa_conservative_m"), float("inf")
            ),
            "encounter_entry_goal_origin": (
                None
                if self._entry_goal_origin is None
                else tuple(float(value) for value in self._entry_goal_origin)
            ),
            "encounter_entry_goal_heading_rad": (
                None
                if self._entry_goal_direction is None
                else float(math.atan2(
                    self._entry_goal_direction[1],
                    self._entry_goal_direction[0],
                ))
            ),
            "encounter_temporary_waypoint": (
                None
                if self._temporary_waypoint is None
                else tuple(float(value) for value in self._temporary_waypoint)
            ),
            "encounter_human_time_to_intersection_s": timing[
                "human_time_to_intersection_s"
            ],
            "encounter_robot_clear_time_s": timing["robot_clear_time_s"],
            "encounter_front_space_clear": timing["front_space_clear"],
            "encounter_front_pass_feasible": timing["front_pass_feasible"],
            "encounter_left_bypass_clearance_m": (
                float(timing["left_bypass_clearance_m"])
                if math.isfinite(float(timing.get(
                    "left_bypass_clearance_m", float("inf")
                ))) else None
            ),
            "encounter_right_bypass_clearance_m": (
                float(timing["right_bypass_clearance_m"])
                if math.isfinite(float(timing.get(
                    "right_bypass_clearance_m", float("inf")
                ))) else None
            ),
        }
        self._selected_track_index = selected_index
        self._last_track = selected
        return diagnostics


__all__ = [
    "EncounterMode",
    "EncounterModeConfig",
    "EncounterModeManager",
    "PassageStrategy",
]
