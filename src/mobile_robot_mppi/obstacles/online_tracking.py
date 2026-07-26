"""Causal single-obstacle LaserScan association for online IMM forecasting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import yaml

from mobile_robot_mppi.core.types import RobotObservation

from .collision_risk import GaussianMixtureObstacleForecast
from .imm import ChangeAwareIMMPredictor, OrdinaryIMMPredictor


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("online tracker config source must be a mapping")
    return value


@dataclass(frozen=True)
class OnlineTrackerConfig:
    obstacle_radius_m: float = 0.25
    sensor_forward_offset_m: float = 0.10
    minimum_cluster_beams: int = 3
    minimum_initial_cluster_beams: int = 3
    maximum_cluster_point_gap_m: float = 0.20
    association_gate_m: float = 0.90
    maximum_unobserved_duration_s: float = 1.50
    forecast_horizon_steps: int = 36
    forecast_dt_s: float = 0.10
    forecast_auxiliary_key: str = "probabilistic_obstacle_forecasts"
    motion_confirmation_enabled: bool = False
    motion_confirmation_required_observations: int = 4
    motion_confirmation_minimum_speed_mps: float = 0.12

    def __post_init__(self):
        finite = (
            float(self.obstacle_radius_m),
            float(self.sensor_forward_offset_m),
            float(self.maximum_cluster_point_gap_m),
            float(self.association_gate_m),
            float(self.maximum_unobserved_duration_s),
            float(self.forecast_dt_s),
            float(self.motion_confirmation_minimum_speed_mps),
        )
        if not np.isfinite(finite).all():
            raise ValueError("online tracker configuration must be finite")
        if (
            finite[0] <= 0.0
            or finite[1] < 0.0
            or finite[2] <= 0.0
            or finite[3] <= 0.0
            or finite[4] <= 0.0
            or finite[5] <= 0.0
            or finite[6] < 0.0
        ):
            raise ValueError("online tracker metric values are invalid")
        if (
            int(self.minimum_cluster_beams) <= 0
            or int(self.minimum_initial_cluster_beams) <= 0
        ):
            raise ValueError("minimum cluster beams must be positive")
        if (
            int(self.minimum_initial_cluster_beams)
            < int(self.minimum_cluster_beams)
        ):
            raise ValueError(
                "initial cluster support cannot be weaker than tracked "
                "cluster support"
            )
        if int(self.forecast_horizon_steps) <= 0:
            raise ValueError("forecast horizon must be positive")
        if int(self.motion_confirmation_required_observations) < 2:
            raise ValueError(
                "motion confirmation requires at least two observations"
            )
        if not str(self.forecast_auxiliary_key):
            raise ValueError("forecast auxiliary key must be nonempty")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]):
        return cls(
            obstacle_radius_m=float(values.get("obstacle_radius_m", 0.25)),
            sensor_forward_offset_m=float(
                values.get("sensor_forward_offset_m", 0.10)
            ),
            minimum_cluster_beams=int(
                values.get("minimum_cluster_beams", 3)
            ),
            minimum_initial_cluster_beams=int(
                values.get(
                    "minimum_initial_cluster_beams",
                    values.get("minimum_cluster_beams", 3),
                )
            ),
            maximum_cluster_point_gap_m=float(
                values.get("maximum_cluster_point_gap_m", 0.20)
            ),
            association_gate_m=float(
                values.get("association_gate_m", 0.90)
            ),
            maximum_unobserved_duration_s=float(
                values.get("maximum_unobserved_duration_s", 1.50)
            ),
            forecast_horizon_steps=int(
                values.get("forecast_horizon_steps", 36)
            ),
            forecast_dt_s=float(values.get("forecast_dt_s", 0.10)),
            forecast_auxiliary_key=str(
                values.get(
                    "forecast_auxiliary_key",
                    "probabilistic_obstacle_forecasts",
                )
            ),
            motion_confirmation_enabled=bool(
                values.get("motion_confirmation_enabled", False)
            ),
            motion_confirmation_required_observations=int(
                values.get(
                    "motion_confirmation_required_observations", 4
                )
            ),
            motion_confirmation_minimum_speed_mps=float(
                values.get(
                    "motion_confirmation_minimum_speed_mps", 0.12
                )
            ),
        )


@dataclass(frozen=True)
class ScanCluster:
    center: np.ndarray
    support_beams: int
    minimum_range_m: float
    beam_indices: tuple

    def __post_init__(self):
        center = np.asarray(self.center, dtype=np.float64).reshape(-1).copy()
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("scan cluster center must be a finite 2-vector")
        if int(self.support_beams) <= 0:
            raise ValueError("scan cluster support must be positive")
        if (
            not np.isfinite(self.minimum_range_m)
            or float(self.minimum_range_m) < 0.0
        ):
            raise ValueError("scan cluster minimum range is invalid")
        center.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "support_beams", int(self.support_beams))
        object.__setattr__(
            self, "minimum_range_m", float(self.minimum_range_m)
        )
        object.__setattr__(
            self,
            "beam_indices",
            tuple(int(value) for value in self.beam_indices),
        )


@dataclass(frozen=True)
class OnlineTrackingUpdate:
    measurement: Optional[np.ndarray]
    forecast: Optional[GaussianMixtureObstacleForecast]
    diagnostics: Mapping[str, object]


class SingleObstacleChangeAwareTracker:
    """Associate one circular LaserScan target using prediction gating."""

    def __init__(self, predictor, config: OnlineTrackerConfig):
        if not isinstance(predictor, OrdinaryIMMPredictor):
            raise TypeError(
                "online tracker requires an OrdinaryIMMPredictor-compatible "
                "predictor"
            )
        self.predictor = predictor
        self.config = config
        self.reset()

    @classmethod
    def from_mapping(
        cls,
        project_root,
        values: Mapping[str, object],
    ):
        root = Path(project_root).resolve()
        ordinary_path = Path(str(values["ordinary_config_path"]))
        predictor_mode = str(
            values.get("predictor_mode", "change_aware")
        ).strip().lower()
        if predictor_mode not in ("change_aware", "ordinary"):
            raise ValueError(
                "predictor_mode must be 'change_aware' or 'ordinary'"
            )
        change_path = Path(str(values.get("change_config_path", "")))
        if not ordinary_path.is_absolute():
            ordinary_path = root / ordinary_path
        if predictor_mode == "change_aware" and not change_path.is_absolute():
            change_path = root / change_path
        ordinary_config = _load_yaml(ordinary_path.resolve())
        ordinary = ordinary_config["ordinary_imm"]
        common = dict(
            observation_std=float(values["observation_std_m"]),
            initial_velocity_std=float(
                ordinary_config["prediction"]["initial_velocity_std"]
            ),
            initial_mode_probabilities=ordinary[
                "initial_mode_probabilities"
            ],
            transition_matrix=ordinary["transition_matrix"],
            turn_rate_radps=ordinary["turn_rate_radps"],
            brake_decay_rate_per_s=ordinary["brake_decay_rate_per_s"],
            process_acceleration_std=ordinary[
                "process_acceleration_std"
            ],
        )
        if predictor_mode == "ordinary":
            predictor = OrdinaryIMMPredictor(**common)
        else:
            change_config = _load_yaml(change_path.resolve())
            candidate_name = str(
                values.get("change_candidate", "dual_975_999")
            )
            detector = change_config["pilot_candidates"][candidate_name]
            response = change_config["change_response"]
            predictor = ChangeAwareIMMPredictor(
                **common,
                nis_threshold=detector["nis_threshold"],
                required_exceedances=detector["required_exceedances"],
                window_observations=detector["window_observations"],
                single_exceedance_threshold=detector.get(
                    "single_exceedance_threshold"
                ),
                reset_mode_probabilities=response[
                    "reset_mode_probabilities"
                ],
                state_covariance_inflation=response[
                    "state_covariance_inflation"
                ],
                recovery_process_noise_scale=response[
                    "recovery_process_noise_scale"
                ],
                recovery_duration_s=response["recovery_duration_s"],
                refractory_period_s=response["refractory_period_s"],
                dropout_guard_after_s=response["dropout_guard_after_s"],
                dropout_covariance_inflation=response[
                    "dropout_covariance_inflation"
                ],
            )
        return cls(predictor, OnlineTrackerConfig.from_mapping(values))

    def reset(self):
        self.predictor.reset()
        self.last_associated_timestamp = None
        self.last_timestamp = None
        self.update_count = 0
        self.associated_count = 0
        self.forecast_count = 0
        self.measurement_history = []

    def _sensor_origin(self, observation):
        theta = float(observation.pose.theta)
        offset = self.config.sensor_forward_offset_m
        return np.asarray(
            (
                observation.pose.x + offset * np.cos(theta),
                observation.pose.y + offset * np.sin(theta),
            ),
            dtype=np.float64,
        )

    def scan_clusters(self, observation: RobotObservation):
        scan = observation.scan
        if scan is None:
            return ()
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(scan.range_min))
            & (ranges < float(scan.range_max) - 1.0e-12)
        )
        indices = np.flatnonzero(valid)
        if not indices.size:
            return ()
        origin = self._sensor_origin(observation)
        angles = (
            float(observation.pose.theta)
            + float(scan.angle_min)
            + indices * float(scan.angle_increment)
        )
        points = origin[None, :] + ranges[indices, None] * np.stack(
            (np.cos(angles), np.sin(angles)), axis=1
        )
        groups = []
        current = [0]
        for local in range(1, indices.size):
            adjacent_beam = int(indices[local] - indices[local - 1]) == 1
            adjacent_point = (
                float(np.linalg.norm(points[local] - points[local - 1]))
                <= self.config.maximum_cluster_point_gap_m
            )
            if adjacent_beam and adjacent_point:
                current.append(local)
            else:
                groups.append(current)
                current = [local]
        groups.append(current)
        if (
            len(groups) > 1
            and int(indices[groups[0][0]]) == 0
            and int(indices[groups[-1][-1]]) == ranges.size - 1
            and float(
                np.linalg.norm(
                    points[groups[0][0]] - points[groups[-1][-1]]
                )
            )
            <= self.config.maximum_cluster_point_gap_m
        ):
            groups[0] = groups[-1] + groups[0]
            groups.pop()

        clusters = []
        minimum_support = (
            self.config.minimum_initial_cluster_beams
            if self.predictor.state is None
            else self.config.minimum_cluster_beams
        )
        for group in groups:
            if len(group) < minimum_support:
                continue
            group_indices = indices[group]
            group_ranges = ranges[group_indices]
            closest_local = int(np.argmin(group_ranges))
            closest_index = int(group_indices[closest_local])
            angle = (
                float(observation.pose.theta)
                + float(scan.angle_min)
                + closest_index * float(scan.angle_increment)
            )
            direction = np.asarray(
                (np.cos(angle), np.sin(angle)), dtype=np.float64
            )
            center = origin + (
                float(group_ranges[closest_local])
                + self.config.obstacle_radius_m
            ) * direction
            clusters.append(
                ScanCluster(
                    center=center,
                    support_beams=len(group),
                    minimum_range_m=float(group_ranges[closest_local]),
                    beam_indices=tuple(int(value) for value in group_indices),
                )
            )
        return tuple(clusters)

    def _association_reference(self, timestamp):
        if self.predictor.state is None:
            return None
        elapsed = float(timestamp) - float(self.predictor.timestamp)
        if elapsed <= 0.0:
            return np.asarray(
                self.predictor.state[:2], dtype=np.float64
            ).copy()
        return self.predictor.forecast(1, elapsed).means[0, :2].copy()

    def _associate(self, clusters: Sequence[ScanCluster], timestamp):
        if not clusters:
            return None, None
        reference = self._association_reference(timestamp)
        if reference is None:
            selected = min(
                clusters,
                key=lambda item: (
                    -item.support_beams,
                    item.minimum_range_m,
                ),
            )
            return selected, None
        distances = np.asarray(
            [
                np.linalg.norm(cluster.center - reference)
                for cluster in clusters
            ],
            dtype=np.float64,
        )
        index = int(np.argmin(distances))
        if float(distances[index]) > self.config.association_gate_m:
            return None, float(distances[index])
        return clusters[index], float(distances[index])

    def update(self, observation: RobotObservation):
        timestamp = float(observation.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("tracker timestamp must be finite")
        if (
            self.last_timestamp is not None
            and timestamp <= float(self.last_timestamp)
        ):
            raise ValueError("tracker timestamps must increase strictly")
        clusters = self.scan_clusters(observation)
        selected, association_distance = self._associate(
            clusters, timestamp
        )
        measurement = (
            None if selected is None else selected.center.copy()
        )
        self.predictor.update(measurement, timestamp)
        self.update_count += 1
        if measurement is not None:
            self.associated_count += 1
            self.last_associated_timestamp = timestamp
            self.measurement_history.append((
                timestamp,
                float(measurement[0]),
                float(measurement[1]),
            ))
            self.measurement_history = self.measurement_history[-8:]
        self.last_timestamp = timestamp
        unobserved_duration = (
            0.0
            if self.last_associated_timestamp is None
            else timestamp - float(self.last_associated_timestamp)
        )
        measurement_velocity = None
        required_motion_observations = int(
            self.config.motion_confirmation_required_observations
        )
        if len(self.measurement_history) >= required_motion_observations:
            history = np.asarray(
                self.measurement_history[-required_motion_observations:],
                dtype=np.float64,
            )
            times = history[:, 0] - float(np.mean(history[:, 0]))
            denominator = float(np.sum(times ** 2))
            if denominator > 1.0e-12:
                measurement_velocity = np.sum(
                    times[:, None] * history[:, 1:3], axis=0
                ) / denominator
        measurement_speed = (
            None
            if measurement_velocity is None
            else float(np.linalg.norm(measurement_velocity))
        )
        motion_confirmed = bool(
            not self.config.motion_confirmation_enabled
            or (
                measurement_speed is not None
                and measurement_speed
                >= self.config.motion_confirmation_minimum_speed_mps
            )
        )
        forecast = None
        valid = bool(
            self.predictor.state is not None
            and self.last_associated_timestamp is not None
            and unobserved_duration
            <= self.config.maximum_unobserved_duration_s + 1.0e-12
            and motion_confirmed
        )
        if valid:
            prediction = self.predictor.forecast(
                self.config.forecast_horizon_steps,
                self.config.forecast_dt_s,
            )
            forecast = GaussianMixtureObstacleForecast.from_prediction(
                prediction,
                timestamp=timestamp,
                dt=self.config.forecast_dt_s,
                radius_m=self.config.obstacle_radius_m,
                source="online_%s" % str(self.predictor.name),
            )
            self.forecast_count += 1
        diagnostics = {
            "enabled": True,
            "cluster_count": len(clusters),
            "associated": measurement is not None,
            "association_distance_m": association_distance,
            "measurement_x": (
                None if measurement is None else float(measurement[0])
            ),
            "measurement_y": (
                None if measurement is None else float(measurement[1])
            ),
            "measurement_velocity_x_mps": (
                None
                if measurement_velocity is None
                else float(measurement_velocity[0])
            ),
            "measurement_velocity_y_mps": (
                None
                if measurement_velocity is None
                else float(measurement_velocity[1])
            ),
            "measurement_speed_mps": measurement_speed,
            "motion_confirmation_enabled": bool(
                self.config.motion_confirmation_enabled
            ),
            "motion_confirmation_required_observations": (
                required_motion_observations
            ),
            "motion_confirmation_minimum_speed_mps": float(
                self.config.motion_confirmation_minimum_speed_mps
            ),
            "motion_confirmed": motion_confirmed,
            "selected_support_beams": (
                0 if selected is None else selected.support_beams
            ),
            "unobserved_duration_s": float(unobserved_duration),
            "forecast_valid": forecast is not None,
            "forecast_count": self.forecast_count,
            "update_count": self.update_count,
            "forecast_availability": float(
                self.forecast_count / self.update_count
            ),
            "predictor_source": str(self.predictor.name),
            "innovation_nis": getattr(
                self.predictor, "last_innovation_nis", None
            ),
            "change_triggered": bool(getattr(
                self.predictor, "last_change_triggered", False
            )),
            "dropout_guard_triggered": (
                bool(getattr(
                    self.predictor,
                    "last_dropout_guard_triggered",
                    False,
                ))
            ),
            "recovery_active": bool(getattr(
                self.predictor, "recovery_active", False
            )),
        }
        return OnlineTrackingUpdate(
            measurement=measurement,
            forecast=forecast,
            diagnostics=diagnostics,
        )


__all__ = [
    "OnlineTrackerConfig",
    "OnlineTrackingUpdate",
    "ScanCluster",
    "SingleObstacleChangeAwareTracker",
]
