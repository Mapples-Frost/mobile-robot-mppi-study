"""Python 3 facade around the protected scan_guard/local obstacle assets."""

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from mobile_robot_mppi.core.types import RobotObservation
from mobile_robot_mppi.obstacles.online_tracking import (
    SingleObstacleChangeAwareTracker,
)
from mobile_robot_mppi.obstacles.multi_online_tracking import (
    MultiObstacleChangeAwareTracker,
)
from mobile_robot_mppi.perception.scan_flow import RobustScanFlowEstimator


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PerceptionResult:
    observation: RobotObservation
    guard: Dict[str, Any]
    diagnostics: Dict[str, Any]


class LegacyScanPipeline:
    def __init__(self, project_root, config: Mapping[str, object]):
        root = Path(project_root).resolve()
        scripts = root / "mppi_hardware_bridge" / "scripts"
        self.scan_guard = _load_module("mppi_scan_guard_compat", scripts / "scan_guard.py")
        self.local_layer = _load_module("mppi_local_layer_compat", scripts / "local_obstacle_layer.py")
        self.config = dict(config)
        self.temporal_scan_flow = RobustScanFlowEstimator(
            self.config.get("temporal_scan_guard", {})
        )
        tracker_config = self.config.get(
            "dynamic_obstacle_tracker", {}
        )
        maximum_tracks = int(tracker_config.get("maximum_tracks", 1))
        tracker_type = (
            MultiObstacleChangeAwareTracker
            if maximum_tracks > 1
            else SingleObstacleChangeAwareTracker
        )
        self.dynamic_obstacle_tracker = (
            tracker_type.from_mapping(root, tracker_config)
            if bool(tracker_config.get("enabled", False))
            else None
        )

    @staticmethod
    def _forecast_sequence(forecast):
        """Normalize single- and multi-track forecasts for the planner API."""

        if forecast is None:
            return ()
        if isinstance(forecast, tuple):
            return forecast
        return (forecast,)

    def reset(self):
        """Reset episode-local temporal state without touching legacy assets."""

        self.temporal_scan_flow.reset()
        if self.dynamic_obstacle_tracker is not None:
            self.dynamic_obstacle_tracker.reset()

    def process(self, observation: RobotObservation) -> PerceptionResult:
        if observation.scan is None:
            tracker_update = (
                None
                if self.dynamic_obstacle_tracker is None
                else self.dynamic_obstacle_tracker.update(observation)
            )
            auxiliary = dict(observation.auxiliary)
            diagnostics = {"mode": "none"}
            if tracker_update is not None:
                auxiliary["dynamic_obstacle_tracker"] = dict(
                    tracker_update.diagnostics
                )
                diagnostics["dynamic_obstacle_tracker"] = dict(
                    tracker_update.diagnostics
                )
                if tracker_update.forecast is not None:
                    key = (
                        self.dynamic_obstacle_tracker.config
                        .forecast_auxiliary_key
                    )
                    auxiliary[key] = self._forecast_sequence(
                        tracker_update.forecast
                    )
            return PerceptionResult(
                replace(observation, auxiliary=auxiliary),
                {"emergency_stop": False, "reason": "no_scan"},
                diagnostics,
            )
        scan = observation.scan
        guard_cfg = self.config.get("scan_guard", {})
        guard = dict(self.scan_guard.analyze_scan_front_sector(
            ranges=scan.ranges.tolist(),
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            front_stop_distance=float(guard_cfg.get("front_stop_distance", 0.34)),
            front_slow_distance=float(guard_cfg.get("front_slow_distance", 0.85)),
            front_angle_deg=float(guard_cfg.get("front_angle_deg", 35.0)),
            hard_stop_distance=float(guard_cfg.get("hard_stop_distance", 0.28)),
            front_soft_block_distance=float(guard_cfg.get("front_soft_block_distance", 0.45)),
            front_slow_min_scale=float(guard_cfg.get("front_slow_min_scale", 0.45)),
            side_stop_distance=float(guard_cfg.get("side_stop_distance", 0.18)),
            side_angle_deg=float(guard_cfg.get("side_angle_deg", 125.0)),
            near_body_stop_radius=float(guard_cfg.get("near_body_stop_radius", 0.20)),
        ))
        flow = self.temporal_scan_flow.update(scan, observation.timestamp)
        flow_values = flow.to_dict()
        guard.update({
            "temporal_scan_valid": flow.valid,
            "temporal_scan_reason": flow.reason,
            "temporal_scan_closing_rate_mps": flow.closing_rate_mps,
            "temporal_scan_clearance_m": flow.clearance_m,
            "temporal_scan_ttc_s": flow.ttc_s,
            "temporal_scan_risk_alpha": flow.risk_alpha,
            "temporal_scan_support_beams": flow.support_beams,
            "temporal_scan_support_fraction": flow.support_fraction,
            "temporal_scan_rejected_jump_fraction": (
                flow.rejected_jump_fraction
            ),
            "temporal_scan_held": flow.held,
        })
        flow_cfg = self.temporal_scan_flow.config
        if flow_cfg.safety_enabled and flow.valid:
            if (
                flow.ttc_s <= flow_cfg.safety_hard_stop_ttc_s
                and not bool(guard.get("emergency_stop", False))
            ):
                guard["emergency_stop"] = True
                guard["should_slow_down"] = False
                guard["slow_scale"] = 0.0
                guard["reason"] = "temporal_collision_risk"
            elif (
                flow.ttc_s <= flow_cfg.safety_slow_ttc_s
                and not bool(guard.get("emergency_stop", False))
                and str(guard.get("reason", "front_clear"))
                != "front_soft_block"
            ):
                legacy_scale = float(guard.get("slow_scale", 1.0))
                temporal_scale = min(legacy_scale, flow_cfg.safety_slow_scale)
                if temporal_scale < legacy_scale - 1e-12:
                    guard["reason"] = "temporal_slowdown"
                guard["should_slow_down"] = True
                guard["slow_scale"] = temporal_scale
        values = scan.obstacle_ranges if scan.obstacle_ranges is not None else scan.ranges
        layer_cfg = self.config.get("local_obstacle_layer", {})
        state = observation.pose.as_array()
        try:
            obstacles, debug = self.local_layer.scan_to_experiment_obstacles_geometric(
                ranges=values.tolist(),
                angle_min=scan.angle_min,
                angle_increment=scan.angle_increment,
                range_min=scan.range_min,
                range_max=scan.range_max,
                current_state_exp=tuple(state),
                max_radius=scan.range_max,
                min_radius=float(layer_cfg.get("min_radius", 0.08)),
                angle_offset_rad=0.0,
                downsample_step=int(layer_cfg.get("downsample_step", 1)),
                obstacle_radius=float(layer_cfg.get("obstacle_radius", 0.08)),
                max_obstacle_count=int(layer_cfg.get("max_obstacle_count", 80)),
                obstacle_inflation=float(layer_cfg.get("obstacle_inflation", 0.0)),
                return_debug=True,
            )
        except Exception as exc:
            obstacles = self.local_layer.scan_to_experiment_obstacles(
                ranges=values.tolist(),
                angle_min=scan.angle_min,
                angle_increment=scan.angle_increment,
                range_min=scan.range_min,
                range_max=scan.range_max,
                current_state_exp=tuple(state),
                max_radius=scan.range_max,
                min_radius=float(layer_cfg.get("min_radius", 0.08)),
                angle_offset_rad=0.0,
                downsample_step=int(layer_cfg.get("fallback_downsample_step", 3)),
                obstacle_radius=float(layer_cfg.get("obstacle_radius", 0.08)),
            )
            debug = {"mode": "point_fallback", "error": str(exc)}
        max_count = int(layer_cfg.get("planner_max_obstacles", 24))
        obstacles = sorted(
            obstacles,
            key=lambda item: (float(item[0]) - observation.pose.x) ** 2 + (float(item[1]) - observation.pose.y) ** 2,
        )[:max_count]
        normalized = tuple(
            (float(item[0]), float(item[1]), float(item[2]) if len(item) >= 3 else 0.08)
            for item in obstacles
        )
        auxiliary = dict(observation.auxiliary)
        auxiliary["temporal_scan_flow"] = flow_values
        diagnostics = dict(debug)
        diagnostics["temporal_scan_flow"] = flow_values
        if self.dynamic_obstacle_tracker is not None:
            tracker_update = self.dynamic_obstacle_tracker.update(
                observation
            )
            tracker_diagnostics = dict(tracker_update.diagnostics)
            auxiliary["dynamic_obstacle_tracker"] = tracker_diagnostics
            diagnostics["dynamic_obstacle_tracker"] = tracker_diagnostics
            guard["dynamic_obstacle_associated"] = bool(
                tracker_diagnostics.get("associated", False)
            )
            measurement_x = tracker_diagnostics.get("measurement_x")
            measurement_y = tracker_diagnostics.get("measurement_y")
            if (
                guard["dynamic_obstacle_associated"]
                and measurement_x is not None
                and measurement_y is not None
            ):
                tracker_cfg = self.dynamic_obstacle_tracker.config
                sensor_x = (
                    observation.pose.x
                    + tracker_cfg.sensor_forward_offset_m
                    * np.cos(observation.pose.theta)
                )
                sensor_y = (
                    observation.pose.y
                    + tracker_cfg.sensor_forward_offset_m
                    * np.sin(observation.pose.theta)
                )
                expected_surface_range = max(
                    0.0,
                    float(np.hypot(
                        float(measurement_x) - sensor_x,
                        float(measurement_y) - sensor_y,
                    ))
                    - tracker_cfg.obstacle_radius_m,
                )
                guard["dynamic_obstacle_surface_range_m"] = (
                    expected_surface_range
                )
                obstacle_bearing = float(np.arctan2(
                    float(measurement_y) - sensor_y,
                    float(measurement_x) - sensor_x,
                ) - observation.pose.theta)
                obstacle_bearing = float(np.arctan2(
                    np.sin(obstacle_bearing),
                    np.cos(obstacle_bearing),
                ))
                away_heading_error = float(np.arctan2(
                    np.sin(obstacle_bearing + np.pi),
                    np.cos(obstacle_bearing + np.pi),
                ))
                guard["dynamic_obstacle_bearing_rad"] = (
                    obstacle_bearing
                )
                guard["dynamic_obstacle_away_heading_error_rad"] = (
                    away_heading_error
                )
                near_body_value = guard.get("min_near_body_range")
                near_body_range = (
                    float(near_body_value)
                    if near_body_value is not None
                    else float("inf")
                )
                match_tolerance = float(
                    self.config.get("scan_guard", {}).get(
                        "dynamic_escape_match_tolerance_m", 0.12
                    )
                )
                guard["dynamic_obstacle_near_body_match"] = bool(
                    np.isfinite(near_body_range)
                    and abs(
                        near_body_range - expected_surface_range
                    )
                    <= match_tolerance
                )
                angle_tolerance = np.deg2rad(float(
                    self.config.get("scan_guard", {}).get(
                        "dynamic_escape_flow_angle_tolerance_deg",
                        20.0,
                    )
                ))
                flow_angle_error = float(np.arctan2(
                    np.sin(
                        flow.center_angle_rad - obstacle_bearing
                    ),
                    np.cos(
                        flow.center_angle_rad - obstacle_bearing
                    ),
                ))
                guard["dynamic_obstacle_scan_flow_match"] = bool(
                    flow.valid
                    and abs(flow_angle_error) <= angle_tolerance
                )
            if tracker_update.forecast is not None:
                key = (
                    self.dynamic_obstacle_tracker.config
                    .forecast_auxiliary_key
                )
                auxiliary[key] = self._forecast_sequence(
                    tracker_update.forecast
                )
        auxiliary["dynamic_obstacle_escape_context"] = {
            "temporal_scan_valid": bool(flow.valid),
            "temporal_scan_ttc_s": float(flow.ttc_s),
            "temporal_scan_safety_hard_stop_ttc_s": float(
                flow_cfg.safety_hard_stop_ttc_s
            ),
            "dynamic_obstacle_scan_flow_match": bool(
                guard.get("dynamic_obstacle_scan_flow_match", False)
            ),
            "dynamic_obstacle_away_heading_error_rad": guard.get(
                "dynamic_obstacle_away_heading_error_rad"
            ),
            "dynamic_obstacle_measurement_velocity_x_mps": (
                None
                if self.dynamic_obstacle_tracker is None
                else tracker_diagnostics.get(
                    "measurement_velocity_x_mps"
                )
            ),
            "dynamic_obstacle_measurement_velocity_y_mps": (
                None
                if self.dynamic_obstacle_tracker is None
                else tracker_diagnostics.get(
                    "measurement_velocity_y_mps"
                )
            ),
            "dynamic_obstacle_forecast_index": (
                0
                if self.dynamic_obstacle_tracker is None
                else int(
                    tracker_diagnostics.get(
                        "nearest_forecast_index", 0
                    )
                    or 0
                )
            ),
        }
        return PerceptionResult(
            replace(
                observation,
                local_obstacles=normalized,
                auxiliary=auxiliary,
            ),
            guard,
            diagnostics,
        )
