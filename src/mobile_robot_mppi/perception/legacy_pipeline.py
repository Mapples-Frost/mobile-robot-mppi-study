"""Python 3 facade around the protected scan_guard/local obstacle assets."""

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from mobile_robot_mppi.core.types import RobotObservation


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

    def process(self, observation: RobotObservation) -> PerceptionResult:
        if observation.scan is None:
            return PerceptionResult(observation, {"emergency_stop": False, "reason": "no_scan"}, {"mode": "none"})
        scan = observation.scan
        guard_cfg = self.config.get("scan_guard", {})
        guard = self.scan_guard.analyze_scan_front_sector(
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
        )
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
        return PerceptionResult(
            replace(observation, local_obstacles=normalized),
            dict(guard),
            dict(debug),
        )
