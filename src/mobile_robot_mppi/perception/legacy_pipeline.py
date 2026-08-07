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
from mobile_robot_mppi.perception.scan_flow import (
    RobustScanFlowEstimator,
    TemporalSafetyHysteresis,
)
from mobile_robot_mppi.real_robot.person_track_manager import PersonTrackManager


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
        local_layer_config = self.config.get("local_obstacle_layer", {})
        self.local_obstacle_hard_filter_enabled = bool(
            local_layer_config.get(
                "local_obstacle_hard_filter_enabled", False
            )
        )
        self.local_obstacle_hard_filter_max_obstacles = int(
            local_layer_config.get(
                "local_obstacle_hard_filter_max_obstacles", 16
            )
        )
        if self.local_obstacle_hard_filter_max_obstacles < 1:
            raise ValueError(
                "local_obstacle_hard_filter_max_obstacles must be positive"
            )
        temporal_scan_config = self.config.get("temporal_scan_guard", {})
        self.temporal_scan_flow = RobustScanFlowEstimator(
            temporal_scan_config
        )
        self.temporal_safety_hysteresis = TemporalSafetyHysteresis(
            enabled=temporal_scan_config.get(
                "safety_state_hysteresis_enabled", False
            ),
            release_clear_frames=temporal_scan_config.get(
                "safety_release_clear_frames", 2
            ),
        )
        self._temporal_safety_last_slow_scale = float(
            temporal_scan_config.get("safety_slow_scale", 0.25)
        )
        tracker_config = self.config.get(
            "dynamic_obstacle_tracker", {}
        )
        self.known_static_filter_enabled = bool(
            tracker_config.get("known_static_filter_enabled", False)
        )
        self.known_static_filter_tolerance_m = float(
            tracker_config.get(
                "known_static_filter_tolerance_m", 0.06
            )
        )
        # The raw scan remains the source for geometric collision guarding.
        # When enabled, temporal closing-rate/TTC inference instead consumes
        # the scan after known-static beam masking.  This prevents wall
        # returns from being interpreted as moving obstacles while preserving
        # the independent raw geometric safety guard.  Default off preserves
        # historical behavior exactly.
        self.known_static_temporal_flow_filter_enabled = bool(
            tracker_config.get(
                "known_static_temporal_flow_filter_enabled", False
            )
        )
        self.known_static_obstacles = tuple(
            tracker_config.get("known_static_obstacles", ())
        )
        # Cluster-centre static rejection.  The scan filter above screens beam
        # endpoints; survivors are clustered, so a cluster centre can still sit
        # on a wall and be tracked as a moving obstacle.  This second test runs
        # on the cluster CENTRE, where real and phantom obstacles separate
        # cleanly: a real obstacle's centre cannot approach a wall closer than
        # its own half-extent plus the scene's dynamic clearance floor.
        # Default off, which reproduces the historical behaviour exactly.
        self.known_static_track_rejection_enabled = bool(
            tracker_config.get("known_static_track_rejection_enabled", False)
        )
        self.known_static_track_rejection_distance_m = float(
            tracker_config.get(
                "known_static_track_rejection_distance_m", 0.15
            )
        )
        if (
            self.known_static_filter_enabled
            and not self.known_static_obstacles
        ):
            raise ValueError(
                "known static filtering requires injected static geometry"
            )
        if (
            self.known_static_track_rejection_enabled
            and not self.known_static_obstacles
        ):
            raise ValueError(
                "known static track rejection requires injected static geometry"
            )
        if self.known_static_track_rejection_distance_m < 0.0:
            raise ValueError(
                "known_static_track_rejection_distance_m must be non-negative"
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
        self.person_track_manager = PersonTrackManager(
            self.config.get("person_tracking", {})
        )
        self.human_point_cloud_auxiliary_key = str(
            self.config.get("person_tracking", {})
            .get("point_cloud", {})
            .get("auxiliary_key", "human_point_cloud_base")
        )
        self.strict_person_forecast_admission = bool(
            self.config.get("person_tracking", {}).get(
                "strict_forecast_admission", False
            )
        )
        if (
            self.known_static_track_rejection_enabled
            and self.dynamic_obstacle_tracker is not None
        ):
            if not hasattr(
                self.dynamic_obstacle_tracker, "static_cluster_rejection"
            ):
                raise ValueError(
                    "known static track rejection requires the multi-target "
                    "tracker; the single-target tracker has no cluster stage"
                )
            self.dynamic_obstacle_tracker.static_cluster_rejection = (
                self._static_cluster_rejection
            )

    def _static_cluster_rejection(self, centers):
        """Reject cluster centres lying on known static geometry.

        Reuses the same geometry as the scan filter, but with
        ``segment_half_thickness=True`` so the comparison is against the true
        wall SURFACE rather than a doubled thickness, and with its own wider
        tolerance.
        """

        return self._known_static_hit_mask(
            centers,
            segment_half_thickness=True,
            tolerance=self.known_static_track_rejection_distance_m,
        )

    def _temporal_flow_scan(self, raw_observation, planner_observation):
        """Choose the scan supplied to temporal dynamic-risk inference.

        The geometric scan guard must continue to inspect the raw scan.  The
        temporal estimator is a separate dynamic-only signal and may use the
        known-static-filtered scan when explicitly enabled.
        """

        if self.known_static_temporal_flow_filter_enabled:
            return planner_observation.scan
        return raw_observation.scan

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
        self.temporal_safety_hysteresis.reset()
        self._temporal_safety_last_slow_scale = float(
            self.temporal_scan_flow.config.safety_slow_scale
        )
        if self.dynamic_obstacle_tracker is not None:
            self.dynamic_obstacle_tracker.reset()
        self.person_track_manager.reset()

    def _annotate_person_tracking(
        self,
        tracker_diagnostics: Mapping[str, Any],
        forecasts,
        observation: RobotObservation,
    ) -> Dict[str, Any]:
        """Attach one causal person identity above low-level cluster slots."""

        return self.person_track_manager.update(
            tracker_diagnostics,
            forecasts=self._forecast_sequence(forecasts),
            timestamp_s=float(observation.timestamp),
            pose=observation.pose.as_array(),
        )

    def _filter_person_forecasts(self, tracker_diagnostics, forecasts):
        """Enforce the person-identity forecast publication contract.

        Low-level tracker slots are useful for geometric hard safety, but an
        ``INVALID`` slot must never reach the probabilistic planner as a
        directional forecast.  The real-robot profile enables this strict
        boundary; the default remains compatibility-preserving for simulation
        and legacy replay callers.
        """

        values = dict(tracker_diagnostics or {})
        sequence = self._forecast_sequence(forecasts)
        if not self.strict_person_forecast_admission:
            values.setdefault("person_forecast_admission_enabled", False)
            values.setdefault("person_forecast_rejected_track_indices", ())
            return values, sequence
        forecast_indices = tuple(
            int(value)
            for value in values.get("forecast_track_indices", ())
        )
        allowed_indices = set()
        for person in values.get("person_tracks", ()) or ():
            person = dict(person or {})
            if str(person.get("forecast_qualification", "INVALID")) in {
                "VALID", "PROVISIONAL"
            }:
                allowed_indices.update(
                    int(value)
                    for value in person.get("forecast_track_indices", ())
                )
        retained = []
        retained_indices = []
        rejected_indices = []
        for track_index, forecast in zip(forecast_indices, sequence):
            if track_index in allowed_indices:
                retained_indices.append(track_index)
                retained.append(forecast)
            else:
                rejected_indices.append(track_index)
        # A malformed index/forecast length is rejected rather than silently
        # re-indexed to a different person.
        if len(sequence) != len(forecast_indices):
            rejected_indices.extend(
                forecast_indices[len(sequence):]
            )
        retained_indices = tuple(retained_indices)
        rejected_indices = tuple(sorted(set(rejected_indices)))
        values.update({
            "person_forecast_admission_enabled": True,
            "person_forecast_admitted_track_indices": retained_indices,
            "person_forecast_rejected_track_indices": rejected_indices,
            "person_forecast_admission_reason": (
                "admitted"
                if retained_indices
                else (
                    "no_valid_person_identity"
                    if sequence
                    else "no_low_level_forecast"
                )
            ),
            "forecast_track_indices": retained_indices,
            "person_forecast_candidate_track_indices": retained_indices,
            "valid_forecast_count": len(retained),
            "forecast_valid": bool(retained),
            "motion_confirmed": bool(retained),
            "forecast_unavailable_reason": (
                "available" if retained else "person_identity_not_qualified"
            ),
        })
        return values, tuple(retained)

    @staticmethod
    def _point_segment_distance(points, start, end):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        delta = end - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1.0e-12:
            return np.linalg.norm(points - start[None, :], axis=1)
        fraction = np.clip(
            np.sum((points - start[None, :]) * delta[None, :], axis=1)
            / denominator,
            0.0,
            1.0,
        )
        projections = start[None, :] + fraction[:, None] * delta[None, :]
        return np.linalg.norm(points - projections, axis=1)

    def _known_static_hit_mask(
        self, points, *, segment_half_thickness=False, tolerance=None
    ):
        points = np.asarray(points, dtype=np.float64)
        matched = np.zeros(points.shape[0], dtype=bool)
        tolerance = (
            self.known_static_filter_tolerance_m
            if tolerance is None
            else float(tolerance)
        )
        for obstacle in self.known_static_obstacles:
            kind = str(obstacle.get("type", "cylinder"))
            if kind == "box":
                position = np.asarray(
                    obstacle["position"][:2], dtype=np.float64
                )
                yaw = float(obstacle.get("yaw", 0.0))
                cosine = float(np.cos(yaw))
                sine = float(np.sin(yaw))
                relative = points - position[None, :]
                local = np.stack(
                    (
                        cosine * relative[:, 0]
                        + sine * relative[:, 1],
                        -sine * relative[:, 0]
                        + cosine * relative[:, 1],
                    ),
                    axis=1,
                )
                half_size = np.asarray(
                    obstacle["size"][:2], dtype=np.float64
                )
                outside = np.maximum(
                    np.abs(local) - half_size[None, :], 0.0
                )
                matched |= (
                    np.linalg.norm(outside, axis=1)
                    <= tolerance
                )
            elif kind == "segment":
                distance = self._point_segment_distance(
                    points, obstacle["start"], obstacle["end"]
                )
                matched |= distance <= (
                    (
                        0.5
                        if segment_half_thickness
                        else 1.0
                    )
                    * float(obstacle.get("thickness", 0.10))
                    + tolerance
                )
            elif kind == "cylinder":
                position = np.asarray(
                    obstacle["position"][:2], dtype=np.float64
                )
                radial = np.linalg.norm(
                    points - position[None, :], axis=1
                )
                matched |= radial <= (
                    float(obstacle.get("radius", 0.25))
                    + tolerance
                )
        return matched

    def _dynamic_tracker_observation(self, observation):
        if (
            not self.known_static_filter_enabled
            or self.dynamic_obstacle_tracker is None
            or observation.scan is None
        ):
            return observation
        scan = observation.scan
        ranges = np.asarray(scan.ranges, dtype=np.float64).copy()
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(scan.range_min))
            & (ranges < float(scan.range_max) - 1.0e-12)
        )
        indices = np.flatnonzero(valid)
        if not indices.size:
            return observation
        tracker_cfg = self.dynamic_obstacle_tracker.config
        theta = float(observation.pose.theta)
        origin = np.asarray(
            (
                observation.pose.x
                + tracker_cfg.sensor_forward_offset_m * np.cos(theta),
                observation.pose.y
                + tracker_cfg.sensor_forward_offset_m * np.sin(theta),
            ),
            dtype=np.float64,
        )
        angles = (
            theta
            + float(scan.angle_min)
            + indices * float(scan.angle_increment)
        )
        endpoints = origin[None, :] + ranges[indices, None] * np.stack(
            (np.cos(angles), np.sin(angles)), axis=1
        )
        static_hits = self._known_static_hit_mask(endpoints)
        ranges[indices[static_hits]] = float(scan.range_max)
        obstacle_ranges = np.where(
            ranges < float(scan.range_max) - 1.0e-12,
            ranges,
            np.inf,
        )
        auxiliary = dict(observation.auxiliary)
        auxiliary["known_static_filter"] = {
            "enabled": True,
            "input_hit_count": int(indices.size),
            "masked_static_hit_count": int(np.sum(static_hits)),
            "residual_dynamic_hit_count": int(
                indices.size - np.sum(static_hits)
            ),
        }
        return replace(
            observation,
            scan=replace(
                scan,
                ranges=ranges,
                obstacle_ranges=obstacle_ranges,
            ),
            auxiliary=auxiliary,
        )

    def process(self, observation: RobotObservation) -> PerceptionResult:
        if observation.scan is None:
            tracker_update = (
                None
                if self.dynamic_obstacle_tracker is None
                else self.dynamic_obstacle_tracker.update(observation)
            )
            auxiliary = dict(observation.auxiliary)
            if self.known_static_obstacles:
                auxiliary["known_static_obstacles"] = (
                    self.known_static_obstacles
                )
            diagnostics = {"mode": "none"}
            if tracker_update is not None:
                tracker_values = dict(tracker_update.diagnostics)
                tracker_forecasts = tracker_update.forecast
                tracker_values = self._annotate_person_tracking(
                    tracker_values, tracker_forecasts, observation
                )
                tracker_values, tracker_forecasts = (
                    self._filter_person_forecasts(
                        tracker_values, tracker_forecasts
                    )
                )
                auxiliary["dynamic_obstacle_tracker"] = tracker_values
                auxiliary["person_tracking"] = {
                    key: value for key, value in tracker_values.items()
                    if str(key).startswith("person_")
                }
                diagnostics["dynamic_obstacle_tracker"] = tracker_values
                diagnostics["person_tracking"] = auxiliary["person_tracking"]
                if tracker_forecasts:
                    key = (
                        self.dynamic_obstacle_tracker.config
                        .forecast_auxiliary_key
                    )
                    auxiliary[key] = tracker_forecasts
            auxiliary.pop(self.human_point_cloud_auxiliary_key, None)
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
        # Filter known-static returns before temporal dynamic inference only
        # when the new opt-in is enabled.  The raw ``scan`` above remains the
        # input to the geometric safety guard.
        planner_observation = self._dynamic_tracker_observation(
            observation
        )
        flow_scan = self._temporal_flow_scan(
            observation, planner_observation
        )
        flow = self.temporal_scan_flow.update(
            flow_scan, observation.timestamp, observation.pose
        )
        flow_values = flow.to_dict()
        # The motion-bootstrap tracker must see the same causal flow estimate
        # that is logged and consumed by the geometric safety layer below.
        # Previously this value was added only to the final output auxiliary,
        # after dynamic_obstacle_tracker.update() had already run, so a close
        # moving person could trigger TTC braking without ever reaching the
        # CA-IMM slot-selection path.
        planner_auxiliary = dict(planner_observation.auxiliary)
        planner_auxiliary["temporal_scan_flow"] = flow_values
        planner_observation = replace(
            planner_observation, auxiliary=planner_auxiliary
        )
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
            "temporal_scan_center_angle_rad": flow.center_angle_rad,
            "temporal_scan_held": flow.held,
        })
        flow_cfg = self.temporal_scan_flow.config
        flow_safety_quality_ok = bool(
            flow.support_beams >= flow_cfg.safety_min_support_beams
            and flow.rejected_jump_fraction
            <= flow_cfg.safety_max_rejected_jump_fraction
        )
        require_quality_for_slowdown = bool(
            self.config.get("temporal_scan_guard", {}).get(
                "safety_slowdown_requires_quality", False
            )
        )
        guard["temporal_scan_safety_quality_ok"] = flow_safety_quality_ok
        guard["temporal_scan_slowdown_quality_required"] = (
            require_quality_for_slowdown
        )
        raw_temporal_safety_state = "clear"
        raw_temporal_slow_scale = 1.0
        if flow_cfg.safety_enabled and flow.valid:
            if (
                flow.ttc_s <= flow_cfg.safety_hard_stop_ttc_s
                and flow_safety_quality_ok
            ):
                raw_temporal_safety_state = "stop"
            elif (
                flow.ttc_s <= flow_cfg.safety_slow_ttc_s
                and (
                    not require_quality_for_slowdown
                    or flow_safety_quality_ok
                )
            ):
                raw_temporal_safety_state = "slow"
                raw_temporal_slow_scale = flow_cfg.safety_slowdown_scale(
                    flow.ttc_s
                )
                self._temporal_safety_last_slow_scale = float(
                    raw_temporal_slow_scale
                )
        temporal_safety_state, temporal_safety_diagnostics = (
            self.temporal_safety_hysteresis.update(
                raw_temporal_safety_state
            )
        )
        guard.update(temporal_safety_diagnostics)
        if (
            temporal_safety_state == "stop"
            and not bool(guard.get("emergency_stop", False))
        ):
            guard["emergency_stop"] = True
            guard["should_slow_down"] = False
            guard["slow_scale"] = 0.0
            guard["reason"] = "temporal_collision_risk"
        elif (
            temporal_safety_state == "slow"
            and not bool(guard.get("emergency_stop", False))
            and str(guard.get("reason", "front_clear"))
            != "front_soft_block"
        ):
            legacy_scale = float(guard.get("slow_scale", 1.0))
            temporal_scale = min(
                legacy_scale,
                raw_temporal_slow_scale
                if raw_temporal_safety_state == "slow"
                else self._temporal_safety_last_slow_scale,
            )
            if temporal_scale < legacy_scale - 1e-12:
                guard["reason"] = "temporal_slowdown"
            guard["should_slow_down"] = True
            guard["slow_scale"] = temporal_scale
        # Safety continues to inspect the raw scan above.  The planner already
        # receives exact injected static geometry, so its scan-derived local
        # obstacle set must contain only residual (potentially dynamic)
        # returns.  Feeding known-static hits into both representations
        # double-counts walls as dense inflated circles and can make every
        # low-risk forward action look worse than stopping.
        planner_scan = planner_observation.scan
        values = (
            planner_scan.obstacle_ranges
            if planner_scan.obstacle_ranges is not None
            else planner_scan.ranges
        )
        layer_cfg = self.config.get("local_obstacle_layer", {})
        state = observation.pose.as_array()
        try:
            obstacles, debug = self.local_layer.scan_to_experiment_obstacles_geometric(
                ranges=values.tolist(),
                angle_min=planner_scan.angle_min,
                angle_increment=planner_scan.angle_increment,
                range_min=planner_scan.range_min,
                range_max=planner_scan.range_max,
                current_state_exp=tuple(state),
                max_radius=planner_scan.range_max,
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
                angle_min=planner_scan.angle_min,
                angle_increment=planner_scan.angle_increment,
                range_min=planner_scan.range_min,
                range_max=planner_scan.range_max,
                current_state_exp=tuple(state),
                max_radius=planner_scan.range_max,
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
        hard_filter_obstacles = list(self.known_static_obstacles)
        if self.local_obstacle_hard_filter_enabled:
            for ox, oy, radius in normalized[
                    :self.local_obstacle_hard_filter_max_obstacles]:
                hard_filter_obstacles.append({
                    "type": "cylinder",
                    "position": [float(ox), float(oy), 0.0],
                    "radius": float(radius),
                    "source": "causal_local_scan",
                })
        if hard_filter_obstacles:
            # Reuse the planner's already tested hard static-feasibility
            # machinery for the current causal scan.  The geometry is rebuilt
            # every cycle, so no moving person is frozen across observations;
            # it simply prevents MPPI's weighted mean from cutting through a
            # currently occupied footprint when dynamic classification drops.
            auxiliary["known_static_obstacles"] = tuple(
                hard_filter_obstacles
            )
        diagnostics = dict(debug)
        diagnostics["local_obstacle_count"] = len(normalized)
        diagnostics["local_obstacle_hard_filter_enabled"] = bool(
            self.local_obstacle_hard_filter_enabled
        )
        diagnostics["local_obstacle_hard_filter_count"] = int(
            max(0, len(hard_filter_obstacles) - len(self.known_static_obstacles))
        )
        diagnostics["temporal_flow_scan_source"] = (
            "known_static_filtered"
            if self.known_static_temporal_flow_filter_enabled
            else "raw"
        )
        diagnostics["temporal_scan_flow"] = flow_values
        static_filter_diagnostics = (
            planner_observation.auxiliary.get("known_static_filter")
        )
        if static_filter_diagnostics is not None:
            static_filter_diagnostics = dict(static_filter_diagnostics)
            static_filter_diagnostics[
                "planner_local_layer_static_hits_removed"
            ] = int(
                static_filter_diagnostics.get(
                    "masked_static_hit_count", 0
                )
            )
            auxiliary["known_static_filter"] = (
                static_filter_diagnostics
            )
            diagnostics["known_static_filter"] = (
                static_filter_diagnostics
            )
        if self.dynamic_obstacle_tracker is not None:
            tracker_update = self.dynamic_obstacle_tracker.update(
                planner_observation
            )
            tracker_diagnostics = dict(tracker_update.diagnostics)
            tracker_forecasts = tracker_update.forecast
            tracker_diagnostics = self._annotate_person_tracking(
                tracker_diagnostics, tracker_forecasts, observation
            )
            tracker_diagnostics, tracker_forecasts = (
                self._filter_person_forecasts(
                    tracker_diagnostics, tracker_forecasts
                )
            )
            auxiliary["dynamic_obstacle_tracker"] = tracker_diagnostics
            diagnostics["dynamic_obstacle_tracker"] = tracker_diagnostics
            auxiliary["person_tracking"] = {
                key: value for key, value in tracker_diagnostics.items()
                if str(key).startswith("person_")
            }
            diagnostics["person_tracking"] = auxiliary["person_tracking"]
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
                # The person-level identity is the single bearing/velocity
                # source for dynamic arbitration.  Low-level nearest-slot
                # geometry remains in the audit trace, but cannot silently
                # swap the tracked human to another leg/background cluster.
                person_x = tracker_diagnostics.get(
                    "person_selected_position_x"
                )
                person_y = tracker_diagnostics.get(
                    "person_selected_position_y"
                )
                if person_x is not None and person_y is not None:
                    measurement_x = person_x
                    measurement_y = person_y
                    guard["dynamic_obstacle_geometry_source"] = (
                        "person_track"
                    )
                else:
                    guard["dynamic_obstacle_geometry_source"] = (
                        "low_level_tracker_slot"
                    )
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
            if tracker_forecasts:
                key = (
                    self.dynamic_obstacle_tracker.config
                    .forecast_auxiliary_key
                )
                auxiliary[key] = tracker_forecasts
        # The full cloud has served its perception-only purpose.  Do not carry
        # hundreds of thousands of points through the CUDA planner or async
        # diagnostic writer.
        auxiliary.pop(self.human_point_cloud_auxiliary_key, None)
        near_body_points = tuple(guard.get("near_body_points", ()))
        known_static_near_body_match = False
        if self.known_static_obstacles and near_body_points:
            local_points = np.asarray(
                [
                    (float(point["x"]), float(point["y"]))
                    for point in near_body_points
                ],
                dtype=np.float64,
            )
            theta = float(observation.pose.theta)
            cosine = float(np.cos(theta))
            sine = float(np.sin(theta))
            rotation = np.asarray(
                ((cosine, -sine), (sine, cosine)),
                dtype=np.float64,
            )
            sensor_offset = (
                0.0
                if self.dynamic_obstacle_tracker is None
                else float(
                    self.dynamic_obstacle_tracker.config
                    .sensor_forward_offset_m
                )
            )
            sensor_origin = np.asarray(
                (
                    observation.pose.x + sensor_offset * cosine,
                    observation.pose.y + sensor_offset * sine,
                ),
                dtype=np.float64,
            )
            world_points = (
                local_points @ rotation.T + sensor_origin[None, :]
            )
            known_static_near_body_match = bool(
                np.any(self._known_static_hit_mask(
                    world_points,
                    segment_half_thickness=True,
                ))
            )
        auxiliary["known_static_near_body_match"] = (
            known_static_near_body_match
        )
        auxiliary["static_near_body_hard_stop"] = bool(
            guard.get("emergency_stop", False)
            and guard.get("reason") == "near_body_hard_stop"
            and known_static_near_body_match
            and not guard.get(
                "dynamic_obstacle_near_body_match", False
            )
            and not guard.get(
                "dynamic_obstacle_scan_flow_match", False
            )
        )
        auxiliary["dynamic_obstacle_escape_context"] = {
            "temporal_scan_valid": bool(flow.valid),
            "temporal_scan_ttc_s": float(flow.ttc_s),
            "temporal_scan_clearance_m": float(flow.clearance_m),
            "temporal_scan_support_beams": int(flow.support_beams),
            "temporal_scan_support_fraction": float(flow.support_fraction),
            "temporal_scan_rejected_jump_fraction": float(
                flow.rejected_jump_fraction
            ),
            "temporal_scan_safety_quality_ok": bool(
                flow.support_beams >= flow_cfg.safety_min_support_beams
                and flow.rejected_jump_fraction
                <= flow_cfg.safety_max_rejected_jump_fraction
            ),
            "temporal_scan_held": bool(flow.held),
            "temporal_scan_safety_raw_state": str(
                guard.get("temporal_scan_safety_raw_state", "clear")
            ),
            "temporal_scan_safety_state": str(
                guard.get("temporal_scan_safety_state", "clear")
            ),
            "temporal_scan_safety_release_pending": bool(
                guard.get("temporal_scan_safety_release_pending", False)
            ),
            "temporal_scan_safety_clear_streak": int(
                guard.get("temporal_scan_safety_clear_streak", 0)
            ),
            "temporal_scan_safety_hard_stop_ttc_s": float(
                flow_cfg.safety_hard_stop_ttc_s
            ),
            "dynamic_obstacle_scan_flow_match": bool(
                guard.get("dynamic_obstacle_scan_flow_match", False)
            ),
            "dynamic_obstacle_away_heading_error_rad": guard.get(
                "dynamic_obstacle_away_heading_error_rad"
            ),
            "dynamic_obstacle_bearing_rad": guard.get(
                "dynamic_obstacle_bearing_rad"
            ),
            "dynamic_obstacle_surface_range_m": guard.get(
                "dynamic_obstacle_surface_range_m"
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
            "dynamic_obstacle_tracker_change_triggered": bool(
                False
                if self.dynamic_obstacle_tracker is None
                else tracker_diagnostics.get("change_triggered", False)
            ),
            "dynamic_obstacle_tracker_forecast_valid": bool(
                False
                if self.dynamic_obstacle_tracker is None
                else tracker_diagnostics.get("forecast_valid", False)
            ),
            "dynamic_obstacle_tracker_associated": bool(
                False
                if self.dynamic_obstacle_tracker is None
                else tracker_diagnostics.get("associated", False)
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
            # Person-level identity is a forecast source/diagnostic contract;
            # it never writes a command and never replaces static filtering.
            "person_forecast_qualification": str(
                tracker_diagnostics.get(
                    "person_forecast_qualification", "INVALID"
                )
            ),
            "person_forecast_source": str(
                tracker_diagnostics.get(
                    "person_forecast_source", "unavailable"
                )
            ),
            "person_selected_id": tracker_diagnostics.get(
                "person_selected_id"
            ),
            "person_identity_continuity": bool(
                tracker_diagnostics.get(
                    "person_identity_continuity", False
                )
            ),
            "person_forecast_candidate_track_indices": tuple(
                int(value)
                for value in tracker_diagnostics.get(
                    "person_forecast_candidate_track_indices", ()
                )
            ),
            "low_level_forecast_track_indices": tuple(
                int(value)
                for value in tracker_diagnostics.get(
                    "low_level_forecast_track_indices", ()
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
