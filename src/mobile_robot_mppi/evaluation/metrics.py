import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.evaluation.tracking import (
    FootprintCorridor,
    TrackingEventMonitor,
)


@dataclass
class EpisodeMetrics:
    goal_x: float
    goal_y: float
    goal_tolerance: float
    control_dt: float = 0.0
    reference_points: Optional[Sequence[Sequence[float]]] = None
    tracking_reference: Optional[PolylineReference] = None
    tracking_corridor: Optional[FootprintCorridor] = None
    tracking_obstacle_positions: Sequence[Sequence[float]] = field(
        default_factory=tuple
    )
    tracking_center_crossing: Optional[Sequence[float]] = None
    tracking_center_crossing_radius: float = 0.5
    records: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        self._tracking_monitor = None
        if self.tracking_reference is not None:
            if self.tracking_corridor is None:
                raise ValueError("tracking corridor is required with a reference")
            self._tracking_monitor = TrackingEventMonitor(
                self.tracking_reference,
                self.tracking_corridor,
                self.tracking_obstacle_positions,
                center_crossing=self.tracking_center_crossing,
                center_crossing_radius=self.tracking_center_crossing_radius,
            )

    def _cross_track_errors(self, values):
        if self.reference_points is None:
            return None
        points = np.asarray(self.reference_points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 2
            or points.shape[1] < 2
            or not np.isfinite(points[:, :2]).all()
        ):
            raise ValueError(
                "reference_points must contain at least two finite XY points"
            )
        query = np.asarray(
            [(row["x"], row["y"]) for row in values], dtype=np.float64
        )
        starts = points[:-1, :2]
        segments = points[1:, :2] - starts
        squared_lengths = np.sum(segments * segments, axis=1)
        valid = squared_lengths > 1e-16
        if not np.any(valid):
            raise ValueError("reference_points must contain a nonzero segment")
        starts = starts[valid]
        segments = segments[valid]
        squared_lengths = squared_lengths[valid]
        relative = query[:, None, :] - starts[None, :, :]
        fraction = np.sum(relative * segments[None, :, :], axis=2)
        fraction = np.clip(fraction / squared_lengths[None, :], 0.0, 1.0)
        projections = starts[None, :, :] + fraction[:, :, None] * segments[None, :, :]
        distance = np.linalg.norm(query[:, None, :] - projections, axis=2)
        return np.min(distance, axis=1)

    def update(self, truth, safety_decision, planner_diagnostics, applied_control=None):
        distance = math.hypot(truth.pose.x - self.goal_x, truth.pose.y - self.goal_y)
        applied = applied_control or safety_decision.executed_control
        prior = dict(planner_diagnostics.get("prior", {}))
        truth_metadata = dict(getattr(truth, "metadata", {}) or {})
        safety_diagnostics = dict(
            getattr(safety_decision, "diagnostics", {}) or {}
        )
        record = {
            "time": truth.timestamp,
            "x": truth.pose.x,
            "y": truth.pose.y,
            "theta": truth.pose.theta,
            "v": truth.twist.v,
            "omega": truth.twist.omega,
            "goal_distance": distance,
            "collision": float(truth.collision),
            "clearance": truth.minimum_clearance,
            "slip_ratio": truth.slip_ratio,
            "dynamic_obstacle_count": int(
                truth_metadata.get("dynamic_obstacle_count", 0)
            ),
            "nearest_dynamic_obstacle_center_distance": (
                float(truth_metadata["nearest_dynamic_obstacle_center_distance"])
                if truth_metadata.get(
                    "nearest_dynamic_obstacle_center_distance"
                ) is not None
                else float("inf")
            ),
            "proposed_v": safety_decision.proposed_control.v,
            "proposed_omega": safety_decision.proposed_control.omega,
            "executed_v": safety_decision.executed_control.v,
            "executed_omega": safety_decision.executed_control.omega,
            "applied_v": applied.v,
            "applied_omega": applied.omega,
            "safety_override": float(safety_decision.overridden),
            "safety_reason": str(safety_decision.reason),
            "dynamic_escape_allowed": float(
                safety_diagnostics.get(
                    "dynamic_escape_allowed", False
                )
            ),
            "dynamic_escape_reactive": float(
                safety_diagnostics.get(
                    "dynamic_escape_reactive", False
                )
            ),
            "dynamic_escape_held": float(
                safety_diagnostics.get("dynamic_escape_held", False)
            ),
            "dynamic_escape_hold_remaining": int(
                safety_diagnostics.get(
                    "dynamic_escape_hold_remaining", 0
                )
            ),
            "dynamic_escape_reverse": float(
                safety_diagnostics.get(
                    "dynamic_escape_reverse", False
                )
            ),
            "dynamic_escape_selected_probability": float(
                safety_diagnostics.get(
                    "dynamic_escape_selected_probability", 0.0
                )
            ),
            "dynamic_escape_stop_probability": float(
                safety_diagnostics.get(
                    "dynamic_escape_stop_probability", 0.0
                )
            ),
            "dynamic_escape_selected_probability_mass": float(
                safety_diagnostics.get(
                    "dynamic_escape_selected_probability_mass", 0.0
                )
            ),
            "dynamic_escape_stop_probability_mass": float(
                safety_diagnostics.get(
                    "dynamic_escape_stop_probability_mass", 0.0
                )
            ),
            "dynamic_escape_probability_mass_enabled": float(
                safety_diagnostics.get(
                    "dynamic_escape_probability_mass_enabled", False
                )
            ),
            "dynamic_escape_probability_mass_fallback": float(
                safety_diagnostics.get(
                    "dynamic_escape_probability_mass_fallback", False
                )
            ),
            "dynamic_escape_probability_mass_relative_threshold": float(
                safety_diagnostics.get(
                    "dynamic_escape_probability_mass_relative_threshold",
                    0.0,
                )
            ),
            "dynamic_escape_vetted_planner_control": float(
                safety_diagnostics.get(
                    "dynamic_escape_vetted_planner_control", False
                )
            ),
            "dynamic_recovery_enabled": float(
                safety_diagnostics.get(
                    "dynamic_recovery_enabled", False
                )
            ),
            "dynamic_recovery_translation_enabled": float(
                safety_diagnostics.get(
                    "dynamic_recovery_translation_enabled", False
                )
            ),
            "dynamic_recovery_minimum_heading_error_rad": float(
                safety_diagnostics.get(
                    "dynamic_recovery_minimum_heading_error_rad", 0.0
                )
            ),
            "dynamic_recovery_active": float(
                safety_diagnostics.get(
                    "dynamic_recovery_active", False
                )
            ),
            "dynamic_recovery_pending": float(
                safety_diagnostics.get(
                    "dynamic_recovery_pending", False
                )
            ),
            "dynamic_recovery_mode": str(
                safety_diagnostics.get(
                    "dynamic_recovery_mode", "inactive"
                )
            ),
            "dynamic_recovery_guard_clear": float(
                safety_diagnostics.get(
                    "dynamic_recovery_guard_clear", False
                )
            ),
            "dynamic_recovery_clear_steps": int(
                safety_diagnostics.get(
                    "dynamic_recovery_clear_steps", 0
                )
            ),
            "dynamic_recovery_release_count": int(
                safety_diagnostics.get(
                    "dynamic_recovery_release_count", 0
                )
            ),
            "planner_temporal_escape_active": float(
                safety_diagnostics.get(
                    "planner_temporal_escape_active", False
                )
            ),
            "temporal_scan_valid": float(
                safety_diagnostics.get("temporal_scan_valid", False)
            ),
            "temporal_scan_closing_rate_mps": float(
                safety_diagnostics.get(
                    "temporal_scan_closing_rate_mps", 0.0
                )
            ),
            "temporal_scan_ttc_s": float(
                safety_diagnostics.get(
                    "temporal_scan_ttc_s", float("inf")
                )
            ),
            "temporal_scan_risk_alpha": float(
                safety_diagnostics.get(
                    "temporal_scan_risk_alpha", 0.0
                )
            ),
            "temporal_scan_support_beams": int(
                safety_diagnostics.get(
                    "temporal_scan_support_beams", 0
                )
            ),
            "temporal_scan_rejected_jump_fraction": float(
                safety_diagnostics.get(
                    "temporal_scan_rejected_jump_fraction", 0.0
                )
            ),
            "temporal_scan_held": float(
                safety_diagnostics.get("temporal_scan_held", False)
            ),
            "reference_id": str(planner_diagnostics.get("reference_id", "unknown")),
            "target_phase": str(planner_diagnostics.get("target_phase", "unknown")),
            "target_x": float(planner_diagnostics.get("target_x", self.goal_x)),
            "target_y": float(planner_diagnostics.get("target_y", self.goal_y)),
            "terminal_heading_gate_active": float(
                planner_diagnostics.get("terminal_heading_gate_active", False)
            ),
            "terminal_bearing_error": float(
                planner_diagnostics.get("terminal_bearing_error", 0.0)
            ),
            "target_bearing_error": float(
                planner_diagnostics.get("target_bearing_error", 0.0)
            ),
            "terminal_translation_scale": float(
                planner_diagnostics.get("terminal_translation_scale", 1.0)
            ),
            "terminal_alignment_active": float(
                planner_diagnostics.get("terminal_alignment_active", False)
            ),
            "terminal_alignment_omega": float(
                planner_diagnostics.get("terminal_alignment_omega", 0.0)
            ),
            "planner_compute_ms": float(planner_diagnostics.get("compute_ms", 0.0)),
            "planner_observation_x": float(
                planner_diagnostics.get("planner_observation_x", 0.0)
            ),
            "planner_observation_y": float(
                planner_diagnostics.get("planner_observation_y", 0.0)
            ),
            "planner_observation_theta": float(
                planner_diagnostics.get(
                    "planner_observation_theta", 0.0
                )
            ),
            "probabilistic_obstacle_risk_enabled": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_risk_enabled", False
                )
            ),
            "probabilistic_obstacle_forecast_count": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_forecast_count", 0
                )
            ),
            "probabilistic_obstacle_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_probability_mass", 0.0
                )
            ),
            "probabilistic_obstacle_union_bound": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_union_bound", 0.0
                )
            ),
            "probabilistic_obstacle_maximum_step_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_maximum_step_probability",
                    0.0,
                )
            ),
            "probabilistic_obstacle_hard_violation": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_hard_violation", False
                )
            ),
            "probabilistic_obstacle_fail_closed": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_fail_closed", False
                )
            ),
            "probabilistic_obstacle_speed_scale": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_speed_scale", 1.0
                )
            ),
            "probabilistic_obstacle_initial_hard_violation": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_initial_hard_violation",
                    False,
                )
            ),
            "probabilistic_obstacle_candidate_feasible_fraction": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_candidate_feasible_fraction",
                    1.0,
                )
            ),
            "probabilistic_obstacle_active_fallback_used": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_active_fallback_used",
                    False,
                )
            ),
            "probabilistic_obstacle_active_fallback_index": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_active_fallback_index", -1
                )
            ),
            "probabilistic_obstacle_active_fallback_kind": str(
                planner_diagnostics.get(
                    "probabilistic_obstacle_active_fallback_kind", "none"
                )
            ),
            "probabilistic_obstacle_stop_maximum_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_stop_maximum_probability",
                    0.0,
                )
            ),
            "probabilistic_obstacle_stop_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_stop_probability_mass", 0.0
                )
            ),
            "probabilistic_obstacle_stopping_feasibility_enabled": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_stopping_feasibility_enabled",
                    False,
                )
            ),
            "probabilistic_obstacle_emergency_candidate_count": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_emergency_candidate_count", 0
                )
            ),
            "probabilistic_obstacle_emergency_candidate_selected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_emergency_candidate_selected",
                    False,
                )
            ),
            "probabilistic_obstacle_temporal_emergency_triggered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_temporal_emergency_triggered",
                    False,
                )
            ),
            "probabilistic_obstacle_temporal_emergency_vetted": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_temporal_emergency_vetted",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_window_enabled": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_window_enabled", False
                )
            ),
            "probabilistic_obstacle_traversal_window_safe": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_window_safe", False
                )
            ),
            "probabilistic_obstacle_traversal_forecast_sufficient": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_forecast_sufficient",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_active": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_active", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_started": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_started", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_completed": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_completed", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_cancelled": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_cancelled", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_closing": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_closing",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_midpoint_guard": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_midpoint_guard",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_retreat_temporal_lattice_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_temporal_lattice_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_completed": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_completed", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_pending": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_pending", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_no_crossing_safe_streak": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_no_crossing_safe_streak",
                    0,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_released_by_no_crossing_clearance": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_released_by_no_crossing_clearance",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_active": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_active",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_safe": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_safe",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_staging_approach_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_staging_approach_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_staging_approach_safe": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_staging_approach_safe",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_staging_target_progress": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_staging_target_progress",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_rearm_hold_overridden_by_temporal_closing": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_rearm_hold_overridden_by_temporal_closing",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_progress": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_candidate_selected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_candidate_selected", False
                )
            ),
            "probabilistic_obstacle_traversal_crossing_progress": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_crossing_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_entry_progress": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_entry_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_clear_progress": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_clear_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_current_progress": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_current_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_maximum_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_maximum_probability", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_probability_mass", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_temporal_corroboration_enabled": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_corroboration_enabled",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_corroboration_maximum_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_corroboration_maximum_probability",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_corroboration_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_corroboration_probability_mass",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_full_horizon_enabled": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_full_horizon_enabled",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_full_horizon_safe": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_full_horizon_safe",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_rejected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_rejected",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_safe": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_safe",
                    True,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_prealign_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_prealign_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_terminal_release_active": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_terminal_release_active",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_margin_s": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_margin_s",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_safe_streak": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_safe_streak",
                    0,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_required_streak": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_required_streak",
                    1,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_waiting": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_waiting",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_released": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_admission_released",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_preserved_for_nearest_safe_exit": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_preserved_for_nearest_safe_exit",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_forward_exit_distance_m": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_forward_exit_distance_m",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_exit_distance_m": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_exit_distance_m",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_forward_exit_optimistic_time_s": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_forward_exit_optimistic_time_s",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_exit_optimistic_time_s": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_exit_optimistic_time_s",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_exit_deadline_ttc_s": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_exit_deadline_ttc_s",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_exit_deadline_guard_triggered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_exit_deadline_guard_triggered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_v": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_v",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_omega": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_omega",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_overridden_by_hard_risk": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_overridden_by_hard_risk",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_overridden_by_temporal_midpoint_guard": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_retreat_overridden_by_temporal_midpoint_guard",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count",
                    0,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_release_condition_met": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_release_condition_met",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_applied": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_applied",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_candidate_count": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_candidate_count",
                    0,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count",
                    0,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_selected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_selected",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_requested": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_requested",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_admitted_count": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_admitted_count",
                    0,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_selected": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_selected",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_raw_triggered": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_raw_triggered",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_temporal_closing_observed": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_temporal_closing_observed",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_latched": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_escape_latched",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_reused": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_escape_reused",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_v": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_v",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_omega": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_omega",
                    0.0,
                )
            ),
            "probabilistic_obstacle_traversal_required_steps": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_required_steps", 0
                )
            ),
            "probabilistic_obstacle_traversal_forecast_index": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_traversal_forecast_index", -1
                )
            ),
            "probabilistic_obstacle_pareto_forward_commit_applied": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_pareto_forward_commit_applied",
                    False,
                )
            ),
            "probabilistic_obstacle_low_risk_forward_commit_applied": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_low_risk_forward_commit_applied",
                    False,
                )
            ),
            "probabilistic_obstacle_preferred_emergency_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_preferred_emergency_probability",
                    0.0,
                )
            ),
            "probabilistic_obstacle_preferred_emergency_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_preferred_emergency_probability_mass",
                    0.0,
                )
            ),
            "probabilistic_obstacle_preferred_emergency_hard_violation": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_preferred_emergency_hard_violation",
                    False,
                )
            ),
            "probabilistic_obstacle_best_cost_emergency_index": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_best_cost_emergency_index", -1
                )
            ),
            "probabilistic_obstacle_best_cost_emergency_v": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_best_cost_emergency_v", 0.0
                )
            ),
            "probabilistic_obstacle_best_cost_emergency_omega": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_best_cost_emergency_omega", 0.0
                )
            ),
            "probabilistic_obstacle_best_cost_emergency_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_best_cost_emergency_probability",
                    0.0,
                )
            ),
            "probabilistic_obstacle_best_cost_emergency_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_best_cost_emergency_probability_mass",
                    0.0,
                )
            ),
            "probabilistic_obstacle_minimum_risk_emergency_index": int(
                planner_diagnostics.get(
                    "probabilistic_obstacle_minimum_risk_emergency_index", -1
                )
            ),
            "probabilistic_obstacle_minimum_risk_emergency_v": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_minimum_risk_emergency_v", 0.0
                )
            ),
            "probabilistic_obstacle_minimum_risk_emergency_omega": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_minimum_risk_emergency_omega", 0.0
                )
            ),
            "probabilistic_obstacle_minimum_risk_emergency_probability": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_minimum_risk_emergency_probability",
                    0.0,
                )
            ),
            "probabilistic_obstacle_minimum_risk_emergency_probability_mass": float(
                planner_diagnostics.get(
                    "probabilistic_obstacle_minimum_risk_emergency_probability_mass",
                    0.0,
                )
            ),
            "dynamic_obstacle_tracker_enabled": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_enabled", False
                )
            ),
            "dynamic_obstacle_tracker_cluster_count": int(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_cluster_count", 0
                )
            ),
            "dynamic_obstacle_tracker_associated": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_associated", False
                )
            ),
            "dynamic_obstacle_tracker_association_distance_m": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_association_distance_m",
                    0.0,
                )
            ),
            "dynamic_obstacle_tracker_measurement_x": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_measurement_x", 0.0
                )
            ),
            "dynamic_obstacle_tracker_measurement_y": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_measurement_y", 0.0
                )
            ),
            "dynamic_obstacle_tracker_support_beams": int(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_support_beams", 0
                )
            ),
            "dynamic_obstacle_tracker_unobserved_duration_s": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_unobserved_duration_s",
                    0.0,
                )
            ),
            "dynamic_obstacle_tracker_forecast_valid": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_forecast_valid", False
                )
            ),
            "dynamic_obstacle_tracker_forecast_availability": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_forecast_availability",
                    0.0,
                )
            ),
            "dynamic_obstacle_tracker_innovation_nis": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_innovation_nis", 0.0
                )
            ),
            "dynamic_obstacle_tracker_change_triggered": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_change_triggered", False
                )
            ),
            "dynamic_obstacle_tracker_dropout_guard_triggered": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_dropout_guard_triggered",
                    False,
                )
            ),
            "dynamic_obstacle_tracker_recovery_active": float(
                planner_diagnostics.get(
                    "dynamic_obstacle_tracker_recovery_active", False
                )
            ),
            "profile_planner_state_reference_ms": float(
                planner_diagnostics.get(
                    "profile_planner_state_reference_ms", 0.0
                )
            ),
            "profile_planner_prior_ms": float(
                planner_diagnostics.get("profile_planner_prior_ms", 0.0)
            ),
            "profile_mppi_sampling_ms": float(
                planner_diagnostics.get("profile_mppi_sampling_ms", 0.0)
            ),
            "profile_mppi_batch_rollout_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_batch_rollout_ms", 0.0
                )
            ),
            "profile_mppi_cost_ms": float(
                planner_diagnostics.get("profile_mppi_cost_ms", 0.0)
            ),
            "profile_mppi_weighting_update_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_weighting_update_ms", 0.0
                )
            ),
            "profile_mppi_final_rollout_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_final_rollout_ms", 0.0
                )
            ),
            "profile_mppi_solve_total_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_solve_total_ms", 0.0
                )
            ),
            "planner_cost_min": float(planner_diagnostics.get("cost_min", 0.0)),
            "planner_cost_mean": float(planner_diagnostics.get("cost_mean", 0.0)),
            "effective_sample_size": float(
                planner_diagnostics.get("effective_sample_size", 0.0)
            ),
            "optimizer_diagnostics_enabled": float(
                planner_diagnostics.get("optimizer_diagnostics_enabled", False)
            ),
            "optimizer_best_candidate_cost": float(
                planner_diagnostics.get("optimizer_best_candidate_cost", 0.0)
            ),
            "optimizer_selected_sequence_cost": float(
                planner_diagnostics.get("optimizer_selected_sequence_cost", 0.0)
            ),
            "optimizer_selected_cost_gap": float(
                planner_diagnostics.get("optimizer_selected_cost_gap", 0.0)
            ),
            "optimizer_best_first_v": float(
                planner_diagnostics.get("optimizer_best_first_v", 0.0)
            ),
            "optimizer_best_first_omega": float(
                planner_diagnostics.get("optimizer_best_first_omega", 0.0)
            ),
            "optimizer_selected_first_v": float(
                planner_diagnostics.get("optimizer_selected_first_v", 0.0)
            ),
            "optimizer_selected_first_omega": float(
                planner_diagnostics.get("optimizer_selected_first_omega", 0.0)
            ),
            "optimizer_first_action_cancellation_ratio": float(
                planner_diagnostics.get(
                    "optimizer_first_action_cancellation_ratio", 1.0
                )
            ),
            "optimizer_initial_proposal_first_v": float(
                planner_diagnostics.get("optimizer_initial_proposal_first_v", 0.0)
            ),
            "optimizer_initial_proposal_first_omega": float(
                planner_diagnostics.get(
                    "optimizer_initial_proposal_first_omega", 0.0
                )
            ),
            "terminal_value_raw_authority_mean": float(
                planner_diagnostics.get(
                    "terminal_value_raw_authority_mean", 0.0
                )
            ),
            "terminal_value_causal_dynamics_cap": float(
                planner_diagnostics.get(
                    "terminal_value_causal_dynamics_cap", 1.0
                )
            ),
            "terminal_value_causal_cap_active_fraction": float(
                planner_diagnostics.get(
                    "terminal_value_causal_cap_active_fraction", 0.0
                )
            ),
            "sample_saturation_fraction": float(
                planner_diagnostics.get("sample_saturation_fraction", 0.0)
            ),
            "path_boundary_candidate_filter_enabled": float(
                planner_diagnostics.get(
                    "path_boundary_candidate_filter_enabled", False
                )
            ),
            "path_boundary_candidate_feasible_fraction": float(
                planner_diagnostics.get(
                    "path_boundary_candidate_feasible_fraction", 1.0
                )
            ),
            "path_boundary_candidate_feasible_fraction_min": float(
                planner_diagnostics.get(
                    "path_boundary_candidate_feasible_fraction_min",
                    planner_diagnostics.get(
                        "path_boundary_candidate_feasible_fraction", 1.0
                    ),
                )
            ),
            "path_boundary_no_feasible_candidates": float(
                planner_diagnostics.get(
                    "path_boundary_no_feasible_candidates", False
                )
            ),
            "path_boundary_no_feasible_iteration_fraction": float(
                planner_diagnostics.get(
                    "path_boundary_no_feasible_iteration_fraction", 0.0
                )
            ),
            "path_boundary_weighted_update_feasible": float(
                planner_diagnostics.get(
                    "path_boundary_weighted_update_feasible", True
                )
            ),
            "path_boundary_final_min_margin": float(
                planner_diagnostics.get(
                    "path_boundary_final_min_margin", float("inf")
                )
            ),
            "path_boundary_fallback_used": float(
                planner_diagnostics.get("path_boundary_fallback_used", False)
            ),
            "path_boundary_fallback_candidate_index": int(
                planner_diagnostics.get(
                    "path_boundary_fallback_candidate_index", -1
                )
            ),
            "residual_support_confidence_mean": float(
                planner_diagnostics.get("residual_support_confidence_mean", 1.0)
            ),
            "residual_support_confidence_min": float(
                planner_diagnostics.get("residual_support_confidence_min", 1.0)
            ),
            "residual_support_reduced_fraction": float(
                planner_diagnostics.get("residual_support_reduced_fraction", 0.0)
            ),
            "residual_support_disabled_fraction": float(
                planner_diagnostics.get("residual_support_disabled_fraction", 0.0)
            ),
            "residual_reliability_enabled": float(
                planner_diagnostics.get("residual_reliability_enabled", False)
            ),
            "residual_reliability_alpha": float(
                planner_diagnostics.get("residual_reliability_alpha", 0.0)
            ),
            "residual_reliability_evidence_alpha": float(
                planner_diagnostics.get(
                    "residual_reliability_evidence_alpha", 0.0
                )
            ),
            "residual_reliability_context_alpha": float(
                planner_diagnostics.get(
                    "residual_reliability_context_alpha", 0.0
                )
            ),
            "residual_reliability_context_value": float(
                planner_diagnostics.get(
                    "residual_reliability_context_value", 0.0
                )
            ),
            "residual_reliability_samples": int(
                planner_diagnostics.get("residual_reliability_samples", 0)
            ),
            "residual_reliability_mean_improvement": float(
                planner_diagnostics.get(
                    "residual_reliability_mean_improvement", 0.0
                )
            ),
            "residual_reliability_lcb": float(
                planner_diagnostics.get("residual_reliability_lcb", 0.0)
            ),
            "residual_reliability_last_relative_improvement": float(
                planner_diagnostics.get(
                    "residual_reliability_last_relative_improvement", 0.0
                )
            ),
            "residual_reliability_nominal_error": float(
                planner_diagnostics.get("residual_reliability_nominal_error", 0.0)
            ),
            "residual_reliability_residual_error": float(
                planner_diagnostics.get("residual_reliability_residual_error", 0.0)
            ),
            "residual_stall_guard_enabled": float(
                planner_diagnostics.get(
                    "residual_stall_guard_enabled", False
                )
            ),
            "residual_stall_guard_authority": float(
                planner_diagnostics.get(
                    "residual_stall_guard_authority", 1.0
                )
            ),
            "residual_stall_guard_latched": float(
                planner_diagnostics.get(
                    "residual_stall_guard_latched", False
                )
            ),
            "residual_stall_guard_qualifying_steps": int(
                planner_diagnostics.get(
                    "residual_stall_guard_qualifying_steps", 0
                )
            ),
            "residual_stall_guard_latch_step": int(
                planner_diagnostics.get(
                    "residual_stall_guard_latch_step", -1
                )
            ),
            "residual_stall_guard_speed_mps": float(
                planner_diagnostics.get(
                    "residual_stall_guard_speed_mps", 0.0
                )
            ),
            "residual_stall_guard_previous_probability": float(
                planner_diagnostics.get(
                    "residual_stall_guard_previous_probability", 1.0
                )
            ),
            "residual_stall_guard_goal_distance_m": float(
                planner_diagnostics.get(
                    "residual_stall_guard_goal_distance_m", 0.0
                )
            ),
            "residual_safety_shield_enabled": float(
                planner_diagnostics.get(
                    "residual_safety_shield_enabled", False
                )
            ),
            "residual_safety_shield_accepted": float(
                planner_diagnostics.get(
                    "residual_safety_shield_accepted", False
                )
            ),
            "residual_safety_shield_fallback_used": float(
                planner_diagnostics.get(
                    "residual_safety_shield_fallback_used", False
                )
            ),
            "residual_safety_shield_selected_source": str(
                planner_diagnostics.get(
                    "residual_safety_shield_selected_source", "none"
                )
            ),
            "residual_safety_shield_nominal_baseline_maximum_probability": float(
                planner_diagnostics.get(
                    "residual_safety_shield_nominal_baseline_maximum_probability",
                    0.0,
                )
            ),
            "residual_safety_shield_candidate_nominal_maximum_probability": float(
                planner_diagnostics.get(
                    "residual_safety_shield_candidate_nominal_maximum_probability",
                    0.0,
                )
            ),
            "residual_safety_shield_candidate_residual_maximum_probability": float(
                planner_diagnostics.get(
                    "residual_safety_shield_candidate_residual_maximum_probability",
                    0.0,
                )
            ),
            "residual_safety_shield_maximum_position_deviation_m": float(
                planner_diagnostics.get(
                    "residual_safety_shield_maximum_position_deviation_m",
                    0.0,
                )
            ),
            "residual_safety_shield_nominal_progress_noninferior": float(
                planner_diagnostics.get(
                    "residual_safety_shield_nominal_progress_noninferior",
                    False,
                )
            ),
            "optimizer": str(planner_diagnostics.get("optimizer", "standard")),
            "rl_driven_total_rollouts": int(
                planner_diagnostics.get("rl_driven_total_rollouts", 0)
            ),
            "paper_total_rollouts": int(
                planner_diagnostics.get("paper_total_rollouts", 0)
            ),
            "paper_candidates_per_iteration": int(
                planner_diagnostics.get(
                    "paper_candidates_per_iteration", 0
                )
            ),
            "paper_standard_fallback_active": float(
                planner_diagnostics.get(
                    "paper_standard_fallback_active", False
                )
            ),
            "paper_standard_fallback_contract": str(
                planner_diagnostics.get(
                    "paper_standard_fallback_contract", "disabled"
                )
            ),
            "paper_guided_unique_sequences": int(
                planner_diagnostics.get(
                    "paper_guided_unique_sequences", 0
                )
            ),
            "paper_guided_reuses": int(
                planner_diagnostics.get("paper_guided_reuses", 0)
            ),
            "reliability_hss_enabled": float(
                planner_diagnostics.get("reliability_hss_enabled", False)
            ),
            "reliability_level": str(
                planner_diagnostics.get("reliability_level", "disabled")
            ),
            "reliability_authority": float(
                planner_diagnostics.get("reliability_authority", 1.0)
            ),
            "reliability_proposal_authority": float(
                planner_diagnostics.get(
                    "reliability_proposal_authority", 1.0
                )
            ),
            "reliability_requested_proposal_authority": float(
                planner_diagnostics.get(
                    "reliability_requested_proposal_authority",
                    planner_diagnostics.get(
                        "reliability_proposal_authority", 1.0
                    ),
                )
            ),
            "reliability_proposal_fallback_fraction": float(
                planner_diagnostics.get(
                    "reliability_proposal_fallback_fraction", 0.0
                )
            ),
            "reliability_proposal_advantage_gate_enabled": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_gate_enabled", False
                )
            ),
            "reliability_proposal_advantage_gate_mode": str(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_gate_mode", "disabled"
                )
            ),
            "reliability_proposal_advantage_gate_shadow": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_gate_shadow", False
                )
            ),
            "reliability_proposal_advantage_updated": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_updated", False
                )
            ),
            "reliability_proposal_advantage_disadvantage": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_disadvantage", False
                )
            ),
            "reliability_proposal_advantage_relative": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_relative", 0.0
                )
            ),
            "reliability_proposal_advantage_observations": int(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_observations", 0
                )
            ),
            "reliability_proposal_advantage_consecutive_disadvantages": int(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_consecutive_disadvantages",
                    0,
                )
            ),
            "reliability_proposal_advantage_latched": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_latched", False
                )
            ),
            "reliability_proposal_advantage_authority_applied": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_authority_applied", 1.0
                )
            ),
            "reliability_proposal_advantage_authority_next": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_authority_next", 1.0
                )
            ),
            "reliability_proposal_advantage_would_authority_next": float(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_would_authority_next", 1.0
                )
            ),
            "reliability_proposal_advantage_relative_disadvantage_margin": (
                float(planner_diagnostics.get(
                    "reliability_proposal_advantage_"
                    "relative_disadvantage_margin",
                    0.0,
                ))
            ),
            "reliability_proposal_advantage_patience": int(
                planner_diagnostics.get(
                    "reliability_proposal_advantage_patience", 0
                )
            ),
            "reliability_counterfactual_enabled": float(
                planner_diagnostics.get(
                    "reliability_counterfactual_enabled", False
                )
            ),
            "reliability_counterfactual_authority": float(
                planner_diagnostics.get(
                    "reliability_counterfactual_authority", 1.0
                )
            ),
            "reliability_counterfactual_advantage": float(
                planner_diagnostics.get(
                    "reliability_counterfactual_advantage", 0.0
                )
            ),
            "reliability_counterfactual_actor_progress": float(
                planner_diagnostics.get(
                    "reliability_counterfactual_actor_progress", 0.0
                )
            ),
            "reliability_counterfactual_baseline_progress": float(
                planner_diagnostics.get(
                    "reliability_counterfactual_baseline_progress", 0.0
                )
            ),
            "reliability_dynamics_confidence": float(
                planner_diagnostics.get("dynamics_confidence", 1.0)
            ),
            "reliability_dynamics_routing_mode": str(
                planner_diagnostics.get(
                    "dynamics_routing_mode", "trust_weighted"
                )
            ),
            "reliability_model_routing_factor": float(
                planner_diagnostics.get("model_routing_factor", 1.0)
            ),
            "reliability_actor_confidence": float(
                planner_diagnostics.get("actor_confidence", 1.0)
            ),
            "reliability_actor_competence_confidence": float(
                planner_diagnostics.get(
                    "actor_competence_confidence", 1.0
                )
            ),
            "reliability_actor_competence_raw_confidence": float(
                planner_diagnostics.get(
                    "actor_competence_raw_confidence", 1.0
                )
            ),
            "reliability_actor_competence_mapped_confidence": float(
                planner_diagnostics.get(
                    "actor_competence_mapped_confidence", 1.0
                )
            ),
            "reliability_actor_competence_guided_yield": float(
                planner_diagnostics.get(
                    "actor_competence_guided_yield", 0.0
                )
            ),
            "reliability_actor_competence_gaussian_yield": float(
                planner_diagnostics.get(
                    "actor_competence_gaussian_yield", 0.0
                )
            ),
            "reliability_ensemble_disagreement_max": float(
                planner_diagnostics.get(
                    "ensemble_disagreement_max", 0.0
                )
            ),
            "reliability_innovation_error_ema": float(
                planner_diagnostics.get("innovation_error_ema", 0.0)
            ),
            "reliability_guided_fraction_applied": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_applied", 0.30
                )
            ),
            "reliability_guided_fraction_next": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_next", 0.30
                )
            ),
            "reliability_guided_fraction_raw_applied": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_raw_applied", 0.30
                )
            ),
            "reliability_guided_fraction_raw_next": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_raw_next", 0.30
                )
            ),
            "terminal_guidance_floor_enabled": float(
                planner_diagnostics.get(
                    "terminal_guidance_floor_enabled", False
                )
            ),
            "terminal_guidance_floor_active": float(
                planner_diagnostics.get(
                    "terminal_guidance_floor_active", False
                )
            ),
            "terminal_guidance_terminal_phase": float(
                planner_diagnostics.get(
                    "terminal_guidance_terminal_phase", False
                )
            ),
            "terminal_guidance_distance": float(
                planner_diagnostics.get(
                    "terminal_guidance_distance", float("inf")
                )
            ),
            "terminal_guidance_radius": float(
                planner_diagnostics.get("terminal_guidance_radius", 0.0)
            ),
            "terminal_guided_fraction_floor": float(
                planner_diagnostics.get(
                    "terminal_guided_fraction_floor", 0.0
                )
            ),
            "completion_handover_enabled": float(
                planner_diagnostics.get(
                    "completion_handover_enabled", False
                )
            ),
            "completion_handover_authority": float(
                planner_diagnostics.get(
                    "completion_handover_authority", 1.0
                )
            ),
            "completion_handover_distance": float(
                planner_diagnostics.get(
                    "completion_handover_distance", float("inf")
                )
            ),
            "terminal_value_completion_authority": float(
                planner_diagnostics.get(
                    "terminal_value_completion_authority", 1.0
                )
            ),
            "paper_guided_elite_count": int(
                planner_diagnostics.get("paper_guided_elite_count", 0)
            ),
            "paper_gaussian_elite_count": int(
                planner_diagnostics.get("paper_gaussian_elite_count", 0)
            ),
            "paper_guided_opportunity_count": int(
                planner_diagnostics.get("paper_guided_opportunity_count", 0)
            ),
            "paper_gaussian_opportunity_count": int(
                planner_diagnostics.get("paper_gaussian_opportunity_count", 0)
            ),
            "paper_same_cycle_guided_cost_filter_enabled": float(
                planner_diagnostics.get(
                    "paper_same_cycle_guided_cost_filter_enabled", False
                )
            ),
            "paper_same_cycle_gaussian_actor_isolated": float(
                planner_diagnostics.get(
                    "paper_same_cycle_gaussian_actor_isolated", False
                )
            ),
            "paper_same_cycle_guided_cost_filter_iterations": int(
                planner_diagnostics.get(
                    "paper_same_cycle_guided_cost_filter_iterations", 0
                )
            ),
            "paper_same_cycle_guided_filtered_candidates": int(
                planner_diagnostics.get(
                    "paper_same_cycle_guided_filtered_candidates", 0
                )
            ),
            "paper_same_cycle_guided_relative_margin": float(
                planner_diagnostics.get(
                    "paper_same_cycle_guided_relative_margin", 0.0
                )
            ),
            "paper_guided_cost_observed": float(
                planner_diagnostics.get("paper_guided_cost_observed", False)
            ),
            "paper_gaussian_cost_observed": float(
                planner_diagnostics.get("paper_gaussian_cost_observed", False)
            ),
            "paper_guided_cost_min": float(
                planner_diagnostics.get("paper_guided_cost_min", 0.0)
            ),
            "paper_gaussian_cost_min": float(
                planner_diagnostics.get("paper_gaussian_cost_min", 0.0)
            ),
            "paper_guided_cost_mean": float(
                planner_diagnostics.get("paper_guided_cost_mean", 0.0)
            ),
            "paper_gaussian_cost_mean": float(
                planner_diagnostics.get("paper_gaussian_cost_mean", 0.0)
            ),
            "paper_guided_cost_p50": float(
                planner_diagnostics.get("paper_guided_cost_p50", 0.0)
            ),
            "paper_gaussian_cost_p50": float(
                planner_diagnostics.get("paper_gaussian_cost_p50", 0.0)
            ),
            "paper_guided_minus_gaussian_cost_min": float(
                planner_diagnostics.get(
                    "paper_guided_minus_gaussian_cost_min", 0.0
                )
            ),
            "paper_guided_minus_gaussian_cost_mean": float(
                planner_diagnostics.get(
                    "paper_guided_minus_gaussian_cost_mean", 0.0
                )
            ),
            "paper_guided_feasible_fraction": float(
                planner_diagnostics.get(
                    "paper_guided_feasible_fraction", 0.0
                )
            ),
            "paper_gaussian_feasible_fraction": float(
                planner_diagnostics.get(
                    "paper_gaussian_feasible_fraction", 0.0
                )
            ),
            "paper_actor_first_v": float(
                planner_diagnostics.get("paper_actor_first_v", 0.0)
            ),
            "paper_actor_first_omega": float(
                planner_diagnostics.get("paper_actor_first_omega", 0.0)
            ),
            "paper_baseline_first_v": float(
                planner_diagnostics.get("paper_baseline_first_v", 0.0)
            ),
            "paper_baseline_first_omega": float(
                planner_diagnostics.get("paper_baseline_first_omega", 0.0)
            ),
            "paper_actor_baseline_mean_abs_delta": float(
                planner_diagnostics.get(
                    "paper_actor_baseline_mean_abs_delta", 0.0
                )
            ),
            "paper_actor_baseline_first_action_l2_delta": float(
                planner_diagnostics.get(
                    "paper_actor_baseline_first_action_l2_delta", 0.0
                )
            ),
            "rl_source_samples": int(
                planner_diagnostics.get("rl_source_samples", 0)
            ),
            "shifted_source_samples": int(
                planner_diagnostics.get("shifted_source_samples", 0)
            ),
            "base_source_samples": int(
                planner_diagnostics.get("base_source_samples", 0)
            ),
            "rl_elite_count": int(
                planner_diagnostics.get("rl_elite_count", 0)
            ),
            "shifted_elite_count": int(
                planner_diagnostics.get("shifted_elite_count", 0)
            ),
            "base_elite_count": int(
                planner_diagnostics.get("base_elite_count", 0)
            ),
            "rl_elite_fraction": float(
                planner_diagnostics.get("rl_elite_fraction", 0.0)
            ),
            "covariance_scale_mean": float(
                planner_diagnostics.get("covariance_scale_mean", 1.0)
            ),
            "terminal_value_enabled": float(
                planner_diagnostics.get("terminal_value_enabled", False)
            ),
            "terminal_value_weight": float(
                planner_diagnostics.get("terminal_value_weight", 0.0)
            ),
            "terminal_value_conservative_enabled": float(
                planner_diagnostics.get(
                    "terminal_value_conservative_enabled", False
                )
            ),
            "terminal_value_authority_mean": float(
                planner_diagnostics.get(
                    "terminal_value_authority_mean", 1.0
                )
            ),
            "terminal_value_authority_min": float(
                planner_diagnostics.get(
                    "terminal_value_authority_min", 1.0
                )
            ),
            "terminal_value_authority_max": float(
                planner_diagnostics.get(
                    "terminal_value_authority_max", 1.0
                )
            ),
            "terminal_value_dynamics_confidence_mean": float(
                planner_diagnostics.get(
                    "terminal_value_dynamics_confidence_mean", 1.0
                )
            ),
            "terminal_value_critic_confidence_mean": float(
                planner_diagnostics.get(
                    "terminal_value_critic_confidence_mean", 1.0
                )
            ),
            "terminal_value_uncertainty_mean": float(
                planner_diagnostics.get(
                    "terminal_value_uncertainty_mean", 0.0
                )
            ),
            "terminal_value_uncertainty_cost_mean": float(
                planner_diagnostics.get(
                    "terminal_value_uncertainty_cost_mean", 0.0
                )
            ),
            "terminal_q_mean": float(
                planner_diagnostics.get("terminal_q_mean", 0.0)
            ),
            "terminal_q_disagreement_mean": float(
                planner_diagnostics.get(
                    "terminal_q_disagreement_mean", 0.0
                )
            ),
            "residual_policy_context_enabled": float(
                planner_diagnostics.get(
                    "residual_policy_context_enabled", False
                )
            ),
            "residual_policy_predicted_abs_mean": float(
                planner_diagnostics.get(
                    "residual_policy_predicted_abs_mean", 0.0
                )
            ),
            "residual_policy_predicted_abs_max": float(
                planner_diagnostics.get(
                    "residual_policy_predicted_abs_max", 0.0
                )
            ),
            "residual_policy_innovation_abs_mean": float(
                planner_diagnostics.get(
                    "residual_policy_innovation_abs_mean", 0.0
                )
            ),
            "residual_policy_innovation_abs_max": float(
                planner_diagnostics.get(
                    "residual_policy_innovation_abs_max", 0.0
                )
            ),
            "residual_policy_disagreement_mean": float(
                planner_diagnostics.get(
                    "residual_policy_disagreement_mean", 0.0
                )
            ),
            "residual_policy_support_mean": float(
                planner_diagnostics.get(
                    "residual_policy_support_mean", 0.0
                )
            ),
            "residual_policy_innovation_valid_fraction": float(
                planner_diagnostics.get(
                    "residual_policy_innovation_valid_fraction", 0.0
                )
            ),
            "residual_policy_authority_enabled": float(
                planner_diagnostics.get(
                    "residual_policy_authority_enabled", False
                )
            ),
            "residual_policy_authority_mean": float(
                planner_diagnostics.get(
                    "residual_policy_authority_mean", 0.0
                )
            ),
            "residual_policy_authority_max": float(
                planner_diagnostics.get(
                    "residual_policy_authority_max", 0.0
                )
            ),
            "anytime_selected_samples": int(
                planner_diagnostics.get("anytime_selected_samples", 0)
            ),
            "anytime_add_samples": float(
                planner_diagnostics.get("anytime_add_samples", False)
            ),
            "anytime_predicted_advantage": float(
                planner_diagnostics.get("anytime_predicted_advantage", 0.0)
            ),
            "anytime_confidence_width": float(
                planner_diagnostics.get("anytime_confidence_width", 0.0)
            ),
            "anytime_score": float(
                planner_diagnostics.get("anytime_score", 0.0)
            ),
            "anytime_decision_refreshed": float(
                planner_diagnostics.get("anytime_decision_refreshed", False)
            ),
            "prior_type": str(prior.get("type", "unknown")),
            "rl_gate_mode": str(prior.get("gate_mode", "disabled")),
            "rl_gate_alpha": float(prior.get("gate_alpha", 0.0)),
            "rl_policy_mode": str(prior.get("policy_mode", "direct")),
            "rl_correction_gate_alpha": float(
                prior.get("correction_gate_alpha", 0.0)
            ),
            "rl_base_action_abs_mean": float(
                prior.get("base_action_abs_mean", 0.0)
            ),
            "rl_unit_correction_abs_mean": float(
                prior.get("unit_correction_abs_mean", 0.0)
            ),
            "rl_applied_correction_abs_mean": float(
                prior.get("applied_correction_abs_mean", 0.0)
            ),
            "rl_applied_correction_abs_max": float(
                prior.get("applied_correction_abs_max", 0.0)
            ),
            "rl_raw_applied_correction_abs_mean": float(
                prior.get("raw_applied_correction_abs_mean", 0.0)
            ),
            "rl_correction_advantage_gate_mode": str(
                prior.get("correction_advantage_gate_mode", "none")
            ),
            "rl_correction_advantage_critic_source": str(
                prior.get("correction_advantage_critic_source", "online")
            ),
            "rl_correction_advantage_threshold": float(
                prior.get("correction_advantage_threshold", 0.0)
            ),
            "rl_correction_advantage_uncertainty_multiplier": float(
                prior.get(
                    "correction_advantage_uncertainty_multiplier", 1.0
                )
            ),
            "rl_correction_advantage_gate_alpha": float(
                prior.get("correction_advantage_gate_alpha", 1.0)
            ),
            "rl_correction_support_gate_enabled": float(
                prior.get("correction_support_gate_enabled", False)
            ),
            "rl_correction_support_confidence": float(
                prior.get("correction_support_confidence", 1.0)
            ),
            "rl_correction_effective_gate_alpha": float(
                prior.get("correction_effective_gate_alpha", 1.0)
            ),
            "rl_unselected_critic_diagnostics_computed": float(
                prior.get(
                    "unselected_critic_diagnostics_computed", False
                )
            ),
            "rl_selected_consensus_lcb": float(
                prior.get("selected_consensus_lcb", 0.0)
            ),
            "rl_online_q1_base": float(prior.get("online_q1_base", 0.0)),
            "rl_online_q2_base": float(prior.get("online_q2_base", 0.0)),
            "rl_online_q1_candidate": float(
                prior.get("online_q1_candidate", 0.0)
            ),
            "rl_online_q2_candidate": float(
                prior.get("online_q2_candidate", 0.0)
            ),
            "rl_target_q1_base": float(prior.get("target_q1_base", 0.0)),
            "rl_target_q2_base": float(prior.get("target_q2_base", 0.0)),
            "rl_target_q1_candidate": float(
                prior.get("target_q1_candidate", 0.0)
            ),
            "rl_target_q2_candidate": float(
                prior.get("target_q2_candidate", 0.0)
            ),
            "rl_online_advantage_q1": float(
                prior.get("online_advantage_q1", 0.0)
            ),
            "rl_online_advantage_q2": float(
                prior.get("online_advantage_q2", 0.0)
            ),
            "rl_online_conservative_advantage": float(
                prior.get("online_conservative_advantage", 0.0)
            ),
            "rl_target_advantage_q1": float(
                prior.get("target_advantage_q1", 0.0)
            ),
            "rl_target_advantage_q2": float(
                prior.get("target_advantage_q2", 0.0)
            ),
            "rl_target_conservative_advantage": float(
                prior.get("target_conservative_advantage", 0.0)
            ),
            "rl_ood_score": float(prior.get("ood_score", 0.0)),
            "rl_scene_complexity_score": float(
                prior.get("scene_complexity_score", 0.0)
            ),
            "rl_scene_complexity_front_proximity": float(
                prior.get("scene_complexity_front_proximity", 0.0)
            ),
            "rl_scene_complexity_constriction": float(
                prior.get("scene_complexity_constriction", 0.0)
            ),
            "rl_scene_complexity_density": float(
                prior.get("scene_complexity_density", 0.0)
            ),
            "rl_scene_complexity_front_clearance_m": float(
                prior.get("scene_complexity_front_clearance_m", 0.0)
            ),
            "rl_scene_complexity_left_clearance_m": float(
                prior.get("scene_complexity_left_clearance_m", 0.0)
            ),
            "rl_scene_complexity_right_clearance_m": float(
                prior.get("scene_complexity_right_clearance_m", 0.0)
            ),
            "rl_scene_complexity_near_obstacle_fraction": float(
                prior.get("scene_complexity_near_obstacle_fraction", 0.0)
            ),
            "rl_scene_complexity_scan_valid": float(
                prior.get("scene_complexity_scan_valid", False)
            ),
            "rl_temporal_closing_gate_alpha": float(
                prior.get("temporal_closing_gate_alpha", 0.0)
            ),
            "rl_temporal_closing_rate_mps": float(
                prior.get("temporal_closing_rate_mps", 0.0)
            ),
            "rl_temporal_closing_held": float(
                prior.get("temporal_closing_held", False)
            ),
            "rl_hazard_activation": float(
                prior.get("hazard_activation", 0.0)
            ),
            "rl_competence_confidence": float(
                prior.get("competence_confidence", 0.0)
            ),
            "rl_baseline_progress_m": float(
                prior.get("baseline_progress_m", 0.0)
            ),
            "rl_baseline_progress_window_s": float(
                prior.get("baseline_progress_window_s", 0.0)
            ),
            "rl_baseline_progress_gate_ready": float(
                prior.get("baseline_progress_gate_ready", False)
            ),
            "rl_baseline_stagnation_activation": float(
                prior.get("baseline_stagnation_activation", 0.0)
            ),
            "rl_baseline_stagnation_held": float(
                prior.get("baseline_stagnation_held", False)
            ),
            "rl_learned_inference_skipped": float(
                prior.get("learned_inference_skipped", False)
            ),
            "profile_prior_fallback_ms": float(
                prior.get("profile_prior_fallback_ms", 0.0)
            ),
            "profile_prior_encode_normalize_ms": float(
                prior.get("profile_prior_encode_normalize_ms", 0.0)
            ),
            "profile_prior_gate_features_ms": float(
                prior.get("profile_prior_gate_features_ms", 0.0)
            ),
            "profile_prior_actor_ms": float(
                prior.get("profile_prior_actor_ms", 0.0)
            ),
            "profile_prior_advantage_ms": float(
                prior.get("profile_prior_advantage_ms", 0.0)
            ),
            "profile_prior_decoder_ms": float(
                prior.get("profile_prior_decoder_ms", 0.0)
            ),
            "profile_prior_outer_gate_ms": float(
                prior.get("profile_prior_outer_gate_ms", 0.0)
            ),
            "profile_prior_total_ms": float(
                prior.get("profile_prior_total_ms", 0.0)
            ),
            "rl_critic_disagreement": float(
                prior.get("critic_disagreement", 0.0)
            ),
            "rl_exploration_activation": float(
                prior.get("exploration_activation", 0.0)
            ),
            "rl_exploration_latch_alpha": float(
                prior.get("exploration_latch_alpha", 0.0)
            ),
            "rl_subgoal_distance": float(
                prior.get("subgoal_distance", 0.0)
            ),
            "rl_subgoal_bearing": float(
                prior.get("subgoal_bearing", 0.0)
            ),
            "rl_subgoal_x_body": float(
                prior.get("subgoal_x_body", 0.0)
            ),
            "rl_subgoal_y_body": float(
                prior.get("subgoal_y_body", 0.0)
            ),
        }
        if self._tracking_monitor is not None:
            record.update(self._tracking_monitor.update(
                truth.pose.as_array(), truth.timestamp
            ).to_dict())
        self.records.append(record)

    def summary(self, termination_reason=None):
        if not self.records:
            return {
                "steps": 0,
                "success": False,
                "termination_reason": termination_reason or "no_steps",
            }
        values = self.records
        path_length = sum(
            math.hypot(values[index]["x"] - values[index - 1]["x"], values[index]["y"] - values[index - 1]["y"])
            for index in range(1, len(values))
        )
        controls = np.asarray([(row["executed_v"], row["executed_omega"]) for row in values])
        jerk = np.diff(controls, axis=0) if len(values) > 1 else np.zeros((0, 2))
        applied_controls = np.asarray(
            [(row["applied_v"], row["applied_omega"]) for row in values]
        )
        applied_jerk = (
            np.diff(applied_controls, axis=0)
            if len(values) > 1
            else np.zeros((0, 2))
        )
        compute = np.asarray([row["planner_compute_ms"] for row in values])
        effective_samples = np.asarray([row["effective_sample_size"] for row in values])
        finite_clearance = [row["clearance"] for row in values if math.isfinite(row["clearance"])]
        finite_dynamic_distance = [
            row["nearest_dynamic_obstacle_center_distance"]
            for row in values
            if math.isfinite(row["nearest_dynamic_obstacle_center_distance"])
        ]
        cross_track = self._cross_track_errors(values)
        collision = bool(any(row["collision"] for row in values))
        success = bool(values[-1]["goal_distance"] <= self.goal_tolerance and not collision)
        deadline_ms = 1000.0 * self.control_dt if self.control_dt > 0.0 else None
        deadline_misses = int(np.sum(compute > deadline_ms)) if deadline_ms is not None else 0
        stuck_steps = sum(
            abs(row["v"]) < 0.02 and row["goal_distance"] > self.goal_tolerance
            for row in values
        )
        spin_steps = sum(abs(row["v"]) < 0.03 and abs(row["omega"]) > 0.5 for row in values)
        safety_reason_counts = {}
        for row in values:
            if row["safety_override"]:
                reason = row["safety_reason"]
                safety_reason_counts[reason] = safety_reason_counts.get(reason, 0) + 1
        result = {
            "steps": len(values),
            "success": success,
            "termination_reason": termination_reason or ("goal_reached" if success else "unknown"),
            "time_to_goal_s": float(values[-1]["time"]) if success else None,
            "collision": collision,
            "final_goal_distance": values[-1]["goal_distance"],
            "trajectory_length": path_length,
            "cross_track_rmse": (
                None
                if cross_track is None
                else float(np.sqrt(np.mean(np.square(cross_track))))
            ),
            "cross_track_mean": (
                None if cross_track is None else float(np.mean(cross_track))
            ),
            "cross_track_max": (
                None if cross_track is None else float(np.max(cross_track))
            ),
            "minimum_clearance": min(finite_clearance) if finite_clearance else None,
            "dynamic_obstacle_count": int(max(
                row.get("dynamic_obstacle_count", 0) for row in values
            )),
            "minimum_dynamic_obstacle_center_distance": (
                min(finite_dynamic_distance) if finite_dynamic_distance else None
            ),
            "mean_abs_omega": float(np.mean(np.abs(controls[:, 1]))),
            "control_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0,
            "applied_control_jerk": (
                float(np.mean(np.linalg.norm(applied_jerk, axis=1)))
                if applied_jerk.size
                else 0.0
            ),
            "stuck_steps": int(stuck_steps),
            "spin_steps": int(spin_steps),
            "mean_slip_ratio": float(np.mean([row["slip_ratio"] for row in values])),
            "safety_interventions": int(sum(row["safety_override"] for row in values)),
            "safety_reason_counts": dict(sorted(safety_reason_counts.items())),
            "dynamic_recovery_active_steps": int(sum(
                row.get("dynamic_recovery_active", 0.0)
                for row in values
            )),
            "dynamic_recovery_align_steps": int(sum(
                row.get("dynamic_recovery_mode") == "align"
                for row in values
            )),
            "dynamic_recovery_advance_steps": int(sum(
                row.get("dynamic_recovery_mode") == "advance"
                for row in values
            )),
            "dynamic_recovery_abort_steps": int(sum(
                row.get("dynamic_recovery_mode") == "aborted"
                for row in values
            )),
            "dynamic_escape_allowed_steps": int(sum(
                row.get("dynamic_escape_allowed", 0.0) for row in values
            )),
            "dynamic_escape_held_steps": int(sum(
                row.get("dynamic_escape_held", 0.0) for row in values
            )),
            "dynamic_escape_probability_mass_enabled_fraction": float(
                np.mean([
                    row.get(
                        "dynamic_escape_probability_mass_enabled", 0.0
                    )
                    for row in values
                ])
            ),
            "dynamic_escape_probability_mass_fallback_steps": int(sum(
                row.get("dynamic_escape_probability_mass_fallback", 0.0)
                for row in values
            )),
            "dynamic_escape_vetted_planner_control_steps": int(sum(
                row.get("dynamic_escape_vetted_planner_control", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_active_fallback_steps": int(sum(
                row.get("probabilistic_obstacle_active_fallback_used", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_mass_fallback_steps": int(sum(
                row.get("probabilistic_obstacle_active_fallback_kind")
                == "minimum_accumulated_risk_active_candidate"
                for row in values
            )),
            "probabilistic_obstacle_emergency_candidate_selected_steps": int(
                sum(
                    row.get(
                        "probabilistic_obstacle_emergency_candidate_selected",
                        0.0,
                    )
                    for row in values
                )
            ),
            "probabilistic_obstacle_temporal_emergency_triggered_steps": int(
                sum(
                    row.get(
                        "probabilistic_obstacle_temporal_emergency_triggered",
                        0.0,
                    )
                    for row in values
                )
            ),
            "probabilistic_obstacle_temporal_emergency_vetted_steps": int(
                sum(
                    row.get(
                        "probabilistic_obstacle_temporal_emergency_vetted",
                        0.0,
                    )
                    for row in values
                )
            ),
            "probabilistic_obstacle_traversal_window_safe_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_window_safe", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_forecast_sufficient_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_forecast_sufficient", 0.0
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_active_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_commit_active", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_started_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_commit_started", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_completed_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_commit_completed", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_cancelled_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_commit_cancelled", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_closing_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_closing",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_midpoint_guard_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_midpoint_guard",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_rejected_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_rejected",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_prealign_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_prealign_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_terminal_release_active_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_terminal_release_active",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_release_condition_met_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_release_condition_met",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_applied_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_applied",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_candidate_count_max": int(max(
                (
                    row.get(
                        "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_candidate_count",
                        0,
                    )
                    for row in values
                ),
                default=0,
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count_max": int(max(
                (
                    row.get(
                        "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count",
                        0,
                    )
                    for row in values
                ),
                default=0,
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_selected_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_selected",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_admitted_count_max": int(max(
                (
                    row.get(
                        "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_admitted_count",
                        0,
                    )
                    for row in values
                ),
                default=0,
            )),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_selected_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_selected",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_waiting_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_waiting",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_admission_released_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_admission_released",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_preserved_for_nearest_safe_exit_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_preserved_for_nearest_safe_exit",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_retreat_requested_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_retreat_requested", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_retreat_temporal_lattice_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_retreat_temporal_lattice_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_retreat_completed_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_retreat_completed", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count_max": int(max(
                (
                    row.get(
                        "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count",
                        0,
                    )
                    for row in values
                ),
                default=0,
            )),
            "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_latched_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_escape_latched",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_reused_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_post_center_temporal_escape_reused",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_pending_steps": int(sum(
                row.get("probabilistic_obstacle_traversal_rearm_pending", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_no_crossing_safe_streak_max": int(max(
                (
                    row.get(
                        "probabilistic_obstacle_traversal_rearm_no_crossing_safe_streak",
                        0.0,
                    )
                    for row in values
                ),
                default=0.0,
            )),
            "probabilistic_obstacle_traversal_rearm_released_by_no_crossing_clearance_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_released_by_no_crossing_clearance",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_active_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_active",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_safe_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_safe",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_staging_approach_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_staging_approach_requested",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_staging_approach_safe_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_staging_approach_safe",
                    0.0,
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk",
                    0.0,
                )
                > 0.5
                for row in values
            )),
            "probabilistic_obstacle_traversal_rearm_hold_overridden_by_temporal_closing_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_rearm_hold_overridden_by_temporal_closing",
                    0.0,
                )
                > 0.5
                for row in values
            )),
            "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested",
                    0.0,
                )
                > 0.5
                for row in values
            )),
            "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered",
                    0.0,
                )
                > 0.5
                for row in values
            )),
            "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered",
                    0.0,
                )
                > 0.5
                for row in values
            )),
            "probabilistic_obstacle_traversal_candidate_selected_steps": int(sum(
                row.get(
                    "probabilistic_obstacle_traversal_candidate_selected", 0.0
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_maximum_probability_max": float(max(
                row.get(
                    "probabilistic_obstacle_traversal_maximum_probability", 0.0
                )
                for row in values
            )),
            "probabilistic_obstacle_traversal_probability_mass_min": float(min(
                row.get("probabilistic_obstacle_traversal_probability_mass", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_traversal_required_steps_max": int(max(
                row.get("probabilistic_obstacle_traversal_required_steps", 0)
                for row in values
            )),
            "probabilistic_obstacle_pareto_forward_commit_steps": int(
                sum(
                    row.get(
                        "probabilistic_obstacle_pareto_forward_commit_applied",
                        0.0,
                    )
                    for row in values
                )
            ),
            "probabilistic_obstacle_low_risk_forward_commit_steps": int(
                sum(
                    row.get(
                        "probabilistic_obstacle_low_risk_forward_commit_applied",
                        0.0,
                    )
                    for row in values
                )
            ),
            "planner_temporal_escape_active_steps": int(sum(
                row.get("planner_temporal_escape_active", 0.0)
                for row in values
            )),
            "probabilistic_obstacle_stopping_feasibility_enabled_fraction": (
                float(np.mean([
                    row.get(
                        "probabilistic_obstacle_stopping_feasibility_enabled",
                        0.0,
                    )
                    for row in values
                ]))
            ),
            "temporal_scan_valid_fraction": float(np.mean([
                row.get("temporal_scan_valid", 0.0) for row in values
            ])),
            "temporal_scan_risk_alpha_mean": float(np.mean([
                row.get("temporal_scan_risk_alpha", 0.0) for row in values
            ])),
            "temporal_scan_risk_alpha_max": float(np.max([
                row.get("temporal_scan_risk_alpha", 0.0) for row in values
            ])),
            "temporal_scan_closing_rate_mps_max": float(np.max([
                row.get("temporal_scan_closing_rate_mps", 0.0)
                for row in values
            ])),
            "temporal_scan_rejected_jump_fraction_max": float(np.max([
                row.get("temporal_scan_rejected_jump_fraction", 0.0)
                for row in values
            ])),
            "temporal_safety_interventions": int(sum(
                row.get("safety_override", 0.0)
                and str(row.get("safety_reason", "")).startswith("temporal_")
                for row in values
            )),
            "planner_compute_ms_mean": float(compute.mean()),
            "planner_compute_ms_p50": float(np.percentile(compute, 50)),
            "planner_compute_ms_p95": float(np.percentile(compute, 95)),
            "planner_compute_ms_p99": float(np.percentile(compute, 99)),
            "planner_compute_ms_max": float(compute.max()),
            "residual_support_confidence_mean": float(np.mean([
                row.get("residual_support_confidence_mean", 1.0) for row in values
            ])),
            "residual_support_confidence_min": float(np.min([
                row.get("residual_support_confidence_min", 1.0) for row in values
            ])),
            "residual_support_reduced_fraction_mean": float(np.mean([
                row.get("residual_support_reduced_fraction", 0.0) for row in values
            ])),
            "residual_support_disabled_fraction_mean": float(np.mean([
                row.get("residual_support_disabled_fraction", 0.0) for row in values
            ])),
            "residual_reliability_enabled_fraction": float(np.mean([
                row.get("residual_reliability_enabled", 0.0)
                for row in values
            ])),
            "residual_reliability_alpha_mean": float(np.mean([
                row.get("residual_reliability_alpha", 0.0)
                for row in values
            ])),
            "residual_reliability_alpha_max": float(np.max([
                row.get("residual_reliability_alpha", 0.0)
                for row in values
            ])),
            "residual_reliability_samples_max": int(np.max([
                row.get("residual_reliability_samples", 0)
                for row in values
            ])),
            "residual_stall_guard_enabled_fraction": float(np.mean([
                row.get("residual_stall_guard_enabled", 0.0)
                for row in values
            ])),
            "residual_stall_guard_latched_fraction": float(np.mean([
                row.get("residual_stall_guard_latched", 0.0)
                for row in values
            ])),
            "residual_stall_guard_latched": bool(any(
                row.get("residual_stall_guard_latched", 0.0)
                for row in values
            )),
            "residual_stall_guard_latch_step": int(next(
                (
                    row.get("residual_stall_guard_latch_step", -1)
                    for row in values
                    if row.get("residual_stall_guard_latched", 0.0)
                ),
                -1,
            )),
            "residual_stall_guard_minimum_authority": float(np.min([
                row.get("residual_stall_guard_authority", 1.0)
                for row in values
            ])),
            "residual_safety_shield_enabled_fraction": float(np.mean([
                row.get("residual_safety_shield_enabled", 0.0)
                for row in values
            ])),
            "residual_safety_shield_acceptance_fraction": float(np.mean([
                row.get("residual_safety_shield_accepted", 0.0)
                for row in values
            ])),
            "residual_safety_shield_fallback_fraction": float(np.mean([
                row.get("residual_safety_shield_fallback_used", 0.0)
                for row in values
            ])),
            "residual_safety_shield_maximum_position_deviation_m": float(
                np.max([
                    row.get(
                        "residual_safety_shield_maximum_position_deviation_m",
                        0.0,
                    )
                    for row in values
                ])
            ),
            "profile_planner_state_reference_ms_mean": float(np.mean([
                row.get("profile_planner_state_reference_ms", 0.0)
                for row in values
            ])),
            "profile_planner_prior_ms_mean": float(np.mean([
                row.get("profile_planner_prior_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_sampling_ms_mean": float(np.mean([
                row.get("profile_mppi_sampling_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_batch_rollout_ms_mean": float(np.mean([
                row.get("profile_mppi_batch_rollout_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_cost_ms_mean": float(np.mean([
                row.get("profile_mppi_cost_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_weighting_update_ms_mean": float(np.mean([
                row.get("profile_mppi_weighting_update_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_final_rollout_ms_mean": float(np.mean([
                row.get("profile_mppi_final_rollout_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_solve_total_ms_mean": float(np.mean([
                row.get("profile_mppi_solve_total_ms", 0.0)
                for row in values
            ])),
            "planner_deadline_ms": deadline_ms,
            "planner_deadline_misses": deadline_misses,
            "planner_deadline_miss_rate": (
                float(deadline_misses / len(values)) if deadline_ms is not None else None
            ),
            "effective_sample_size_mean": float(effective_samples.mean()),
            "effective_sample_size_min": float(effective_samples.min()),
            "sample_saturation_fraction_mean": float(
                np.mean([row["sample_saturation_fraction"] for row in values])
            ),
            "path_boundary_candidate_filter_enabled_fraction": float(np.mean([
                row.get("path_boundary_candidate_filter_enabled", False)
                for row in values
            ])),
            "path_boundary_candidate_feasible_fraction_mean": float(np.mean([
                row.get("path_boundary_candidate_feasible_fraction", 1.0)
                for row in values
            ])),
            "path_boundary_candidate_feasible_fraction_min": float(np.min([
                row.get(
                    "path_boundary_candidate_feasible_fraction_min",
                    row.get("path_boundary_candidate_feasible_fraction", 1.0),
                )
                for row in values
            ])),
            "path_boundary_no_feasible_decision_fraction": float(np.mean([
                row.get("path_boundary_no_feasible_candidates", False)
                for row in values
            ])),
            "path_boundary_no_feasible_iteration_fraction_mean": float(np.mean([
                row.get("path_boundary_no_feasible_iteration_fraction", 0.0)
                for row in values
            ])),
            "path_boundary_weighted_update_infeasible_fraction": float(np.mean([
                not row.get("path_boundary_weighted_update_feasible", True)
                for row in values
            ])),
            "path_boundary_fallback_steps": int(sum(
                row.get("path_boundary_fallback_used", False)
                for row in values
            )),
            "path_boundary_fallback_fraction": float(np.mean([
                row.get("path_boundary_fallback_used", False)
                for row in values
            ])),
            "path_boundary_final_min_margin_min": float(np.min([
                row.get("path_boundary_final_min_margin", 0.0)
                for row in values
            ])),
            "optimizer": str(values[-1].get("optimizer", "standard")),
            "rl_driven_total_rollouts_mean": float(np.mean([
                row.get("rl_driven_total_rollouts", 0) for row in values
            ])),
            "paper_total_rollouts_mean": float(np.mean([
                row.get("paper_total_rollouts", 0) for row in values
            ])),
            "paper_candidates_per_iteration_mean": float(np.mean([
                row.get("paper_candidates_per_iteration", 0)
                for row in values
            ])),
            "paper_standard_fallback_active_fraction": float(np.mean([
                row.get("paper_standard_fallback_active", 0.0)
                for row in values
            ])),
            "paper_standard_fallback_contract": str(
                next((
                    row.get(
                        "paper_standard_fallback_contract", "disabled"
                    )
                    for row in reversed(values)
                    if row.get("paper_standard_fallback_active", 0.0)
                ), "disabled")
            ),
            "paper_guided_unique_sequences_mean": float(np.mean([
                row.get("paper_guided_unique_sequences", 0)
                for row in values
            ])),
            "paper_guided_reuses_total": int(sum(
                row.get("paper_guided_reuses", 0) for row in values
            )),
            "reliability_hss_enabled_fraction": float(np.mean([
                row.get("reliability_hss_enabled", 0.0)
                for row in values
            ])),
            "reliability_authority_mean": float(np.mean([
                row.get("reliability_authority", 1.0)
                for row in values
            ])),
            "reliability_authority_min": float(np.min([
                row.get("reliability_authority", 1.0)
                for row in values
            ])),
            "reliability_authority_max": float(np.max([
                row.get("reliability_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_authority_mean": float(np.mean([
                row.get("reliability_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_authority_min": float(np.min([
                row.get("reliability_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_authority_max": float(np.max([
                row.get("reliability_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_requested_proposal_authority_mean": float(np.mean([
                row.get("reliability_requested_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_fallback_fraction_mean": float(
                np.mean([
                    row.get(
                        "reliability_proposal_fallback_fraction", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_gate_enabled_fraction": float(
                np.mean([
                    row.get(
                        "reliability_proposal_advantage_gate_enabled", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_gate_mode": str(
                values[-1].get(
                    "reliability_proposal_advantage_gate_mode", "disabled"
                )
            ),
            "reliability_proposal_advantage_gate_shadow_fraction": float(
                np.mean([
                    row.get(
                        "reliability_proposal_advantage_gate_shadow", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_observed_fraction": float(
                np.mean([
                    row.get(
                        "reliability_proposal_advantage_updated", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_disadvantage_fraction": float(
                sum(
                    row.get(
                        "reliability_proposal_advantage_disadvantage", 0.0
                    )
                    for row in values
                )
                / max(
                    sum(
                        row.get(
                            "reliability_proposal_advantage_updated", 0.0
                        )
                        for row in values
                    ),
                    1.0,
                )
            ),
            "reliability_proposal_advantage_relative_mean": float(np.mean([
                row.get("reliability_proposal_advantage_relative", 0.0)
                for row in values
                if row.get("reliability_proposal_advantage_updated", 0.0)
            ])) if any(
                row.get("reliability_proposal_advantage_updated", 0.0)
                for row in values
            ) else 0.0,
            "reliability_proposal_advantage_relative_min": float(np.min([
                row.get("reliability_proposal_advantage_relative", 0.0)
                for row in values
                if row.get("reliability_proposal_advantage_updated", 0.0)
            ])) if any(
                row.get("reliability_proposal_advantage_updated", 0.0)
                for row in values
            ) else 0.0,
            "reliability_proposal_advantage_relative_max": float(np.max([
                row.get("reliability_proposal_advantage_relative", 0.0)
                for row in values
                if row.get("reliability_proposal_advantage_updated", 0.0)
            ])) if any(
                row.get("reliability_proposal_advantage_updated", 0.0)
                for row in values
            ) else 0.0,
            "reliability_proposal_advantage_authority_applied_mean": float(
                np.mean([
                    row.get(
                        "reliability_proposal_advantage_authority_applied",
                        1.0,
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_authority_applied_min": float(
                np.min([
                    row.get(
                        "reliability_proposal_advantage_authority_applied",
                        1.0,
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_would_authority_next_min": float(
                np.min([
                    row.get(
                        "reliability_proposal_advantage_would_authority_next",
                        1.0,
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_latched_fraction": float(
                np.mean([
                    row.get(
                        "reliability_proposal_advantage_latched", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_proposal_advantage_latch_step": next((
                int(index)
                for index, row in enumerate(values)
                if row.get("reliability_proposal_advantage_latched", 0.0)
            ), -1),
            "reliability_proposal_advantage_observations_max": int(np.max([
                row.get("reliability_proposal_advantage_observations", 0)
                for row in values
            ])),
            "reliability_counterfactual_enabled_fraction": float(
                np.mean([
                    row.get("reliability_counterfactual_enabled", 0.0)
                    for row in values
                ])
            ),
            "reliability_counterfactual_authority_mean": float(
                np.mean([
                    row.get("reliability_counterfactual_authority", 1.0)
                    for row in values
                ])
            ),
            "reliability_counterfactual_authority_min": float(
                np.min([
                    row.get("reliability_counterfactual_authority", 1.0)
                    for row in values
                ])
            ),
            "reliability_counterfactual_advantage_mean": float(
                np.mean([
                    row.get("reliability_counterfactual_advantage", 0.0)
                    for row in values
                ])
            ),
            "reliability_counterfactual_actor_progress_mean": float(
                np.mean([
                    row.get(
                        "reliability_counterfactual_actor_progress", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_counterfactual_baseline_progress_mean": float(
                np.mean([
                    row.get(
                        "reliability_counterfactual_baseline_progress", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_dynamics_confidence_mean": float(np.mean([
                row.get("reliability_dynamics_confidence", 1.0)
                for row in values
            ])),
            "reliability_dynamics_confidence_min": float(np.min([
                row.get("reliability_dynamics_confidence", 1.0)
                for row in values
            ])),
            "reliability_dynamics_confidence_max": float(np.max([
                row.get("reliability_dynamics_confidence", 1.0)
                for row in values
            ])),
            "reliability_dynamics_routing_mode": str(
                values[-1].get(
                    "reliability_dynamics_routing_mode",
                    "trust_weighted",
                )
            ),
            "reliability_model_routing_factor_mean": float(np.mean([
                row.get("reliability_model_routing_factor", 1.0)
                for row in values
            ])),
            "reliability_model_routing_factor_min": float(np.min([
                row.get("reliability_model_routing_factor", 1.0)
                for row in values
            ])),
            "reliability_model_routing_factor_max": float(np.max([
                row.get("reliability_model_routing_factor", 1.0)
                for row in values
            ])),
            "reliability_actor_support_confidence_mean": float(
                np.mean([
                    row.get("reliability_actor_confidence", 1.0)
                    for row in values
                ])
            ),
            "reliability_actor_support_confidence_min": float(
                np.min([
                    row.get("reliability_actor_confidence", 1.0)
                    for row in values
                ])
            ),
            "reliability_actor_support_confidence_max": float(
                np.max([
                    row.get("reliability_actor_confidence", 1.0)
                    for row in values
                ])
            ),
            "reliability_actor_competence_mean": float(np.mean([
                row.get(
                    "reliability_actor_competence_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_min": float(np.min([
                row.get(
                    "reliability_actor_competence_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_max": float(np.max([
                row.get(
                    "reliability_actor_competence_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_raw_mean": float(np.mean([
                row.get(
                    "reliability_actor_competence_raw_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_mapped_mean": float(np.mean([
                row.get(
                    "reliability_actor_competence_mapped_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_guided_yield_mean": float(
                np.mean([
                    row.get(
                        "reliability_actor_competence_guided_yield",
                        0.0,
                    )
                    for row in values
                ])
            ),
            "reliability_actor_competence_gaussian_yield_mean": float(
                np.mean([
                    row.get(
                        "reliability_actor_competence_gaussian_yield",
                        0.0,
                    )
                    for row in values
                ])
            ),
            "reliability_low_fraction": float(np.mean([
                row.get("reliability_level", "disabled") == "low"
                for row in values
            ])),
            "reliability_medium_fraction": float(np.mean([
                row.get("reliability_level", "disabled") == "medium"
                for row in values
            ])),
            "reliability_high_fraction": float(np.mean([
                row.get("reliability_level", "disabled") == "high"
                for row in values
            ])),
            "reliability_guided_fraction_applied_mean": float(np.mean([
                row.get("reliability_guided_fraction_applied", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_next_mean": float(np.mean([
                row.get("reliability_guided_fraction_next", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_applied_mean": float(np.mean([
                row.get("reliability_guided_fraction_raw_applied", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_applied_min": float(np.min([
                row.get("reliability_guided_fraction_raw_applied", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_next_mean": float(np.mean([
                row.get("reliability_guided_fraction_raw_next", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_next_min": float(np.min([
                row.get("reliability_guided_fraction_raw_next", 0.30)
                for row in values
            ])),
            "terminal_guidance_floor_active_fraction": float(np.mean([
                row.get("terminal_guidance_floor_active", 0.0)
                for row in values
            ])),
            "terminal_guidance_tracking_active_fraction": float(np.mean([
                row.get("terminal_guidance_floor_active", 0.0)
                for row in values
                if row.get("target_phase") == "tracking"
            ])) if any(
                row.get("target_phase") == "tracking" for row in values
            ) else 0.0,
            "terminal_guidance_terminal_active_fraction": float(np.mean([
                row.get("terminal_guidance_floor_active", 0.0)
                for row in values
                if row.get("target_phase") in (
                    "terminal_approach", "terminal"
                )
            ])) if any(
                row.get("target_phase") in (
                    "terminal_approach", "terminal"
                )
                for row in values
            ) else 0.0,
            "completion_handover_enabled_fraction": float(np.mean([
                row.get("completion_handover_enabled", 0.0)
                for row in values
            ])),
            "completion_handover_authority_mean": float(np.mean([
                row.get("completion_handover_authority", 1.0)
                for row in values
            ])),
            "completion_handover_authority_min": float(np.min([
                row.get("completion_handover_authority", 1.0)
                for row in values
            ])),
            "terminal_value_completion_authority_mean": float(np.mean([
                row.get("terminal_value_completion_authority", 1.0)
                for row in values
            ])),
            "terminal_value_completion_authority_min": float(np.min([
                row.get("terminal_value_completion_authority", 1.0)
                for row in values
            ])),
            "paper_guided_elite_count_total": int(sum(
                row.get("paper_guided_elite_count", 0)
                for row in values
            )),
            "paper_gaussian_elite_count_total": int(sum(
                row.get("paper_gaussian_elite_count", 0)
                for row in values
            )),
            "paper_guided_opportunity_count_total": int(sum(
                row.get("paper_guided_opportunity_count", 0)
                for row in values
            )),
            "paper_gaussian_opportunity_count_total": int(sum(
                row.get("paper_gaussian_opportunity_count", 0)
                for row in values
            )),
            "paper_same_cycle_guided_cost_filter_enabled_fraction": float(
                np.mean([
                    row.get(
                        "paper_same_cycle_guided_cost_filter_enabled", 0.0
                    )
                    for row in values
                ])
            ),
            "paper_same_cycle_gaussian_actor_isolated_fraction": float(
                np.mean([
                    row.get(
                        "paper_same_cycle_gaussian_actor_isolated", 0.0
                    )
                    for row in values
                ])
            ),
            "paper_same_cycle_guided_cost_filter_active_fraction": float(
                np.mean([
                    row.get(
                        "paper_same_cycle_guided_cost_filter_iterations", 0
                    ) > 0
                    for row in values
                ])
            ),
            "paper_same_cycle_guided_cost_filter_iterations_total": int(sum(
                row.get(
                    "paper_same_cycle_guided_cost_filter_iterations", 0
                )
                for row in values
            )),
            "paper_same_cycle_guided_filtered_candidates_total": int(sum(
                row.get(
                    "paper_same_cycle_guided_filtered_candidates", 0
                )
                for row in values
            )),
            "paper_guided_cost_observed_fraction": float(np.mean([
                row.get("paper_guided_cost_observed", 0.0)
                for row in values
            ])),
            "paper_guided_minus_gaussian_cost_min_mean": float(np.mean([
                row.get("paper_guided_minus_gaussian_cost_min", 0.0)
                for row in values
                if row.get("paper_guided_cost_observed", 0.0)
            ])) if any(
                row.get("paper_guided_cost_observed", 0.0)
                for row in values
            ) else 0.0,
            "paper_guided_minus_gaussian_cost_mean_mean": float(np.mean([
                row.get("paper_guided_minus_gaussian_cost_mean", 0.0)
                for row in values
                if row.get("paper_guided_cost_observed", 0.0)
            ])) if any(
                row.get("paper_guided_cost_observed", 0.0)
                for row in values
            ) else 0.0,
            "paper_guided_feasible_fraction_mean": float(np.mean([
                row.get("paper_guided_feasible_fraction", 0.0)
                for row in values
                if row.get("paper_guided_cost_observed", 0.0)
            ])) if any(
                row.get("paper_guided_cost_observed", 0.0)
                for row in values
            ) else 0.0,
            "paper_gaussian_feasible_fraction_mean": float(np.mean([
                row.get("paper_gaussian_feasible_fraction", 0.0)
                for row in values
            ])),
            "paper_actor_baseline_mean_abs_delta_mean": float(np.mean([
                row.get("paper_actor_baseline_mean_abs_delta", 0.0)
                for row in values
            ])),
            "paper_actor_baseline_first_action_l2_delta_mean": float(np.mean([
                row.get(
                    "paper_actor_baseline_first_action_l2_delta", 0.0
                )
                for row in values
            ])),
            "rl_elite_fraction_mean": float(np.mean([
                row.get("rl_elite_fraction", 0.0) for row in values
            ])),
            "rl_elite_count_total": int(sum(
                row.get("rl_elite_count", 0) for row in values
            )),
            "shifted_elite_count_total": int(sum(
                row.get("shifted_elite_count", 0) for row in values
            )),
            "base_elite_count_total": int(sum(
                row.get("base_elite_count", 0) for row in values
            )),
            "covariance_scale_mean": float(np.mean([
                row.get("covariance_scale_mean", 1.0) for row in values
            ])),
            "terminal_value_enabled_fraction": float(np.mean([
                row.get("terminal_value_enabled", 0.0) for row in values
            ])),
            "terminal_value_conservative_enabled_fraction": float(
                np.mean([
                    row.get(
                        "terminal_value_conservative_enabled", 0.0
                    )
                    for row in values
                ])
            ),
            "terminal_value_authority_mean": float(np.mean([
                row.get("terminal_value_authority_mean", 1.0)
                for row in values
            ])),
            "terminal_value_authority_min": float(np.min([
                row.get("terminal_value_authority_min", 1.0)
                for row in values
            ])),
            "terminal_value_authority_max": float(np.max([
                row.get("terminal_value_authority_max", 1.0)
                for row in values
            ])),
            "terminal_value_raw_authority_mean": float(np.mean([
                row.get("terminal_value_raw_authority_mean", 0.0)
                for row in values
            ])),
            "terminal_value_causal_dynamics_cap_mean": float(np.mean([
                row.get("terminal_value_causal_dynamics_cap", 1.0)
                for row in values
            ])),
            "terminal_value_causal_dynamics_cap_min": float(np.min([
                row.get("terminal_value_causal_dynamics_cap", 1.0)
                for row in values
            ])),
            "terminal_value_causal_cap_active_fraction": float(np.mean([
                row.get(
                    "terminal_value_causal_cap_active_fraction", 0.0
                )
                for row in values
            ])),
            "terminal_value_dynamics_confidence_mean": float(np.mean([
                row.get(
                    "terminal_value_dynamics_confidence_mean", 1.0
                )
                for row in values
            ])),
            "terminal_value_critic_confidence_mean": float(np.mean([
                row.get(
                    "terminal_value_critic_confidence_mean", 1.0
                )
                for row in values
            ])),
            "terminal_value_uncertainty_mean": float(np.mean([
                row.get("terminal_value_uncertainty_mean", 0.0)
                for row in values
            ])),
            "terminal_value_uncertainty_cost_mean": float(np.mean([
                row.get("terminal_value_uncertainty_cost_mean", 0.0)
                for row in values
            ])),
            "terminal_q_mean": float(np.mean([
                row.get("terminal_q_mean", 0.0) for row in values
            ])),
            "terminal_q_disagreement_mean": float(np.mean([
                row.get("terminal_q_disagreement_mean", 0.0)
                for row in values
            ])),
            "residual_policy_context_enabled_fraction": float(np.mean([
                row.get("residual_policy_context_enabled", 0.0)
                for row in values
            ])),
            "residual_policy_predicted_abs_mean": float(np.mean([
                row.get("residual_policy_predicted_abs_mean", 0.0)
                for row in values
            ])),
            "residual_policy_predicted_abs_max": float(np.max([
                row.get("residual_policy_predicted_abs_max", 0.0)
                for row in values
            ])),
            "residual_policy_innovation_abs_mean": float(np.mean([
                row.get("residual_policy_innovation_abs_mean", 0.0)
                for row in values
            ])),
            "residual_policy_innovation_abs_max": float(np.max([
                row.get("residual_policy_innovation_abs_max", 0.0)
                for row in values
            ])),
            "residual_policy_disagreement_mean": float(np.mean([
                row.get("residual_policy_disagreement_mean", 0.0)
                for row in values
            ])),
            "residual_policy_support_mean": float(np.mean([
                row.get("residual_policy_support_mean", 0.0)
                for row in values
            ])),
            "residual_policy_innovation_valid_fraction": float(np.mean([
                row.get("residual_policy_innovation_valid_fraction", 0.0)
                for row in values
            ])),
            "residual_policy_authority_enabled_fraction": float(np.mean([
                row.get("residual_policy_authority_enabled", 0.0)
                for row in values
            ])),
            "residual_policy_authority_mean": float(np.mean([
                row.get("residual_policy_authority_mean", 0.0)
                for row in values
            ])),
            "residual_policy_authority_max": float(np.max([
                row.get("residual_policy_authority_max", 0.0)
                for row in values
            ])),
            "anytime_mean_samples": float(np.mean([
                row.get("anytime_selected_samples", 0) for row in values
            ])),
            "anytime_add_fraction": float(np.mean([
                row.get("anytime_add_samples", 0.0) for row in values
            ])),
            "anytime_predicted_advantage_mean": float(np.mean([
                row.get("anytime_predicted_advantage", 0.0)
                for row in values
            ])),
            "anytime_confidence_width_mean": float(np.mean([
                row.get("anytime_confidence_width", 0.0) for row in values
            ])),
            "anytime_decision_refresh_fraction": float(np.mean([
                row.get("anytime_decision_refreshed", 0.0) for row in values
            ])),
            "rl_gate_alpha_mean": float(
                np.mean([row.get("rl_gate_alpha", 0.0) for row in values])
            ),
            "rl_gate_fallback_steps": int(
                sum(
                    row.get("prior_type") == "rl_sac"
                    and row.get("rl_gate_alpha", 0.0) <= 1e-12
                    for row in values
                )
            ),
            "rl_policy_mode": str(values[-1].get("rl_policy_mode", "direct")),
            "rl_correction_gate_alpha_mean": float(np.mean([
                row.get("rl_correction_gate_alpha", 0.0) for row in values
            ])),
            "rl_base_action_abs_mean": float(np.mean([
                row.get("rl_base_action_abs_mean", 0.0) for row in values
            ])),
            "rl_unit_correction_abs_mean": float(np.mean([
                row.get("rl_unit_correction_abs_mean", 0.0) for row in values
            ])),
            "rl_applied_correction_abs_mean": float(np.mean([
                row.get("rl_applied_correction_abs_mean", 0.0) for row in values
            ])),
            "rl_applied_correction_abs_max": float(np.max([
                row.get("rl_applied_correction_abs_max", 0.0) for row in values
            ])),
            "rl_raw_applied_correction_abs_mean": float(np.mean([
                row.get("rl_raw_applied_correction_abs_mean", 0.0)
                for row in values
            ])),
            "rl_correction_advantage_gate_mode": str(values[-1].get(
                "rl_correction_advantage_gate_mode", "none"
            )),
            "rl_correction_advantage_critic_source": str(values[-1].get(
                "rl_correction_advantage_critic_source", "online"
            )),
            "rl_correction_advantage_threshold": float(values[-1].get(
                "rl_correction_advantage_threshold", 0.0
            )),
            "rl_correction_advantage_uncertainty_multiplier": float(
                values[-1].get(
                    "rl_correction_advantage_uncertainty_multiplier", 1.0
                )
            ),
            "rl_correction_advantage_gate_alpha_mean": float(np.mean([
                row.get("rl_correction_advantage_gate_alpha", 1.0)
                for row in values
            ])),
            "rl_correction_support_gate_enabled": bool(
                values[-1].get(
                    "rl_correction_support_gate_enabled", False
                )
            ),
            "rl_correction_support_confidence_mean": float(np.mean([
                row.get("rl_correction_support_confidence", 1.0)
                for row in values
            ])),
            "rl_correction_support_confidence_min": float(np.min([
                row.get("rl_correction_support_confidence", 1.0)
                for row in values
            ])),
            "rl_correction_effective_gate_alpha_mean": float(np.mean([
                row.get("rl_correction_effective_gate_alpha", 1.0)
                for row in values
            ])),
            "rl_unselected_critic_diagnostics_computed_fraction": float(
                np.mean([
                    row.get(
                        "rl_unselected_critic_diagnostics_computed", 0.0
                    )
                    for row in values
                ])
            ),
            "rl_selected_consensus_lcb_mean": float(np.mean([
                row.get("rl_selected_consensus_lcb", 0.0)
                for row in values
            ])),
            "rl_online_conservative_advantage_mean": float(np.mean([
                row.get("rl_online_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_online_conservative_advantage_min": float(np.min([
                row.get("rl_online_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_online_conservative_advantage_max": float(np.max([
                row.get("rl_online_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_online_positive_advantage_fraction": float(np.mean([
                row.get("rl_online_conservative_advantage", 0.0) >= 0.0
                for row in values
            ])),
            "rl_target_conservative_advantage_mean": float(np.mean([
                row.get("rl_target_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_target_conservative_advantage_min": float(np.min([
                row.get("rl_target_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_target_conservative_advantage_max": float(np.max([
                row.get("rl_target_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_target_positive_advantage_fraction": float(np.mean([
                row.get("rl_target_conservative_advantage", 0.0) >= 0.0
                for row in values
            ])),
            "rl_ood_score_mean": float(
                np.mean([row.get("rl_ood_score", 0.0) for row in values])
            ),
            "rl_ood_score_max": float(
                np.max([row.get("rl_ood_score", 0.0) for row in values])
            ),
            "rl_scene_complexity_score_mean": float(np.mean([
                row.get("rl_scene_complexity_score", 0.0) for row in values
            ])),
            "rl_scene_complexity_score_max": float(np.max([
                row.get("rl_scene_complexity_score", 0.0) for row in values
            ])),
            "rl_scene_complexity_front_proximity_mean": float(np.mean([
                row.get("rl_scene_complexity_front_proximity", 0.0)
                for row in values
            ])),
            "rl_scene_complexity_constriction_mean": float(np.mean([
                row.get("rl_scene_complexity_constriction", 0.0)
                for row in values
            ])),
            "rl_scene_complexity_density_mean": float(np.mean([
                row.get("rl_scene_complexity_density", 0.0)
                for row in values
            ])),
            "rl_scene_complexity_scan_valid_fraction": float(np.mean([
                row.get("rl_scene_complexity_scan_valid", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_gate_alpha_mean": float(np.mean([
                row.get("rl_temporal_closing_gate_alpha", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_gate_alpha_max": float(np.max([
                row.get("rl_temporal_closing_gate_alpha", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_rate_mps_max": float(np.max([
                row.get("rl_temporal_closing_rate_mps", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_held_fraction": float(np.mean([
                row.get("rl_temporal_closing_held", 0.0)
                for row in values
            ])),
            "rl_hazard_activation_mean": float(np.mean([
                row.get("rl_hazard_activation", 0.0) for row in values
            ])),
            "rl_competence_confidence_mean": float(np.mean([
                row.get("rl_competence_confidence", 0.0) for row in values
            ])),
            "rl_baseline_progress_m_mean": float(np.mean([
                row.get("rl_baseline_progress_m", 0.0) for row in values
            ])),
            "rl_baseline_progress_gate_ready_fraction": float(np.mean([
                row.get("rl_baseline_progress_gate_ready", 0.0)
                for row in values
            ])),
            "rl_baseline_stagnation_activation_mean": float(np.mean([
                row.get("rl_baseline_stagnation_activation", 0.0)
                for row in values
            ])),
            "rl_baseline_stagnation_activation_max": float(np.max([
                row.get("rl_baseline_stagnation_activation", 0.0)
                for row in values
            ])),
            "rl_baseline_stagnation_held_fraction": float(np.mean([
                row.get("rl_baseline_stagnation_held", 0.0)
                for row in values
            ])),
            "rl_gate_active_fraction": float(np.mean([
                row.get("rl_gate_alpha", 0.0) > 0.05 for row in values
            ])),
            "rl_learned_inference_skip_fraction": float(np.mean([
                row.get("rl_learned_inference_skipped", 0.0)
                for row in values
            ])),
            "profile_prior_fallback_ms_mean": float(np.mean([
                row.get("profile_prior_fallback_ms", 0.0)
                for row in values
            ])),
            "profile_prior_encode_normalize_ms_mean": float(np.mean([
                row.get("profile_prior_encode_normalize_ms", 0.0)
                for row in values
            ])),
            "profile_prior_gate_features_ms_mean": float(np.mean([
                row.get("profile_prior_gate_features_ms", 0.0)
                for row in values
            ])),
            "profile_prior_actor_ms_mean": float(np.mean([
                row.get("profile_prior_actor_ms", 0.0)
                for row in values
            ])),
            "profile_prior_advantage_ms_mean": float(np.mean([
                row.get("profile_prior_advantage_ms", 0.0)
                for row in values
            ])),
            "profile_prior_decoder_ms_mean": float(np.mean([
                row.get("profile_prior_decoder_ms", 0.0)
                for row in values
            ])),
            "profile_prior_outer_gate_ms_mean": float(np.mean([
                row.get("profile_prior_outer_gate_ms", 0.0)
                for row in values
            ])),
            "profile_prior_total_ms_mean": float(np.mean([
                row.get("profile_prior_total_ms", 0.0)
                for row in values
            ])),
            "rl_critic_disagreement_mean": float(np.mean([
                row.get("rl_critic_disagreement", 0.0) for row in values
            ])),
            "rl_critic_disagreement_max": float(np.max([
                row.get("rl_critic_disagreement", 0.0) for row in values
            ])),
            "rl_exploration_activation_mean": float(np.mean([
                row.get("rl_exploration_activation", 0.0) for row in values
            ])),
            "rl_exploration_latch_alpha_mean": float(np.mean([
                row.get("rl_exploration_latch_alpha", 0.0) for row in values
            ])),
            "rl_subgoal_distance_mean": float(
                np.mean([row.get("rl_subgoal_distance", 0.0) for row in values])
            ),
            "rl_subgoal_abs_bearing_mean": float(
                np.mean([
                    abs(row.get("rl_subgoal_bearing", 0.0)) for row in values
                ])
            ),
        }
        if self._tracking_monitor is not None:
            signed = np.asarray([
                row["signed_cross_track_error"] for row in values
            ], dtype=np.float64)
            heading = np.asarray([
                row["tangent_heading_error"] for row in values
            ], dtype=np.float64)
            margins = np.asarray([
                row["minimum_footprint_boundary_margin"] for row in values
            ], dtype=np.float64)
            result.update({
                "path_progress_m_max": float(max(
                    row["path_arc_length"] for row in values
                )),
                "path_progress_ratio_max": float(max(
                    row["path_progress_ratio"] for row in values
                )),
                "signed_cross_track_error_mean": float(np.mean(signed)),
                "cross_track_p95": float(np.percentile(np.abs(signed), 95)),
                "integrated_deviation_area": float(
                    np.sum(np.abs(signed)) * self.control_dt
                ),
                "tangent_heading_rmse": float(
                    np.sqrt(np.mean(np.square(heading)))
                ),
                "minimum_footprint_boundary_margin": float(np.min(margins)),
                "boundary_violation_steps": int(np.sum(margins < 0.0)),
                "boundary_safe_success": bool(success and np.all(margins >= 0.0)),
                "obstacle_pass_events": int(sum(
                    row["obstacle_pass_event"] for row in values
                )),
                "recovery_events": int(sum(
                    row["recovery_event"] for row in values
                )),
                "maximum_recovery_time": float(max(
                    row["recovery_time"] for row in values
                )),
                "maximum_recovery_distance": float(max(
                    row["recovery_distance"] for row in values
                )),
                "center_crossing_count": int(max(
                    row["center_crossing_count"] for row in values
                )),
            })
        return result
