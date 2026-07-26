from typing import Mapping

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import ControlCommand, SafetyDecision


class ScanGuardArbiter:
    """Final control boundary; neither Memory nor RL can bypass it."""

    def __init__(self, action_spec: ActionSpec, config=None):
        self.action_spec = action_spec
        self.config = dict(config or {})
        self.front_soft_block_max_speed = float(
            self.config.get("front_soft_block_max_speed", 0.0)
        )
        if self.front_soft_block_max_speed < 0.0:
            raise ValueError("front_soft_block_max_speed must be non-negative")
        self.dynamic_escape_enabled = bool(
            self.config.get("dynamic_escape_enabled", False)
        )
        self.dynamic_escape_max_speed = float(
            self.config.get("dynamic_escape_max_speed", 0.20)
        )
        self.dynamic_escape_min_risk_improvement = float(
            self.config.get(
                "dynamic_escape_min_risk_improvement", 0.05
            )
        )
        self.dynamic_escape_min_probability_mass_improvement = float(
            self.config.get(
                "dynamic_escape_min_probability_mass_improvement", 0.05
            )
        )
        self.dynamic_escape_min_probability_mass_relative_improvement = float(
            self.config.get(
                "dynamic_escape_min_probability_mass_relative_improvement",
                0.0,
            )
        )
        self.dynamic_escape_probability_mass_enabled = bool(
            self.config.get(
                "dynamic_escape_probability_mass_enabled", False
            )
        )
        self.dynamic_escape_hold_enabled = bool(
            self.config.get("dynamic_escape_hold_enabled", False)
        )
        self.dynamic_escape_hold_steps = int(
            self.config.get("dynamic_escape_hold_steps", 3)
        )
        self.dynamic_escape_hold_min_probability = float(
            self.config.get("dynamic_escape_hold_min_probability", 0.20)
        )
        self.dynamic_escape_reactive_enabled = bool(
            self.config.get(
                "dynamic_escape_reactive_enabled", False
            )
        )
        self.dynamic_escape_use_vetted_planner_control = bool(
            self.config.get(
                "dynamic_escape_use_vetted_planner_control", False
            )
        )
        self.dynamic_escape_preserve_hard_fallback_planner_control = bool(
            self.config.get(
                "dynamic_escape_preserve_hard_fallback_planner_control",
                False,
            )
        )
        self.dynamic_escape_veto_forward_when_reactive_reverse = bool(
            self.config.get(
                "dynamic_escape_veto_forward_when_reactive_reverse", False
            )
        )
        self.dynamic_escape_trigger_ttc_s = float(
            self.config.get("dynamic_escape_trigger_ttc_s", 1.20)
        )
        self.dynamic_escape_turn_gain = float(
            self.config.get("dynamic_escape_turn_gain", 1.5)
        )
        self.dynamic_escape_min_speed = float(
            self.config.get("dynamic_escape_min_speed", 0.20)
        )
        self.dynamic_escape_reverse_speed = float(
            self.config.get("dynamic_escape_reverse_speed", 0.0)
        )
        self.dynamic_escape_uncertainty_fusion_enabled = bool(
            self.config.get(
                "dynamic_escape_uncertainty_fusion_enabled", False
            )
        )
        self.dynamic_escape_uncertainty_nis_threshold = float(
            self.config.get(
                "dynamic_escape_uncertainty_nis_threshold", 4.0
            )
        )
        self.dynamic_escape_uncertainty_trigger_ttc_s = float(
            self.config.get(
                "dynamic_escape_uncertainty_trigger_ttc_s",
                self.dynamic_escape_trigger_ttc_s,
            )
        )
        self.dynamic_escape_direction_commit_steps = int(
            self.config.get("dynamic_escape_direction_commit_steps", 0)
        )
        # v12 uses a finite two-phase forward corridor (turn, then coast
        # straight) instead of repeatedly replaying the same saturated turn.
        self.dynamic_escape_corridor_enabled = bool(
            self.config.get("dynamic_escape_corridor_enabled", False)
        )
        self.dynamic_escape_corridor_turn_steps = int(
            self.config.get("dynamic_escape_corridor_turn_steps", 3)
        )
        self.dynamic_escape_corridor_commit_steps = int(
            self.config.get("dynamic_escape_corridor_commit_steps", 10)
        )
        self.dynamic_escape_corridor_speed = float(
            self.config.get(
                "dynamic_escape_corridor_speed",
                self.dynamic_escape_max_speed,
            )
        )
        self.dynamic_recovery_enabled = bool(
            self.config.get("dynamic_recovery_enabled", False)
        )
        self.dynamic_recovery_entry_probability = float(
            self.config.get(
                "dynamic_recovery_entry_probability", 0.10
            )
        )
        self.dynamic_recovery_abort_probability = float(
            self.config.get(
                "dynamic_recovery_abort_probability", 0.15
            )
        )
        self.dynamic_recovery_clear_hold_steps = int(
            self.config.get("dynamic_recovery_clear_hold_steps", 5)
        )
        self.dynamic_recovery_heading_tolerance_rad = float(
            self.config.get(
                "dynamic_recovery_heading_tolerance_rad", 0.30
            )
        )
        self.dynamic_recovery_minimum_heading_error_rad = float(
            self.config.get(
                "dynamic_recovery_minimum_heading_error_rad", 0.0
            )
        )
        self.dynamic_recovery_min_speed = float(
            self.config.get("dynamic_recovery_min_speed", 0.30)
        )
        self.dynamic_recovery_translation_enabled = bool(
            self.config.get(
                "dynamic_recovery_translation_enabled", True
            )
        )
        self.dynamic_recovery_turn_gain = float(
            self.config.get("dynamic_recovery_turn_gain", 1.50)
        )
        self.dynamic_recovery_release_steps = int(
            self.config.get("dynamic_recovery_release_steps", 5)
        )
        self.dynamic_recovery_progressive_acceleration_enabled = bool(
            self.config.get(
                "dynamic_recovery_progressive_acceleration_enabled", False
            )
        )
        self.dynamic_recovery_ramp_start_speed = float(
            self.config.get("dynamic_recovery_ramp_start_speed", 0.0)
        )
        self.dynamic_recovery_ramp_steps = int(
            self.config.get("dynamic_recovery_ramp_steps", 1)
        )
        self.dynamic_recovery_minimum_forward_commit_steps = int(
            self.config.get(
                "dynamic_recovery_minimum_forward_commit_steps", 0
            )
        )
        self.dynamic_recovery_progress_watch_enabled = bool(
            self.config.get(
                "dynamic_recovery_progress_watch_enabled", False
            )
        )
        self.dynamic_recovery_progress_watch_steps = int(
            self.config.get("dynamic_recovery_progress_watch_steps", 5)
        )
        self.dynamic_recovery_progress_watch_minimum_progress_m = float(
            self.config.get(
                "dynamic_recovery_progress_watch_minimum_progress_m", 0.04
            )
        )
        self.dynamic_recovery_progress_watch_max_reentries = int(
            self.config.get(
                "dynamic_recovery_progress_watch_max_reentries", 1
            )
        )
        self.dynamic_recovery_progress_watch_timeout_steps = int(
            self.config.get(
                "dynamic_recovery_progress_watch_timeout_steps", 60
            )
        )
        self.dynamic_recovery_alignment_creep_enabled = bool(
            self.config.get(
                "dynamic_recovery_alignment_creep_enabled", False
            )
        )
        self.dynamic_recovery_alignment_creep_speed = float(
            self.config.get("dynamic_recovery_alignment_creep_speed", 0.0)
        )
        self.dynamic_recovery_alignment_creep_minimum_heading_error_rad = float(
            self.config.get(
                "dynamic_recovery_alignment_creep_minimum_heading_error_rad",
                0.0,
            )
        )
        self.dynamic_recovery_alignment_creep_clearance_trend_enabled = bool(
            self.config.get(
                "dynamic_recovery_alignment_creep_clearance_trend_enabled",
                False,
            )
        )
        self.dynamic_recovery_alignment_creep_clearance_tolerance_m = float(
            self.config.get(
                "dynamic_recovery_alignment_creep_clearance_tolerance_m",
                0.0,
            )
        )
        self.dynamic_recovery_alignment_creep_clearance_hold_steps = int(
            self.config.get(
                "dynamic_recovery_alignment_creep_clearance_hold_steps", 1
            )
        )
        self.dynamic_recovery_alignment_creep_minimum_scan_clearance_m = float(
            self.config.get(
                "dynamic_recovery_alignment_creep_minimum_scan_clearance_m",
                0.0,
            )
        )
        self.dynamic_recovery_goal_release_distance_m = float(
            self.config.get(
                "dynamic_recovery_goal_release_distance_m", 0.45
            )
        )
        self.dynamic_deadline_supervisor_enabled = bool(
            self.config.get("dynamic_deadline_supervisor_enabled", False)
        )
        self.dynamic_deadline_episode_steps = int(
            self.config.get("dynamic_deadline_episode_steps", 0)
        )
        self.dynamic_deadline_control_period_s = float(
            self.config.get("dynamic_deadline_control_period_s", 0.1)
        )
        self.dynamic_deadline_position_tolerance_m = float(
            self.config.get("dynamic_deadline_position_tolerance_m", 0.3)
        )
        self.dynamic_deadline_reserve_steps = int(
            self.config.get("dynamic_deadline_reserve_steps", 2)
        )
        self.dynamic_deadline_authority_reserve_steps = int(
            self.config.get(
                "dynamic_deadline_authority_reserve_steps",
                self.dynamic_deadline_reserve_steps,
            )
        )
        self.dynamic_deadline_clear_hold_steps = int(
            self.config.get("dynamic_deadline_clear_hold_steps", 5)
        )
        self.dynamic_deadline_minimum_required_speed = float(
            self.config.get(
                "dynamic_deadline_minimum_required_speed", 0.2
            )
        )
        self.dynamic_deadline_speed_floor = float(
            self.config.get("dynamic_deadline_speed_floor", 0.3)
        )
        self.dynamic_deadline_speed_margin = float(
            self.config.get("dynamic_deadline_speed_margin", 1.1)
        )
        self.dynamic_deadline_trigger_margin_mps = float(
            self.config.get("dynamic_deadline_trigger_margin_mps", 0.02)
        )
        self.dynamic_deadline_heading_tolerance_rad = float(
            self.config.get(
                "dynamic_deadline_heading_tolerance_rad", 0.2
            )
        )
        if self.dynamic_escape_max_speed < 0.0:
            raise ValueError(
                "dynamic_escape_max_speed must be non-negative"
            )
        if (
            self.dynamic_escape_min_risk_improvement < 0.0
            or self.dynamic_escape_min_probability_mass_improvement < 0.0
            or not 0.0
            <= self.dynamic_escape_min_probability_mass_relative_improvement
            <= 1.0
        ):
            raise ValueError(
                "dynamic escape risk improvements must be non-negative"
            )
        if (
            self.dynamic_escape_trigger_ttc_s <= 0.0
            or self.dynamic_escape_turn_gain <= 0.0
            or self.dynamic_escape_min_speed < 0.0
            or self.dynamic_escape_min_speed
            > self.dynamic_escape_max_speed
            or self.dynamic_escape_reverse_speed < 0.0
            or self.dynamic_escape_uncertainty_nis_threshold <= 0.0
            or self.dynamic_escape_uncertainty_trigger_ttc_s <= 0.0
            or self.dynamic_escape_direction_commit_steps < 0
            or self.dynamic_escape_corridor_turn_steps < 1
            or self.dynamic_escape_corridor_commit_steps
            < self.dynamic_escape_corridor_turn_steps
            or self.dynamic_escape_corridor_speed < 0.0
            or self.dynamic_escape_corridor_speed
            > self.dynamic_escape_max_speed
        ):
            raise ValueError(
                "dynamic reactive escape parameters are invalid"
            )
        if (
            self.dynamic_escape_hold_steps < 1
            or not 0.0 <= self.dynamic_escape_hold_min_probability <= 1.0
        ):
            raise ValueError("dynamic escape hold parameters are invalid")
        if self.dynamic_recovery_enabled and (
            not 0.0
            <= self.dynamic_recovery_entry_probability
            <= self.dynamic_recovery_abort_probability
            < 1.0
            or self.dynamic_recovery_clear_hold_steps < 1
            or not 0.0
            < self.dynamic_recovery_heading_tolerance_rad
            < 0.5 * np.pi
            or not 0.0
            <= self.dynamic_recovery_minimum_heading_error_rad
            < np.pi
            or self.dynamic_recovery_min_speed < 0.0
            or self.dynamic_recovery_min_speed
            > self.dynamic_escape_max_speed
            or self.dynamic_recovery_turn_gain <= 0.0
            or self.dynamic_recovery_release_steps < 1
            or self.dynamic_recovery_minimum_forward_commit_steps < 0
            or self.dynamic_recovery_goal_release_distance_m <= 0.0
        ):
            raise ValueError(
                "dynamic recovery parameters are invalid"
            )
        if self.dynamic_recovery_progressive_acceleration_enabled and (
            self.dynamic_recovery_ramp_steps < 1
            or self.dynamic_recovery_ramp_start_speed < 0.0
            or self.dynamic_recovery_ramp_start_speed
            > self.dynamic_recovery_min_speed
        ):
            raise ValueError(
                "dynamic recovery acceleration ramp parameters are invalid"
            )
        if self.dynamic_recovery_progress_watch_enabled and (
            self.dynamic_recovery_progress_watch_steps < 1
            or self.dynamic_recovery_progress_watch_minimum_progress_m < 0.0
            or self.dynamic_recovery_progress_watch_max_reentries < 1
            or self.dynamic_recovery_progress_watch_timeout_steps
            < self.dynamic_recovery_progress_watch_steps
        ):
            raise ValueError(
                "dynamic recovery progress watch parameters are invalid"
            )
        if self.dynamic_recovery_alignment_creep_enabled and (
            self.dynamic_recovery_alignment_creep_speed <= 0.0
            or self.dynamic_recovery_alignment_creep_speed
            > self.dynamic_recovery_min_speed
            or not 0.0
            <= self.dynamic_recovery_alignment_creep_minimum_heading_error_rad
            < np.pi
            or self.dynamic_recovery_alignment_creep_clearance_tolerance_m
            < 0.0
            or self.dynamic_recovery_alignment_creep_clearance_hold_steps < 1
            or self.dynamic_recovery_alignment_creep_minimum_scan_clearance_m
            < 0.0
        ):
            raise ValueError(
                "dynamic recovery alignment creep parameters are invalid"
            )
        if self.dynamic_deadline_supervisor_enabled and (
            self.dynamic_deadline_episode_steps < 1
            or self.dynamic_deadline_control_period_s <= 0.0
            or self.dynamic_deadline_position_tolerance_m <= 0.0
            or self.dynamic_deadline_reserve_steps < 0
            or self.dynamic_deadline_reserve_steps
            >= self.dynamic_deadline_episode_steps
            or self.dynamic_deadline_authority_reserve_steps < 0
            or self.dynamic_deadline_authority_reserve_steps
            >= self.dynamic_deadline_episode_steps
            or self.dynamic_deadline_clear_hold_steps < 1
            or self.dynamic_deadline_minimum_required_speed < 0.0
            or self.dynamic_deadline_minimum_required_speed
            > self.dynamic_escape_max_speed
            or self.dynamic_deadline_speed_floor < 0.0
            or self.dynamic_deadline_speed_floor
            > self.dynamic_escape_max_speed
            or self.dynamic_deadline_speed_margin < 1.0
            or self.dynamic_deadline_trigger_margin_mps < 0.0
            or not 0.0
            < self.dynamic_deadline_heading_tolerance_rad
            < 0.5 * np.pi
        ):
            raise ValueError(
                "dynamic deadline supervisor parameters are invalid"
            )
        self.reset()

    def reset(self):
        self._dynamic_escape_seen = False
        self._dynamic_escape_hold_remaining = 0
        self._dynamic_escape_hold_values = None
        self._dynamic_escape_direction_commit_remaining = 0
        self._dynamic_escape_direction_commit_values = None
        self._dynamic_escape_corridor_remaining = 0
        self._dynamic_escape_corridor_turn_remaining = 0
        self._dynamic_escape_corridor_turn_sign = 0.0
        self._dynamic_recovery_active = False
        self._dynamic_recovery_clear_steps = 0
        self._dynamic_recovery_release_count = 0
        self._dynamic_recovery_advance_steps = 0
        self._dynamic_recovery_alignment_creep_latched = False
        self._dynamic_recovery_previous_scan_clearance = float("nan")
        self._dynamic_recovery_clearance_trend_steps = 0
        self._dynamic_recovery_progress_watch_active = False
        self._dynamic_recovery_progress_watch_remaining = 0
        self._dynamic_recovery_progress_watch_samples = []
        self._dynamic_recovery_progress_reentry_count = 0
        self._dynamic_deadline_decision_count = 0
        self._dynamic_deadline_conflict_seen = False
        self._dynamic_deadline_clear_steps = 0

    def arbitrate(
        self,
        proposed: ControlCommand,
        guard_result: Mapping[str, object],
        planning_context: Mapping[str, object] = None,
    ):
        self._dynamic_deadline_decision_count += 1
        values = self.action_spec.clip(proposed.values)
        reason = str(guard_result.get("reason", "front_clear"))
        context = dict(planning_context or {})
        selected_probability = float(
            context.get(
                "probabilistic_obstacle_maximum_step_probability",
                1.0,
            )
        )
        stop_probability = float(
            context.get(
                "probabilistic_obstacle_stop_maximum_probability",
                0.0,
            )
        )
        selected_probability_mass = float(
            context.get("probabilistic_obstacle_probability_mass", 0.0)
        )
        stop_probability_mass = float(
            context.get(
                "probabilistic_obstacle_stop_probability_mass", 0.0
            )
        )
        maximum_probability_improved = bool(
            stop_probability - selected_probability
            >= self.dynamic_escape_min_risk_improvement
        )
        saturated_probability_mass_improved = bool(
            self.dynamic_escape_probability_mass_enabled
            and np.isclose(
                selected_probability,
                stop_probability,
                atol=1.0e-12,
                rtol=0.0,
            )
            and stop_probability_mass - selected_probability_mass
            >= self.dynamic_escape_min_probability_mass_improvement
            and (
                stop_probability_mass - selected_probability_mass
            ) / max(abs(stop_probability_mass), 1.0e-12)
            >= self.dynamic_escape_min_probability_mass_relative_improvement
        )
        planned_escape_allowed = bool(
            self.dynamic_escape_enabled
            and guard_result.get("emergency_stop", False)
            and reason == "near_body_hard_stop"
            and guard_result.get(
                "dynamic_obstacle_near_body_match", False
            )
            and context.get(
                "probabilistic_obstacle_active_fallback_used", False
            )
            and int(context.get(
                "probabilistic_obstacle_active_fallback_index", -1
            )) >= 0
            and (
                maximum_probability_improved
                or saturated_probability_mass_improved
            )
        )
        fresh_reactive_escape_allowed = bool(
            self.dynamic_escape_enabled
            and self.dynamic_escape_reactive_enabled
            and context.get(
                "probabilistic_obstacle_active_avoidance_enabled",
                False,
            )
            and guard_result.get(
                "dynamic_obstacle_scan_flow_match", False
            )
            and guard_result.get("temporal_scan_valid", False)
            and float(guard_result.get(
                "temporal_scan_ttc_s", float("inf")
            )) <= self.dynamic_escape_trigger_ttc_s
        )
        tracker_innovation_nis = float(
            context.get("dynamic_obstacle_tracker_innovation_nis", 0.0)
            or 0.0
        )
        tracker_change_triggered = bool(
            context.get(
                "dynamic_obstacle_tracker_change_triggered", False
            )
        )
        temporal_ttc_s = float(
            guard_result.get("temporal_scan_ttc_s", float("inf"))
        )
        uncertainty_fusion_escape_allowed = bool(
            self.dynamic_escape_enabled
            and self.dynamic_escape_reactive_enabled
            and self.dynamic_escape_uncertainty_fusion_enabled
            and context.get(
                "probabilistic_obstacle_active_avoidance_enabled",
                False,
            )
            and guard_result.get(
                "dynamic_obstacle_scan_flow_match", False
            )
            and guard_result.get("temporal_scan_valid", False)
            and np.isfinite(temporal_ttc_s)
            and temporal_ttc_s
            <= self.dynamic_escape_uncertainty_trigger_ttc_s
            and (
                tracker_change_triggered
                or tracker_innovation_nis
                >= self.dynamic_escape_uncertainty_nis_threshold
            )
        )
        held_reactive_escape_allowed = bool(
            self.dynamic_escape_enabled
            and self.dynamic_escape_reactive_enabled
            and self.dynamic_escape_hold_enabled
            and not self.dynamic_escape_use_vetted_planner_control
            and self._dynamic_escape_hold_remaining > 0
            and self._dynamic_escape_hold_values is not None
            and context.get(
                "probabilistic_obstacle_active_avoidance_enabled",
                False,
            )
            and selected_probability
            >= self.dynamic_escape_hold_min_probability
        )
        reactive_escape_allowed = bool(
            fresh_reactive_escape_allowed
            or held_reactive_escape_allowed
            or uncertainty_fusion_escape_allowed
        )
        if (
            not fresh_reactive_escape_allowed
            and not held_reactive_escape_allowed
        ):
            self._dynamic_escape_hold_remaining = 0
            self._dynamic_escape_hold_values = None
        dynamic_escape_allowed = bool(
            planned_escape_allowed or reactive_escape_allowed
        )
        planner_temporal_escape_active = bool(
            context.get(
                "probabilistic_obstacle_temporal_emergency_vetted", False
            )
            and context.get(
                "probabilistic_obstacle_emergency_candidate_selected", False
            )
        )
        recovery_mode = "inactive"
        recovery_speed_floor = 0.0
        recovery_risk_ramp_fraction = 0.0
        recovery_forward_commit_active = False
        recovery_alignment_creep_active = False
        recovery_progress_watch_triggered = False
        recovery_progress_watch_progress_m = 0.0
        deadline_supervisor_active = False
        deadline_required_speed = 0.0
        deadline_speed_floor = 0.0
        deadline_steps_remaining = max(
            0,
            self.dynamic_deadline_episode_steps
            - self._dynamic_deadline_decision_count
            + 1,
        )
        deadline_available_steps = max(
            0,
            deadline_steps_remaining - self.dynamic_deadline_reserve_steps,
        )
        deadline_authority_available_steps = max(
            0,
            deadline_steps_remaining
            - self.dynamic_deadline_authority_reserve_steps,
        )
        deadline_urgency_exhausted = False
        recovery_scan_clearance = float("nan")
        scan_clearance_values = []
        for key in (
            "min_front_range",
            "min_side_range",
            "temporal_scan_clearance_m",
        ):
            raw_clearance = guard_result.get(key)
            if raw_clearance is None:
                continue
            try:
                candidate_clearance = float(raw_clearance)
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate_clearance) and candidate_clearance >= 0.0:
                scan_clearance_values.append(candidate_clearance)
        if scan_clearance_values:
            recovery_scan_clearance = float(min(scan_clearance_values))
        recovery_clearance_trend_ready = bool(
            not self.dynamic_recovery_alignment_creep_clearance_trend_enabled
        )
        recovery_guard_clear = bool(
            not guard_result.get("emergency_stop", False)
            and reason == "front_clear"
            and not guard_result.get("should_slow_down", False)
        )
        recovery_ttc_clear = bool(
            not (
                guard_result.get(
                    "dynamic_obstacle_scan_flow_match", False
                )
                and guard_result.get("temporal_scan_valid", False)
                and float(guard_result.get(
                    "temporal_scan_ttc_s", float("inf")
                )) <= self.dynamic_escape_trigger_ttc_s
            )
        )
        recovery_heading_error = float(
            context.get(
                "target_bearing_error",
                context.get("terminal_bearing_error", float("nan")),
            )
        )
        recovery_goal_distance = float(
            context.get("terminal_control_distance", float("inf"))
        )
        if dynamic_escape_allowed or planner_temporal_escape_active:
            if self.dynamic_deadline_supervisor_enabled:
                self._dynamic_deadline_conflict_seen = True
                self._dynamic_deadline_clear_steps = 0
            self._dynamic_escape_seen = True
            self._dynamic_recovery_active = False
            self._dynamic_recovery_clear_steps = 0
            self._dynamic_recovery_release_count = 0
            self._dynamic_recovery_advance_steps = 0
            self._dynamic_recovery_alignment_creep_latched = False
            self._dynamic_recovery_previous_scan_clearance = float("nan")
            self._dynamic_recovery_clearance_trend_steps = 0
            self._dynamic_recovery_progress_watch_active = False
            self._dynamic_recovery_progress_watch_remaining = 0
            self._dynamic_recovery_progress_watch_samples = []
        elif self.dynamic_recovery_enabled and self._dynamic_escape_seen:
            recovery_clear = bool(
                recovery_guard_clear
                and recovery_ttc_clear
                and selected_probability
                <= self.dynamic_recovery_entry_probability
            )
            recovery_abort = bool(
                not recovery_guard_clear
                or not recovery_ttc_clear
                or selected_probability
                >= self.dynamic_recovery_abort_probability
            )
            if (
                self._dynamic_recovery_progress_watch_active
                and not self._dynamic_recovery_active
            ):
                if (
                    recovery_abort
                    or not recovery_clear
                    or not np.isfinite(recovery_goal_distance)
                    or recovery_goal_distance
                    <= self.dynamic_recovery_goal_release_distance_m
                ):
                    self._dynamic_recovery_progress_watch_active = False
                    self._dynamic_recovery_progress_watch_remaining = 0
                    self._dynamic_recovery_progress_watch_samples = []
                    self._dynamic_escape_seen = False
                    recovery_mode = "progress_watch_cancelled"
                else:
                    self._dynamic_recovery_progress_watch_remaining -= 1
                    self._dynamic_recovery_progress_watch_samples.append(
                        recovery_goal_distance
                    )
                    maximum_samples = (
                        self.dynamic_recovery_progress_watch_steps + 1
                    )
                    if (
                        len(self._dynamic_recovery_progress_watch_samples)
                        > maximum_samples
                    ):
                        self._dynamic_recovery_progress_watch_samples.pop(0)
                    if (
                        len(self._dynamic_recovery_progress_watch_samples)
                        == maximum_samples
                    ):
                        recovery_progress_watch_progress_m = float(
                            self._dynamic_recovery_progress_watch_samples[0]
                            - self._dynamic_recovery_progress_watch_samples[-1]
                        )
                        if (
                            recovery_progress_watch_progress_m
                            < self.dynamic_recovery_progress_watch_minimum_progress_m
                            and self._dynamic_recovery_progress_reentry_count
                            < self.dynamic_recovery_progress_watch_max_reentries
                        ):
                            self._dynamic_recovery_active = True
                            self._dynamic_recovery_progress_watch_active = False
                            self._dynamic_recovery_progress_watch_remaining = 0
                            self._dynamic_recovery_progress_watch_samples = []
                            self._dynamic_recovery_progress_reentry_count += 1
                            self._dynamic_recovery_release_count = 0
                            self._dynamic_recovery_advance_steps = 0
                            self._dynamic_recovery_alignment_creep_latched = False
                            self._dynamic_recovery_previous_scan_clearance = (
                                float("nan")
                            )
                            self._dynamic_recovery_clearance_trend_steps = 0
                            recovery_progress_watch_triggered = True
                            recovery_mode = "progress_reentered"
                    if (
                        not self._dynamic_recovery_active
                        and self._dynamic_recovery_progress_watch_remaining <= 0
                    ):
                        self._dynamic_recovery_progress_watch_active = False
                        self._dynamic_recovery_progress_watch_samples = []
                        self._dynamic_escape_seen = False
                        recovery_mode = "progress_watch_timeout"
                    elif not self._dynamic_recovery_active:
                        recovery_mode = "progress_watch"
            elif self._dynamic_recovery_active and recovery_abort:
                self._dynamic_recovery_active = False
                self._dynamic_recovery_clear_steps = 0
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
                self._dynamic_recovery_alignment_creep_latched = False
                self._dynamic_recovery_previous_scan_clearance = float("nan")
                self._dynamic_recovery_clearance_trend_steps = 0
                self._dynamic_recovery_progress_watch_active = False
                self._dynamic_recovery_progress_watch_remaining = 0
                self._dynamic_recovery_progress_watch_samples = []
                recovery_mode = "aborted"
            elif not self._dynamic_recovery_active:
                if recovery_clear:
                    self._dynamic_recovery_clear_steps += 1
                else:
                    self._dynamic_recovery_clear_steps = 0
                if (
                    self._dynamic_recovery_clear_steps
                    >= self.dynamic_recovery_clear_hold_steps
                ):
                    if (
                        np.isfinite(recovery_heading_error)
                        and abs(recovery_heading_error)
                        >= self.dynamic_recovery_minimum_heading_error_rad
                    ):
                        self._dynamic_recovery_active = True
                        self._dynamic_recovery_release_count = 0
                        self._dynamic_recovery_advance_steps = 0
                        self._dynamic_recovery_alignment_creep_latched = False
                        self._dynamic_recovery_previous_scan_clearance = float(
                            "nan"
                        )
                        self._dynamic_recovery_clearance_trend_steps = 0
                        self._dynamic_recovery_progress_watch_active = False
                        self._dynamic_recovery_progress_watch_remaining = 0
                        self._dynamic_recovery_progress_watch_samples = []
                        recovery_mode = "entered"
                    else:
                        self._dynamic_escape_seen = False
                        self._dynamic_recovery_clear_steps = 0
                        self._dynamic_recovery_advance_steps = 0
                        self._dynamic_recovery_alignment_creep_latched = False
                        self._dynamic_recovery_previous_scan_clearance = float(
                            "nan"
                        )
                        self._dynamic_recovery_clearance_trend_steps = 0
                        self._dynamic_recovery_progress_watch_active = False
                        self._dynamic_recovery_progress_watch_remaining = 0
                        self._dynamic_recovery_progress_watch_samples = []
                recovery_mode = "not_needed"
        deadline_guard_clear = bool(
            recovery_guard_clear
            and recovery_ttc_clear
            and selected_probability
            <= self.dynamic_recovery_entry_probability
            and not context.get(
                "probabilistic_obstacle_hard_violation", False
            )
        )
        if (
            self.dynamic_deadline_supervisor_enabled
            and self._dynamic_deadline_conflict_seen
            and not dynamic_escape_allowed
            and not planner_temporal_escape_active
            and not self._dynamic_recovery_active
            and not self._dynamic_recovery_progress_watch_active
            and deadline_guard_clear
        ):
            self._dynamic_deadline_clear_steps += 1
        elif self.dynamic_deadline_supervisor_enabled and (
            not deadline_guard_clear
            or dynamic_escape_allowed
            or planner_temporal_escape_active
        ):
            self._dynamic_deadline_clear_steps = 0

        proposed_deadline_v = (
            float(values[self.action_spec.index("v_cmd")])
            if "v_cmd" in self.action_spec.names
            else 0.0
        )
        deadline_distance_remaining = max(
            0.0,
            recovery_goal_distance
            - self.dynamic_deadline_position_tolerance_m,
        )
        if not self.dynamic_deadline_supervisor_enabled:
            deadline_required_speed = 0.0
        elif deadline_available_steps > 0:
            deadline_required_speed = float(
                deadline_distance_remaining
                / (
                    deadline_available_steps
                    * self.dynamic_deadline_control_period_s
                )
            )
        elif deadline_distance_remaining > 0.0:
            # The urgency horizon is exhausted, but an independently frozen
            # authority horizon may still allow a final guarded cycle.  Keep
            # diagnostics finite and request only the existing bounded escape
            # speed; the safety gates below retain higher priority.
            deadline_required_speed = self.dynamic_escape_max_speed
            deadline_urgency_exhausted = True
        deadline_supervisor_active = bool(
            self.dynamic_deadline_supervisor_enabled
            and self._dynamic_deadline_conflict_seen
            and not dynamic_escape_allowed
            and not planner_temporal_escape_active
            and not self._dynamic_recovery_active
            and not self._dynamic_recovery_progress_watch_active
            and deadline_guard_clear
            and self._dynamic_deadline_clear_steps
            >= self.dynamic_deadline_clear_hold_steps
            and np.isfinite(recovery_goal_distance)
            and recovery_goal_distance
            > self.dynamic_deadline_position_tolerance_m
            and np.isfinite(recovery_heading_error)
            and abs(recovery_heading_error)
            <= self.dynamic_deadline_heading_tolerance_rad
            and deadline_authority_available_steps > 0
            and deadline_required_speed
            >= self.dynamic_deadline_minimum_required_speed
            and deadline_required_speed
            > proposed_deadline_v
            + self.dynamic_deadline_trigger_margin_mps
        )
        if self._dynamic_recovery_active:
            if self.dynamic_recovery_alignment_creep_clearance_trend_enabled:
                previous_clearance = (
                    self._dynamic_recovery_previous_scan_clearance
                )
                if (
                    np.isfinite(recovery_scan_clearance)
                    and np.isfinite(previous_clearance)
                    and recovery_scan_clearance
                    + self.dynamic_recovery_alignment_creep_clearance_tolerance_m
                    >= previous_clearance
                ):
                    self._dynamic_recovery_clearance_trend_steps += 1
                else:
                    self._dynamic_recovery_clearance_trend_steps = 0
                self._dynamic_recovery_previous_scan_clearance = (
                    recovery_scan_clearance
                )
                recovery_clearance_trend_ready = bool(
                    np.isfinite(recovery_scan_clearance)
                    and recovery_scan_clearance
                    >= self.dynamic_recovery_alignment_creep_minimum_scan_clearance_m
                    and self._dynamic_recovery_clearance_trend_steps
                    >= self.dynamic_recovery_alignment_creep_clearance_hold_steps
                )
            else:
                recovery_clearance_trend_ready = True
        away_heading_error = float(guard_result.get(
            "dynamic_obstacle_away_heading_error_rad", 0.0
        ))
        obstacle_bearing = float(
            guard_result.get("dynamic_obstacle_bearing_rad", 0.0)
        )
        geometric_forward_escape = bool(
            self.dynamic_escape_uncertainty_fusion_enabled
            and reactive_escape_allowed
            and np.isfinite(obstacle_bearing)
            and abs(obstacle_bearing) <= 0.5 * np.pi
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        )
        corridor_commit_active = bool(
            self.dynamic_escape_corridor_enabled
            and self._dynamic_escape_corridor_remaining > 0
            and context.get(
                "probabilistic_obstacle_active_avoidance_enabled",
                False,
            )
            and guard_result.get(
                "dynamic_obstacle_scan_flow_match", False
            )
            and guard_result.get("temporal_scan_valid", False)
            and not guard_result.get("emergency_stop", False)
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        )
        corridor_escape_active = bool(
            corridor_commit_active or geometric_forward_escape
        )
        corridor_turning = False
        committed_geometric_escape = bool(
            geometric_forward_escape
            and self._dynamic_escape_direction_commit_remaining > 0
            and self._dynamic_escape_direction_commit_values is not None
        )
        reactive_reverse_required = bool(
            reactive_escape_allowed
            and self.dynamic_escape_reverse_speed > 0.0
            and abs(away_heading_error) > 0.5 * np.pi
            and not geometric_forward_escape
        )
        vetted_forward_reverse_veto = False
        reverse_escape = False
        hard_fallback_planner_control = bool(
            self.dynamic_escape_preserve_hard_fallback_planner_control
            and dynamic_escape_allowed
            and context.get(
                "probabilistic_obstacle_active_fallback_used", False
            )
            and context.get(
                "probabilistic_obstacle_hard_violation", False
            )
            and str(
                context.get(
                    "probabilistic_obstacle_active_fallback_kind",
                    "",
                )
            )
            in {
                "minimum_accumulated_risk_active_candidate",
                "minimum_risk_active_candidate",
                "risk_equivalent_forward_progress_candidate",
            }
            and "v_cmd" in self.action_spec.names
            and float(values[self.action_spec.index("v_cmd")]) >= 0.0
        )
        if dynamic_escape_allowed or corridor_commit_active:
            if self.dynamic_escape_corridor_enabled and (
                geometric_forward_escape or corridor_commit_active
            ):
                if self._dynamic_escape_corridor_remaining <= 0:
                    self._dynamic_escape_corridor_turn_sign = (
                        -1.0 if obstacle_bearing >= 0.0 else 1.0
                    )
                    self._dynamic_escape_corridor_remaining = (
                        self.dynamic_escape_corridor_commit_steps
                    )
                    self._dynamic_escape_corridor_turn_remaining = min(
                        self.dynamic_escape_corridor_turn_steps,
                        self._dynamic_escape_corridor_remaining,
                    )
                v_index = self.action_spec.index("v_cmd")
                omega_index = self.action_spec.index("omega_cmd")
                values[v_index] = min(
                    self.dynamic_escape_corridor_speed,
                    self.action_spec.upper[v_index],
                )
                corridor_turning = (
                    self._dynamic_escape_corridor_turn_remaining > 0
                )
                if corridor_turning:
                    values[omega_index] = (
                        self.action_spec.upper[omega_index]
                        if self._dynamic_escape_corridor_turn_sign > 0.0
                        else self.action_spec.lower[omega_index]
                    )
                    self._dynamic_escape_corridor_turn_remaining -= 1
                else:
                    values[omega_index] = 0.0
                self._dynamic_escape_corridor_remaining -= 1
                values = self.action_spec.clip(values)
                reverse_escape = False
                reason = "dynamic_corridor_escape"
            elif committed_geometric_escape:
                values = self.action_spec.clip(
                    self._dynamic_escape_direction_commit_values
                )
                self._dynamic_escape_direction_commit_remaining -= 1
                reverse_escape = False
                reason = "dynamic_active_escape"
            elif geometric_forward_escape:
                # A front-half-plane obstacle with short TTC should be
                # cleared by a committed forward arc.  Reversing along the
                # closing line caused the repeated reverse/turn failure mode.
                v_index = self.action_spec.index("v_cmd")
                omega_index = self.action_spec.index("omega_cmd")
                values[v_index] = min(
                    self.dynamic_escape_max_speed,
                    self.action_spec.upper[v_index],
                )
                turn_sign = -1.0 if obstacle_bearing >= 0.0 else 1.0
                values[omega_index] = (
                    self.action_spec.upper[omega_index]
                    if turn_sign > 0.0
                    else self.action_spec.lower[omega_index]
                )
                values = self.action_spec.clip(values)
                reverse_escape = False
                if self.dynamic_escape_direction_commit_steps > 0:
                    self._dynamic_escape_direction_commit_values = (
                        values.copy()
                    )
                    self._dynamic_escape_direction_commit_remaining = (
                        self.dynamic_escape_direction_commit_steps - 1
                    )
                reason = "dynamic_active_escape"
            else:
                use_vetted_planner_control = (
                    self.dynamic_escape_use_vetted_planner_control
                    or hard_fallback_planner_control
                )
                if use_vetted_planner_control and (
                    self.dynamic_escape_veto_forward_when_reactive_reverse
                    and fresh_reactive_escape_allowed
                    and reactive_reverse_required
                    and "v_cmd" in self.action_spec.names
                ):
                    proposed_v = float(
                        values[self.action_spec.index("v_cmd")]
                    )
                    vetted_forward_reverse_veto = bool(proposed_v > 0.0)
                    use_vetted_planner_control = (
                        not vetted_forward_reverse_veto
                    )
                if use_vetted_planner_control:
                    # Preserve the planner command only when it was already
                    # risk-vetted; otherwise use the reactive escape below.
                    values = self.action_spec.clip(proposed.values)
                    self._dynamic_escape_hold_remaining = 0
                    self._dynamic_escape_hold_values = None
                    if "v_cmd" in self.action_spec.names:
                        reverse_escape = bool(
                            values[self.action_spec.index("v_cmd")] < 0.0
                        )
                    reason = "dynamic_active_escape"
                elif (
                    held_reactive_escape_allowed
                    and not fresh_reactive_escape_allowed
                ):
                    values = self.action_spec.clip(
                        self._dynamic_escape_hold_values
                    )
                    if "v_cmd" in self.action_spec.names:
                        reverse_escape = bool(
                            values[self.action_spec.index("v_cmd")] < 0.0
                        )
                    self._dynamic_escape_hold_remaining -= 1
                    reason = "dynamic_active_escape"
                else:
                    reverse_escape = bool(reactive_reverse_required)
                    if "v_cmd" in self.action_spec.names:
                        index = self.action_spec.index("v_cmd")
                        if reverse_escape:
                            values[index] = max(
                                -self.dynamic_escape_reverse_speed,
                                self.action_spec.lower[index],
                            )
                        else:
                            minimum_speed = (
                                self.dynamic_escape_min_speed
                                if reactive_escape_allowed
                                else 0.0
                            )
                            values[index] = min(
                                max(minimum_speed, values[index]),
                                self.dynamic_escape_max_speed,
                            )
                    if (
                        reactive_escape_allowed
                        and "omega_cmd" in self.action_spec.names
                    ):
                        omega_index = self.action_spec.index("omega_cmd")
                        heading_error = away_heading_error
                        if reverse_escape:
                            heading_error = float(np.arctan2(
                                np.sin(away_heading_error - np.pi),
                                np.cos(away_heading_error - np.pi),
                            ))
                        values[omega_index] = np.clip(
                            self.dynamic_escape_turn_gain
                            * heading_error,
                            self.action_spec.lower[omega_index],
                            self.action_spec.upper[omega_index],
                        )
                    if (
                        fresh_reactive_escape_allowed
                        and self.dynamic_escape_hold_enabled
                    ):
                        self._dynamic_escape_hold_values = values.copy()
                        self._dynamic_escape_hold_remaining = (
                            self.dynamic_escape_hold_steps
                        )
                    reason = "dynamic_active_escape"
        elif self._dynamic_recovery_active:
            heading_error = float(
                context.get(
                    "target_bearing_error",
                    context.get(
                        "terminal_bearing_error", float("nan")
                    ),
                )
            )
            goal_distance = recovery_goal_distance
            proposed_v = (
                float(values[self.action_spec.index("v_cmd")])
                if "v_cmd" in self.action_spec.names
                else 0.0
            )
            if goal_distance <= (
                self.dynamic_recovery_goal_release_distance_m
            ):
                self._dynamic_recovery_active = False
                self._dynamic_escape_seen = False
                self._dynamic_recovery_clear_steps = 0
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
                self._dynamic_recovery_alignment_creep_latched = False
                self._dynamic_recovery_previous_scan_clearance = float("nan")
                self._dynamic_recovery_clearance_trend_steps = 0
                self._dynamic_recovery_progress_watch_active = False
                self._dynamic_recovery_progress_watch_remaining = 0
                self._dynamic_recovery_progress_watch_samples = []
                recovery_mode = "goal_release"
            elif not np.isfinite(heading_error):
                self._dynamic_recovery_active = False
                self._dynamic_recovery_clear_steps = 0
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
                self._dynamic_recovery_alignment_creep_latched = False
                self._dynamic_recovery_previous_scan_clearance = float("nan")
                self._dynamic_recovery_clearance_trend_steps = 0
                self._dynamic_recovery_progress_watch_active = False
                self._dynamic_recovery_progress_watch_remaining = 0
                self._dynamic_recovery_progress_watch_samples = []
                recovery_mode = "missing_heading_abort"
            elif abs(heading_error) > (
                self.dynamic_recovery_heading_tolerance_rad
            ):
                if "v_cmd" in self.action_spec.names:
                    v_index = self.action_spec.index("v_cmd")
                    creep_heading_eligible = bool(
                        self._dynamic_recovery_alignment_creep_latched
                        or abs(heading_error)
                        >= self.dynamic_recovery_alignment_creep_minimum_heading_error_rad
                    )
                    if (
                        self.dynamic_recovery_alignment_creep_enabled
                        and creep_heading_eligible
                        and recovery_clearance_trend_ready
                    ):
                        self._dynamic_recovery_alignment_creep_latched = True
                        values[v_index] = np.clip(
                            self.dynamic_recovery_alignment_creep_speed,
                            self.action_spec.lower[v_index],
                            min(
                                self.action_spec.upper[v_index],
                                self.dynamic_escape_max_speed,
                            ),
                        )
                        recovery_speed_floor = float(values[v_index])
                        recovery_alignment_creep_active = True
                    else:
                        values[v_index] = 0.0
                if "omega_cmd" in self.action_spec.names:
                    omega_index = self.action_spec.index("omega_cmd")
                    values[omega_index] = np.clip(
                        self.dynamic_recovery_turn_gain * heading_error,
                        self.action_spec.lower[omega_index],
                        self.action_spec.upper[omega_index],
                    )
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
                recovery_mode = (
                    "align_creep"
                    if recovery_alignment_creep_active
                    else "align"
                )
                reason = (
                    "dynamic_recovery_align_creep"
                    if recovery_alignment_creep_active
                    else "dynamic_recovery_align"
                )
            elif not self.dynamic_recovery_translation_enabled:
                self._dynamic_recovery_active = False
                self._dynamic_escape_seen = False
                self._dynamic_recovery_clear_steps = 0
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
                self._dynamic_recovery_alignment_creep_latched = False
                self._dynamic_recovery_previous_scan_clearance = float("nan")
                self._dynamic_recovery_clearance_trend_steps = 0
                self._dynamic_recovery_progress_watch_active = False
                self._dynamic_recovery_progress_watch_remaining = 0
                self._dynamic_recovery_progress_watch_samples = []
                recovery_mode = "aligned_release"
            else:
                self._dynamic_recovery_advance_steps += 1
                recovery_speed_floor = self.dynamic_recovery_min_speed
                recovery_risk_ramp_fraction = 1.0
                if self.dynamic_recovery_progressive_acceleration_enabled:
                    ramp_denominator = max(
                        1, self.dynamic_recovery_ramp_steps - 1
                    )
                    time_fraction = float(np.clip(
                        (self._dynamic_recovery_advance_steps - 1)
                        / ramp_denominator,
                        0.0,
                        1.0,
                    ))
                    risk_denominator = (
                        self.dynamic_recovery_abort_probability
                        - self.dynamic_recovery_entry_probability
                    )
                    recovery_risk_ramp_fraction = float(np.clip(
                        (
                            self.dynamic_recovery_abort_probability
                            - selected_probability
                        ) / max(risk_denominator, 1.0e-12),
                        0.0,
                        1.0,
                    ))
                    ramp_fraction = min(
                        time_fraction, recovery_risk_ramp_fraction
                    )
                    recovery_speed_floor = float(
                        self.dynamic_recovery_ramp_start_speed
                        + ramp_fraction
                        * (
                            self.dynamic_recovery_min_speed
                            - self.dynamic_recovery_ramp_start_speed
                        )
                    )
                if "v_cmd" in self.action_spec.names:
                    v_index = self.action_spec.index("v_cmd")
                    values[v_index] = min(
                        max(
                            recovery_speed_floor,
                            values[v_index],
                        ),
                        self.dynamic_escape_max_speed,
                    )
                if "omega_cmd" in self.action_spec.names:
                    omega_index = self.action_spec.index("omega_cmd")
                    values[omega_index] = np.clip(
                        self.dynamic_recovery_turn_gain * heading_error,
                        self.action_spec.lower[omega_index],
                        self.action_spec.upper[omega_index],
                    )
                if proposed_v >= 0.8 * self.dynamic_recovery_min_speed:
                    self._dynamic_recovery_release_count += 1
                else:
                    self._dynamic_recovery_release_count = 0
                recovery_mode = "advance"
                reason = "dynamic_recovery_advance"
                recovery_forward_commit_active = bool(
                    self._dynamic_recovery_advance_steps
                    <= self.dynamic_recovery_minimum_forward_commit_steps
                )
                if (
                    self._dynamic_recovery_release_count
                    >= self.dynamic_recovery_release_steps
                    and self._dynamic_recovery_advance_steps
                    >= self.dynamic_recovery_minimum_forward_commit_steps
                ):
                    self._dynamic_recovery_active = False
                    self._dynamic_recovery_clear_steps = 0
                    self._dynamic_recovery_release_count = 0
                    self._dynamic_recovery_advance_steps = 0
                    self._dynamic_recovery_alignment_creep_latched = False
                    self._dynamic_recovery_previous_scan_clearance = float(
                        "nan"
                    )
                    self._dynamic_recovery_clearance_trend_steps = 0
                    if (
                        self.dynamic_recovery_progress_watch_enabled
                        and self._dynamic_recovery_progress_reentry_count
                        < self.dynamic_recovery_progress_watch_max_reentries
                        and np.isfinite(goal_distance)
                        and goal_distance
                        > self.dynamic_recovery_goal_release_distance_m
                    ):
                        self._dynamic_escape_seen = True
                        self._dynamic_recovery_progress_watch_active = True
                        self._dynamic_recovery_progress_watch_remaining = (
                            self.dynamic_recovery_progress_watch_timeout_steps
                        )
                        self._dynamic_recovery_progress_watch_samples = [
                            goal_distance
                        ]
                        recovery_mode = "planner_release_watch"
                    else:
                        self._dynamic_escape_seen = False
                        self._dynamic_recovery_progress_watch_active = False
                        self._dynamic_recovery_progress_watch_remaining = 0
                        self._dynamic_recovery_progress_watch_samples = []
                        recovery_mode = "planner_release"
        elif deadline_supervisor_active:
            if "v_cmd" in self.action_spec.names:
                v_index = self.action_spec.index("v_cmd")
                deadline_speed_floor = float(min(
                    self.dynamic_escape_max_speed,
                    max(
                        self.dynamic_deadline_speed_floor,
                        self.dynamic_deadline_speed_margin
                        * deadline_required_speed,
                    ),
                ))
                values[v_index] = max(values[v_index], deadline_speed_floor)
            reason = "dynamic_deadline_supervisor"
        elif bool(guard_result.get("emergency_stop", False)):
            if "v_cmd" in self.action_spec.names:
                values[self.action_spec.index("v_cmd")] = 0.0
        elif reason == "front_soft_block":
            # The legacy ROS bridge has a stateful, separately tested creep
            # recovery.  The Python-3 research runtime does not.  Treating a
            # soft block as scale=1 would therefore command full translation
            # inside the body stopping envelope.  Stop translation while
            # retaining omega, so MPPI can turn away without bypassing safety.
            if "v_cmd" in self.action_spec.names:
                index = self.action_spec.index("v_cmd")
                values[index] = min(
                    max(0.0, values[index]), self.front_soft_block_max_speed
                )
        elif bool(guard_result.get("should_slow_down", False)):
            if "v_cmd" in self.action_spec.names:
                index = self.action_spec.index("v_cmd")
                values[index] = max(0.0, values[index]) * float(guard_result.get("slow_scale", 1.0))
        if not dynamic_escape_allowed and not corridor_commit_active:
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            self._dynamic_escape_corridor_remaining = 0
            self._dynamic_escape_corridor_turn_remaining = 0
            self._dynamic_escape_corridor_turn_sign = 0.0
        executed = ControlCommand(values, proposed.timestamp, "safety_arbitration")
        overridden = not np.allclose(executed.values, proposed.values, rtol=0.0, atol=1e-12)
        diagnostics = dict(guard_result)
        diagnostics["dynamic_escape_allowed"] = dynamic_escape_allowed
        diagnostics["dynamic_escape_reactive"] = (
            reactive_escape_allowed
        )
        diagnostics["dynamic_escape_uncertainty_fusion"] = (
            uncertainty_fusion_escape_allowed
        )
        diagnostics["dynamic_escape_geometric_forward"] = (
            geometric_forward_escape
        )
        diagnostics["dynamic_escape_corridor_active"] = (
            corridor_escape_active
        )
        diagnostics["dynamic_escape_corridor_turning"] = (
            corridor_turning
        )
        diagnostics["dynamic_escape_corridor_remaining"] = int(
            self._dynamic_escape_corridor_remaining
        )
        diagnostics["dynamic_escape_corridor_turn_remaining"] = int(
            self._dynamic_escape_corridor_turn_remaining
        )
        diagnostics["dynamic_escape_direction_commit_remaining"] = int(
            self._dynamic_escape_direction_commit_remaining
        )
        diagnostics["dynamic_escape_held"] = bool(
            held_reactive_escape_allowed
            and not fresh_reactive_escape_allowed
        )
        diagnostics["dynamic_escape_hold_remaining"] = int(
            self._dynamic_escape_hold_remaining
        )
        diagnostics["dynamic_escape_reverse"] = bool(
            dynamic_escape_allowed
            and reactive_escape_allowed
            and reverse_escape
        )
        diagnostics["dynamic_escape_selected_probability"] = (
            selected_probability
        )
        diagnostics["dynamic_escape_stop_probability"] = stop_probability
        diagnostics["dynamic_escape_selected_probability_mass"] = (
            selected_probability_mass
        )
        diagnostics["dynamic_escape_stop_probability_mass"] = (
            stop_probability_mass
        )
        diagnostics["dynamic_escape_probability_mass_fallback"] = (
            saturated_probability_mass_improved
        )
        diagnostics["dynamic_escape_probability_mass_enabled"] = (
            self.dynamic_escape_probability_mass_enabled
        )
        diagnostics["dynamic_escape_probability_mass_relative_threshold"] = (
            self.dynamic_escape_min_probability_mass_relative_improvement
        )
        diagnostics["dynamic_escape_vetted_planner_control"] = bool(
            dynamic_escape_allowed
            and (
                self.dynamic_escape_use_vetted_planner_control
                or hard_fallback_planner_control
            )
            and not vetted_forward_reverse_veto
        )
        diagnostics[
            "dynamic_escape_hard_fallback_planner_control"
        ] = bool(
            hard_fallback_planner_control
            and not vetted_forward_reverse_veto
        )
        diagnostics["dynamic_escape_vetted_forward_reverse_veto"] = bool(
            vetted_forward_reverse_veto
        )
        diagnostics["dynamic_recovery_enabled"] = (
            self.dynamic_recovery_enabled
        )
        diagnostics["planner_temporal_escape_active"] = (
            planner_temporal_escape_active
        )
        diagnostics["dynamic_recovery_translation_enabled"] = (
            self.dynamic_recovery_translation_enabled
        )
        diagnostics[
            "dynamic_recovery_progressive_acceleration_enabled"
        ] = self.dynamic_recovery_progressive_acceleration_enabled
        diagnostics["dynamic_recovery_ramp_start_speed"] = (
            self.dynamic_recovery_ramp_start_speed
        )
        diagnostics["dynamic_recovery_ramp_steps"] = (
            self.dynamic_recovery_ramp_steps
        )
        diagnostics["dynamic_recovery_minimum_forward_commit_steps"] = (
            self.dynamic_recovery_minimum_forward_commit_steps
        )
        diagnostics["dynamic_recovery_progress_watch_enabled"] = (
            self.dynamic_recovery_progress_watch_enabled
        )
        diagnostics["dynamic_recovery_progress_watch_steps"] = (
            self.dynamic_recovery_progress_watch_steps
        )
        diagnostics[
            "dynamic_recovery_progress_watch_minimum_progress_m"
        ] = self.dynamic_recovery_progress_watch_minimum_progress_m
        diagnostics["dynamic_recovery_progress_watch_max_reentries"] = (
            self.dynamic_recovery_progress_watch_max_reentries
        )
        diagnostics["dynamic_recovery_progress_watch_timeout_steps"] = (
            self.dynamic_recovery_progress_watch_timeout_steps
        )
        diagnostics["dynamic_recovery_alignment_creep_enabled"] = (
            self.dynamic_recovery_alignment_creep_enabled
        )
        diagnostics["dynamic_recovery_alignment_creep_speed"] = (
            self.dynamic_recovery_alignment_creep_speed
        )
        diagnostics[
            "dynamic_recovery_alignment_creep_minimum_heading_error_rad"
        ] = self.dynamic_recovery_alignment_creep_minimum_heading_error_rad
        diagnostics[
            "dynamic_recovery_alignment_creep_clearance_trend_enabled"
        ] = self.dynamic_recovery_alignment_creep_clearance_trend_enabled
        diagnostics[
            "dynamic_recovery_alignment_creep_clearance_tolerance_m"
        ] = self.dynamic_recovery_alignment_creep_clearance_tolerance_m
        diagnostics[
            "dynamic_recovery_alignment_creep_clearance_hold_steps"
        ] = self.dynamic_recovery_alignment_creep_clearance_hold_steps
        diagnostics[
            "dynamic_recovery_alignment_creep_minimum_scan_clearance_m"
        ] = self.dynamic_recovery_alignment_creep_minimum_scan_clearance_m
        diagnostics["dynamic_recovery_minimum_heading_error_rad"] = (
            self.dynamic_recovery_minimum_heading_error_rad
        )
        diagnostics["dynamic_recovery_active"] = (
            self._dynamic_recovery_active
        )
        diagnostics["dynamic_recovery_pending"] = (
            self._dynamic_escape_seen
        )
        diagnostics["dynamic_recovery_mode"] = recovery_mode
        diagnostics["dynamic_recovery_guard_clear"] = (
            recovery_guard_clear
        )
        diagnostics["dynamic_recovery_clear_steps"] = (
            self._dynamic_recovery_clear_steps
        )
        diagnostics["dynamic_recovery_release_count"] = (
            self._dynamic_recovery_release_count
        )
        diagnostics["dynamic_recovery_advance_count"] = (
            self._dynamic_recovery_advance_steps
        )
        diagnostics["dynamic_recovery_speed_floor"] = (
            recovery_speed_floor
        )
        diagnostics["dynamic_recovery_risk_ramp_fraction"] = (
            recovery_risk_ramp_fraction
        )
        diagnostics["dynamic_recovery_forward_commit_active"] = (
            recovery_forward_commit_active
        )
        diagnostics["dynamic_recovery_alignment_creep_active"] = (
            recovery_alignment_creep_active
        )
        diagnostics["dynamic_recovery_scan_clearance_m"] = (
            recovery_scan_clearance
        )
        diagnostics["dynamic_recovery_clearance_trend_steps"] = (
            self._dynamic_recovery_clearance_trend_steps
        )
        diagnostics["dynamic_recovery_clearance_trend_ready"] = (
            recovery_clearance_trend_ready
        )
        diagnostics["dynamic_recovery_progress_watch_active"] = (
            self._dynamic_recovery_progress_watch_active
        )
        diagnostics["dynamic_recovery_progress_watch_remaining"] = int(
            self._dynamic_recovery_progress_watch_remaining
        )
        diagnostics["dynamic_recovery_progress_watch_sample_count"] = len(
            self._dynamic_recovery_progress_watch_samples
        )
        diagnostics["dynamic_recovery_progress_watch_progress_m"] = (
            recovery_progress_watch_progress_m
        )
        diagnostics["dynamic_recovery_progress_watch_triggered"] = (
            recovery_progress_watch_triggered
        )
        diagnostics["dynamic_recovery_progress_reentry_count"] = int(
            self._dynamic_recovery_progress_reentry_count
        )
        diagnostics["dynamic_deadline_supervisor_enabled"] = (
            self.dynamic_deadline_supervisor_enabled
        )
        diagnostics["dynamic_deadline_supervisor_active"] = (
            deadline_supervisor_active
        )
        diagnostics["dynamic_deadline_conflict_seen"] = (
            self._dynamic_deadline_conflict_seen
        )
        diagnostics["dynamic_deadline_guard_clear"] = deadline_guard_clear
        diagnostics["dynamic_deadline_clear_steps"] = int(
            self._dynamic_deadline_clear_steps
        )
        diagnostics["dynamic_deadline_decision_count"] = int(
            self._dynamic_deadline_decision_count
        )
        diagnostics["dynamic_deadline_steps_remaining"] = int(
            deadline_steps_remaining
        )
        diagnostics["dynamic_deadline_available_steps"] = int(
            deadline_available_steps
        )
        diagnostics["dynamic_deadline_urgency_available_steps"] = int(
            deadline_available_steps
        )
        diagnostics["dynamic_deadline_authority_available_steps"] = int(
            deadline_authority_available_steps
        )
        diagnostics["dynamic_deadline_urgency_exhausted"] = (
            deadline_urgency_exhausted
        )
        diagnostics["dynamic_deadline_required_speed"] = float(
            deadline_required_speed
        )
        diagnostics["dynamic_deadline_speed_floor"] = float(
            deadline_speed_floor
        )
        return SafetyDecision(
            proposed, executed, overridden, reason, diagnostics
        )
