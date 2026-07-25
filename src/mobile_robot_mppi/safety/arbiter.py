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
        self.dynamic_recovery_alignment_creep_enabled = bool(
            self.config.get(
                "dynamic_recovery_alignment_creep_enabled", False
            )
        )
        self.dynamic_recovery_alignment_creep_speed = float(
            self.config.get("dynamic_recovery_alignment_creep_speed", 0.0)
        )
        self.dynamic_recovery_goal_release_distance_m = float(
            self.config.get(
                "dynamic_recovery_goal_release_distance_m", 0.45
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
        if self.dynamic_recovery_alignment_creep_enabled and (
            self.dynamic_recovery_alignment_creep_speed <= 0.0
            or self.dynamic_recovery_alignment_creep_speed
            > self.dynamic_recovery_min_speed
        ):
            raise ValueError(
                "dynamic recovery alignment creep parameters are invalid"
            )
        self.reset()

    def reset(self):
        self._dynamic_escape_seen = False
        self._dynamic_escape_hold_remaining = 0
        self._dynamic_escape_hold_values = None
        self._dynamic_recovery_active = False
        self._dynamic_recovery_clear_steps = 0
        self._dynamic_recovery_release_count = 0
        self._dynamic_recovery_advance_steps = 0

    def arbitrate(
        self,
        proposed: ControlCommand,
        guard_result: Mapping[str, object],
        planning_context: Mapping[str, object] = None,
    ):
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
            fresh_reactive_escape_allowed or held_reactive_escape_allowed
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
        if dynamic_escape_allowed or planner_temporal_escape_active:
            self._dynamic_escape_seen = True
            self._dynamic_recovery_active = False
            self._dynamic_recovery_clear_steps = 0
            self._dynamic_recovery_release_count = 0
            self._dynamic_recovery_advance_steps = 0
        elif self.dynamic_recovery_enabled and self._dynamic_escape_seen:
            recovery_heading_error = float(
                context.get(
                    "target_bearing_error",
                    context.get(
                        "terminal_bearing_error", float("nan")
                    ),
                )
            )
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
            if self._dynamic_recovery_active and recovery_abort:
                self._dynamic_recovery_active = False
                self._dynamic_recovery_clear_steps = 0
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
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
                        recovery_mode = "entered"
                    else:
                        self._dynamic_escape_seen = False
                        self._dynamic_recovery_clear_steps = 0
                        self._dynamic_recovery_advance_steps = 0
                        recovery_mode = "not_needed"
        reverse_escape = False
        if dynamic_escape_allowed:
            if self.dynamic_escape_use_vetted_planner_control:
                # The proposed command is the first action of the trajectory
                # already evaluated by the probabilistic planner.  Do not
                # replace it with a bearing-only heuristic whose risk was never
                # scored against the obstacle forecast.  Planner-controlled
                # mode also disables cross-cycle command holding because a
                # previously vetted command is not certified for a new state.
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
                away_heading_error = float(guard_result.get(
                    "dynamic_obstacle_away_heading_error_rad", 0.0
                ))
                reverse_escape = bool(
                    reactive_escape_allowed
                    and self.dynamic_escape_reverse_speed > 0.0
                    and abs(away_heading_error) > 0.5 * np.pi
                )
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
            goal_distance = float(
                context.get("terminal_control_distance", float("inf"))
            )
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
                recovery_mode = "goal_release"
            elif not np.isfinite(heading_error):
                self._dynamic_recovery_active = False
                self._dynamic_recovery_clear_steps = 0
                self._dynamic_recovery_release_count = 0
                self._dynamic_recovery_advance_steps = 0
                recovery_mode = "missing_heading_abort"
            elif abs(heading_error) > (
                self.dynamic_recovery_heading_tolerance_rad
            ):
                if "v_cmd" in self.action_spec.names:
                    v_index = self.action_spec.index("v_cmd")
                    if self.dynamic_recovery_alignment_creep_enabled:
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
                    self._dynamic_escape_seen = False
                    self._dynamic_recovery_clear_steps = 0
                    self._dynamic_recovery_release_count = 0
                    self._dynamic_recovery_advance_steps = 0
                    recovery_mode = "planner_release"
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
        executed = ControlCommand(values, proposed.timestamp, "safety_arbitration")
        overridden = not np.allclose(executed.values, proposed.values, rtol=0.0, atol=1e-12)
        diagnostics = dict(guard_result)
        diagnostics["dynamic_escape_allowed"] = dynamic_escape_allowed
        diagnostics["dynamic_escape_reactive"] = (
            reactive_escape_allowed
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
            and self.dynamic_escape_use_vetted_planner_control
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
        diagnostics["dynamic_recovery_alignment_creep_enabled"] = (
            self.dynamic_recovery_alignment_creep_enabled
        )
        diagnostics["dynamic_recovery_alignment_creep_speed"] = (
            self.dynamic_recovery_alignment_creep_speed
        )
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
        return SafetyDecision(
            proposed, executed, overridden, reason, diagnostics
        )
