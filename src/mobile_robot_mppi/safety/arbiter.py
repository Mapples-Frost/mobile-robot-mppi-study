import math
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
        self.physical_front_speed_governor_enabled = bool(
            self.config.get("physical_front_speed_governor_enabled", False)
        )
        self.physical_front_speed_governor_clearance_m = float(
            self.config.get("physical_front_speed_governor_clearance_m", 0.35)
        )
        self.physical_front_speed_governor_reaction_s = float(
            self.config.get("physical_front_speed_governor_reaction_s", 0.50)
        )
        self.physical_front_speed_governor_deceleration_mps2 = float(
            self.config.get(
                "physical_front_speed_governor_deceleration_mps2", 0.80
            )
        )
        # The raw near-body guard is intentionally omnidirectional, but the
        # final control boundary also knows the commanded translation sign. A
        # rear-only return cannot be struck while the base keeps translating
        # forward. This opt-in rule releases only that case; reverse motion,
        # front/side points and malformed evidence remain fail-closed.
        self.directional_motion_guard_enabled = bool(
            self.config.get("directional_motion_guard_enabled", False)
        )
        self.directional_forward_protected_half_angle_deg = float(
            self.config.get(
                "directional_forward_protected_half_angle_deg", 100.0
            )
        )
        self.directional_motion_minimum_speed_mps = float(
            self.config.get("directional_motion_minimum_speed_mps", 0.02)
        )
        # A rear-only close approach should make the robot leave the person,
        # even when the stochastic planner happens to propose reverse on that
        # cycle.  This synthesis is opt-in and requires a freshly observed
        # open front sector; mixed/front evidence remains fail-closed.
        self.rear_pass_through_force_forward_enabled = bool(
            self.config.get(
                "rear_pass_through_force_forward_enabled", False
            )
        )
        self.rear_pass_through_min_front_clearance_m = float(
            self.config.get(
                "rear_pass_through_min_front_clearance_m", 0.90
            )
        )
        self.rear_pass_through_min_forward_speed_mps = float(
            self.config.get(
                "rear_pass_through_min_forward_speed_mps", 0.35
            )
        )
        self.rear_pass_through_max_omega_radps = float(
            self.config.get("rear_pass_through_max_omega_radps", 0.30)
        )
        self.rear_pass_through_min_turn_omega_radps = float(
            self.config.get(
                "rear_pass_through_min_turn_omega_radps", 0.15
            )
        )
        self.rear_pass_through_force_straight_enabled = bool(
            self.config.get(
                "rear_pass_through_force_straight_enabled", False
            )
        )
        self.rear_pass_through_direction_release_steps = int(
            self.config.get(
                "rear_pass_through_direction_release_steps", 3
            )
        )
        if self.physical_front_speed_governor_clearance_m < 0.0:
            raise ValueError(
                "physical_front_speed_governor_clearance_m must be non-negative"
            )
        if self.physical_front_speed_governor_reaction_s < 0.0:
            raise ValueError(
                "physical_front_speed_governor_reaction_s must be non-negative"
            )
        if self.physical_front_speed_governor_deceleration_mps2 <= 0.0:
            raise ValueError(
                "physical_front_speed_governor_deceleration_mps2 must be positive"
            )
        if (
            not 90.0
            <= self.directional_forward_protected_half_angle_deg
            < 180.0
            or self.directional_motion_minimum_speed_mps < 0.0
            or self.rear_pass_through_min_front_clearance_m <= 0.0
            or self.rear_pass_through_min_forward_speed_mps <= 0.0
            or self.rear_pass_through_max_omega_radps < 0.0
            or self.rear_pass_through_min_turn_omega_radps < 0.0
            or self.rear_pass_through_min_turn_omega_radps
            > self.rear_pass_through_max_omega_radps
            or self.rear_pass_through_direction_release_steps < 1
        ):
            raise ValueError("directional motion guard parameters are invalid")
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
        self.dynamic_escape_use_vetted_emergency_candidate = bool(
            self.config.get(
                "dynamic_escape_use_vetted_emergency_candidate", False
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
        # --- static reverse escape (flagged, default OFF) -------------------
        # `near_body_hard_stop` fires omnidirectionally on range alone
        # (scan_guard.py:251, no angular gate).  Escape from it is gated on
        # `dynamic_obstacle_near_body_match`, which is false beside a wall, so
        # a robot that halts within `near_body_stop_radius` of static geometry
        # has no legal exit: planner-proposed reverse commands are zeroed at
        # the emergency branch below.  This permits reverse in exactly that
        # case, proposal-only and heavily bounded.  Default off; every
        # precondition fails closed on missing evidence.
        self.static_reverse_escape_enabled = bool(
            self.config.get("static_reverse_escape_enabled", False)
        )
        self.static_reverse_escape_hold_last_proposal_enabled = bool(
            self.config.get(
                "static_reverse_escape_hold_last_proposal_enabled", False
            )
        )
        self.static_reverse_escape_hold_at_speed_cap_enabled = bool(
            self.config.get(
                "static_reverse_escape_hold_at_speed_cap_enabled", False
            )
        )
        self.static_reverse_escape_max_speed = float(
            self.config.get("static_reverse_escape_max_speed", 0.10)
        )
        self.static_reverse_escape_rear_sector_deg = float(
            self.config.get("static_reverse_escape_rear_sector_deg", 120.0)
        )
        self.static_reverse_escape_min_rear_range = float(
            self.config.get("static_reverse_escape_min_rear_range", 0.55)
        )
        self.static_reverse_escape_max_consecutive_steps = int(
            self.config.get("static_reverse_escape_max_consecutive_steps", 40)
        )
        self.static_reverse_escape_cooldown_steps = int(
            self.config.get("static_reverse_escape_cooldown_steps", 20)
        )
        # A single run is bounded, but runs may resume after the cooldown.
        # This caps the cumulative authority over the whole episode: if the
        # robot needs more reverse than this to escape, stopping is correct.
        self.static_reverse_escape_max_total_steps = int(
            self.config.get("static_reverse_escape_max_total_steps", 200)
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
        self.dynamic_escape_frontal_commit_steps = int(
            self.config.get(
                "dynamic_escape_frontal_commit_steps",
                self.dynamic_escape_direction_commit_steps,
            )
        )
        # A noisy leg-cluster regression can momentarily look like a lateral
        # reversal during an otherwise head-on approach.  Require material
        # lateral speed before such a refresh is allowed to invalidate an
        # already selected passage side.  The default keeps historical callers
        # unchanged; the physical profile opts into the stricter gate.
        self.dynamic_escape_direction_refresh_minimum_lateral_speed_mps = float(
            self.config.get(
                "dynamic_escape_direction_refresh_minimum_lateral_speed_mps",
                0.0,
            )
        )
        # A physical leg cluster can swap sides for one scan even while the
        # person keeps walking in one direction.  Explicit direction changes
        # therefore need a short run of consistent estimator confirmations;
        # first-time prediction acquisition remains immediate below.
        self.dynamic_escape_direction_refresh_confirmation_steps = int(
            self.config.get(
                "dynamic_escape_direction_refresh_confirmation_steps", 1
            )
        )
        self.dynamic_escape_measured_reversal_confirmation_steps = int(
            self.config.get(
                "dynamic_escape_measured_reversal_confirmation_steps",
                self.dynamic_escape_direction_refresh_confirmation_steps,
            )
        )
        # A causal lateral velocity is the physical crossing direction.  The
        # higher-level preferred-heading heuristic may change sides as its
        # sampled passage costs move, but it must not overrule the simple
        # counter-motion contract used by the real robot: turn opposite the
        # pedestrian's measured lateral motion.  This threshold is separate
        # from the reversal gate so first acquisition can happen early while
        # later direction changes still require confirmation.
        self.dynamic_escape_crossing_minimum_lateral_speed_mps = float(
            self.config.get(
                "dynamic_escape_crossing_minimum_lateral_speed_mps",
                self.dynamic_escape_direction_refresh_minimum_lateral_speed_mps,
            )
        )
        # Temporal scan flow becomes valid one or two control frames before a
        # tracked forecast.  An opt-in physical pre-turn spends those frames
        # establishing a passage side instead of merely braking while gateway
        # and chassis latency consume the remaining TTC.
        self.dynamic_escape_temporal_preturn_enabled = bool(
            self.config.get("dynamic_escape_temporal_preturn_enabled", False)
        )
        self.dynamic_escape_temporal_preturn_speed = float(
            self.config.get("dynamic_escape_temporal_preturn_speed", 0.0)
        )
        self.dynamic_escape_temporal_preturn_max_bearing_rad = float(
            self.config.get(
                "dynamic_escape_temporal_preturn_max_bearing_rad",
                0.5 * np.pi,
            )
        )
        self.dynamic_escape_coast_direction_lock_enabled = bool(
            self.config.get(
                "dynamic_escape_coast_direction_lock_enabled", False
            )
        )
        self.dynamic_escape_geometric_single_commit_enabled = bool(
            self.config.get(
                "dynamic_escape_geometric_single_commit_enabled", False
            )
        )
        self.dynamic_escape_geometric_rearm_clear_steps = int(
            self.config.get(
                "dynamic_escape_geometric_rearm_clear_steps", 5
            )
        )
        # After the short saturated turn, retain only enough steering to track
        # the forecast-relative passage heading.  A zero-yaw coast drove the
        # physical robot along a stale tangent; replaying full yaw indefinitely
        # produced the earlier large-circle failure.  The maximum is opt-in so
        # non-deployment configurations preserve their historical behaviour.
        self.dynamic_escape_coast_turn_gain = float(
            self.config.get("dynamic_escape_coast_turn_gain", 0.50)
        )
        self.dynamic_escape_coast_max_omega_radps = float(
            self.config.get("dynamic_escape_coast_max_omega_radps", 0.0)
        )
        self.dynamic_escape_coast_steps = int(
            self.config.get("dynamic_escape_coast_steps", -1)
        )
        self.dynamic_escape_passage_completion_enabled = bool(
            self.config.get(
                "dynamic_escape_passage_completion_enabled", False
            )
        )
        self.dynamic_escape_passage_completion_min_bearing_rad = float(
            self.config.get(
                "dynamic_escape_passage_completion_min_bearing_rad", 1.0
            )
        )
        self.dynamic_escape_passage_completion_min_front_clearance_m = float(
            self.config.get(
                "dynamic_escape_passage_completion_min_front_clearance_m",
                1.5,
            )
        )
        self.dynamic_escape_passage_completion_min_goal_counter_heading_rad = (
            float(self.config.get(
                "dynamic_escape_passage_completion_min_goal_counter_heading_rad",
                0.55,
            ))
        )
        self.dynamic_escape_frontal_entry_speed = float(
            self.config.get(
                "dynamic_escape_frontal_entry_speed",
                self.dynamic_escape_max_speed,
            )
        )
        self.dynamic_escape_persistent_front_retry_enabled = bool(
            self.config.get(
                "dynamic_escape_persistent_front_retry_enabled", False
            )
        )
        self.dynamic_escape_vetted_reverse_retry_steps = int(
            self.config.get("dynamic_escape_vetted_reverse_retry_steps", 0)
        )
        self.dynamic_escape_persistent_front_max_retries = int(
            self.config.get("dynamic_escape_persistent_front_max_retries", 0)
        )
        self.dynamic_escape_persistent_front_retry_commit_steps = int(
            self.config.get(
                "dynamic_escape_persistent_front_retry_commit_steps",
                self.dynamic_escape_frontal_commit_steps,
            )
        )
        # A passage-side commitment is useful only while it still leaves a
        # plausible route back to the goal.  The physical deployment supplies
        # a body-frame goal bearing and opts into this finite release gate;
        # infinity preserves the historical behaviour for other profiles.
        self.dynamic_escape_goal_divergence_release_rad = float(
            self.config.get(
                "dynamic_escape_goal_divergence_release_rad", float("inf")
            )
        )
        self.dynamic_escape_post_retry_reverse_hold_enabled = bool(
            self.config.get(
                "dynamic_escape_post_retry_reverse_hold_enabled", False
            )
        )
        self.dynamic_escape_reverse_goal_realign_gain = float(
            self.config.get("dynamic_escape_reverse_goal_realign_gain", 0.0)
        )
        self.dynamic_escape_reverse_goal_realign_max_omega_radps = float(
            self.config.get(
                "dynamic_escape_reverse_goal_realign_max_omega_radps", 0.0
            )
        )
        self.dynamic_escape_post_retry_side_forward_speed = float(
            self.config.get(
                "dynamic_escape_post_retry_side_forward_speed", 0.0
            )
        )
        self.dynamic_escape_post_retry_side_forward_min_bearing_rad = float(
            self.config.get(
                "dynamic_escape_post_retry_side_forward_min_bearing_rad",
                float("inf"),
            )
        )
        self.dynamic_escape_post_retry_side_forward_min_surface_range_m = float(
            self.config.get(
                "dynamic_escape_post_retry_side_forward_min_surface_range_m",
                float("inf"),
            )
        )
        # Close-range dynamic escape is a bounded transaction: establish a
        # passage side in place, then create a small amount of room by reversing
        # only when the rear sector is positively observed clear.  Forward
        # motion remains forbidden by the near-body hard stop.
        self.dynamic_escape_hard_stop_enabled = bool(
            self.config.get("dynamic_escape_hard_stop_enabled", False)
        )
        self.dynamic_escape_hard_stop_turn_steps = int(
            self.config.get("dynamic_escape_hard_stop_turn_steps", 3)
        )
        self.dynamic_escape_hard_stop_reverse_steps = int(
            self.config.get("dynamic_escape_hard_stop_reverse_steps", 4)
        )
        self.dynamic_escape_hard_stop_reverse_speed = float(
            self.config.get("dynamic_escape_hard_stop_reverse_speed", 0.20)
        )
        self.dynamic_escape_hard_stop_reverse_max_omega_radps = float(
            self.config.get(
                "dynamic_escape_hard_stop_reverse_max_omega_radps",
                self.dynamic_escape_coast_max_omega_radps,
            )
        )
        self.dynamic_escape_hard_stop_reverse_turn_decay_enabled = bool(
            self.config.get(
                "dynamic_escape_hard_stop_reverse_turn_decay_enabled", False
            )
        )
        self.dynamic_escape_hard_stop_min_rear_range = float(
            self.config.get("dynamic_escape_hard_stop_min_rear_range", 0.80)
        )
        self.dynamic_escape_hard_stop_rear_sector_deg = float(
            self.config.get("dynamic_escape_hard_stop_rear_sector_deg", 120.0)
        )
        # A close-range transaction must not restart merely because a
        # different leg/person track becomes the current forecast source.
        # Historical configurations retain refresh authority; the physical
        # profile disables it for the bounded hard-stop transaction only.
        self.dynamic_escape_hard_stop_direction_refresh_enabled = bool(
            self.config.get(
                "dynamic_escape_hard_stop_direction_refresh_enabled", True
            )
        )
        # If the rear was genuinely blocked during the finite reverse phase,
        # permit one reverse-only retry when that same sector later becomes
        # positively clear.  This avoids both wasting a newly opened exit and
        # infinitely restarting the turn/reverse transaction.
        self.dynamic_escape_hard_stop_rear_clear_retry_enabled = bool(
            self.config.get(
                "dynamic_escape_hard_stop_rear_clear_retry_enabled", False
            )
        )
        self.dynamic_escape_hard_stop_rear_clear_retry_steps = int(
            self.config.get(
                "dynamic_escape_hard_stop_rear_clear_retry_steps",
                self.dynamic_escape_hard_stop_reverse_steps,
            )
        )
        # Blocked rear observations and actual reverse motion are separate
        # resources.  A blocked scan may spend a finite wait budget, but must
        # never silently consume a reverse frame that was not executed.
        self.dynamic_escape_hard_stop_rear_blocked_wait_steps = int(
            self.config.get(
                "dynamic_escape_hard_stop_rear_blocked_wait_steps",
                self.dynamic_escape_hard_stop_reverse_steps,
            )
        )
        self.dynamic_escape_hard_stop_completed_hold_steps = int(
            self.config.get(
                "dynamic_escape_hard_stop_completed_hold_steps", 1
            )
        )
        # After the finite transaction, a live obstacle centre that has moved
        # behind the lateral plane and a positively clear front corridor make
        # forward translation an opening motion.  This is intentionally an
        # opt-in physical escape; historical profiles remain fail-closed.
        self.dynamic_escape_hard_stop_side_rear_release_enabled = bool(
            self.config.get(
                "dynamic_escape_hard_stop_side_rear_release_enabled", False
            )
        )
        self.dynamic_escape_hard_stop_side_rear_release_min_bearing_rad = float(
            self.config.get(
                "dynamic_escape_hard_stop_side_rear_release_min_bearing_rad",
                math.pi,
            )
        )
        self.dynamic_escape_hard_stop_side_rear_release_min_front_range_m = float(
            self.config.get(
                "dynamic_escape_hard_stop_side_rear_release_min_front_range_m",
                float("inf"),
            )
        )
        self.dynamic_escape_hard_stop_side_rear_release_speed = float(
            self.config.get(
                "dynamic_escape_hard_stop_side_rear_release_speed", 0.0
            )
        )
        self.dynamic_escape_hard_stop_side_rear_release_hold_min_bearing_rad = float(
            self.config.get(
                "dynamic_escape_hard_stop_side_rear_release_hold_min_bearing_rad",
                self.dynamic_escape_hard_stop_side_rear_release_min_bearing_rad,
            )
        )
        self.dynamic_escape_hard_stop_side_rear_release_abort_steps = int(
            self.config.get(
                "dynamic_escape_hard_stop_side_rear_release_abort_steps", 1
            )
        )
        self.dynamic_escape_max_zero_translation_turn_steps = int(
            self.config.get(
                "dynamic_escape_max_zero_translation_turn_steps", 9
            )
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
            or self.dynamic_escape_frontal_commit_steps < 0
            or self.dynamic_escape_direction_refresh_minimum_lateral_speed_mps
            < 0.0
            or self.dynamic_escape_direction_refresh_confirmation_steps < 1
            or self.dynamic_escape_measured_reversal_confirmation_steps < 1
            or self.dynamic_escape_crossing_minimum_lateral_speed_mps < 0.0
            or self.dynamic_escape_temporal_preturn_speed < 0.0
            or self.dynamic_escape_temporal_preturn_speed
            > self.dynamic_escape_max_speed
            or not 0.0
            < self.dynamic_escape_temporal_preturn_max_bearing_rad
            <= np.pi
            or self.dynamic_escape_geometric_rearm_clear_steps < 1
            or self.dynamic_escape_coast_turn_gain <= 0.0
            or self.dynamic_escape_coast_max_omega_radps < 0.0
            or self.dynamic_escape_coast_steps < -1
            or not 0.0
            < self.dynamic_escape_passage_completion_min_bearing_rad
            <= np.pi
            or self.dynamic_escape_passage_completion_min_front_clearance_m
            <= 0.0
            or self.dynamic_escape_passage_completion_min_goal_counter_heading_rad
            < 0.0
            or self.dynamic_escape_frontal_entry_speed < 0.0
            or self.dynamic_escape_frontal_entry_speed
            > self.dynamic_escape_max_speed
            or self.dynamic_escape_hard_stop_turn_steps < 1
            or self.dynamic_escape_hard_stop_reverse_steps < 1
            or self.dynamic_escape_hard_stop_reverse_speed <= 0.0
            or self.dynamic_escape_hard_stop_reverse_max_omega_radps < 0.0
            or self.dynamic_escape_vetted_reverse_retry_steps < 0
            or self.dynamic_escape_persistent_front_max_retries < 0
            or self.dynamic_escape_persistent_front_retry_commit_steps < 0
            or self.dynamic_escape_goal_divergence_release_rad <= 0.0
            or self.dynamic_escape_reverse_goal_realign_gain < 0.0
            or self.dynamic_escape_reverse_goal_realign_max_omega_radps < 0.0
            or self.dynamic_escape_post_retry_side_forward_speed < 0.0
            or self.dynamic_escape_post_retry_side_forward_speed
            > self.dynamic_escape_max_speed
            or self.dynamic_escape_post_retry_side_forward_min_bearing_rad
            <= 0.0
            or self.dynamic_escape_post_retry_side_forward_min_surface_range_m
            <= 0.0
            or self.dynamic_escape_hard_stop_min_rear_range <= 0.0
            or self.dynamic_escape_hard_stop_rear_clear_retry_steps < 1
            or self.dynamic_escape_hard_stop_rear_blocked_wait_steps < 1
            or self.dynamic_escape_hard_stop_completed_hold_steps < 0
            or not 0.5 * np.pi
            < self.dynamic_escape_hard_stop_side_rear_release_min_bearing_rad
            <= np.pi
            or self.dynamic_escape_hard_stop_side_rear_release_min_front_range_m
            <= 0.0
            or self.dynamic_escape_hard_stop_side_rear_release_speed < 0.0
            or self.dynamic_escape_hard_stop_side_rear_release_speed
            > self.dynamic_escape_max_speed
            or not 0.5 * np.pi
            <= self.dynamic_escape_hard_stop_side_rear_release_hold_min_bearing_rad
            <= self.dynamic_escape_hard_stop_side_rear_release_min_bearing_rad
            or self.dynamic_escape_hard_stop_side_rear_release_abort_steps < 1
            or self.dynamic_escape_max_zero_translation_turn_steps < 1
            or not 0.0
            < self.dynamic_escape_hard_stop_rear_sector_deg
            <= 180.0
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
            self.static_reverse_escape_max_speed <= 0.0
            or not 0.0 < self.static_reverse_escape_rear_sector_deg <= 180.0
            or self.static_reverse_escape_min_rear_range <= 0.0
            or self.static_reverse_escape_max_consecutive_steps < 1
            or self.static_reverse_escape_cooldown_steps < 0
            or self.static_reverse_escape_max_total_steps
            < self.static_reverse_escape_max_consecutive_steps
        ):
            raise ValueError(
                "static reverse escape parameters are invalid"
            )
        if (
            self.static_reverse_escape_enabled
            and self.static_reverse_escape_min_rear_range
            <= self.static_reverse_escape_max_speed
            * self.static_reverse_escape_max_consecutive_steps
            * 0.1
        ):
            # Refuse a configuration whose own reverse budget could consume the
            # rear headroom it is checked against.  0.1 s is the control period.
            raise ValueError(
                "static reverse escape budget exceeds its required rear "
                "headroom; reduce max_speed or max_consecutive_steps"
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
        self._dynamic_escape_geometric_commit_consumed = False
        self._dynamic_escape_geometric_clear_streak = 0
        self._dynamic_escape_geometric_turn_sign = 0.0
        self._dynamic_escape_geometric_direction_prediction_backed = False
        self._dynamic_escape_measured_reversal_candidate_sign = 0.0
        self._dynamic_escape_measured_reversal_confirmation_count = 0
        self._dynamic_escape_frontal_encounter_latched = False
        self._dynamic_escape_coast_remaining = 0
        self._dynamic_escape_vetted_reverse_steps = 0
        self._dynamic_escape_persistent_front_retry_count = 0
        self._dynamic_escape_hard_stop_turn_remaining = 0
        self._dynamic_escape_hard_stop_reverse_remaining = 0
        self._dynamic_escape_hard_stop_turn_sign = 0.0
        self._dynamic_escape_hard_stop_consumed = False
        self._dynamic_escape_hard_stop_clear_streak = 0
        self._dynamic_escape_hard_stop_rear_blocked_latched = False
        self._dynamic_escape_hard_stop_rear_clear_retry_used = False
        self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = 0
        self._dynamic_escape_hard_stop_completed_hold_count = 0
        self._dynamic_escape_hard_stop_side_rear_release_latched = False
        self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
        self._rear_pass_through_turn_sign = 0.0
        self._rear_pass_through_direction_clear_streak = 0
        self._dynamic_escape_zero_translation_turn_steps = 0
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
        self._static_reverse_escape_run_steps = 0
        self._static_reverse_escape_cooldown_remaining = 0
        self._static_reverse_escape_engaged_count = 0
        self._static_reverse_escape_last_v = 0.0

    @staticmethod
    def _rear_sector_clear(guard_result, sector_deg, minimum_range):
        """True only if the rear sector is observed and demonstrably open.

        Uses the full valid-return list, not just the near-body subset, so the
        test is positive rear headroom rather than mere absence of a near-body
        return.  Returns False when the sector carries no observations at all:
        an unobserved rear is not a clear rear.
        """
        points = guard_result.get("raw_points_base", None)
        if not isinstance(points, (list, tuple)) or not points:
            return False
        half_width = math.radians(0.5 * float(sector_deg))
        observed = 0
        minimum = float("inf")
        for point in points:
            try:
                angle = float(point["base_angle"])
                distance = float(point["range"])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(angle) or not math.isfinite(distance):
                continue
            # Angular distance from the rear axis (+/- pi), wrap-safe.
            if abs(math.pi - abs(angle)) > half_width:
                continue
            observed += 1
            if distance < minimum:
                minimum = distance
        if observed <= 0:
            return False
        return minimum >= float(minimum_range)

    def _rear_near_body_clear(self, guard_result):
        return self._rear_sector_clear(
            guard_result,
            self.static_reverse_escape_rear_sector_deg,
            self.static_reverse_escape_min_rear_range,
        )

    def _static_reverse_escape_permitted(self, reason, guard_result, proposed_v):
        """All preconditions for permitting a planner-proposed reverse."""
        if not self.static_reverse_escape_enabled:
            return False
        # Only the omnidirectional near-body stop.  Every other emergency
        # cause keeps its unconditional zeroing.
        if reason != "near_body_hard_stop":
            return False
        # Proposal-only: the arbiter never synthesises reverse motion.
        if not (proposed_v < 0.0):
            return False
        # Static only.  If anything suggests the near-body obstruction is a
        # moving object, defer to the dynamic escape branch and stay stopped.
        if bool(guard_result.get("dynamic_obstacle_near_body_match", False)):
            return False
        if bool(guard_result.get("dynamic_obstacle_scan_flow_match", False)):
            return False
        # Require positive evidence that something is actually near the body;
        # without it this is not the trap and the stop stands.
        if int(guard_result.get("valid_near_body_count", 0) or 0) <= 0:
            return False
        if self._static_reverse_escape_cooldown_remaining > 0:
            return False
        if (
            self._static_reverse_escape_run_steps
            >= self.static_reverse_escape_max_consecutive_steps
        ):
            return False
        if (
            self._static_reverse_escape_engaged_count
            >= self.static_reverse_escape_max_total_steps
        ):
            return False
        return self._rear_near_body_clear(guard_result)

    @staticmethod
    def _guard_point_bearing(point):
        """Return a wrap-safe base-frame bearing or None for malformed data."""
        try:
            if "base_angle" in point:
                angle = float(point["base_angle"])
            else:
                angle = math.atan2(float(point["y"]), float(point["x"]))
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(angle):
            return None
        return float(math.atan2(math.sin(angle), math.cos(angle)))

    def _rear_pass_through_evidence(self, guard_result):
        """Return rear-only evidence for the *causal* safety trigger.

        A tracker may contain several body/leg fragments at once.  Its nearest
        centre therefore cannot release an unrelated front stop.  Directional
        release is intentionally source-scoped: near-body stops are decided
        from the actual near-body points and temporal stops from the temporal
        flow centre.  Ordinary ``front_clear`` cycles never acquire global
        rear authority from a tracker-only bearing.
        """
        if not self.directional_motion_guard_enabled:
            return False, None, ()
        reason = str(guard_result.get("reason", "front_clear"))
        if reason not in {
            "front_clear",
            "near_body_hard_stop",
            "temporal_collision_risk",
            "temporal_slowdown",
        }:
            return False, None, ()
        half_angle = math.radians(
            self.directional_forward_protected_half_angle_deg
        )
        near_body_bearings = []
        near_body_points = guard_result.get("near_body_points", ())
        if isinstance(near_body_points, (list, tuple)):
            for point in near_body_points:
                angle = self._guard_point_bearing(point)
                if angle is None:
                    return False, None, ()
                near_body_bearings.append(angle)

        temporal_valid = bool(guard_result.get("temporal_scan_valid", False))
        temporal_ttc = float(
            guard_result.get("temporal_scan_ttc_s", float("inf"))
        )
        temporal_angle = guard_result.get("temporal_scan_center_angle_rad")
        temporal_bearing = None
        if (
            temporal_valid
            and temporal_angle is not None
            and math.isfinite(float(temporal_angle))
            and temporal_ttc <= self.dynamic_escape_trigger_ttc_s
        ):
            temporal_bearing = float(temporal_angle)

        if reason == "near_body_hard_stop":
            evidence = tuple(near_body_bearings)
            sources = tuple("near_body" for _ in evidence)
            # A simultaneous front temporal threat vetoes release even when
            # every near-body return happens to lie behind the chassis.
            contradictory = (
                temporal_bearing is not None
                and abs(float(temporal_bearing)) <= half_angle
            )
        else:
            evidence = (() if temporal_bearing is None
                        else (float(temporal_bearing),))
            sources = (() if temporal_bearing is None
                       else ("temporal_flow",))
            # Conversely, a temporal rear return cannot release a concurrent
            # front/side near-body point.
            contradictory = any(
                abs(float(angle)) <= half_angle
                for angle in near_body_bearings
            )

        if not evidence or contradictory:
            return False, None, ()
        wrapped = tuple(
            float(math.atan2(math.sin(angle), math.cos(angle)))
            for angle in evidence
        )
        if any(abs(angle) <= half_angle for angle in wrapped):
            return False, None, sources
        nearest_rear_bearing = min(
            wrapped, key=lambda angle: abs(abs(angle) - math.pi)
        )
        return True, nearest_rear_bearing, sources

    @staticmethod
    def _finite_clearance(guard_result, key):
        value = guard_result.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _select_dynamic_escape_turn_sign(
        self, guard_result, context, obstacle_bearing
    ):
        """Choose one generalized, forward-only passage side.

        The probabilistic planner converts forecast-relative obstacle motion
        into a preferred robot heading.  Its sign is authoritative for
        crossings and oblique approaches.  If that signal is unavailable for
        a near head-on approach, choose the side with more measured clearance;
        only then fall back to turning away from the current bearing.
        """
        if bool(context.get("encounter_control_authoritative", False)):
            try:
                encounter_side = int(context.get(
                    "encounter_control_locked_steering_side", 0
                ) or 0)
            except (TypeError, ValueError):
                encounter_side = 0
            if encounter_side != 0:
                return (
                    1.0 if encounter_side > 0 else -1.0,
                    "encounter_mode_locked_side",
                )

        heading_error = context.get(
            "probabilistic_obstacle_preferred_escape_heading_error_rad"
        )
        try:
            heading_error = float(heading_error)
        except (TypeError, ValueError):
            heading_error = float("nan")
        lateral_speed = context.get(
            "probabilistic_obstacle_motion_lateral_body_mps", 0.0
        )
        lateral_fraction = context.get(
            "probabilistic_obstacle_motion_lateral_fraction", 0.0
        )
        try:
            lateral_speed = float(lateral_speed)
            lateral_fraction = float(lateral_fraction)
        except (TypeError, ValueError):
            lateral_speed = 0.0
            lateral_fraction = 0.0
        strong_lateral_crossing = bool(
            context.get(
                "probabilistic_obstacle_forward_lateral_countermotion_applied",
                False,
            )
            and math.isfinite(lateral_speed)
            and abs(lateral_speed)
            >= self.dynamic_escape_crossing_minimum_lateral_speed_mps
            and math.isfinite(lateral_fraction)
            and lateral_fraction >= 0.50
        )
        frontal_geometry = bool(
            math.isfinite(obstacle_bearing)
            and abs(obstacle_bearing) < math.radians(20.0)
            and not strong_lateral_crossing
        )
        if not frontal_geometry and strong_lateral_crossing:
            return (
                -float(np.sign(lateral_speed)),
                "measured_lateral_countermotion",
            )

        if (
            not frontal_geometry
            and math.isfinite(heading_error)
            and abs(heading_error) >= 0.08
        ):
            return float(np.sign(heading_error)), "predicted_relative_motion"

        if (
            frontal_geometry
            and self._dynamic_escape_frontal_encounter_latched
            and self._dynamic_escape_geometric_turn_sign != 0.0
        ):
            # Direct approaches do not have a meaningful left/right human
            # crossing direction.  Keep the already established clearance
            # side instead of allowing alternating leg fragments to swap it.
            return self._dynamic_escape_geometric_turn_sign, "previous_side"

        left_clearance = self._finite_clearance(
            guard_result, "min_left_side_range"
        )
        right_clearance = self._finite_clearance(
            guard_result, "min_right_side_range"
        )
        if left_clearance is not None and right_clearance is not None:
            clearance_delta = left_clearance - right_clearance
            if abs(clearance_delta) >= 0.05:
                return float(np.sign(clearance_delta)), "measured_side_clearance"

        if math.isfinite(heading_error) and abs(heading_error) >= 0.08:
            return float(np.sign(heading_error)), "predicted_relative_motion"

        if math.isfinite(obstacle_bearing) and abs(obstacle_bearing) >= 0.05:
            return (-1.0 if obstacle_bearing >= 0.0 else 1.0), "obstacle_bearing"
        if self._dynamic_escape_geometric_turn_sign != 0.0:
            return self._dynamic_escape_geometric_turn_sign, "previous_side"
        return -1.0, "deterministic_right_tie_break"

    def arbitrate(
        self,
        proposed: ControlCommand,
        guard_result: Mapping[str, object],
        planning_context: Mapping[str, object] = None,
    ):
        self._dynamic_deadline_decision_count += 1
        values = self.action_spec.clip(proposed.values)
        guard_result = dict(guard_result)
        context = dict(planning_context or {})
        encounter_control_authoritative = bool(
            context.get("encounter_control_authoritative", False)
        )
        encounter_rear_pass_inhibited = bool(
            encounter_control_authoritative
            and context.get("encounter_control_inhibit_rear_pass", False)
        )
        encounter_dynamic_escape_inhibited = bool(
            encounter_control_authoritative
            and context.get("encounter_control_inhibit_dynamic_escape", False)
        )
        encounter_hard_stop_escape_retained = bool(
            encounter_control_authoritative
            and context.get(
                "encounter_control_hard_stop_escape_motion_allowed", False
            )
        )
        encounter_stop_only_hard_safety = bool(
            encounter_control_authoritative
            and context.get(
                "encounter_control_stop_only_hard_safety", False
            )
        )
        reason = str(guard_result.get("reason", "front_clear"))
        proposed_v_for_direction = (
            float(values[self.action_spec.index("v_cmd")])
            if "v_cmd" in self.action_spec.names
            else 0.0
        )
        (
            rear_only_evidence,
            rear_pass_through_bearing,
            rear_pass_through_sources,
        ) = self._rear_pass_through_evidence(guard_result)
        rear_pass_through_raw_evidence = bool(rear_only_evidence)
        rear_pass_through_suppressed = bool(
            encounter_rear_pass_inhibited and rear_only_evidence
        )
        if encounter_rear_pass_inhibited:
            rear_only_evidence = False
            rear_pass_through_bearing = None
            rear_pass_through_sources = ()
        rear_pass_front_clearance = self._finite_clearance(
            guard_result, "min_front_range"
        )
        rear_pass_force_forward_ready = bool(
            self.rear_pass_through_force_forward_enabled
            and rear_only_evidence
            and rear_pass_front_clearance is not None
            and rear_pass_front_clearance
            >= self.rear_pass_through_min_front_clearance_m
            and "v_cmd" in self.action_spec.names
        )
        rear_pass_through_active = bool(
            rear_only_evidence
            and (
                proposed_v_for_direction
                > self.directional_motion_minimum_speed_mps
                or rear_pass_force_forward_ready
            )
        )
        rear_reverse_blocked = bool(
            rear_only_evidence
            and proposed_v_for_direction
            < -self.directional_motion_minimum_speed_mps
        )
        rear_pass_candidate_turn_sign = 0.0
        if rear_pass_through_bearing is not None:
            rear_pass_lateral = math.sin(float(rear_pass_through_bearing))
            if abs(rear_pass_lateral) >= 0.25:
                rear_pass_candidate_turn_sign = float(
                    np.sign(rear_pass_lateral)
                )
        rear_pass_direction_lock_started = False
        if rear_only_evidence:
            # A Mid-360 leg/flow centre can alternate between the two sides of
            # the rear axis while it observes the same person.  Reversing the
            # yaw sign every scan makes an otherwise continuous forward exit
            # visibly weave.  Lock the first causal rear-side sign until the
            # rear trigger has been absent for several complete cycles.
            self._rear_pass_through_direction_clear_streak = 0
            if (
                self._rear_pass_through_turn_sign == 0.0
                and rear_pass_candidate_turn_sign != 0.0
            ):
                self._rear_pass_through_turn_sign = (
                    rear_pass_candidate_turn_sign
                )
                rear_pass_direction_lock_started = True
        else:
            self._rear_pass_through_direction_clear_streak += 1
            if (
                self._rear_pass_through_direction_clear_streak
                >= self.rear_pass_through_direction_release_steps
            ):
                self._rear_pass_through_turn_sign = 0.0
        if rear_pass_through_active and reason in {
            "front_clear",
            "near_body_hard_stop",
            "temporal_collision_risk",
            "temporal_slowdown",
        }:
            guard_result["directional_guard_original_reason"] = reason
            guard_result["emergency_stop"] = False
            guard_result["should_slow_down"] = False
            guard_result["slow_scale"] = 1.0
            guard_result["reason"] = "rear_pass_through"
            reason = "rear_pass_through"
            if rear_pass_force_forward_ready:
                v_index = self.action_spec.index("v_cmd")
                values[v_index] = min(
                    self.action_spec.upper[v_index],
                    max(
                        float(values[v_index]),
                        self.rear_pass_through_min_forward_speed_mps,
                    ),
                )
                if "omega_cmd" in self.action_spec.names:
                    omega_index = self.action_spec.index("omega_cmd")
                    if self.rear_pass_through_force_straight_enabled:
                        # Once all causal evidence is behind the lateral plane,
                        # forward translation already increases separation.
                        # The older rear-corner steering rule imposed +/-0.6
                        # rad/s for tens of frames after a successful pass and
                        # dragged the robot away from its goal.  This physical
                        # profile removes that downstream steering authority.
                        omega_value = 0.0
                    else:
                        omega_limit = min(
                            self.rear_pass_through_max_omega_radps,
                            abs(float(self.action_spec.lower[omega_index])),
                            abs(float(self.action_spec.upper[omega_index])),
                        )
                        omega_value = float(np.clip(
                            values[omega_index], -omega_limit, omega_limit
                        ))
                        turn_sign = self._rear_pass_through_turn_sign
                        if turn_sign != 0.0:
                            # Turn toward the obstacle's initially observed
                            # lateral side so the *rear* of the chassis swings
                            # away. Historical profiles retain this behaviour.
                            omega_value = turn_sign * min(
                                omega_limit,
                                max(
                                    abs(omega_value),
                                    self.rear_pass_through_min_turn_omega_radps,
                                ),
                            )
                        else:
                            omega_value = 0.0
                    values[omega_index] = omega_value
        static_reverse_escape_held = False
        if self._static_reverse_escape_cooldown_remaining > 0:
            self._static_reverse_escape_cooldown_remaining -= 1
        if reason != "near_body_hard_stop":
            # Left the trap by any route: close the run and serve the cooldown
            # before another reverse may be granted.
            if self._static_reverse_escape_run_steps > 0:
                self._static_reverse_escape_cooldown_remaining = (
                    self.static_reverse_escape_cooldown_steps
                )
            self._static_reverse_escape_run_steps = 0
            self._static_reverse_escape_last_v = 0.0
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
            and not rear_only_evidence
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
            and not rear_only_evidence
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
        if encounter_dynamic_escape_inhibited:
            # The semantic manager owns the passage side and MPPI already
            # optimized against its temporary route.  Revoke all stale
            # ordinary escape/recovery transactions; the instantaneous hard
            # stop below remains untouched and may still veto translation.
            dynamic_escape_allowed = False
            planned_escape_allowed = False
            reactive_escape_allowed = False
            fresh_reactive_escape_allowed = False
            held_reactive_escape_allowed = False
            uncertainty_fusion_escape_allowed = False
            self._dynamic_escape_hold_remaining = 0
            self._dynamic_escape_hold_values = None
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            # A close-range turn/reverse transaction is also moving-command
            # authority.  Clear it while the semantic transaction is active so
            # it cannot resume from stale counters at rejoin handoff.
            self._dynamic_escape_hard_stop_turn_remaining = 0
            self._dynamic_escape_hard_stop_reverse_remaining = 0
            self._dynamic_escape_hard_stop_turn_sign = 0.0
            self._dynamic_escape_hard_stop_consumed = False
            self._dynamic_escape_hard_stop_rear_blocked_latched = False
            self._dynamic_escape_hard_stop_rear_clear_retry_used = False
            self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = 0
            self._dynamic_escape_hard_stop_completed_hold_count = 0
            self._dynamic_escape_hard_stop_side_rear_release_latched = False
            self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
            self._dynamic_escape_corridor_remaining = 0
            self._dynamic_escape_corridor_turn_remaining = 0
            self._dynamic_escape_corridor_turn_sign = 0.0
            self._dynamic_escape_geometric_commit_consumed = False
            self._dynamic_escape_geometric_clear_streak = 0
            self._dynamic_escape_geometric_turn_sign = 0.0
            self._dynamic_escape_geometric_direction_prediction_backed = False
            self._dynamic_escape_frontal_encounter_latched = False
            self._dynamic_escape_coast_remaining = 0
            self._dynamic_escape_vetted_reverse_steps = 0
            self._dynamic_escape_persistent_front_retry_count = 0
            # Ordinary forecast/scan escape no longer owns the command, but
            # the bounded close-range turn/reverse transaction is part of the
            # retained hard-safety boundary.  Keeping its state lets a front
            # near-body stop actually create separation when the measured rear
            # sector is clear; resetting it every encounter cycle would leave
            # the robot permanently stopped in front of a close pedestrian.
            if not encounter_hard_stop_escape_retained:
                self._dynamic_escape_hard_stop_turn_remaining = 0
                self._dynamic_escape_hard_stop_reverse_remaining = 0
                self._dynamic_escape_hard_stop_turn_sign = 0.0
                self._dynamic_escape_hard_stop_consumed = False
                self._dynamic_escape_hard_stop_rear_blocked_latched = False
                self._dynamic_escape_hard_stop_side_rear_release_latched = False
            self._dynamic_recovery_active = False
            self._dynamic_escape_seen = False
            self._dynamic_recovery_clear_steps = 0
            self._dynamic_recovery_release_count = 0
            self._dynamic_recovery_advance_steps = 0
            self._dynamic_recovery_alignment_creep_latched = False
            self._dynamic_recovery_progress_watch_active = False
            self._dynamic_recovery_progress_watch_remaining = 0
            self._dynamic_recovery_progress_watch_samples = []
            self._dynamic_deadline_conflict_seen = False
            self._dynamic_deadline_clear_steps = 0
        if rear_only_evidence:
            # Once all causal evidence lies in the rear cone, stale avoidance
            # transactions must not replay their earlier stop/saturated-turn
            # prefix. A new front/side observation re-enters the normal
            # fail-closed path on that same cycle.
            dynamic_escape_allowed = False
            planned_escape_allowed = False
            reactive_escape_allowed = False
            fresh_reactive_escape_allowed = False
            held_reactive_escape_allowed = False
            uncertainty_fusion_escape_allowed = False
            self._dynamic_escape_hold_remaining = 0
            self._dynamic_escape_hold_values = None
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            self._dynamic_escape_corridor_remaining = 0
            self._dynamic_escape_corridor_turn_remaining = 0
            self._dynamic_escape_corridor_turn_sign = 0.0
            # Crossing completion is stronger re-arm evidence than waiting
            # for several perfectly empty scans: the causal threat has passed
            # behind the chassis.  A later front approach is a new encounter.
            self._dynamic_escape_geometric_commit_consumed = False
            self._dynamic_escape_geometric_clear_streak = 0
            self._dynamic_escape_geometric_turn_sign = 0.0
            self._dynamic_escape_geometric_direction_prediction_backed = False
            self._dynamic_escape_frontal_encounter_latched = False
            self._dynamic_escape_coast_remaining = 0
            self._dynamic_escape_vetted_reverse_steps = 0
            self._dynamic_escape_persistent_front_retry_count = 0
        geometric_threat_evidence_active = bool(
            not rear_only_evidence
            and (
                context.get(
                    "probabilistic_obstacle_active_avoidance_enabled", False
                )
                or guard_result.get("dynamic_obstacle_near_body_match", False)
                or (
                    guard_result.get("temporal_scan_valid", False)
                    and np.isfinite(temporal_ttc_s)
                    and temporal_ttc_s <= self.dynamic_escape_trigger_ttc_s
                )
            )
        )
        if self.dynamic_escape_geometric_single_commit_enabled:
            if geometric_threat_evidence_active:
                self._dynamic_escape_geometric_clear_streak = 0
            else:
                self._dynamic_escape_geometric_clear_streak += 1
                if (
                    self._dynamic_escape_geometric_clear_streak
                    >= self.dynamic_escape_geometric_rearm_clear_steps
                ):
                    self._dynamic_escape_geometric_commit_consumed = False
                    self._dynamic_escape_geometric_turn_sign = 0.0
                    self._dynamic_escape_geometric_direction_prediction_backed = (
                        False
                    )
                    self._dynamic_escape_frontal_encounter_latched = False
                    self._dynamic_escape_coast_remaining = 0
                    self._dynamic_escape_vetted_reverse_steps = 0
                    self._dynamic_escape_persistent_front_retry_count = 0
        else:
            self._dynamic_escape_geometric_commit_consumed = False
            self._dynamic_escape_geometric_clear_streak = 0
            self._dynamic_escape_geometric_turn_sign = 0.0
            self._dynamic_escape_geometric_direction_prediction_backed = False
        planner_temporal_escape_active = bool(
            not encounter_dynamic_escape_inhibited
            and context.get(
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
                "physical_goal_bearing_error_rad",
                context.get(
                    "target_bearing_error",
                    context.get("terminal_bearing_error", float("nan")),
                ),
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
        geometric_turn_source = "inactive"
        prediction_direction_refresh_requested = bool(
            context.get(
                "probabilistic_obstacle_escape_direction_refreshed", False
            )
        )
        try:
            prediction_direction_refresh_confirmation_count = int(
                context.get(
                    "probabilistic_obstacle_escape_direction_reversal_confirmation_count",
                    0,
                )
                or 0
            )
        except (TypeError, ValueError):
            prediction_direction_refresh_confirmation_count = 0
        prediction_direction_refresh_confirmed = bool(
            not prediction_direction_refresh_requested
            or self.dynamic_escape_direction_refresh_confirmation_steps <= 1
            or prediction_direction_refresh_confirmation_count
            >= self.dynamic_escape_direction_refresh_confirmation_steps
        )
        lateral_motion_speed = context.get(
            "probabilistic_obstacle_motion_lateral_body_mps"
        )
        try:
            lateral_motion_speed = float(lateral_motion_speed)
        except (TypeError, ValueError):
            lateral_motion_speed = float("nan")
        refresh_minimum_speed = (
            self.dynamic_escape_direction_refresh_minimum_lateral_speed_mps
        )
        prediction_direction_refresh_lateral_evidence = bool(
            refresh_minimum_speed <= 0.0
            or (
                math.isfinite(lateral_motion_speed)
                and abs(lateral_motion_speed) >= refresh_minimum_speed
            )
            or (
                prediction_direction_refresh_requested
                and prediction_direction_refresh_confirmed
            )
        )
        preferred_heading_error = context.get(
            "probabilistic_obstacle_preferred_escape_heading_error_rad"
        )
        try:
            preferred_heading_error = float(preferred_heading_error)
        except (TypeError, ValueError):
            preferred_heading_error = float("nan")
        lateral_motion_fraction = context.get(
            "probabilistic_obstacle_motion_lateral_fraction", 0.0
        )
        try:
            lateral_motion_fraction = float(lateral_motion_fraction)
        except (TypeError, ValueError):
            lateral_motion_fraction = 0.0
        measured_lateral_countermotion_available = bool(
            context.get(
                "probabilistic_obstacle_forward_lateral_countermotion_applied",
                False,
            )
            and math.isfinite(lateral_motion_speed)
            and abs(lateral_motion_speed)
            >= self.dynamic_escape_crossing_minimum_lateral_speed_mps
            and math.isfinite(lateral_motion_fraction)
            and lateral_motion_fraction >= 0.50
        )
        preferred_turn_sign = (
            -float(np.sign(lateral_motion_speed))
            if measured_lateral_countermotion_available
            else float(np.sign(preferred_heading_error))
            if math.isfinite(preferred_heading_error)
            and abs(preferred_heading_error) >= 0.08
            else 0.0
        )
        strong_lateral_crossing = bool(
            measured_lateral_countermotion_available
            and prediction_direction_refresh_lateral_evidence
            and preferred_turn_sign != 0.0
        )
        late_prediction_forward_sector = bool(
            math.isfinite(obstacle_bearing)
            and abs(obstacle_bearing) <= math.radians(
                min(
                    90.0,
                    self.directional_forward_protected_half_angle_deg,
                )
            )
        )
        late_prediction_oblique_geometry = bool(
            math.isfinite(obstacle_bearing)
            # A frontal leg cluster can report a large lateral velocity for one
            # frame while the person is actually approaching head-on.  Do not
            # let that point-cloud fragmentation replace the clearance-selected
            # passage side.  Genuine crossings in the recorded physical runs
            # first appeared outside this 20 degree frontal cone.
            and abs(obstacle_bearing) >= math.radians(20.0)
        )
        measured_reversal_candidate = bool(
            measured_lateral_countermotion_available
            and not self._dynamic_escape_frontal_encounter_latched
            and late_prediction_forward_sector
            and late_prediction_oblique_geometry
            and preferred_turn_sign != 0.0
            and self._dynamic_escape_geometric_turn_sign != 0.0
            and preferred_turn_sign
            != self._dynamic_escape_geometric_turn_sign
        )
        if measured_reversal_candidate:
            if (
                preferred_turn_sign
                == self._dynamic_escape_measured_reversal_candidate_sign
            ):
                self._dynamic_escape_measured_reversal_confirmation_count += 1
            else:
                self._dynamic_escape_measured_reversal_candidate_sign = (
                    preferred_turn_sign
                )
                self._dynamic_escape_measured_reversal_confirmation_count = 1
        else:
            self._dynamic_escape_measured_reversal_candidate_sign = 0.0
            self._dynamic_escape_measured_reversal_confirmation_count = 0
        measured_reversal_confirmation_count = int(
            self._dynamic_escape_measured_reversal_confirmation_count
        )
        measured_reversal_confirmed = bool(
            measured_reversal_candidate
            and measured_reversal_confirmation_count
            >= self.dynamic_escape_measured_reversal_confirmation_steps
        )
        late_prediction_available = bool(
            context.get(
                "probabilistic_obstacle_forward_lateral_countermotion_applied",
                False,
            )
            and not self._dynamic_escape_frontal_encounter_latched
            and prediction_direction_refresh_lateral_evidence
            and preferred_turn_sign != 0.0
            and self._dynamic_escape_geometric_turn_sign != 0.0
            and not self._dynamic_escape_geometric_direction_prediction_backed
            and late_prediction_forward_sector
            and late_prediction_oblique_geometry
        )
        late_prediction_direction_refresh = bool(
            late_prediction_available
            and preferred_turn_sign
            != self._dynamic_escape_geometric_turn_sign
        )
        prediction_direction_refresh_geometry_allowed = bool(
            not self._dynamic_escape_frontal_encounter_latched
            and (
                self._dynamic_escape_geometric_direction_prediction_backed
                or (
                late_prediction_oblique_geometry
                    and strong_lateral_crossing
                )
            )
        )
        prediction_direction_refreshed = bool(
            (
                prediction_direction_refresh_requested
                and prediction_direction_refresh_confirmed
                and prediction_direction_refresh_lateral_evidence
                and prediction_direction_refresh_geometry_allowed
            )
            or late_prediction_direction_refresh
            or measured_reversal_confirmed
        )
        if (
            late_prediction_available
            and not late_prediction_direction_refresh
        ):
            # The first strong crossing prediction agrees with the geometric
            # fallback, so the selected side is now prediction-backed.  Later
            # noisy leg-cluster reversals must use the planner's ordinary
            # confirmation path instead of repeatedly exercising this one-time
            # acquisition rule.
            self._dynamic_escape_geometric_direction_prediction_backed = True
        if prediction_direction_refreshed:
            # A CA-IMM/causal-regression direction change starts a new finite
            # turn transaction immediately.  This is not a timer re-trigger:
            # it is new motion evidence that invalidates the old passage side.
            self._dynamic_escape_geometric_commit_consumed = False
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            self._dynamic_escape_coast_remaining = 0
            self._dynamic_escape_measured_reversal_candidate_sign = 0.0
            self._dynamic_escape_measured_reversal_confirmation_count = 0
        dynamic_hard_stop_prediction_evidence = bool(
            context.get(
                "probabilistic_obstacle_active_avoidance_enabled", False
            )
        )
        dynamic_hard_stop_geometric_event = bool(
            self.dynamic_escape_hard_stop_enabled
            and not encounter_stop_only_hard_safety
            and (
                not encounter_dynamic_escape_inhibited
                or encounter_hard_stop_escape_retained
            )
            and reason in {"near_body_hard_stop", "hard_stop"}
            and guard_result.get("emergency_stop", False)
            and not rear_only_evidence
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        )
        dynamic_hard_stop_event = bool(
            dynamic_hard_stop_geometric_event
            and (
                guard_result.get("dynamic_obstacle_near_body_match", False)
                or guard_result.get(
                    "dynamic_obstacle_scan_flow_match", False
                )
                or dynamic_hard_stop_prediction_evidence
            )
            and np.isfinite(obstacle_bearing)
            and abs(obstacle_bearing) <= math.radians(
                self.directional_forward_protected_half_angle_deg
            )
        )
        hard_stop_transaction_pending = bool(
            self.dynamic_escape_hard_stop_enabled
            and not self._dynamic_escape_hard_stop_consumed
            and (
                self._dynamic_escape_hard_stop_turn_remaining > 0
                or self._dynamic_escape_hard_stop_reverse_remaining > 0
            )
        )
        if dynamic_hard_stop_geometric_event:
            self._dynamic_escape_hard_stop_clear_streak = 0
        elif hard_stop_transaction_pending:
            # The Mid-360 non-repetitive scan can miss the same close person
            # for one frame.  That is not clearance evidence and must not
            # pause or cancel a transaction that has already entered the
            # 0.50 m hard-stop envelope.
            self._dynamic_escape_hard_stop_clear_streak = 0
        else:
            self._dynamic_escape_hard_stop_clear_streak += 1
            if (
                self._dynamic_escape_hard_stop_clear_streak
                >= self.dynamic_escape_geometric_rearm_clear_steps
            ):
                self._dynamic_escape_hard_stop_turn_remaining = 0
                self._dynamic_escape_hard_stop_reverse_remaining = 0
                self._dynamic_escape_hard_stop_turn_sign = 0.0
                self._dynamic_escape_hard_stop_consumed = False
                self._dynamic_escape_hard_stop_rear_blocked_latched = False
                self._dynamic_escape_hard_stop_rear_clear_retry_used = False
                self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = 0
                self._dynamic_escape_hard_stop_completed_hold_count = 0
                self._dynamic_escape_hard_stop_side_rear_release_latched = False
                self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
        hard_stop_direction_refresh_applied = bool(
            prediction_direction_refreshed
            and dynamic_hard_stop_event
            and self.dynamic_escape_hard_stop_direction_refresh_enabled
        )
        if hard_stop_direction_refresh_applied:
            # A genuine forecast reversal invalidates the close-range side in
            # the same cycle, just as it invalidates the forward arc above.
            self._dynamic_escape_hard_stop_turn_remaining = 0
            self._dynamic_escape_hard_stop_reverse_remaining = 0
            self._dynamic_escape_hard_stop_consumed = False
            self._dynamic_escape_hard_stop_rear_blocked_latched = False
            self._dynamic_escape_hard_stop_rear_clear_retry_used = False
            self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = 0
            self._dynamic_escape_hard_stop_completed_hold_count = 0
            self._dynamic_escape_hard_stop_side_rear_release_latched = False
            self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
        hard_stop_front_clearance = self._finite_clearance(
            guard_result, "min_front_range"
        )
        hard_stop_side_rear_base_available = bool(
            self.dynamic_escape_hard_stop_side_rear_release_enabled
            and self._dynamic_escape_hard_stop_consumed
            and dynamic_hard_stop_geometric_event
            and (
                guard_result.get("dynamic_obstacle_near_body_match", False)
                or guard_result.get("dynamic_obstacle_scan_flow_match", False)
                or dynamic_hard_stop_prediction_evidence
            )
            and hard_stop_front_clearance is not None
            and hard_stop_front_clearance
            >= self.dynamic_escape_hard_stop_side_rear_release_min_front_range_m
        )
        hard_stop_side_rear_acquire = bool(
            hard_stop_side_rear_base_available
            and np.isfinite(obstacle_bearing)
            and abs(obstacle_bearing)
            >= self.dynamic_escape_hard_stop_side_rear_release_min_bearing_rad
        )
        hard_stop_side_rear_hold_bearing = (
            abs(obstacle_bearing)
            if np.isfinite(obstacle_bearing)
            else 0.0
        )
        try:
            hard_stop_temporal_bearing = float(
                guard_result.get(
                    "temporal_scan_center_angle_rad", float("nan")
                )
            )
            hard_stop_temporal_ttc = float(
                guard_result.get("temporal_scan_ttc_s", float("inf"))
            )
        except (TypeError, ValueError):
            hard_stop_temporal_bearing = float("nan")
            hard_stop_temporal_ttc = float("inf")
        if (
            guard_result.get("temporal_scan_valid", False)
            and np.isfinite(hard_stop_temporal_bearing)
            and hard_stop_temporal_ttc <= self.dynamic_escape_trigger_ttc_s
        ):
            hard_stop_side_rear_hold_bearing = max(
                hard_stop_side_rear_hold_bearing,
                abs(hard_stop_temporal_bearing),
            )
        if not hard_stop_side_rear_base_available:
            self._dynamic_escape_hard_stop_side_rear_release_latched = False
            self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
        elif hard_stop_side_rear_acquire:
            self._dynamic_escape_hard_stop_side_rear_release_latched = True
            self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
        elif self._dynamic_escape_hard_stop_side_rear_release_latched:
            if (
                hard_stop_side_rear_hold_bearing
                >= self.dynamic_escape_hard_stop_side_rear_release_hold_min_bearing_rad
            ):
                self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
            else:
                self._dynamic_escape_hard_stop_side_rear_release_abort_count += 1
                if (
                    self._dynamic_escape_hard_stop_side_rear_release_abort_count
                    >= self.dynamic_escape_hard_stop_side_rear_release_abort_steps
                ):
                    self._dynamic_escape_hard_stop_side_rear_release_latched = False
        hard_stop_side_rear_geometry_available = bool(
            hard_stop_side_rear_base_available
            and self._dynamic_escape_hard_stop_side_rear_release_latched
        )
        hard_stop_rear_clear_retry_started = False
        if (
            self.dynamic_escape_hard_stop_rear_clear_retry_enabled
            and dynamic_hard_stop_geometric_event
            and self._dynamic_escape_hard_stop_consumed
            and not hard_stop_side_rear_geometry_available
            and self._dynamic_escape_hard_stop_rear_blocked_latched
            and not self._dynamic_escape_hard_stop_rear_clear_retry_used
            and self._rear_sector_clear(
                guard_result,
                self.dynamic_escape_hard_stop_rear_sector_deg,
                self.dynamic_escape_hard_stop_min_rear_range,
            )
        ):
            self._dynamic_escape_hard_stop_reverse_remaining = (
                self.dynamic_escape_hard_stop_rear_clear_retry_steps
            )
            self._dynamic_escape_hard_stop_consumed = False
            self._dynamic_escape_hard_stop_rear_blocked_latched = False
            self._dynamic_escape_hard_stop_rear_clear_retry_used = True
            self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = (
                self.dynamic_escape_hard_stop_rear_blocked_wait_steps
            )
            self._dynamic_escape_hard_stop_completed_hold_count = 0
            hard_stop_rear_clear_retry_started = True
        hard_stop_transaction_active = bool(
            self.dynamic_escape_hard_stop_enabled
            and not self._dynamic_escape_hard_stop_consumed
            and (
                self._dynamic_escape_hard_stop_turn_remaining > 0
                or self._dynamic_escape_hard_stop_reverse_remaining > 0
            )
        )
        hard_stop_transaction_completed_candidate = bool(
            self.dynamic_escape_hard_stop_enabled
            and self._dynamic_escape_hard_stop_consumed
            and self._dynamic_escape_hard_stop_turn_sign != 0.0
            and dynamic_hard_stop_geometric_event
        )
        hard_stop_side_rear_release_available = bool(
            hard_stop_transaction_completed_candidate
            and hard_stop_side_rear_geometry_available
        )
        hard_stop_transaction_completed_hold = bool(
            hard_stop_transaction_completed_candidate
            and not hard_stop_side_rear_release_available
            and self._dynamic_escape_hard_stop_completed_hold_count
            < self.dynamic_escape_hard_stop_completed_hold_steps
        )
        hard_stop_transaction_completed_fail_closed = bool(
            hard_stop_transaction_completed_candidate
            and not hard_stop_side_rear_release_available
            and not hard_stop_transaction_completed_hold
        )
        front_geometric_escape_available = bool(
            reactive_escape_allowed
            and np.isfinite(obstacle_bearing)
            and abs(obstacle_bearing) <= 0.5 * np.pi
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        )
        goal_diverged_from_escape_side = bool(
            np.isfinite(recovery_heading_error)
            and np.isfinite(self.dynamic_escape_goal_divergence_release_rad)
            and self._dynamic_escape_geometric_turn_sign != 0.0
            and recovery_heading_error
            * self._dynamic_escape_geometric_turn_sign
            < 0.0
            and abs(recovery_heading_error)
            >= self.dynamic_escape_goal_divergence_release_rad
        )
        passage_completion_front_clearance = self._finite_clearance(
            guard_result, "min_front_range"
        )
        frontal_passage_completion_ready = bool(
            self.dynamic_escape_passage_completion_enabled
            and self._dynamic_escape_frontal_encounter_latched
            and self._dynamic_escape_geometric_turn_sign != 0.0
            and np.isfinite(obstacle_bearing)
            and abs(obstacle_bearing)
            >= self.dynamic_escape_passage_completion_min_bearing_rad
            and obstacle_bearing
            * self._dynamic_escape_geometric_turn_sign < 0.0
            and passage_completion_front_clearance is not None
            and passage_completion_front_clearance
            >= self.dynamic_escape_passage_completion_min_front_clearance_m
            and np.isfinite(recovery_heading_error)
            and recovery_heading_error
            * self._dynamic_escape_geometric_turn_sign < 0.0
            and abs(recovery_heading_error)
            >= self.dynamic_escape_passage_completion_min_goal_counter_heading_rad
        )
        geometric_goal_release_applied = bool(
            goal_diverged_from_escape_side
            and (
                self._dynamic_escape_direction_commit_remaining > 0
                or self._dynamic_escape_coast_remaining != 0
            )
        )
        geometric_passage_completion_applied = bool(
            frontal_passage_completion_ready
            and (
                self._dynamic_escape_direction_commit_remaining > 0
                or self._dynamic_escape_coast_remaining != 0
            )
        )
        if (
            geometric_goal_release_applied
            or geometric_passage_completion_applied
        ):
            # Recorded run 003752 kept a saturated right turn alive after the
            # goal had moved 75--106 degrees to the left of the chassis.  End
            # both phases atomically.  A frontal pass also completes as soon
            # as the person is outside the forward corridor, the front opens,
            # and the goal lies counter to the avoidance turn.  In either
            # case a stale coast must not continue the same yaw transaction.
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            self._dynamic_escape_coast_remaining = 0
        persistent_front_retry_candidate = bool(
            self.dynamic_escape_persistent_front_retry_enabled
            and self.dynamic_escape_vetted_reverse_retry_steps > 0
            and self.dynamic_escape_persistent_front_max_retries > 0
            and front_geometric_escape_available
            and self._dynamic_escape_geometric_commit_consumed
            and self._dynamic_escape_coast_remaining == 0
            and self._dynamic_escape_vetted_reverse_steps
            >= self.dynamic_escape_vetted_reverse_retry_steps
            and self._dynamic_escape_persistent_front_retry_count
            < self.dynamic_escape_persistent_front_max_retries
        )
        persistent_front_retry_goal_rejected = bool(
            persistent_front_retry_candidate
            and goal_diverged_from_escape_side
        )
        persistent_front_retry_rearmed = bool(
            persistent_front_retry_candidate
            and not persistent_front_retry_goal_rejected
        )
        if persistent_front_retry_rearmed:
            # The first finite arc did not move the live obstacle out of the
            # front half-plane and MPPI has spent its small reverse allowance.
            # Retry the established passage side once instead of accepting an
            # unbounded sequence of risk-vetted reverse samples.
            self._dynamic_escape_geometric_commit_consumed = False
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            self._dynamic_escape_vetted_reverse_steps = 0
            self._dynamic_escape_persistent_front_retry_count += 1
        elif persistent_front_retry_goal_rejected:
            # Replaying the old side when the chassis already faces far away
            # from the goal only deepens the failure.  Consume the one retry so
            # the final bounded reverse/hold policy takes authority instead.
            self._dynamic_escape_persistent_front_retry_count = (
                self.dynamic_escape_persistent_front_max_retries
            )
        post_retry_reverse_exhausted = bool(
            self.dynamic_escape_post_retry_reverse_hold_enabled
            and self.dynamic_escape_vetted_reverse_retry_steps > 0
            and front_geometric_escape_available
            and self._dynamic_escape_geometric_commit_consumed
            and self._dynamic_escape_coast_remaining == 0
            and self._dynamic_escape_persistent_front_retry_count
            >= self.dynamic_escape_persistent_front_max_retries
            and self._dynamic_escape_vetted_reverse_steps
            >= self.dynamic_escape_vetted_reverse_retry_steps
        )
        try:
            dynamic_surface_range = float(
                guard_result.get(
                    "dynamic_obstacle_surface_range_m", float("nan")
                )
            )
        except (TypeError, ValueError):
            dynamic_surface_range = float("nan")
        post_retry_side_forward_available = bool(
            post_retry_reverse_exhausted
            and not guard_result.get("emergency_stop", False)
            and self.dynamic_escape_post_retry_side_forward_speed > 0.0
            and np.isfinite(obstacle_bearing)
            and abs(obstacle_bearing)
            >= self.dynamic_escape_post_retry_side_forward_min_bearing_rad
            and np.isfinite(dynamic_surface_range)
            and dynamic_surface_range
            >= self.dynamic_escape_post_retry_side_forward_min_surface_range_m
        )
        geometric_forward_escape = bool(
            self.dynamic_escape_uncertainty_fusion_enabled
            and front_geometric_escape_available
            and (
                not self.dynamic_escape_geometric_single_commit_enabled
                or not self._dynamic_escape_geometric_commit_consumed
            )
        )
        corridor_entry_allowed = bool(
            self.dynamic_escape_corridor_enabled
            and front_geometric_escape_available
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
            corridor_commit_active or corridor_entry_allowed
        )
        corridor_turning = False
        committed_geometric_escape = bool(
            self.dynamic_escape_uncertainty_fusion_enabled
            and self._dynamic_escape_direction_commit_remaining > 0
            and self._dynamic_escape_direction_commit_values is not None
            and (
                not guard_result.get("emergency_stop", False)
                or (
                    self._dynamic_escape_geometric_turn_sign != 0.0
                    and not dynamic_hard_stop_geometric_event
                )
            )
            and not rear_only_evidence
        )
        geometric_forward_coast = bool(
            self.dynamic_escape_uncertainty_fusion_enabled
            and self.dynamic_escape_geometric_single_commit_enabled
            and self._dynamic_escape_geometric_commit_consumed
            and self._dynamic_escape_coast_remaining != 0
            and (
                front_geometric_escape_available
                or (
                    geometric_threat_evidence_active
                    and not rear_only_evidence
                    and np.isfinite(obstacle_bearing)
                    and abs(obstacle_bearing) <= 0.5 * np.pi
                    and "v_cmd" in self.action_spec.names
                    and "omega_cmd" in self.action_spec.names
                )
            )
            and (
                not guard_result.get("emergency_stop", False)
                or (
                    self._dynamic_escape_geometric_turn_sign != 0.0
                    and not dynamic_hard_stop_geometric_event
                )
            )
        )
        geometric_temporal_escape_override = bool(
            guard_result.get("emergency_stop", False)
            and not dynamic_hard_stop_geometric_event
            and self._dynamic_escape_geometric_turn_sign != 0.0
            and (committed_geometric_escape or geometric_forward_coast)
        )
        prediction_backed_temporal_escape_override = bool(
            geometric_temporal_escape_override
            and self._dynamic_escape_geometric_direction_prediction_backed
        )
        reactive_reverse_required = bool(
            reactive_escape_allowed
            and self.dynamic_escape_reverse_speed > 0.0
            and abs(away_heading_error) > 0.5 * np.pi
            and not geometric_forward_escape
        )
        try:
            temporal_preturn_bearing = float(
                guard_result.get(
                    "temporal_scan_center_angle_rad", float("nan")
                )
            )
            temporal_preturn_ttc = float(
                guard_result.get("temporal_scan_ttc_s", float("inf"))
            )
        except (TypeError, ValueError):
            temporal_preturn_bearing = float("nan")
            temporal_preturn_ttc = float("inf")
        temporal_preturn_active = bool(
            self.dynamic_escape_temporal_preturn_enabled
            and not encounter_dynamic_escape_inhibited
            and reason == "temporal_slowdown"
            and not guard_result.get("emergency_stop", False)
            and guard_result.get("temporal_scan_valid", False)
            and np.isfinite(temporal_preturn_bearing)
            and abs(temporal_preturn_bearing)
            <= self.dynamic_escape_temporal_preturn_max_bearing_rad
            and np.isfinite(temporal_preturn_ttc)
            and temporal_preturn_ttc <= self.dynamic_escape_trigger_ttc_s
            and not rear_only_evidence
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        )
        temporal_preturn_applied = False
        vetted_forward_reverse_veto = False
        reverse_escape = False
        hard_stop_escape_phase = "inactive"
        hard_stop_reverse_authorized = False
        hard_stop_rear_clear = False
        geometric_coast_direction_locked = False
        frontal_entry_speed_applied = False
        reverse_goal_realign_applied = False
        post_retry_reverse_hold_applied = False
        post_retry_side_forward_applied = False
        hard_stop_reverse_turn_scale = 1.0
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
        if (
            (dynamic_hard_stop_event
             and not self._dynamic_escape_hard_stop_consumed)
            or hard_stop_transaction_active
            or hard_stop_transaction_completed_hold
        ):
            # A near-body transaction supersedes the earlier forward pass.
            # Never resume its stale coast after turning/reversing in place.
            self._dynamic_escape_coast_remaining = 0
            self._dynamic_escape_vetted_reverse_steps = 0
            v_index = self.action_spec.index("v_cmd")
            omega_index = self.action_spec.index("omega_cmd")
            if (
                not self._dynamic_escape_hard_stop_consumed
                and self._dynamic_escape_hard_stop_turn_remaining <= 0
                and self._dynamic_escape_hard_stop_reverse_remaining <= 0
            ):
                if (
                    self._dynamic_escape_frontal_encounter_latched
                    and self._dynamic_escape_geometric_turn_sign != 0.0
                ):
                    turn_sign = self._dynamic_escape_geometric_turn_sign
                    geometric_turn_source = "frontal_encounter_side"
                else:
                    turn_sign, geometric_turn_source = (
                        self._select_dynamic_escape_turn_sign(
                            guard_result, context, obstacle_bearing
                        )
                    )
                self._dynamic_escape_hard_stop_turn_sign = turn_sign
                self._dynamic_escape_geometric_turn_sign = turn_sign
                self._dynamic_escape_frontal_encounter_latched = bool(
                    self._dynamic_escape_frontal_encounter_latched
                    or (
                        math.isfinite(obstacle_bearing)
                        and abs(obstacle_bearing) < math.radians(20.0)
                        and not strong_lateral_crossing
                    )
                )
                self._dynamic_escape_geometric_direction_prediction_backed = (
                    geometric_turn_source in {
                        "predicted_relative_motion",
                        "measured_lateral_countermotion",
                    }
                )
                self._dynamic_escape_hard_stop_turn_remaining = (
                    self.dynamic_escape_hard_stop_turn_steps
                )
                self._dynamic_escape_hard_stop_reverse_remaining = (
                    self.dynamic_escape_hard_stop_reverse_steps
                )
                self._dynamic_escape_hard_stop_rear_blocked_latched = False
                self._dynamic_escape_hard_stop_rear_clear_retry_used = False
                self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = (
                    self.dynamic_escape_hard_stop_rear_blocked_wait_steps
                )
                self._dynamic_escape_hard_stop_completed_hold_count = 0
                self._dynamic_escape_hard_stop_side_rear_release_latched = False
                self._dynamic_escape_hard_stop_side_rear_release_abort_count = 0
            values[v_index] = 0.0
            if self._dynamic_escape_hard_stop_turn_remaining > 0:
                values[omega_index] = (
                    self.action_spec.upper[omega_index]
                    if self._dynamic_escape_hard_stop_turn_sign > 0.0
                    else self.action_spec.lower[omega_index]
                )
                self._dynamic_escape_hard_stop_turn_remaining -= 1
                hard_stop_escape_phase = "turn_in_place"
            elif self._dynamic_escape_hard_stop_reverse_remaining > 0:
                hard_stop_rear_clear = self._rear_sector_clear(
                    guard_result,
                    self.dynamic_escape_hard_stop_rear_sector_deg,
                    self.dynamic_escape_hard_stop_min_rear_range,
                )
                if hard_stop_rear_clear:
                    values[v_index] = max(
                        -self.dynamic_escape_hard_stop_reverse_speed,
                        self.action_spec.lower[v_index],
                    )
                    self._dynamic_escape_hard_stop_rear_blocked_latched = False
                    hard_stop_reverse_authorized = True
                    hard_stop_escape_phase = (
                        "rear_clear_retry_reverse"
                        if hard_stop_rear_clear_retry_started
                        else "rear_clear_reverse"
                    )
                else:
                    self._dynamic_escape_hard_stop_rear_blocked_latched = True
                    self._dynamic_escape_hard_stop_rear_blocked_wait_remaining -= 1
                    if (
                        self._dynamic_escape_hard_stop_rear_blocked_wait_remaining
                        <= 0
                    ):
                        self._dynamic_escape_hard_stop_reverse_remaining = 0
                        self._dynamic_escape_hard_stop_consumed = True
                        hard_stop_escape_phase = "rear_blocked_wait_exhausted"
                    else:
                        hard_stop_escape_phase = "rear_blocked_turn_only"
                bounded_omega = min(
                    self.dynamic_escape_hard_stop_reverse_max_omega_radps,
                    abs(float(self.action_spec.upper[omega_index])),
                    abs(float(self.action_spec.lower[omega_index])),
                )
                if self.dynamic_escape_hard_stop_reverse_turn_decay_enabled:
                    hard_stop_reverse_turn_scale = min(
                        1.0,
                        max(
                            0.0,
                            float(
                                self._dynamic_escape_hard_stop_reverse_remaining
                            )
                            / float(self.dynamic_escape_hard_stop_reverse_steps),
                        ),
                    )
                    if (
                        np.isfinite(recovery_heading_error)
                        and self._dynamic_escape_hard_stop_turn_sign != 0.0
                        and recovery_heading_error
                        * self._dynamic_escape_hard_stop_turn_sign
                        < 0.0
                        and np.isfinite(
                            self.dynamic_escape_goal_divergence_release_rad
                        )
                    ):
                        goal_turn_scale = np.clip(
                            1.0
                            - abs(recovery_heading_error)
                            / self.dynamic_escape_goal_divergence_release_rad,
                            0.0,
                            1.0,
                        )
                        hard_stop_reverse_turn_scale = min(
                            hard_stop_reverse_turn_scale,
                            float(goal_turn_scale),
                        )
                values[omega_index] = (
                    bounded_omega
                    * self._dynamic_escape_hard_stop_turn_sign
                    * hard_stop_reverse_turn_scale
                )
                if hard_stop_rear_clear:
                    self._dynamic_escape_hard_stop_reverse_remaining -= 1
                    if self._dynamic_escape_hard_stop_reverse_remaining <= 0:
                        self._dynamic_escape_hard_stop_consumed = True
            else:
                values[omega_index] = 0.0
                hard_stop_escape_phase = "bounded_transaction_complete"
                self._dynamic_escape_hard_stop_consumed = True
                self._dynamic_escape_hard_stop_completed_hold_count += 1
            values = self.action_spec.clip(values)
            reverse_escape = bool(values[v_index] < 0.0)
            reason = "dynamic_hard_stop_escape"
        elif hard_stop_side_rear_release_available:
            # The transaction has put the live obstacle behind the lateral
            # plane while the measured front corridor is open.  A short,
            # straight forward command increases separation; continuing to
            # hold (0, 0) here reproduced the 42-cycle physical deadlock.
            v_index = self.action_spec.index("v_cmd")
            omega_index = self.action_spec.index("omega_cmd")
            values[v_index] = min(
                self.dynamic_escape_hard_stop_side_rear_release_speed,
                self.action_spec.upper[v_index],
            )
            values[omega_index] = 0.0
            values = self.action_spec.clip(values)
            guard_result["emergency_stop"] = False
            guard_result["should_slow_down"] = False
            guard_result["slow_scale"] = 1.0
            self._dynamic_escape_hard_stop_reverse_remaining = 0
            self._dynamic_escape_hard_stop_rear_blocked_wait_remaining = 0
            self._dynamic_escape_hard_stop_rear_clear_retry_used = True
            reverse_escape = False
            hard_stop_escape_phase = "side_rear_forward_release"
            reason = "dynamic_hard_stop_side_rear_release"
        elif hard_stop_transaction_completed_fail_closed:
            # No direction of translation is positively certified.  End the
            # completed transaction's ownership after its finite diagnostic
            # hold, but retain the underlying instantaneous hard stop.
            v_index = self.action_spec.index("v_cmd")
            omega_index = self.action_spec.index("omega_cmd")
            values[v_index] = 0.0
            values[omega_index] = 0.0
            values = self.action_spec.clip(values)
            hard_stop_escape_phase = "completed_fail_closed"
        elif (
            dynamic_escape_allowed
            or corridor_commit_active
            or committed_geometric_escape
            or geometric_forward_coast
        ):
            if self.dynamic_escape_corridor_enabled and (
                corridor_entry_allowed or corridor_commit_active
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
                if (
                    persistent_front_retry_rearmed
                    and self._dynamic_escape_geometric_turn_sign != 0.0
                ):
                    turn_sign = self._dynamic_escape_geometric_turn_sign
                    geometric_turn_source = "persistent_front_previous_side"
                else:
                    turn_sign, geometric_turn_source = (
                        self._select_dynamic_escape_turn_sign(
                            guard_result, context, obstacle_bearing
                        )
                    )
                self._dynamic_escape_geometric_turn_sign = turn_sign
                self._dynamic_escape_vetted_reverse_steps = 0
                self._dynamic_escape_frontal_encounter_latched = bool(
                    self._dynamic_escape_frontal_encounter_latched
                    or (
                        math.isfinite(obstacle_bearing)
                        and abs(obstacle_bearing) < math.radians(20.0)
                        and not strong_lateral_crossing
                    )
                )
                self._dynamic_escape_geometric_direction_prediction_backed = (
                    geometric_turn_source in {
                        "predicted_relative_motion",
                        "measured_lateral_countermotion",
                    }
                )
                if self._dynamic_escape_frontal_encounter_latched:
                    input_v = float(values[v_index])
                    values[v_index] = min(
                        values[v_index],
                        self.dynamic_escape_frontal_entry_speed,
                    )
                    frontal_entry_speed_applied = bool(
                        values[v_index] < input_v - 1.0e-12
                    )
                values[omega_index] = (
                    self.action_spec.upper[omega_index]
                    if turn_sign > 0.0
                    else self.action_spec.lower[omega_index]
                )
                values = self.action_spec.clip(values)
                reverse_escape = False
                commit_steps = (
                    self.dynamic_escape_frontal_commit_steps
                    if self._dynamic_escape_frontal_encounter_latched
                    else self.dynamic_escape_direction_commit_steps
                )
                if persistent_front_retry_rearmed:
                    commit_steps = min(
                        commit_steps,
                        self.dynamic_escape_persistent_front_retry_commit_steps,
                    )
                if commit_steps > 0:
                    self._dynamic_escape_direction_commit_values = (
                        values.copy()
                    )
                    self._dynamic_escape_direction_commit_remaining = (
                        commit_steps - 1
                    )
                if self.dynamic_escape_geometric_single_commit_enabled:
                    self._dynamic_escape_geometric_commit_consumed = True
                    self._dynamic_escape_geometric_clear_streak = 0
                    self._dynamic_escape_coast_remaining = (
                        self.dynamic_escape_coast_steps
                    )
                reason = "dynamic_active_escape"
            elif geometric_forward_coast:
                # The finite saturated turn has established a passage side.
                # Continue translating on that tangent without replaying the
                # turn or accepting a reverse candidate.  This supplies the
                # visible "go around, then straighten" behaviour and prevents
                # an unlimited arc when the forecast remains active.
                v_index = self.action_spec.index("v_cmd")
                omega_index = self.action_spec.index("omega_cmd")
                values[v_index] = min(
                    self.dynamic_escape_max_speed,
                    self.action_spec.upper[v_index],
                )
                preferred_heading_error = context.get(
                    "probabilistic_obstacle_preferred_escape_heading_error_rad"
                )
                try:
                    preferred_heading_error = float(preferred_heading_error)
                except (TypeError, ValueError):
                    preferred_heading_error = float("nan")
                if (
                    math.isfinite(preferred_heading_error)
                    and abs(preferred_heading_error) >= 0.08
                    and self.dynamic_escape_coast_max_omega_radps > 0.0
                ):
                    if (
                        self.dynamic_escape_coast_direction_lock_enabled
                        and self._dynamic_escape_geometric_turn_sign != 0.0
                    ):
                        preferred_heading_error = math.copysign(
                            abs(preferred_heading_error),
                            self._dynamic_escape_geometric_turn_sign,
                        )
                        geometric_coast_direction_locked = True
                    values[omega_index] = np.clip(
                        self.dynamic_escape_coast_turn_gain
                        * preferred_heading_error,
                        -self.dynamic_escape_coast_max_omega_radps,
                        self.dynamic_escape_coast_max_omega_radps,
                    )
                    geometric_turn_source = "predicted_relative_motion_coast"
                elif (
                    self._dynamic_escape_frontal_encounter_latched
                    and self._dynamic_escape_geometric_turn_sign != 0.0
                    and self.dynamic_escape_coast_max_omega_radps > 0.0
                ):
                    values[omega_index] = (
                        self._dynamic_escape_geometric_turn_sign
                        * self.dynamic_escape_coast_max_omega_radps
                    )
                    geometric_turn_source = "frontal_encounter_coast"
                else:
                    values[omega_index] = 0.0
                values = self.action_spec.clip(values)
                if self._dynamic_escape_coast_remaining > 0:
                    self._dynamic_escape_coast_remaining -= 1
                reverse_escape = False
                reason = "dynamic_active_escape"
            else:
                use_vetted_planner_control = (
                    self.dynamic_escape_use_vetted_planner_control
                    or hard_fallback_planner_control
                    or (
                        self.dynamic_escape_use_vetted_emergency_candidate
                        and context.get(
                            "probabilistic_obstacle_emergency_candidate_selected",
                            False,
                        )
                        and context.get(
                            "probabilistic_obstacle_temporal_emergency_vetted",
                            False,
                        )
                    )
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
                        if (
                            reverse_escape
                            and not post_retry_reverse_exhausted
                            and front_geometric_escape_available
                            and np.isfinite(recovery_heading_error)
                            and self._dynamic_escape_geometric_turn_sign != 0.0
                            and recovery_heading_error
                            * self._dynamic_escape_geometric_turn_sign
                            < 0.0
                            and self.dynamic_escape_reverse_goal_realign_gain
                            > 0.0
                            and self.dynamic_escape_reverse_goal_realign_max_omega_radps
                            > 0.0
                            and "omega_cmd" in self.action_spec.names
                        ):
                            omega_index = self.action_spec.index("omega_cmd")
                            values[omega_index] = np.clip(
                                self.dynamic_escape_reverse_goal_realign_gain
                                * recovery_heading_error,
                                -self.dynamic_escape_reverse_goal_realign_max_omega_radps,
                                self.dynamic_escape_reverse_goal_realign_max_omega_radps,
                            )
                            values = self.action_spec.clip(values)
                            reverse_goal_realign_applied = True
                        if reverse_escape and post_retry_reverse_exhausted:
                            # The encounter has already spent its finite first
                            # attempt, retry and reverse allowance.  Holding is
                            # safer and far more goal-directed than accepting
                            # another arbitrary MPPI reverse arc indefinitely.
                            if post_retry_side_forward_available:
                                values[self.action_spec.index("v_cmd")] = min(
                                    self.dynamic_escape_post_retry_side_forward_speed,
                                    self.action_spec.upper[
                                        self.action_spec.index("v_cmd")
                                    ],
                                )
                                if "omega_cmd" in self.action_spec.names:
                                    values[
                                        self.action_spec.index("omega_cmd")
                                    ] = 0.0
                                post_retry_side_forward_applied = True
                            else:
                                values[self.action_spec.index("v_cmd")] = 0.0
                                if "omega_cmd" in self.action_spec.names:
                                    values[
                                        self.action_spec.index("omega_cmd")
                                    ] = 0.0
                                post_retry_reverse_hold_applied = True
                            reverse_escape = False
                        if reverse_escape and front_geometric_escape_available:
                            self._dynamic_escape_vetted_reverse_steps += 1
                        # This is a finite authority budget, not a consecutive
                        # streak.  Stochastic MPPI alternated forward/reverse
                        # every scan in physical run 022756, so resetting on a
                        # single forward sample let it accumulate 49 reverse
                        # commands without ever reaching the four-step retry.
                        # Clear/rear encounter release and the explicit retry
                        # transaction reset the budget above.
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
        elif temporal_preturn_active:
            # Scan flow has established a closing front-sector threat, but the
            # tracker has not yet exported a forecast.  Start the same finite
            # passage side now; waiting in straight-line slowdown consumed two
            # of the roughly thirteen TTC frames in runs 022708/022756, plus
            # two more frames of gateway/chassis response latency.
            v_index = self.action_spec.index("v_cmd")
            omega_index = self.action_spec.index("omega_cmd")
            turn_sign, geometric_turn_source = (
                self._select_dynamic_escape_turn_sign(
                    guard_result, context, temporal_preturn_bearing
                )
            )
            self._dynamic_escape_geometric_turn_sign = turn_sign
            self._dynamic_escape_geometric_direction_prediction_backed = False
            self._dynamic_escape_frontal_encounter_latched = bool(
                self._dynamic_escape_frontal_encounter_latched
                or abs(temporal_preturn_bearing) < math.radians(20.0)
            )
            values[v_index] = min(
                self.dynamic_escape_temporal_preturn_speed,
                self.action_spec.upper[v_index],
            )
            values[omega_index] = (
                self.action_spec.upper[omega_index]
                if turn_sign > 0.0
                else self.action_spec.lower[omega_index]
            )
            values = self.action_spec.clip(values)
            temporal_preturn_applied = True
            # Keep the established reason so existing dynamic path authority
            # preserves this scan-vetted command through the path supervisor.
            reason = "temporal_slowdown"
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
                index = self.action_spec.index("v_cmd")
                if self._static_reverse_escape_permitted(
                    reason, guard_result, float(values[index])
                ):
                    # Honour the planner's own reverse, capped.  Never faster
                    # than the cap, never a sign the planner did not propose.
                    values[index] = max(
                        float(values[index]),
                        -abs(self.static_reverse_escape_max_speed),
                    )
                    self._static_reverse_escape_last_v = (
                        -abs(self.static_reverse_escape_max_speed)
                        if self.static_reverse_escape_hold_at_speed_cap_enabled
                        else float(values[index])
                    )
                    self._static_reverse_escape_run_steps += 1
                    self._static_reverse_escape_engaged_count += 1
                    reason = "static_reverse_escape"
                elif (
                    self.static_reverse_escape_hold_last_proposal_enabled
                    and float(values[index]) >= 0.0
                    and self._static_reverse_escape_last_v < 0.0
                    and self._static_reverse_escape_permitted(
                        reason,
                        guard_result,
                        self._static_reverse_escape_last_v,
                    )
                ):
                    # Continue only a planner-originated reverse sign and
                    # magnitude. Every held step rechecks the same static-only
                    # evidence, rear clearance and bounded authority.
                    values[index] = self._static_reverse_escape_last_v
                    self._static_reverse_escape_run_steps += 1
                    self._static_reverse_escape_engaged_count += 1
                    static_reverse_escape_held = True
                    reason = "static_reverse_escape"
                else:
                    values[index] = 0.0
                    # A non-reverse planner sample does not mean the robot has
                    # left the static trap. Keep the bounded run open across
                    # that one-cycle proposal gap; otherwise stochastic sign
                    # jitter immediately starts a cooldown and makes the
                    # configured multi-step reverse budget unreachable. Every
                    # later negative proposal is still re-checked against the
                    # dynamic exclusions, near-body evidence and rear scan.
                    proposal_gap_inside_same_trap = bool(
                        self.static_reverse_escape_enabled
                        and reason == "near_body_hard_stop"
                        and float(proposed.values[index]) >= 0.0
                        and self._static_reverse_escape_run_steps > 0
                        and self._static_reverse_escape_run_steps
                        < self.static_reverse_escape_max_consecutive_steps
                        and self._static_reverse_escape_engaged_count
                        < self.static_reverse_escape_max_total_steps
                    )
                    if (
                        self._static_reverse_escape_run_steps > 0
                        and not proposal_gap_inside_same_trap
                    ):
                        # A run has ended: impose the cooldown before another.
                        self._static_reverse_escape_cooldown_remaining = (
                            self.static_reverse_escape_cooldown_steps
                        )
                        self._static_reverse_escape_run_steps = 0
                        self._static_reverse_escape_last_v = 0.0
            else:
                self._static_reverse_escape_run_steps = 0
                self._static_reverse_escape_last_v = 0.0
            if (
                encounter_stop_only_hard_safety
                and "omega_cmd" in self.action_spec.names
            ):
                values[self.action_spec.index("omega_cmd")] = 0.0
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
                # A front-sector slowdown constrains motion *toward* the
                # obstacle.  Mapping a planner-vetted reverse command through
                # max(0, v) traps the robot against the same obstacle and
                # defeats static/dynamic candidate certification upstream.
                # Emergency and near-body hard stops are handled above and
                # remain fail-closed.
                if values[index] > 0.0:
                    values[index] *= float(
                        guard_result.get("slow_scale", 1.0)
                    )
                    if self.physical_front_speed_governor_enabled:
                        front_range = guard_result.get("min_front_range")
                        if front_range is not None and np.isfinite(front_range):
                            available = max(
                                0.0,
                                float(front_range)
                                - self.physical_front_speed_governor_clearance_m,
                            )
                            deceleration = (
                                self.physical_front_speed_governor_deceleration_mps2
                            )
                            reaction = self.physical_front_speed_governor_reaction_s
                            speed_cap = max(
                                0.0,
                                -deceleration * reaction
                                + math.sqrt(
                                    (deceleration * reaction) ** 2
                                    + 2.0 * deceleration * available
                                ),
                            )
                            values[index] = min(values[index], speed_cap)
        # Apply the physical front-clearance governor to the final selected
        # command, including active-avoidance and planner-vetted branches.  The
        # earlier implementation applied it only inside the ordinary slowdown
        # branch, allowing a +0.50 m/s escape candidate to bypass the same
        # stopping envelope at 0.55--0.65 m in the 085438 frontal run.
        physical_front_speed_governor_applied = False
        if (
            self.physical_front_speed_governor_enabled
            and "v_cmd" in self.action_spec.names
        ):
            v_index = self.action_spec.index("v_cmd")
            front_range = guard_result.get("min_front_range")
            if (
                values[v_index] > 0.0
                and front_range is not None
                and np.isfinite(front_range)
            ):
                available = max(
                    0.0,
                    float(front_range)
                    - self.physical_front_speed_governor_clearance_m,
                )
                deceleration = (
                    self.physical_front_speed_governor_deceleration_mps2
                )
                reaction = self.physical_front_speed_governor_reaction_s
                speed_cap = max(
                    0.0,
                    -deceleration * reaction
                    + math.sqrt(
                        (deceleration * reaction) ** 2
                        + 2.0 * deceleration * available
                    ),
                )
                if values[v_index] > speed_cap:
                    values[v_index] = speed_cap
                    physical_front_speed_governor_applied = True
        if (
            not dynamic_escape_allowed
            and not corridor_commit_active
            and not committed_geometric_escape
            and not geometric_forward_coast
        ):
            self._dynamic_escape_direction_commit_remaining = 0
            self._dynamic_escape_direction_commit_values = None
            self._dynamic_escape_corridor_remaining = 0
            self._dynamic_escape_corridor_turn_remaining = 0
            self._dynamic_escape_corridor_turn_sign = 0.0
        dynamic_zero_translation_turn_guard = bool(
            reason in {
                "dynamic_active_escape",
                "dynamic_corridor_escape",
                "dynamic_hard_stop_escape",
                "dynamic_recovery_align",
                "dynamic_recovery_align_creep",
                "temporal_collision_risk",
                "temporal_slowdown",
            }
            or (
                guard_result.get("emergency_stop", False)
                and (
                    guard_result.get(
                        "dynamic_obstacle_near_body_match", False
                    )
                    or guard_result.get(
                        "dynamic_obstacle_scan_flow_match", False
                    )
                    or context.get(
                        "probabilistic_obstacle_active_avoidance_enabled",
                        False,
                    )
                )
            )
        )
        dynamic_zero_translation_turn_suppressed = False
        dynamic_recovery_zero_turn_budget_release = False
        if "v_cmd" in self.action_spec.names and "omega_cmd" in (
            self.action_spec.names
        ):
            v_index = self.action_spec.index("v_cmd")
            omega_index = self.action_spec.index("omega_cmd")
            zero_translation = abs(float(values[v_index])) < 0.02
            turning = abs(float(values[omega_index])) >= 0.10
            if dynamic_zero_translation_turn_guard and zero_translation:
                if turning:
                    if (
                        self._dynamic_escape_zero_translation_turn_steps
                        >= self.dynamic_escape_max_zero_translation_turn_steps
                    ):
                        values[omega_index] = 0.0
                        dynamic_zero_translation_turn_suppressed = True
                        if (
                            self._dynamic_recovery_active
                            and reason in {
                                "dynamic_recovery_align",
                                "dynamic_recovery_align_creep",
                            }
                        ):
                            # Recovery cannot satisfy its heading tolerance once
                            # the independent zero-translation yaw budget has
                            # expired.  Releasing it here prevents an absorbing
                            # (v=0, omega=0) state and returns authority to the
                            # planner on the following cycle.
                            self._dynamic_recovery_active = False
                            self._dynamic_escape_seen = False
                            self._dynamic_recovery_clear_steps = 0
                            self._dynamic_recovery_release_count = 0
                            self._dynamic_recovery_advance_steps = 0
                            self._dynamic_recovery_alignment_creep_latched = (
                                False
                            )
                            self._dynamic_recovery_previous_scan_clearance = (
                                float("nan")
                            )
                            self._dynamic_recovery_clearance_trend_steps = 0
                            self._dynamic_recovery_progress_watch_active = False
                            self._dynamic_recovery_progress_watch_remaining = 0
                            self._dynamic_recovery_progress_watch_samples = []
                            dynamic_recovery_zero_turn_budget_release = True
                            reason = "dynamic_recovery_budget_release"
                    else:
                        self._dynamic_escape_zero_translation_turn_steps += 1
                # Once the bounded budget is consumed, a suppressed zero-yaw
                # cycle must not re-arm it.  Translation or a genuinely clear
                # guard below is the only release.
            else:
                self._dynamic_escape_zero_translation_turn_steps = 0

        executed = ControlCommand(values, proposed.timestamp, "safety_arbitration")
        overridden = not np.allclose(executed.values, proposed.values, rtol=0.0, atol=1e-12)
        diagnostics = dict(guard_result)
        if (
            self.physical_front_speed_governor_enabled
            and "v_cmd" in self.action_spec.names
        ):
            diagnostics["physical_front_speed_governor_enabled"] = True
            diagnostics["physical_front_speed_governor_applied"] = bool(
                physical_front_speed_governor_applied
                or (
                    proposed.values[self.action_spec.index("v_cmd")]
                    > values[self.action_spec.index("v_cmd")] + 1e-12
                    and guard_result.get("reason")
                    == "front_obstacle_slow"
                )
            )
        diagnostics["dynamic_escape_allowed"] = dynamic_escape_allowed
        diagnostics["encounter_control_stop_only_hard_safety"] = bool(
            encounter_stop_only_hard_safety
        )
        diagnostics["encounter_control_hard_stop_escape_retained"] = bool(
            encounter_hard_stop_escape_retained
        )
        diagnostics["final_motion_owner"] = (
            "hard_stop"
            if encounter_stop_only_hard_safety
            and guard_result.get("emergency_stop", False)
            else "encounter"
            if encounter_control_authoritative
            else "safety"
            if overridden
            else "mppi"
        )
        diagnostics["directional_motion_guard_enabled"] = bool(
            self.directional_motion_guard_enabled
        )
        diagnostics["rear_pass_through_active"] = bool(
            rear_pass_through_active
        )
        diagnostics["rear_pass_through_raw_evidence"] = bool(
            rear_pass_through_raw_evidence
        )
        diagnostics["rear_pass_through_suppressed_by_encounter"] = bool(
            rear_pass_through_suppressed
        )
        diagnostics["encounter_control_authoritative"] = bool(
            encounter_control_authoritative
        )
        diagnostics["encounter_control_rear_pass_inhibited"] = bool(
            encounter_rear_pass_inhibited
        )
        diagnostics["encounter_control_dynamic_escape_inhibited"] = bool(
            encounter_dynamic_escape_inhibited
        )
        diagnostics["encounter_control_hard_safety_retained"] = bool(
            context.get("encounter_control_hard_safety_retained", False)
        )
        diagnostics["encounter_control_hard_stop_escape_retained"] = bool(
            encounter_hard_stop_escape_retained
        )
        diagnostics["rear_reverse_blocked"] = bool(rear_reverse_blocked)
        diagnostics["rear_pass_through_force_forward_enabled"] = bool(
            self.rear_pass_through_force_forward_enabled
        )
        diagnostics["rear_pass_through_force_forward_ready"] = bool(
            rear_pass_force_forward_ready
        )
        diagnostics["rear_pass_through_force_straight_enabled"] = bool(
            self.rear_pass_through_force_straight_enabled
        )
        diagnostics["rear_pass_through_front_clearance_m"] = (
            None
            if rear_pass_front_clearance is None
            else float(rear_pass_front_clearance)
        )
        diagnostics["rear_pass_through_bearing_rad"] = (
            None
            if rear_pass_through_bearing is None
            else float(rear_pass_through_bearing)
        )
        diagnostics["rear_pass_through_sources"] = tuple(
            rear_pass_through_sources
        )
        diagnostics["rear_pass_through_candidate_turn_sign"] = float(
            rear_pass_candidate_turn_sign
        )
        diagnostics["rear_pass_through_turn_sign"] = float(
            self._rear_pass_through_turn_sign
        )
        diagnostics["rear_pass_through_direction_lock_started"] = bool(
            rear_pass_direction_lock_started
        )
        diagnostics["rear_pass_through_direction_clear_streak"] = int(
            self._rear_pass_through_direction_clear_streak
        )
        diagnostics["dynamic_escape_max_zero_translation_turn_steps"] = int(
            self.dynamic_escape_max_zero_translation_turn_steps
        )
        diagnostics["dynamic_escape_zero_translation_turn_steps"] = int(
            self._dynamic_escape_zero_translation_turn_steps
        )
        diagnostics["dynamic_escape_zero_translation_turn_suppressed"] = bool(
            dynamic_zero_translation_turn_suppressed
        )
        diagnostics["dynamic_escape_reactive"] = (
            reactive_escape_allowed
        )
        diagnostics["dynamic_escape_uncertainty_fusion"] = (
            uncertainty_fusion_escape_allowed
        )
        diagnostics["dynamic_escape_geometric_forward"] = (
            geometric_forward_escape
        )
        diagnostics["dynamic_escape_geometric_forward_coast"] = bool(
            geometric_forward_coast
        )
        diagnostics["dynamic_escape_geometric_turn_sign"] = float(
            self._dynamic_escape_geometric_turn_sign
        )
        diagnostics["dynamic_escape_geometric_turn_source"] = str(
            geometric_turn_source
        )
        diagnostics["dynamic_escape_temporal_preturn_enabled"] = bool(
            self.dynamic_escape_temporal_preturn_enabled
        )
        diagnostics["dynamic_escape_temporal_preturn_applied"] = bool(
            temporal_preturn_applied
        )
        diagnostics["dynamic_escape_temporal_preturn_bearing_rad"] = float(
            temporal_preturn_bearing
        )
        diagnostics["dynamic_escape_measured_lateral_countermotion"] = bool(
            measured_lateral_countermotion_available
        )
        diagnostics["dynamic_escape_measured_reversal_candidate"] = bool(
            measured_reversal_candidate
        )
        diagnostics[
            "dynamic_escape_measured_reversal_confirmation_count"
        ] = int(measured_reversal_confirmation_count)
        diagnostics["dynamic_escape_measured_reversal_confirmed"] = bool(
            measured_reversal_confirmed
        )
        diagnostics["dynamic_escape_prediction_direction_refreshed"] = bool(
            prediction_direction_refreshed
        )
        diagnostics[
            "dynamic_escape_prediction_direction_late_acquisition_available"
        ] = bool(late_prediction_available)
        diagnostics[
            "dynamic_escape_prediction_direction_late_acquisition_forward_sector"
        ] = bool(late_prediction_forward_sector)
        diagnostics[
            "dynamic_escape_prediction_direction_late_acquisition_oblique_geometry"
        ] = bool(late_prediction_oblique_geometry)
        diagnostics[
            "dynamic_escape_prediction_direction_late_acquisition_applied"
        ] = bool(late_prediction_direction_refresh)
        diagnostics[
            "dynamic_escape_geometric_direction_prediction_backed"
        ] = bool(
            self._dynamic_escape_geometric_direction_prediction_backed
        )
        diagnostics[
            "dynamic_escape_frontal_encounter_latched"
        ] = bool(self._dynamic_escape_frontal_encounter_latched)
        diagnostics[
            "dynamic_escape_prediction_direction_refresh_requested"
        ] = bool(prediction_direction_refresh_requested)
        diagnostics[
            "dynamic_escape_prediction_direction_refresh_confirmation_count"
        ] = int(prediction_direction_refresh_confirmation_count)
        diagnostics[
            "dynamic_escape_prediction_direction_refresh_confirmed"
        ] = bool(prediction_direction_refresh_confirmed)
        diagnostics[
            "dynamic_escape_prediction_direction_refresh_rejected"
        ] = bool(
            prediction_direction_refresh_requested
            and not prediction_direction_refreshed
        )
        diagnostics[
            "dynamic_escape_prediction_direction_refresh_geometry_allowed"
        ] = bool(prediction_direction_refresh_geometry_allowed)
        diagnostics[
            "dynamic_escape_prediction_direction_lateral_speed_mps"
        ] = (
            float(lateral_motion_speed)
            if math.isfinite(lateral_motion_speed)
            else None
        )
        diagnostics["dynamic_escape_coast_direction_locked"] = bool(
            geometric_coast_direction_locked
        )
        diagnostics["dynamic_escape_coast_remaining"] = int(
            self._dynamic_escape_coast_remaining
        )
        diagnostics["dynamic_escape_frontal_entry_speed_applied"] = bool(
            frontal_entry_speed_applied
        )
        diagnostics["dynamic_escape_persistent_front_retry_enabled"] = bool(
            self.dynamic_escape_persistent_front_retry_enabled
        )
        diagnostics["dynamic_escape_persistent_front_retry_rearmed"] = bool(
            persistent_front_retry_rearmed
        )
        diagnostics[
            "dynamic_escape_persistent_front_retry_goal_rejected"
        ] = bool(persistent_front_retry_goal_rejected)
        diagnostics["dynamic_escape_persistent_front_retry_count"] = int(
            self._dynamic_escape_persistent_front_retry_count
        )
        diagnostics["dynamic_escape_vetted_reverse_steps"] = int(
            self._dynamic_escape_vetted_reverse_steps
        )
        diagnostics["dynamic_escape_goal_diverged_from_escape_side"] = bool(
            goal_diverged_from_escape_side
        )
        diagnostics["dynamic_escape_geometric_goal_release_applied"] = bool(
            geometric_goal_release_applied
        )
        diagnostics["dynamic_escape_frontal_passage_completion_ready"] = bool(
            frontal_passage_completion_ready
        )
        diagnostics[
            "dynamic_escape_geometric_passage_completion_applied"
        ] = bool(geometric_passage_completion_applied)
        diagnostics["dynamic_escape_post_retry_reverse_exhausted"] = bool(
            post_retry_reverse_exhausted
        )
        diagnostics["dynamic_escape_post_retry_reverse_hold_applied"] = bool(
            post_retry_reverse_hold_applied
        )
        diagnostics["dynamic_escape_post_retry_side_forward_available"] = bool(
            post_retry_side_forward_available
        )
        diagnostics["dynamic_escape_post_retry_side_forward_applied"] = bool(
            post_retry_side_forward_applied
        )
        diagnostics["dynamic_escape_reverse_goal_realign_applied"] = bool(
            reverse_goal_realign_applied
        )
        diagnostics["dynamic_escape_hard_stop_enabled"] = bool(
            self.dynamic_escape_hard_stop_enabled
        )
        diagnostics["dynamic_escape_hard_stop_event"] = bool(
            dynamic_hard_stop_event
        )
        diagnostics["dynamic_escape_hard_stop_geometric_event"] = bool(
            dynamic_hard_stop_geometric_event
        )
        diagnostics["dynamic_escape_hard_stop_transaction_active"] = bool(
            (dynamic_hard_stop_event
             and not self._dynamic_escape_hard_stop_consumed)
            or hard_stop_transaction_active
            or hard_stop_transaction_completed_hold
        )
        diagnostics["dynamic_escape_hard_stop_transaction_held"] = bool(
            (
                hard_stop_transaction_active
                or hard_stop_transaction_completed_hold
            )
            and not dynamic_hard_stop_event
        )
        diagnostics[
            "dynamic_escape_hard_stop_prediction_evidence"
        ] = bool(dynamic_hard_stop_prediction_evidence)
        diagnostics["dynamic_escape_hard_stop_phase"] = str(
            hard_stop_escape_phase
        )
        diagnostics["dynamic_escape_hard_stop_reverse_authorized"] = bool(
            hard_stop_reverse_authorized
        )
        diagnostics["dynamic_escape_hard_stop_reverse_turn_scale"] = float(
            hard_stop_reverse_turn_scale
        )
        diagnostics["dynamic_escape_hard_stop_rear_clear"] = bool(
            hard_stop_rear_clear
        )
        diagnostics[
            "dynamic_escape_hard_stop_direction_refresh_enabled"
        ] = bool(self.dynamic_escape_hard_stop_direction_refresh_enabled)
        diagnostics[
            "dynamic_escape_hard_stop_direction_refresh_applied"
        ] = bool(hard_stop_direction_refresh_applied)
        diagnostics[
            "dynamic_escape_hard_stop_rear_blocked_latched"
        ] = bool(self._dynamic_escape_hard_stop_rear_blocked_latched)
        diagnostics[
            "dynamic_escape_hard_stop_rear_clear_retry_started"
        ] = bool(hard_stop_rear_clear_retry_started)
        diagnostics[
            "dynamic_escape_hard_stop_rear_clear_retry_used"
        ] = bool(self._dynamic_escape_hard_stop_rear_clear_retry_used)
        diagnostics[
            "dynamic_escape_hard_stop_rear_blocked_wait_remaining"
        ] = int(self._dynamic_escape_hard_stop_rear_blocked_wait_remaining)
        diagnostics[
            "dynamic_escape_hard_stop_completed_hold_count"
        ] = int(self._dynamic_escape_hard_stop_completed_hold_count)
        diagnostics[
            "dynamic_escape_hard_stop_completed_fail_closed"
        ] = bool(hard_stop_transaction_completed_fail_closed)
        diagnostics[
            "dynamic_escape_hard_stop_side_rear_release_available"
        ] = bool(hard_stop_side_rear_release_available)
        diagnostics[
            "dynamic_escape_hard_stop_side_rear_release_applied"
        ] = bool(
            hard_stop_escape_phase == "side_rear_forward_release"
        )
        diagnostics[
            "dynamic_escape_hard_stop_side_rear_release_latched"
        ] = bool(self._dynamic_escape_hard_stop_side_rear_release_latched)
        diagnostics[
            "dynamic_escape_hard_stop_side_rear_release_abort_count"
        ] = int(self._dynamic_escape_hard_stop_side_rear_release_abort_count)
        diagnostics[
            "dynamic_escape_hard_stop_side_rear_release_hold_bearing_rad"
        ] = float(hard_stop_side_rear_hold_bearing)
        diagnostics["dynamic_escape_hard_stop_turn_remaining"] = int(
            self._dynamic_escape_hard_stop_turn_remaining
        )
        diagnostics["dynamic_escape_hard_stop_reverse_remaining"] = int(
            self._dynamic_escape_hard_stop_reverse_remaining
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
        diagnostics["dynamic_escape_geometric_single_commit_enabled"] = bool(
            self.dynamic_escape_geometric_single_commit_enabled
        )
        diagnostics["dynamic_escape_geometric_commit_consumed"] = bool(
            self._dynamic_escape_geometric_commit_consumed
        )
        diagnostics["dynamic_escape_geometric_rearm_clear_streak"] = int(
            self._dynamic_escape_geometric_clear_streak
        )
        diagnostics["dynamic_escape_geometric_threat_evidence_active"] = bool(
            geometric_threat_evidence_active
        )
        diagnostics[
            "dynamic_escape_prediction_backed_temporal_override"
        ] = bool(prediction_backed_temporal_escape_override)
        diagnostics[
            "dynamic_escape_geometric_temporal_override"
        ] = bool(geometric_temporal_escape_override)
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
        diagnostics[
            "dynamic_recovery_zero_turn_budget_release"
        ] = bool(dynamic_recovery_zero_turn_budget_release)
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
        diagnostics["static_reverse_escape_enabled"] = bool(
            self.static_reverse_escape_enabled
        )
        diagnostics["static_reverse_escape_engaged"] = bool(
            reason == "static_reverse_escape"
        )
        diagnostics["static_reverse_escape_hold_last_proposal_enabled"] = bool(
            self.static_reverse_escape_hold_last_proposal_enabled
        )
        diagnostics["static_reverse_escape_hold_at_speed_cap_enabled"] = bool(
            self.static_reverse_escape_hold_at_speed_cap_enabled
        )
        diagnostics["static_reverse_escape_held"] = bool(
            static_reverse_escape_held
        )
        diagnostics["static_reverse_escape_run_steps"] = int(
            self._static_reverse_escape_run_steps
        )
        diagnostics["static_reverse_escape_engaged_count"] = int(
            self._static_reverse_escape_engaged_count
        )
        diagnostics["static_reverse_escape_cooldown_remaining"] = int(
            self._static_reverse_escape_cooldown_remaining
        )
        return SafetyDecision(
            proposed, executed, overridden, reason, diagnostics
        )
