"""Dimension-agnostic MPPI controller with plugin cost and prior ports."""

import hashlib
import os
import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Memoisation of `MppiController._known_static_map_clearance`.  Enabled by
# default; it is provably result-preserving (the function is pure and the key
# is a digest of the input bytes) and removes ~49% of calls.  Set the
# environment variable MPPI_DISABLE_STATIC_CLEARANCE_CACHE=1 to bypass it,
# which exists so the cache can be A/B timed and so a regression test can
# assert both paths agree bit-for-bit.
STATIC_CLEARANCE_CACHE_ENABLED = (
    os.environ.get("MPPI_DISABLE_STATIC_CLEARANCE_CACHE", "") not in ("1", "true", "True")
)

from mobile_robot_mppi.core.spaces import ActionSpec, StateSpec
from mobile_robot_mppi.core.types import ControlCommand, PlanResult, RobotObservation
from mobile_robot_mppi.obstacles.collision_risk import (
    CollisionRiskConfig,
    GaussianMixtureObstacleForecast,
    evaluate_collision_risk,
    evaluate_collision_risk_cuda,
)
from mobile_robot_mppi.planning.candidate_diagnostics import (
    reverse_candidate_diagnostics,
)
from mobile_robot_mppi.planning.static_astar import plan_static_astar_path
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior, PriorOutput


def _wrap_periodic(state: np.ndarray, state_spec: StateSpec) -> np.ndarray:
    for index in state_spec.periodic_indices:
        state[..., index] = np.arctan2(np.sin(state[..., index]), np.cos(state[..., index]))
    return state


def integrate_batch(dynamics, state, control, dt, state_spec, method="euler"):
    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if dt <= 0.0 or not np.isfinite(state).all() or not np.isfinite(control).all():
        raise ValueError("integration inputs must be finite and dt positive")
    if method == "euler":
        result = state + dt * dynamics.derivative(state, control)
    elif method == "rk4":
        k1 = dynamics.derivative(state, control)
        k2 = dynamics.derivative(state + 0.5 * dt * k1, control)
        k3 = dynamics.derivative(state + 0.5 * dt * k2, control)
        k4 = dynamics.derivative(state + dt * k3, control)
        result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    else:
        raise ValueError("unknown integration method: %s" % method)
    if not np.isfinite(result).all():
        raise FloatingPointError("rollout integration produced NaN or Inf")
    return _wrap_periodic(result, state_spec)


@dataclass(frozen=True)
class MppiConfig:
    horizon: int = 25
    num_samples: int = 400
    dt: float = 0.1
    command_delay_s: float = 0.0
    temperature: float = 4.0
    noise_sigma: Tuple[float, ...] = (0.12, 0.35)
    # Temporal correlation structure of the sampled control perturbation.
    # "iid" reproduces the historical sampler exactly and is the default, so
    # every frozen protocol keeps bit-identical behaviour. Alternatives
    # ("piecewise:<k>", "ar1:<tau_s>") preserve the per-step marginal variance
    # and change only the correlation across horizon steps.
    noise_basis: str = "iid"
    # Segment obstacles: when False (default) the known-static-map clearance
    # subtracts the full scene ``thickness``, reproducing historical behaviour
    # exactly. When True it subtracts ``0.5 * thickness``, matching the geometry
    # MuJoCo actually builds (model_factory.py) and the convention already used
    # by mujoco_plant, scene_feasibility and static_astar.
    known_static_map_segment_half_thickness: bool = False
    integrator: str = "rk4"
    goal_running_weight: float = 1.0
    goal_terminal_weight: float = 12.0
    heading_weight: float = 0.2
    path_preview_enabled: bool = False
    path_preview_speed_mps: float = 0.45
    path_preview_heading_weight: float = 0.0
    path_boundary_enabled: bool = False
    path_boundary_buffer: float = 0.0
    path_boundary_weight: float = 0.0
    path_boundary_violation_penalty: float = 0.0
    path_boundary_candidate_filter_enabled: bool = False
    terminal_velocity_weight: float = 0.0
    terminal_yaw_rate_weight: float = 0.0
    terminal_bearing_weight: float = 0.0
    terminal_translation_speed_limit: Optional[float] = None
    terminal_translation_heading_gate_rad: Optional[float] = None
    terminal_alignment_yaw_gain: Optional[float] = None
    terminal_control_radius: Optional[float] = None
    # Lower bound on the terminal heading gate's translation scale.  The gate
    # exists to force in-place alignment, but it is applied to the *final*
    # action, after avoidance has chosen a command, so a scale of 0.0 also
    # cancels evasive translation.  A differential-drive robot whose v is
    # zeroed cannot change its position at all, so clearance can only decay
    # while it rotates.  Raising this floor keeps the gate's alignment and
    # speed-limit behaviour while preserving translation authority.
    # 0.0 reproduces the historical gate exactly.
    terminal_translation_minimum_scale: float = 0.0
    control_weight: float = 0.05
    control_rate_weight: float = 0.08
    obstacle_weight: float = 30.0
    obstacle_influence: float = 0.55
    robot_radius: float = 0.25
    collision_penalty: float = 5000.0
    known_static_map_cost_enabled: bool = False
    known_static_map_influence_m: float = 0.0
    known_static_map_weight: float = 0.0
    known_static_map_collision_penalty: float = 0.0
    known_static_map_candidate_filter_enabled: bool = False
    probabilistic_reference_authority_enabled: bool = False
    probabilistic_reference_authority_filter_alpha: float = 0.85
    probabilistic_reference_authority_minimum: float = 0.15
    probabilistic_reference_authority_risk_source: str = (
        "maximum_step_probability"
    )
    probabilistic_reference_progress_weight: float = 0.0
    static_astar_replan_enabled: bool = False
    static_astar_replan_static_hard_stop_enabled: bool = False
    static_astar_replan_corner_clamped_lookahead_enabled: bool = False
    static_astar_replan_sharp_corner_clamped_lookahead_enabled: bool = False
    static_astar_replan_sharp_corner_minimum_turn_rad: float = (
        np.pi / 4.0
    )
    # Floor on the clamped sharp-corner target's distance ahead of the robot.
    # The unfloored clamp puts the target exactly on the corner vertex, so the
    # pursuit distance collapses to zero on arrival and the robot can orbit the
    # vertex without its projection passing it -- the clamp then never
    # releases.  Measured on seed 791101302: route progress pinned at 3.48 m of
    # a 29.76 m route (0.070 route fraction) against 15.19 m without the clamp.
    # 0.0 reproduces the unfloored clamp exactly.
    static_astar_replan_sharp_corner_minimum_lookahead_m: float = 0.0
    # Optional readiness gate for the sharp-corner clamp.  ``None`` keeps the
    # historical unconditional clamp.  A positive value defers the first
    # sharp-corner clamp until that route-progress distance remains, then
    # retains the same distance as the target floor.
    static_astar_replan_sharp_corner_deferred_clamp_distance_m: Optional[
        float
    ] = None
    static_astar_replan_resolution_m: float = 0.10
    static_astar_replan_clearance_margin_m: float = 0.08
    static_astar_replan_deviation_m: float = 0.70
    static_astar_replan_stagnation_steps: int = 30
    static_astar_replan_minimum_progress_m: float = 0.12
    static_astar_replan_cooldown_steps: int = 30
    static_astar_replan_bounds_padding_m: float = 0.50
    probabilistic_obstacle_risk_enabled: bool = False
    probabilistic_obstacle_risk_weight: float = 0.0
    # Candidate-vs-forecast risk is a dense [K,H,M] operation.  Keep this
    # opt-in so CPU-only evaluation protocols retain their exact runtime path.
    probabilistic_obstacle_risk_cuda_enabled: bool = False
    probabilistic_obstacle_risk_cuda_device: str = "cuda"
    probabilistic_obstacle_hard_threshold: float = 0.20
    probabilistic_obstacle_hard_penalty: float = 10000.0
    probabilistic_obstacle_safety_margin: float = 0.10
    probabilistic_obstacle_minimum_std: float = 0.01
    probabilistic_obstacle_forecast_key: str = (
        "probabilistic_obstacle_forecasts"
    )
    probabilistic_obstacle_missing_forecast_action: str = "raise"
    probabilistic_obstacle_hard_violation_action: str = "penalize"
    probabilistic_obstacle_hard_violation_progress_tiebreak_enabled: bool = False
    probabilistic_obstacle_hard_violation_progress_mass_tolerance: float = 0.05
    probabilistic_obstacle_candidate_filter_enabled: bool = False
    probabilistic_obstacle_stopping_feasibility_enabled: bool = False
    probabilistic_obstacle_emergency_candidates_enabled: bool = False
    probabilistic_obstacle_emergency_candidate_prefix_steps: int = 5
    probabilistic_obstacle_emergency_candidate_hold_tail_enabled: bool = False
    probabilistic_obstacle_emergency_candidate_trigger_ttc_s: float = 0.0
    probabilistic_obstacle_emergency_candidate_trigger_distance_m: float = 0.0
    probabilistic_obstacle_emergency_candidate_critical_distance_m: float = 0.0
    probabilistic_obstacle_emergency_candidate_intent_hold_steps: int = 0
    # Real-robot emergency takeover must be supported by a trustworthy causal
    # scan flow.  These gates are opt-in so frozen simulation profiles retain
    # their historical admission rule.
    probabilistic_obstacle_emergency_min_support_beams: int = 0
    probabilistic_obstacle_emergency_max_rejected_jump_fraction: float = 1.0
    probabilistic_obstacle_emergency_require_scan_quality: bool = False
    probabilistic_obstacle_emergency_candidate_slew_enabled: bool = False
    # Emergency lattice members are normally scored with the same MPPI batch.
    # Historical replay profiles may retain the old direct fallback, while the
    # physical profile explicitly disables that second final-control writer.
    probabilistic_obstacle_emergency_candidate_direct_fallback_enabled: bool = True
    # A reverse first command must remain prediction-feasible even when the
    # generic risk fallback has no jointly feasible sample.  The physical
    # profile combines this with a raw-scan rear-sector veto downstream.
    probabilistic_obstacle_reverse_candidate_filter_enabled: bool = False
    probabilistic_obstacle_emergency_forecast_corroboration_enabled: bool = False
    probabilistic_obstacle_front_obstacle_forward_turn_enabled: bool = False
    probabilistic_obstacle_counterflow_escape_enabled: bool = False
    probabilistic_obstacle_counterflow_weight: float = 1.0
    # Differential-drive realization of a crossing prediction.  For a
    # significant body-frame lateral obstacle velocity, choose a forward-biased
    # robot direction whose lateral component has the opposite sign.  This
    # avoids asking the real chassis to face a counterflow vector behind it
    # before it can begin the lateral pass.  Default off preserves the frozen
    # simulation treatment; the physical profile enables it explicitly.
    probabilistic_obstacle_forward_lateral_countermotion_enabled: bool = False
    probabilistic_obstacle_forward_lateral_countermotion_weight: float = 1.20
    probabilistic_obstacle_forward_lateral_minimum_speed_mps: float = 0.10
    probabilistic_obstacle_forward_lateral_minimum_fraction: float = 0.25
    probabilistic_obstacle_forward_lateral_reversal_confirm_steps: int = 2
    # Counterflow rotates the escape away from the obstacle's tangent, which can
    # swing the preferred direction behind the robot.  Every collision measured
    # for the full arm on Map3-Redesign-D happened that way: penetration of only
    # 0.003-0.027 m, the traversal window disengaged (commit_active=0,
    # window_safe=0), and four of six impacts at NEGATIVE velocity.  The robot
    # reverses into a carrier rather than failing to cross one, and 66-71% of
    # reversing steps occur under counterflow escape.
    #
    # The guard vetoes the counterflow term when the direction it produces would
    # carry the robot toward a tracked obstacle that is already inside the guard
    # radius; the escape falls back to the pure perpendicular, which moves across
    # the obstacle's motion instead of back into it.  Tuning thresholds cannot
    # express this: raising the safety margin or the staging hold made matters
    # worse on every one of twelve screened variants, because both add retreat
    # authority and retreat is the behaviour that causes these collisions.
    probabilistic_obstacle_escape_rear_occupancy_guard_enabled: bool = False
    probabilistic_obstacle_escape_rear_occupancy_guard_radius_m: float = 0.70
    # Control-level rear guard.
    #
    # The direction-level guard above was measured and never fired: 0 vetoes over
    # 12 episodes while counterflow applied 733 times.  Counterflow steers AWAY
    # from the obstacle that triggered it, so that direction almost never points
    # at anything.  The causal step is one further down: the robot is
    # differential drive, so realising a world-frame direction behind its current
    # heading means COMMANDING REVERSE, and it then backs into a DIFFERENT
    # obstacle that nothing was checking.
    #
    # Every collision measured for the full arm on Map3-Redesign-D fits that:
    # impacts at -0.23 to -0.37 m/s, penetration 0.003-0.027 m, traversal window
    # disengaged, and contact with a carrier the robot was never facing.
    #
    # This guard acts on the commanded velocity itself.  When v_cmd is negative
    # and a tracked obstacle lies within `radius_m` inside a rear sector of
    # half-width `halfangle_deg` about the robot's backward axis, the reverse
    # command is scaled by `scale` (0.0 suppresses it outright).  Forward motion
    # is never touched, so the guard cannot cause the deadlock that tighter
    # margins and staging holds produced.
    probabilistic_obstacle_reverse_rear_guard_enabled: bool = False
    probabilistic_obstacle_reverse_rear_guard_radius_m: float = 0.75
    probabilistic_obstacle_reverse_rear_guard_halfangle_deg: float = 80.0
    probabilistic_obstacle_reverse_rear_guard_scale: float = 0.0
    # Checking only the obstacle's CURRENT position was measured and never
    # fired: 0 activations while the robot commanded reverse on 46 of 595 steps.
    # The carriers run at 0.54 m/s, faster than the robot reverses, so the
    # collision happens because a carrier ARRIVES where the robot is backing
    # into -- the obstacle is not behind the robot yet at the moment the reverse
    # is commanded.  The guard therefore scans the forecast horizon, not just
    # the present, and vetoes when any predicted position within the lookahead
    # falls in the rear sector.
    probabilistic_obstacle_reverse_rear_guard_lookahead_steps: int = 6
    probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled: bool = False
    # A directional transaction may prefer only reverse or only forward
    # emergency templates.  If that preference selects a hard-violating
    # candidate while the same already-evaluated emergency lattice contains a
    # feasible member, the frozen hard threshold must remain authoritative.
    # Default off preserves the registered method exactly.
    probabilistic_obstacle_emergency_feasibility_over_direction_enabled: bool = False
    probabilistic_obstacle_emergency_candidate_forward_risk_ceiling: float = 0.0
    probabilistic_obstacle_emergency_candidate_forward_mass_ceiling: float = 0.0
    probabilistic_obstacle_emergency_candidate_rearm_ttc_s: float = 0.0
    probabilistic_obstacle_emergency_candidate_rearm_clear_steps: int = 1
    probabilistic_obstacle_terminal_intent_safe_stop_enabled: bool = False
    probabilistic_obstacle_traversal_window_enabled: bool = False
    probabilistic_obstacle_traversal_window_horizon_steps: int = 60
    probabilistic_obstacle_traversal_window_activation_distance_m: float = 1.2
    probabilistic_obstacle_traversal_window_cross_track_m: float = 0.45
    probabilistic_obstacle_traversal_window_clearance_margin_m: float = 0.10
    probabilistic_obstacle_traversal_window_probability_ceiling: float = 0.05
    probabilistic_obstacle_traversal_window_mass_ceiling: float = 0.50
    probabilistic_obstacle_traversal_window_clear_hold_steps: int = 3
    probabilistic_obstacle_traversal_window_translation_heading_gate_rad: float = 0.0
    probabilistic_obstacle_traversal_window_terminal_target_bearing_enabled: bool = False
    # Terminal capture is a single, forecast-certified endpoint trajectory.
    # It is intentionally disabled by default so the historical planner path
    # remains unchanged.  When enabled, it uses the existing traversal slot
    # and risk/static filters rather than overriding the safety arbiter.
    probabilistic_obstacle_terminal_capture_candidate_enabled: bool = False
    probabilistic_obstacle_traversal_window_abort_probability: float = 0.0
    probabilistic_obstacle_traversal_window_temporal_abort_mass_floor: float = 0.0
    probabilistic_obstacle_traversal_window_temporal_abort_full_horizon_corroboration_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_abort_current_hazard_only_enabled: bool = False
    probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled: bool = False
    # Scope the traversal commit certificate to the carrier whose route crossing
    # is actually being traversed, instead of the union over every tracked
    # obstacle.  See _traversal_certificate_obstacles for the measurements.
    probabilistic_obstacle_traversal_window_commit_certificate_crossing_obstacle_only: bool = False
    probabilistic_obstacle_traversal_window_commit_clear_hold_tail_enabled: bool = False
    probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps: int = 1
    probabilistic_obstacle_traversal_window_commit_admission_prealign_enabled: bool = False
    probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_enabled: bool = False
    # When enabled, an uncommitted staging hold requires the existing
    # emergency TTC trigger to be active (or the raw emergency trigger to be
    # corroborated).  The historical path intentionally keeps using any
    # causal closing observation when this is false.
    probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_current_hazard_only_enabled: bool = False
    # Number of CONSECUTIVE steps of uncommitted staging hold after which the
    # robot retreats to a standoff instead of continuing to hold in place.
    # 0 keeps the historical behaviour of holding indefinitely.
    #
    # The uncommitted hold stages in place "until online geometry either exposes
    # a certifiable crossing or the closing signal clears".  Neither is
    # guaranteed: on chapter 3 a certified-admissible carrier drives along the
    # robot's route for 30.6% of its loop, so the closing signal frequently
    # never clears and no certifiable crossing ever appears.  Retreat cannot
    # rescue it either, because retreat_requested is gated behind
    # abort_started_commit -- an already-STARTED commit that then aborts -- and
    # chapter 3 commits 18 times in 50 seeds against chapter 1's 7855.  Measured
    # consequence: retreat fires on 1 step out of 36152 (chapter 1: 7.85%), the
    # robot holds at the route entrance, and 37 of 38 collisions happen after
    # step 400 with a median final goal distance of 4.16 m.
    probabilistic_obstacle_traversal_window_uncommitted_hold_retreat_steps: int = 0
    # Derive the traversal window's extent from the forecast's predicted route
    # intrusion span instead of padding the single closest crossing point by a
    # fixed clearance radius.
    #
    # Measured defect (chapter 3): the certified window is
    # clear_progress - entry_progress = 1.400 m at EVERY percentile (p10 = p90),
    # i.e. a constant 2 x 0.67 m radius, while carrier 2's actual intrusion spans
    # 2.23 m of route. The plan therefore covers 63% of the hazard: the robot
    # certifies clearing 1.4 m, clears exactly that, and is still inside the
    # remaining 0.8 m when the carrier returns. 33 of 38 collisions (87%) occur
    # inside that intrusion zone.
    #
    # The forecast localises the hazard correctly -- while approaching the zone
    # it places the crossing inside the true span on 80% of certified steps -- so
    # the span is available; only the extent applied to it was wrong.
    #
    # The window is taken as the UNION of the historical fixed-radius window and
    # the forecast-derived span, so it can only ever grow. A forecast that grazes
    # the route at a single point therefore still yields the old window rather
    # than a degenerate zero-length one.
    probabilistic_obstacle_traversal_window_forecast_extent_enabled: bool = False
    # Zone-occupancy scheduling: hold BEFORE entering the hazard zone until the
    # zone is geometrically clear for the whole transit, then go.
    #
    # Every earlier chapter-3 intervention routed through the Gaussian-mixture
    # risk bound, which SATURATES at exactly 1.0 on 47.4% of certificates -- it is
    # uninformative precisely when it matters, so no gate downstream of it can
    # work. This predicate instead uses forecast MEANS and a route-span test,
    # which cannot saturate.
    #
    # Premise measured on chapter 3: the hazard zone (11.86-14.09 m) is occupied
    # in regular 4.6 s intervals separated by 10.8 s clear gaps; 8 of 9 gaps fit
    # the 5.0 s transit, and the worst-case wait for a usable gap is 4.6 s.
    # Verifying a 5.0 s clear transit needs 50 forecast steps against a 64-step
    # horizon, so the check is available causally.
    probabilistic_obstacle_zone_occupancy_hold_enabled: bool = False
    # Nominal speed used to convert remaining zone distance into transit steps.
    probabilistic_obstacle_zone_occupancy_transit_speed_mps: float = 0.45
    # Extra clear steps required beyond the estimated transit, as margin.
    probabilistic_obstacle_zone_occupancy_clear_margin_steps: int = 5
    # How far ahead along the route to look for a hazard zone. The previous
    # placement inherited the traversal window's 1.2 m activation distance, which
    # left only a 0.53 m sliver in which the hold could fire -- it fired 0 times
    # across 12 episodes. Holding before a zone requires seeing it earlier.
    probabilistic_obstacle_zone_occupancy_lookahead_m: float = 4.0
    probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_terminal_release_enabled: bool = False
    probabilistic_obstacle_traversal_window_commit_admission_exit_deadline_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_abort_nearest_exit_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_midpoint_guard_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_exit_deadline_guard_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_no_crossing_clear_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_release_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_staging_approach_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_staging_frontier_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled: bool = False
    probabilistic_obstacle_traversal_window_rearm_temporal_closing_lattice_override_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_retreat_post_intent_all_hard_forward_filter_enabled: bool = False
    probabilistic_obstacle_traversal_window_temporal_midpoint_retreat_reverse_filter_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_low_ttc_nonforward_coverage_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_low_ttc_prefix_boundary_handoff_enabled: bool = False
    probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled: bool = False
    probabilistic_obstacle_traversal_window_retreat_margin_m: float = 0.0
    probabilistic_obstacle_traversal_window_retreat_hard_risk_override_enabled: bool = False
    probabilistic_obstacle_traversal_window_retreat_completion_frozen_frame_enabled: bool = False
    probabilistic_obstacle_traversal_window_retreat_completion_frozen_reference_enabled: bool = False
    probabilistic_obstacle_speed_governor_enabled: bool = False
    probabilistic_obstacle_speed_governor_start_ratio: float = 0.50
    # Keep the speed governor active while active-avoidance motion is selected.
    # False reproduces the historical bypass, under which commanded speed RISES
    # during evasion (applied_v p90 0.694 m/s pre-collision vs 0.530 m/s in safe
    # operation) and the feasible candidate set collapses to 0.425.
    probabilistic_obstacle_speed_governor_applies_during_active_avoidance: bool = False
    importance_sampling_correction: bool = False
    previous_sequence_blend: float = 0.5
    safety_recovery_prefix_steps: int = 0
    # Soft slowdowns and actuator interpolation still close the previous-
    # action loop, but must not be mistaken for a blocked translation.  Only
    # an actually stopped command may zero the future recovery prefix.
    safety_recovery_translation_stop_threshold: float = 1.0e-6
    profile_components: bool = False
    optimizer_diagnostics_enabled: bool = False
    seed: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], action_dim: int):
        sigma = values.get("noise_sigma", (0.12, 0.35))
        if np.isscalar(sigma):
            sigma = (float(sigma),) * int(action_dim)
        return cls(
            horizon=int(values.get("horizon", 25)),
            num_samples=int(values.get("num_samples", 400)),
            dt=float(values.get("dt", values.get("control_dt", 0.1))),
            command_delay_s=float(values.get("command_delay_s", 0.0)),
            temperature=float(values.get("temperature", 4.0)),
            noise_sigma=tuple(float(v) for v in sigma),
            noise_basis=str(values.get("noise_basis", "iid")),
            known_static_map_segment_half_thickness=bool(
                values.get("known_static_map_segment_half_thickness", False)
            ),
            integrator=str(values.get("integrator", "rk4")),
            goal_running_weight=float(values.get("goal_running_weight", 1.0)),
            goal_terminal_weight=float(values.get("goal_terminal_weight", 12.0)),
            heading_weight=float(values.get("heading_weight", 0.2)),
            path_preview_enabled=bool(values.get("path_preview_enabled", False)),
            path_preview_speed_mps=float(
                values.get("path_preview_speed_mps", 0.45)
            ),
            path_preview_heading_weight=float(
                values.get("path_preview_heading_weight", 0.0)
            ),
            path_boundary_enabled=bool(
                values.get("path_boundary_enabled", False)
            ),
            path_boundary_buffer=float(
                values.get("path_boundary_buffer", 0.0)
            ),
            path_boundary_weight=float(
                values.get("path_boundary_weight", 0.0)
            ),
            path_boundary_violation_penalty=float(
                values.get("path_boundary_violation_penalty", 0.0)
            ),
            path_boundary_candidate_filter_enabled=bool(
                values.get("path_boundary_candidate_filter_enabled", False)
            ),
            terminal_velocity_weight=float(values.get("terminal_velocity_weight", 0.0)),
            terminal_yaw_rate_weight=float(values.get("terminal_yaw_rate_weight", 0.0)),
            terminal_bearing_weight=float(values.get("terminal_bearing_weight", 0.0)),
            terminal_translation_speed_limit=(
                None
                if values.get("terminal_translation_speed_limit") is None
                else float(values["terminal_translation_speed_limit"])
            ),
            terminal_translation_heading_gate_rad=(
                None
                if values.get("terminal_translation_heading_gate_rad") is None
                else float(values["terminal_translation_heading_gate_rad"])
            ),
            terminal_alignment_yaw_gain=(
                None
                if values.get("terminal_alignment_yaw_gain") is None
                else float(values["terminal_alignment_yaw_gain"])
            ),
            terminal_control_radius=(
                None
                if values.get("terminal_control_radius") is None
                else float(values["terminal_control_radius"])
            ),
            terminal_translation_minimum_scale=float(
                values.get("terminal_translation_minimum_scale", 0.0)
            ),
            control_weight=float(values.get("control_weight", 0.05)),
            control_rate_weight=float(values.get("control_rate_weight", 0.08)),
            obstacle_weight=float(values.get("obstacle_weight", 30.0)),
            obstacle_influence=float(values.get("obstacle_influence", 0.55)),
            robot_radius=float(values.get("robot_radius", 0.25)),
            collision_penalty=float(values.get("collision_penalty", 5000.0)),
            known_static_map_cost_enabled=bool(
                values.get("known_static_map_cost_enabled", False)
            ),
            known_static_map_influence_m=float(
                values.get("known_static_map_influence_m", 0.0)
            ),
            known_static_map_weight=float(
                values.get("known_static_map_weight", 0.0)
            ),
            known_static_map_collision_penalty=float(
                values.get("known_static_map_collision_penalty", 0.0)
            ),
            known_static_map_candidate_filter_enabled=bool(
                values.get(
                    "known_static_map_candidate_filter_enabled", False
                )
            ),
            probabilistic_reference_authority_enabled=bool(
                values.get(
                    "probabilistic_reference_authority_enabled", False
                )
            ),
            probabilistic_reference_authority_filter_alpha=float(
                values.get(
                    "probabilistic_reference_authority_filter_alpha", 0.85
                )
            ),
            probabilistic_reference_authority_minimum=float(
                values.get(
                    "probabilistic_reference_authority_minimum", 0.15
                )
            ),
            probabilistic_reference_authority_risk_source=str(
                values.get(
                    "probabilistic_reference_authority_risk_source",
                    "maximum_step_probability",
                )
            ),
            probabilistic_reference_progress_weight=float(
                values.get(
                    "probabilistic_reference_progress_weight", 0.0
                )
            ),
            static_astar_replan_enabled=bool(
                values.get("static_astar_replan_enabled", False)
            ),
            static_astar_replan_static_hard_stop_enabled=bool(
                values.get(
                    "static_astar_replan_static_hard_stop_enabled",
                    False,
                )
            ),
            static_astar_replan_corner_clamped_lookahead_enabled=bool(
                values.get(
                    "static_astar_replan_corner_clamped_lookahead_enabled",
                    False,
                )
            ),
            static_astar_replan_sharp_corner_clamped_lookahead_enabled=bool(
                values.get(
                    "static_astar_replan_sharp_corner_clamped_lookahead_enabled",
                    False,
                )
            ),
            static_astar_replan_sharp_corner_minimum_turn_rad=float(
                values.get(
                    "static_astar_replan_sharp_corner_minimum_turn_rad",
                    np.pi / 4.0,
                )
            ),
            static_astar_replan_sharp_corner_minimum_lookahead_m=float(
                values.get(
                    "static_astar_replan_sharp_corner_minimum_lookahead_m",
                    0.0,
                )
            ),
            static_astar_replan_sharp_corner_deferred_clamp_distance_m=(
                None
                if values.get(
                    "static_astar_replan_sharp_corner_deferred_clamp_distance_m",
                    None,
                )
                is None
                else float(
                    values[
                        "static_astar_replan_sharp_corner_deferred_clamp_distance_m"
                    ]
                )
            ),
            static_astar_replan_resolution_m=float(
                values.get("static_astar_replan_resolution_m", 0.10)
            ),
            static_astar_replan_clearance_margin_m=float(
                values.get(
                    "static_astar_replan_clearance_margin_m", 0.08
                )
            ),
            static_astar_replan_deviation_m=float(
                values.get("static_astar_replan_deviation_m", 0.70)
            ),
            static_astar_replan_stagnation_steps=int(
                values.get("static_astar_replan_stagnation_steps", 30)
            ),
            static_astar_replan_minimum_progress_m=float(
                values.get(
                    "static_astar_replan_minimum_progress_m", 0.12
                )
            ),
            static_astar_replan_cooldown_steps=int(
                values.get("static_astar_replan_cooldown_steps", 30)
            ),
            static_astar_replan_bounds_padding_m=float(
                values.get(
                    "static_astar_replan_bounds_padding_m", 0.50
                )
            ),
            probabilistic_obstacle_risk_enabled=bool(
                values.get("probabilistic_obstacle_risk_enabled", False)
            ),
            probabilistic_obstacle_risk_weight=float(
                values.get("probabilistic_obstacle_risk_weight", 0.0)
            ),
            probabilistic_obstacle_risk_cuda_enabled=bool(
                values.get("probabilistic_obstacle_risk_cuda_enabled", False)
            ),
            probabilistic_obstacle_risk_cuda_device=str(
                values.get("probabilistic_obstacle_risk_cuda_device", "cuda")
            ),
            probabilistic_obstacle_hard_threshold=float(
                values.get("probabilistic_obstacle_hard_threshold", 0.20)
            ),
            probabilistic_obstacle_hard_penalty=float(
                values.get("probabilistic_obstacle_hard_penalty", 10000.0)
            ),
            probabilistic_obstacle_safety_margin=float(
                values.get("probabilistic_obstacle_safety_margin", 0.10)
            ),
            probabilistic_obstacle_minimum_std=float(
                values.get("probabilistic_obstacle_minimum_std", 0.01)
            ),
            probabilistic_obstacle_forecast_key=str(
                values.get(
                    "probabilistic_obstacle_forecast_key",
                    "probabilistic_obstacle_forecasts",
                )
            ),
            probabilistic_obstacle_missing_forecast_action=str(
                values.get(
                    "probabilistic_obstacle_missing_forecast_action",
                    "raise",
                )
            ),
            probabilistic_obstacle_hard_violation_action=str(
                values.get(
                    "probabilistic_obstacle_hard_violation_action",
                    "penalize",
                )
            ),
            probabilistic_obstacle_hard_violation_progress_tiebreak_enabled=bool(
                values.get(
                    "probabilistic_obstacle_hard_violation_progress_tiebreak_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_hard_violation_progress_mass_tolerance=float(
                values.get(
                    "probabilistic_obstacle_hard_violation_progress_mass_tolerance",
                    0.05,
                )
            ),
            probabilistic_obstacle_candidate_filter_enabled=bool(
                values.get(
                    "probabilistic_obstacle_candidate_filter_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_stopping_feasibility_enabled=bool(
                values.get(
                    "probabilistic_obstacle_stopping_feasibility_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_candidates_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_candidates_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_candidate_prefix_steps=int(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_prefix_steps",
                    5,
                )
            ),
            probabilistic_obstacle_emergency_candidate_hold_tail_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_hold_tail_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_candidate_trigger_ttc_s=float(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_trigger_ttc_s",
                    0.0,
                )
            ),
            probabilistic_obstacle_emergency_candidate_trigger_distance_m=float(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_trigger_distance_m",
                    0.0,
                )
            ),
            probabilistic_obstacle_emergency_candidate_critical_distance_m=float(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_critical_distance_m",
                    0.0,
                )
            ),
            probabilistic_obstacle_emergency_candidate_intent_hold_steps=int(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_intent_hold_steps",
                    0,
                )
            ),
            probabilistic_obstacle_emergency_forecast_corroboration_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_forecast_corroboration_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_front_obstacle_forward_turn_enabled=bool(
                values.get(
                    "probabilistic_obstacle_front_obstacle_forward_turn_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_counterflow_escape_enabled=bool(
                values.get(
                    "probabilistic_obstacle_counterflow_escape_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_counterflow_weight=float(
                values.get(
                    "probabilistic_obstacle_counterflow_weight",
                    1.0,
                )
            ),
            probabilistic_obstacle_escape_rear_occupancy_guard_enabled=bool(
                values.get(
                    "probabilistic_obstacle_escape_rear_occupancy_guard_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_escape_rear_occupancy_guard_radius_m=float(
                values.get(
                    "probabilistic_obstacle_escape_rear_occupancy_guard_radius_m",
                    0.70,
                )
            ),
            probabilistic_obstacle_reverse_rear_guard_enabled=bool(
                values.get(
                    "probabilistic_obstacle_reverse_rear_guard_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_reverse_rear_guard_radius_m=float(
                values.get(
                    "probabilistic_obstacle_reverse_rear_guard_radius_m",
                    0.75,
                )
            ),
            probabilistic_obstacle_reverse_rear_guard_halfangle_deg=float(
                values.get(
                    "probabilistic_obstacle_reverse_rear_guard_halfangle_deg",
                    80.0,
                )
            ),
            probabilistic_obstacle_reverse_rear_guard_scale=float(
                values.get(
                    "probabilistic_obstacle_reverse_rear_guard_scale",
                    0.0,
                )
            ),
            probabilistic_obstacle_reverse_rear_guard_lookahead_steps=int(
                values.get(
                    "probabilistic_obstacle_reverse_rear_guard_lookahead_steps",
                    6,
                )
            ),
            probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_feasibility_over_direction_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_feasibility_over_direction_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_candidate_forward_risk_ceiling=float(
                values.get(
                    'probabilistic_obstacle_emergency_candidate_forward_risk_ceiling',
                    0.0,
                )
            ),
            probabilistic_obstacle_emergency_candidate_forward_mass_ceiling=float(
                values.get(
                    'probabilistic_obstacle_emergency_candidate_forward_mass_ceiling',
                    0.0,
                )
            ),
            probabilistic_obstacle_emergency_candidate_rearm_ttc_s=float(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_rearm_ttc_s",
                    0.0,
                )
            ),
            probabilistic_obstacle_emergency_candidate_rearm_clear_steps=int(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_rearm_clear_steps",
                    1,
                )
            ),
            probabilistic_obstacle_terminal_intent_safe_stop_enabled=bool(
                values.get(
                    "probabilistic_obstacle_terminal_intent_safe_stop_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_enabled", False
                )
            ),
            probabilistic_obstacle_traversal_window_horizon_steps=int(
                values.get(
                    "probabilistic_obstacle_traversal_window_horizon_steps", 60
                )
            ),
            probabilistic_obstacle_traversal_window_activation_distance_m=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_activation_distance_m",
                    1.2,
                )
            ),
            probabilistic_obstacle_traversal_window_cross_track_m=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_cross_track_m", 0.45
                )
            ),
            probabilistic_obstacle_traversal_window_clearance_margin_m=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_clearance_margin_m",
                    0.10,
                )
            ),
            probabilistic_obstacle_traversal_window_probability_ceiling=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_probability_ceiling",
                    0.05,
                )
            ),
            probabilistic_obstacle_traversal_window_mass_ceiling=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_mass_ceiling", 0.50
                )
            ),
            probabilistic_obstacle_traversal_window_clear_hold_steps=int(
                values.get(
                    "probabilistic_obstacle_traversal_window_clear_hold_steps", 3
                )
            ),
            probabilistic_obstacle_traversal_window_translation_heading_gate_rad=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_translation_heading_gate_rad",
                    0.0,
                )
            ),
            probabilistic_obstacle_traversal_window_terminal_target_bearing_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_terminal_target_bearing_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_terminal_capture_candidate_enabled=bool(
                values.get(
                    "probabilistic_obstacle_terminal_capture_candidate_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_abort_probability=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_abort_probability",
                    0.0,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_mass_floor=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_abort_mass_floor",
                    0.0,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_full_horizon_corroboration_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_abort_full_horizon_corroboration_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_current_hazard_only_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_abort_current_hazard_only_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_commit_certificate_crossing_obstacle_only=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_commit_certificate_crossing_obstacle_only",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_commit_clear_hold_tail_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_commit_clear_hold_tail_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps=int(
                values.get(
                    "probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps",
                    1,
                )
            ),
            probabilistic_obstacle_traversal_window_commit_admission_prealign_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_commit_admission_prealign_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_current_hazard_only_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_current_hazard_only_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_uncommitted_hold_retreat_steps=int(
                values.get(
                    "probabilistic_obstacle_traversal_window_uncommitted_hold_retreat_steps",
                    0,
                )
            ),
            probabilistic_obstacle_traversal_window_forecast_extent_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_forecast_extent_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_zone_occupancy_hold_enabled=bool(
                values.get(
                    "probabilistic_obstacle_zone_occupancy_hold_enabled", False
                )
            ),
            probabilistic_obstacle_zone_occupancy_transit_speed_mps=float(
                values.get(
                    "probabilistic_obstacle_zone_occupancy_transit_speed_mps",
                    0.45,
                )
            ),
            probabilistic_obstacle_zone_occupancy_clear_margin_steps=int(
                values.get(
                    "probabilistic_obstacle_zone_occupancy_clear_margin_steps",
                    5,
                )
            ),
            probabilistic_obstacle_zone_occupancy_lookahead_m=float(
                values.get(
                    "probabilistic_obstacle_zone_occupancy_lookahead_m", 4.0
                )
            ),
            probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_terminal_release_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_terminal_release_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_commit_admission_exit_deadline_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_commit_admission_exit_deadline_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_nearest_exit_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_abort_nearest_exit_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_midpoint_guard_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_midpoint_guard_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_exit_deadline_guard_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_exit_deadline_guard_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_no_crossing_clear_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_no_crossing_clear_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_release_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_release_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_staging_approach_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_staging_approach_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_staging_frontier_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_staging_frontier_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_rearm_temporal_closing_lattice_override_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_rearm_temporal_closing_lattice_override_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_retreat_post_intent_all_hard_forward_filter_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_retreat_post_intent_all_hard_forward_filter_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_temporal_midpoint_retreat_reverse_filter_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_temporal_midpoint_retreat_reverse_filter_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_low_ttc_nonforward_coverage_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_low_ttc_nonforward_coverage_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_low_ttc_prefix_boundary_handoff_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_low_ttc_prefix_boundary_handoff_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_retreat_margin_m=float(
                values.get(
                    "probabilistic_obstacle_traversal_window_retreat_margin_m",
                    0.0,
                )
            ),
            probabilistic_obstacle_traversal_window_retreat_hard_risk_override_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_retreat_hard_risk_override_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_retreat_completion_frozen_frame_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_retreat_completion_frozen_frame_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_traversal_window_retreat_completion_frozen_reference_enabled=bool(
                values.get(
                    "probabilistic_obstacle_traversal_window_retreat_completion_frozen_reference_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_speed_governor_enabled=bool(
                values.get(
                    "probabilistic_obstacle_speed_governor_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_speed_governor_start_ratio=float(
                values.get(
                    "probabilistic_obstacle_speed_governor_start_ratio",
                    0.50,
                )
            ),
            probabilistic_obstacle_speed_governor_applies_during_active_avoidance=bool(
                values.get(
                    "probabilistic_obstacle_speed_governor_applies_during_active_avoidance",
                    False,
                )
            ),
            importance_sampling_correction=bool(
                values.get("importance_sampling_correction", False)
            ),
            previous_sequence_blend=float(values.get("previous_sequence_blend", 0.5)),
            safety_recovery_prefix_steps=int(
                values.get("safety_recovery_prefix_steps", 0)
            ),
            safety_recovery_translation_stop_threshold=float(
                values.get(
                    "safety_recovery_translation_stop_threshold", 1.0e-6
                )
            ),
            probabilistic_obstacle_emergency_min_support_beams=int(
                values.get("probabilistic_obstacle_emergency_min_support_beams", 0)
            ),
            probabilistic_obstacle_emergency_max_rejected_jump_fraction=float(
                values.get(
                    "probabilistic_obstacle_emergency_max_rejected_jump_fraction",
                    1.0,
                )
            ),
            probabilistic_obstacle_emergency_require_scan_quality=bool(
                values.get(
                    "probabilistic_obstacle_emergency_require_scan_quality",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_candidate_slew_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_slew_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_emergency_candidate_direct_fallback_enabled=bool(
                values.get(
                    "probabilistic_obstacle_emergency_candidate_direct_fallback_enabled",
                    True,
                )
            ),
            probabilistic_obstacle_reverse_candidate_filter_enabled=bool(
                values.get(
                    "probabilistic_obstacle_reverse_candidate_filter_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_forward_lateral_countermotion_enabled=bool(
                values.get(
                    "probabilistic_obstacle_forward_lateral_countermotion_enabled",
                    False,
                )
            ),
            probabilistic_obstacle_forward_lateral_countermotion_weight=float(
                values.get(
                    "probabilistic_obstacle_forward_lateral_countermotion_weight",
                    1.20,
                )
            ),
            probabilistic_obstacle_forward_lateral_minimum_speed_mps=float(
                values.get(
                    "probabilistic_obstacle_forward_lateral_minimum_speed_mps",
                    0.10,
                )
            ),
            probabilistic_obstacle_forward_lateral_minimum_fraction=float(
                values.get(
                    "probabilistic_obstacle_forward_lateral_minimum_fraction",
                    0.25,
                )
            ),
            probabilistic_obstacle_forward_lateral_reversal_confirm_steps=int(
                values.get(
                    "probabilistic_obstacle_forward_lateral_reversal_confirm_steps",
                    2,
                )
            ),
            profile_components=bool(
                values.get("profile_components", False)
            ),
            optimizer_diagnostics_enabled=bool(
                values.get("optimizer_diagnostics_enabled", False)
            ),
            seed=int(values.get("seed", 0)),
        )

    def validate(self, action_dim: int) -> None:
        if self.horizon <= 0 or self.num_samples <= 0 or self.dt <= 0.0:
            raise ValueError("MPPI horizon, samples, and dt must be positive")
        if (
            not np.isfinite(self.command_delay_s)
            or self.command_delay_s < 0.0
            or self.command_delay_s > self.dt + 1e-12
        ):
            raise ValueError(
                "command_delay_s must be finite and within one control interval"
            )
        if self.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive")
        if len(self.noise_sigma) != action_dim or any(value <= 0.0 for value in self.noise_sigma):
            raise ValueError("noise_sigma must contain one positive value per action dimension")
        if self.noise_basis != "iid":
            # Fail loudly at validation rather than silently sampling i.i.d.
            # if a protocol carries a malformed basis spec.
            from mobile_robot_mppi.sampling.bases import build_basis

            build_basis(self.noise_basis, self.dt)
        numeric_costs = (
            self.goal_running_weight,
            self.goal_terminal_weight,
            self.heading_weight,
            self.path_preview_heading_weight,
            self.path_boundary_buffer,
            self.path_boundary_weight,
            self.path_boundary_violation_penalty,
            self.terminal_velocity_weight,
            self.terminal_yaw_rate_weight,
            self.terminal_bearing_weight,
            self.control_weight,
            self.control_rate_weight,
            self.obstacle_weight,
            self.obstacle_influence,
            self.robot_radius,
            self.collision_penalty,
            self.known_static_map_influence_m,
            self.known_static_map_weight,
            self.known_static_map_collision_penalty,
            self.probabilistic_reference_authority_filter_alpha,
            self.probabilistic_reference_authority_minimum,
            self.probabilistic_reference_progress_weight,
            self.static_astar_replan_resolution_m,
            self.static_astar_replan_clearance_margin_m,
            self.static_astar_replan_deviation_m,
            self.static_astar_replan_minimum_progress_m,
            self.static_astar_replan_bounds_padding_m,
            self.static_astar_replan_sharp_corner_minimum_turn_rad,
            self.probabilistic_obstacle_risk_weight,
            self.probabilistic_obstacle_hard_threshold,
            self.probabilistic_obstacle_hard_penalty,
            self.probabilistic_obstacle_safety_margin,
            self.probabilistic_obstacle_minimum_std,
        )
        if not np.isfinite(numeric_costs).all() or any(value < 0.0 for value in numeric_costs):
            raise ValueError("MPPI cost and geometry parameters must be finite and non-negative")
        if self.known_static_map_cost_enabled and (
            self.known_static_map_weight <= 0.0
            or self.known_static_map_collision_penalty <= 0.0
        ):
            raise ValueError(
                "known static-map cost requires positive weight and "
                "collision penalty"
            )
        if (
            self.known_static_map_candidate_filter_enabled
            and not self.known_static_map_cost_enabled
        ):
            raise ValueError(
                "known static-map candidate filtering requires the exact "
                "static-map cost"
            )
        if not 0.0 <= self.probabilistic_reference_authority_filter_alpha < 1.0:
            raise ValueError(
                "reference-authority filter alpha must be in [0,1)"
            )
        if not 0.0 <= self.probabilistic_reference_authority_minimum <= 1.0:
            raise ValueError(
                "minimum reference authority must be within [0,1]"
            )
        if self.probabilistic_reference_authority_risk_source not in (
            "maximum_step_probability",
            "accumulated_probability_mass",
        ):
            raise ValueError(
                "reference-authority risk source must be maximum-step "
                "probability or accumulated probability mass"
            )
        if (
            self.static_astar_replan_stagnation_steps < 0
            or self.static_astar_replan_cooldown_steps < 0
        ):
            raise ValueError(
                "static A* replan step counts must be non-negative"
            )
        if self.static_astar_replan_enabled and (
            self.static_astar_replan_resolution_m <= 0.0
            or self.static_astar_replan_deviation_m <= 0.0
            or self.static_astar_replan_minimum_progress_m <= 0.0
            or self.static_astar_replan_bounds_padding_m <= 0.0
        ):
            raise ValueError(
                "enabled static A* replanning requires positive geometry "
                "and progress parameters"
            )
        if not (
            0.0
            < self.static_astar_replan_sharp_corner_minimum_turn_rad
            <= np.pi
        ):
            raise ValueError(
                "static A* sharp-corner turn threshold must be in (0, pi]"
            )
        if (
            self.probabilistic_reference_authority_enabled
            and not self.probabilistic_obstacle_risk_enabled
        ):
            raise ValueError(
                "probabilistic reference authority requires probabilistic "
                "obstacle risk"
            )
        if (
            self.probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_release_enabled
            and (
                not self.probabilistic_obstacle_traversal_window_rearm_no_crossing_clear_enabled
                or not self.probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_enabled
            )
        ):
            raise ValueError(
                "certified rearm-handoff release requires no-crossing "
                "clearance and certified handoff"
            )
        if (
            not np.isfinite(self.path_preview_speed_mps)
            or self.path_preview_speed_mps <= 0.0
        ):
            raise ValueError("path_preview_speed_mps must be finite and positive")
        if self.path_boundary_enabled and (
            not self.path_preview_enabled
            or self.path_boundary_violation_penalty <= 0.0
        ):
            raise ValueError(
                "path boundary enforcement requires path preview and a positive violation penalty"
            )
        if (
            self.path_boundary_candidate_filter_enabled
            and not self.path_boundary_enabled
        ):
            raise ValueError(
                "path boundary candidate filtering requires path boundary enforcement"
            )
        if self.integrator not in ("euler", "rk4"):
            raise ValueError("MPPI integrator must be 'euler' or 'rk4'")
        if not 0.0 < self.probabilistic_obstacle_hard_threshold <= 1.0:
            raise ValueError(
                "probabilistic obstacle hard threshold must be in (0,1]"
            )
        if self.probabilistic_obstacle_minimum_std <= 0.0:
            raise ValueError(
                "probabilistic obstacle minimum std must be positive"
            )
        if (
            self.probabilistic_obstacle_risk_cuda_enabled
            and not self.probabilistic_obstacle_risk_cuda_device.startswith("cuda")
        ):
            raise ValueError(
                "CUDA probabilistic obstacle risk requires a CUDA device"
            )
        if (
            self.probabilistic_obstacle_risk_enabled
            and self.probabilistic_obstacle_risk_weight <= 0.0
            and self.probabilistic_obstacle_hard_penalty <= 0.0
        ):
            raise ValueError(
                "enabled probabilistic obstacle risk requires a positive cost"
            )
        if not self.probabilistic_obstacle_forecast_key:
            raise ValueError(
                "probabilistic obstacle forecast key must be nonempty"
            )
        if self.probabilistic_obstacle_missing_forecast_action not in (
            "raise",
            "stop",
            "scan_only",
        ):
            raise ValueError(
                "probabilistic obstacle missing-forecast action must be "
                "'raise', 'stop' or 'scan_only'"
            )
        if self.probabilistic_obstacle_hard_violation_action not in (
            "penalize",
            "stop",
            "active_avoidance",
            "active_avoidance_motion",
        ):
            raise ValueError(
                "probabilistic obstacle hard-violation action must be "
                "'penalize', 'stop', 'active_avoidance', or "
                "'active_avoidance_motion'"
            )
        if (
            not np.isfinite(
                self.probabilistic_obstacle_hard_violation_progress_mass_tolerance
            )
            or not 0.0
            <= self.probabilistic_obstacle_hard_violation_progress_mass_tolerance
            <= 1.0
        ):
            raise ValueError(
                "hard-violation progress mass tolerance must be in [0,1]"
            )
        if not (
            0.0
            < self.probabilistic_obstacle_speed_governor_start_ratio
            < 1.0
        ):
            raise ValueError(
                "probabilistic obstacle speed-governor start ratio must "
                "be in (0,1)"
            )
        if not 0.0 <= self.previous_sequence_blend <= 1.0:
            raise ValueError("previous_sequence_blend must be between zero and one")
        if self.terminal_translation_speed_limit is not None and (
            not np.isfinite(self.terminal_translation_speed_limit)
            or self.terminal_translation_speed_limit <= 0.0
        ):
            raise ValueError(
                "terminal_translation_speed_limit must be finite and positive"
            )
        if self.terminal_translation_heading_gate_rad is not None and (
            not np.isfinite(self.terminal_translation_heading_gate_rad)
            or not 0.0 < self.terminal_translation_heading_gate_rad <= np.pi
        ):
            raise ValueError(
                "terminal_translation_heading_gate_rad must be in (0, pi]"
            )
        if self.terminal_alignment_yaw_gain is not None and (
            not np.isfinite(self.terminal_alignment_yaw_gain)
            or self.terminal_alignment_yaw_gain <= 0.0
        ):
            raise ValueError(
                "terminal_alignment_yaw_gain must be finite and positive"
            )
        if self.terminal_control_radius is not None and (
            not np.isfinite(self.terminal_control_radius)
            or self.terminal_control_radius <= 0.0
        ):
            raise ValueError(
                "terminal_control_radius must be finite and positive"
            )
        if (
            not np.isfinite(self.terminal_translation_minimum_scale)
            or not 0.0 <= self.terminal_translation_minimum_scale <= 1.0
        ):
            raise ValueError(
                "terminal_translation_minimum_scale must be in [0, 1]"
            )
        if (
            not np.isfinite(
                self.static_astar_replan_sharp_corner_minimum_lookahead_m
            )
            or self.static_astar_replan_sharp_corner_minimum_lookahead_m < 0.0
        ):
            raise ValueError(
                "static_astar_replan_sharp_corner_minimum_lookahead_m must be "
                "finite and non-negative"
            )
        if self.static_astar_replan_sharp_corner_deferred_clamp_distance_m is not None and (
            not np.isfinite(
                self.static_astar_replan_sharp_corner_deferred_clamp_distance_m
            )
            or self.static_astar_replan_sharp_corner_deferred_clamp_distance_m
            <= 0.0
        ):
            raise ValueError(
                "static_astar_replan_sharp_corner_deferred_clamp_distance_m "
                "must be None or finite and positive"
            )
        if not 0 <= self.safety_recovery_prefix_steps <= self.horizon:
            raise ValueError("safety_recovery_prefix_steps must be within the horizon")
        if (
            not np.isfinite(
                self.safety_recovery_translation_stop_threshold
            )
            or self.safety_recovery_translation_stop_threshold < 0.0
        ):
            raise ValueError(
                "safety recovery translation stop threshold must be finite "
                "and non-negative"
            )
        if (
            self.probabilistic_obstacle_emergency_candidates_enabled
            and not (
                1
                <= self.probabilistic_obstacle_emergency_candidate_prefix_steps
                <= self.horizon
            )
        ):
            raise ValueError(
                "probabilistic emergency candidate prefix must be within the horizon"
            )
        if (
            self.probabilistic_obstacle_emergency_candidates_enabled
            and self.num_samples < 8
        ):
            raise ValueError(
                "probabilistic emergency candidates require at least 8 samples"
            )
        if (
            not np.isfinite(
                self.probabilistic_obstacle_emergency_candidate_trigger_ttc_s
            )
            or self.probabilistic_obstacle_emergency_candidate_trigger_ttc_s
            < 0.0
        ):
            raise ValueError(
                "probabilistic emergency candidate trigger TTC must be "
                "finite and non-negative"
            )
        trigger_distance = (
            self.probabilistic_obstacle_emergency_candidate_trigger_distance_m
        )
        critical_distance = (
            self.probabilistic_obstacle_emergency_candidate_critical_distance_m
        )
        if (
            not np.isfinite(trigger_distance)
            or trigger_distance < 0.0
            or not np.isfinite(critical_distance)
            or critical_distance < 0.0
            or critical_distance > trigger_distance
        ):
            raise ValueError(
                "probabilistic emergency candidate distances must be finite "
                "and satisfy 0 <= critical <= trigger"
            )
        if (
            not np.isfinite(self.probabilistic_obstacle_counterflow_weight)
            or self.probabilistic_obstacle_counterflow_weight < 0.0
        ):
            raise ValueError(
                "probabilistic counterflow weight must be finite and "
                "non-negative"
            )
        if self.probabilistic_obstacle_emergency_min_support_beams < 0:
            raise ValueError(
                "probabilistic emergency minimum support must be non-negative"
            )
        if not 0.0 <= self.probabilistic_obstacle_emergency_max_rejected_jump_fraction <= 1.0:
            raise ValueError(
                "probabilistic emergency rejected-jump ceiling must be in [0, 1]"
            )
        if (
            not np.isfinite(
                self.probabilistic_obstacle_forward_lateral_countermotion_weight
            )
            or self.probabilistic_obstacle_forward_lateral_countermotion_weight
            <= 0.0
            or not np.isfinite(
                self.probabilistic_obstacle_forward_lateral_minimum_speed_mps
            )
            or self.probabilistic_obstacle_forward_lateral_minimum_speed_mps
            < 0.0
            or not 0.0
            <= self.probabilistic_obstacle_forward_lateral_minimum_fraction
            <= 1.0
            or self
            .probabilistic_obstacle_forward_lateral_reversal_confirm_steps
            < 1
        ):
            raise ValueError(
                "probabilistic forward-lateral countermotion parameters are "
                "invalid"
            )
        if self.probabilistic_obstacle_emergency_candidate_intent_hold_steps < 0:
            raise ValueError(
                "probabilistic emergency candidate intent hold must be "
                "non-negative"
            )
        forward_risk_ceiling = (
            self.probabilistic_obstacle_emergency_candidate_forward_risk_ceiling
        )
        forward_mass_ceiling = (
            self.probabilistic_obstacle_emergency_candidate_forward_mass_ceiling
        )
        if (
            not np.isfinite(forward_risk_ceiling)
            or not 0.0 <= forward_risk_ceiling <= 1.0
            or not np.isfinite(forward_mass_ceiling)
            or forward_mass_ceiling < 0.0
        ):
            raise ValueError(
                'probabilistic emergency forward risk ceilings must be '
                'finite and non-negative'
            )
        if (forward_risk_ceiling > 0.0) != (forward_mass_ceiling > 0.0):
            raise ValueError(
                'probabilistic emergency forward risk and mass ceilings '
                'must be enabled together'
            )
        rearm_ttc = (
            self.probabilistic_obstacle_emergency_candidate_rearm_ttc_s
        )
        if not np.isfinite(rearm_ttc) or rearm_ttc < 0.0:
            raise ValueError(
                "probabilistic emergency candidate rearm TTC must be "
                "finite and non-negative"
            )
        if (
            rearm_ttc > 0.0
            and rearm_ttc
            < self.probabilistic_obstacle_emergency_candidate_trigger_ttc_s
        ):
            raise ValueError(
                "probabilistic emergency candidate rearm TTC must not be "
                "smaller than the trigger TTC"
            )
        if (
            self.probabilistic_obstacle_emergency_candidate_rearm_clear_steps
            < 1
        ):
            raise ValueError(
                "probabilistic emergency candidate rearm clear steps must be "
                "positive"
            )
        if (
            self.probabilistic_obstacle_terminal_intent_safe_stop_enabled
            and not (
                self.probabilistic_obstacle_risk_enabled
                and self.probabilistic_obstacle_emergency_candidates_enabled
                and self.probabilistic_obstacle_stopping_feasibility_enabled
                and self
                .probabilistic_obstacle_emergency_forecast_corroboration_enabled
            )
        ):
            raise ValueError(
                "terminal intent safe stop requires probabilistic risk, "
                "emergency candidates, stopping feasibility, and forecast "
                "corroboration"
            )
        if (
            self
            .probabilistic_obstacle_traversal_window_terminal_target_bearing_enabled
            and not self.probabilistic_obstacle_traversal_window_enabled
        ):
            raise ValueError(
                "terminal traversal target-bearing steering requires the "
                "traversal window"
            )
        if (
            self.probabilistic_obstacle_terminal_capture_candidate_enabled
            and not (
                self.probabilistic_obstacle_risk_enabled
                and self.probabilistic_obstacle_traversal_window_enabled
            )
        ):
            raise ValueError(
                "terminal capture candidate requires probabilistic risk "
                "and the traversal window"
            )
        traversal_values = (
            self.probabilistic_obstacle_traversal_window_activation_distance_m,
            self.probabilistic_obstacle_traversal_window_cross_track_m,
            self.probabilistic_obstacle_traversal_window_clearance_margin_m,
            self.probabilistic_obstacle_traversal_window_probability_ceiling,
            self.probabilistic_obstacle_traversal_window_mass_ceiling,
            self.probabilistic_obstacle_traversal_window_translation_heading_gate_rad,
            self.probabilistic_obstacle_traversal_window_abort_probability,
            self.probabilistic_obstacle_traversal_window_temporal_abort_mass_floor,
            self.probabilistic_obstacle_traversal_window_retreat_margin_m,
        )
        if (
            not np.isfinite(traversal_values).all()
            or any(value < 0.0 for value in traversal_values)
            or self.probabilistic_obstacle_traversal_window_horizon_steps
            < self.horizon
            or self.probabilistic_obstacle_traversal_window_clear_hold_steps < 0
            or self.probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps
            < 1
            or self.probabilistic_obstacle_traversal_window_probability_ceiling
            > 1.0
            or self.probabilistic_obstacle_traversal_window_translation_heading_gate_rad
            > 0.5 * np.pi
            or self.probabilistic_obstacle_traversal_window_abort_probability
            > 1.0
        ):
            raise ValueError("probabilistic traversal-window settings are invalid")
        if self.probabilistic_obstacle_traversal_window_enabled and (
            not self.probabilistic_obstacle_risk_enabled
            or not self.probabilistic_obstacle_emergency_candidates_enabled
            or self.probabilistic_obstacle_traversal_window_activation_distance_m
            <= 0.0
            or self.probabilistic_obstacle_traversal_window_cross_track_m <= 0.0
            or self.probabilistic_obstacle_traversal_window_probability_ceiling
            <= 0.0
            or self.probabilistic_obstacle_traversal_window_mass_ceiling <= 0.0
        ):
            raise ValueError(
                "enabled traversal-window certification requires probability "
                "risk, emergency candidates, and positive bounds"
            )


@dataclass(frozen=True)
class SequenceEvaluation:
    """Predicted trajectories and base costs for fixed control sequences.

    This public result contract is deliberately independent of MPPI sampling
    weights.  It supports paired model diagnostics where nominal, learned and
    physical rollouts must score the exact same commanded sequences.
    """

    trajectories: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class SequenceWeights:
    """Importance-adjusted costs and normalized MPPI sequence weights."""

    adjusted_costs: np.ndarray
    weights: np.ndarray
    effective_sample_size: float


class MppiController:
    def __init__(
        self,
        dynamics,
        state_spec: StateSpec,
        action_spec: ActionSpec,
        config: MppiConfig,
        sampling_prior=None,
        memory_cost: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        config.validate(action_spec.dimension)
        if int(dynamics.state_dim) != state_spec.dimension:
            raise ValueError("prediction dynamics and state spec dimensions differ")
        if int(dynamics.control_dim) != action_spec.dimension:
            raise ValueError("prediction dynamics and action spec dimensions differ")
        self.dynamics = dynamics
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.config = config
        self.sampling_prior = sampling_prior or GoalWarmStartPrior()
        self.memory_cost = memory_cost
        self.rng = np.random.RandomState(config.seed)
        self.previous_sequence = None
        self.previous_action = np.zeros(action_spec.dimension, dtype=np.float64)
        self._safety_blocked = False
        self._reliability_previous_state = None
        self._reliability_pending_control = None
        self._delay_preceding_action = self.previous_action.copy()
        self._previous_probabilistic_risk = 1.0
        # Read-only audit state for the dense forecast-risk backend.  This is
        # deliberately separate from the risk result contract so CPU and CUDA
        # evaluations remain numerically/public-API compatible.
        self._last_probabilistic_risk_backend = "disabled"
        self._probabilistic_reference_risk_filtered = 0.0
        self._static_astar_progress_anchor = None
        self._static_astar_stagnation_steps = 0
        self._static_astar_cooldown_steps = 0
        self._static_astar_replan_count = 0
        self._static_astar_static_hard_stop_latched = False
        self._static_astar_last_diagnostics = {
            "enabled": bool(config.static_astar_replan_enabled),
            "triggered": False,
            "reason": "none",
            "route_points": 0,
            "cross_track_m": 0.0,
            "stagnation_steps": 0,
        }
        self._probabilistic_emergency_intent_remaining = 0
        self._probabilistic_emergency_direction = None
        self._probabilistic_emergency_pending_direction = None
        self._probabilistic_emergency_pending_direction_count = 0
        self._probabilistic_emergency_latched_pattern = None
        self._probabilistic_emergency_latched_heading = None
        self._probabilistic_emergency_rearm_ready = True
        self._probabilistic_emergency_rearm_clear_count = 0
        self._probabilistic_traversal_crossing_progress = None
        self._probabilistic_traversal_entry_progress = None
        self._probabilistic_traversal_clear_progress = None
        self._probabilistic_traversal_commit_started = False
        self._probabilistic_traversal_retreat_progress = None
        self._probabilistic_traversal_uncommitted_hold_steps = 0
        self._probabilistic_traversal_retreat_target_position = None
        self._probabilistic_traversal_retreat_target_tangent = None
        self._probabilistic_traversal_retreat_reference = None
        self._probabilistic_traversal_rearm_pending = False
        self._probabilistic_traversal_retreat_temporal_lattice = False
        self._probabilistic_traversal_admission_safe_streak = 0
        self._probabilistic_traversal_admission_signature = None
        self._probabilistic_traversal_exit_deadline_retreat_active = False
        self._probabilistic_traversal_exit_deadline_retreat_pattern = None
        self._probabilistic_traversal_exit_deadline_retreat_heading = None
        self._probabilistic_traversal_post_center_temporal_pattern = None
        self._probabilistic_traversal_post_center_temporal_heading = None

    def reset(self, seed=None):
        """Reset receding-horizon state and optionally reseed sampling.

        The optional episode seed is essential for paired experiments: plant,
        sensors and MPPI perturbations must all belong to the same replicate.
        Existing runtime callers that omit it preserve the configured seed.
        """

        resolved_seed = self.config.seed if seed is None else int(seed)
        if not 0 <= resolved_seed <= 2 ** 32 - 1:
            raise ValueError("MPPI reset seed must be in [0, 2**32 - 1]")
        self.rng = np.random.RandomState(resolved_seed)
        self.previous_sequence = None
        self.previous_action = np.zeros(self.action_spec.dimension, dtype=np.float64)
        self._safety_blocked = False
        self._reliability_previous_state = None
        self._reliability_pending_control = None
        self._delay_preceding_action = self.previous_action.copy()
        self._previous_probabilistic_risk = 1.0
        self._probabilistic_reference_risk_filtered = 0.0
        self._static_astar_progress_anchor = None
        self._static_astar_stagnation_steps = 0
        self._static_astar_cooldown_steps = 0
        self._static_astar_replan_count = 0
        self._static_astar_static_hard_stop_latched = False
        self._static_astar_last_diagnostics = {
            "enabled": bool(self.config.static_astar_replan_enabled),
            "triggered": False,
            "reason": "none",
            "route_points": 0,
            "cross_track_m": 0.0,
            "stagnation_steps": 0,
        }
        self._probabilistic_emergency_intent_remaining = 0
        self._probabilistic_emergency_direction = None
        self._probabilistic_emergency_pending_direction = None
        self._probabilistic_emergency_pending_direction_count = 0
        self._probabilistic_emergency_latched_pattern = None
        self._probabilistic_emergency_latched_heading = None
        self._probabilistic_emergency_rearm_ready = True
        self._probabilistic_emergency_rearm_clear_count = 0
        self._probabilistic_traversal_crossing_progress = None
        self._probabilistic_traversal_entry_progress = None
        self._probabilistic_traversal_clear_progress = None
        self._probabilistic_traversal_commit_started = False
        self._probabilistic_traversal_retreat_progress = None
        self._probabilistic_traversal_uncommitted_hold_steps = 0
        self._probabilistic_traversal_retreat_target_position = None
        self._probabilistic_traversal_retreat_target_tangent = None
        self._probabilistic_traversal_retreat_reference = None
        self._probabilistic_traversal_rearm_pending = False
        self._probabilistic_traversal_retreat_temporal_lattice = False
        self._probabilistic_traversal_admission_safe_streak = 0
        self._probabilistic_traversal_admission_signature = None
        self._probabilistic_traversal_exit_deadline_retreat_active = False
        self._probabilistic_traversal_exit_deadline_retreat_pattern = None
        self._probabilistic_traversal_exit_deadline_retreat_heading = None
        self._probabilistic_traversal_post_center_temporal_pattern = None
        self._probabilistic_traversal_post_center_temporal_heading = None
        residual_reset = getattr(
            getattr(self.dynamics, "residual", None), "reset", None
        )
        if callable(residual_reset):
            residual_reset()
        reset = getattr(self.sampling_prior, "reset", None)
        if callable(reset):
            reset()

    def state_from_observation(self, observation: RobotObservation) -> np.ndarray:
        values = {
            "x": observation.pose.x,
            "y": observation.pose.y,
            "theta": observation.pose.theta,
            "v": observation.twist.v,
            "omega": observation.twist.omega,
        }
        values.update(observation.auxiliary)
        try:
            state = np.asarray([values[name] for name in self.state_spec.names], dtype=np.float64)
        except KeyError as exc:
            raise ValueError("observation does not provide state channel %s" % exc)
        return self.state_spec.validate(state)

    def _maybe_replan_static_reference(
        self, state, observation, reference
    ) -> Dict[str, Any]:
        """Refresh only the static A* reference after deviation/stagnation."""

        diagnostics = {
            "enabled": bool(self.config.static_astar_replan_enabled),
            "triggered": False,
            "reason": "none",
            "route_points": 0,
            "cross_track_m": 0.0,
            "stagnation_steps": int(
                self._static_astar_stagnation_steps
            ),
            "count": int(self._static_astar_replan_count),
        }
        required = (
            "project",
            "replace_points",
            "points",
            "progress",
        )
        if (
            not self.config.static_astar_replan_enabled
            or not all(hasattr(reference, name) for name in required)
        ):
            self._static_astar_last_diagnostics = diagnostics
            return diagnostics
        position = np.asarray(state, dtype=np.float64)[
            list(self.state_spec.position_indices)
        ]
        projection = reference.project(
            position, minimum_progress=float(reference.progress)
        )
        progress = float(projection.progress)
        diagnostics["cross_track_m"] = float(
            projection.cross_track_error
        )
        if self._static_astar_progress_anchor is None:
            self._static_astar_progress_anchor = progress
        if (
            progress - float(self._static_astar_progress_anchor)
            >= self.config.static_astar_replan_minimum_progress_m
        ):
            self._static_astar_progress_anchor = progress
            self._static_astar_stagnation_steps = 0
        else:
            self._static_astar_stagnation_steps += 1
        diagnostics["stagnation_steps"] = int(
            self._static_astar_stagnation_steps
        )
        static_hard_stop = bool(
            observation.auxiliary.get(
                "static_near_body_hard_stop", False
            )
        )
        if not static_hard_stop:
            self._static_astar_static_hard_stop_latched = False
        if self._static_astar_cooldown_steps > 0:
            self._static_astar_cooldown_steps -= 1
            diagnostics["reason"] = "cooldown"
            self._static_astar_last_diagnostics = diagnostics
            return diagnostics

        reason = None
        if (
            projection.cross_track_error
            > self.config.static_astar_replan_deviation_m
        ):
            reason = "deviation"
        elif (
            self.config.static_astar_replan_static_hard_stop_enabled
            and static_hard_stop
            and not self._static_astar_static_hard_stop_latched
        ):
            reason = "static_hard_stop"
        elif (
            self.config.static_astar_replan_stagnation_steps > 0
            and self._static_astar_stagnation_steps
            >= self.config.static_astar_replan_stagnation_steps
        ):
            reason = "stagnation"
        if reason is None:
            self._static_astar_last_diagnostics = diagnostics
            return diagnostics

        static_obstacles = tuple(
            observation.auxiliary.get("known_static_obstacles", ())
        )
        if not static_obstacles:
            diagnostics["reason"] = "missing_static_map"
            self._static_astar_last_diagnostics = diagnostics
            return diagnostics
        goal = np.asarray(reference.points[-1], dtype=np.float64)
        if float(np.linalg.norm(goal - position)) <= max(
            2.0 * float(getattr(reference, "tolerance", 0.2)),
            self.config.static_astar_replan_resolution_m,
        ):
            diagnostics["reason"] = "near_goal"
            self._static_astar_last_diagnostics = diagnostics
            return diagnostics
        try:
            route = plan_static_astar_path(
                position,
                goal,
                static_obstacles,
                robot_radius=self.config.robot_radius,
                clearance_margin=(
                    self.config.static_astar_replan_clearance_margin_m
                ),
                resolution=self.config.static_astar_replan_resolution_m,
                bounds_padding=(
                    self.config.static_astar_replan_bounds_padding_m
                ),
            )
            reference.replace_points(route)
            reference.corner_clamped_lookahead_enabled = bool(
                self.config
                .static_astar_replan_corner_clamped_lookahead_enabled
            )
            reference.sharp_corner_clamped_lookahead_enabled = bool(
                self.config
                .static_astar_replan_sharp_corner_clamped_lookahead_enabled
            )
            reference.sharp_corner_minimum_turn_rad = float(
                self.config
                .static_astar_replan_sharp_corner_minimum_turn_rad
            )
            reference.sharp_corner_minimum_lookahead_m = float(
                self.config
                .static_astar_replan_sharp_corner_minimum_lookahead_m
            )
            reference.sharp_corner_deferred_clamp_distance_m = (
                self.config
                .static_astar_replan_sharp_corner_deferred_clamp_distance_m
            )
        except (RuntimeError, ValueError) as exc:
            if reason == "static_hard_stop":
                self._static_astar_static_hard_stop_latched = True
            diagnostics["reason"] = "failed:%s" % str(exc)
            self._static_astar_cooldown_steps = (
                self.config.static_astar_replan_cooldown_steps
            )
            self._static_astar_last_diagnostics = diagnostics
            return diagnostics
        if reason == "static_hard_stop":
            self._static_astar_static_hard_stop_latched = True
        self._static_astar_progress_anchor = 0.0
        self._static_astar_stagnation_steps = 0
        self._static_astar_cooldown_steps = (
            self.config.static_astar_replan_cooldown_steps
        )
        self._static_astar_replan_count += 1
        diagnostics.update({
            "triggered": True,
            "reason": reason,
            "route_points": int(len(route)),
            "stagnation_steps": 0,
            "count": int(self._static_astar_replan_count),
        })
        self._static_astar_last_diagnostics = diagnostics
        return diagnostics

    def _probabilistic_reference_authority(
        self, candidate_risk, prior_index
    ) -> Tuple[float, float]:
        if (
            not self.config.probabilistic_reference_authority_enabled
            or candidate_risk is None
        ):
            self._probabilistic_reference_risk_filtered = 0.0
            return 1.0, 0.0
        index = int(np.clip(
            prior_index,
            0,
            len(candidate_risk.maximum_step_probability) - 1,
        ))
        if (
            self.config.probabilistic_reference_authority_risk_source
            == "accumulated_probability_mass"
        ):
            raw = float(
                candidate_risk.accumulated_probability_mass[index]
            )
        else:
            raw = float(
                candidate_risk.maximum_step_probability[index]
            )
        raw = float(np.clip(raw, 0.0, 1.0))
        alpha = (
            self.config
            .probabilistic_reference_authority_filter_alpha
        )
        self._probabilistic_reference_risk_filtered = float(
            alpha * self._probabilistic_reference_risk_filtered
            + (1.0 - alpha) * raw
        )
        authority = max(
            self.config.probabilistic_reference_authority_minimum,
            1.0 - self._probabilistic_reference_risk_filtered,
        )
        return float(authority), raw

    def _prior(self, observation, reference) -> PriorOutput:
        output = self.sampling_prior.propose(
            observation, reference, self.config.horizon, self.action_spec
        )
        mean = np.asarray(output.mean, dtype=np.float64)
        expected = (self.config.horizon, self.action_spec.dimension)
        if mean.shape != expected or not np.isfinite(mean).all():
            raise ValueError("sampling prior returned an invalid mean")
        metadata = dict(output.metadata)
        if self.previous_sequence is not None:
            shifted = np.empty_like(self.previous_sequence)
            shifted[:-1] = self.previous_sequence[1:]
            shifted[-1] = self.previous_sequence[-1]
            blend = self.config.previous_sequence_blend
            mean = (1.0 - blend) * mean + blend * shifted
        if (
            self._safety_blocked
            and self.config.safety_recovery_prefix_steps > 0
            and "v_cmd" in self.action_spec.names
        ):
            v_index = self.action_spec.index("v_cmd")
            mean[:self.config.safety_recovery_prefix_steps, v_index] = 0.0
            metadata["safety_recovery"] = True
            metadata["safety_recovery_prefix_steps"] = int(
                self.config.safety_recovery_prefix_steps
            )
        else:
            metadata["safety_recovery"] = False
        return PriorOutput(
            np.clip(mean, self.action_spec.lower, self.action_spec.upper),
            output.covariance,
            metadata,
            output.proposals,
        )

    def observe_safety_decision(self, decision) -> None:
        """Close the loop between MPPI's proposal and the command actually sent.

        A forward stop must not erase angular control.  When scan_guard blocks
        translation, the next proposal therefore starts with a zero-translation
        prefix while retaining the angular sequence, allowing an in-place turn.
        """
        executed = np.asarray(decision.executed_control.values, dtype=np.float64).reshape(-1)
        proposed = np.asarray(decision.proposed_control.values, dtype=np.float64).reshape(-1)
        expected = (self.action_spec.dimension,)
        if (
            executed.shape != expected
            or proposed.shape != expected
            or not np.isfinite(executed).all()
            or not np.isfinite(proposed).all()
        ):
            raise ValueError("safety feedback actions must be finite and match action dimension")
        delay_fraction = float(self.config.command_delay_s / self.config.dt)
        self._reliability_pending_control = (
            delay_fraction * self._delay_preceding_action
            + (1.0 - delay_fraction) * executed
        )
        self.previous_action = executed.copy()
        blocked = False
        if "v_cmd" in self.action_spec.names:
            v_index = self.action_spec.index("v_cmd")
            blocked = bool(
                decision.overridden
                and proposed[v_index] > executed[v_index] + 1e-12
                and abs(executed[v_index])
                <= self.config.safety_recovery_translation_stop_threshold
            )
            if blocked and self.previous_sequence is not None:
                prefix = max(1, self.config.safety_recovery_prefix_steps)
                self.previous_sequence[:prefix, v_index] = executed[v_index]
        self._safety_blocked = blocked
        prior_observer = getattr(self.sampling_prior, "observe_safety_decision", None)
        if callable(prior_observer):
            prior_observer(decision)

    def _observe_residual_reliability(self, current_state):
        residual = getattr(self.dynamics, "residual", None)
        ungated = getattr(residual, "ungated_derivative", None)
        nominal = getattr(self.dynamics, "nominal", None)
        if residual is None or nominal is None:
            return
        if (
            self._reliability_previous_state is None
            or self._reliability_pending_control is None
        ):
            return

        previous = self._reliability_previous_state
        control = self._reliability_pending_control
        self.observe_completed_transition(
            previous,
            control,
            current_state,
            residual_derivative=(ungated if callable(ungated) else None),
        )
        self._reliability_pending_control = None

    def _observe_residual_context(self, state, target):
        residual = getattr(self.dynamics, "residual", None)
        observer = getattr(residual, "observe_context", None)
        if not callable(observer):
            return
        observer(
            state,
            np.asarray((target.pose.x, target.pose.y), dtype=np.float64),
            self._previous_probabilistic_risk,
        )

    def observe_completed_transition(
        self,
        previous_state,
        applied_control,
        current_state,
        residual_derivative=None,
    ):
        """Update residual diagnostics from one already completed transition.

        This public hook is shared by online MPPI execution and offline direct
        Actor training.  It is intentionally called only after ``current_state``
        has been measured, so no future plant information can leak into the
        command that generated the transition.
        """

        residual = getattr(self.dynamics, "residual", None)
        nominal = getattr(self.dynamics, "nominal", None)
        if residual is None or nominal is None:
            return False
        return self._observe_residual_prediction_errors(
            previous_state,
            applied_control,
            current_state,
            residual=residual,
            nominal=nominal,
            residual_derivative=residual_derivative,
        )

    def _observe_residual_prediction_errors(
        self,
        previous_state,
        applied_control,
        current_state,
        *,
        residual,
        nominal,
        residual_derivative=None,
    ):
        """Update one explicit causal residual evidence source.

        The normal controller passes its prediction residual.  Controllers
        with a preregistered reliability sidecar may call the same helper for
        that independent source, keeping prediction dynamics and HSS evidence
        separate without changing either model's error definition.
        """

        observers = []
        seen = set()
        current = residual
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            observer = getattr(current, "observe_prediction_errors", None)
            if callable(observer):
                observers.append(observer)
            child = getattr(current, "residual", None)
            current = None if child is current else child
        if not observers:
            return False
        previous = self.state_spec.validate(previous_state)
        control = np.asarray(applied_control, dtype=np.float64).reshape(-1)
        if (
            control.shape != (self.action_spec.dimension,)
            or not np.isfinite(control).all()
        ):
            raise ValueError("completed-transition control is invalid")
        observed = self.state_spec.validate(current_state)
        derivative = residual_derivative
        if derivative is None:
            derivative = getattr(residual, "ungated_derivative", None)
        if not callable(derivative):
            derivative = residual.derivative

        class _ResidualCombined:
            state_dim = int(self.dynamics.state_dim)
            control_dim = int(self.dynamics.control_dim)

            def derivative(inner_self, state, action, time=None):
                return np.asarray(
                    nominal.derivative(state, action, time), dtype=np.float64
                ) + np.asarray(derivative(state, action, time), dtype=np.float64)

        nominal_prediction = integrate_batch(
            nominal, previous, control, self.config.dt,
            self.state_spec, self.config.integrator,
        )
        residual_prediction = integrate_batch(
            _ResidualCombined(), previous, control, self.config.dt,
            self.state_spec, self.config.integrator,
        )
        nominal_error = nominal_prediction - observed
        residual_error = residual_prediction - observed
        for index in self.state_spec.periodic_indices:
            nominal_error[index] = np.arctan2(
                np.sin(nominal_error[index]), np.cos(nominal_error[index])
            )
            residual_error[index] = np.arctan2(
                np.sin(residual_error[index]), np.cos(residual_error[index])
            )
        for observer in observers:
            observer(nominal_error.copy(), residual_error.copy())
        return True

    def _sample(self, prior: PriorOutput, rng=None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        shape = (self.config.num_samples, self.config.horizon, self.action_spec.dimension)
        if prior.covariance is None:
            if self.config.noise_basis == "iid":
                # Unchanged production path. Kept as a literal expression so the
                # default draws the identical RNG stream in the identical order.
                noise = rng.normal(size=shape) * np.asarray(self.config.noise_sigma)[None, None, :]
            else:
                # Imported locally so the default path does not change at all.
                from mobile_robot_mppi.sampling.bases import build_basis

                noise = build_basis(self.config.noise_basis, self.config.dt).sample(
                    rng,
                    self.config.num_samples,
                    self.config.horizon,
                    np.asarray(self.config.noise_sigma, dtype=np.float64),
                )
        else:
            covariance = np.asarray(prior.covariance, dtype=np.float64)
            if covariance.shape == (self.action_spec.dimension, self.action_spec.dimension):
                noise = rng.multivariate_normal(
                    np.zeros(self.action_spec.dimension), covariance,
                    size=(self.config.num_samples, self.config.horizon),
                )
            else:
                raise ValueError("prior covariance must have shape [action_dim, action_dim]")
        samples = prior.mean[None, :, :] + noise
        samples = np.clip(samples, self.action_spec.lower, self.action_spec.upper)
        samples[0] = prior.mean
        return samples

    def _sampling_covariance(self, prior: PriorOutput) -> np.ndarray:
        if prior.covariance is None:
            sigma = np.asarray(self.config.noise_sigma, dtype=np.float64)
            return np.diag(sigma ** 2)
        covariance = np.asarray(prior.covariance, dtype=np.float64)
        expected = (self.action_spec.dimension, self.action_spec.dimension)
        if covariance.shape != expected or not np.isfinite(covariance).all():
            raise ValueError("prior covariance must be a finite [action_dim, action_dim] matrix")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("prior covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("prior covariance must be positive definite") from exc
        return covariance

    def _rear_obstacle_blocks_reverse(self, state, forecasts):
        """True when a tracked obstacle sits behind the robot, within reach.

        "Behind" is measured against the robot's own heading, not against the
        escape direction, because the collisions happen when the controller
        realises a rearward world-frame direction by driving backwards into
        something it is not facing.

        Returns ``False`` when disabled, when heading is unavailable, or when no
        obstacle lies in the rear sector, so the default path is unchanged.
        """

        if not self.config.probabilistic_obstacle_reverse_rear_guard_enabled:
            return False
        if "theta" not in self.state_spec.names:
            return False
        radius = float(
            self.config.probabilistic_obstacle_reverse_rear_guard_radius_m
        )
        half_angle = float(
            self.config.probabilistic_obstacle_reverse_rear_guard_halfangle_deg
        )
        if not np.isfinite(radius) or radius <= 0.0:
            return False
        if not np.isfinite(half_angle) or half_angle <= 0.0:
            return False
        cos_limit = float(np.cos(np.radians(min(half_angle, 180.0))))
        state = np.asarray(state, dtype=np.float64)
        origin = state[list(self.state_spec.position_indices)][:2]
        theta = float(state[self.state_spec.index("theta")])
        # Unit vector pointing out of the robot's back.
        backward = np.asarray(
            (-np.cos(theta), -np.sin(theta)), dtype=np.float64
        )
        lookahead = int(
            self.config
            .probabilistic_obstacle_reverse_rear_guard_lookahead_steps
        )
        if lookahead < 0:
            lookahead = 0
        for forecast in forecasts:
            means = np.asarray(forecast.component_means, dtype=np.float64)
            weights = np.asarray(forecast.component_weights, dtype=np.float64)
            if means.ndim != 3 or means.shape[0] == 0:
                continue
            horizon = min(lookahead, means.shape[0] - 1)
            # Scan the present AND the forecast: the carrier that causes the
            # impact is usually still arriving when the reverse is commanded.
            for step in range(horizon + 1):
                predicted = np.sum(
                    weights[step][..., None] * means[step], axis=0
                )
                offset = predicted[:2] - origin
                distance = float(np.linalg.norm(offset))
                if not np.isfinite(distance) or distance > radius:
                    continue
                if distance <= 1.0e-9:
                    return True
                if float(np.dot(backward, offset / distance)) >= cos_limit:
                    return True
        return False

    def _escape_direction_enters_occupancy(
        self, robot_xy, direction, forecasts
    ):
        """True when moving along ``direction`` closes on an obstacle already near.

        The escape lattice steers the robot along a preferred direction without
        asking whether that direction is occupied.  Counterflow can rotate it
        behind the robot, and the measured consequence was six grazing
        collisions -- 0.003 to 0.027 m of penetration, four of them at negative
        velocity, all with the traversal window disengaged.

        An obstacle vetoes the direction when it is BOTH inside the guard radius
        AND ahead of the robot along that direction, i.e. the move would close on
        it.  An obstacle inside the radius but off to the side or behind the
        motion is left alone, so the guard suppresses only the reversals that
        actually run into something.

        Returns ``False`` unless the guard is enabled, so the default
        configuration is bit-identical to before.
        """

        if not (
            self.config
            .probabilistic_obstacle_escape_rear_occupancy_guard_enabled
        ):
            return False
        radius = float(
            self.config
            .probabilistic_obstacle_escape_rear_occupancy_guard_radius_m
        )
        if not np.isfinite(radius) or radius <= 0.0:
            return False
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1.0e-9:
            return False
        direction = direction / norm
        origin = np.asarray(robot_xy, dtype=np.float64)
        for forecast in forecasts:
            means = np.asarray(forecast.component_means, dtype=np.float64)
            weights = np.asarray(forecast.component_weights, dtype=np.float64)
            if means.ndim != 3 or means.shape[0] == 0:
                continue
            current = np.sum(weights[0][..., None] * means[0], axis=0)
            offset = current[:2] - origin[:2]
            distance = float(np.linalg.norm(offset))
            if not np.isfinite(distance) or distance > radius:
                continue
            if distance <= 1.0e-9:
                return True
            if float(np.dot(direction, offset / distance)) > 0.0:
                return True
        return False

    @staticmethod
    def _forward_lateral_countermotion_direction(
        obstacle_motion,
        theta,
        lateral_weight,
        minimum_lateral_speed_mps,
        minimum_lateral_fraction,
    ):
        """Return a forward-biased direction opposite lateral human motion."""

        motion = np.asarray(obstacle_motion, dtype=np.float64)
        speed = float(np.linalg.norm(motion))
        forward = np.asarray((np.cos(theta), np.sin(theta)), dtype=np.float64)
        left = np.asarray((-np.sin(theta), np.cos(theta)), dtype=np.float64)
        longitudinal_speed = float(np.dot(motion, forward))
        lateral_speed = float(np.dot(motion, left))
        lateral_fraction = abs(lateral_speed) / max(speed, 1.0e-9)
        if (
            not np.isfinite(motion).all()
            or speed <= 1.0e-9
            or abs(lateral_speed) < float(minimum_lateral_speed_mps)
            or lateral_fraction < float(minimum_lateral_fraction)
        ):
            return None, longitudinal_speed, lateral_speed, lateral_fraction
        direction = (
            forward
            - np.sign(lateral_speed) * float(lateral_weight) * left
        )
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        return direction, longitudinal_speed, lateral_speed, lateral_fraction

    def _probabilistic_emergency_context(
        self, observation, state=None, probabilistic_obstacles=()
    ):
        """Return causal scan evidence used to activate the escape lattice."""

        threshold = float(
            self.config
            .probabilistic_obstacle_emergency_candidate_trigger_ttc_s
        )
        trigger_distance = float(
            self.config
            .probabilistic_obstacle_emergency_candidate_trigger_distance_m
        )
        if observation is None or (
            threshold <= 0.0 and trigger_distance <= 0.0
        ):
            self._probabilistic_emergency_intent_remaining = 0
            self._probabilistic_emergency_direction = None
            self._probabilistic_emergency_pending_direction = None
            self._probabilistic_emergency_pending_direction_count = 0
            self._probabilistic_emergency_latched_pattern = None
            self._probabilistic_emergency_latched_heading = None
            self._probabilistic_emergency_rearm_ready = True
            self._probabilistic_emergency_rearm_clear_count = 0
            return {
                "triggered": False,
                "raw_triggered": False,
                "closing_observed": False,
                "scan_valid": False,
                "scan_flow_match": False,
                "scan_quality_ok": False,
                "scan_support_beams": 0,
                "scan_rejected_jump_fraction": 1.0,
                "forecast_evidence_ok": False,
                "intent_held": False,
                "away_heading_error_rad": None,
                "obstacle_bearing_rad": None,
                "front_obstacle_forward_turn": False,
                "near_distance_triggered": False,
                "critical_distance_triggered": False,
                "forecast_corroboration_enabled": bool(
                    self.config
                    .probabilistic_obstacle_emergency_forecast_corroboration_enabled
                ),
                "forecast_corroborated": False,
                "forecast_stop_maximum_probability": 0.0,
                "surface_range_m": float("inf"),
                "reserve_reverse_coverage": False,
                "ttc_s": float("inf"),
                "safety_hard_stop_ttc_s": 0.0,
                "rearm_ready": True,
                "rearm_clear_count": 0,
                "escape_direction_refreshed": False,
                "escape_direction_alignment": 1.0,
                "preferred_escape_heading_error_rad": None,
                "escape_direction_source": "unavailable",
                "forward_lateral_countermotion_applied": False,
                "obstacle_motion_longitudinal_body_mps": 0.0,
                "obstacle_motion_lateral_body_mps": 0.0,
                "obstacle_motion_lateral_fraction": 0.0,
                "escape_direction_reversal_confirmation_count": 0,
            }
        context = dict(
            observation.auxiliary.get(
                "dynamic_obstacle_escape_context", {}
            )
        )
        ttc_s = float(context.get("temporal_scan_ttc_s", float("inf")))
        scan_valid = bool(context.get("temporal_scan_valid", False))
        scan_flow_match = bool(
            context.get("dynamic_obstacle_scan_flow_match", False)
        )
        scan_support_beams = int(
            context.get("temporal_scan_support_beams", 0) or 0
        )
        scan_rejected_jump_fraction = float(
            context.get("temporal_scan_rejected_jump_fraction", 1.0)
            or 0.0
        )
        if not np.isfinite(scan_rejected_jump_fraction):
            scan_rejected_jump_fraction = 1.0
        scan_quality_ok = bool(
            not self.config.probabilistic_obstacle_emergency_require_scan_quality
            or (
                scan_valid
                and scan_support_beams
                >= self.config.probabilistic_obstacle_emergency_min_support_beams
                and scan_rejected_jump_fraction
                <= self.config.probabilistic_obstacle_emergency_max_rejected_jump_fraction
            )
        )
        safety_hard_stop_ttc_s = float(
            context.get("temporal_scan_safety_hard_stop_ttc_s", 0.0)
        )
        closing_observed = bool(
            scan_valid
            and scan_flow_match
            and np.isfinite(ttc_s)
            and ttc_s > 0.0
        )
        surface_range_value = context.get(
            "dynamic_obstacle_surface_range_m"
        )
        surface_range_m = (
            float(surface_range_value)
            if surface_range_value is not None
            else float("inf")
        )
        near_distance_triggered = bool(
            trigger_distance > 0.0
            and np.isfinite(surface_range_m)
            and surface_range_m <= trigger_distance
        )
        critical_distance = float(
            self.config
            .probabilistic_obstacle_emergency_candidate_critical_distance_m
        )
        critical_distance_triggered = bool(
            critical_distance > 0.0
            and np.isfinite(surface_range_m)
            and surface_range_m <= critical_distance
        )
        forecast_corroboration_enabled = bool(
            self.config
            .probabilistic_obstacle_emergency_forecast_corroboration_enabled
        )
        forecast_stop_maximum_probability = 0.0
        forecast_corroborated = not forecast_corroboration_enabled
        if (
            forecast_corroboration_enabled
            and state is not None
            and probabilistic_obstacles
        ):
            stop_sequence = np.zeros(
                (self.config.horizon, self.action_spec.dimension),
                dtype=np.float64,
            )
            stop_trajectory = self.rollout(
                np.asarray(state, dtype=np.float64),
                stop_sequence,
            )[0]
            stop_risk = self._probabilistic_collision_risk(
                stop_trajectory[None, ...],
                probabilistic_obstacles,
            )
            forecast_stop_maximum_probability = float(
                stop_risk.maximum_step_probability[0]
            )
            forecast_corroborated = bool(
                forecast_stop_maximum_probability
                >= self.config.probabilistic_obstacle_hard_threshold
                - 1.0e-12
            )
        forecast_evidence_ok = bool(
            context.get("dynamic_obstacle_tracker_forecast_valid", False)
            and context.get("dynamic_obstacle_tracker_associated", False)
        )
        noncritical_trigger = bool(
            (
                threshold > 0.0
                and closing_observed
                and scan_quality_ok
                and ttc_s <= threshold
            )
            or (
                near_distance_triggered
                and scan_quality_ok
            )
        )
        raw_triggered = bool(
            critical_distance_triggered
            or (
                noncritical_trigger
                and forecast_corroborated
                and (
                    not forecast_corroboration_enabled
                    or not self.config.probabilistic_obstacle_emergency_require_scan_quality
                    or forecast_evidence_ok
                )
            )
        )
        start_intent = bool(
            raw_triggered and self._probabilistic_emergency_rearm_ready
        )
        near_intent_unresolved = bool(
            critical_distance_triggered
            or (
                near_distance_triggered
                and forecast_corroborated
            )
        )
        intent_held = bool(
            not start_intent
            and (
                near_intent_unresolved
                or self._probabilistic_emergency_intent_remaining > 0
            )
        )
        triggered = bool(start_intent or intent_held)
        away_heading_error = context.get(
            "dynamic_obstacle_away_heading_error_rad"
        )
        obstacle_bearing = context.get("dynamic_obstacle_bearing_rad")
        if obstacle_bearing is not None:
            obstacle_bearing = float(obstacle_bearing)
            if not np.isfinite(obstacle_bearing):
                obstacle_bearing = None
        if away_heading_error is not None:
            away_heading_error = float(away_heading_error)
            if not np.isfinite(away_heading_error):
                away_heading_error = None
        result = {
            "triggered": triggered,
            "raw_triggered": raw_triggered,
            "closing_observed": closing_observed,
            "scan_valid": scan_valid,
            "scan_flow_match": scan_flow_match,
            "scan_quality_ok": scan_quality_ok,
            "scan_support_beams": scan_support_beams,
            "scan_rejected_jump_fraction": scan_rejected_jump_fraction,
            "forecast_evidence_ok": forecast_evidence_ok,
            "intent_held": intent_held,
            "away_heading_error_rad": away_heading_error,
            "obstacle_bearing_rad": obstacle_bearing,
            "front_obstacle_forward_turn": bool(
                self.config
                .probabilistic_obstacle_front_obstacle_forward_turn_enabled
                and obstacle_bearing is not None
                and abs(obstacle_bearing) <= 0.5 * np.pi
            ),
            "near_distance_triggered": near_distance_triggered,
            "critical_distance_triggered": critical_distance_triggered,
            "forecast_corroboration_enabled": (
                forecast_corroboration_enabled
            ),
            "forecast_corroborated": forecast_corroborated,
            "forecast_stop_maximum_probability": (
                forecast_stop_maximum_probability
            ),
            "surface_range_m": surface_range_m,
            "reserve_reverse_coverage": critical_distance_triggered,
            "ttc_s": ttc_s,
            "safety_hard_stop_ttc_s": safety_hard_stop_ttc_s,
            "escape_direction_refreshed": False,
            "escape_direction_alignment": 1.0,
            "preferred_escape_heading_error_rad": None,
            "escape_direction_source": "unavailable",
            "forward_lateral_countermotion_applied": False,
            "obstacle_motion_longitudinal_body_mps": 0.0,
            "obstacle_motion_lateral_body_mps": 0.0,
            "obstacle_motion_lateral_fraction": 0.0,
            "escape_direction_reversal_confirmation_count": 0,
        }
        if (
            start_intent
            and self._probabilistic_emergency_latched_pattern is None
            and away_heading_error is not None
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        ):
            v_index = self.action_spec.index("v_cmd")
            omega_index = self.action_spec.index("omega_cmd")
            reverse = abs(away_heading_error) > 0.5 * np.pi
            motion_heading_error = away_heading_error
            if reverse:
                motion_heading_error = float(np.arctan2(
                    np.sin(away_heading_error - np.pi),
                    np.cos(away_heading_error - np.pi),
                ))
            # If maximum translation already carries the robot broadly away
            # from the obstacle, preserve that separation rate instead of
            # spending the short TTC window on a tight arc.  Large heading
            # errors still receive a saturated turn.
            yaw_rate = 0.0
            if abs(motion_heading_error) > 1.0:
                yaw_rate = (
                    self.action_spec.upper[omega_index]
                    if motion_heading_error >= 0.0
                    else self.action_spec.lower[omega_index]
                )
            self._probabilistic_emergency_latched_pattern = (
                float(
                    self.action_spec.lower[v_index]
                    if reverse else self.action_spec.upper[v_index]
                ),
                float(yaw_rate),
            )
        if (
            triggered
            and self._probabilistic_emergency_latched_pattern is not None
        ):
            latched_pattern = self._probabilistic_emergency_latched_pattern
            if (
                self._probabilistic_emergency_latched_heading is not None
                and state is not None
                and "theta" in self.state_spec.names
                and "omega_cmd" in self.action_spec.names
            ):
                theta = float(
                    np.asarray(state, dtype=np.float64)[
                        self.state_spec.index("theta")
                    ]
                )
                heading_error = float(np.arctan2(
                    np.sin(
                        self._probabilistic_emergency_latched_heading - theta
                    ),
                    np.cos(
                        self._probabilistic_emergency_latched_heading - theta
                    ),
                ))
                omega_index = self.action_spec.index("omega_cmd")
                latched_pattern = (
                    float(latched_pattern[0]),
                    float(np.clip(
                        1.5 * heading_error,
                        self.action_spec.lower[omega_index],
                        self.action_spec.upper[omega_index],
                    )),
                )
            result["latched_escape_pattern"] = latched_pattern
        if (
            triggered
            and state is not None
            and probabilistic_obstacles
            and "theta" in self.state_spec.names
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
        ):
            forecasts = tuple(probabilistic_obstacles)
            forecast_index = int(
                context.get("dynamic_obstacle_forecast_index", 0) or 0
            )
            if not 0 <= forecast_index < len(forecasts):
                forecast_index = 0
            forecast = forecasts[forecast_index]
            result["escape_forecast_index"] = forecast_index
            component_means = np.asarray(
                forecast.component_means, dtype=np.float64
            )
            component_weights = np.asarray(
                forecast.component_weights, dtype=np.float64
            )
            mixture_means = np.sum(
                component_weights[..., None] * component_means, axis=1
            )
            lookahead = min(4, mixture_means.shape[0] - 1)
            measured_velocity = np.asarray((
                context.get(
                    "dynamic_obstacle_measurement_velocity_x_mps",
                    np.nan,
                ),
                context.get(
                    "dynamic_obstacle_measurement_velocity_y_mps",
                    np.nan,
                ),
            ), dtype=np.float64)
            if (
                np.isfinite(measured_velocity).all()
                and float(np.linalg.norm(measured_velocity)) > 0.05
            ):
                # A short causal regression over associated LaserScan centers
                # supplies direction when the IMM has not yet adapted to a
                # maneuver change.  Probability certification below still
                # uses the full frozen tracker forecast.
                obstacle_motion = measured_velocity
                result["escape_direction_source"] = (
                    "causal_scan_measurement_regression"
                )
            else:
                obstacle_motion = (
                    mixture_means[lookahead] - mixture_means[0]
                )
                result["escape_direction_source"] = "tracker_forecast"
            motion_norm = float(np.linalg.norm(obstacle_motion))
            if motion_norm > 1.0e-6:
                tangent = obstacle_motion / motion_norm
                perpendicular = np.asarray(
                    (-tangent[1], tangent[0]), dtype=np.float64
                )
                robot_xy = np.asarray(state, dtype=np.float64)[
                    list(self.state_spec.position_indices)
                ]
                theta = float(
                    np.asarray(state, dtype=np.float64)[
                        self.state_spec.index("theta")
                    ]
                )
                (
                    forward_lateral_direction,
                    longitudinal_body_speed,
                    lateral_body_speed,
                    lateral_motion_fraction,
                ) = self._forward_lateral_countermotion_direction(
                    obstacle_motion,
                    theta,
                    self.config
                    .probabilistic_obstacle_forward_lateral_countermotion_weight,
                    self.config
                    .probabilistic_obstacle_forward_lateral_minimum_speed_mps,
                    self.config
                    .probabilistic_obstacle_forward_lateral_minimum_fraction,
                )
                result["obstacle_motion_longitudinal_body_mps"] = (
                    longitudinal_body_speed
                )
                result["obstacle_motion_lateral_body_mps"] = lateral_body_speed
                result["obstacle_motion_lateral_fraction"] = (
                    lateral_motion_fraction
                )
                forward_lateral_applied = bool(
                    self.config
                    .probabilistic_obstacle_forward_lateral_countermotion_enabled
                    and forward_lateral_direction is not None
                )
                result["forward_lateral_countermotion_applied"] = (
                    forward_lateral_applied
                )
                forward_lateral_hold = bool(
                    self.config
                    .probabilistic_obstacle_forward_lateral_countermotion_enabled
                    and forward_lateral_direction is None
                    and self._probabilistic_emergency_direction is not None
                )
                if forward_lateral_applied:
                    measured_preferred_direction = forward_lateral_direction
                    result["escape_direction_source"] = (
                        str(result["escape_direction_source"])
                        + "_forward_lateral_countermotion"
                    )
                elif forward_lateral_hold:
                    # Once a crossing side exists, a single low-lateral or
                    # nearly stationary measurement is not evidence for a new
                    # side.  Retain the transaction until either a significant
                    # opposite motion is confirmed or the encounter re-arms.
                    measured_preferred_direction = np.asarray(
                        self._probabilistic_emergency_direction,
                        dtype=np.float64,
                    ).copy()
                    result["escape_direction_source"] = (
                        "held_during_insignificant_lateral_motion"
                    )
                else:
                    lateral_side = float(np.dot(
                        robot_xy - mixture_means[0], perpendicular
                    ))
                    if abs(lateral_side) <= 1.0e-9:
                        lateral_side = 1.0
                    measured_preferred_direction = (
                        np.sign(lateral_side) * perpendicular
                    )
                if (
                    self.config
                    .probabilistic_obstacle_counterflow_escape_enabled
                    and not forward_lateral_applied
                    and not forward_lateral_hold
                ):
                    counterflow_direction = (
                        measured_preferred_direction
                        - self.config.probabilistic_obstacle_counterflow_weight
                        * tangent
                    )
                    counterflow_norm = float(
                        np.linalg.norm(counterflow_direction)
                    )
                    if counterflow_norm > 1.0e-9:
                        candidate_direction = (
                            counterflow_direction / counterflow_norm
                        )
                        vetoed = self._escape_direction_enters_occupancy(
                            robot_xy, candidate_direction, forecasts
                        )
                        result[
                            "counterflow_rear_occupancy_vetoed"
                        ] = bool(vetoed)
                        if vetoed:
                            # Keep the pure perpendicular: it moves ACROSS the
                            # obstacle's motion rather than back into whatever
                            # the counterflow term was steering toward.
                            result["counterflow_escape_applied"] = False
                        else:
                            measured_preferred_direction = candidate_direction
                            result["counterflow_escape_applied"] = True
                    else:
                        result["counterflow_escape_applied"] = False
                else:
                    result["counterflow_escape_applied"] = False
                previous_direction = self._probabilistic_emergency_direction
                direction_alignment = 1.0
                if previous_direction is not None:
                    previous_direction = np.asarray(
                        previous_direction, dtype=np.float64
                    )
                    direction_alignment = float(np.clip(
                        np.dot(
                            previous_direction
                            / max(float(np.linalg.norm(previous_direction)), 1.0e-9),
                            measured_preferred_direction
                            / max(float(np.linalg.norm(measured_preferred_direction)), 1.0e-9),
                        ),
                        -1.0,
                        1.0,
                    ))
                tracker_change_triggered = bool(
                    context.get(
                        "dynamic_obstacle_tracker_change_triggered", False
                    )
                )
                reversal_confirmation_count = 0
                reversal_confirmed = False
                if (
                    previous_direction is not None
                    and forward_lateral_applied
                    and direction_alignment < 0.0
                    and not tracker_change_triggered
                ):
                    pending_direction = (
                        self._probabilistic_emergency_pending_direction
                    )
                    pending_alignment = -1.0
                    if pending_direction is not None:
                        pending_direction = np.asarray(
                            pending_direction, dtype=np.float64
                        )
                        pending_alignment = float(np.clip(
                            np.dot(
                                pending_direction
                                / max(float(np.linalg.norm(
                                    pending_direction
                                )), 1.0e-9),
                                measured_preferred_direction
                                / max(float(np.linalg.norm(
                                    measured_preferred_direction
                                )), 1.0e-9),
                            ),
                            -1.0,
                            1.0,
                        ))
                    if pending_alignment >= np.cos(np.deg2rad(20.0)):
                        self._probabilistic_emergency_pending_direction_count += 1
                    else:
                        self._probabilistic_emergency_pending_direction = (
                            measured_preferred_direction.copy()
                        )
                        self._probabilistic_emergency_pending_direction_count = 1
                    reversal_confirmation_count = int(
                        self._probabilistic_emergency_pending_direction_count
                    )
                    reversal_confirmed = bool(
                        reversal_confirmation_count
                        >= self.config
                        .probabilistic_obstacle_forward_lateral_reversal_confirm_steps
                    )
                else:
                    self._probabilistic_emergency_pending_direction = None
                    self._probabilistic_emergency_pending_direction_count = 0
                direction_refresh = bool(
                    previous_direction is not None
                    and (
                        (
                            direction_alignment < 0.0
                            and not forward_lateral_applied
                        )
                        or (
                            tracker_change_triggered
                            and direction_alignment < np.cos(0.25 * np.pi)
                        )
                        or reversal_confirmed
                    )
                )
                if previous_direction is None or direction_refresh:
                    self._probabilistic_emergency_direction = (
                        measured_preferred_direction.copy()
                    )
                if direction_refresh:
                    # The old intent latch represented a now-invalid human
                    # trajectory.  Remove it on this same solve so candidate
                    # generation and the physical arbiter both see the new
                    # counter-motion direction immediately.
                    self._probabilistic_emergency_latched_pattern = None
                    self._probabilistic_emergency_latched_heading = None
                    result.pop("latched_escape_pattern", None)
                    self._probabilistic_emergency_pending_direction = None
                    self._probabilistic_emergency_pending_direction_count = 0
                result["escape_direction_refreshed"] = direction_refresh
                result["escape_direction_alignment"] = direction_alignment
                result[
                    "escape_direction_reversal_confirmation_count"
                ] = reversal_confirmation_count
                preferred_direction = np.asarray(
                    self._probabilistic_emergency_direction,
                    dtype=np.float64,
                )
                result["preferred_escape_direction_x"] = float(
                    preferred_direction[0]
                )
                result["preferred_escape_direction_y"] = float(
                    preferred_direction[1]
                )
                theta = float(
                    np.asarray(state, dtype=np.float64)[
                        self.state_spec.index("theta")
                    ]
                )
                preferred_heading = float(np.arctan2(
                    preferred_direction[1], preferred_direction[0]
                ))
                result["preferred_escape_heading_error_rad"] = float(
                    np.arctan2(
                        np.sin(preferred_heading - theta),
                        np.cos(preferred_heading - theta),
                    )
                )
                v_index = self.action_spec.index("v_cmd")
                omega_index = self.action_spec.index("omega_cmd")

                def motion_pattern(direction):
                    desired_heading = float(np.arctan2(
                        direction[1], direction[0]
                    ))
                    forward_error = float(np.arctan2(
                        np.sin(desired_heading - theta),
                        np.cos(desired_heading - theta),
                    ))
                    reverse_error = float(np.arctan2(
                        np.sin(desired_heading - np.pi - theta),
                        np.cos(desired_heading - np.pi - theta),
                    ))
                    reverse = abs(reverse_error) < abs(forward_error)
                    heading_error = reverse_error if reverse else forward_error
                    return (
                        float(
                            self.action_spec.lower[v_index]
                            if reverse else self.action_spec.upper[v_index]
                        ),
                        float(np.clip(
                            1.5 * heading_error,
                            self.action_spec.lower[omega_index],
                            self.action_spec.upper[omega_index],
                        )),
                    )

                result["forecast_escape_patterns"] = (
                    motion_pattern(preferred_direction),
                    motion_pattern(-preferred_direction),
                )
        if start_intent:
            self._probabilistic_emergency_intent_remaining = int(
                self.config
                .probabilistic_obstacle_emergency_candidate_intent_hold_steps
            )
            self._probabilistic_emergency_rearm_ready = False
            self._probabilistic_emergency_rearm_clear_count = 0
        elif near_distance_triggered:
            # A still-close obstacle is not a one-shot event. Keep the chosen
            # direction latched and spend the configured hold only after the
            # measured separation rises above the trigger distance.
            self._probabilistic_emergency_rearm_clear_count = 0
        elif self._probabilistic_emergency_intent_remaining > 0:
            self._probabilistic_emergency_intent_remaining -= 1
            if self._probabilistic_emergency_intent_remaining == 0:
                self._probabilistic_emergency_direction = None
                self._probabilistic_emergency_pending_direction = None
                self._probabilistic_emergency_pending_direction_count = 0
                self._probabilistic_emergency_latched_pattern = None
                self._probabilistic_emergency_latched_heading = None
        else:
            self._probabilistic_emergency_direction = None
            self._probabilistic_emergency_pending_direction = None
            self._probabilistic_emergency_pending_direction_count = 0
            self._probabilistic_emergency_latched_pattern = None
            self._probabilistic_emergency_latched_heading = None
            rearm_ttc = float(
                self.config
                .probabilistic_obstacle_emergency_candidate_rearm_ttc_s
            )
            if rearm_ttc <= 0.0:
                rearm_ttc = threshold
            rearm_clear = bool(
                not raw_triggered
                and (
                    not np.isfinite(ttc_s)
                    or ttc_s >= rearm_ttc
                )
            )
            if rearm_clear:
                self._probabilistic_emergency_rearm_clear_count += 1
            else:
                self._probabilistic_emergency_rearm_clear_count = 0
            if self._probabilistic_emergency_rearm_clear_count >= int(
                self.config
                .probabilistic_obstacle_emergency_candidate_rearm_clear_steps
            ):
                self._probabilistic_emergency_rearm_ready = True
                self._probabilistic_emergency_rearm_clear_count = 0
        result["rearm_ready"] = bool(
            self._probabilistic_emergency_rearm_ready
        )
        result["rearm_clear_count"] = int(
            self._probabilistic_emergency_rearm_clear_count
        )
        return result

    def _inject_probabilistic_emergency_candidates(
        self, samples, prior, emergency_context=None
    ):
        """Replace six budgeted samples with deterministic escape maneuvers."""

        self._probabilistic_emergency_nonforward_coverage_applied = False
        self._probabilistic_emergency_forward_exit_coverage_applied = False
        mask = np.zeros(self.config.num_samples, dtype=bool)
        if not self.config.probabilistic_obstacle_emergency_candidates_enabled:
            return mask
        if (
            "v_cmd" not in self.action_spec.names
            or "omega_cmd" not in self.action_spec.names
        ):
            return mask
        v_index = self.action_spec.index("v_cmd")
        omega_index = self.action_spec.index("omega_cmd")
        prefix = self.config.probabilistic_obstacle_emergency_candidate_prefix_steps
        base_patterns = (
            (self.action_spec.upper[v_index], 0.0),
            (self.action_spec.lower[v_index], 0.0),
            (self.action_spec.upper[v_index], self.action_spec.upper[omega_index]),
            (self.action_spec.upper[v_index], self.action_spec.lower[omega_index]),
            (self.action_spec.lower[v_index], self.action_spec.upper[omega_index]),
            (self.action_spec.lower[v_index], self.action_spec.lower[omega_index]),
        )
        patterns = base_patterns
        emergency_context = dict(emergency_context or {})
        forecast_patterns = tuple(
            emergency_context.get("forecast_escape_patterns", ())
        )
        latched_pattern = emergency_context.get("latched_escape_pattern")
        away_heading_error = emergency_context.get("away_heading_error_rad")
        obstacle_bearing = emergency_context.get("obstacle_bearing_rad")
        front_obstacle_forward_turn = bool(
            emergency_context.get("front_obstacle_forward_turn", False)
        )
        away_pattern = None
        if (
            front_obstacle_forward_turn
            and obstacle_bearing is not None
            and np.isfinite(obstacle_bearing)
        ):
            turn_sign = -1.0 if float(obstacle_bearing) >= 0.0 else 1.0
            away_pattern = (
                float(self.action_spec.upper[v_index]),
                float(
                    self.action_spec.upper[omega_index]
                    if turn_sign > 0.0
                    else self.action_spec.lower[omega_index]
                ),
            )
        elif away_heading_error is not None and np.isfinite(away_heading_error):
            away_heading_error = float(away_heading_error)
            reverse = abs(away_heading_error) > 0.5 * np.pi
            motion_heading_error = away_heading_error
            if reverse:
                motion_heading_error = float(np.arctan2(
                    np.sin(away_heading_error - np.pi),
                    np.cos(away_heading_error - np.pi),
                ))
            away_pattern = (
                self.action_spec.lower[v_index]
                if reverse else self.action_spec.upper[v_index],
                float(np.clip(
                    1.5 * motion_heading_error,
                    self.action_spec.lower[omega_index],
                    self.action_spec.upper[omega_index],
                )),
            )
        if (
            latched_pattern is not None
            or away_pattern is not None
            or forecast_patterns
        ):
            ordered = (
                ([] if latched_pattern is None else [latched_pattern])
                + ([] if away_pattern is None else [away_pattern])
                + list(forecast_patterns)
                + list(patterns)
            )
            unique = []
            for pattern in ordered:
                pattern = tuple(float(value) for value in pattern)
                if not any(np.allclose(
                    pattern, existing, atol=1.0e-12, rtol=0.0
                ) for existing in unique):
                    unique.append(pattern)
            if emergency_context.get(
                "post_center_forward_exit_commit_requested", False
            ):
                self._probabilistic_emergency_forward_exit_coverage_applied = True
                # Once the robot has crossed the frozen conflict center,
                # reversing prolongs occupancy and can re-enter the obstacle
                # path.  Preserve straight/left/right forward templates in
                # the same six slots so the final risk guard can enforce the
                # already committed exit direction without adding rollouts.
                forward_templates = [
                    tuple(float(value) for value in pattern)
                    for pattern in base_patterns
                    if pattern[0] > 0.0
                ]
                preferred_capacity = max(0, 6 - len(forward_templates))
                preferred = []
                for pattern in unique:
                    if any(np.allclose(
                        pattern,
                        forward,
                        atol=1.0e-12,
                        rtol=0.0,
                    ) for forward in forward_templates):
                        continue
                    preferred.append(pattern)
                    if len(preferred) >= preferred_capacity:
                        break
                patterns = tuple(preferred + forward_templates)
            elif (
                emergency_context.get(
                    "post_center_low_ttc_nonforward_coverage_requested",
                    False,
                )
                or emergency_context.get(
                    "reserve_reverse_coverage", False
                )
            ):
                self._probabilistic_emergency_nonforward_coverage_applied = True
                # Forecast, away-heading and latched preferences can otherwise
                # occupy all six fixed lattice slots with forward patterns.
                # During an unresolved post-center hard-stop TTC dropout,
                # reserve the three existing reverse templates so the action
                # guard has straight/left/right non-forward evidence every
                # cycle.  Candidate values, slot count and rollout budget do
                # not change.
                reverse_templates = [
                    tuple(float(value) for value in pattern)
                    for pattern in base_patterns
                    if pattern[0] <= 0.0
                ]
                preferred_capacity = max(0, 6 - len(reverse_templates))
                preferred = []
                for pattern in unique:
                    if any(np.allclose(
                        pattern,
                        reverse,
                        atol=1.0e-12,
                        rtol=0.0,
                    ) for reverse in reverse_templates):
                        continue
                    preferred.append(pattern)
                    if len(preferred) >= preferred_capacity:
                        break
                patterns = tuple(preferred + reverse_templates)
            else:
                patterns = tuple(unique[:6])
        start = self.config.num_samples - len(patterns)
        proposal_mean = getattr(prior, "mean", None)
        if proposal_mean is None or callable(proposal_mean):
            proposal_mean = prior
        proposal_mean = np.asarray(proposal_mean, dtype=np.float64)
        for offset, (speed, yaw_rate) in enumerate(patterns):
            index = start + offset
            samples[index] = proposal_mean
            samples[index, :prefix, v_index] = speed
            samples[index, :prefix, omega_index] = yaw_rate
            if self.config.probabilistic_obstacle_emergency_candidate_slew_enabled:
                # Emergency candidates remain decisive, but enter through the
                # same actuator contract as nominal MPPI.  Without this pass a
                # 0 -> +/-v and 0 -> +/-omega jump was injected into the first
                # command and the next 1.2 s of the prefix, producing the
                # observed reverse/turn snap and command staircasing.
                previous = np.asarray(
                    self.previous_action, dtype=np.float64
                ).copy()
                for step in range(prefix):
                    samples[index, step] = self.action_spec.clip(
                        samples[index, step], previous, self.config.dt
                    )
                    previous = samples[index, step].copy()
            if (
                self.config
                .probabilistic_obstacle_emergency_candidate_hold_tail_enabled
            ):
                samples[index, prefix:, v_index] = 0.0
                samples[index, prefix:, omega_index] = 0.0
            mask[index] = True
        return mask

    def _clear_probabilistic_traversal_commit(self):
        self._probabilistic_traversal_crossing_progress = None
        self._probabilistic_traversal_entry_progress = None
        self._probabilistic_traversal_clear_progress = None
        self._probabilistic_traversal_commit_started = False
        self._probabilistic_traversal_admission_safe_streak = 0
        self._probabilistic_traversal_admission_signature = None
        self._clear_probabilistic_traversal_post_center_temporal_escape()

    def _clear_probabilistic_traversal_exit_deadline_retreat_escape(self):
        """Release the causal escape bound to one deadline retreat."""

        self._probabilistic_traversal_exit_deadline_retreat_active = False
        self._probabilistic_traversal_exit_deadline_retreat_pattern = None
        self._probabilistic_traversal_exit_deadline_retreat_heading = None

    def _start_probabilistic_traversal_exit_deadline_retreat_escape(self):
        """Start a fresh deadline-retreat transaction before action choice."""

        self._probabilistic_traversal_exit_deadline_retreat_active = bool(
            self.config
            .probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled
        )
        self._probabilistic_traversal_exit_deadline_retreat_pattern = None
        self._probabilistic_traversal_exit_deadline_retreat_heading = None

    def _probabilistic_traversal_exit_deadline_retreat_escape_pattern(
        self, state
    ):
        """Return the deadline-retreat escape with heading stabilization."""

        pattern = self._probabilistic_traversal_exit_deadline_retreat_pattern
        if pattern is None:
            return None
        pattern = tuple(float(value) for value in pattern)
        if (
            self._probabilistic_traversal_exit_deadline_retreat_heading is None
            or state is None
            or "theta" not in self.state_spec.names
            or "omega_cmd" not in self.action_spec.names
        ):
            return pattern
        theta = float(
            np.asarray(state, dtype=np.float64)[
                self.state_spec.index("theta")
            ]
        )
        heading_error = float(np.arctan2(
            np.sin(
                self._probabilistic_traversal_exit_deadline_retreat_heading
                - theta
            ),
            np.cos(
                self._probabilistic_traversal_exit_deadline_retreat_heading
                - theta
            ),
        ))
        omega_index = self.action_spec.index("omega_cmd")
        return (
            float(pattern[0]),
            float(np.clip(
                1.5 * heading_error,
                self.action_spec.lower[omega_index],
                self.action_spec.upper[omega_index],
            )),
        )

    def _latch_probabilistic_traversal_exit_deadline_retreat_escape(
        self, pattern, state
    ):
        """Latch the first selected causal escape for one deadline retreat."""

        if (
            not self.config
            .probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled
            or not self._probabilistic_traversal_exit_deadline_retreat_active
            or self._probabilistic_traversal_exit_deadline_retreat_pattern
            is not None
        ):
            return False
        pattern = tuple(float(value) for value in pattern)
        if len(pattern) != 2 or not np.isfinite(pattern).all():
            return False
        self._probabilistic_traversal_exit_deadline_retreat_pattern = pattern
        self._probabilistic_traversal_exit_deadline_retreat_heading = None
        if (
            state is not None
            and "theta" in self.state_spec.names
            and "omega_cmd" in self.action_spec.names
        ):
            theta = float(
                np.asarray(state, dtype=np.float64)[
                    self.state_spec.index("theta")
                ]
            )
            target_heading = theta + pattern[1] * float(
                self.config
                .probabilistic_obstacle_emergency_candidate_prefix_steps
            ) * float(self.config.dt)
            self._probabilistic_traversal_exit_deadline_retreat_heading = float(
                np.arctan2(np.sin(target_heading), np.cos(target_heading))
            )
        return True

    def _bind_probabilistic_traversal_exit_deadline_retreat_escape(
        self, emergency_context, traversal_context, state
    ):
        """Keep one deadline-retreat escape authoritative for its retreat."""

        emergency_context = dict(emergency_context or {})
        traversal_context = dict(traversal_context or {})
        active = bool(
            self.config
            .probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled
            and self._probabilistic_traversal_exit_deadline_retreat_active
            and traversal_context.get("retreat_requested", False)
            and traversal_context.get(
                "retreat_temporal_lattice_requested", False
            )
        )
        pattern = (
            self._probabilistic_traversal_exit_deadline_retreat_escape_pattern(
                state
            )
            if active
            else None
        )
        if not active:
            self._clear_probabilistic_traversal_exit_deadline_retreat_escape()
        elif pattern is not None:
            emergency_context["latched_escape_pattern"] = pattern
        traversal_context[
            "exit_deadline_retreat_escape_transaction_active"
        ] = active
        traversal_context["exit_deadline_retreat_escape_reused"] = bool(
            active and pattern is not None
        )
        return emergency_context, traversal_context

    def _clear_probabilistic_traversal_post_center_temporal_escape(self):
        """Release the causal post-center escape transaction."""

        self._probabilistic_traversal_post_center_temporal_pattern = None
        self._probabilistic_traversal_post_center_temporal_heading = None

    def _probabilistic_traversal_post_center_temporal_escape_pattern(
        self, state
    ):
        """Return the latched escape with one-turn heading stabilization."""

        pattern = self._probabilistic_traversal_post_center_temporal_pattern
        if pattern is None:
            return None
        pattern = tuple(float(value) for value in pattern)
        if (
            self._probabilistic_traversal_post_center_temporal_heading is None
            or state is None
            or "theta" not in self.state_spec.names
            or "omega_cmd" not in self.action_spec.names
        ):
            return pattern
        theta = float(
            np.asarray(state, dtype=np.float64)[
                self.state_spec.index("theta")
            ]
        )
        heading_error = float(np.arctan2(
            np.sin(
                self._probabilistic_traversal_post_center_temporal_heading
                - theta
            ),
            np.cos(
                self._probabilistic_traversal_post_center_temporal_heading
                - theta
            ),
        ))
        omega_index = self.action_spec.index("omega_cmd")
        return (
            float(pattern[0]),
            float(np.clip(
                1.5 * heading_error,
                self.action_spec.lower[omega_index],
                self.action_spec.upper[omega_index],
            )),
        )

    def _latch_probabilistic_traversal_post_center_temporal_escape(
        self, pattern, state
    ):
        """Latch the first selected causal escape for one active commit."""

        if (
            not self.config
            .probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled
            or self._probabilistic_traversal_post_center_temporal_pattern
            is not None
        ):
            return False
        pattern = tuple(float(value) for value in pattern)
        if len(pattern) != 2 or not np.isfinite(pattern).all():
            return False
        self._probabilistic_traversal_post_center_temporal_pattern = pattern
        self._probabilistic_traversal_post_center_temporal_heading = None
        if (
            state is not None
            and "theta" in self.state_spec.names
            and "omega_cmd" in self.action_spec.names
        ):
            theta = float(
                np.asarray(state, dtype=np.float64)[
                    self.state_spec.index("theta")
                ]
            )
            target_heading = theta + pattern[1] * float(
                self.config
                .probabilistic_obstacle_emergency_candidate_prefix_steps
            ) * float(self.config.dt)
            self._probabilistic_traversal_post_center_temporal_heading = float(
                np.arctan2(np.sin(target_heading), np.cos(target_heading))
            )
        return True

    def _bind_probabilistic_traversal_post_center_temporal_escape(
        self, emergency_context, traversal_context, state
    ):
        """Bind one escape transaction while the same causal warning holds."""

        emergency_context = dict(emergency_context or {})
        traversal_context = dict(traversal_context or {})
        active = bool(
            self.config
            .probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled
            and self.config
            .probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled
            and traversal_context.get("candidate_requested", False)
            and traversal_context.get("commit_started", False)
            and not traversal_context.get("retreat_requested", False)
            and not traversal_context.get("rearm_pending", False)
            and traversal_context.get("temporal_emergency_raw_triggered", False)
            and float(traversal_context.get(
                "temporal_corroboration_probability_mass", 0.0
            )) >= float(
                self.config
                .probabilistic_obstacle_traversal_window_temporal_abort_mass_floor
            )
            and float(traversal_context.get("current_progress", 0.0))
            >= float(traversal_context.get("crossing_progress", 0.0))
            - 1.0e-9
            and float(traversal_context.get("current_progress", 0.0))
            < float(traversal_context.get("clear_progress", 0.0))
            - 1.0e-9
        )
        pattern = (
            self._probabilistic_traversal_post_center_temporal_escape_pattern(
                state
            )
            if active
            else None
        )
        if not active:
            self._clear_probabilistic_traversal_post_center_temporal_escape()
        elif pattern is not None:
            # The transaction-specific causal maneuver outranks the ordinary
            # short intent latch, which is intentionally free to expire and
            # rearm independently.
            emergency_context["latched_escape_pattern"] = pattern
        traversal_context["post_center_temporal_escape_transaction_active"] = (
            active
        )
        traversal_context["post_center_temporal_escape_reused"] = bool(
            active and pattern is not None
        )
        return emergency_context, traversal_context

    def _update_probabilistic_traversal_admission_streak(
        self,
        *,
        safe,
        forecast_index,
        crossing_progress,
        entry_progress,
        clear_progress,
    ):
        """Track consecutive safe certificates for one causal crossing geometry."""

        if not safe:
            self._probabilistic_traversal_admission_safe_streak = 0
            self._probabilistic_traversal_admission_signature = None
            return 0
        signature = (
            int(forecast_index),
            float(crossing_progress),
            float(entry_progress),
            float(clear_progress),
        )
        previous = self._probabilistic_traversal_admission_signature
        v_index = self.action_spec.index("v_cmd")
        geometry_tolerance = max(
            1.0e-9,
            float(self.action_spec.upper[v_index]) * float(self.config.dt),
        )
        same_geometry = bool(
            previous is not None
            and int(previous[0]) == signature[0]
            and np.allclose(
                np.asarray(previous[1:], dtype=np.float64),
                np.asarray(signature[1:], dtype=np.float64),
                atol=geometry_tolerance,
                rtol=0.0,
            )
        )
        self._probabilistic_traversal_admission_safe_streak = (
            self._probabilistic_traversal_admission_safe_streak + 1
            if same_geometry
            else 1
        )
        self._probabilistic_traversal_admission_signature = signature
        return int(self._probabilistic_traversal_admission_safe_streak)

    def _probabilistic_traversal_retreat_candidate(self, state, reference):
        v_index = self.action_spec.index("v_cmd")
        omega_index = self.action_spec.index("omega_cmd")
        theta_index = self.state_spec.index("theta")
        commands = np.zeros(
            (self.config.horizon, self.action_spec.dimension),
            dtype=np.float64,
        )
        commands[:, v_index] = float(self.action_spec.lower[v_index])
        current_progress = float(getattr(reference, "progress", 0.0))
        frozen_tangent = self._probabilistic_traversal_retreat_target_tangent
        tangent = (
            float(np.arctan2(frozen_tangent[1], frozen_tangent[0]))
            if (
                self.config
                .probabilistic_obstacle_traversal_window_retreat_completion_frozen_frame_enabled
                and frozen_tangent is not None
            )
            else float(np.asarray(reference.poses_at_progress(
                np.asarray([current_progress], dtype=np.float64)
            ))[0, 2])
        )
        heading_error = float(np.arctan2(
            np.sin(tangent - state[theta_index]),
            np.cos(tangent - state[theta_index]),
        ))
        commands[0, omega_index] = float(np.clip(
            1.5 * heading_error,
            self.action_spec.lower[omega_index],
            self.action_spec.upper[omega_index],
        ))
        return commands

    def _probabilistic_traversal_hold_candidate(self):
        return np.zeros(
            (self.config.horizon, self.action_spec.dimension),
            dtype=np.float64,
        )

    def _probabilistic_traversal_prealign_candidate(self, sequence):
        """Keep a traversal candidate's steering while forbidding translation."""

        if sequence is None:
            return self._probabilistic_traversal_hold_candidate()
        result = np.asarray(sequence, dtype=np.float64).copy()
        expected = (self.config.horizon, self.action_spec.dimension)
        if result.shape != expected or not np.isfinite(result).all():
            return self._probabilistic_traversal_hold_candidate()
        result[:, self.action_spec.index("v_cmd")] = 0.0
        return result

    def _probabilistic_traversal_crossing(
        self, reference, probabilistic_obstacles, current_progress
    ):
        """Infer the nearest route crossing from causal forecast geometry."""

        candidates = []
        cross_track_limit = float(
            self.config.probabilistic_obstacle_traversal_window_cross_track_m
        )
        activation_distance = float(
            self.config
            .probabilistic_obstacle_traversal_window_activation_distance_m
        )
        for forecast_index, forecast in enumerate(probabilistic_obstacles):
            means = np.asarray(forecast.component_means, dtype=np.float64)
            weights = np.asarray(forecast.component_weights, dtype=np.float64)
            mixture_mean = np.sum(weights[..., None] * means, axis=1)
            projection = reference.project_batch(mixture_mean)
            closest_index = int(np.argmin(projection.cross_track_error))
            cross_track = float(projection.cross_track_error[closest_index])
            crossing_progress = float(projection.progress[closest_index])
            clearance = float(
                self.config.robot_radius
                + forecast.radius_m
                + self.config.probabilistic_obstacle_safety_margin
                + self.config
                .probabilistic_obstacle_traversal_window_clearance_margin_m
            )
            distance_ahead = crossing_progress - float(current_progress)
            # Route span over which this forecast is predicted to intrude, by the
            # same criterion the fixed radius stands in for. None when the
            # forecast never enters the tube.
            span_low = None
            span_high = None
            intruding = (
                np.asarray(projection.cross_track_error, dtype=np.float64)
                <= clearance
            )
            if bool(intruding.any()):
                intruding_progress = np.asarray(
                    projection.progress, dtype=np.float64
                )[intruding]
                span_low = float(intruding_progress.min())
                span_high = float(intruding_progress.max())
            if (
                cross_track <= cross_track_limit
                and distance_ahead >= -clearance
                and distance_ahead <= activation_distance
            ):
                candidates.append((
                    max(0.0, distance_ahead),
                    crossing_progress,
                    clearance,
                    int(forecast_index),
                    cross_track,
                    span_low,
                    span_high,
                ))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[4]))

    def _forecast_hazard_zones(self, reference, probabilistic_obstacles):
        """Route spans each forecast is predicted to intrude.

        Independent of any activation distance, and deliberately NOT derived from
        the detected crossing: crossing detection only fires within the traversal
        window's activation distance, which is why the earlier placement of the
        occupancy hold could never act early enough to hold before a zone.
        """

        zones = []
        for forecast in probabilistic_obstacles:
            means = np.asarray(forecast.component_means, dtype=np.float64)
            weights = np.asarray(forecast.component_weights, dtype=np.float64)
            mixture_mean = np.sum(weights[..., None] * means, axis=1)
            if mixture_mean.shape[0] <= 0:
                continue
            projection = reference.project_batch(mixture_mean)
            tube = float(
                self.config.robot_radius
                + forecast.radius_m
                + self.config.probabilistic_obstacle_safety_margin
            )
            inside = (
                np.asarray(projection.cross_track_error, dtype=np.float64)
                <= tube
            )
            if not bool(inside.any()):
                continue
            progress = np.asarray(projection.progress, dtype=np.float64)[inside]
            zones.append((float(progress.min()), float(progress.max())))
        return zones

    def _zone_occupied_within(
        self,
        reference,
        probabilistic_obstacles,
        zone_low,
        zone_high,
        step_start,
        step_end,
    ):
        """Will any forecast mean sit inside the route zone within `steps`?

        Purely geometric: projects each forecast's mixture mean onto the route
        and tests route-span membership plus cross-track proximity. No
        probability, no mixture upper bound, so it cannot saturate the way the
        collision-risk bound does.

        Returns (occupied, first_clear_step). ``occupied`` is True when the zone
        is blocked at any step in the window, in which case entering now risks
        being inside it when the obstacle arrives.
        """

        # A step RANGE, not a horizon. The zone is derived from a forecast that
        # intrudes there, so "occupied within N steps" is circular -- it is always
        # true. The decidable question is whether the zone is occupied during the
        # interval the robot would actually be inside it.
        lo_step = max(0, int(step_start))
        hi_step = int(step_end)
        if hi_step <= lo_step:
            return False, 0
        occupied_at = None
        for forecast in probabilistic_obstacles:
            means = np.asarray(forecast.component_means, dtype=np.float64)
            weights = np.asarray(forecast.component_weights, dtype=np.float64)
            mixture_mean = np.sum(weights[..., None] * means, axis=1)
            end = min(hi_step, mixture_mean.shape[0])
            if end <= lo_step:
                continue
            projection = reference.project_batch(mixture_mean[lo_step:end])
            progress = np.asarray(projection.progress, dtype=np.float64)
            cross = np.asarray(projection.cross_track_error, dtype=np.float64)
            tube = float(
                self.config.robot_radius
                + forecast.radius_m
                + self.config.probabilistic_obstacle_safety_margin
            )
            inside = (
                (progress >= float(zone_low))
                & (progress <= float(zone_high))
                & (cross <= tube)
            )
            if bool(inside.any()):
                first = lo_step + int(np.argmax(inside))
                occupied_at = first if occupied_at is None else min(
                    occupied_at, first
                )
        if occupied_at is None:
            return False, 0
        return True, occupied_at

    def _traversal_certificate_obstacles(
        self, probabilistic_obstacles, crossing_forecast_index
    ):
        """Obstacles the traversal commit certificate integrates risk over.

        Every obstacle by default, which is the historical behaviour. When
        commit_certificate_crossing_obstacle_only is set, just the obstacle whose
        crossing is being traversed -- but only if the index actually identifies
        one. An absent or out-of-range index falls back to the full set, because
        silently certifying against nothing would turn a missing index into a
        safety hole rather than an error.

        The traversal window is a per-crossing mechanism, but the certificate
        integrates risk over the union of every tracked obstacle. All three
        complex scenes carry three carriers, so what differs is how much of the
        horizon they sweep. Measured over the 50-seed confirmatory sets, per
        certificate evaluation:

            map        certs   median bound   ==1.0    full-horizon safe   commits
            chapter1   23954          0.156   25.3%                45.3%      7855
            chapter2    6110          0.213   16.0%                38.4%      1060
            chapter3    2869          1.000   47.4%                 7.4%        18

        On chapter 3 the union bound saturates at exactly 1.0 on nearly half of
        all evaluations, so the admission is unsatisfiable and the window commits
        18 times against chapter 1's 7855 -- the proposed method's central
        mechanism is suppressed by two orders of magnitude, and no ceiling value
        recovers it (raising the ceiling from 0.08 to 0.30 admits only 20.5%).

        Narrowing the certificate does not narrow global safety: the
        risk-augmented MPPI cost, the hard probability threshold, the residual
        safety shield and the emergency candidate layer all still consider every
        obstacle.
        """

        if not (
            self.config
            .probabilistic_obstacle_traversal_window_commit_certificate_crossing_obstacle_only
        ):
            return probabilistic_obstacles
        if crossing_forecast_index is None:
            return probabilistic_obstacles
        index = int(crossing_forecast_index)
        if not 0 <= index < len(probabilistic_obstacles):
            return probabilistic_obstacles
        return [probabilistic_obstacles[index]]

    def _probabilistic_traversal_candidate(
        self,
        state,
        reference,
        probabilistic_obstacles,
        current_progress,
        clear_progress,
        stop_at_target=False,
        hold_after_target=False,
        target_bearing_steering=False,
        goal_position=None,
        goal_tolerance=None,
        crossing_forecast_index=None,
    ):
        """Build and certify one fixed-budget route traversal proposal.

        The longer certificate is evaluated only for this deterministic
        proposal.  The main MPPI horizon and rollout count remain unchanged.

        ``crossing_forecast_index`` names the obstacle whose route crossing this
        traversal is for.  It only matters when
        commit_certificate_crossing_obstacle_only is set, in which case the risk
        integral is taken over that obstacle alone; see the config field for why.
        """

        horizon = min(
            int(
                self.config
                .probabilistic_obstacle_traversal_window_horizon_steps
            ),
            *(forecast.horizon for forecast in probabilistic_obstacles),
        )
        result = {
            "safe": False,
            "forecast_sufficient": False,
            "maximum_probability": 1.0,
            "probability_mass": float(horizon),
            "temporal_corroboration_maximum_probability": 1.0,
            "temporal_corroboration_probability_mass": float(horizon),
            "commit_admission_full_horizon_safe": False,
            "required_steps": 0,
            "sequence": None,
        }
        if horizon < self.config.horizon:
            return result
        v_index = self.action_spec.index("v_cmd")
        omega_index = self.action_spec.index("omega_cmd")
        theta_index = self.state_spec.index("theta")
        commands = np.zeros(
            (horizon, self.action_spec.dimension), dtype=np.float64
        )
        commands[:, v_index] = float(self.action_spec.upper[v_index])
        target_pose = np.asarray(
            reference.poses_at_progress(np.asarray([clear_progress]))[0],
            dtype=np.float64,
        )
        target_heading = float(target_pose[2])
        terminal_goal = (
            None
            if goal_position is None
            else np.asarray(goal_position, dtype=np.float64).reshape(-1)[:2]
        )
        if terminal_goal is not None and (
            terminal_goal.shape != (2,) or not np.isfinite(terminal_goal).all()
        ):
            return result
        if target_bearing_steering:
            position = np.asarray(state, dtype=np.float64)[
                list(self.state_spec.position_indices)
            ]
            target_delta = (
                terminal_goal - position
                if terminal_goal is not None
                else target_pose[:2] - position
            )
            if float(np.linalg.norm(target_delta)) > 1.0e-9:
                # A route tangent describes motion on the route, but it
                # cannot recover a robot that is laterally displaced from a
                # terminal endpoint.  Aim the deterministic certificate at
                # the endpoint itself so scalar projection progress cannot
                # masquerade as two-dimensional arrival.
                target_heading = float(np.arctan2(
                    target_delta[1], target_delta[0]
                ))
        heading_error = float(np.arctan2(
            np.sin(target_heading - state[theta_index]),
            np.cos(target_heading - state[theta_index]),
        ))
        maximum_yaw_rate = float(
            self.action_spec.upper[omega_index]
            if heading_error >= 0.0
            else -self.action_spec.lower[omega_index]
        )
        translation_heading_gate = float(
            self.config
            .probabilistic_obstacle_traversal_window_translation_heading_gate_rad
        )
        turn_steps = 0
        if abs(heading_error) > translation_heading_gate:
            turn_steps = int(np.ceil(
                (abs(heading_error) - translation_heading_gate)
                / max(maximum_yaw_rate * self.config.dt, 1.0e-12)
            ))
            turn_steps = max(1, min(turn_steps, horizon))
        if turn_steps:
            commands[:turn_steps, v_index] = 0.0
            commands[:turn_steps, omega_index] = float(np.clip(
                heading_error / max(turn_steps * self.config.dt, 1.0e-12),
                self.action_spec.lower[omega_index],
                self.action_spec.upper[omega_index],
            ))
        elif abs(heading_error) > 1.0e-9:
            commands[0, omega_index] = float(np.clip(
                1.5 * heading_error,
                self.action_spec.lower[omega_index],
                self.action_spec.upper[omega_index],
            ))
        def rollout_candidate(candidate_commands):
            prediction_controls = self._prediction_controls(
                candidate_commands[None, ...]
            )
            if bool(getattr(self.dynamics, "supports_rollout_batch", False)):
                return np.asarray(
                    self.dynamics.rollout_batch(
                        np.asarray(state, dtype=np.float64),
                        prediction_controls,
                        self.config.dt,
                        self.state_spec,
                        self.config.integrator,
                    ),
                    dtype=np.float64,
                )[0]
            candidate_trajectory = np.empty(
                (horizon + 1, self.state_spec.dimension), dtype=np.float64
            )
            candidate_trajectory[0] = state
            for step in range(horizon):
                candidate_trajectory[step + 1] = integrate_batch(
                    self.dynamics,
                    candidate_trajectory[step],
                    prediction_controls[0, step],
                    self.config.dt,
                    self.state_spec,
                    self.config.integrator,
                )
            return candidate_trajectory

        trajectory = rollout_candidate(commands)
        positions = trajectory[1:, list(self.state_spec.position_indices)]
        projected = reference.project_batch(
            positions, minimum_progress=float(current_progress)
        ).progress
        if terminal_goal is None:
            cleared = np.flatnonzero(
                projected >= float(clear_progress) - 1.0e-9
            )
        else:
            tolerance = (
                float(reference.tolerance)
                if goal_tolerance is None
                else float(goal_tolerance)
            )
            if not np.isfinite(tolerance) or tolerance <= 0.0:
                return result
            cleared = np.flatnonzero(
                np.linalg.norm(positions - terminal_goal[None, :], axis=1)
                <= tolerance
            )
        if not cleared.size:
            return result
        if hold_after_target:
            # The crossing certificate ends at the frozen clear exit.  Keep
            # the injected candidate inside that same transaction scope
            # instead of letting an unrelated maximum-speed tail reach later
            # static geometry and invalidate an otherwise certified crossing.
            reach_index = int(cleared[0])
            commands[reach_index + 1:, :] = 0.0
            trajectory = rollout_candidate(commands)
            positions = trajectory[
                1:, list(self.state_spec.position_indices)
            ]
            projected = reference.project_batch(
                positions, minimum_progress=float(current_progress)
            ).progress
            if terminal_goal is None:
                cleared = np.flatnonzero(
                    projected >= float(clear_progress) - 1.0e-9
                )
            else:
                tolerance = (
                    float(reference.tolerance)
                    if goal_tolerance is None
                    else float(goal_tolerance)
                )
                cleared = np.flatnonzero(
                    np.linalg.norm(
                        positions - terminal_goal[None, :], axis=1
                    )
                    <= tolerance
                )
            if not cleared.size:
                return result
        if stop_at_target:
            # Rearm staging may advance to the already frozen crossing entry,
            # but its open-loop certificate must not continue through the
            # conflict region.  Stop after the first predicted state reaches
            # that target, then re-evaluate the unchanged probabilistic
            # horizon.  Only the first command is executed before replanning.
            reach_index = int(cleared[0])
            commands[reach_index + 1:, :] = 0.0

            if terminal_goal is None:
                # The first maximum-speed command that reaches the entry can
                # overshoot it by one control interval.  Fit that one already
                # budgeted command to the greatest non-crossing speed.  The
                # existing 1e-9 geometric tolerance is used only for numerical
                # bisection; no behavioral threshold or rollout candidate is
                # introduced.
                maximum_reach_speed = float(commands[reach_index, v_index])
                lower_speed = 0.0
                upper_speed = maximum_reach_speed
                fitted_commands = commands.copy()
                for _ in range(40):
                    trial_speed = 0.5 * (lower_speed + upper_speed)
                    fitted_commands[reach_index, v_index] = trial_speed
                    trial_trajectory = rollout_candidate(fitted_commands)
                    trial_positions = trial_trajectory[
                        1:, list(self.state_spec.position_indices)
                    ]
                    trial_progress = reference.project_batch(
                        trial_positions,
                        minimum_progress=float(current_progress),
                    ).progress
                    if float(np.max(trial_progress)) <= (
                        float(clear_progress) + 1.0e-9
                    ):
                        lower_speed = trial_speed
                    else:
                        upper_speed = trial_speed
                fitted_commands[reach_index, v_index] = lower_speed
                commands = fitted_commands
            trajectory = rollout_candidate(commands)
            positions = trajectory[
                1:, list(self.state_spec.position_indices)
            ]
            projected = reference.project_batch(
                positions, minimum_progress=float(current_progress)
            ).progress
            if terminal_goal is None:
                cleared = np.flatnonzero(
                    projected >= float(clear_progress) - 1.0e-9
                )
                overshot = (
                    float(np.max(projected))
                    > float(clear_progress) + 1.0e-9
                )
            else:
                tolerance = (
                    float(reference.tolerance)
                    if goal_tolerance is None
                    else float(goal_tolerance)
                )
                cleared = np.flatnonzero(
                    np.linalg.norm(
                        positions - terminal_goal[None, :], axis=1
                    )
                    <= tolerance
                )
                overshot = False
            if not cleared.size or overshot:
                return result
        end_step = min(
            horizon,
            int(cleared[0])
            + 1
            + int(
                self.config
                .probabilistic_obstacle_traversal_window_clear_hold_steps
            ),
        )
        risk_end_step = end_step
        if (
            self.config
            .probabilistic_obstacle_traversal_window_temporal_abort_full_horizon_corroboration_enabled
            or self.config
            .probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled
        ):
            risk_end_step = max(
                end_step,
                min(horizon, int(self.config.horizon)),
            )
        certificate_obstacles = self._traversal_certificate_obstacles(
            probabilistic_obstacles, crossing_forecast_index
        )
        risk = evaluate_collision_risk(
            positions[None, :risk_end_step, :],
            certificate_obstacles,
            CollisionRiskConfig(
                robot_radius_m=self.config.robot_radius,
                safety_margin_m=(
                    self.config.probabilistic_obstacle_safety_margin
                ),
                minimum_position_std_m=(
                    self.config.probabilistic_obstacle_minimum_std
                ),
                hard_probability_threshold=(
                    self.config.probabilistic_obstacle_hard_threshold
                ),
            ),
        )
        short_step_probability = np.asarray(
            risk.step_probability_upper_bound[0, :end_step],
            dtype=np.float64,
        )
        maximum_probability = float(np.max(short_step_probability))
        probability_mass = float(np.sum(short_step_probability))
        temporal_corroboration_maximum_probability = float(
            risk.maximum_step_probability[0]
        )
        temporal_corroboration_probability_mass = float(
            risk.accumulated_probability_mass[0]
        )
        safe = bool(
            maximum_probability
            <= self.config
            .probabilistic_obstacle_traversal_window_probability_ceiling
            and probability_mass
            <= self.config
            .probabilistic_obstacle_traversal_window_mass_ceiling
        )
        commit_admission_full_horizon_safe = bool(
            temporal_corroboration_maximum_probability
            <= self.config
            .probabilistic_obstacle_traversal_window_probability_ceiling
            and temporal_corroboration_probability_mass
            <= self.config
            .probabilistic_obstacle_traversal_window_mass_ceiling
        )
        result.update({
            "safe": safe,
            "forecast_sufficient": True,
            "maximum_probability": maximum_probability,
            "probability_mass": probability_mass,
            "temporal_corroboration_maximum_probability": (
                temporal_corroboration_maximum_probability
            ),
            "temporal_corroboration_probability_mass": (
                temporal_corroboration_probability_mass
            ),
            "commit_admission_full_horizon_safe": (
                commit_admission_full_horizon_safe
            ),
            "required_steps": int(end_step),
            "sequence": commands[: self.config.horizon].copy(),
        })
        return result

    def _probabilistic_traversal_window_context(
        self,
        state,
        reference,
        probabilistic_obstacles,
        temporal_emergency_triggered=False,
        temporal_emergency_raw_triggered=False,
        temporal_emergency_closing_observed=False,
        temporal_emergency_ttc_s=float("inf"),
        terminal_phase=False,
    ):
        """Certify, start, and persist a causal crossing-window commit."""

        # Consecutive-step counter for the uncommitted staging hold. Cleared on
        # entry and restored only on a step that actually holds, so the count is
        # exactly the current unbroken run rather than a lifetime total.
        previous_uncommitted_hold_steps = (
            self._probabilistic_traversal_uncommitted_hold_steps
        )
        self._probabilistic_traversal_uncommitted_hold_steps = 0

        result = {
            "enabled": bool(
                self.config.probabilistic_obstacle_traversal_window_enabled
            ),
            "candidate_requested": False,
            "window_safe": False,
            "commit_active": False,
            "commit_started": False,
            "commit_completed": False,
            "commit_cancelled": False,
            "commit_cancelled_by_temporal_closing": False,
            "temporal_abort_current_hazard_signal": False,
            "commit_cancelled_by_temporal_midpoint_guard": False,
            "commit_cancelled_by_temporal_exit_deadline_guard": False,
            "commit_admission_full_horizon_safe": False,
            "commit_admission_rejected": False,
            "commit_admission_exit_deadline_safe": True,
            "commit_admission_exit_deadline_rejected": False,
            "commit_admission_exit_deadline_hold_requested": False,
            "commit_admission_prealign_requested": False,
            "uncommitted_temporal_staging_hold_requested": False,
            "uncommitted_hold_retreat_triggered": False,
            "zone_occupancy_hold_active": False,
            "uncommitted_temporal_staging_terminal_release_active": bool(
                self.config
                .probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_terminal_release_enabled
                and terminal_phase
            ),
            "terminal_phase": bool(terminal_phase),
            "terminal_target_bearing_steering_active": False,
            "terminal_capture_active": False,
            "terminal_capture_released": False,
            "commit_admission_exit_deadline_margin_s": 0.0,
            "commit_admission_safe_streak": 0,
            "commit_admission_required_streak": int(
                self.config
                .probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps
            ),
            "commit_admission_waiting": False,
            "commit_admission_released": False,
            "commit_preserved_for_nearest_safe_exit": False,
            "forward_exit_distance_m": 0.0,
            "retreat_exit_distance_m": 0.0,
            "forward_exit_optimistic_time_s": 0.0,
            "retreat_exit_optimistic_time_s": 0.0,
            "temporal_exit_deadline_ttc_s": float("inf"),
            "temporal_exit_deadline_guard_triggered": False,
            "temporal_closing_observed": bool(
                temporal_emergency_closing_observed
            ),
            "exit_deadline_retreat_escape_transaction_active": bool(
                self._probabilistic_traversal_exit_deadline_retreat_active
            ),
            "exit_deadline_retreat_escape_reused": False,
            "retreat_requested": False,
            "retreat_temporal_lattice_requested": bool(
                self._probabilistic_traversal_retreat_temporal_lattice
            ),
            "retreat_completed": False,
            "retreat_completion_projection_alias_rejected": False,
            "retreat_target_x": 0.0,
            "retreat_target_y": 0.0,
            "retreat_signed_distance_m": 0.0,
            "retreat_frozen_reference_progress": 0.0,
            "rearm_pending": bool(
                self._probabilistic_traversal_rearm_pending
            ),
            "rearm_no_crossing_safe_streak": 0,
            "rearm_released_by_no_crossing_clearance": False,
            "rearm_released_by_certified_handoff": False,
            "rearm_no_crossing_certified_handoff_active": False,
            "rearm_no_crossing_certified_handoff_safe": False,
            "rearm_staging_approach_requested": False,
            "rearm_staging_approach_safe": False,
            "rearm_staging_target_progress": 0.0,
            "post_center_forward_exit_commit_active": False,
            "retreat_progress": 0.0,
            "forecast_sufficient": False,
            "crossing_progress": 0.0,
            "entry_progress": 0.0,
            "clear_progress": 0.0,
            "current_progress": 0.0,
            "maximum_probability": 0.0,
            "probability_mass": 0.0,
            "required_steps": 0,
            "forecast_index": -1,
            "sequence": None,
        }
        if not result["enabled"]:
            self._clear_probabilistic_traversal_commit()
            self._probabilistic_traversal_retreat_progress = None
            self._probabilistic_traversal_retreat_target_position = None
            self._probabilistic_traversal_retreat_target_tangent = None
            self._probabilistic_traversal_retreat_reference = None
            self._probabilistic_traversal_rearm_pending = False
            self._probabilistic_traversal_retreat_temporal_lattice = False
            self._clear_probabilistic_traversal_exit_deadline_retreat_escape()
            return result
        if (
            self._probabilistic_traversal_rearm_pending
            and not probabilistic_obstacles
        ):
            self._probabilistic_traversal_admission_safe_streak = 0
            self._probabilistic_traversal_admission_signature = None
            result.update({
                "candidate_requested": True,
                "rearm_pending": True,
                "sequence": self._probabilistic_traversal_hold_candidate(),
            })
            return result
        required_contract = (
            probabilistic_obstacles
            and hasattr(reference, "project")
            and hasattr(reference, "project_batch")
            and hasattr(reference, "poses_at_progress")
            and hasattr(reference, "total_length")
            and "v_cmd" in self.action_spec.names
            and "omega_cmd" in self.action_spec.names
            and "theta" in self.state_spec.names
        )
        if not required_contract:
            if self._probabilistic_traversal_clear_progress is None:
                self._probabilistic_traversal_admission_safe_streak = 0
                self._probabilistic_traversal_admission_signature = None
            return result
        position = np.asarray(state, dtype=np.float64)[
            list(self.state_spec.position_indices)
        ]
        current_progress = float(reference.project(
            position,
            minimum_progress=float(getattr(reference, "progress", 0.0)),
        ).progress)
        result["current_progress"] = current_progress
        terminal_goal = (
            np.asarray(reference.points[-1], dtype=np.float64)[:2]
            if hasattr(reference, "points")
            else None
        )
        terminal_capture_released = bool(
            terminal_goal is None
            or float(np.linalg.norm(position - terminal_goal))
            <= float(getattr(reference, "tolerance", 0.0))
            or current_progress
            >= float(reference.total_length) - 1.0e-9
        )
        result["terminal_capture_released"] = terminal_capture_released
        if (
            self.config
            .probabilistic_obstacle_terminal_capture_candidate_enabled
            and terminal_phase
            and not terminal_capture_released
            and terminal_goal is not None
            and not temporal_emergency_triggered
            and not temporal_emergency_raw_triggered
            and not temporal_emergency_closing_observed
            and not self._probabilistic_traversal_rearm_pending
            and self._probabilistic_traversal_retreat_progress is None
            and self._probabilistic_traversal_clear_progress is None
        ):
            terminal_capture = self._probabilistic_traversal_candidate(
                state,
                reference,
                probabilistic_obstacles,
                current_progress,
                float(reference.total_length),
                stop_at_target=True,
                target_bearing_steering=True,
                goal_position=terminal_goal,
                goal_tolerance=float(reference.tolerance),
            )
            full_horizon_required = bool(
                self.config
                .probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled
            )
            terminal_capture_safe = bool(
                terminal_capture["safe"]
                and (
                    not full_horizon_required
                    or terminal_capture["commit_admission_full_horizon_safe"]
                )
                and terminal_capture["sequence"] is not None
            )
            if terminal_capture_safe:
                result.update({
                    "candidate_requested": True,
                    "window_safe": True,
                    "commit_admission_full_horizon_safe": bool(
                        terminal_capture[
                            "commit_admission_full_horizon_safe"
                        ]
                    ),
                    "forecast_sufficient": bool(
                        terminal_capture["forecast_sufficient"]
                    ),
                    "maximum_probability": float(
                        terminal_capture["maximum_probability"]
                    ),
                    "probability_mass": float(
                        terminal_capture["probability_mass"]
                    ),
                    "temporal_corroboration_maximum_probability": float(
                        terminal_capture[
                            "temporal_corroboration_maximum_probability"
                        ]
                    ),
                    "temporal_corroboration_probability_mass": float(
                        terminal_capture[
                            "temporal_corroboration_probability_mass"
                        ]
                    ),
                    "required_steps": int(terminal_capture["required_steps"]),
                    "clear_progress": float(reference.total_length),
                    "terminal_capture_active": True,
                    "sequence": terminal_capture["sequence"],
                })
                return result
        if self._probabilistic_traversal_retreat_progress is not None:
            # Retreat is the one traversal phase where physical motion is
            # intentionally allowed to move behind the monotonic online
            # reference progress.  Using ``current_progress`` here would pin
            # the completion check to that forward-only floor and latch the
            # controller in reverse forever after a pre-crossing abort.
            physical_progress = float(reference.project(position).progress)
            retreat_progress = float(
                self._probabilistic_traversal_retreat_progress
            )
            result["retreat_progress"] = retreat_progress
            projected_completion = bool(
                physical_progress <= retreat_progress + 1.0e-9
            )
            frozen_position = (
                self._probabilistic_traversal_retreat_target_position
            )
            frozen_tangent = (
                self._probabilistic_traversal_retreat_target_tangent
            )
            frozen_frame_active = bool(
                self.config
                .probabilistic_obstacle_traversal_window_retreat_completion_frozen_frame_enabled
                and frozen_position is not None
                and frozen_tangent is not None
            )
            retreat_signed_distance = (
                float(np.dot(
                    position - frozen_position,
                    frozen_tangent,
                ))
                if frozen_frame_active
                else 0.0
            )
            frozen_frame_completion = bool(
                retreat_signed_distance <= 1.0e-9
            )
            frozen_reference = self._probabilistic_traversal_retreat_reference
            frozen_reference_active = bool(
                self.config
                .probabilistic_obstacle_traversal_window_retreat_completion_frozen_reference_enabled
                and frozen_reference is not None
            )
            frozen_reference_progress = (
                float(frozen_reference.project(position).progress)
                if frozen_reference_active
                else 0.0
            )
            frozen_reference_completion = bool(
                frozen_reference_progress <= retreat_progress + 1.0e-9
            )
            retreat_completed = bool(
                frozen_reference_completion
                if frozen_reference_active
                else (
                    frozen_frame_completion
                    if frozen_frame_active
                    else projected_completion
                )
            )
            result.update({
                "retreat_completion_projection_alias_rejected": bool(
                    (
                        frozen_reference_active
                        and (
                            projected_completion
                            or (frozen_frame_active and frozen_frame_completion)
                        )
                        and not frozen_reference_completion
                    )
                    or (
                        not frozen_reference_active
                        and frozen_frame_active
                        and projected_completion
                        and not frozen_frame_completion
                    )
                ),
                "retreat_target_x": float(
                    frozen_position[0] if frozen_position is not None else 0.0
                ),
                "retreat_target_y": float(
                    frozen_position[1] if frozen_position is not None else 0.0
                ),
                "retreat_signed_distance_m": float(retreat_signed_distance),
                "retreat_frozen_reference_progress": float(
                    frozen_reference_progress
                ),
            })
            if retreat_completed:
                self._probabilistic_traversal_retreat_progress = None
                self._probabilistic_traversal_retreat_target_position = None
                self._probabilistic_traversal_retreat_target_tangent = None
                self._probabilistic_traversal_retreat_reference = None
                self._probabilistic_traversal_retreat_temporal_lattice = False
                self._clear_probabilistic_traversal_exit_deadline_retreat_escape()
                # The controller intentionally moved behind the otherwise
                # monotonic live path reference.  Rebase only at this explicit
                # retreat transaction boundary so the next certificate uses
                # the robot's real staging position and traversal duration.
                reference.progress = physical_progress
                current_progress = physical_progress
                result["current_progress"] = current_progress
                self._probabilistic_traversal_rearm_pending = True
                result["retreat_completed"] = True
                result["rearm_pending"] = True
            else:
                result.update({
                    "candidate_requested": True,
                    "retreat_requested": True,
                    "sequence": self._probabilistic_traversal_retreat_candidate(
                        state, reference
                    ),
                })
                return result
        active = self._probabilistic_traversal_clear_progress is not None
        if active and current_progress >= (
            float(self._probabilistic_traversal_clear_progress) - 1.0e-9
        ):
            result["commit_completed"] = True
            self._clear_probabilistic_traversal_commit()
            active = False
        # ---- zone-occupancy scheduling ---------------------------------
        # Runs BEFORE crossing detection, so it is not bounded by the traversal
        # window's activation distance. If a hazard zone lies ahead within the
        # lookahead and is predicted occupied during the interval the robot would
        # be transiting it, hold short of it. Forecast means and route geometry
        # only -- no risk bound, which saturates at 1.0 on 47% of certificates.
        if (
            self.config.probabilistic_obstacle_zone_occupancy_hold_enabled
            and not active
        ):
            speed = max(
                1.0e-3,
                float(
                    self.config
                    .probabilistic_obstacle_zone_occupancy_transit_speed_mps
                ),
            )
            step_m = speed * float(self.config.dt)
            lookahead = float(
                self.config.probabilistic_obstacle_zone_occupancy_lookahead_m
            )
            margin = int(
                self.config
                .probabilistic_obstacle_zone_occupancy_clear_margin_steps
            )
            ahead = [
                (lo, hi)
                for lo, hi in self._forecast_hazard_zones(
                    reference, probabilistic_obstacles
                )
                if lo > float(current_progress)
                and (lo - float(current_progress)) <= lookahead
            ]
            if ahead:
                zone_low, zone_high = min(ahead, key=lambda z: z[0])
                arrive = int(
                    np.floor((zone_low - float(current_progress)) / step_m)
                )
                clear = int(
                    np.ceil((zone_high - float(current_progress)) / step_m)
                ) + margin
                blocked, _first = self._zone_occupied_within(
                    reference,
                    probabilistic_obstacles,
                    zone_low,
                    zone_high,
                    arrive,
                    clear,
                )
                if blocked:
                    result.update({
                        "candidate_requested": True,
                        "zone_occupancy_hold_active": True,
                        "sequence": (
                            self._probabilistic_traversal_hold_candidate()
                        ),
                    })
                    return result
        crossing = None
        if active:
            crossing = (
                0.0,
                float(self._probabilistic_traversal_crossing_progress),
                float(
                    self._probabilistic_traversal_clear_progress
                    - self._probabilistic_traversal_crossing_progress
                ),
                -1,
                0.0,
                # An already-active traversal reuses its stored entry/clear
                # progress below, so no forecast span is needed here.
                None,
                None,
            )
        else:
            crossing = self._probabilistic_traversal_crossing(
                reference, probabilistic_obstacles, current_progress
            )
        if crossing is None:
            if self._probabilistic_traversal_rearm_pending:
                no_crossing_clear_streak = 0
                no_crossing_handoff = None
                no_crossing_handoff_safe = False
                if (
                    self.config
                    .probabilistic_obstacle_traversal_window_rearm_no_crossing_clear_enabled
                ):
                    certified_handoff_enabled = bool(
                        self.config
                        .probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_enabled
                    )
                    if certified_handoff_enabled:
                        v_index = self.action_spec.index("v_cmd")
                        handoff_progress = min(
                            float(reference.total_length),
                            current_progress
                            + max(
                                0.0, float(self.action_spec.upper[v_index])
                            ) * float(self.config.dt),
                        )
                        no_crossing_handoff = (
                            self._probabilistic_traversal_candidate(
                                state,
                                reference,
                                probabilistic_obstacles,
                                current_progress,
                                handoff_progress,
                            )
                        )
                        no_crossing_handoff_safe = bool(
                            not temporal_emergency_closing_observed
                            and no_crossing_handoff["safe"]
                            and no_crossing_handoff[
                                "commit_admission_full_horizon_safe"
                            ]
                        )
                    no_crossing_clear_streak = (
                        self._update_probabilistic_traversal_admission_streak(
                            safe=(
                                no_crossing_handoff_safe
                                if certified_handoff_enabled
                                else not temporal_emergency_closing_observed
                            ),
                            forecast_index=-1,
                            crossing_progress=current_progress,
                            entry_progress=current_progress,
                            clear_progress=(
                                handoff_progress
                                if certified_handoff_enabled
                                else current_progress
                            ),
                        )
                    )
                    required_streak = int(
                        self.config
                        .probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps
                    )
                    if no_crossing_clear_streak >= required_streak:
                        if certified_handoff_enabled:
                            result.update({
                                "candidate_requested": True,
                                "window_safe": bool(
                                    no_crossing_handoff["safe"]
                                ),
                                "commit_admission_full_horizon_safe": bool(
                                    no_crossing_handoff[
                                        "commit_admission_full_horizon_safe"
                                    ]
                                ),
                                "rearm_pending": True,
                                "rearm_no_crossing_safe_streak": int(
                                    no_crossing_clear_streak
                                ),
                                "rearm_no_crossing_certified_handoff_active": True,
                                "rearm_no_crossing_certified_handoff_safe": True,
                                "forecast_sufficient": bool(
                                    no_crossing_handoff[
                                        "forecast_sufficient"
                                    ]
                                ),
                                "maximum_probability": float(
                                    no_crossing_handoff[
                                        "maximum_probability"
                                    ]
                                ),
                                "probability_mass": float(
                                    no_crossing_handoff["probability_mass"]
                                ),
                                "temporal_corroboration_maximum_probability": float(
                                    no_crossing_handoff[
                                        "temporal_corroboration_maximum_probability"
                                    ]
                                ),
                                "temporal_corroboration_probability_mass": float(
                                    no_crossing_handoff[
                                        "temporal_corroboration_probability_mass"
                                    ]
                                ),
                                "required_steps": int(
                                    no_crossing_handoff["required_steps"]
                                ),
                                "sequence": no_crossing_handoff["sequence"],
                            })
                            return result
                        self._probabilistic_traversal_rearm_pending = False
                        self._probabilistic_traversal_admission_safe_streak = 0
                        self._probabilistic_traversal_admission_signature = None
                        result.update({
                            "rearm_pending": False,
                            "rearm_no_crossing_safe_streak": int(
                                no_crossing_clear_streak
                            ),
                            "rearm_released_by_no_crossing_clearance": True,
                        })
                        return result
                result.update({
                    "candidate_requested": True,
                    "rearm_pending": True,
                    "rearm_no_crossing_safe_streak": int(
                        no_crossing_clear_streak
                    ),
                    "rearm_no_crossing_certified_handoff_safe": bool(
                        no_crossing_handoff_safe
                    ),
                    "sequence": self._probabilistic_traversal_hold_candidate(),
                })
            elif not active:
                self._probabilistic_traversal_admission_safe_streak = 0
                self._probabilistic_traversal_admission_signature = None
                staging_hold_current_hazard = bool(
                    temporal_emergency_closing_observed
                )
                if (
                    staging_hold_current_hazard
                    and self.config
                    .probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_current_hazard_only_enabled
                ):
                    trigger_ttc = float(
                        self.config
                        .probabilistic_obstacle_emergency_candidate_trigger_ttc_s
                    )
                    staging_hold_current_hazard = bool(
                        temporal_emergency_raw_triggered
                        or (
                            trigger_ttc > 0.0
                            and np.isfinite(temporal_emergency_ttc_s)
                            and temporal_emergency_ttc_s <= trigger_ttc
                        )
                    )
                if (
                    self.config
                    .probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_enabled
                    and staging_hold_current_hazard
                    and not result[
                        "uncommitted_temporal_staging_terminal_release_active"
                    ]
                    and self._probabilistic_traversal_retreat_progress is None
                    and not self._probabilistic_traversal_exit_deadline_retreat_active
                ):
                    # A causal LaserScan closing signal may precede the
                    # constant-velocity forecast's recognition of a future
                    # route crossing after an obstacle reverses.  Preserve the
                    # route entrance by staging in place until online geometry
                    # either exposes a certifiable crossing or the closing
                    # signal clears.  This uses the existing traversal slot;
                    # no future plant truth, obstacle identity, or extra
                    # rollout is introduced.
                    held_steps = previous_uncommitted_hold_steps + 1
                    self._probabilistic_traversal_uncommitted_hold_steps = (
                        held_steps
                    )
                    retreat_after = int(
                        self.config
                        .probabilistic_obstacle_traversal_window_uncommitted_hold_retreat_steps
                    )
                    retreat_margin = float(
                        self.config
                        .probabilistic_obstacle_traversal_window_retreat_margin_m
                    )
                    if (
                        retreat_after > 0
                        and held_steps >= retreat_after
                        and retreat_margin > 0.0
                    ):
                        # Holding has stopped being a wait and become a stall:
                        # neither a certifiable crossing nor a clear closing
                        # signal has appeared in `retreat_after` consecutive
                        # steps, so staging in place is just standing at the
                        # route entrance while the carrier keeps closing.  Back
                        # off to a standoff instead and let the existing retreat
                        # executor drive it; that path already tolerates moving
                        # behind the monotonic online progress floor.
                        physical_progress = float(
                            reference.project(position).progress
                        )
                        self._probabilistic_traversal_retreat_progress = max(
                            0.0, physical_progress - retreat_margin
                        )
                        self._probabilistic_traversal_uncommitted_hold_steps = 0
                        result["uncommitted_hold_retreat_triggered"] = True
                    else:
                        result.update({
                            "candidate_requested": True,
                            "uncommitted_temporal_staging_hold_requested": True,
                            "sequence": (
                                self._probabilistic_traversal_hold_candidate()
                            ),
                        })
            return result
        (
            _,
            crossing_progress,
            clearance,
            forecast_index,
            _,
            crossing_span_low,
            crossing_span_high,
        ) = crossing
        entry_progress = max(0.0, crossing_progress - clearance)
        clear_progress = min(
            float(reference.total_length), crossing_progress + clearance
        )
        if (
            self.config
            .probabilistic_obstacle_traversal_window_forecast_extent_enabled
            and crossing_span_low is not None
            and crossing_span_high is not None
        ):
            # Union with the fixed-radius window: never narrower than before.
            entry_progress = max(0.0, min(entry_progress, crossing_span_low))
            clear_progress = min(
                float(reference.total_length),
                max(clear_progress, crossing_span_high),
            )
        if active:
            entry_progress = float(
                self._probabilistic_traversal_entry_progress
            )
            clear_progress = float(
                self._probabilistic_traversal_clear_progress
            )
        terminal_target_bearing_steering_active = bool(
            self.config
            .probabilistic_obstacle_traversal_window_terminal_target_bearing_enabled
            and terminal_phase
            and clear_progress
            >= float(reference.total_length) - 1.0e-9
            and current_progress >= crossing_progress - 1.0e-9
        )
        result["terminal_target_bearing_steering_active"] = bool(
            terminal_target_bearing_steering_active
        )
        certificate = self._probabilistic_traversal_candidate(
            state,
            reference,
            probabilistic_obstacles,
            current_progress,
            clear_progress,
            hold_after_target=bool(
                self.config
                .probabilistic_obstacle_traversal_window_commit_clear_hold_tail_enabled
            ),
            target_bearing_steering=bool(
                terminal_target_bearing_steering_active
            ),
            crossing_forecast_index=forecast_index,
        )
        rearm_staging_approach_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_rearm_staging_approach_enabled
            and self._probabilistic_traversal_rearm_pending
            and not active
            and current_progress < entry_progress - 1.0e-9
        )
        rearm_staging_candidate = None
        rearm_staging_approach_safe = False
        rearm_staging_target_progress = entry_progress
        if rearm_staging_approach_requested:
            if (
                self.config
                .probabilistic_obstacle_traversal_window_rearm_staging_frontier_enabled
            ):
                staging_v_index = self.action_spec.index("v_cmd")
                rearm_staging_target_progress = min(
                    entry_progress,
                    current_progress
                    + max(
                        0.0,
                        float(self.action_spec.upper[staging_v_index]),
                    ) * float(self.config.dt),
                )
            rearm_staging_candidate = (
                self._probabilistic_traversal_candidate(
                    state,
                    reference,
                    probabilistic_obstacles,
                    current_progress,
                    rearm_staging_target_progress,
                    stop_at_target=True,
                )
            )
            rearm_staging_approach_safe = bool(
                rearm_staging_candidate["safe"]
                and rearm_staging_candidate[
                    "commit_admission_full_horizon_safe"
                ]
                and rearm_staging_candidate["sequence"] is not None
            )
        started = bool(
            active
            and (
                self._probabilistic_traversal_commit_started
                or current_progress >= entry_progress - 1.0e-9
            )
        )
        abort_probability = float(
            self.config
            .probabilistic_obstacle_traversal_window_abort_probability
        )
        temporal_abort_mass_floor = float(
            self.config
            .probabilistic_obstacle_traversal_window_temporal_abort_mass_floor
        )
        temporal_abort_current_hazard_signal = bool(
            temporal_emergency_raw_triggered
            or (
                temporal_emergency_closing_observed
                and np.isfinite(temporal_emergency_ttc_s)
                and temporal_emergency_ttc_s > 0.0
                and self.config
                .probabilistic_obstacle_emergency_candidate_trigger_ttc_s
                > 0.0
                and temporal_emergency_ttc_s
                <= self.config
                .probabilistic_obstacle_emergency_candidate_trigger_ttc_s
            )
        )
        temporal_abort_signal = bool(
            temporal_abort_current_hazard_signal
            if self.config
            .probabilistic_obstacle_traversal_window_temporal_abort_current_hazard_only_enabled
            else temporal_emergency_triggered
        )
        temporal_abort = bool(
            temporal_abort_signal
            and certificate["forecast_sufficient"]
            and certificate["temporal_corroboration_probability_mass"]
            >= temporal_abort_mass_floor
        )
        full_horizon_admission_enabled = bool(
            self.config
            .probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled
        )
        admission_certificate_safe = bool(
            certificate["safe"]
            and (
                not full_horizon_admission_enabled
                or certificate["commit_admission_full_horizon_safe"]
            )
        )
        admission_v_index = self.action_spec.index("v_cmd")
        admission_forward_speed = max(
            1.0e-12, float(self.action_spec.upper[admission_v_index])
        )
        admission_forward_exit_optimistic_time = max(
            0.0, clear_progress - current_progress
        ) / admission_forward_speed
        admission_exit_deadline_rejected = bool(
            not active
            and self.config
            .probabilistic_obstacle_traversal_window_commit_admission_exit_deadline_enabled
            and temporal_emergency_closing_observed
            and np.isfinite(temporal_emergency_ttc_s)
            and temporal_emergency_ttc_s > 0.0
            and admission_forward_exit_optimistic_time
            >= temporal_emergency_ttc_s
        )
        admission_exit_deadline_safe = bool(
            not admission_exit_deadline_rejected
        )
        admission_exit_deadline_margin = (
            float(temporal_emergency_ttc_s)
            - admission_forward_exit_optimistic_time
            if temporal_emergency_closing_observed
            and np.isfinite(temporal_emergency_ttc_s)
            else 0.0
        )
        admission_safe = bool(
            admission_certificate_safe and admission_exit_deadline_safe
        )
        admission_rejected = bool(
            certificate["safe"]
            and (
                (
                    full_horizon_admission_enabled
                    and not certificate["commit_admission_full_horizon_safe"]
                )
                or admission_exit_deadline_rejected
            )
        )
        admission_required_streak = int(
            self.config
            .probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps
        )
        admission_safe_streak = 0
        if not active:
            admission_safe_streak = (
                self._update_probabilistic_traversal_admission_streak(
                    safe=admission_safe,
                    forecast_index=forecast_index,
                    crossing_progress=crossing_progress,
                    entry_progress=entry_progress,
                    clear_progress=clear_progress,
                )
            )
        admission_stable = bool(
            admission_safe
            and (
                active
                or admission_safe_streak >= admission_required_streak
            )
        )
        admission_waiting = bool(
            not active
            and admission_certificate_safe
            and not admission_stable
        )
        admission_exit_deadline_hold_requested = bool(
            not active and admission_exit_deadline_rejected
        )
        admission_hold_requested = bool(
            admission_waiting or admission_exit_deadline_hold_requested
        )
        admission_prealign_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_commit_admission_prealign_enabled
            and not active
            and not admission_stable
            and not temporal_emergency_raw_triggered
            and not self._probabilistic_traversal_rearm_pending
            and self._probabilistic_traversal_retreat_progress is None
            and certificate.get("sequence") is not None
        )
        admission_released = False
        retreat_margin = float(
            self.config
            .probabilistic_obstacle_traversal_window_retreat_margin_m
        )
        retreat_exit_progress = max(0.0, entry_progress - retreat_margin)
        forward_exit_distance = max(0.0, clear_progress - current_progress)
        retreat_exit_distance = max(
            0.0, current_progress - retreat_exit_progress
        )
        preserve_for_nearest_safe_exit = bool(
            active
            and started
            and current_progress < crossing_progress - 1.0e-9
            and temporal_abort
            and certificate["safe"]
            and self.config
            .probabilistic_obstacle_traversal_window_temporal_abort_nearest_exit_enabled
            and forward_exit_distance <= retreat_exit_distance
        )
        v_index = self.action_spec.index("v_cmd")
        midpoint_guard_band = max(
            0.0, float(self.action_spec.upper[v_index]) * self.config.dt
        )
        forward_speed = max(
            1.0e-12, float(self.action_spec.upper[v_index])
        )
        reverse_speed = max(
            1.0e-12, -float(self.action_spec.lower[v_index])
        )
        forward_exit_optimistic_time = (
            forward_exit_distance / forward_speed
        )
        retreat_exit_optimistic_time = (
            retreat_exit_distance / reverse_speed
        )
        temporal_exit_deadline_ttc = float(temporal_emergency_ttc_s)
        temporal_exit_deadline_guard_abort = bool(
            active
            and started
            and current_progress < crossing_progress - 1.0e-9
            and temporal_emergency_raw_triggered
            and np.isfinite(temporal_exit_deadline_ttc)
            and temporal_exit_deadline_ttc > 0.0
            and self.config
            .probabilistic_obstacle_traversal_window_temporal_exit_deadline_guard_enabled
            and min(
                forward_exit_optimistic_time,
                retreat_exit_optimistic_time,
            ) >= temporal_exit_deadline_ttc
        )
        temporal_midpoint_guard_abort = bool(
            active
            and started
            and current_progress < crossing_progress - 1.0e-9
            and temporal_abort_signal
            and certificate["forecast_sufficient"]
            and certificate["safe"]
            and self.config
            .probabilistic_obstacle_traversal_window_temporal_midpoint_guard_enabled
            and retreat_exit_distance <= forward_exit_distance
            and forward_exit_distance - retreat_exit_distance
            <= midpoint_guard_band + 1.0e-12
        )
        abort_started_commit = bool(
            active
            and started
            and current_progress < crossing_progress - 1.0e-9
            and not preserve_for_nearest_safe_exit
            and (
                temporal_abort
                or temporal_midpoint_guard_abort
                or temporal_exit_deadline_guard_abort
                or (
                    abort_probability > 0.0
                    and (
                        not certificate["forecast_sufficient"]
                        or certificate["maximum_probability"]
                        >= abort_probability
                    )
                )
            )
        )
        if abort_started_commit:
            result["commit_cancelled"] = True
            result["commit_cancelled_by_temporal_closing"] = bool(
                temporal_abort
                or temporal_midpoint_guard_abort
                or temporal_exit_deadline_guard_abort
            )
            result["commit_cancelled_by_temporal_midpoint_guard"] = bool(
                temporal_midpoint_guard_abort
            )
            result[
                "commit_cancelled_by_temporal_exit_deadline_guard"
            ] = bool(temporal_exit_deadline_guard_abort)
            if retreat_margin > 0.0:
                self._probabilistic_traversal_retreat_progress = max(
                    0.0, entry_progress - retreat_margin
                )
                if (
                    self.config
                    .probabilistic_obstacle_traversal_window_retreat_completion_frozen_frame_enabled
                ):
                    retreat_target_pose = np.asarray(
                        reference.poses_at_progress(np.asarray([
                            self._probabilistic_traversal_retreat_progress
                        ], dtype=np.float64))[0],
                        dtype=np.float64,
                    )
                    self._probabilistic_traversal_retreat_target_position = (
                        retreat_target_pose[:2].copy()
                    )
                    self._probabilistic_traversal_retreat_target_tangent = (
                        np.asarray([
                            np.cos(retreat_target_pose[2]),
                            np.sin(retreat_target_pose[2]),
                        ], dtype=np.float64)
                    )
                if (
                    self.config
                    .probabilistic_obstacle_traversal_window_retreat_completion_frozen_reference_enabled
                ):
                    # Online A* replaces the live route in place.  Preserve
                    # the complete route coordinate at the abort boundary so
                    # completion remains ordered on curved paths and cannot
                    # alias through either a new scalar origin or a local
                    # tangent half-plane.
                    self._probabilistic_traversal_retreat_reference = (
                        copy.deepcopy(reference)
                    )
                self._probabilistic_traversal_retreat_temporal_lattice = bool(
                    temporal_midpoint_guard_abort
                    or temporal_exit_deadline_guard_abort
                )
            else:
                self._probabilistic_traversal_retreat_target_position = None
                self._probabilistic_traversal_retreat_target_tangent = None
                self._probabilistic_traversal_retreat_reference = None
                self._probabilistic_traversal_retreat_temporal_lattice = False
            if (
                temporal_exit_deadline_guard_abort
                and self._probabilistic_traversal_retreat_progress is not None
            ):
                self._start_probabilistic_traversal_exit_deadline_retreat_escape()
            else:
                self._clear_probabilistic_traversal_exit_deadline_retreat_escape()
            self._probabilistic_traversal_rearm_pending = False
            self._clear_probabilistic_traversal_commit()
            active = False
            started = False
        elif active and not started and not admission_safe:
            result["commit_cancelled"] = True
            result["commit_admission_rejected"] = admission_rejected
            self._clear_probabilistic_traversal_commit()
            active = False
        elif not active and admission_stable:
            self._probabilistic_traversal_crossing_progress = crossing_progress
            self._probabilistic_traversal_entry_progress = entry_progress
            self._probabilistic_traversal_clear_progress = clear_progress
            self._probabilistic_traversal_commit_started = False
            self._probabilistic_traversal_rearm_pending = False
            active = True
            started = bool(current_progress >= entry_progress - 1.0e-9)
            admission_released = True
            self._probabilistic_traversal_admission_safe_streak = 0
            self._probabilistic_traversal_admission_signature = None
        requested = bool(active and (started or certificate["safe"]))
        if admission_hold_requested:
            requested = True
        if admission_prealign_requested:
            requested = True
        retreat_requested = bool(
            abort_started_commit
            and self._probabilistic_traversal_retreat_progress is not None
        )
        if retreat_requested:
            requested = True
        rearm_pending = bool(
            self._probabilistic_traversal_rearm_pending and not active
        )
        if rearm_pending:
            requested = True
        post_center_forward_exit_commit_active = bool(
            self.config
            .probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled
            and active
            and started
            and current_progress >= crossing_progress - 1.0e-9
            and current_progress < clear_progress - 1.0e-9
        )
        result.update({
            "candidate_requested": requested,
            "window_safe": bool(certificate["safe"]),
            "commit_admission_full_horizon_safe": bool(
                certificate["commit_admission_full_horizon_safe"]
            ),
            "commit_admission_rejected": bool(
                result["commit_admission_rejected"]
                or (not active and admission_rejected)
            ),
            "commit_admission_exit_deadline_safe": bool(
                admission_exit_deadline_safe
            ),
            "commit_admission_exit_deadline_rejected": bool(
                admission_exit_deadline_rejected
            ),
            "commit_admission_exit_deadline_hold_requested": bool(
                admission_exit_deadline_hold_requested
            ),
            "commit_admission_prealign_requested": bool(
                admission_prealign_requested
            ),
            "commit_admission_exit_deadline_margin_s": float(
                admission_exit_deadline_margin
            ),
            "commit_admission_safe_streak": int(admission_safe_streak),
            "commit_admission_required_streak": int(
                admission_required_streak
            ),
            "commit_admission_waiting": bool(admission_waiting),
            "commit_admission_released": bool(admission_released),
            "commit_preserved_for_nearest_safe_exit": bool(
                preserve_for_nearest_safe_exit
            ),
            "forward_exit_distance_m": float(forward_exit_distance),
            "retreat_exit_distance_m": float(retreat_exit_distance),
            "forward_exit_optimistic_time_s": float(
                forward_exit_optimistic_time
            ),
            "retreat_exit_optimistic_time_s": float(
                retreat_exit_optimistic_time
            ),
            "temporal_exit_deadline_ttc_s": float(
                temporal_exit_deadline_ttc
            ),
            "temporal_exit_deadline_guard_triggered": bool(
                temporal_exit_deadline_guard_abort
            ),
            "temporal_closing_observed": bool(
                temporal_emergency_closing_observed
            ),
            "temporal_abort_current_hazard_signal": bool(
                temporal_abort_current_hazard_signal
            ),
            "exit_deadline_retreat_escape_transaction_active": bool(
                self._probabilistic_traversal_exit_deadline_retreat_active
            ),
            "exit_deadline_retreat_escape_reused": False,
            "commit_active": bool(active),
            "commit_started": bool(started),
            "forecast_sufficient": bool(certificate["forecast_sufficient"]),
            "crossing_progress": float(crossing_progress),
            "entry_progress": float(entry_progress),
            "clear_progress": float(clear_progress),
            "maximum_probability": float(certificate["maximum_probability"]),
            "probability_mass": float(certificate["probability_mass"]),
            "temporal_corroboration_maximum_probability": float(
                certificate["temporal_corroboration_maximum_probability"]
            ),
            "temporal_corroboration_probability_mass": float(
                certificate["temporal_corroboration_probability_mass"]
            ),
            "required_steps": int(certificate["required_steps"]),
            "forecast_index": int(forecast_index),
            "retreat_requested": retreat_requested,
            "retreat_temporal_lattice_requested": bool(
                self._probabilistic_traversal_retreat_temporal_lattice
            ),
            "rearm_pending": rearm_pending,
            "rearm_staging_approach_requested": bool(
                rearm_staging_approach_requested
            ),
            "rearm_staging_approach_safe": bool(
                rearm_staging_approach_safe
            ),
            "rearm_staging_target_progress": float(
                rearm_staging_target_progress
            ),
            "post_center_forward_exit_commit_active": bool(
                post_center_forward_exit_commit_active
            ),
            "retreat_progress": float(
                self._probabilistic_traversal_retreat_progress
                if self._probabilistic_traversal_retreat_progress is not None
                else 0.0
            ),
            "sequence": (
                self._probabilistic_traversal_retreat_candidate(state, reference)
                if retreat_requested
                else rearm_staging_candidate["sequence"]
                if (
                    rearm_pending
                    and rearm_staging_approach_safe
                    and rearm_staging_candidate is not None
                )
                else self._probabilistic_traversal_hold_candidate()
                if rearm_pending
                else self._probabilistic_traversal_prealign_candidate(
                    certificate["sequence"]
                )
                if admission_prealign_requested
                else self._probabilistic_traversal_hold_candidate()
                if admission_hold_requested
                else certificate["sequence"]
            ),
        })
        return result

    def _inject_probabilistic_traversal_candidate(
        self, samples, traversal_context
    ):
        sequence = traversal_context.get("sequence")
        if (
            not traversal_context.get("candidate_requested", False)
            or sequence is None
        ):
            return -1
        # The final six slots belong to the existing emergency lattice.  Slot
        # K-7 is reserved from the same fixed K budget for the certified route
        # traversal proposal.
        index = int(self.config.num_samples - 7)
        samples[index] = np.asarray(sequence, dtype=np.float64)
        return index

    def _importance_sampling_cost(self, nominal, perturbations, covariance):
        """Return the MPPI likelihood-ratio correction for each rollout.

        For a Gaussian proposal with mean ``nominal`` and covariance Sigma,
        the practical MPPI correction is

            lambda * R * sum_t nominal_t^T Sigma^-1 epsilon_t,

        where this implementation uses ``control_weight`` as the scalar
        control-cost matrix R.  Keeping R explicit prevents a small sampling
        variance from unintentionally collapsing all weight onto one rollout.

        It is intentionally optional so archived experiments can reproduce the
        repository's former weighted-sampling behavior exactly.
        """
        if not self.config.importance_sampling_correction:
            return np.zeros(perturbations.shape[0], dtype=np.float64)
        inverse_covariance = np.linalg.inv(covariance)
        return self.config.temperature * self.config.control_weight * np.einsum(
            "ha,ab,kha->k", nominal, inverse_covariance, perturbations
        )

    def rollout(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        initial_state = self.state_spec.validate(
            np.asarray(initial_state, dtype=np.float64).reshape(-1)
        )
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 2:
            controls = controls[None, ...]
        expected_tail = (self.config.horizon, self.action_spec.dimension)
        if controls.ndim != 3 or controls.shape[1:] != expected_tail:
            raise ValueError(
                "rollout controls must have shape [K,%d,%d]"
                % expected_tail
            )
        if controls.shape[0] < 1 or not np.isfinite(controls).all():
            raise ValueError("rollout controls must contain a finite batch")
        prediction_controls = self._prediction_controls(controls)
        if bool(getattr(self.dynamics, "supports_rollout_batch", False)):
            trajectory = np.asarray(
                self.dynamics.rollout_batch(
                    initial_state,
                    prediction_controls,
                    self.config.dt,
                    self.state_spec,
                    self.config.integrator,
                ),
                dtype=np.float64,
            )
            expected_shape = (
                controls.shape[0],
                self.config.horizon + 1,
                self.state_spec.dimension,
            )
            if (
                trajectory.shape != expected_shape
                or not np.isfinite(trajectory).all()
            ):
                raise FloatingPointError(
                    "batch-rollout fast path returned an invalid trajectory"
                )
            return trajectory
        batch = controls.shape[0]
        trajectory = np.empty((batch, self.config.horizon + 1, self.state_spec.dimension), dtype=np.float64)
        trajectory[:, 0, :] = np.repeat(initial_state[None, :], batch, axis=0)
        for step in range(self.config.horizon):
            trajectory[:, step + 1, :] = integrate_batch(
                self.dynamics,
                trajectory[:, step, :],
                prediction_controls[:, step, :],
                self.config.dt,
                self.state_spec,
                self.config.integrator,
            )
        return trajectory

    def evaluate_control_sequences(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        target,
        obstacles: Iterable[Sequence[float]] = (),
        reference=None,
        probabilistic_obstacles=(),
    ) -> SequenceEvaluation:
        """Roll out and score a fixed, bounded batch without sampling it.

        The method is intended for counterfactual and model-ranking studies.
        It neither mutates receding-horizon state nor adds the proposal-density
        importance correction used internally by MPPI.
        """

        values = self._validate_evaluation_controls(controls)
        trajectories = self.rollout(initial_state, values)
        costs = self.cost_trajectories(
            trajectories,
            values,
            target,
            obstacles,
            reference=reference,
            probabilistic_obstacles=probabilistic_obstacles,
        )
        return SequenceEvaluation(trajectories=trajectories, costs=costs)

    def cost_trajectories(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        target,
        obstacles: Iterable[Sequence[float]] = (),
        reference=None,
        probabilistic_obstacles=(),
    ) -> np.ndarray:
        """Apply the controller's base cost contract to fixed trajectories."""

        values = self._validate_evaluation_controls(controls)
        paths = np.asarray(trajectories, dtype=np.float64)
        expected = (
            values.shape[0],
            self.config.horizon + 1,
            self.state_spec.dimension,
        )
        if paths.shape != expected:
            raise ValueError(
                "evaluation trajectories must have shape [K,%d,%d]"
                % (self.config.horizon + 1, self.state_spec.dimension)
            )
        if not np.isfinite(paths).all():
            raise ValueError("evaluation trajectories must be finite")
        costs = np.asarray(
            self._cost(
                paths,
                values,
                target,
                tuple(obstacles),
                reference=reference,
                probabilistic_obstacles=probabilistic_obstacles,
            ),
            dtype=np.float64,
        )
        if costs.shape != (values.shape[0],) or not np.isfinite(costs).all():
            raise FloatingPointError("trajectory cost produced NaN, Inf, or invalid shape")
        return costs

    def _validate_evaluation_controls(self, controls: np.ndarray) -> np.ndarray:
        values = np.asarray(controls, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, ...]
        expected_tail = (self.config.horizon, self.action_spec.dimension)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                "evaluation controls must have shape [K,%d,%d]"
                % expected_tail
            )
        if values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError("evaluation controls must contain a finite batch")
        tolerance = 1e-12
        if (
            np.any(values < self.action_spec.lower[None, None, :] - tolerance)
            or np.any(values > self.action_spec.upper[None, None, :] + tolerance)
        ):
            raise ValueError("evaluation controls exceed action bounds")
        return values

    def importance_weights(
        self,
        nominal: np.ndarray,
        candidates: np.ndarray,
        base_costs: np.ndarray,
        covariance: Optional[np.ndarray] = None,
    ) -> SequenceWeights:
        """Reweight an externally fixed proposal batch using MPPI semantics."""

        values = self._validate_evaluation_controls(candidates)
        nominal = np.asarray(nominal, dtype=np.float64)
        expected_nominal = (self.config.horizon, self.action_spec.dimension)
        if nominal.shape != expected_nominal or not np.isfinite(nominal).all():
            raise ValueError("nominal sequence has invalid shape or values")
        if (
            np.any(nominal < self.action_spec.lower[None, :] - 1e-12)
            or np.any(nominal > self.action_spec.upper[None, :] + 1e-12)
        ):
            raise ValueError("nominal sequence exceeds action bounds")
        costs = np.asarray(base_costs, dtype=np.float64).reshape(-1)
        if costs.shape != (values.shape[0],) or not np.isfinite(costs).all():
            raise ValueError("base costs must be one finite value per candidate")
        covariance = self._sampling_covariance(
            PriorOutput(nominal, covariance, {})
        )
        correction = self._importance_sampling_cost(
            nominal, values - nominal[None, :, :], covariance
        )
        adjusted = costs + correction
        beta = float(np.min(adjusted))
        exponent = np.clip(
            -(adjusted - beta) / self.config.temperature, -700.0, 0.0
        )
        weights = np.exp(exponent)
        weight_sum = float(np.sum(weights))
        if np.isfinite(weight_sum) and weight_sum > 0.0:
            # Do not floor a representable subnormal sum: doing so leaves all
            # weights near zero, and squaring them later produces an infinite
            # effective sample size. Exact normalization is stable here.
            weights /= weight_sum
        else:
            finite_costs = np.where(np.isfinite(costs), costs, np.inf)
            fallback_index = (
                int(np.argmin(finite_costs))
                if np.any(np.isfinite(finite_costs)) else 0
            )
            weights[:] = 0.0
            weights[fallback_index] = 1.0
        effective_sample_size = float(1.0 / np.sum(weights ** 2))
        if not np.isfinite(weights).all() or not np.isfinite(effective_sample_size):
            raise FloatingPointError("MPPI importance weights are not finite")

        return SequenceWeights(adjusted, weights, effective_sample_size)

    def _prediction_controls(self, controls: np.ndarray) -> np.ndarray:
        """Map commanded sequences to interval-average delayed commands.

        MuJoCo schedules each zero-order-held command after a known delay. For
        delays no longer than one controller interval, the interval average is
        exactly the convex combination of the preceding and current command.
        The preceding value for the first rollout step is the last command that
        passed safety arbitration.
        """

        values = np.asarray(controls, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != self.action_spec.dimension:
            raise ValueError("prediction controls must have shape [K,H,action_dim]")
        fraction = float(self.config.command_delay_s / self.config.dt)
        if fraction <= 1e-12:
            return values
        preceding = np.empty_like(values)
        preceding[:, 0, :] = self.previous_action
        preceding[:, 1:, :] = values[:, :-1, :]
        return fraction * preceding + (1.0 - fraction) * values

    def _known_static_map_clearance(
        self, trajectories, known_static_obstacles
    ):
        """Exact signed footprint clearance to frozen static geometry.

        Memoised on the exact content of ``trajectories`` and obstacle
        geometry. The function is pure -- geometry and trajectories in,
        signed clearance out -- and is called from
        eleven distinct sites, roughly 11.7 times per control step, of which
        49.3% repeat an identical trajectory array (measured over 25 steps:
        292 calls, 148 unique). Caching therefore removes about half the work
        while returning bit-identical values.

        The key is a digest of the array's bytes, never its identity: the
        planner reuses and mutates trajectory buffers in place, so an
        identity-keyed cache would return stale clearances.
        """

        xy = np.asarray(trajectories, dtype=np.float64)[
            ..., list(self.state_spec.position_indices)
        ]
        if not STATIC_CLEARANCE_CACHE_ENABLED:
            return self._known_static_map_clearance_uncached(
                xy, known_static_obstacles
            )
        cache = getattr(self, "_static_clearance_cache", None)
        if cache is None:
            cache = self._static_clearance_cache = {}
        contiguous = np.ascontiguousarray(xy)
        obstacle_digest = hashlib.blake2b(
            repr(tuple(known_static_obstacles)).encode("utf-8"),
            digest_size=16,
        ).digest()
        key = (
            contiguous.shape,
            hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest(),
            obstacle_digest,
        )
        hit = cache.get(key)
        if hit is not None:
            return hit
        minimum = self._known_static_map_clearance_uncached(
            xy, known_static_obstacles
        )
        # Bounded so a long episode cannot grow the cache without limit;
        # the reuse that matters is always within one control step.
        if len(cache) >= 64:
            cache.clear()
        minimum.setflags(write=False)
        cache[key] = minimum
        return minimum

    def _known_static_map_clearance_uncached(
        self, xy, known_static_obstacles
    ):
        """Signed footprint clearance geometry, with no memoisation.

        Split out so the cache can be disabled for A/B timing and so a
        regression test can assert the cached and uncached paths agree
        bit-for-bit.
        """

        minimum = np.full(xy.shape[:2], np.inf, dtype=np.float64)
        for obstacle in known_static_obstacles:
            kind = str(obstacle.get("type", "cylinder"))
            if kind == "box":
                center = np.asarray(
                    obstacle["position"][:2], dtype=np.float64
                )
                yaw = float(obstacle.get("yaw", 0.0))
                cosine = float(np.cos(yaw))
                sine = float(np.sin(yaw))
                relative = xy - center
                local_x = (
                    cosine * relative[..., 0]
                    + sine * relative[..., 1]
                )
                local_y = (
                    -sine * relative[..., 0]
                    + cosine * relative[..., 1]
                )
                half_size = np.asarray(
                    obstacle["size"][:2], dtype=np.float64
                )
                offset_x = np.abs(local_x) - half_size[0]
                offset_y = np.abs(local_y) - half_size[1]
                outside = np.hypot(
                    np.maximum(offset_x, 0.0),
                    np.maximum(offset_y, 0.0),
                )
                inside = np.minimum(
                    np.maximum(offset_x, offset_y), 0.0
                )
                surface_distance = outside + inside
            elif kind == "segment":
                start = np.asarray(
                    obstacle["start"][:2], dtype=np.float64
                )
                end = np.asarray(
                    obstacle["end"][:2], dtype=np.float64
                )
                vector = end - start
                denominator = float(np.dot(vector, vector))
                if denominator <= 1.0e-12:
                    centerline_distance = np.linalg.norm(
                        xy - start, axis=-1
                    )
                else:
                    fraction = np.clip(
                        np.sum((xy - start) * vector, axis=-1)
                        / denominator,
                        0.0,
                        1.0,
                    )
                    projection = (
                        start
                        + fraction[..., None] * vector
                    )
                    centerline_distance = np.linalg.norm(
                        xy - projection, axis=-1
                    )
                # Scene segment ``thickness`` is the full MuJoCo box width.
                # ``model_factory`` builds the geom with half-extent
                # ``0.5 * thickness``, and ``mujoco_plant``, ``scene_feasibility``
                # and ``static_astar`` all measure against that half width.
                # Subtracting the full thickness here treats the wall as twice
                # its true half-width, which is over-conservative by
                # ``0.5 * thickness`` (0.10-0.14 m on the chapter-1 maps).
                #
                # The corrected convention is gated so every frozen result
                # remains bit-reproducible under the default.
                segment_extent = float(obstacle.get("thickness", 0.10))
                if self.config.known_static_map_segment_half_thickness:
                    segment_extent *= 0.5
                surface_distance = centerline_distance - segment_extent
            elif kind == "cylinder":
                center = np.asarray(
                    obstacle["position"][:2], dtype=np.float64
                )
                surface_distance = (
                    np.linalg.norm(xy - center, axis=-1)
                    - float(obstacle.get("radius", 0.25))
                )
            else:
                raise ValueError(
                    "unsupported known static obstacle type: %s" % kind
                )
            minimum = np.minimum(
                minimum,
                surface_distance - self.config.robot_radius,
            )
        if not np.isfinite(minimum).all():
            raise FloatingPointError(
                "known static-map clearance contains NaN or Inf"
            )
        # Bounded so a long episode cannot grow the cache without limit; the
        # reuse that matters is always within one control step.
        return minimum


    def _cost(
        self,
        trajectories,
        controls,
        target,
        obstacles,
        reference=None,
        probabilistic_obstacles=(),
        known_static_obstacles=(),
        path_boundary_margins=None,
        probabilistic_risk=None,
        reference_authority=1.0,
    ):
        reference_authority = float(reference_authority)
        if (
            not np.isfinite(reference_authority)
            or not 0.0 <= reference_authority <= 1.0
        ):
            raise ValueError("reference authority must be within [0,1]")
        xy_indices = self.state_spec.position_indices
        xy = trajectories[..., list(xy_indices)]
        target_xy = np.asarray((target.pose.x, target.pose.y), dtype=np.float64)
        preview = getattr(reference, "preview_poses", None)
        path_preview_active = bool(
            self.config.path_preview_enabled and callable(preview)
        )
        if path_preview_active:
            initial_offset = float(getattr(reference, "lookahead_distance", 0.0))
            offsets = initial_offset + np.arange(
                self.config.horizon + 1, dtype=np.float64
            ) * self.config.dt * self.config.path_preview_speed_mps
            reference_poses = np.asarray(preview(offsets), dtype=np.float64)
            if reference_poses.shape != (self.config.horizon + 1, 3):
                raise ValueError("path preview must have shape [H+1,3]")
            distance_sq = np.sum(
                (xy - reference_poses[None, :, :2]) ** 2, axis=-1
            )
        else:
            reference_poses = None
            distance_sq = np.sum((xy - target_xy) ** 2, axis=-1)
        costs = (
            reference_authority
            * self.config.goal_running_weight
            * np.sum(distance_sq[:, 1:-1], axis=1)
        )
        costs += (
            reference_authority
            * self.config.goal_terminal_weight
            * distance_sq[:, -1]
        )
        if (
            self.config.probabilistic_reference_progress_weight > 0.0
            and reference is not None
            and all(
                hasattr(reference, name)
                for name in (
                    "points",
                    "segment_lengths",
                    "cumulative",
                    "progress",
                )
            )
        ):
            terminal_progress = self._reference_terminal_progress(
                xy[:, -1, :], reference
            )
            progress_gain = np.maximum(
                0.0,
                terminal_progress - float(reference.progress),
            )
            # Preserve a direction-of-travel incentive while the dynamic-risk
            # signal relaxes exact path/subgoal attraction. This is a soft
            # reference-cost term, not a discrete navigation decision.
            costs -= (
                self.config.probabilistic_reference_progress_weight
                * progress_gain
            )
        if (
            path_preview_active
            and self.config.path_preview_heading_weight > 0.0
            and "theta" in self.state_spec.names
        ):
            theta = trajectories[:, 1:, self.state_spec.index("theta")]
            heading_error = np.arctan2(
                np.sin(theta - reference_poses[None, 1:, 2]),
                np.cos(theta - reference_poses[None, 1:, 2]),
            )
            costs += (
                reference_authority
                * self.config.path_preview_heading_weight
                * np.sum(heading_error ** 2, axis=1)
            )
        if self.config.path_boundary_enabled:
            margins = path_boundary_margins
            if margins is None:
                margins = self._path_boundary_margins(
                    trajectories,
                    reference,
                )
            buffered_excess = np.maximum(
                0.0, self.config.path_boundary_buffer - margins[:, 1:]
            )
            costs += self.config.path_boundary_weight * np.sum(
                buffered_excess ** 2, axis=1
            )
            boundary_violation = np.any(margins[:, 1:] < 0.0, axis=1)
            costs += (
                self.config.path_boundary_violation_penalty
                * boundary_violation.astype(np.float64)
            )
        if target.heading_tolerance is not None and "theta" in self.state_spec.names:
            theta = trajectories[:, -1, self.state_spec.index("theta")]
            error = np.arctan2(np.sin(theta - target.pose.theta), np.cos(theta - target.pose.theta))
            costs += self.config.heading_weight * error ** 2
        if "v" in self.state_spec.names:
            terminal_v = trajectories[:, -1, self.state_spec.index("v")]
            costs += self.config.terminal_velocity_weight * terminal_v ** 2
        if "omega" in self.state_spec.names:
            terminal_omega = trajectories[:, -1, self.state_spec.index("omega")]
            costs += self.config.terminal_yaw_rate_weight * terminal_omega ** 2
        if (
            self.config.terminal_bearing_weight > 0.0
            and target.phase in ("terminal_approach", "terminal")
            and "theta" in self.state_spec.names
        ):
            final_dx = target_xy[0] - xy[:, -1, 0]
            final_dy = target_xy[1] - xy[:, -1, 1]
            final_distance = np.hypot(final_dx, final_dy)
            desired_heading = np.arctan2(final_dy, final_dx)
            final_theta = trajectories[:, -1, self.state_spec.index("theta")]
            bearing_error = np.arctan2(
                np.sin(final_theta - desired_heading),
                np.cos(final_theta - desired_heading),
            )
            # At the exact target the bearing is undefined; position success
            # should not impose an arbitrary world-frame orientation.
            bearing_error = np.where(final_distance > 1e-12, bearing_error, 0.0)
            costs += self.config.terminal_bearing_weight * bearing_error ** 2
        costs += self.config.control_weight * np.sum(controls ** 2, axis=(1, 2))
        previous = np.broadcast_to(
            self.previous_action[None, None, :],
            (controls.shape[0], 1, controls.shape[2]),
        )
        deltas = np.diff(controls, axis=1, prepend=previous)
        costs += self.config.control_rate_weight * np.sum(deltas ** 2, axis=(1, 2))
        for obstacle in obstacles:
            ox, oy = float(obstacle[0]), float(obstacle[1])
            radius = float(obstacle[2]) if len(obstacle) >= 3 else 0.08
            clearance = np.sqrt((xy[..., 0] - ox) ** 2 + (xy[..., 1] - oy) ** 2)
            clearance -= radius + self.config.robot_radius
            collision = np.any(clearance <= 0.0, axis=1)
            influence = np.maximum(0.0, self.config.obstacle_influence - clearance)
            costs += self.config.obstacle_weight * np.sum(influence ** 2, axis=1)
            costs += self.config.collision_penalty * collision.astype(np.float64)
        if (
            self.config.known_static_map_cost_enabled
            and known_static_obstacles
        ):
            static_clearance = self._known_static_map_clearance(
                trajectories, known_static_obstacles
            )
            static_influence = np.maximum(
                0.0,
                self.config.known_static_map_influence_m
                - static_clearance,
            )
            costs += self.config.known_static_map_weight * np.sum(
                static_influence[:, 1:] ** 2, axis=1
            )
            static_collision = np.any(
                static_clearance[:, 1:] <= 0.0, axis=1
            )
            costs += (
                self.config.known_static_map_collision_penalty
                * static_collision.astype(np.float64)
            )
        if (
            self.config.probabilistic_obstacle_risk_enabled
            and probabilistic_obstacles
        ):
            risk = probabilistic_risk
            if risk is None:
                risk = self._probabilistic_collision_risk(
                    trajectories, probabilistic_obstacles
                )
            costs += (
                self.config.probabilistic_obstacle_risk_weight
                * risk.accumulated_probability_mass
            )
            costs += (
                self.config.probabilistic_obstacle_hard_penalty
                * risk.hard_violation.astype(np.float64)
            )
        if self.memory_cost is not None:
            for index in range(trajectories.shape[0]):
                costs[index] += float(self.memory_cost(trajectories[index], controls[index]))
        return costs

    def _reference_terminal_progress(self, positions, reference):
        """Vectorized final-state progress on a soft polyline reference."""

        values = np.asarray(positions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                "reference progress positions must have shape [K,2]"
            )
        starts = np.asarray(reference.points[:-1], dtype=np.float64)
        vectors = np.diff(
            np.asarray(reference.points, dtype=np.float64), axis=0
        )
        lengths = np.asarray(
            reference.segment_lengths, dtype=np.float64
        )
        cumulative = np.asarray(
            reference.cumulative[:-1], dtype=np.float64
        )
        relative = values[:, None, :] - starts[None, :, :]
        fractions = np.sum(
            relative * vectors[None, :, :], axis=2
        )
        fractions = np.clip(
            fractions / lengths[None, :] ** 2, 0.0, 1.0
        )
        projections = (
            starts[None, :, :]
            + fractions[..., None] * vectors[None, :, :]
        )
        squared_distances = np.sum(
            (projections - values[:, None, :]) ** 2, axis=2
        )
        candidate_progress = (
            cumulative[None, :] + fractions * lengths[None, :]
        )
        floor = float(reference.progress)
        backtrack = float(
            getattr(reference, "projection_backtrack_distance", 0.0)
        )
        forward = float(
            getattr(reference, "projection_forward_distance", float("inf"))
        )
        admissible = (
            candidate_progress >= floor - backtrack
        ) & (
            candidate_progress <= floor + forward
        )
        has_admissible = np.any(admissible, axis=1)
        nearest = np.argmin(
            np.where(admissible, squared_distances, np.inf), axis=1
        )
        projected = candidate_progress[
            np.arange(values.shape[0]), nearest
        ]
        projected = np.where(has_admissible, projected, floor)
        return np.maximum(floor, projected)

    def _probabilistic_collision_risk(
        self, trajectories, probabilistic_obstacles
    ):
        forecasts = tuple(probabilistic_obstacles)
        if not forecasts:
            raise ValueError(
                "probabilistic obstacle risk is enabled but forecasts are absent"
            )
        if any(
            not isinstance(item, GaussianMixtureObstacleForecast)
            for item in forecasts
        ):
            raise TypeError(
                "probabilistic obstacle forecasts use an invalid contract"
            )
        for forecast in forecasts:
            if not np.isclose(
                forecast.dt,
                self.config.dt,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "probabilistic obstacle forecast dt must match MPPI dt"
                )
        xy = np.asarray(trajectories, dtype=np.float64)[
            :, 1:, list(self.state_spec.position_indices)
        ]
        risk_config = CollisionRiskConfig(
            robot_radius_m=self.config.robot_radius,
            safety_margin_m=(
                self.config.probabilistic_obstacle_safety_margin
            ),
            minimum_position_std_m=(
                self.config.probabilistic_obstacle_minimum_std
            ),
            hard_probability_threshold=(
                self.config.probabilistic_obstacle_hard_threshold
            ),
        )
        if self.config.probabilistic_obstacle_risk_cuda_enabled:
            self._last_probabilistic_risk_backend = "cuda"
            return evaluate_collision_risk_cuda(
                xy,
                forecasts,
                risk_config,
                device=self.config.probabilistic_obstacle_risk_cuda_device,
            )
        self._last_probabilistic_risk_backend = "cpu"
        return evaluate_collision_risk(xy, forecasts, risk_config)

    @staticmethod
    def _emergency_first_step_boundary_handoff(
        full_horizon_eligible,
        first_step_eligible,
        emergency_candidate_mask,
        *,
        enabled,
    ):
        """Admit only emergency candidates whose immediate step is bounded."""

        full = np.asarray(full_horizon_eligible, dtype=bool)
        first = np.asarray(first_step_eligible, dtype=bool)
        emergency = np.asarray(emergency_candidate_mask, dtype=bool)
        if (
            full.ndim != 1
            or first.shape != full.shape
            or emergency.shape != full.shape
        ):
            raise ValueError("emergency boundary handoff masks differ")
        admitted = np.zeros(full.shape, dtype=bool)
        result = full.copy()
        if enabled:
            admitted = emergency & first & ~full
            result[admitted] = True
        return result, admitted

    def _apply_probabilistic_obstacle_action_guard(
        self,
        state,
        action,
        sequence,
        trajectory,
        samples,
        costs,
        probabilistic_obstacles,
        candidate_eligible=None,
        candidate_risk=None,
        emergency_candidate_mask=None,
        temporal_emergency_triggered=False,
        traversal_context=None,
        traversal_candidate_index=-1,
        candidate_static_clearance=None,
    ):
        """Apply the deployed final probability-risk action contract.

        Alternative optimizers may generate and weight candidates differently,
        but the command leaving MPPI must retain the same active-avoidance,
        stop-is-safest, fail-closed and speed-governor semantics as standard
        MPPI.  This helper does not sample or add candidates; it only selects
        from the optimizer's already-budgeted final candidate batch.
        """

        if not self.config.probabilistic_obstacle_risk_enabled:
            return action, sequence, trajectory, {
                "probabilistic_obstacle_risk_enabled": False,
                "probabilistic_obstacle_forecast_count": 0,
                "probabilistic_obstacle_probability_mass": 0.0,
                "probabilistic_obstacle_union_bound": 0.0,
                "probabilistic_obstacle_maximum_step_probability": 0.0,
                "probabilistic_obstacle_hard_violation": False,
                "probabilistic_obstacle_fail_closed": False,
                "probabilistic_obstacle_speed_scale": 1.0,
                "probabilistic_obstacle_initial_hard_violation": False,
                "probabilistic_obstacle_candidate_feasible_fraction": 1.0,
                "probabilistic_obstacle_active_fallback_used": False,
                "probabilistic_obstacle_active_fallback_index": -1,
                "probabilistic_obstacle_active_fallback_kind": "none",
                "probabilistic_obstacle_risk_equivalent_forward_progress_applied": False,
                "probabilistic_obstacle_stop_maximum_probability": 0.0,
                "probabilistic_obstacle_stop_probability_mass": 0.0,
                "probabilistic_obstacle_stopping_feasibility_enabled": False,
                "probabilistic_obstacle_emergency_candidate_count": 0,
                "probabilistic_obstacle_emergency_candidate_selected": False,
                "probabilistic_obstacle_emergency_candidate_direct_fallback_enabled": bool(
                    self.config
                    .probabilistic_obstacle_emergency_candidate_direct_fallback_enabled
                ),
                "probabilistic_obstacle_emergency_candidate_direct_fallback_suppressed": False,
                "probabilistic_obstacle_temporal_emergency_triggered": False,
                "probabilistic_obstacle_temporal_emergency_vetted": False,
                "probabilistic_obstacle_terminal_intent_safe_stop_selected": False,
                "probabilistic_obstacle_traversal_window_enabled": False,
                "probabilistic_obstacle_traversal_terminal_target_bearing_steering_active": False,
                "probabilistic_obstacle_terminal_capture_candidate_active": False,
                "probabilistic_obstacle_traversal_candidate_selected": False,
                "probabilistic_obstacle_traversal_candidate_geometry_eligible": False,
                "probabilistic_obstacle_traversal_candidate_static_minimum_clearance": 0.0,
                "probabilistic_obstacle_traversal_temporal_abort_current_hazard_signal": False,
                "probabilistic_obstacle_traversal_temporal_corroboration_enabled": False,
                "probabilistic_obstacle_traversal_temporal_corroboration_maximum_probability": 0.0,
                "probabilistic_obstacle_traversal_temporal_corroboration_probability_mass": 0.0,
                "probabilistic_obstacle_traversal_commit_admission_full_horizon_enabled": False,
                "probabilistic_obstacle_traversal_commit_admission_full_horizon_safe": False,
                "probabilistic_obstacle_traversal_commit_admission_rejected": False,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_enabled": False,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_safe": True,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected": False,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested": False,
                "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested": False,
                "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected": False,
                "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override": False,
                "probabilistic_obstacle_traversal_uncommitted_temporal_staging_terminal_release_active": False,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override": False,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered": False,
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_margin_s": 0.0,
                "probabilistic_obstacle_traversal_commit_admission_safe_streak": 0,
                "probabilistic_obstacle_traversal_commit_admission_required_streak": 1,
                "probabilistic_obstacle_traversal_commit_admission_waiting": False,
                "probabilistic_obstacle_traversal_commit_admission_released": False,
                "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_midpoint_guard": False,
                "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard": False,
                "probabilistic_obstacle_traversal_commit_preserved_for_nearest_safe_exit": False,
                "probabilistic_obstacle_traversal_retreat_temporal_lattice_requested": False,
                "probabilistic_obstacle_traversal_retreat_completion_projection_alias_rejected": False,
                "probabilistic_obstacle_traversal_retreat_target_x": 0.0,
                "probabilistic_obstacle_traversal_retreat_target_y": 0.0,
                "probabilistic_obstacle_traversal_retreat_signed_distance_m": 0.0,
                "probabilistic_obstacle_traversal_forward_exit_distance_m": 0.0,
                "probabilistic_obstacle_traversal_retreat_exit_distance_m": 0.0,
                "probabilistic_obstacle_traversal_forward_exit_optimistic_time_s": 0.0,
                "probabilistic_obstacle_traversal_retreat_exit_optimistic_time_s": 0.0,
                "probabilistic_obstacle_traversal_temporal_exit_deadline_ttc_s": 0.0,
                "probabilistic_obstacle_traversal_temporal_exit_deadline_guard_triggered": False,
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active": False,
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched": False,
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused": False,
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_v": 0.0,
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_omega": 0.0,
                "probabilistic_obstacle_traversal_retreat_overridden_by_hard_risk": False,
                "probabilistic_obstacle_traversal_retreat_overridden_by_temporal_midpoint_guard": False,
                "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk": False,
                "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_release_condition_met": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_requested": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_applied": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_candidate_count": 0,
                "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_requested": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count": 0,
                "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_selected": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_requested": False,
                "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_admitted_count": 0,
                "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_selected": False,
                "probabilistic_obstacle_traversal_rearm_staging_approach_requested": False,
                "probabilistic_obstacle_traversal_rearm_staging_approach_safe": False,
                "probabilistic_obstacle_traversal_rearm_staging_target_progress": 0.0,
                "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested": False,
                "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied": False,
                "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count": 0,
                "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered": False,
                "probabilistic_obstacle_traversal_post_center_temporal_raw_triggered": False,
                "probabilistic_obstacle_traversal_temporal_closing_observed": False,
                "probabilistic_obstacle_traversal_post_center_temporal_escape_latched": False,
                "probabilistic_obstacle_traversal_post_center_temporal_escape_reused": False,
                "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_v": 0.0,
                "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_omega": 0.0,
                "probabilistic_obstacle_active_avoidance_enabled": False,
            }

        forecasts = tuple(probabilistic_obstacles)
        values = np.asarray(samples, dtype=np.float64)
        candidate_costs = np.asarray(costs, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] < 1:
            raise ValueError("probabilistic action guard requires candidates")
        if candidate_costs.shape != (values.shape[0],):
            raise ValueError("probabilistic action guard cost shape differs")
        if candidate_eligible is None:
            eligible_mask = np.ones(values.shape[0], dtype=bool)
        else:
            eligible_mask = np.asarray(candidate_eligible, dtype=bool)
            if eligible_mask.shape != (values.shape[0],):
                raise ValueError(
                    "probabilistic action guard eligibility shape differs"
                )
        if emergency_candidate_mask is None:
            emergency_mask = np.zeros(values.shape[0], dtype=bool)
        else:
            emergency_mask = np.asarray(
                emergency_candidate_mask, dtype=bool
            )
            if emergency_mask.shape != (values.shape[0],):
                raise ValueError(
                    "probabilistic emergency candidate mask shape differs"
                )
        static_clearance_trace = None
        if candidate_static_clearance is not None:
            static_clearance_trace = np.asarray(
                candidate_static_clearance, dtype=np.float64
            )
            if (
                static_clearance_trace.ndim != 2
                or static_clearance_trace.shape[0] != values.shape[0]
            ):
                raise ValueError(
                    "probabilistic candidate static-clearance shape differs"
                )

        selected_risk = self._probabilistic_collision_risk(
            np.asarray(trajectory, dtype=np.float64)[None, ...], forecasts
        )
        initial_hard_violation = bool(selected_risk.hard_violation[0])
        fallback_used = False
        fallback_candidate_index = -1
        fallback_kind = "none"
        risk_equivalent_forward_progress_applied = False
        stop_maximum_probability = 0.0
        stop_probability_mass = 0.0
        fail_closed = False
        hard_action = self.config.probabilistic_obstacle_hard_violation_action
        evaluated_candidates = candidate_risk
        risk_candidate_feasible = np.ones(values.shape[0], dtype=bool)
        temporal_emergency_vetted = False
        terminal_intent_safe_stop_selected = False
        pareto_forward_commit_applied = False
        low_risk_forward_commit_applied = False
        emergency_feasibility_over_direction_applied = False
        preferred_emergency_probability = 0.0
        preferred_emergency_probability_mass = 0.0
        preferred_emergency_hard_violation = False
        best_cost_emergency_index = -1
        minimum_risk_emergency_index = -1
        traversal_context = dict(traversal_context or {})
        traversal_index = int(traversal_candidate_index)
        terminal_capture_active = bool(
            traversal_context.get("terminal_capture_active", False)
        )
        traversal_candidate_geometry_eligible = bool(
            0 <= traversal_index < values.shape[0]
            and eligible_mask[traversal_index]
        )
        traversal_candidate_static_minimum_clearance = (
            float(np.min(static_clearance_trace[traversal_index]))
            if (
                static_clearance_trace is not None
                and 0 <= traversal_index < values.shape[0]
            )
            else 0.0
        )
        traversal_selected = False
        traversal_retreat_overridden_by_hard_risk = False
        traversal_retreat_overridden_by_temporal_midpoint_guard = False
        traversal_commit_overridden_by_post_center_hard_risk = False
        traversal_commit_overridden_by_post_center_temporal_risk = False
        traversal_admission_exit_deadline_hold_overridden_by_hard_risk = False
        traversal_uncommitted_temporal_staging_hold_overridden_by_hard_risk = False
        traversal_admission_exit_deadline_forward_lattice_filtered = False
        traversal_rearm_hold_overridden_by_hard_risk = False
        traversal_rearm_hold_overridden_by_temporal_closing = False
        traversal_temporal_retreat_post_intent_forward_filter_requested = False
        traversal_temporal_retreat_post_intent_forward_lattice_filtered = False
        traversal_temporal_midpoint_retreat_forward_filter_requested = False
        traversal_temporal_midpoint_retreat_forward_lattice_filtered = False
        traversal_post_center_low_ttc_continuity_requested = False
        traversal_post_center_low_ttc_continuity_forward_lattice_filtered = False
        traversal_post_center_low_ttc_continuity_hard_risk_fallback = False
        traversal_post_center_low_ttc_continuity_release_condition_met = False
        traversal_post_center_forward_exit_lattice_filtered = False
        traversal_post_center_forward_exit_commit_requested = bool(
            traversal_context.get(
                "post_center_forward_exit_commit_requested", False
            )
        )
        traversal_post_center_forward_exit_coverage_applied = bool(
            getattr(
                self,
                "_probabilistic_emergency_forward_exit_coverage_applied",
                False,
            )
        )
        traversal_post_center_forward_exit_candidate_count = (
            int(np.sum(
                emergency_mask
                & (
                    values[:, 0, self.action_spec.index("v_cmd")]
                    >= 0.0
                )
            ))
            if "v_cmd" in self.action_spec.names
            else 0
        )
        traversal_post_center_low_ttc_nonforward_coverage_requested = bool(
            traversal_context.get(
                "post_center_low_ttc_nonforward_coverage_requested", False
            )
        )
        traversal_post_center_low_ttc_nonforward_coverage_applied = bool(
            getattr(
                self,
                "_probabilistic_emergency_nonforward_coverage_applied",
                False,
            )
        )
        traversal_post_center_low_ttc_nonforward_candidate_count = (
            int(np.sum(
                emergency_mask
                & (
                    values[:, 0, self.action_spec.index("v_cmd")]
                    <= 0.0
                )
            ))
            if "v_cmd" in self.action_spec.names
            else 0
        )
        traversal_post_center_low_ttc_first_step_boundary_handoff_requested = bool(
            traversal_context.get(
                "post_center_low_ttc_first_step_boundary_handoff_requested",
                False,
            )
        )
        first_step_boundary_handoff_mask = np.asarray(
            traversal_context.get(
                "_post_center_low_ttc_first_step_boundary_handoff_mask",
                np.zeros(values.shape[0], dtype=bool),
            ),
            dtype=bool,
        )
        if first_step_boundary_handoff_mask.shape != (values.shape[0],):
            raise ValueError("first-step boundary handoff mask shape differs")
        traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count = int(
            np.sum(first_step_boundary_handoff_mask)
        )
        traversal_post_center_low_ttc_prefix_boundary_handoff_requested = bool(
            traversal_context.get(
                "post_center_low_ttc_prefix_boundary_handoff_requested",
                False,
            )
        )
        exit_deadline_retreat_escape_transaction_active = bool(
            traversal_context.get(
                "exit_deadline_retreat_escape_transaction_active", False
            )
        )
        exit_deadline_retreat_escape_latched = False
        traversal_started = bool(
            traversal_context.get("commit_started", False)
        )

        if (
            traversal_context.get("candidate_requested", False)
            and 0 <= traversal_index < values.shape[0]
            and eligible_mask[traversal_index]
        ):
            if evaluated_candidates is None:
                candidate_trajectories = self.rollout(state, values)
                evaluated_candidates = self._probabilistic_collision_risk(
                    candidate_trajectories, forecasts
                )
            risk_candidate_feasible = ~evaluated_candidates.hard_violation
            retreat_requested = bool(
                traversal_context.get("retreat_requested", False)
            )
            rearm_pending = bool(
                traversal_context.get("rearm_pending", False)
            )
            traversal_rearm_hold_overridden_by_hard_risk = bool(
                rearm_pending
                and self.config
                .probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled
                and traversal_context.get(
                    "temporal_emergency_raw_triggered", False
                )
                and not risk_candidate_feasible[traversal_index]
            )
            admission_waiting = bool(
                traversal_context.get("commit_admission_waiting", False)
            )
            admission_exit_deadline_hold_requested = bool(
                traversal_context.get(
                    "commit_admission_exit_deadline_hold_requested", False
                )
            )
            admission_hold_requested = bool(
                admission_waiting or admission_exit_deadline_hold_requested
            )
            admission_prealign_requested = bool(
                traversal_context.get(
                    "commit_admission_prealign_requested", False
                )
            )
            uncommitted_temporal_staging_hold_requested = bool(
                traversal_context.get(
                    "uncommitted_temporal_staging_hold_requested", False
                )
            )
            admission_hold_requested = bool(
                admission_hold_requested or admission_prealign_requested
            )
            transactional_hold_requested = bool(
                admission_hold_requested
                or uncommitted_temporal_staging_hold_requested
            )
            traversal_rearm_hold_overridden_by_temporal_closing = bool(
                rearm_pending
                and not admission_hold_requested
                and self.config
                .probabilistic_obstacle_traversal_window_rearm_temporal_closing_lattice_override_enabled
                and traversal_context.get(
                    "temporal_emergency_raw_triggered", False
                )
                and risk_candidate_feasible[traversal_index]
            )
            traversal_admission_exit_deadline_hold_overridden_by_hard_risk = bool(
                admission_exit_deadline_hold_requested
                and not risk_candidate_feasible[traversal_index]
            )
            traversal_uncommitted_temporal_staging_hold_overridden_by_hard_risk = bool(
                uncommitted_temporal_staging_hold_requested
                and not risk_candidate_feasible[traversal_index]
            )
            if traversal_uncommitted_temporal_staging_hold_overridden_by_hard_risk:
                # The staging counter represents consecutive *executed* hold
                # actions.  A requested hold that is rejected by the hard-risk
                # lattice never actually holds the robot, so it must not
                # contribute toward the retreat threshold on the next cycle.
                # Reset here, after the selected candidate is known, while
                # leaving the default-disabled path untouched.
                self._probabilistic_traversal_uncommitted_hold_steps = 0
            traversal_commit_overridden_by_post_center_hard_risk = bool(
                traversal_started
                and not retreat_requested
                and not rearm_pending
                and self.config
                .probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled
                and float(traversal_context.get("current_progress", 0.0))
                >= float(traversal_context.get("crossing_progress", 0.0))
                - 1.0e-9
                and float(traversal_context.get("current_progress", 0.0))
                < float(traversal_context.get("clear_progress", 0.0))
                - 1.0e-9
                and not risk_candidate_feasible[traversal_index]
            )
            traversal_commit_overridden_by_post_center_temporal_risk = bool(
                traversal_started
                and not retreat_requested
                and not rearm_pending
                and self.config
                .probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled
                and bool(
                    traversal_context.get(
                        "temporal_emergency_raw_triggered",
                        temporal_emergency_triggered,
                    )
                )
                and float(
                    traversal_context.get(
                        "temporal_corroboration_probability_mass", 0.0
                    )
                )
                >= float(
                    self.config
                    .probabilistic_obstacle_traversal_window_temporal_abort_mass_floor
                )
                and float(traversal_context.get("current_progress", 0.0))
                >= float(traversal_context.get("crossing_progress", 0.0))
                - 1.0e-9
                and float(traversal_context.get("current_progress", 0.0))
                < float(traversal_context.get("clear_progress", 0.0))
                - 1.0e-9
            )
            post_center_transaction_scope = bool(
                traversal_started
                and not retreat_requested
                and not rearm_pending
                and float(traversal_context.get("current_progress", 0.0))
                >= float(traversal_context.get("crossing_progress", 0.0))
                - 1.0e-9
                and float(traversal_context.get("current_progress", 0.0))
                < float(traversal_context.get("clear_progress", 0.0))
                - 1.0e-9
            )
            temporal_scan_ttc_s = float(
                traversal_context.get(
                    "temporal_emergency_ttc_s", float("inf")
                )
            )
            temporal_scan_hard_stop_ttc_s = float(
                traversal_context.get(
                    "temporal_emergency_safety_hard_stop_ttc_s", 0.0
                )
            )
            temporal_scan_hard_stop_active = bool(
                traversal_context.get(
                    "temporal_emergency_scan_valid", False
                )
                and temporal_scan_hard_stop_ttc_s > 0.0
                and np.isfinite(temporal_scan_ttc_s)
                and 0.0 < temporal_scan_ttc_s
                <= temporal_scan_hard_stop_ttc_s
            )
            traversal_post_center_low_ttc_continuity_requested = bool(
                self.config
                .probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled
                and traversal_commit_overridden_by_post_center_hard_risk
                and post_center_transaction_scope
                and temporal_scan_hard_stop_active
                and not traversal_context.get(
                    "temporal_emergency_closing_observed", False
                )
                and not traversal_context.get(
                    "temporal_emergency_rearm_ready", True
                )
                and not traversal_post_center_forward_exit_commit_requested
            )
            traversal_post_center_low_ttc_continuity_release_condition_met = bool(
                self.config
                .probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled
                and post_center_transaction_scope
                and (
                    traversal_context.get(
                        "temporal_emergency_rearm_ready", True
                    )
                    or not temporal_scan_hard_stop_active
                )
            )
            traversal_retreat_overridden_by_hard_risk = bool(
                retreat_requested
                and self.config
                .probabilistic_obstacle_traversal_window_retreat_hard_risk_override_enabled
                and not risk_candidate_feasible[traversal_index]
            )
            traversal_retreat_overridden_by_temporal_midpoint_guard = bool(
                retreat_requested
                and traversal_context.get(
                    "retreat_temporal_lattice_requested", False
                )
                and self.config
                .probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled
                and (
                    not self.config
                    .probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled
                    or traversal_context.get(
                        "temporal_emergency_raw_triggered", False
                    )
                    or exit_deadline_retreat_escape_transaction_active
                )
            )
            traversal_temporal_retreat_post_intent_forward_filter_requested = bool(
                traversal_retreat_overridden_by_temporal_midpoint_guard
                and self.config
                .probabilistic_obstacle_traversal_window_temporal_retreat_post_intent_all_hard_forward_filter_enabled
                and traversal_context.get(
                    "temporal_retreat_raw_lattice_requested", False
                )
                and not temporal_emergency_triggered
            )
            traversal_temporal_midpoint_retreat_forward_filter_requested = bool(
                traversal_retreat_overridden_by_temporal_midpoint_guard
                and self.config
                .probabilistic_obstacle_traversal_window_temporal_midpoint_retreat_reverse_filter_enabled
                and traversal_context.get(
                    "temporal_retreat_raw_lattice_requested", False
                )
                and not exit_deadline_retreat_escape_transaction_active
            )
            if (
                (
                    traversal_started
                    and not traversal_commit_overridden_by_post_center_hard_risk
                    and not traversal_commit_overridden_by_post_center_temporal_risk
                )
                or (
                    retreat_requested
                    and not traversal_retreat_overridden_by_hard_risk
                    and not traversal_retreat_overridden_by_temporal_midpoint_guard
                )
                or (
                    rearm_pending
                    and not traversal_rearm_hold_overridden_by_hard_risk
                    and not traversal_rearm_hold_overridden_by_temporal_closing
                )
                or (
                    transactional_hold_requested
                    and risk_candidate_feasible[traversal_index]
                )
                or (
                    not retreat_requested
                    and not rearm_pending
                    and not traversal_commit_overridden_by_post_center_hard_risk
                    and not traversal_commit_overridden_by_post_center_temporal_risk
                    and risk_candidate_feasible[traversal_index]
                )
            ):
                fallback_candidate_index = traversal_index
                fallback_kind = (
                    "terminal_capture_candidate"
                    if terminal_capture_active
                    else "traversal_uncommitted_temporal_staging_hold"
                    if uncommitted_temporal_staging_hold_requested
                    else "traversal_admission_prealign"
                    if admission_prealign_requested
                    else "traversal_admission_exit_deadline_hold"
                    if admission_exit_deadline_hold_requested
                    else "traversal_admission_stability_hold"
                    if admission_waiting
                    else "traversal_rearm_no_crossing_certified_handoff"
                    if traversal_context.get(
                        "rearm_no_crossing_certified_handoff_active", False
                    )
                    else "certified_traversal_window_candidate"
                )
                fallback_used = True
                traversal_selected = True
                if (
                    not retreat_requested
                    and not rearm_pending
                    and not transactional_hold_requested
                    and not terminal_capture_active
                ):
                    self._probabilistic_traversal_commit_started = True
                    traversal_started = True
                sequence = values[traversal_index].copy()
                action = self.action_spec.clip(sequence[0])
                sequence[0] = action
                trajectory = self.rollout(state, sequence)[0]
                selected_risk = self._probabilistic_collision_risk(
                    trajectory[None, ...], forecasts
                )
                if (
                    self.config
                    .probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_release_enabled
                    and rearm_pending
                    and traversal_context.get(
                        "rearm_no_crossing_certified_handoff_active",
                        False,
                    )
                    and traversal_context.get(
                        "rearm_no_crossing_certified_handoff_safe",
                        False,
                    )
                    and risk_candidate_feasible[traversal_index]
                ):
                    # Release only after the already-certified handoff
                    # candidate has actually won final arbitration.  Clearing
                    # in the context builder would drop the transaction even
                    # when a same-cycle hard-risk override rejects it.
                    self._probabilistic_traversal_rearm_pending = False
                    self._probabilistic_traversal_admission_safe_streak = 0
                    self._probabilistic_traversal_admission_signature = None
                    traversal_context["rearm_pending"] = False
                    traversal_context[
                        "rearm_released_by_certified_handoff"
                    ] = True

        terminal_intent_safe_stop_requested = bool(
            self.config
            .probabilistic_obstacle_terminal_intent_safe_stop_enabled
            and traversal_context.get("terminal_phase", False)
            and temporal_emergency_triggered
            and not traversal_context.get(
                "temporal_emergency_raw_triggered", False
            )
            and not traversal_context.get("commit_active", False)
            and not traversal_context.get("retreat_requested", False)
            and not traversal_context.get("rearm_pending", False)
        )
        if (
            terminal_intent_safe_stop_requested
            and not traversal_selected
            and hard_action in (
                "active_avoidance",
                "active_avoidance_motion",
            )
        ):
            stop_sequence = np.zeros(
                (self.config.horizon, self.action_spec.dimension),
                dtype=np.float64,
            )
            stop_trajectory = self.rollout(state, stop_sequence)[0]
            stop_risk = self._probabilistic_collision_risk(
                stop_trajectory[None, ...], forecasts
            )
            stop_maximum_probability = float(
                stop_risk.maximum_step_probability[0]
            )
            stop_probability_mass = float(
                stop_risk.accumulated_probability_mass[0]
            )
            if not bool(stop_risk.hard_violation[0]):
                # The raw causal warning has cleared, so only the configured
                # intent hysteresis remains.  Near a terminal goal, continuing
                # a previously latched radial translation can discard a safe
                # arrival window.  Reuse the existing full-horizon stop
                # certificate and the unchanged hard threshold; active
                # warnings and traversal transactions retain priority.
                terminal_intent_safe_stop_selected = True
                fallback_used = True
                fallback_kind = "terminal_intent_safe_stop"
                action = np.zeros(
                    self.action_spec.dimension, dtype=np.float64
                )
                sequence = stop_sequence
                trajectory = stop_trajectory
                selected_risk = stop_risk

        if (
            (
                temporal_emergency_triggered
                or traversal_retreat_overridden_by_temporal_midpoint_guard
                or traversal_commit_overridden_by_post_center_hard_risk
                    or traversal_commit_overridden_by_post_center_temporal_risk
                    or traversal_admission_exit_deadline_hold_overridden_by_hard_risk
                    or traversal_uncommitted_temporal_staging_hold_overridden_by_hard_risk
                    or traversal_rearm_hold_overridden_by_hard_risk
                    or traversal_rearm_hold_overridden_by_temporal_closing
            )
            and not traversal_selected
            and not terminal_intent_safe_stop_selected
            and (
                not temporal_emergency_triggered
                or self.config
                .probabilistic_obstacle_emergency_candidate_direct_fallback_enabled
            )
            and hard_action in (
                "active_avoidance",
                "active_avoidance_motion",
            )
            and np.any(emergency_mask & eligible_mask)
        ):
            if evaluated_candidates is None:
                candidate_trajectories = self.rollout(state, values)
                evaluated_candidates = self._probabilistic_collision_risk(
                    candidate_trajectories, forecasts
                )
            risk_candidate_feasible = ~evaluated_candidates.hard_violation
            all_emergency_indices = np.flatnonzero(
                emergency_mask & eligible_mask
            )
            emergency_index = -1
            if all_emergency_indices.size:
                preferred_index = int(all_emergency_indices[0])
                preferred_emergency_probability = float(
                    evaluated_candidates
                    .maximum_step_probability[preferred_index]
                )
                preferred_emergency_probability_mass = float(
                    evaluated_candidates
                    .accumulated_probability_mass[preferred_index]
                )
                preferred_emergency_hard_violation = bool(
                    evaluated_candidates.hard_violation[preferred_index]
                )
                if not preferred_emergency_hard_violation:
                    # Temporal scan flow is an independent safety signal: once
                    # it identifies a closing object, a below-threshold radial
                    # escape remains certified even when an under-confident
                    # forecast assigns the nominal trajectory a slightly lower
                    # numerical risk.  This prevents direction oscillation.
                    emergency_index = preferred_index
                else:
                    stop_sequence_for_preference = np.zeros(
                        (
                            self.config.horizon,
                            self.action_spec.dimension,
                        ),
                        dtype=np.float64,
                    )
                    stop_trajectory_for_preference = self.rollout(
                        state, stop_sequence_for_preference
                    )[0]
                    stop_risk_for_preference = (
                        self._probabilistic_collision_risk(
                            stop_trajectory_for_preference[None, ...],
                            forecasts,
                        )
                    )
                    stop_maximum_probability = float(
                        stop_risk_for_preference
                        .maximum_step_probability[0]
                    )
                    stop_probability_mass = float(
                        stop_risk_for_preference
                        .accumulated_probability_mass[0]
                    )
                    if self._risk_is_strictly_better(
                        preferred_emergency_probability,
                        preferred_emergency_probability_mass,
                        stop_maximum_probability,
                        stop_probability_mass,
                        allow_equal_maximum_mass=True,
                    ):
                        emergency_index = preferred_index

                # If every lattice member violates the hard threshold, the
                # former contract kept the first radial intent whenever it was
                # only marginally better than stopping.  In a moving-obstacle
                # chase geometry that can ignore a substantially lower-risk
                # lateral member of the same already-budgeted lattice.  Rank
                # all emergency candidates before applying the feasibility
                # subset; no new rollout or non-causal signal is introduced.
                minimum_risk_pool = all_emergency_indices
                traversal_temporal_retreat_post_intent_forward_lattice_filtered = bool(
                    traversal_temporal_retreat_post_intent_forward_filter_requested
                    and np.all(
                        evaluated_candidates.hard_violation[
                            all_emergency_indices
                        ]
                    )
                )
                if (
                    (
                        traversal_admission_exit_deadline_hold_overridden_by_hard_risk
                        or traversal_temporal_retreat_post_intent_forward_lattice_filtered
                        or traversal_temporal_midpoint_retreat_forward_filter_requested
                    )
                    and "v_cmd" in self.action_spec.names
                ):
                    reverse_indices = all_emergency_indices[
                        values[
                            all_emergency_indices,
                            0,
                            self.action_spec.index("v_cmd"),
                        ] < 0.0
                    ]
                    if reverse_indices.size:
                        minimum_risk_pool = reverse_indices
                        if traversal_admission_exit_deadline_hold_overridden_by_hard_risk:
                            traversal_admission_exit_deadline_forward_lattice_filtered = True
                        if traversal_temporal_midpoint_retreat_forward_filter_requested:
                            traversal_temporal_midpoint_retreat_forward_lattice_filtered = True
                if (
                    traversal_post_center_forward_exit_commit_requested
                    and "v_cmd" in self.action_spec.names
                ):
                    v_index = self.action_spec.index("v_cmd")
                    nonreverse_indices = all_emergency_indices[
                        values[all_emergency_indices, 0, v_index] >= 0.0
                    ]
                    if nonreverse_indices.size:
                        traversal_post_center_forward_exit_lattice_filtered = bool(
                            np.any(
                                values[
                                    all_emergency_indices, 0, v_index
                                ] < 0.0
                            )
                        )
                        minimum_risk_pool = nonreverse_indices
                if (
                    traversal_post_center_low_ttc_continuity_requested
                    and "v_cmd" in self.action_spec.names
                ):
                    v_index = self.action_spec.index("v_cmd")
                    nonforward_indices = all_emergency_indices[
                        values[all_emergency_indices, 0, v_index] <= 0.0
                    ]
                    if nonforward_indices.size:
                        traversal_post_center_low_ttc_continuity_forward_lattice_filtered = bool(
                            np.any(
                                values[
                                    all_emergency_indices, 0, v_index
                                ] > 0.0
                            )
                        )
                        traversal_post_center_low_ttc_continuity_hard_risk_fallback = bool(
                            np.all(
                                evaluated_candidates.hard_violation[
                                    nonforward_indices
                                ]
                            )
                        )
                        minimum_risk_pool = nonforward_indices
                minimum_all_order = np.lexsort((
                    candidate_costs[minimum_risk_pool],
                    evaluated_candidates.accumulated_probability_mass[
                        minimum_risk_pool
                    ],
                    evaluated_candidates.maximum_step_probability[
                        minimum_risk_pool
                    ],
                ))
                minimum_risk_emergency_index = int(
                    minimum_risk_pool[minimum_all_order[0]]
                )
                if traversal_admission_exit_deadline_hold_overridden_by_hard_risk:
                    # The causal deadline already proves that continuing the
                    # generic forward proposal cannot clear the crossing in
                    # time.  When the zero-speed hold is itself hard-risk,
                    # choose the least-risk member of the existing six-slot
                    # lattice rather than returning authority to that generic
                    # proposal.  No candidate or rollout is added here.
                    emergency_index = minimum_risk_emergency_index
                if traversal_rearm_hold_overridden_by_hard_risk:
                    # A rearm hold is a safe default only while its same-cycle
                    # forecast remains below the frozen hard threshold.  A
                    # persistent causal scan warning must retain access to the
                    # already-budgeted lattice after the global intent expires.
                    emergency_index = minimum_risk_emergency_index
                if traversal_temporal_retreat_post_intent_forward_lattice_filtered:
                    # Once the one-shot scan intent expires, the active
                    # transaction is explicitly retreating toward its frozen
                    # staging target.  Keep the six-slot evidence budget but
                    # do not let a late all-hard ranking flip that transaction
                    # back to a forward member.
                    emergency_index = minimum_risk_emergency_index
                if traversal_temporal_midpoint_retreat_forward_lattice_filtered:
                    # A midpoint abort has already committed the controller to
                    # a frozen staging target behind the robot.  Keep all six
                    # evaluated budget slots for evidence, but execute the
                    # least-risk existing reverse member from the first raw-
                    # bound retreat cycle so the one-shot scan intent cannot
                    # advance away from that transaction target.
                    emergency_index = minimum_risk_emergency_index
                if traversal_post_center_low_ttc_continuity_requested:
                    # The scan-flow estimator still has a valid, critically
                    # low TTC and the previous matched encounter has not
                    # rearmed.  A brief tracker/scan angle mismatch must not
                    # turn a near-body stop into forward translation.  Rank
                    # only the existing non-forward lattice members; the
                    # six-slot rollout budget and risk evidence are unchanged.
                    emergency_index = minimum_risk_emergency_index
                if traversal_post_center_forward_exit_commit_requested:
                    # Crossing the frozen conflict center is a one-way
                    # transaction.  Select the least-risk non-reverse member
                    # of the existing lattice so the robot clears the moving
                    # obstacle path instead of backing into it again.
                    emergency_index = minimum_risk_emergency_index
                if (
                    preferred_emergency_hard_violation
                    and not traversal_commit_overridden_by_post_center_temporal_risk
                    and not exit_deadline_retreat_escape_transaction_active
                    and emergency_index >= 0
                    and minimum_risk_emergency_index != emergency_index
                    and self._risk_is_strictly_better(
                        evaluated_candidates.maximum_step_probability[
                            minimum_risk_emergency_index
                        ],
                        evaluated_candidates.accumulated_probability_mass[
                            minimum_risk_emergency_index
                        ],
                        evaluated_candidates.maximum_step_probability[
                            emergency_index
                        ],
                        evaluated_candidates.accumulated_probability_mass[
                            emergency_index
                        ],
                        allow_equal_maximum_mass=True,
                    )
                ):
                    emergency_index = minimum_risk_emergency_index

            emergency_indices = all_emergency_indices[
                risk_candidate_feasible[all_emergency_indices]
            ]
            if (
                traversal_post_center_forward_exit_commit_requested
                and emergency_indices.size
                and "v_cmd" in self.action_spec.names
            ):
                nonreverse_feasible_indices = emergency_indices[
                    values[
                        emergency_indices,
                        0,
                        self.action_spec.index("v_cmd"),
                    ] >= 0.0
                ]
                if nonreverse_feasible_indices.size:
                    emergency_indices = nonreverse_feasible_indices
            if (
                traversal_post_center_low_ttc_continuity_requested
                and emergency_indices.size
                and "v_cmd" in self.action_spec.names
            ):
                nonforward_feasible_indices = emergency_indices[
                    values[
                        emergency_indices,
                        0,
                        self.action_spec.index("v_cmd"),
                    ] <= 0.0
                ]
                if nonforward_feasible_indices.size:
                    emergency_indices = nonforward_feasible_indices
            if emergency_indices.size:
                if (
                    self.config
                    .probabilistic_obstacle_emergency_feasibility_over_direction_enabled
                    and emergency_index >= 0
                    and evaluated_candidates.hard_violation[emergency_index]
                ):
                    # Direction is advisory once it would execute a
                    # hard-violating member despite a feasible member in the
                    # same fixed six-slot lattice.  Reuse the existing risk
                    # ordering; no rollout, threshold, or truth input changes.
                    feasible_order = np.lexsort((
                        candidate_costs[emergency_indices],
                        evaluated_candidates.accumulated_probability_mass[
                            emergency_indices
                        ],
                        evaluated_candidates.maximum_step_probability[
                            emergency_indices
                        ],
                    ))
                    emergency_index = int(
                        emergency_indices[feasible_order[0]]
                    )
                    emergency_feasibility_over_direction_applied = True
                best_cost_emergency_index = int(
                    emergency_indices[
                        np.argmin(candidate_costs[emergency_indices])
                    ]
                )
                minimum_risk_order = np.lexsort((
                    candidate_costs[emergency_indices],
                    evaluated_candidates.accumulated_probability_mass[
                        emergency_indices
                    ],
                    evaluated_candidates.maximum_step_probability[
                        emergency_indices
                    ],
                ))
                minimum_risk_emergency_index = int(
                    emergency_indices[minimum_risk_order[0]]
                )
                strict_pareto_forward = (
                    emergency_index >= 0
                    and minimum_risk_emergency_index != emergency_index
                    and minimum_risk_emergency_index
                    == best_cost_emergency_index
                    and self._risk_is_strictly_better(
                        evaluated_candidates.maximum_step_probability[
                            minimum_risk_emergency_index
                        ],
                        evaluated_candidates.accumulated_probability_mass[
                            minimum_risk_emergency_index
                        ],
                        evaluated_candidates.maximum_step_probability[
                            emergency_index
                        ],
                        evaluated_candidates.accumulated_probability_mass[
                            emergency_index
                        ],
                        allow_equal_maximum_mass=True,
                    )
                )
                low_risk_cost_forward = (
                    emergency_index >= 0
                    and best_cost_emergency_index != emergency_index
                    and self.config
                    .probabilistic_obstacle_emergency_candidate_forward_risk_ceiling
                    > 0.0
                    and self.config
                    .probabilistic_obstacle_emergency_candidate_forward_mass_ceiling
                    > 0.0
                    and not evaluated_candidates.hard_violation[
                        best_cost_emergency_index
                    ]
                    and evaluated_candidates.maximum_step_probability[
                        best_cost_emergency_index
                    ]
                    <= self.config
                    .probabilistic_obstacle_emergency_candidate_forward_risk_ceiling
                    and evaluated_candidates.accumulated_probability_mass[
                        best_cost_emergency_index
                    ]
                    <= self.config
                    .probabilistic_obstacle_emergency_candidate_forward_mass_ceiling
                )
                forward_commit_index = (
                    minimum_risk_emergency_index
                    if strict_pareto_forward
                    else best_cost_emergency_index
                    if low_risk_cost_forward
                    else -1
                )
                if (
                    self.config
                    .probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled
                    and not traversal_commit_overridden_by_post_center_temporal_risk
                    and not exit_deadline_retreat_escape_transaction_active
                    and not traversal_admission_exit_deadline_hold_overridden_by_hard_risk
                    and not traversal_rearm_hold_overridden_by_hard_risk
                    and not traversal_rearm_hold_overridden_by_temporal_closing
                    and not traversal_temporal_retreat_post_intent_forward_lattice_filtered
                    and not traversal_temporal_midpoint_retreat_forward_lattice_filtered
                    and not traversal_post_center_low_ttc_continuity_requested
                    and emergency_index >= 0
                    and forward_commit_index >= 0
                    and "v_cmd" in self.action_spec.names
                    and "omega_cmd" in self.action_spec.names
                    and "theta" in self.state_spec.names
                    and values[
                        forward_commit_index,
                        0,
                        self.action_spec.index("v_cmd"),
                    ] > 0.0
                    and values[
                        emergency_index,
                        0,
                        self.action_spec.index("v_cmd"),
                    ] < 0.0
                ):
                    emergency_index = forward_commit_index
                    pareto_forward_commit_applied = True
                    low_risk_forward_commit_applied = bool(
                        low_risk_cost_forward and not strict_pareto_forward
                    )
                    self._probabilistic_emergency_latched_pattern = (
                        float(values[
                            emergency_index,
                            0,
                            self.action_spec.index("v_cmd"),
                        ]),
                        float(values[
                            emergency_index,
                            0,
                            self.action_spec.index("omega_cmd"),
                        ]),
                    )
                    theta = float(
                        np.asarray(state, dtype=np.float64)[
                            self.state_spec.index("theta")
                        ]
                    )
                    turn_duration = float(
                        self.config
                        .probabilistic_obstacle_emergency_candidate_prefix_steps
                        * self.config.dt
                    )
                    self._probabilistic_emergency_latched_heading = float(
                        np.arctan2(
                            np.sin(
                                theta
                                + self._probabilistic_emergency_latched_pattern[1]
                                * turn_duration
                            ),
                            np.cos(
                                theta
                                + self._probabilistic_emergency_latched_pattern[1]
                                * turn_duration
                            ),
                        )
                    )
            if emergency_index < 0 and emergency_indices.size:
                # If the preferred intent is not certifiable, fall back to the
                # safest feasible member of the same evaluated lattice.
                order = np.lexsort((
                    candidate_costs[emergency_indices],
                    evaluated_candidates
                    .accumulated_probability_mass[emergency_indices],
                    evaluated_candidates
                    .maximum_step_probability[emergency_indices],
                ))
                ranked_indices = list(emergency_indices[order])
                selected_probability = float(
                    selected_risk.maximum_step_probability[0]
                )
                selected_mass = float(
                    selected_risk.accumulated_probability_mass[0]
                )
                for candidate_index in ranked_indices:
                    candidate_index = int(candidate_index)
                    emergency_probability = float(
                        evaluated_candidates
                        .maximum_step_probability[candidate_index]
                    )
                    emergency_mass = float(
                        evaluated_candidates
                        .accumulated_probability_mass[candidate_index]
                    )
                    risk_not_worse = bool(
                        emergency_probability
                        < selected_probability - 1.0e-12
                        or (
                            np.isclose(
                                emergency_probability,
                                selected_probability,
                                atol=1.0e-12,
                                rtol=0.0,
                            )
                            and emergency_mass <= selected_mass + 1.0e-12
                        )
                    )
                    if risk_not_worse:
                        emergency_index = candidate_index
                        break
            if emergency_index >= 0:
                if exit_deadline_retreat_escape_transaction_active:
                    exit_deadline_retreat_escape_latched = (
                        self._latch_probabilistic_traversal_exit_deadline_retreat_escape(
                            (
                                values[
                                    emergency_index,
                                    0,
                                    self.action_spec.index("v_cmd"),
                                ],
                                values[
                                    emergency_index,
                                    0,
                                    self.action_spec.index("omega_cmd"),
                                ],
                            ),
                            state,
                        )
                    )
                if traversal_commit_overridden_by_post_center_temporal_risk:
                    self._latch_probabilistic_traversal_post_center_temporal_escape(
                        (
                            values[
                                emergency_index,
                                0,
                                self.action_spec.index("v_cmd"),
                            ],
                            values[
                                emergency_index,
                                0,
                                self.action_spec.index("omega_cmd"),
                            ],
                        ),
                        state,
                    )
                fallback_candidate_index = emergency_index
                fallback_kind = (
                    "exit_deadline_retreat_emergency_candidate"
                    if exit_deadline_retreat_escape_transaction_active
                    else "post_center_forward_exit_emergency_candidate"
                    if traversal_post_center_forward_exit_commit_requested
                    else "post_center_low_ttc_continuity_emergency_candidate"
                    if traversal_post_center_low_ttc_continuity_requested
                    else "post_center_hard_risk_emergency_candidate"
                    if traversal_commit_overridden_by_post_center_hard_risk
                    else "post_center_temporal_corroborated_emergency_candidate"
                    if traversal_commit_overridden_by_post_center_temporal_risk
                    else "admission_exit_deadline_hold_hard_risk_emergency_candidate"
                    if traversal_admission_exit_deadline_hold_overridden_by_hard_risk
                    else "uncommitted_temporal_staging_hold_hard_risk_emergency_candidate"
                    if traversal_uncommitted_temporal_staging_hold_overridden_by_hard_risk
                    else "rearm_hold_hard_risk_emergency_candidate"
                    if traversal_rearm_hold_overridden_by_hard_risk
                    else "rearm_hold_temporal_closing_emergency_candidate"
                    if traversal_rearm_hold_overridden_by_temporal_closing
                    else "temporal_retreat_post_intent_emergency_candidate"
                    if traversal_temporal_retreat_post_intent_forward_lattice_filtered
                    else "temporal_midpoint_retreat_emergency_candidate"
                    if traversal_temporal_midpoint_retreat_forward_lattice_filtered
                    else "temporal_scan_vetted_emergency_candidate"
                )
                fallback_used = True
                temporal_emergency_vetted = True
                sequence = values[emergency_index].copy()
                # Safety lattice commands are already bound-clipped and
                # must not lose the early escape window to the nominal
                # command slew limiter.
                action = self.action_spec.clip(sequence[0])
                sequence[0] = action
                trajectory = self.rollout(state, sequence)[0]
                selected_risk = self._probabilistic_collision_risk(
                    trajectory[None, ...], forecasts
                )

        if (
            not temporal_emergency_vetted
            and not traversal_selected
            and not terminal_intent_safe_stop_selected
            and initial_hard_violation
            and hard_action in (
                "active_avoidance",
                "active_avoidance_motion",
            )
        ):
            if evaluated_candidates is None:
                candidate_trajectories = self.rollout(state, values)
                evaluated_candidates = self._probabilistic_collision_risk(
                    candidate_trajectories, forecasts
                )
            risk_candidate_feasible = ~evaluated_candidates.hard_violation
            eligible = np.flatnonzero(eligible_mask)
            if eligible.size == 0:
                eligible = np.arange(values.shape[0], dtype=np.int64)
            feasible = eligible[risk_candidate_feasible[eligible]]
            stop_sequence = np.zeros(
                (self.config.horizon, self.action_spec.dimension),
                dtype=np.float64,
            )
            stop_trajectory = self.rollout(state, stop_sequence)[0]
            stop_risk = self._probabilistic_collision_risk(
                stop_trajectory[None, ...], forecasts
            )
            stop_maximum_probability = float(
                stop_risk.maximum_step_probability[0]
            )
            stop_probability_mass = float(
                stop_risk.accumulated_probability_mass[0]
            )
            if feasible.size:
                fallback_candidate_index = int(
                    feasible[np.argmin(candidate_costs[feasible])]
                )
                fallback_kind = "feasible_active_candidate"
            else:
                candidate_order = np.lexsort(
                    (
                        candidate_costs[eligible],
                        evaluated_candidates
                        .accumulated_probability_mass[eligible],
                        evaluated_candidates
                        .maximum_step_probability[eligible],
                    )
                )
                safest_index = int(eligible[candidate_order[0]])
                safest_probability = float(
                    evaluated_candidates
                    .maximum_step_probability[safest_index]
                )
                safest_probability_mass = float(
                    evaluated_candidates
                    .accumulated_probability_mass[safest_index]
                )
                if self._risk_is_strictly_better(
                    safest_probability,
                    safest_probability_mass,
                    stop_maximum_probability,
                    stop_probability_mass,
                    allow_equal_maximum_mass=(
                        self.config
                        .probabilistic_obstacle_stopping_feasibility_enabled
                    ),
                ):
                    fallback_candidate_index = safest_index
                    fallback_kind = (
                        "minimum_accumulated_risk_active_candidate"
                        if np.isclose(
                            safest_probability,
                            stop_maximum_probability,
                            atol=1.0e-12,
                            rtol=0.0,
                        )
                        else "minimum_risk_active_candidate"
                        )
                    if (
                        hard_action == "active_avoidance_motion"
                        and self.config
                        .probabilistic_obstacle_hard_violation_progress_tiebreak_enabled
                        and "v_cmd" in self.action_spec.names
                    ):
                        v_index = self.action_spec.index("v_cmd")
                        mass_limit = safest_probability_mass * (
                            1.0
                            + self.config
                            .probabilistic_obstacle_hard_violation_progress_mass_tolerance
                        ) + 1.0e-12
                        progress_mask = (
                            eligible_mask
                            & (
                                evaluated_candidates.maximum_step_probability
                                <= safest_probability + 1.0e-12
                            )
                            & (
                                evaluated_candidates.accumulated_probability_mass
                                <= mass_limit
                            )
                            & (values[:, 0, v_index] >= 0.0)
                        )
                        progress_indices = np.flatnonzero(progress_mask)
                        if progress_indices.size:
                            progress_order = np.lexsort(
                                (
                                    candidate_costs[progress_indices],
                                    -values[progress_indices, 0, v_index],
                                )
                            )
                            progress_index = int(
                                progress_indices[progress_order[0]]
                            )
                            if (
                                values[progress_index, 0, v_index]
                                > values[safest_index, 0, v_index]
                                + 1.0e-12
                            ):
                                fallback_candidate_index = progress_index
                                fallback_kind = (
                                    "risk_equivalent_forward_progress_candidate"
                                )
                                risk_equivalent_forward_progress_applied = True
                else:
                        fallback_kind = "stop_is_safest_candidate"
            fallback_used = True
            if fallback_candidate_index >= 0:
                sequence = values[fallback_candidate_index].copy()
                if emergency_mask[fallback_candidate_index]:
                    action = self.action_spec.clip(sequence[0])
                else:
                    action = self.action_spec.clip(
                        sequence[0], self.previous_action, self.config.dt
                    )
                sequence[0] = action
                trajectory = self.rollout(state, sequence)[0]
            else:
                action = np.zeros(
                    self.action_spec.dimension, dtype=np.float64
                )
                sequence = stop_sequence
                trajectory = stop_trajectory
                fail_closed = True
            selected_risk = self._probabilistic_collision_risk(
                trajectory[None, ...], forecasts
            )
        elif evaluated_candidates is not None:
            risk_candidate_feasible = ~evaluated_candidates.hard_violation

        hard_violation = bool(selected_risk.hard_violation[0])
        if hard_violation and hard_action == "stop":
            fail_closed = True
            fallback_used = True
            fallback_kind = "unconditional_stop"
            action = np.zeros(self.action_spec.dimension, dtype=np.float64)
            sequence = np.zeros(
                (self.config.horizon, self.action_spec.dimension),
                dtype=np.float64,
            )
            trajectory = self.rollout(state, sequence)[0]

        maximum_step_probability = float(
            selected_risk.maximum_step_probability[0]
        )
        speed_scale = 1.0
        active_avoidance_motion_selected = bool(
            hard_action == "active_avoidance_motion"
            and fallback_used
            and fallback_candidate_index >= 0
        )
        governor_ignores_active_avoidance = bool(
            self.config
            .probabilistic_obstacle_speed_governor_applies_during_active_avoidance
        )
        if (
            self.config.probabilistic_obstacle_speed_governor_enabled
            and not traversal_started
            and (
                governor_ignores_active_avoidance
                or not active_avoidance_motion_selected
            )
        ):
            hard_threshold = self.config.probabilistic_obstacle_hard_threshold
            soft_threshold = (
                self.config.probabilistic_obstacle_speed_governor_start_ratio
                * hard_threshold
            )
            speed_scale = float(np.clip(
                (hard_threshold - maximum_step_probability)
                / (hard_threshold - soft_threshold),
                0.0,
                1.0,
            ))

        post_center_temporal_escape_pattern = (
            self._probabilistic_traversal_post_center_temporal_pattern
        )
        if post_center_temporal_escape_pattern is None:
            post_center_temporal_escape_pattern = (0.0, 0.0)
        exit_deadline_retreat_escape_pattern = (
            self._probabilistic_traversal_exit_deadline_retreat_pattern
        )
        if exit_deadline_retreat_escape_pattern is None:
            exit_deadline_retreat_escape_pattern = (0.0, 0.0)

        emergency_candidate_trace = []
        emergency_indices_for_trace = np.flatnonzero(emergency_mask)
        if (
            evaluated_candidates is not None
            and emergency_indices_for_trace.size
        ):
            v_index = (
                self.action_spec.index("v_cmd")
                if "v_cmd" in self.action_spec.names
                else None
            )
            omega_index = (
                self.action_spec.index("omega_cmd")
                if "omega_cmd" in self.action_spec.names
                else None
            )
            preferred_trace_index = int(emergency_indices_for_trace[0])
            for candidate_index in emergency_indices_for_trace:
                index = int(candidate_index)
                emergency_candidate_trace.append({
                    "index": index,
                    "first_v": (
                        float(values[index, 0, v_index])
                        if v_index is not None else 0.0
                    ),
                    "first_omega": (
                        float(values[index, 0, omega_index])
                        if omega_index is not None else 0.0
                    ),
                    "maximum_probability": float(
                        evaluated_candidates
                        .maximum_step_probability[index]
                    ),
                    "probability_mass": float(
                        evaluated_candidates
                        .accumulated_probability_mass[index]
                    ),
                    "hard_violation": bool(
                        evaluated_candidates.hard_violation[index]
                    ),
                    "eligible": bool(eligible_mask[index]),
                    "cost": float(candidate_costs[index]),
                    "selected": bool(index == fallback_candidate_index),
                    "preferred": bool(index == preferred_trace_index),
                    "minimum_risk": bool(
                        index == minimum_risk_emergency_index
                    ),
                    "best_cost": bool(index == best_cost_emergency_index),
                    "sequence": values[index].tolist(),
                    "risk_by_step": (
                        np.asarray(
                            evaluated_candidates
                            .step_probability_upper_bound[index],
                            dtype=np.float64,
                        ).tolist()
                        if hasattr(
                            evaluated_candidates,
                            "step_probability_upper_bound",
                        )
                        else []
                    ),
                    "static_clearance_by_step": (
                        static_clearance_trace[index].tolist()
                        if static_clearance_trace is not None else []
                    ),
                })

        diagnostics = {
            "probabilistic_obstacle_risk_enabled": True,
            "probabilistic_obstacle_forecast_count": len(forecasts),
            "probabilistic_obstacle_probability_mass": float(
                selected_risk.accumulated_probability_mass[0]
            ),
            "probabilistic_obstacle_union_bound": float(
                selected_risk.horizon_union_bound[0]
            ),
            "probabilistic_obstacle_maximum_step_probability": float(
                selected_risk.maximum_step_probability[0]
            ),
            "probabilistic_obstacle_hard_violation": hard_violation,
            "probabilistic_obstacle_fail_closed": fail_closed,
            "probabilistic_obstacle_speed_scale": speed_scale,
            "probabilistic_obstacle_speed_governor_bypassed_for_active_avoidance": (
                active_avoidance_motion_selected
            ),
            "probabilistic_obstacle_initial_hard_violation": (
                initial_hard_violation
            ),
            "probabilistic_obstacle_candidate_feasible_fraction": float(
                np.mean(risk_candidate_feasible)
                if evaluated_candidates is not None else 1.0
            ),
            "probabilistic_obstacle_active_fallback_used": fallback_used,
            "probabilistic_obstacle_active_fallback_index": (
                fallback_candidate_index
            ),
            "probabilistic_obstacle_active_fallback_kind": fallback_kind,
            "probabilistic_obstacle_risk_equivalent_forward_progress_applied": (
                risk_equivalent_forward_progress_applied
            ),
            "probabilistic_obstacle_stop_maximum_probability": (
                stop_maximum_probability
            ),
            "probabilistic_obstacle_stop_probability_mass": (
                stop_probability_mass
            ),
            "probabilistic_obstacle_stopping_feasibility_enabled": bool(
                self.config
                .probabilistic_obstacle_stopping_feasibility_enabled
            ),
            "probabilistic_obstacle_emergency_candidate_count": int(
                np.sum(emergency_mask)
            ),
            "probabilistic_obstacle_emergency_candidate_selected": bool(
                fallback_candidate_index >= 0
                and emergency_mask[fallback_candidate_index]
            ),
            "probabilistic_obstacle_emergency_candidate_direct_fallback_enabled": bool(
                self.config
                .probabilistic_obstacle_emergency_candidate_direct_fallback_enabled
            ),
            "probabilistic_obstacle_emergency_candidate_direct_fallback_suppressed": bool(
                temporal_emergency_triggered
                and not self.config
                .probabilistic_obstacle_emergency_candidate_direct_fallback_enabled
            ),
            "probabilistic_obstacle_emergency_candidate_trace": (
                emergency_candidate_trace
            ),
            "probabilistic_obstacle_temporal_emergency_triggered": bool(
                temporal_emergency_triggered
            ),
            "probabilistic_obstacle_temporal_emergency_vetted": bool(
                temporal_emergency_vetted
            ),
            "probabilistic_obstacle_terminal_intent_safe_stop_selected": bool(
                terminal_intent_safe_stop_selected
            ),
            "probabilistic_obstacle_traversal_window_enabled": bool(
                traversal_context.get("enabled", False)
            ),
            "probabilistic_obstacle_traversal_terminal_target_bearing_steering_active": bool(
                traversal_context.get(
                    "terminal_target_bearing_steering_active", False
                )
            ),
            "probabilistic_obstacle_terminal_capture_candidate_active": bool(
                traversal_context.get("terminal_capture_active", False)
            ),
            "probabilistic_obstacle_traversal_window_safe": bool(
                traversal_context.get("window_safe", False)
            ),
            "probabilistic_obstacle_traversal_forecast_sufficient": bool(
                traversal_context.get("forecast_sufficient", False)
            ),
            "probabilistic_obstacle_traversal_commit_active": bool(
                traversal_context.get("commit_active", False)
            ),
            "probabilistic_obstacle_traversal_commit_started": bool(
                traversal_started
            ),
            "probabilistic_obstacle_traversal_commit_completed": bool(
                traversal_context.get("commit_completed", False)
            ),
            "probabilistic_obstacle_traversal_commit_cancelled": bool(
                traversal_context.get("commit_cancelled", False)
            ),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_closing": bool(
                traversal_context.get(
                    "commit_cancelled_by_temporal_closing", False
                )
            ),
            "probabilistic_obstacle_traversal_temporal_abort_current_hazard_signal": bool(
                traversal_context.get(
                    "temporal_abort_current_hazard_signal", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_midpoint_guard": bool(
                traversal_context.get(
                    "commit_cancelled_by_temporal_midpoint_guard", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard": bool(
                traversal_context.get(
                    "commit_cancelled_by_temporal_exit_deadline_guard",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_retreat_requested": bool(
                traversal_context.get("retreat_requested", False)
            ),
            "probabilistic_obstacle_traversal_retreat_temporal_lattice_requested": bool(
                traversal_context.get(
                    "retreat_temporal_lattice_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested": bool(
                traversal_context.get(
                    "temporal_retreat_raw_lattice_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_retreat_completed": bool(
                traversal_context.get("retreat_completed", False)
            ),
            "probabilistic_obstacle_traversal_retreat_completion_projection_alias_rejected": bool(
                traversal_context.get(
                    "retreat_completion_projection_alias_rejected", False
                )
            ),
            "probabilistic_obstacle_traversal_retreat_target_x": float(
                traversal_context.get("retreat_target_x", 0.0)
            ),
            "probabilistic_obstacle_traversal_retreat_target_y": float(
                traversal_context.get("retreat_target_y", 0.0)
            ),
            "probabilistic_obstacle_traversal_retreat_signed_distance_m": float(
                traversal_context.get("retreat_signed_distance_m", 0.0)
            ),
            "probabilistic_obstacle_traversal_retreat_frozen_reference_progress": float(
                traversal_context.get(
                    "retreat_frozen_reference_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_rearm_pending": bool(
                traversal_context.get("rearm_pending", False)
            ),
            "probabilistic_obstacle_traversal_rearm_no_crossing_safe_streak": int(
                traversal_context.get("rearm_no_crossing_safe_streak", 0)
            ),
            "probabilistic_obstacle_traversal_rearm_released_by_no_crossing_clearance": bool(
                traversal_context.get(
                    "rearm_released_by_no_crossing_clearance", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_released_by_certified_handoff": bool(
                traversal_context.get(
                    "rearm_released_by_certified_handoff", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_active": bool(
                traversal_context.get(
                    "rearm_no_crossing_certified_handoff_active", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_safe": bool(
                traversal_context.get(
                    "rearm_no_crossing_certified_handoff_safe", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_staging_approach_requested": bool(
                traversal_context.get(
                    "rearm_staging_approach_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_staging_approach_safe": bool(
                traversal_context.get(
                    "rearm_staging_approach_safe", False
                )
            ),
            "probabilistic_obstacle_traversal_rearm_staging_target_progress": float(
                traversal_context.get(
                    "rearm_staging_target_progress", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_retreat_progress": float(
                traversal_context.get("retreat_progress", 0.0)
            ),
            "probabilistic_obstacle_traversal_candidate_selected": bool(
                traversal_selected
            ),
            "probabilistic_obstacle_traversal_candidate_geometry_eligible": bool(
                traversal_candidate_geometry_eligible
            ),
            "probabilistic_obstacle_traversal_candidate_static_minimum_clearance": float(
                traversal_candidate_static_minimum_clearance
            ),
            "probabilistic_obstacle_traversal_crossing_progress": float(
                traversal_context.get("crossing_progress", 0.0)
            ),
            "probabilistic_obstacle_traversal_entry_progress": float(
                traversal_context.get("entry_progress", 0.0)
            ),
            "probabilistic_obstacle_traversal_clear_progress": float(
                traversal_context.get("clear_progress", 0.0)
            ),
            "probabilistic_obstacle_traversal_current_progress": float(
                traversal_context.get("current_progress", 0.0)
            ),
            "probabilistic_obstacle_traversal_maximum_probability": float(
                traversal_context.get("maximum_probability", 0.0)
            ),
            "probabilistic_obstacle_traversal_probability_mass": float(
                traversal_context.get("probability_mass", 0.0)
            ),
            "probabilistic_obstacle_traversal_temporal_corroboration_enabled": bool(
                self.config
                .probabilistic_obstacle_traversal_window_temporal_abort_full_horizon_corroboration_enabled
            ),
            "probabilistic_obstacle_traversal_temporal_corroboration_maximum_probability": float(
                traversal_context.get(
                    "temporal_corroboration_maximum_probability", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_temporal_corroboration_probability_mass": float(
                traversal_context.get(
                    "temporal_corroboration_probability_mass", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_hold_retreat_triggered": bool(
                traversal_context.get(
                    "uncommitted_hold_retreat_triggered", False
                )
            ),
            "probabilistic_obstacle_zone_occupancy_hold_active": bool(
                traversal_context.get("zone_occupancy_hold_active", False)
            ),
            "probabilistic_obstacle_traversal_commit_admission_full_horizon_enabled": bool(
                self.config
                .probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled
            ),
            "probabilistic_obstacle_traversal_commit_admission_full_horizon_safe": bool(
                traversal_context.get(
                    "commit_admission_full_horizon_safe", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_rejected": bool(
                traversal_context.get("commit_admission_rejected", False)
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_enabled": bool(
                self.config
                .probabilistic_obstacle_traversal_window_commit_admission_exit_deadline_enabled
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_safe": bool(
                traversal_context.get(
                    "commit_admission_exit_deadline_safe", True
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected": bool(
                traversal_context.get(
                    "commit_admission_exit_deadline_rejected", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested": bool(
                traversal_context.get(
                    "commit_admission_exit_deadline_hold_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_prealign_requested": bool(
                traversal_context.get(
                    "commit_admission_prealign_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested": bool(
                traversal_context.get(
                    "uncommitted_temporal_staging_hold_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected": bool(
                traversal_selected
                and traversal_context.get(
                    "uncommitted_temporal_staging_hold_requested", False
                )
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override": bool(
                traversal_uncommitted_temporal_staging_hold_overridden_by_hard_risk
            ),
            "probabilistic_obstacle_traversal_uncommitted_temporal_staging_terminal_release_active": bool(
                traversal_context.get(
                    "uncommitted_temporal_staging_terminal_release_active",
                    False,
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override": bool(
                traversal_admission_exit_deadline_hold_overridden_by_hard_risk
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered": bool(
                traversal_admission_exit_deadline_forward_lattice_filtered
            ),
            "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk": bool(
                traversal_rearm_hold_overridden_by_hard_risk
            ),
            "probabilistic_obstacle_traversal_rearm_hold_overridden_by_temporal_closing": bool(
                traversal_rearm_hold_overridden_by_temporal_closing
            ),
            "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered": bool(
                traversal_temporal_retreat_post_intent_forward_lattice_filtered
            ),
            "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered": bool(
                traversal_temporal_midpoint_retreat_forward_lattice_filtered
            ),
            "probabilistic_obstacle_traversal_commit_admission_exit_deadline_margin_s": float(
                traversal_context.get(
                    "commit_admission_exit_deadline_margin_s", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_commit_admission_safe_streak": int(
                traversal_context.get("commit_admission_safe_streak", 0)
            ),
            "probabilistic_obstacle_traversal_commit_admission_required_streak": int(
                traversal_context.get("commit_admission_required_streak", 1)
            ),
            "probabilistic_obstacle_traversal_commit_admission_waiting": bool(
                traversal_context.get("commit_admission_waiting", False)
            ),
            "probabilistic_obstacle_traversal_commit_admission_released": bool(
                traversal_context.get("commit_admission_released", False)
            ),
            "probabilistic_obstacle_traversal_commit_preserved_for_nearest_safe_exit": bool(
                traversal_context.get(
                    "commit_preserved_for_nearest_safe_exit", False
                )
            ),
            "probabilistic_obstacle_traversal_forward_exit_distance_m": float(
                traversal_context.get("forward_exit_distance_m", 0.0)
            ),
            "probabilistic_obstacle_traversal_retreat_exit_distance_m": float(
                traversal_context.get("retreat_exit_distance_m", 0.0)
            ),
            "probabilistic_obstacle_traversal_forward_exit_optimistic_time_s": float(
                traversal_context.get(
                    "forward_exit_optimistic_time_s", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_retreat_exit_optimistic_time_s": float(
                traversal_context.get(
                    "retreat_exit_optimistic_time_s", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_temporal_exit_deadline_ttc_s": float(
                traversal_context.get(
                    "temporal_exit_deadline_ttc_s", 0.0
                )
            ),
            "probabilistic_obstacle_traversal_temporal_exit_deadline_guard_triggered": bool(
                traversal_context.get(
                    "temporal_exit_deadline_guard_triggered", False
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active": bool(
                exit_deadline_retreat_escape_transaction_active
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched": bool(
                exit_deadline_retreat_escape_latched
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused": bool(
                traversal_context.get(
                    "exit_deadline_retreat_escape_reused", False
                )
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_v": float(
                exit_deadline_retreat_escape_pattern[0]
            ),
            "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_pattern_omega": float(
                exit_deadline_retreat_escape_pattern[1]
            ),
            "probabilistic_obstacle_traversal_retreat_overridden_by_hard_risk": bool(
                traversal_retreat_overridden_by_hard_risk
            ),
            "probabilistic_obstacle_traversal_retreat_overridden_by_temporal_midpoint_guard": bool(
                traversal_retreat_overridden_by_temporal_midpoint_guard
            ),
            "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk": bool(
                traversal_commit_overridden_by_post_center_hard_risk
            ),
            "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk": bool(
                traversal_commit_overridden_by_post_center_temporal_risk
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested": bool(
                traversal_post_center_forward_exit_commit_requested
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied": bool(
                traversal_post_center_forward_exit_coverage_applied
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count": int(
                traversal_post_center_forward_exit_candidate_count
            ),
            "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered": bool(
                traversal_post_center_forward_exit_lattice_filtered
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested": bool(
                traversal_post_center_low_ttc_continuity_requested
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered": bool(
                traversal_post_center_low_ttc_continuity_forward_lattice_filtered
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback": bool(
                traversal_post_center_low_ttc_continuity_hard_risk_fallback
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_release_condition_met": bool(
                traversal_post_center_low_ttc_continuity_release_condition_met
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_requested": bool(
                traversal_post_center_low_ttc_nonforward_coverage_requested
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_coverage_applied": bool(
                traversal_post_center_low_ttc_nonforward_coverage_applied
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_nonforward_candidate_count": int(
                traversal_post_center_low_ttc_nonforward_candidate_count
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_requested": bool(
                traversal_post_center_low_ttc_first_step_boundary_handoff_requested
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count": int(
                traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_first_step_boundary_handoff_selected": bool(
                traversal_post_center_low_ttc_first_step_boundary_handoff_requested
                and
                0 <= fallback_candidate_index < values.shape[0]
                and first_step_boundary_handoff_mask[
                    fallback_candidate_index
                ]
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_requested": bool(
                traversal_post_center_low_ttc_prefix_boundary_handoff_requested
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_admitted_count": int(
                traversal_post_center_low_ttc_first_step_boundary_handoff_admitted_count
                if traversal_post_center_low_ttc_prefix_boundary_handoff_requested
                else 0
            ),
            "probabilistic_obstacle_traversal_post_center_low_ttc_prefix_boundary_handoff_selected": bool(
                traversal_post_center_low_ttc_prefix_boundary_handoff_requested
                and 0 <= fallback_candidate_index < values.shape[0]
                and first_step_boundary_handoff_mask[
                    fallback_candidate_index
                ]
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_raw_triggered": bool(
                traversal_context.get(
                    "temporal_emergency_raw_triggered", False
                )
            ),
            "probabilistic_obstacle_traversal_temporal_closing_observed": bool(
                traversal_context.get(
                    "temporal_emergency_closing_observed", False
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_latched": bool(
                self._probabilistic_traversal_post_center_temporal_pattern
                is not None
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_reused": bool(
                traversal_context.get(
                    "post_center_temporal_escape_reused", False
                )
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_v": float(
                post_center_temporal_escape_pattern[0]
            ),
            "probabilistic_obstacle_traversal_post_center_temporal_escape_pattern_omega": float(
                post_center_temporal_escape_pattern[1]
            ),
            "probabilistic_obstacle_traversal_required_steps": int(
                traversal_context.get("required_steps", 0)
            ),
            "probabilistic_obstacle_traversal_forecast_index": int(
                traversal_context.get("forecast_index", -1)
            ),
            "probabilistic_obstacle_pareto_forward_commit_applied": bool(
                pareto_forward_commit_applied
            ),
            "probabilistic_obstacle_emergency_feasibility_over_direction_applied": bool(
                emergency_feasibility_over_direction_applied
            ),
            'probabilistic_obstacle_low_risk_forward_commit_applied': bool(
                low_risk_forward_commit_applied
            ),
            "probabilistic_obstacle_preferred_emergency_probability": float(
                preferred_emergency_probability
            ),
            "probabilistic_obstacle_preferred_emergency_probability_mass": float(
                preferred_emergency_probability_mass
            ),
            "probabilistic_obstacle_preferred_emergency_hard_violation": bool(
                preferred_emergency_hard_violation
            ),
            "probabilistic_obstacle_best_cost_emergency_index": int(
                best_cost_emergency_index
            ),
            "probabilistic_obstacle_best_cost_emergency_v": float(
                values[best_cost_emergency_index, 0,
                       self.action_spec.index("v_cmd")]
                if best_cost_emergency_index >= 0
                and "v_cmd" in self.action_spec.names else 0.0
            ),
            "probabilistic_obstacle_best_cost_emergency_omega": float(
                values[best_cost_emergency_index, 0,
                       self.action_spec.index("omega_cmd")]
                if best_cost_emergency_index >= 0
                and "omega_cmd" in self.action_spec.names else 0.0
            ),
            "probabilistic_obstacle_best_cost_emergency_probability": float(
                evaluated_candidates.maximum_step_probability[
                    best_cost_emergency_index
                ] if best_cost_emergency_index >= 0 else 0.0
            ),
            "probabilistic_obstacle_best_cost_emergency_probability_mass": float(
                evaluated_candidates.accumulated_probability_mass[
                    best_cost_emergency_index
                ] if best_cost_emergency_index >= 0 else 0.0
            ),
            "probabilistic_obstacle_minimum_risk_emergency_index": int(
                minimum_risk_emergency_index
            ),
            "probabilistic_obstacle_minimum_risk_emergency_v": float(
                values[minimum_risk_emergency_index, 0,
                       self.action_spec.index("v_cmd")]
                if minimum_risk_emergency_index >= 0
                and "v_cmd" in self.action_spec.names else 0.0
            ),
            "probabilistic_obstacle_minimum_risk_emergency_omega": float(
                values[minimum_risk_emergency_index, 0,
                       self.action_spec.index("omega_cmd")]
                if minimum_risk_emergency_index >= 0
                and "omega_cmd" in self.action_spec.names else 0.0
            ),
            "probabilistic_obstacle_minimum_risk_emergency_probability": float(
                evaluated_candidates.maximum_step_probability[
                    minimum_risk_emergency_index
                ] if minimum_risk_emergency_index >= 0 else 0.0
            ),
            "probabilistic_obstacle_minimum_risk_emergency_probability_mass": float(
                evaluated_candidates.accumulated_probability_mass[
                    minimum_risk_emergency_index
                ] if minimum_risk_emergency_index >= 0 else 0.0
            ),
            "probabilistic_obstacle_active_avoidance_enabled": bool(
                hard_action in (
                    "active_avoidance",
                    "active_avoidance_motion",
                )
            ),
        }
        if not fail_closed and speed_scale < 1.0:
            if "v_cmd" in self.action_spec.names:
                speed_index = self.action_spec.index("v_cmd")
                action[speed_index] *= speed_scale
                sequence[0, speed_index] = action[speed_index]
            else:
                action *= speed_scale
                sequence[0] = action
            trajectory = self.rollout(state, sequence)[0]
        return action, sequence, trajectory, diagnostics

    @staticmethod
    def _risk_is_strictly_better(
        candidate_maximum_probability,
        candidate_probability_mass,
        stop_maximum_probability,
        stop_probability_mass,
        allow_equal_maximum_mass=False,
    ):
        """Compare motion against stopping under the configured risk order."""

        candidate_maximum_probability = float(candidate_maximum_probability)
        candidate_probability_mass = float(candidate_probability_mass)
        stop_maximum_probability = float(stop_maximum_probability)
        stop_probability_mass = float(stop_probability_mass)
        values = np.asarray((
            candidate_maximum_probability,
            candidate_probability_mass,
            stop_maximum_probability,
            stop_probability_mass,
        ))
        if not np.isfinite(values).all():
            raise ValueError("probabilistic fallback risks must be finite")
        tolerance = 1.0e-12
        if candidate_maximum_probability < (
            stop_maximum_probability - tolerance
        ):
            return True
        return bool(
            allow_equal_maximum_mass
            and np.isclose(
                candidate_maximum_probability,
                stop_maximum_probability,
                atol=tolerance,
                rtol=0.0,
            )
            and candidate_probability_mass < stop_probability_mass - tolerance
        )

    def _path_boundary_margins(
        self,
        trajectories,
        reference,
    ):
        """Return spatial footprint clearance to the path corridor.

        Positive values mean that the circular planning footprint remains
        inside the corridor.  Each predicted state is projected onto the
        nearest admissible path segment with monotonic per-candidate progress.
        The corridor is spatial, not a time-indexed tube: a stationary braking
        candidate must not become infeasible merely because a preview point
        continues around a downstream bend.  This helper is shared by the soft
        path cost and the optional hard candidate filter.
        """

        paths = np.asarray(trajectories, dtype=np.float64)
        if paths.ndim != 3 or paths.shape[1] != self.config.horizon + 1:
            raise ValueError("boundary trajectories must have shape [K,H+1,nx]")
        footprint_radius = getattr(reference, "footprint_radius", None)
        if (
            not hasattr(reference, "points")
            or not hasattr(reference, "segment_lengths")
            or not hasattr(reference, "cumulative")
            or footprint_radius is None
            or getattr(reference, "corridor_half_width", None) is None
        ):
            raise ValueError(
                "path boundary enforcement requires corridor-aware reference"
            )
        xy = paths[..., list(self.state_spec.position_indices)]
        starts = np.asarray(reference.points[:-1], dtype=np.float64)
        vectors = np.diff(np.asarray(reference.points, dtype=np.float64), axis=0)
        lengths = np.asarray(reference.segment_lengths, dtype=np.float64)
        cumulative = np.asarray(reference.cumulative[:-1], dtype=np.float64)
        floors = np.full(paths.shape[0], float(reference.progress), dtype=np.float64)
        margins = np.empty(paths.shape[:2], dtype=np.float64)
        profile = np.asarray(
            getattr(reference, "corridor_half_width_profile", ()),
            dtype=np.float64,
        )
        for step in range(paths.shape[1]):
            positions = xy[:, step, :]
            relative = positions[:, None, :] - starts[None, :, :]
            fractions = np.sum(relative * vectors[None, :, :], axis=2)
            fractions = np.clip(fractions / lengths[None, :] ** 2, 0.0, 1.0)
            projections = starts[None, :, :] + fractions[..., None] * vectors[None, :, :]
            squared_distances = np.sum(
                (projections - positions[:, None, :]) ** 2, axis=2
            )
            candidate_progress = cumulative[None, :] + fractions * lengths[None, :]
            admissible = (
                candidate_progress
                >= floors[:, None] - float(reference.projection_backtrack_distance)
            ) & (
                candidate_progress
                <= floors[:, None] + float(reference.projection_forward_distance)
            )
            has_admissible = np.any(admissible, axis=1)
            nearest = np.argmin(
                np.where(admissible, squared_distances, np.inf), axis=1
            )
            projected = candidate_progress[np.arange(paths.shape[0]), nearest]
            projected = np.where(has_admissible, projected, floors)
            projected = np.maximum(floors, projected)
            reference_poses = reference.poses_at_progress(projected)
            tangent = reference_poses[:, 2]
            delta = positions - reference_poses[:, :2]
            lateral_error = (
                -np.sin(tangent) * delta[:, 0]
                + np.cos(tangent) * delta[:, 1]
            )
            if profile.size:
                half_widths = np.interp(
                    projected / float(reference.total_length),
                    profile[:, 0],
                    profile[:, 1],
                )
            else:
                half_widths = np.full(
                    paths.shape[0],
                    float(reference.corridor_half_width),
                    dtype=np.float64,
                )
            margins[:, step] = (
                half_widths - float(footprint_radius) - np.abs(lateral_error)
            )
            floors = projected
        if not np.isfinite(margins).all():
            raise FloatingPointError("path boundary margins contain NaN or Inf")
        return margins

    def _solve_plan(
        self, state, prior, target, obstacles, rng, observation=None,
        reference=None,
    ):
        tracker_diagnostics = (
            {}
            if observation is None
            else dict(
                observation.auxiliary.get(
                    "dynamic_obstacle_tracker", {}
                )
            )
        )
        known_static_obstacles = (
            ()
            if observation is None
            else tuple(
                observation.auxiliary.get(
                    "known_static_obstacles", ()
                )
            )
        )
        # Frozen static geometry is optional per live LiDAR frame.  If a
        # frame contains only dynamic/unknown returns, keep ordinary MPPI and
        # scan-derived filtering active and let known-map terms contribute
        # zero instead of aborting the armed run.
        known_static_map_cost_suppressed_no_geometry = bool(
            self.config.known_static_map_cost_enabled
            and not known_static_obstacles
        )
        probabilistic_obstacles = ()
        if self.config.probabilistic_obstacle_risk_enabled:
            if observation is None:
                raise ValueError(
                    "probabilistic obstacle risk requires an observation"
                )
            probabilistic_obstacles = tuple(
                observation.auxiliary.get(
                    self.config.probabilistic_obstacle_forecast_key, ()
                )
            )
            if not probabilistic_obstacles:
                missing_forecast_action = (
                    self.config
                    .probabilistic_obstacle_missing_forecast_action
                )
                if (
                    missing_forecast_action == "stop"
                ):
                    action = np.zeros(
                        self.action_spec.dimension, dtype=np.float64
                    )
                    sequence = np.zeros(
                        (
                            self.config.horizon,
                            self.action_spec.dimension,
                        ),
                        dtype=np.float64,
                    )
                    trajectory = self.rollout(state, sequence)[0]
                    diagnostics = {
                        "cost_min": 0.0,
                        "cost_mean": 0.0,
                        "cost_std": 0.0,
                        "effective_sample_size": 0.0,
                        "effective_sample_fraction": 0.0,
                        "reference_id": target.reference_id,
                        "target_x": float(target.pose.x),
                        "target_y": float(target.pose.y),
                        "target_theta": float(target.pose.theta),
                        "target_is_terminal": bool(
                            target.is_terminal
                        ),
                        "target_phase": str(target.phase),
                        "prior": dict(prior.metadata),
                        "probabilistic_obstacle_risk_enabled": True,
                        "probabilistic_obstacle_forecast_count": 0,
                        "probabilistic_obstacle_probability_mass": 0.0,
                        "probabilistic_obstacle_union_bound": 0.0,
                        "probabilistic_obstacle_maximum_step_probability": 0.0,
                        "probabilistic_obstacle_hard_violation": True,
                        "probabilistic_obstacle_fail_closed": True,
                        "probabilistic_obstacle_speed_scale": 0.0,
                        "dynamic_obstacle_tracker_enabled": bool(
                            tracker_diagnostics.get("enabled", False)
                        ),
                        "dynamic_obstacle_tracker_cluster_count": int(
                            tracker_diagnostics.get("cluster_count", 0)
                        ),
                        "dynamic_obstacle_tracker_associated": bool(
                            tracker_diagnostics.get(
                                "associated", False
                            )
                        ),
                        "dynamic_obstacle_tracker_association_distance_m": float(
                            tracker_diagnostics.get(
                                "association_distance_m", 0.0
                            )
                            or 0.0
                        ),
                        "dynamic_obstacle_tracker_measurement_x": float(
                            tracker_diagnostics.get(
                                "measurement_x", 0.0
                            )
                            or 0.0
                        ),
                        "dynamic_obstacle_tracker_measurement_y": float(
                            tracker_diagnostics.get(
                                "measurement_y", 0.0
                            )
                            or 0.0
                        ),
                        "dynamic_obstacle_tracker_support_beams": int(
                            tracker_diagnostics.get(
                                "selected_support_beams", 0
                            )
                        ),
                        "dynamic_obstacle_tracker_unobserved_duration_s": float(
                            tracker_diagnostics.get(
                                "unobserved_duration_s", 0.0
                            )
                        ),
                        "dynamic_obstacle_tracker_forecast_valid": False,
                        "dynamic_obstacle_tracker_forecast_availability": float(
                            tracker_diagnostics.get(
                                "forecast_availability", 0.0
                            )
                        ),
                        "dynamic_obstacle_tracker_innovation_nis": float(
                            tracker_diagnostics.get(
                                "innovation_nis", 0.0
                            )
                            or 0.0
                        ),
                        "dynamic_obstacle_tracker_change_triggered": bool(
                            tracker_diagnostics.get(
                                "change_triggered", False
                            )
                        ),
                        "dynamic_obstacle_tracker_dropout_guard_triggered": bool(
                            tracker_diagnostics.get(
                                "dropout_guard_triggered", False
                            )
                        ),
                        "dynamic_obstacle_tracker_recovery_active": bool(
                            tracker_diagnostics.get(
                                "recovery_active", False
                            )
                        ),
                    }
                    if self.config.profile_components:
                        diagnostics.update(
                            {
                                "profile_mppi_sampling_ms": 0.0,
                                "profile_mppi_batch_rollout_ms": 0.0,
                                "profile_mppi_cost_ms": 0.0,
                                "profile_mppi_weighting_update_ms": 0.0,
                                "profile_mppi_final_rollout_ms": 0.0,
                                "profile_mppi_solve_total_ms": 0.0,
                            }
                        )
                    return action, sequence, trajectory, diagnostics
                if missing_forecast_action == "raise":
                    raise ValueError(
                        "probabilistic obstacle risk is enabled but the "
                        "observation contains no forecasts"
                    )
            for forecast in probabilistic_obstacles:
                if not isinstance(
                    forecast, GaussianMixtureObstacleForecast
                ):
                    raise TypeError(
                        "observation contains an invalid obstacle forecast"
                    )
                if not np.isclose(
                    forecast.timestamp,
                    observation.timestamp,
                    atol=1.0e-9,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "obstacle forecast timestamp must match observation"
                    )
        emergency_context = {
            "triggered": False,
            "counterflow_escape_applied": False,
            "counterflow_rear_occupancy_vetoed": False,
            "preferred_escape_direction_x": 0.0,
            "preferred_escape_direction_y": 0.0,
            "preferred_escape_heading_error_rad": 0.0,
            "escape_direction_refreshed": False,
            "escape_direction_alignment": 1.0,
            "escape_direction_source": "unavailable",
        }
        if probabilistic_obstacles:
            # Standard MPPI uses the same causal forecast-relative emergency
            # lattice as the RL-driven optimizer. Previously it injected only
            # context-free templates, leaving the predictor-aware generator
            # unreachable in the nominal controller.
            emergency_context = self._probabilistic_emergency_context(
                observation, state, probabilistic_obstacles
            )
        profiling = self.config.profile_components
        solve_started = time.perf_counter() if profiling else None
        stage_started = solve_started
        profile = {}

        def mark(name):
            nonlocal stage_started
            if not profiling:
                return
            now = time.perf_counter()
            profile["profile_mppi_%s_ms" % name] = 1000.0 * (
                now - stage_started
            )
            stage_started = now

        samples = self._sample(prior, rng=rng)
        emergency_candidate_mask = np.zeros(
            self.config.num_samples, dtype=bool
        )
        hard_boundary_filter = bool(
            self.config.path_boundary_candidate_filter_enabled
        )
        hard_static_filter = bool(
            self.config.known_static_map_candidate_filter_enabled
            and known_static_obstacles
        )
        risk_candidate_filter = bool(
            self.config.probabilistic_obstacle_risk_enabled
            and probabilistic_obstacles
            and self.config
            .probabilistic_obstacle_candidate_filter_enabled
        )
        if (
            hard_boundary_filter
            or hard_static_filter
            or risk_candidate_filter
        ):
            # Reserve one of the already budgeted candidates for a deterministic
            # braking sequence.  This does not increase K and gives the
            # fail-closed branch a reproducible control sequence when every
            # stochastic proposal is footprint-infeasible.
            #
            # ``_sample`` reserves candidate zero for the exact prior mean.  Do
            # not silently discard that sole unperturbed warm start: in a long
            # control horizon, every Gaussian candidate can otherwise corrupt
            # a coherent turn/drive sequence.  Preserve it in candidate one
            # before installing the braking candidate.  Both candidates remain
            # inside the original K budget.
            if self.config.num_samples > 1:
                samples[1] = samples[0]
            samples[0] = 0.0
            samples[0, 0] = self.action_spec.clip(
                samples[0, 0], self.previous_action, self.config.dt
            )
        if (
            self.config.probabilistic_obstacle_risk_enabled
            and probabilistic_obstacles
            and self.config
            .probabilistic_obstacle_emergency_candidates_enabled
            and emergency_context["triggered"]
        ):
            emergency_candidate_mask = (
                self._inject_probabilistic_emergency_candidates(
                    samples, prior, emergency_context
                )
            )
        terminal_dx = float(
            target.pose.x - state[self.state_spec.position_indices[0]]
        )
        terminal_dy = float(
            target.pose.y - state[self.state_spec.position_indices[1]]
        )
        terminal_distance = float(np.hypot(terminal_dx, terminal_dy))
        target_bearing_error = 0.0
        if (
            "theta" in self.state_spec.names
            and terminal_distance > 1.0e-12
        ):
            target_heading = float(np.arctan2(
                terminal_dy, terminal_dx
            ))
            theta = float(state[self.state_spec.index("theta")])
            target_bearing_error = float(np.arctan2(
                np.sin(target_heading - theta),
                np.cos(target_heading - theta),
            ))
        terminal_control_region_active = bool(
            self.config.terminal_control_radius is None
            or terminal_distance <= self.config.terminal_control_radius
        )
        terminal_heading_gate_active = bool(
            self.config.terminal_translation_heading_gate_rad is not None
            and target.phase in ("terminal_approach", "terminal")
            and terminal_control_region_active
            and "v_cmd" in self.action_spec.names
            and "theta" in self.state_spec.names
        )
        terminal_bearing_error = 0.0
        terminal_translation_scale = 1.0
        if terminal_heading_gate_active:
            if terminal_distance > 1e-12:
                terminal_bearing_error = target_bearing_error
                gate = float(self.config.terminal_translation_heading_gate_rad)
                gate_cosine = float(np.cos(gate))
                if abs(terminal_bearing_error) >= gate:
                    terminal_translation_scale = 0.0
                else:
                    # Smooth recovery avoids a stop/go discontinuity as the
                    # body aligns with the terminal target bearing.
                    terminal_translation_scale = max(
                        0.0,
                        (float(np.cos(terminal_bearing_error)) - gate_cosine)
                        / max(1.0 - gate_cosine, 1e-12),
                    )
                # See MppiConfig.terminal_translation_minimum_scale: the gate
                # scales the final action, so a zero scale also cancels
                # evasive translation.  0.0 is the historical gate exactly.
                terminal_translation_scale = max(
                    terminal_translation_scale,
                    float(self.config.terminal_translation_minimum_scale),
                )
        terminal_speed_limit_active = bool(
            self.config.terminal_translation_speed_limit is not None
            and target.phase in ("terminal_approach", "terminal")
            and terminal_control_region_active
            and "v_cmd" in self.action_spec.names
        )
        if terminal_speed_limit_active:
            v_index = self.action_spec.index("v_cmd")
            samples[..., v_index] = np.minimum(
                samples[..., v_index],
                self.config.terminal_translation_speed_limit,
            )
        if terminal_heading_gate_active:
            v_index = self.action_spec.index("v_cmd")
            # The bearing gate is a current-step feasibility constraint.  The
            # remaining horizon stays free so MPPI can plan translation after
            # the predicted in-place alignment rather than becoming blind to
            # the value of turning.
            samples[:, 0, v_index] *= terminal_translation_scale
        if (
            hard_boundary_filter
            or hard_static_filter
            or risk_candidate_filter
        ):
            # Candidate rollouts must use the same first command that can
            # actually pass the actuator slew-rate contract.  Otherwise a
            # nominally feasible sample may become infeasible only after the
            # selected command is clipped below.
            regular_candidates = ~emergency_candidate_mask
            samples[regular_candidates, 0, :] = self.action_spec.clip(
                samples[regular_candidates, 0, :],
                self.previous_action,
                self.config.dt,
            )
        # Effective perturbations include actuator-bound clipping.  Expressing
        # the update this way makes the MPPI control law explicit while
        # remaining numerically equivalent to the historical weighted average.
        perturbations = samples - prior.mean[None, :, :]
        covariance = self._sampling_covariance(prior)
        mark("sampling")
        trajectories = self.rollout(state, samples)
        mark("batch_rollout")
        candidate_risk = None
        if (
            probabilistic_obstacles
            and (
                risk_candidate_filter
                or self.config
                .probabilistic_reference_authority_enabled
            )
        ):
            candidate_risk = self._probabilistic_collision_risk(
                trajectories, probabilistic_obstacles
            )
        reference_prior_index = (
            1
            if (
                self.config.num_samples > 1
                and (
                    hard_boundary_filter
                    or hard_static_filter
                    or risk_candidate_filter
                )
            )
            else 0
        )
        reference_authority, reference_risk_raw = (
            self._probabilistic_reference_authority(
                candidate_risk, reference_prior_index
            )
        )
        costs = self._cost(
            trajectories,
            samples,
            target,
            obstacles,
            reference=reference,
            probabilistic_obstacles=probabilistic_obstacles,
            known_static_obstacles=known_static_obstacles,
            probabilistic_risk=candidate_risk,
            reference_authority=reference_authority,
        )
        mark("cost")
        correction = self._importance_sampling_cost(prior.mean, perturbations, covariance)
        costs = costs + correction
        boundary_candidate_feasible = np.ones(
            self.config.num_samples, dtype=bool
        )
        boundary_candidate_min_margin = np.full(
            self.config.num_samples, np.nan, dtype=np.float64
        )
        if hard_boundary_filter:
            candidate_margins = self._path_boundary_margins(
                trajectories, reference
            )
            boundary_candidate_min_margin = np.min(
                candidate_margins[:, 1:], axis=1
            )
            boundary_candidate_feasible = boundary_candidate_min_margin >= 0.0
        static_candidate_feasible = np.ones(
            self.config.num_samples, dtype=bool
        )
        static_candidate_min_clearance = np.full(
            self.config.num_samples, np.nan, dtype=np.float64
        )
        if hard_static_filter:
            static_clearance = self._known_static_map_clearance(
                trajectories, known_static_obstacles
            )
            static_candidate_min_clearance = np.min(
                static_clearance[:, 1:], axis=1
            )
            static_candidate_feasible = (
                static_candidate_min_clearance >= 0.0
            )
        risk_candidate_feasible = np.ones(
            self.config.num_samples, dtype=bool
        )
        if risk_candidate_filter:
            if candidate_risk is None:
                candidate_risk = self._probabilistic_collision_risk(
                    trajectories, probabilistic_obstacles
                )
            risk_candidate_feasible = ~candidate_risk.hard_violation
        reverse_candidate_mask = np.zeros(
            self.config.num_samples, dtype=bool
        )
        reverse_candidate_prediction_feasible = np.ones(
            self.config.num_samples, dtype=bool
        )
        if (
            self.config
            .probabilistic_obstacle_reverse_candidate_filter_enabled
            and "v_cmd" in self.action_spec.names
        ):
            reverse_v_index = self.action_spec.index("v_cmd")
            reverse_candidate_mask = (
                samples[:, 0, reverse_v_index] < -1.0e-9
            )
            # When forecasts are available, a reverse command never regains
            # eligibility through the generic "no jointly feasible" fallback
            # if its predicted trajectory is a hard violation.  Static scan
            # geometry is already represented by static_candidate_feasible.
            reverse_candidate_prediction_feasible = (
                ~reverse_candidate_mask | risk_candidate_feasible
            )
        candidate_eligible = (
            boundary_candidate_feasible
            & static_candidate_feasible
            & reverse_candidate_prediction_feasible
        )
        jointly_feasible = (
            candidate_eligible & risk_candidate_feasible
        )
        beta = float(np.min(costs))
        exponent = np.clip(-(costs - beta) / self.config.temperature, -700.0, 0.0)
        if (
            risk_candidate_filter
            and np.any(jointly_feasible)
        ):
            optimizer_feasible = jointly_feasible
            exponent = np.where(jointly_feasible, exponent, -np.inf)
        elif (
            (hard_boundary_filter or hard_static_filter)
            and np.any(candidate_eligible)
        ):
            optimizer_feasible = candidate_eligible
            exponent = np.where(
                candidate_eligible, exponent, -np.inf
            )
        elif risk_candidate_filter and np.any(risk_candidate_feasible):
            optimizer_feasible = risk_candidate_feasible
            exponent = np.where(
                risk_candidate_feasible, exponent, -np.inf
            )
        else:
            optimizer_feasible = np.ones(
                self.config.num_samples, dtype=bool
            )
        weights = np.exp(exponent)

        weight_sum = float(np.sum(weights))
        if np.isfinite(weight_sum) and weight_sum > 0.0:
            weights /= weight_sum
        else:
            finite_costs = np.where(np.isfinite(costs), costs, np.inf)
            fallback_index = (
                int(np.argmin(finite_costs))
                if np.any(np.isfinite(finite_costs)) else 0
            )
            weights[:] = 0.0
            weights[fallback_index] = 1.0
        boundary_no_feasible_candidates = bool(
            hard_boundary_filter and not np.any(boundary_candidate_feasible)
        )
        static_no_feasible_candidates = bool(
            hard_static_filter and not np.any(static_candidate_feasible)
        )
        if (
            boundary_no_feasible_candidates
            or static_no_feasible_candidates
        ):
            # Candidate zero is the fixed braking sequence inserted above.
            # Use it deterministically instead of allowing an infeasible RL or
            # Gaussian proposal to dominate merely through a lower soft cost.
            weights[:] = 0.0
            weights[0] = 1.0
            optimizer_feasible = np.zeros(
                self.config.num_samples, dtype=bool
            )
            optimizer_feasible[0] = True
        v_index_diagnostic = (
            self.action_spec.index("v_cmd")
            if "v_cmd" in self.action_spec.names else 0
        )
        weighting_indices = np.flatnonzero(optimizer_feasible)
        candidate_diagnostics = reverse_candidate_diagnostics(
            samples=samples,
            costs=costs,
            optimizer_feasible=optimizer_feasible,
            jointly_feasible=jointly_feasible,
            static_feasible=static_candidate_feasible,
            risk_feasible=risk_candidate_feasible,
            weighting_indices=weighting_indices,
            weighting_weights=weights[weighting_indices],
            v_index=v_index_diagnostic,
            prefix_steps=12,
            weighting_population_kind="full_candidate_set",
        )
        weighted_perturbation = np.sum(weights[:, None, None] * perturbations, axis=0)
        sequence = prior.mean + weighted_perturbation
        sequence = np.clip(sequence, self.action_spec.lower, self.action_spec.upper)
        if terminal_speed_limit_active:
            sequence[:, v_index] = np.minimum(
                sequence[:, v_index],
                self.config.terminal_translation_speed_limit,
            )
        if terminal_heading_gate_active:
            sequence[0, v_index] *= terminal_translation_scale
        terminal_alignment_active = bool(
            terminal_heading_gate_active
            and self.config.terminal_alignment_yaw_gain is not None
            and "omega_cmd" in self.action_spec.names
        )
        terminal_alignment_omega = 0.0
        if terminal_alignment_active:
            omega_index = self.action_spec.index("omega_cmd")
            terminal_alignment_omega = (
                float(self.config.terminal_alignment_yaw_gain)
                * terminal_bearing_error
            )
            # Keep the local bearing controller authoritative throughout the
            # terminal phase.  Blending back to unconstrained MPPI at small
            # errors caused a limit cycle because position-only rollout costs
            # do not make the instantaneous turn direction identifiable.
            sequence[0, omega_index] = terminal_alignment_omega
        action = self.action_spec.clip(
            sequence[0], self.previous_action, self.config.dt
        )
        if terminal_speed_limit_active:
            action[v_index] = min(
                action[v_index], self.config.terminal_translation_speed_limit
            )
        if terminal_heading_gate_active:
            # Like an emergency translation stop, terminal alignment may need
            # to decelerate faster than the nominal command slew limit.  It
            # never changes angular control and therefore preserves the
            # rotate-in-place degree of freedom.
            action[v_index] *= terminal_translation_scale
        sequence[0] = action
        nominal_action_before_risk = np.asarray(action, dtype=np.float64).copy()
        mark("weighting_update")
        updated_trajectory = self.rollout(state, sequence)[0]
        boundary_weighted_update_feasible = True
        boundary_fallback_used = False
        boundary_fallback_candidate_index = -1
        if hard_boundary_filter:
            updated_margin = float(np.min(
                self._path_boundary_margins(
                    updated_trajectory[None, ...], reference
                )[0, 1:]
            ))
            boundary_weighted_update_feasible = bool(updated_margin >= 0.0)
            if (
                not boundary_weighted_update_feasible
                and np.any(boundary_candidate_feasible)
            ):
                feasible_indices = np.flatnonzero(boundary_candidate_feasible)
                boundary_fallback_candidate_index = int(
                    feasible_indices[np.argmin(costs[feasible_indices])]
                )
                sequence = samples[boundary_fallback_candidate_index].copy()
                action = self.action_spec.clip(
                    sequence[0], self.previous_action, self.config.dt
                )
                sequence[0] = action
                updated_trajectory = self.rollout(state, sequence)[0]
                boundary_fallback_used = True
                updated_margin = float(np.min(
                    self._path_boundary_margins(
                        updated_trajectory[None, ...], reference
                    )[0, 1:]
                ))
                boundary_weighted_update_feasible = bool(updated_margin >= 0.0)
        else:
            updated_margin = 0.0
        static_weighted_update_feasible = True
        static_fallback_used = False
        static_fallback_candidate_index = -1
        static_updated_min_clearance = 0.0
        if hard_static_filter:
            static_updated_min_clearance = float(np.min(
                self._known_static_map_clearance(
                    updated_trajectory[None, ...],
                    known_static_obstacles,
                )[0, 1:]
            ))
            static_weighted_update_feasible = bool(
                static_updated_min_clearance >= 0.0
            )
            if (
                not static_weighted_update_feasible
                and np.any(candidate_eligible)
            ):
                feasible_mask = candidate_eligible.copy()
                if (
                    risk_candidate_filter
                    and np.any(
                        feasible_mask & risk_candidate_feasible
                    )
                ):
                    feasible_mask &= risk_candidate_feasible
                feasible_indices = np.flatnonzero(feasible_mask)
                static_fallback_candidate_index = int(
                    feasible_indices[np.argmin(costs[feasible_indices])]
                )
                sequence = samples[
                    static_fallback_candidate_index
                ].copy()
                action = self.action_spec.clip(
                    sequence[0],
                    self.previous_action,
                    self.config.dt,
                )
                sequence[0] = action
                updated_trajectory = self.rollout(
                    state, sequence
                )[0]
                static_fallback_used = True
                static_updated_min_clearance = float(np.min(
                    self._known_static_map_clearance(
                        updated_trajectory[None, ...],
                        known_static_obstacles,
                    )[0, 1:]
                ))
                static_weighted_update_feasible = bool(
                    static_updated_min_clearance >= 0.0
                )
        optimizer_diagnostics = {
            "optimizer_diagnostics_enabled": bool(
                self.config.optimizer_diagnostics_enabled
            ),
            "optimizer_candidate_count": int(self.config.num_samples),
            "optimizer_boundary_feasible_count": int(
                np.sum(boundary_candidate_feasible)
            ),
            "optimizer_static_feasible_count": int(
                np.sum(static_candidate_feasible)
            ),
            "optimizer_risk_feasible_count": int(
                np.sum(risk_candidate_feasible)
            ),
            "optimizer_jointly_feasible_count": int(
                np.sum(jointly_feasible)
            ),
            "optimizer_emergency_candidate_count": int(
                np.sum(emergency_candidate_mask)
            ),
            "optimizer_reverse_candidate_filter_enabled": bool(
                self.config
                .probabilistic_obstacle_reverse_candidate_filter_enabled
            ),
            "optimizer_reverse_candidate_count": int(
                np.sum(reverse_candidate_mask)
            ),
            "optimizer_reverse_candidate_prediction_rejected_count": int(
                np.sum(
                    reverse_candidate_mask
                    & ~reverse_candidate_prediction_feasible
                )
            ),
            "optimizer_emergency_candidate_selected_index": -1,
            "optimizer_fallback_used": False,
            "optimizer_fallback_kind": "none",
            "optimizer_best_candidate_cost": 0.0,
            "optimizer_selected_sequence_cost": 0.0,
            "optimizer_selected_cost_gap": 0.0,
            "optimizer_best_first_v": 0.0,
            "optimizer_best_first_omega": 0.0,
            "optimizer_selected_first_v": 0.0,
            "optimizer_selected_first_omega": 0.0,
            "optimizer_first_action_cancellation_ratio": 1.0,
        }
        if self.config.optimizer_diagnostics_enabled:
            eligible = (
                np.flatnonzero(candidate_eligible)
                if (
                    (hard_boundary_filter or hard_static_filter)
                    and np.any(candidate_eligible)
                )
                else np.arange(self.config.num_samples, dtype=np.int64)
            )
            best_index = int(eligible[np.argmin(costs[eligible])])
            best_action = self.action_spec.clip(
                samples[best_index, 0], self.previous_action, self.config.dt
            )
            selected_running = float(self._cost(
                updated_trajectory[None, ...],
                sequence[None, ...],
                target,
                obstacles,
                reference=reference,
                probabilistic_obstacles=probabilistic_obstacles,
                known_static_obstacles=known_static_obstacles,
                reference_authority=reference_authority,
            )[0])
            selected_correction = float(self._importance_sampling_cost(
                prior.mean,
                (sequence - prior.mean)[None, ...],
                covariance,
            )[0])
            selected_cost = selected_running + selected_correction
            best_norm = float(np.linalg.norm(best_action))
            selected_norm = float(np.linalg.norm(action))
            v_index_diag = (
                self.action_spec.index("v_cmd")
                if "v_cmd" in self.action_spec.names else None
            )
            omega_index_diag = (
                self.action_spec.index("omega_cmd")
                if "omega_cmd" in self.action_spec.names else None
            )
            optimizer_diagnostics = {
                "optimizer_diagnostics_enabled": True,
                "optimizer_candidate_count": int(self.config.num_samples),
                "optimizer_boundary_feasible_count": int(
                    np.sum(boundary_candidate_feasible)
                ),
                "optimizer_static_feasible_count": int(
                    np.sum(static_candidate_feasible)
                ),
                "optimizer_risk_feasible_count": int(
                    np.sum(risk_candidate_feasible)
                ),
                "optimizer_jointly_feasible_count": int(
                    np.sum(jointly_feasible)
                ),
                "optimizer_emergency_candidate_count": int(
                    np.sum(emergency_candidate_mask)
                ),
                "optimizer_reverse_candidate_filter_enabled": bool(
                    self.config
                    .probabilistic_obstacle_reverse_candidate_filter_enabled
                ),
                "optimizer_reverse_candidate_count": int(
                    np.sum(reverse_candidate_mask)
                ),
                "optimizer_reverse_candidate_prediction_rejected_count": int(
                    np.sum(
                        reverse_candidate_mask
                        & ~reverse_candidate_prediction_feasible
                    )
                ),
                "optimizer_emergency_candidate_selected_index": -1,
                "optimizer_fallback_used": False,
                "optimizer_fallback_kind": "none",
                "optimizer_best_candidate_cost": float(costs[best_index]),
                "optimizer_selected_sequence_cost": selected_cost,
                "optimizer_selected_cost_gap": (
                    selected_cost - float(costs[best_index])
                ),
                "optimizer_best_first_v": (
                    float(best_action[v_index_diag])
                    if v_index_diag is not None else 0.0
                ),
                "optimizer_best_first_omega": (
                    float(best_action[omega_index_diag])
                    if omega_index_diag is not None else 0.0
                ),
                "optimizer_selected_first_v": (
                    float(action[v_index_diag])
                    if v_index_diag is not None else 0.0
                ),
                "optimizer_selected_first_omega": (
                    float(action[omega_index_diag])
                    if omega_index_diag is not None else 0.0
                ),
                "optimizer_first_action_cancellation_ratio": (
                    selected_norm / max(best_norm, 1e-12)
                ),
            }
        mark("final_rollout")
        fallback_candidate_index = -1
        fallback_used = False
        fallback_kind = "none"
        if (
            self.config.probabilistic_obstacle_risk_enabled
            and probabilistic_obstacles
        ):
            selected_risk = self._probabilistic_collision_risk(
                updated_trajectory[None, ...],
                probabilistic_obstacles,
            )
            initial_hard_violation = bool(
                selected_risk.hard_violation[0]
            )
            fallback_used = False
            fallback_candidate_index = -1
            fallback_kind = "none"
            risk_equivalent_forward_progress_applied = False
            stop_maximum_probability = 0.0
            stop_probability_mass = 0.0
            fail_closed = False
            hard_action = (
                self.config
                .probabilistic_obstacle_hard_violation_action
            )
            if initial_hard_violation and hard_action in (
                "active_avoidance",
                "active_avoidance_motion",
            ):
                if candidate_risk is None:
                    candidate_risk = self._probabilistic_collision_risk(
                        trajectories, probabilistic_obstacles
                    )
                    risk_candidate_feasible = (
                        ~candidate_risk.hard_violation
                    )
                eligible = np.flatnonzero(candidate_eligible)
                if eligible.size == 0:
                    eligible = np.arange(
                        self.config.num_samples, dtype=np.int64
                    )
                feasible = eligible[
                    risk_candidate_feasible[eligible]
                ]
                stop_sequence = np.zeros(
                    (
                        self.config.horizon,
                        self.action_spec.dimension,
                    ),
                    dtype=np.float64,
                )
                stop_trajectory = self.rollout(
                    state, stop_sequence
                )[0]
                stop_risk = self._probabilistic_collision_risk(
                    stop_trajectory[None, ...],
                    probabilistic_obstacles,
                )
                stop_maximum_probability = float(
                    stop_risk.maximum_step_probability[0]
                )
                stop_probability_mass = float(
                    stop_risk.accumulated_probability_mass[0]
                )
                if feasible.size:
                    fallback_candidate_index = int(
                        feasible[np.argmin(costs[feasible])]
                    )
                    fallback_kind = "feasible_active_candidate"
                else:
                    candidate_order = np.lexsort(
                        (
                            costs[eligible],
                            candidate_risk
                            .accumulated_probability_mass[eligible],
                            candidate_risk
                            .maximum_step_probability[eligible],
                        )
                    )
                    safest_index = int(eligible[candidate_order[0]])
                    safest_probability = float(
                        candidate_risk
                        .maximum_step_probability[safest_index]
                    )
                    safest_probability_mass = float(
                        candidate_risk
                        .accumulated_probability_mass[safest_index]
                    )
                    if self._risk_is_strictly_better(
                        safest_probability,
                        safest_probability_mass,
                        stop_maximum_probability,
                        stop_probability_mass,
                        allow_equal_maximum_mass=(
                            self.config
                            .probabilistic_obstacle_stopping_feasibility_enabled
                        ),
                    ):
                        fallback_candidate_index = safest_index
                        fallback_kind = (
                            "minimum_accumulated_risk_active_candidate"
                            if np.isclose(
                                safest_probability,
                                stop_maximum_probability,
                                atol=1.0e-12,
                                rtol=0.0,
                            )
                            else "minimum_risk_active_candidate"
                        )
                    if (
                        hard_action == "active_avoidance_motion"
                        and self.config
                        .probabilistic_obstacle_hard_violation_progress_tiebreak_enabled
                        and "v_cmd" in self.action_spec.names
                    ):
                        v_index = self.action_spec.index("v_cmd")
                        mass_limit = safest_probability_mass * (
                            1.0
                            + self.config
                            .probabilistic_obstacle_hard_violation_progress_mass_tolerance
                        ) + 1.0e-12
                        progress_mask = (
                            candidate_eligible
                            & (
                                candidate_risk.maximum_step_probability
                                <= safest_probability + 1.0e-12
                            )
                            & (
                                candidate_risk.accumulated_probability_mass
                                <= mass_limit
                            )
                            & (samples[:, 0, v_index] >= 0.0)
                        )
                        progress_indices = np.flatnonzero(progress_mask)
                        if progress_indices.size:
                            progress_order = np.lexsort(
                                (
                                    costs[progress_indices],
                                    -samples[progress_indices, 0, v_index],
                                )
                            )
                            progress_index = int(
                                progress_indices[progress_order[0]]
                            )
                            if (
                                samples[progress_index, 0, v_index]
                                > samples[safest_index, 0, v_index]
                                + 1.0e-12
                            ):
                                fallback_candidate_index = progress_index
                                fallback_kind = (
                                    "risk_equivalent_forward_progress_candidate"
                                )
                                risk_equivalent_forward_progress_applied = True
                    else:
                        fallback_kind = "stop_is_safest_candidate"
                fallback_used = True
                if fallback_candidate_index >= 0:
                    sequence = samples[
                        fallback_candidate_index
                    ].copy()
                    # Emergency candidates are generated with a slew-limited
                    # prefix when enabled, and the selected first command is
                    # always clipped once more at the final authority
                    # boundary.  This prevents a direct sign flip even when a
                    # fallback is selected after risk filtering.
                    action = self.action_spec.clip(
                        sequence[0],
                        self.previous_action,
                        self.config.dt,
                    )
                    sequence[0] = action
                    updated_trajectory = self.rollout(
                        state, sequence
                    )[0]
                else:
                    action = np.zeros(
                        self.action_spec.dimension, dtype=np.float64
                    )
                    sequence = stop_sequence
                    updated_trajectory = stop_trajectory
                    fail_closed = True
                selected_risk = self._probabilistic_collision_risk(
                    updated_trajectory[None, ...],
                    probabilistic_obstacles,
                )
            hard_violation = bool(selected_risk.hard_violation[0])
            if hard_violation and hard_action == "stop":
                fail_closed = True
                fallback_used = True
                fallback_kind = "unconditional_stop"
                action = np.zeros(
                    self.action_spec.dimension, dtype=np.float64
                )
                sequence = np.zeros(
                    (
                        self.config.horizon,
                        self.action_spec.dimension,
                    ),
                    dtype=np.float64,
                )
                updated_trajectory = self.rollout(state, sequence)[0]
            maximum_step_probability = float(
                selected_risk.maximum_step_probability[0]
            )
            speed_scale = 1.0
            active_avoidance_motion_selected = bool(
                hard_action == "active_avoidance_motion"
                and fallback_used
                and fallback_candidate_index >= 0
            )
            if (
                self.config
                .probabilistic_obstacle_speed_governor_enabled
                and not active_avoidance_motion_selected
            ):
                hard_threshold = (
                    self.config.probabilistic_obstacle_hard_threshold
                )
                soft_threshold = (
                    self.config
                    .probabilistic_obstacle_speed_governor_start_ratio
                    * hard_threshold
                )
                speed_scale = float(np.clip(
                    (hard_threshold - maximum_step_probability)
                    / (hard_threshold - soft_threshold),
                    0.0,
                    1.0,
                ))
            probabilistic_risk_diagnostics = {
                "probabilistic_obstacle_risk_enabled": True,
                "probabilistic_obstacle_forecast_count": len(
                    probabilistic_obstacles
                ),
                "probabilistic_obstacle_probability_mass": float(
                    selected_risk.accumulated_probability_mass[0]
                ),
                "probabilistic_obstacle_union_bound": float(
                    selected_risk.horizon_union_bound[0]
                ),
                "probabilistic_obstacle_maximum_step_probability": float(
                    selected_risk.maximum_step_probability[0]
                ),
                "probabilistic_obstacle_hard_violation": hard_violation,
                "probabilistic_obstacle_fail_closed": fail_closed,
                "probabilistic_obstacle_speed_scale": speed_scale,
                "probabilistic_obstacle_speed_governor_bypassed_for_active_avoidance": (
                    active_avoidance_motion_selected
                ),
                "probabilistic_obstacle_initial_hard_violation": (
                    initial_hard_violation
                ),
                "probabilistic_obstacle_nominal_first_action": [
                    float(value) for value in nominal_action_before_risk
                ],
                "probabilistic_obstacle_selected_first_action": [
                    float(value) for value in action
                ],
                "probabilistic_obstacle_fallback_action_delta_norm": float(
                    np.linalg.norm(action - nominal_action_before_risk)
                ),
                "probabilistic_obstacle_candidate_feasible_fraction": float(
                    np.mean(risk_candidate_feasible)
                    if candidate_risk is not None
                    else 1.0
                ),
                "probabilistic_obstacle_active_fallback_used": (
                    fallback_used
                ),
                "probabilistic_obstacle_active_fallback_index": (
                    fallback_candidate_index
                ),
                "probabilistic_obstacle_active_fallback_kind": (
                    fallback_kind
                ),
                "probabilistic_obstacle_risk_equivalent_forward_progress_applied": (
                    risk_equivalent_forward_progress_applied
                ),
                "probabilistic_obstacle_stop_maximum_probability": (
                    stop_maximum_probability
                ),
                "probabilistic_obstacle_stop_probability_mass": (
                    stop_probability_mass
                ),
                "probabilistic_obstacle_stopping_feasibility_enabled": bool(
                    self.config
                    .probabilistic_obstacle_stopping_feasibility_enabled
                ),
                "probabilistic_obstacle_emergency_candidate_count": int(
                    np.sum(emergency_candidate_mask)
                ),
                "probabilistic_obstacle_emergency_candidate_direct_fallback_enabled": bool(
                    self.config
                    .probabilistic_obstacle_emergency_candidate_direct_fallback_enabled
                ),
                "probabilistic_obstacle_emergency_candidate_direct_fallback_suppressed": bool(
                    emergency_context.get("triggered", False)
                    and not self.config
                    .probabilistic_obstacle_emergency_candidate_direct_fallback_enabled
                ),
                "probabilistic_obstacle_emergency_candidate_mppi_weight_mass": float(
                    np.sum(weights[emergency_candidate_mask])
                    if emergency_candidate_mask.any()
                    else 0.0
                ),
                "probabilistic_obstacle_emergency_candidate_slew_enabled": bool(
                    self.config
                    .probabilistic_obstacle_emergency_candidate_slew_enabled
                ),
                "probabilistic_obstacle_emergency_candidate_prefix_steps": int(
                    self.config
                    .probabilistic_obstacle_emergency_candidate_prefix_steps
                ),
                "probabilistic_obstacle_emergency_candidate_intent_hold_steps": int(
                    self.config
                    .probabilistic_obstacle_emergency_candidate_intent_hold_steps
                ),
                "probabilistic_obstacle_emergency_candidate_selected": bool(
                    fallback_candidate_index >= 0
                    and emergency_candidate_mask[fallback_candidate_index]
                ),
                "probabilistic_obstacle_temporal_emergency_triggered": bool(
                    emergency_context["triggered"]
                ),
                "probabilistic_obstacle_emergency_raw_triggered": bool(
                    emergency_context.get("raw_triggered", False)
                ),
                "probabilistic_obstacle_emergency_scan_quality_ok": bool(
                    emergency_context.get("scan_quality_ok", False)
                ),
                "probabilistic_obstacle_emergency_scan_support_beams": int(
                    emergency_context.get("scan_support_beams", 0) or 0
                ),
                "probabilistic_obstacle_emergency_scan_rejected_jump_fraction": float(
                    emergency_context.get("scan_rejected_jump_fraction", 1.0)
                    or 0.0
                ),
                "probabilistic_obstacle_emergency_forecast_evidence_ok": bool(
                    emergency_context.get("forecast_evidence_ok", False)
                ),
                "probabilistic_obstacle_emergency_rearm_ready": bool(
                    emergency_context.get("rearm_ready", False)
                ),
                "probabilistic_obstacle_emergency_rearm_clear_count": int(
                    emergency_context.get("rearm_clear_count", 0) or 0
                ),
                "probabilistic_obstacle_temporal_emergency_vetted": bool(
                    emergency_context["triggered"]
                    and fallback_candidate_index >= 0
                    and emergency_candidate_mask[
                        fallback_candidate_index
                    ]
                ),
                "probabilistic_obstacle_counterflow_escape_applied": bool(
                    emergency_context.get(
                        "counterflow_escape_applied", False
                    )
                ),
                "probabilistic_obstacle_counterflow_rear_occupancy_vetoed": bool(
                    emergency_context.get(
                        "counterflow_rear_occupancy_vetoed", False
                    )
                ),
                "probabilistic_obstacle_preferred_escape_direction_x": float(
                    emergency_context.get(
                        "preferred_escape_direction_x", 0.0
                    )
                ),
                "probabilistic_obstacle_preferred_escape_direction_y": float(
                    emergency_context.get(
                        "preferred_escape_direction_y", 0.0
                    )
                ),
                "probabilistic_obstacle_preferred_escape_heading_error_rad": float(
                    emergency_context.get(
                        "preferred_escape_heading_error_rad", 0.0
                    ) or 0.0
                ),
                "probabilistic_obstacle_escape_direction_refreshed": bool(
                    emergency_context.get(
                        "escape_direction_refreshed", False
                    )
                ),
                "probabilistic_obstacle_escape_direction_alignment": float(
                    emergency_context.get(
                        "escape_direction_alignment", 1.0
                    )
                ),
                "probabilistic_obstacle_escape_direction_source": str(
                    emergency_context.get(
                        "escape_direction_source", "unavailable"
                    )
                ),
                "probabilistic_obstacle_active_avoidance_enabled": bool(
                    hard_action in (
                        "active_avoidance",
                        "active_avoidance_motion",
                    )
                ),
                "probabilistic_obstacle_risk_backend": str(
                    self._last_probabilistic_risk_backend
                ),
            }
            if not fail_closed and speed_scale < 1.0:
                if "v_cmd" in self.action_spec.names:
                    speed_index = self.action_spec.index("v_cmd")
                    action[speed_index] *= speed_scale
                    sequence[0, speed_index] = action[speed_index]
                else:
                    action *= speed_scale
                    sequence[0] = action
                updated_trajectory = self.rollout(state, sequence)[0]
        else:
            self._last_probabilistic_risk_backend = "disabled"
            probabilistic_risk_diagnostics = {
                "probabilistic_obstacle_risk_enabled": bool(
                    self.config.probabilistic_obstacle_risk_enabled
                ),
                "probabilistic_obstacle_scan_only_fallback": bool(
                    self.config.probabilistic_obstacle_risk_enabled
                    and not probabilistic_obstacles
                    and self.config
                    .probabilistic_obstacle_missing_forecast_action
                    == "scan_only"
                ),
                "probabilistic_obstacle_forecast_count": 0,
                "probabilistic_obstacle_probability_mass": 0.0,
                "probabilistic_obstacle_union_bound": 0.0,
                "probabilistic_obstacle_maximum_step_probability": 0.0,
                "probabilistic_obstacle_hard_violation": False,
                "probabilistic_obstacle_fail_closed": False,
                "probabilistic_obstacle_speed_scale": 1.0,
                "probabilistic_obstacle_initial_hard_violation": False,
                "probabilistic_obstacle_candidate_feasible_fraction": 1.0,
                "probabilistic_obstacle_active_fallback_used": False,
                "probabilistic_obstacle_active_fallback_index": -1,
                "probabilistic_obstacle_active_fallback_kind": "none",
                "probabilistic_obstacle_risk_equivalent_forward_progress_applied": False,
                "probabilistic_obstacle_stop_maximum_probability": 0.0,
                "probabilistic_obstacle_stop_probability_mass": 0.0,
                "probabilistic_obstacle_stopping_feasibility_enabled": False,
                "probabilistic_obstacle_emergency_candidate_count": 0,
                "probabilistic_obstacle_emergency_candidate_selected": False,
                "probabilistic_obstacle_temporal_emergency_triggered": False,
                "probabilistic_obstacle_temporal_emergency_vetted": False,
                "probabilistic_obstacle_counterflow_escape_applied": False,
                "probabilistic_obstacle_counterflow_rear_occupancy_vetoed": False,
                "probabilistic_obstacle_preferred_escape_direction_x": 0.0,
                "probabilistic_obstacle_preferred_escape_direction_y": 0.0,
                "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
                "probabilistic_obstacle_escape_direction_refreshed": False,
                "probabilistic_obstacle_escape_direction_alignment": 1.0,
                "probabilistic_obstacle_escape_direction_source": "unavailable",
                "probabilistic_obstacle_active_avoidance_enabled": False,
                "probabilistic_obstacle_risk_backend": str(
                    self._last_probabilistic_risk_backend
                ),
            }
        online_tracker_diagnostics = {
            "dynamic_obstacle_tracker_enabled": bool(
                tracker_diagnostics.get("enabled", False)
            ),
            "dynamic_obstacle_tracker_cluster_count": int(
                tracker_diagnostics.get("cluster_count", 0)
            ),
            "dynamic_obstacle_tracker_associated": bool(
                tracker_diagnostics.get("associated", False)
            ),
            "dynamic_obstacle_tracker_association_distance_m": float(
                tracker_diagnostics.get(
                    "association_distance_m", 0.0
                )
                or 0.0
            ),
            "dynamic_obstacle_tracker_measurement_x": float(
                tracker_diagnostics.get("measurement_x", 0.0) or 0.0
            ),
            "dynamic_obstacle_tracker_measurement_y": float(
                tracker_diagnostics.get("measurement_y", 0.0) or 0.0
            ),
            "dynamic_obstacle_tracker_support_beams": int(
                tracker_diagnostics.get("selected_support_beams", 0)
            ),
            "dynamic_obstacle_tracker_unobserved_duration_s": float(
                tracker_diagnostics.get("unobserved_duration_s", 0.0)
            ),
            "dynamic_obstacle_tracker_forecast_valid": bool(
                tracker_diagnostics.get("forecast_valid", False)
            ),
            "dynamic_obstacle_tracker_forecast_availability": float(
                tracker_diagnostics.get("forecast_availability", 0.0)
            ),
            "dynamic_obstacle_tracker_innovation_nis": float(
                tracker_diagnostics.get("innovation_nis", 0.0) or 0.0
            ),
            "dynamic_obstacle_tracker_change_triggered": bool(
                tracker_diagnostics.get("change_triggered", False)
            ),
            "dynamic_obstacle_tracker_dropout_guard_triggered": bool(
                tracker_diagnostics.get(
                    "dropout_guard_triggered", False
                )
            ),
            "dynamic_obstacle_tracker_recovery_active": bool(
                tracker_diagnostics.get("recovery_active", False)
            ),
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
        known_static_minimum_clearance = 0.0
        if (
            self.config.known_static_map_cost_enabled
            and known_static_obstacles
        ):
            known_static_minimum_clearance = float(np.min(
                self._known_static_map_clearance(
                    updated_trajectory[None, ...],
                    known_static_obstacles,
                )[0, 1:]
            ))
        if self.config.optimizer_diagnostics_enabled:
            optimizer_diagnostics.update({
                "optimizer_emergency_candidate_selected_index": int(
                    fallback_candidate_index
                    if fallback_candidate_index >= 0
                    and emergency_candidate_mask[fallback_candidate_index]
                    else -1
                ),
                "optimizer_fallback_used": bool(fallback_used),
                "optimizer_fallback_kind": str(fallback_kind),
            })
        diagnostics = {
            "cost_min": float(costs.min()),
            "cost_mean": float(costs.mean()),
            "cost_std": float(costs.std()),
            "cost_q10": float(np.quantile(costs, 0.10)),
            "cost_q50": float(np.quantile(costs, 0.50)),
            "cost_q90": float(np.quantile(costs, 0.90)),
            "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
            "effective_sample_fraction": float(
                (1.0 / np.sum(weights ** 2)) / self.config.num_samples
            ),
            "path_boundary_candidate_filter_enabled": hard_boundary_filter,
            "path_boundary_candidate_feasible_count": int(
                np.sum(boundary_candidate_feasible)
            ),
            "path_boundary_candidate_feasible_fraction": float(
                np.mean(boundary_candidate_feasible)
            ),
            "path_boundary_candidate_min_margin": float(
                np.nanmin(boundary_candidate_min_margin)
                if hard_boundary_filter else 0.0
            ),
            "path_boundary_no_feasible_candidates": (
                boundary_no_feasible_candidates
            ),
            "path_boundary_weighted_update_feasible": (
                boundary_weighted_update_feasible
            ),
            "path_boundary_final_min_margin": updated_margin,
            "path_boundary_fallback_used": boundary_fallback_used,
            "path_boundary_fallback_candidate_index": (
                boundary_fallback_candidate_index
            ),
            "known_static_map_candidate_filter_enabled": (
                hard_static_filter
            ),
            "known_static_map_candidate_feasible_count": int(
                np.sum(static_candidate_feasible)
            ),
            "known_static_map_candidate_feasible_fraction": float(
                np.mean(static_candidate_feasible)
            ),
            "known_static_map_candidate_min_clearance": float(
                np.nanmin(static_candidate_min_clearance)
                if hard_static_filter else 0.0
            ),
            "known_static_map_no_feasible_candidates": (
                static_no_feasible_candidates
            ),
            "known_static_map_weighted_update_feasible": (
                static_weighted_update_feasible
            ),
            "known_static_map_fallback_used": static_fallback_used,
            "known_static_map_fallback_candidate_index": int(
                static_fallback_candidate_index
            ),
            "known_static_map_weighted_update_min_clearance": float(
                static_updated_min_clearance
            ),
            "known_static_map_cost_enabled": bool(
                self.config.known_static_map_cost_enabled
            ),
            "known_static_map_cost_suppressed_no_geometry": (
                known_static_map_cost_suppressed_no_geometry
            ),
            "known_static_map_obstacle_count": len(
                known_static_obstacles
            ),
            "known_static_map_minimum_clearance": (
                known_static_minimum_clearance
            ),
            "probabilistic_reference_authority_enabled": bool(
                self.config
                .probabilistic_reference_authority_enabled
            ),
            "probabilistic_reference_risk_raw": float(
                reference_risk_raw
            ),
            "probabilistic_reference_risk_filtered": float(
                self._probabilistic_reference_risk_filtered
            ),
            "probabilistic_reference_authority": float(
                reference_authority
            ),
            "probabilistic_reference_progress_weight": float(
                self.config.probabilistic_reference_progress_weight
            ),
            **candidate_diagnostics,
            **optimizer_diagnostics,
            "importance_sampling_correction": bool(
                self.config.importance_sampling_correction
            ),
            "importance_cost_mean": float(correction.mean()),
            "weighted_perturbation_norm": float(
                np.linalg.norm(weighted_perturbation)
            ),
            "sample_saturation_fraction": float(
                np.mean(
                    (samples <= self.action_spec.lower[None, None, :])
                    | (samples >= self.action_spec.upper[None, None, :])
                )
            ),
            "reference_id": target.reference_id,
            "target_x": float(target.pose.x),
            "target_y": float(target.pose.y),
            "target_theta": float(target.pose.theta),
            "target_is_terminal": bool(target.is_terminal),
            "target_phase": str(target.phase),
            "terminal_speed_limit_active": terminal_speed_limit_active,
            "terminal_translation_speed_limit": (
                0.0
                if self.config.terminal_translation_speed_limit is None
                else float(self.config.terminal_translation_speed_limit)
            ),
            "terminal_heading_gate_active": terminal_heading_gate_active,
            "terminal_translation_heading_gate_rad": (
                0.0
                if self.config.terminal_translation_heading_gate_rad is None
                else float(self.config.terminal_translation_heading_gate_rad)
            ),
            "terminal_bearing_error": terminal_bearing_error,
            "target_bearing_error": target_bearing_error,
            "terminal_translation_scale": terminal_translation_scale,
            "terminal_alignment_active": terminal_alignment_active,
            "terminal_alignment_yaw_gain": (
                0.0
                if self.config.terminal_alignment_yaw_gain is None
                else float(self.config.terminal_alignment_yaw_gain)
            ),
            "terminal_alignment_omega": terminal_alignment_omega,
            "terminal_control_radius": (
                0.0
                if self.config.terminal_control_radius is None
                else float(self.config.terminal_control_radius)
            ),
            "terminal_control_distance": terminal_distance,
            "terminal_control_region_active": terminal_control_region_active,
            "prior": dict(prior.metadata),
            **probabilistic_risk_diagnostics,
            **online_tracker_diagnostics,
        }
        residual = getattr(self.dynamics, "residual", None)
        confidence = getattr(residual, "confidence", None)
        if callable(confidence):
            support_controls = self._prediction_controls(
                sequence[None, :, :]
            )[0]
            support = np.asarray(
                confidence(updated_trajectory[:-1], support_controls), dtype=np.float64
            ).reshape(-1)
            diagnostics.update({
                "residual_support_confidence_mean": float(np.mean(support)),
                "residual_support_confidence_min": float(np.min(support)),
                "residual_support_reduced_fraction": float(np.mean(support < 1.0 - 1e-12)),
                "residual_support_disabled_fraction": float(np.mean(support <= 1e-12)),
            })
        else:
            diagnostics.update({
                "residual_support_confidence_mean": 1.0,
                "residual_support_confidence_min": 1.0,
                "residual_support_reduced_fraction": 0.0,
                "residual_support_disabled_fraction": 0.0,
            })
        reliability = getattr(residual, "diagnostics", None)
        if callable(reliability):
            diagnostics.update(reliability())
        else:
            diagnostics.update({
                "residual_reliability_enabled": False,
                "residual_reliability_alpha": 1.0 if residual is not None else 0.0,
                "residual_reliability_evidence_alpha": 1.0 if residual is not None else 0.0,
                "residual_reliability_context_alpha": 1.0,
                "residual_reliability_context_value": 0.0,
                "residual_reliability_samples": 0,
                "residual_reliability_mean_improvement": 0.0,
                "residual_reliability_lcb": 0.0,
                "residual_reliability_last_relative_improvement": 0.0,
                "residual_reliability_nominal_error": 0.0,
                "residual_reliability_residual_error": 0.0,
            })
        # Control-level rear guard, applied last so it governs whatever the
        # planner finally chose, regardless of which mechanism produced it.
        reverse_guard_active = False
        reverse_guard_before = 0.0
        reverse_guard_after = 0.0
        if (
            self.config.probabilistic_obstacle_reverse_rear_guard_enabled
            and "v_cmd" in self.action_spec.names
        ):
            v_index = self.action_spec.names.index("v_cmd")
            commanded = float(np.asarray(action, dtype=np.float64)[v_index])
            if commanded < 0.0 and self._rear_obstacle_blocks_reverse(
                state, probabilistic_obstacles
            ):
                scale = float(
                    self.config
                    .probabilistic_obstacle_reverse_rear_guard_scale
                )
                if not np.isfinite(scale) or scale < 0.0:
                    scale = 0.0
                action = np.array(action, dtype=np.float64, copy=True)
                action[v_index] = commanded * min(scale, 1.0)
                reverse_guard_active = True
                reverse_guard_before = commanded
                reverse_guard_after = float(action[v_index])
        diagnostics.update({
            "probabilistic_obstacle_reverse_rear_guard_active": bool(
                reverse_guard_active
            ),
            "probabilistic_obstacle_reverse_rear_guard_v_before": float(
                reverse_guard_before
            ),
            "probabilistic_obstacle_reverse_rear_guard_v_after": float(
                reverse_guard_after
            ),
        })
        if profiling:
            profile["profile_mppi_solve_total_ms"] = 1000.0 * (
                time.perf_counter() - solve_started
            )
            diagnostics.update(profile)
        return action, sequence, updated_trajectory, diagnostics

    def preview_plan(self, observation: RobotObservation, reference) -> PlanResult:
        """Evaluate the next MPPI plan without advancing controller state.

        A cloned RNG supplies common random numbers for paired candidate-prior
        comparisons.  The real controller RNG, receding-horizon sequence and
        previous action are not changed.  Stateful prior/reference restoration
        is handled by the offline environment wrapper that owns those objects.
        """

        started = time.perf_counter()
        state = self.state_from_observation(observation)
        target = reference.target_at(observation.timestamp, state)
        state_reference_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        prior = self._prior(observation, reference)
        prior_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        preview_rng = np.random.RandomState()
        preview_rng.set_state(self.rng.get_state())
        action, sequence, trajectory, diagnostics = self._solve_plan(
            state,
            prior,
            target,
            observation.local_obstacles,
            preview_rng,
            observation,
            reference,
        )
        diagnostics["compute_ms"] = 1000.0 * (
            time.perf_counter() - started
        )
        diagnostics.update(
            {
                "planner_observation_x": float(
                    state[self.state_spec.position_indices[0]]
                ),
                "planner_observation_y": float(
                    state[self.state_spec.position_indices[1]]
                ),
                "planner_observation_theta": float(
                    state[self.state_spec.index("theta")]
                    if "theta" in self.state_spec.names
                    else observation.pose.theta
                ),
            }
        )
        if self.config.profile_components:
            diagnostics.update({
                "profile_planner_state_reference_ms": 1000.0 * (
                    state_reference_finished - started
                ),
                "profile_planner_prior_ms": 1000.0 * (
                    prior_finished - state_reference_finished
                ),
            })
        diagnostics["preview"] = True
        return PlanResult(
            proposed_control=ControlCommand(action, observation.timestamp, "mppi_preview"),
            control_sequence=sequence,
            predicted_trajectory=trajectory,
            diagnostics=diagnostics,
        )

    def plan(self, observation: RobotObservation, reference) -> PlanResult:
        started = time.perf_counter()
        state = self.state_from_observation(observation)
        self._observe_residual_reliability(state)
        self._reliability_previous_state = state.copy()
        static_replan = self._maybe_replan_static_reference(
            state, observation, reference
        )
        target = reference.target_at(observation.timestamp, state)
        self._observe_residual_context(state, target)
        state_reference_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        prior = self._prior(observation, reference)
        prior_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        action, sequence, updated_trajectory, diagnostics = self._solve_plan(
            state,
            prior,
            target,
            observation.local_obstacles,
            self.rng,
            observation,
            reference,
        )
        current_risk = float(
            diagnostics.get(
                "probabilistic_obstacle_maximum_step_probability", 0.0
            )
        )
        self._previous_probabilistic_risk = (
            current_risk if np.isfinite(current_risk) else 1.0
        )
        self.previous_sequence = sequence.copy()
        self._delay_preceding_action = self.previous_action.copy()
        self.previous_action = action.copy()
        setter = getattr(self.sampling_prior, "set_previous", None)
        if callable(setter):
            setter(sequence)
        diagnostics["compute_ms"] = 1000.0 * (
            time.perf_counter() - started
        )
        diagnostics.update(
            {
                "planner_observation_x": float(
                    state[self.state_spec.position_indices[0]]
                ),
                "planner_observation_y": float(
                    state[self.state_spec.position_indices[1]]
                ),
                "planner_observation_theta": float(
                    state[self.state_spec.index("theta")]
                    if "theta" in self.state_spec.names
                    else observation.pose.theta
                ),
                "static_astar_replan_enabled": bool(
                    static_replan["enabled"]
                ),
                "static_astar_replan_triggered": bool(
                    static_replan["triggered"]
                ),
                "static_astar_replan_reason": str(
                    static_replan["reason"]
                ),
                "static_astar_replan_route_points": int(
                    static_replan["route_points"]
                ),
                "static_astar_replan_cross_track_m": float(
                    static_replan["cross_track_m"]
                ),
                "static_astar_replan_stagnation_steps": int(
                    static_replan["stagnation_steps"]
                ),
                "static_astar_replan_count": int(
                    static_replan["count"]
                ),
            }
        )
        if self.config.profile_components:
            diagnostics.update({
                "profile_planner_state_reference_ms": 1000.0 * (
                    state_reference_finished - started
                ),
                "profile_planner_prior_ms": 1000.0 * (
                    prior_finished - state_reference_finished
                ),
            })
        diagnostics["preview"] = False
        return PlanResult(
            proposed_control=ControlCommand(action, observation.timestamp, "mppi"),
            control_sequence=sequence,
            predicted_trajectory=updated_trajectory,
            diagnostics=diagnostics,
        )
