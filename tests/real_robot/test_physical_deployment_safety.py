from pathlib import Path
import math
from types import SimpleNamespace

import pytest

from deploy.raspberry_pi5_scout.build_pi5_full_config import (
    _physical_front_envelope,
    build_pi5_full_config,
)
from deploy.raspberry_pi5_scout.run_remote_cuda_full import (
    _DynamicPathGuardSupervisor,
    _deskew_livox_frame,
    _dynamic_hazard_sector,
    _direction_reversal_guard,
    _immediate_translation_stop_requested,
    _path_deviation_guard,
    _physical_goal_context,
    _physical_tracker_motion_context,
    _physical_command_slew_guard,
    _scout_fault_labels,
    _wait_for_complete_chassis_status,
)
from deploy.raspberry_pi5_scout.run_remote_pi_gateway import (
    _control_mode_interlock_reason,
)
from mobile_robot_mppi.core.spaces import action_spec_from_config
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter
from mobile_robot_mppi.real_robot import (
    LivoxPointCloudFrame,
    LivoxScanAdapter,
)
import numpy as np


def _weight_root(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in (
        "icode_stage2_task_aware.pt",
        "actor_full_proposed.pt",
        "hss_member1.pt",
        "hss_member2.pt",
        "hss_member3.pt",
    ):
        (weights / name).write_bytes(b"test")
    return tmp_path


def test_physical_limits_are_shared_with_planner_and_safety(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        goal_x=5.0,
        goal_y=2.0,
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    assert config["action_space"]["lower"] == [-0.3, -0.6]
    assert config["action_space"]["upper"] == [0.5, 0.6]
    assert config["rl"]["allow_controller_action_subset"] is True
    assert config["rl"]["allow_actor_action_subspace"] is True
    envelope = _physical_front_envelope(0.5)
    guard = config["perception"]["scan_guard"]
    assert guard["hard_stop_distance"] == pytest.approx(
        envelope["hard_stop_distance_m"]
    )
    assert guard["front_soft_block_distance"] == pytest.approx(
        envelope["soft_block_distance_m"]
    )
    assert guard["front_slow_distance"] == pytest.approx(
        envelope["slow_distance_m"]
    )
    assert guard["front_soft_block_max_speed"] == 0.0
    assert guard["physical_front_speed_governor_enabled"] is True
    local_layer = config["perception"]["local_obstacle_layer"]
    assert local_layer["local_obstacle_hard_filter_enabled"] is True
    assert local_layer[
        "local_obstacle_hard_filter_max_obstacles"
    ] == 16
    assert envelope["hard_stop_distance_m"] == pytest.approx(0.50)
    assert envelope["slow_distance_m"] == pytest.approx(0.75625)
    temporal = config["perception"]["temporal_scan_guard"]
    assert temporal["safety_enabled"] is True
    assert temporal["safety_hard_stop_ttc_s"] == pytest.approx(0.80)
    assert temporal["ego_motion_compensation_enabled"] is True
    assert temporal["safety_continuous_slowdown_enabled"] is True
    assert guard["dynamic_escape_trigger_ttc_s"] == pytest.approx(3.00)
    assert guard["dynamic_escape_uncertainty_fusion_enabled"] is True
    assert guard["dynamic_escape_direction_commit_steps"] == 4
    assert guard["dynamic_escape_frontal_commit_steps"] == 10
    assert guard["dynamic_escape_coast_steps"] == 6
    assert guard["dynamic_escape_frontal_entry_speed"] == pytest.approx(0.20)
    assert guard["dynamic_escape_persistent_front_retry_enabled"] is True
    assert guard["dynamic_escape_vetted_reverse_retry_steps"] == 4
    assert guard["dynamic_escape_persistent_front_max_retries"] == 1
    assert guard[
        "dynamic_escape_persistent_front_retry_commit_steps"
    ] == 4
    assert guard[
        "dynamic_escape_goal_divergence_release_rad"
    ] == pytest.approx(0.90)
    assert guard[
        "dynamic_escape_post_retry_reverse_hold_enabled"
    ] is True
    assert guard[
        "dynamic_escape_reverse_goal_realign_max_omega_radps"
    ] == pytest.approx(0.30)
    assert guard[
        "dynamic_escape_post_retry_side_forward_speed"
    ] == pytest.approx(0.20)
    assert guard[
        "dynamic_escape_post_retry_side_forward_min_bearing_rad"
    ] == pytest.approx(0.65)
    assert guard[
        "dynamic_escape_post_retry_side_forward_min_surface_range_m"
    ] == pytest.approx(0.70)
    assert guard[
        "dynamic_escape_direction_refresh_minimum_lateral_speed_mps"
    ] == pytest.approx(0.35)
    assert guard[
        "dynamic_escape_direction_refresh_confirmation_steps"
    ] == 2
    assert guard["dynamic_escape_coast_direction_lock_enabled"] is True
    assert guard["dynamic_escape_geometric_single_commit_enabled"] is True
    assert guard["dynamic_escape_geometric_rearm_clear_steps"] == 3
    assert guard["dynamic_escape_coast_turn_gain"] == pytest.approx(0.55)
    assert guard["dynamic_escape_coast_max_omega_radps"] == pytest.approx(0.30)
    assert guard["dynamic_recovery_enabled"] is False
    assert guard["dynamic_recovery_translation_enabled"] is False
    assert guard["dynamic_escape_reverse_speed"] == pytest.approx(0.30)
    assert config["planner"][
        "probabilistic_obstacle_emergency_candidate_trigger_distance_m"
    ] == pytest.approx(1.50)
    assert guard["dynamic_escape_hard_stop_enabled"] is True
    assert guard["dynamic_escape_hard_stop_turn_steps"] == 4
    assert guard["dynamic_escape_hard_stop_reverse_steps"] == 12
    assert guard["dynamic_escape_hard_stop_reverse_speed"] == pytest.approx(0.30)
    assert guard[
        "dynamic_escape_hard_stop_reverse_max_omega_radps"
    ] == pytest.approx(0.45)
    assert guard[
        "dynamic_escape_hard_stop_reverse_turn_decay_enabled"
    ] is True
    assert guard["dynamic_escape_hard_stop_min_rear_range"] == pytest.approx(0.80)
    assert guard[
        "dynamic_escape_hard_stop_direction_refresh_enabled"
    ] is False
    assert guard[
        "dynamic_escape_hard_stop_rear_clear_retry_enabled"
    ] is True
    assert guard[
        "dynamic_escape_hard_stop_rear_clear_retry_steps"
    ] == 6
    assert guard[
        "dynamic_escape_hard_stop_rear_blocked_wait_steps"
    ] == 6
    assert guard[
        "dynamic_escape_hard_stop_completed_hold_steps"
    ] == 2
    assert guard[
        "dynamic_escape_hard_stop_side_rear_release_enabled"
    ] is True
    assert guard[
        "dynamic_escape_hard_stop_side_rear_release_min_bearing_rad"
    ] == pytest.approx(1.75)
    assert guard[
        "dynamic_escape_hard_stop_side_rear_release_min_front_range_m"
    ] == pytest.approx(0.90)
    assert guard[
        "dynamic_escape_hard_stop_side_rear_release_speed"
    ] == pytest.approx(0.35)
    assert guard[
        "dynamic_escape_hard_stop_side_rear_release_hold_min_bearing_rad"
    ] == pytest.approx(math.pi / 2.0)
    assert guard[
        "dynamic_escape_hard_stop_side_rear_release_abort_steps"
    ] == 3
    assert guard["directional_motion_guard_enabled"] is True
    assert guard[
        "directional_forward_protected_half_angle_deg"
    ] == pytest.approx(100.0)
    assert guard["rear_pass_through_force_forward_enabled"] is True
    assert guard[
        "rear_pass_through_min_front_clearance_m"
    ] == pytest.approx(0.90)
    assert guard[
        "rear_pass_through_min_forward_speed_mps"
    ] == pytest.approx(0.35)
    planner = config["planner"]
    assert planner["known_static_map_cost_enabled"] is True
    assert planner["known_static_map_candidate_filter_enabled"] is True
    assert planner[
        "probabilistic_obstacle_emergency_candidate_trigger_ttc_s"
    ] == pytest.approx(3.00)
    assert planner[
        "probabilistic_obstacle_emergency_candidate_trigger_distance_m"
    ] == pytest.approx(1.50)
    assert planner[
        "probabilistic_obstacle_forward_lateral_countermotion_enabled"
    ] is True
    assert planner[
        "probabilistic_obstacle_forward_lateral_countermotion_weight"
    ] == pytest.approx(1.20)
    tracker = config["perception"]["dynamic_obstacle_tracker"]
    assert tracker["motion_confirmation_enabled"] is False


def test_physical_tracker_motion_fallback_drives_crossing_opposite_side(
        tmp_path):
    """Replay 232323 cycle 34 where MPPI exported no direction."""
    config = build_pi5_full_config(
        _weight_root(tmp_path), max_v_mps=0.5,
        max_reverse_v_mps=0.3, max_omega_radps=0.6,
    )
    context = _physical_tracker_motion_context(
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_forward_lateral_countermotion_applied": (
                False
            ),
            "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
        },
        {
            "associated": True,
            "forecast_valid": True,
            "motion_confirmed": True,
            "selected_support_beams": 25,
            "nearest_track_index": 2,
            "mapless_dynamic_track_indices": (2,),
            "measurement_velocity_x_mps": 0.09,
            "measurement_velocity_y_mps": 0.52,
        },
        0.0,
        config["planner"],
    )

    assert context["physical_tracker_motion_fallback_applied"] is True
    assert context[
        "probabilistic_obstacle_motion_lateral_body_mps"
    ] > 0.0
    assert context[
        "probabilistic_obstacle_preferred_escape_heading_error_rad"
    ] < 0.0

    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    decision = arbiter.arbitrate(
        ControlCommand([0.35, 0.60]),
        {
            "emergency_stop": False,
            "should_slow_down": True,
            "reason": "temporal_slowdown",
            "dynamic_obstacle_scan_flow_match": True,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 2.10,
            "dynamic_obstacle_bearing_rad": -0.48,
            "min_left_side_range": 1.6,
            "min_right_side_range": 0.8,
            "min_front_range": 1.48,
        },
        context,
    )
    assert decision.executed_control.omega == pytest.approx(-0.60)
    assert decision.diagnostics[
        "dynamic_escape_geometric_turn_source"
    ] == "predicted_relative_motion"


def test_physical_goal_context_uses_wrapped_body_frame_bearing():
    context = _physical_goal_context(
        {"kept": True},
        pose_x=0.0,
        pose_y=0.0,
        pose_yaw=math.pi - 0.05,
        goal_x=-1.0,
        goal_y=0.05,
    )

    assert context["kept"] is True
    assert context["physical_goal_distance_m"] == pytest.approx(
        math.hypot(1.0, 0.05)
    )
    assert abs(context["physical_goal_bearing_error_rad"]) < 0.06


def test_physical_tracker_motion_fallback_rejects_non_dynamic_track(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path), max_v_mps=0.5,
        max_reverse_v_mps=0.3, max_omega_radps=0.6,
    )
    context = _physical_tracker_motion_context(
        {},
        {
            "associated": True,
            "forecast_valid": True,
            "motion_confirmed": True,
            "selected_support_beams": 25,
            "nearest_track_index": 0,
            "mapless_dynamic_track_indices": (2,),
            "measurement_velocity_x_mps": 0.0,
            "measurement_velocity_y_mps": 0.8,
        },
        0.0,
        config["planner"],
    )
    assert context["physical_tracker_motion_fallback_applied"] is False


def test_physical_strong_lateral_motion_refreshes_stale_planner_side(
        tmp_path):
    """Replay the stale two-frame direction in 091643 cycle 11."""
    config = build_pi5_full_config(
        _weight_root(tmp_path), max_v_mps=0.5,
        max_reverse_v_mps=0.3, max_omega_radps=0.6,
    )
    context = _physical_tracker_motion_context(
        {
            "probabilistic_obstacle_forward_lateral_countermotion_applied": (
                True
            ),
            "probabilistic_obstacle_motion_lateral_body_mps": 0.56,
            "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.54,
            "probabilistic_obstacle_escape_direction_refreshed": False,
        },
        {},
        0.0,
        config["planner"],
    )
    assert context["physical_tracker_motion_fallback_applied"] is True
    assert context["physical_tracker_motion_refresh_requested"] is True
    assert context[
        "probabilistic_obstacle_escape_direction_refreshed"
    ] is True
    assert context[
        "probabilistic_obstacle_preferred_escape_heading_error_rad"
    ] < 0.0


def test_physical_dynamic_coast_is_finite_then_returns_planner(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path), max_v_mps=0.5,
        max_reverse_v_mps=0.3, max_omega_radps=0.6,
    )
    guard_config = config["perception"]["scan_guard"]
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]), guard_config
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.5,
        "dynamic_obstacle_bearing_rad": -0.45,
        "min_left_side_range": 1.6,
        "min_right_side_range": 0.8,
        "min_front_range": 1.45,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.55,
    }

    for _ in range(guard_config["dynamic_escape_direction_commit_steps"]):
        decision = arbiter.arbitrate(
            ControlCommand([0.35, 0.10]), guard, context
        )
        assert decision.executed_control.omega == pytest.approx(-0.60)
    for expected in reversed(range(guard_config["dynamic_escape_coast_steps"])):
        decision = arbiter.arbitrate(
            ControlCommand([0.35, 0.10]), guard, context
        )
        assert decision.executed_control.omega < 0.0
        assert decision.diagnostics[
            "dynamic_escape_coast_remaining"
        ] == expected

    released = arbiter.arbitrate(
        ControlCommand([0.35, 0.10]), guard, context
    )
    assert released.executed_control.omega == pytest.approx(0.10)
    assert released.diagnostics["dynamic_escape_geometric_forward_coast"] is False
    assert released.diagnostics["dynamic_escape_coast_remaining"] == 0


def test_physical_limit_validation_matches_pi_gateway(tmp_path):
    root = _weight_root(tmp_path)
    with pytest.raises(ValueError, match="reverse"):
        build_pi5_full_config(
            root,
            max_v_mps=0.5,
            max_reverse_v_mps=0.31,
            max_omega_radps=0.6,
        )
    with pytest.raises(ValueError, match="angular"):
        build_pi5_full_config(
            root,
            max_v_mps=0.5,
            max_reverse_v_mps=0.3,
            max_omega_radps=0.61,
        )


def test_path_guard_never_increases_speed_and_stops_large_deviation():
    clear_v, clear = _path_deviation_guard(
        0.0, 0.0, 0.0, 5.0, 2.0, 0.30
    )
    assert clear_v == pytest.approx(0.30)
    assert clear["active"] is False

    limited_v, limited = _path_deviation_guard(
        2.8, 3.2, 0.9, 5.0, 2.0, 0.50
    )
    assert limited_v == 0.0
    assert limited["active"] is True
    assert limited["cross_track_error_m"] > 1.2

    reverse_v, reverse = _path_deviation_guard(
        2.8, 3.2, 0.9, 5.0, 2.0, -0.10
    )
    assert reverse_v == -0.10
    assert reverse["active"] is True


def test_dynamic_manoeuvre_authority_then_goal_directed_rejoin():
    supervisor = _DynamicPathGuardSupervisor()
    arguments = (1.83, 1.14, 1.25, 5.0, 2.0, 0.50)

    active_v, active = supervisor.apply(
        *arguments,
        safety_reason="dynamic_active_escape",
        emergency_stop=False,
    )
    assert active_v == pytest.approx(0.50)
    assert active["active"] is False
    assert active["bypassed"] is True
    assert active["would_be_reason"] == "heading+heading_stop"

    rejoin_v, rejoin = supervisor.apply(
        *arguments,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=0.60,
        maximum_omega_radps=0.60,
        selected_probability=0.01,
    )
    assert rejoin_v == pytest.approx(0.20)
    assert rejoin["reason"] == "goal_heading_rejoin"
    assert rejoin["commanded_omega_override_radps"] == pytest.approx(-0.60)

    aligned_v, aligned = supervisor.apply(
        0.0, 0.0, 0.0, 5.0, 2.0, 0.50,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=0.10,
        selected_probability=0.0,
    )
    assert aligned_v == pytest.approx(0.50)
    assert aligned["commanded_omega_override_radps"] == pytest.approx(
        0.570760, abs=1e-5
    )
    assert aligned["goal_rejoin_latched"] is True

    supervisor.reset()
    clear_v, clear = supervisor.apply(
        0.0, 0.0, 0.0, 5.0, 2.0, 0.50,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=0.10,
        selected_probability=0.0,
    )
    assert clear_v == pytest.approx(0.50)
    assert clear["commanded_omega_override_radps"] is None


def test_path_supervisor_is_transparent_before_any_dynamic_event():
    supervisor = _DynamicPathGuardSupervisor()
    proposed_v = 0.50
    proposed_omega = 0.41
    output_v, diagnostics = supervisor.apply(
        2.8, 3.2, 0.9, 5.0, 2.0, proposed_v,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=proposed_omega,
        selected_probability=0.0,
    )
    assert output_v == pytest.approx(proposed_v)
    assert diagnostics["reason"] == "planner_authority"
    assert diagnostics["bypassed"] is True
    assert diagnostics["would_be_active"] is True
    assert diagnostics["goal_rejoin_latched"] is False
    assert diagnostics["commanded_omega_override_radps"] is None


def test_goal_rejoin_releases_back_to_planner_after_stable_clear_alignment():
    supervisor = _DynamicPathGuardSupervisor()
    goal_yaw = math.atan2(2.0, 5.0)
    supervisor.apply(
        0.0, 0.0, goal_yaw, 5.0, 2.0, 0.5,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.6,
    )

    results = []
    for _ in range(3):
        results.append(supervisor.apply(
            0.0, 0.0, goal_yaw, 5.0, 2.0, 0.5,
            safety_reason="front_clear",
            proposed_omega=0.12,
            selected_probability=0.0,
            selected_probability_mass=0.0,
            hazard_active=False,
        ))

    released_v, released = results[-1]
    assert released_v == pytest.approx(0.5)
    assert released["reason"] == "planner_authority"
    assert released["goal_rejoin_released_to_planner"] is True
    assert released["goal_rejoin_latched"] is False
    assert released["commanded_omega_override_radps"] is None


def test_20260804_large_circle_continuation_is_rejected_and_turned_home():
    supervisor = _DynamicPathGuardSupervisor()
    # Cycle 57 was the final fresh escape decision.  It is still honoured.
    escape_v, escape = supervisor.apply(
        2.24, 0.97, 1.20, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        emergency_stop=False,
        proposed_omega=0.60,
        selected_probability=0.0013,
    )
    assert escape_v == pytest.approx(0.50)
    assert escape["reason"] == "dynamic_authority"

    # Cycle 58 was front-clear, but the removed ten-step grace replayed the
    # same left turn at full speed and initiated the observed large circle.
    rejoin_v, rejoin = supervisor.apply(
        2.29, 1.08, 1.40, 5.0, 2.0, 0.50,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=0.60,
        maximum_omega_radps=0.60,
        selected_probability=0.0016,
    )
    assert rejoin_v == pytest.approx(0.20)
    assert rejoin["reason"] == "goal_heading_rejoin"
    assert rejoin["commanded_omega_override_radps"] == pytest.approx(-0.60)


def test_dynamic_rejoin_never_synthesizes_motion_above_fresh_risk_ceiling():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        emergency_stop=False,
    )
    guarded_v, guarded = supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.50,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=0.60,
        selected_probability=0.20,
    )
    assert guarded_v == 0.0
    assert guarded["commanded_omega_override_radps"] is None


def test_goal_rejoin_latch_converges_under_repeated_turn_away_proposals():
    supervisor = _DynamicPathGuardSupervisor()
    x, y, yaw = 2.27, 1.04, 0.99
    supervisor.apply(
        x, y, yaw, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        emergency_stop=False,
        proposed_omega=0.60,
    )
    initial_error = None
    final_error = None
    released = False
    for _ in range(8):
        v, diagnostics = supervisor.apply(
            x, y, yaw, 5.0, 2.0, 0.50,
            safety_reason="front_clear",
            emergency_stop=False,
            proposed_omega=0.60,
            selected_probability=0.0,
        )
        omega = diagnostics["commanded_omega_override_radps"]
        if initial_error is None:
            initial_error = abs(diagnostics["heading_error_rad"])
        final_error = abs(diagnostics["heading_error_rad"])
        if omega is None:
            assert diagnostics["goal_rejoin_released_to_planner"] is True
            released = True
            break
        dt = 0.30  # Exercise the measured forecast-cycle command hold.
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = math.atan2(
            math.sin(yaw + omega * dt), math.cos(yaw + omega * dt)
        )
    assert released is True
    assert final_error <= 0.25
    assert final_error < 0.50 * initial_error
    assert x > 2.80


def test_live_crossing_uses_smooth_rejoin_until_forecast_clears():
    supervisor = _DynamicPathGuardSupervisor()
    escape_v, escape = supervisor.apply(
        2.02, 0.84, 0.90, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.51,
        selected_probability=0.006,
        selected_probability_mass=0.053,
        hazard_active=True,
    )
    assert escape_v == pytest.approx(0.50)
    assert escape["commanded_omega_override_radps"] is None

    # Snapshot 023638 cycle 50 used to reverse directly from +0.51 to -0.60.
    # Keep moving, but soften only the opposite rejoin turn while the same
    # forecast remains live.
    passage_v, passage = supervisor.apply(
        2.11, 0.96, 1.07, 5.0, 2.0, 0.50,
        safety_reason="front_clear",
        proposed_omega=0.44,
        selected_probability=0.002,
        selected_probability_mass=0.016,
        hazard_active=True,
    )
    assert passage_v == pytest.approx(0.50)
    assert passage["reason"] == "dynamic_passage_commit"
    assert passage["commanded_omega_override_radps"] == pytest.approx(-0.30)

    # Brief forecast/scan fragmentation cannot reverse the selected side.
    # Goalward rejoin resumes only after the six-cycle evidence hold clears.
    clear_v = None
    clear = None
    for _ in range(6):
        clear_v, clear = supervisor.apply(
            2.22, 1.19, 1.23, 5.0, 2.0, 0.50,
            safety_reason="front_clear",
            proposed_omega=0.60,
            selected_probability=0.0,
            selected_probability_mass=0.0,
            hazard_active=False,
        )
    assert clear_v == pytest.approx(0.20)
    assert clear["reason"] == "goal_heading_rejoin"
    assert clear["commanded_omega_override_radps"] == pytest.approx(-0.60)


def test_rear_only_forecast_releases_weak_passage_rejoin_limit():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        1.78, -0.15, -0.68, 5.0, 0.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=-0.60,
        hazard_active=True,
    )
    # Arm the short scan-gap bridge used by a real rear pass.  Rear-sector
    # evidence on the following front-clear frame must cancel this bridge,
    # otherwise two stale planner-yaw frames precede goal rejoin.
    supervisor.apply(
        1.90, -0.28, -0.75, 5.0, 0.0, 0.50,
        safety_reason="rear_pass_through",
        proposed_omega=0.30,
        hazard_active=False,
        rear_only_hazard=True,
    )

    front_hazard, rear_only_hazard = _dynamic_hazard_sector(
        True,
        {
            "dynamic_obstacle_bearing_rad": 2.13,
            "temporal_scan_valid": False,
        },
    )
    assert front_hazard is False
    assert rear_only_hazard is True
    output_v, rejoin = supervisor.apply(
        2.01, -0.40, -0.82, 5.0, 0.0, 0.50,
        safety_reason="front_clear",
        proposed_omega=0.49,
        maximum_omega_radps=0.60,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=front_hazard,
        rear_only_hazard=rear_only_hazard,
    )

    assert output_v == pytest.approx(0.20)
    assert rejoin["reason"] == "goal_heading_rejoin"
    assert rejoin["hazard_active"] is False
    assert rejoin["rear_only_hazard"] is True
    assert rejoin["rear_pass_through_hold_active"] is False
    assert rejoin["commanded_omega_override_radps"] == pytest.approx(0.60)


def test_front_temporal_evidence_keeps_hazard_front_protected():
    front_hazard, rear_only_hazard = _dynamic_hazard_sector(
        True,
        {
            "dynamic_obstacle_bearing_rad": 2.13,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.2,
            "temporal_scan_center_angle_rad": 0.4,
        },
    )
    assert front_hazard is True
    assert rear_only_hazard is False


def test_post_escape_live_hazard_preserves_bounded_reverse_and_steering():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        3.10, 0.78, 0.04, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=-0.60,
        selected_probability=0.22,
        selected_probability_mass=0.952,
        hazard_active=True,
    )

    for _ in range(4):
        reverse_v, reverse = supervisor.apply(
            3.61, 0.76, 0.23, 5.0, 2.0, -0.30,
            safety_reason="front_clear",
            proposed_omega=0.60,
            selected_probability=0.001,
            selected_probability_mass=0.002,
            hazard_active=True,
        )
        assert reverse_v == pytest.approx(-0.30)
        assert reverse["reason"] == "dynamic_hazard_reverse"
        assert reverse["commanded_omega_override_radps"] is None

    # One sparse frame cannot turn reverse into forward.  Only six consecutive
    # clear observations expire the bounded hazard hold and start goal rejoin.
    for _ in range(5):
        held_v, held = supervisor.apply(
            3.61, 0.76, 0.50, 5.0, 2.0, -0.24,
            safety_reason="front_clear",
            proposed_omega=0.40,
            selected_probability=0.0,
            selected_probability_mass=0.0,
            hazard_active=False,
        )
        assert held_v == pytest.approx(-0.24)
        assert held["reason"] == "dynamic_hazard_reverse"
    rejoin_v, rejoin = supervisor.apply(
        3.61, 0.76, 0.50, 5.0, 2.0, -0.24,
        safety_reason="front_clear",
        proposed_omega=0.40,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=False,
    )
    assert rejoin_v > 0.0
    assert rejoin["reason"] == "stale_reverse_goal_rejoin"
    assert rejoin["commanded_omega_override_radps"] > 0.0


def test_20260804_goal_behind_safe_reverse_is_not_flipped_forward():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        3.80, 1.15, -1.80, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=-0.60,
        hazard_active=False,
    )

    # 20260804_014957 cycle 224: the path target was 152 degrees behind and
    # MPPI/ScanGuard supplied a valid reverse.  The old rejoin branch replaced
    # it with +0.20 m/s, which necessarily drove away from the target.
    output_v, diagnostics = supervisor.apply(
        3.848370715133504,
        1.1915533709355022,
        -1.8930460000089593,
        5.0,
        2.0,
        -0.30,
        safety_reason="front_clear",
        proposed_omega=-0.60,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=False,
    )

    assert output_v == pytest.approx(-0.20)
    assert diagnostics["reason"] == "goal_behind_reverse_rejoin"
    assert diagnostics["commanded_omega_override_radps"] is None
    assert diagnostics["goal_progress_projection_mps"] > 0.0


def test_20260804_goal_behind_forward_proposal_turns_without_departing():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        2.30, 1.60, 2.00, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.60,
        hazard_active=False,
    )

    # 20260804_020913 cycles 62--71: a positive command with a rear-hemisphere
    # target produced sustained negative goal progress.  With no rear-clearance
    # evidence, the supervisor may turn but must not invent reverse motion.
    output_v, diagnostics = supervisor.apply(
        2.2134581305902548,
        1.6963273740443063,
        2.166026000001257,
        5.0,
        2.0,
        0.50,
        safety_reason="front_clear",
        proposed_omega=0.60,
        maximum_omega_radps=0.60,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=False,
    )

    assert output_v == 0.0
    assert diagnostics["reason"] == "goal_behind_turn_rejoin"
    assert diagnostics["commanded_omega_override_radps"] == pytest.approx(
        -0.60
    )
    assert diagnostics["goal_progress_projection_mps"] == 0.0

    # A pose/measurement discontinuity across +/-pi cannot reverse the chosen
    # turn side on the following cycle.
    wrapped_v, wrapped = supervisor.apply(
        1.9692636906208443,
        1.9675749448856652,
        2.592431999999411,
        5.0,
        2.0,
        0.44,
        safety_reason="front_clear",
        proposed_omega=0.40,
        maximum_omega_radps=0.60,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=False,
    )
    assert wrapped_v == 0.0
    assert wrapped["heading_error_rad"] > 3.0
    assert wrapped["commanded_omega_override_radps"] == pytest.approx(-0.60)


def test_goal_behind_forward_escape_remains_authoritative_while_hazard_live():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        2.30, 1.60, 2.00, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.60,
        hazard_active=True,
    )
    output_v, diagnostics = supervisor.apply(
        2.21, 1.70, 2.17, 5.0, 2.0, 0.50,
        safety_reason="front_clear",
        proposed_omega=0.60,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=True,
    )

    assert output_v == pytest.approx(0.50)
    assert diagnostics["reason"] == "dynamic_passage_commit"
    assert diagnostics["hazard_active"] is True


def test_goal_behind_turn_rejoin_converges_without_a_full_rotation():
    supervisor = _DynamicPathGuardSupervisor()
    x, y, yaw = 2.21, 1.70, 2.17
    supervisor.apply(
        x, y, yaw, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.60,
        hazard_active=False,
    )

    turn_steps = 0
    while turn_steps < 30:
        output_v, diagnostics = supervisor.apply(
            x, y, yaw, 5.0, 2.0, 0.50,
            safety_reason="front_clear",
            proposed_omega=0.60,
            maximum_omega_radps=0.60,
            selected_probability=0.0,
            selected_probability_mass=0.0,
            hazard_active=False,
        )
        omega = diagnostics["commanded_omega_override_radps"]
        if diagnostics["reason"] != "goal_behind_turn_rejoin":
            assert output_v > 0.0
            break
        assert output_v == 0.0
        assert omega == pytest.approx(-0.60)
        yaw = math.atan2(
            math.sin(yaw + float(omega) * 0.10),
            math.cos(yaw + float(omega) * 0.10),
        )
        turn_steps += 1
    else:
        pytest.fail("goal-behind turn did not return to forward rejoin")

    assert 1 <= turn_steps < 20


def test_explicit_safety_reverse_remains_authoritative():
    supervisor = _DynamicPathGuardSupervisor()
    reverse_v, reverse = supervisor.apply(
        3.0, 1.0, 0.5, 5.0, 2.0, -0.30,
        safety_reason="dynamic_corridor_escape",
        proposed_omega=0.60,
        selected_probability=0.20,
        selected_probability_mass=2.0,
        hazard_active=True,
    )
    assert reverse_v == pytest.approx(-0.30)
    assert reverse["reason"] == "dynamic_authority"
    assert reverse["commanded_omega_override_radps"] is None


def test_front_dynamic_escape_preserves_safety_arbiter_reverse_and_steering():
    supervisor = _DynamicPathGuardSupervisor()
    forward_v, forward = supervisor.apply(
        3.0, 1.0, 0.5, 5.0, 2.0, -0.30,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.60,
        selected_probability=0.30,
        selected_probability_mass=2.0,
        hazard_active=True,
    )
    assert forward_v == pytest.approx(-0.30)
    assert forward["reason"] == "dynamic_authority"
    assert forward["commanded_omega_override_radps"] is None


def test_rear_only_dynamic_authority_turns_toward_goal():
    supervisor = _DynamicPathGuardSupervisor()
    output_v, decision = supervisor.apply(
        1.65, -0.31, 1.27, 5.0, 0.0, 0.50,
        safety_reason="rear_pass_through",
        proposed_omega=0.30,
        maximum_omega_radps=0.60,
        hazard_active=False,
        rear_only_hazard=True,
    )

    assert output_v == pytest.approx(0.50)
    assert decision["dynamic_authority"] is True
    assert decision["rear_only_goal_steer_active"] is True
    assert decision["heading_error_rad"] < 0.0
    assert decision["commanded_omega_override_radps"] == pytest.approx(-0.60)


def test_rear_only_goal_steer_does_not_flip_at_pi_wrap():
    supervisor = _DynamicPathGuardSupervisor()
    _, before_wrap = supervisor.apply(
        0.52, -0.12, -2.79, 5.0, 0.0, 0.50,
        safety_reason="rear_pass_through",
        proposed_omega=-0.30,
        maximum_omega_radps=0.60,
        rear_only_hazard=True,
    )
    _, after_wrap = supervisor.apply(
        0.53, -0.12, 3.10, 5.0, 0.0, 0.50,
        safety_reason="rear_pass_through",
        proposed_omega=-0.30,
        maximum_omega_radps=0.60,
        rear_only_hazard=True,
    )

    assert before_wrap["heading_error_rad"] > 0.0
    assert after_wrap["heading_error_rad"] < 0.0
    assert before_wrap["commanded_omega_override_radps"] == pytest.approx(0.60)
    assert after_wrap["commanded_omega_override_radps"] == pytest.approx(0.60)
    assert after_wrap["rear_only_goal_turn_sign"] == pytest.approx(1.0)


def test_front_dynamic_hard_stop_selects_side_then_reverses_only_if_rear_clear(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    action_spec = action_spec_from_config(config["action_space"])
    front_point = {
        "base_angle": 0.0,
        "range": 0.30,
        "x": 0.30,
        "y": 0.0,
    }
    clear_rear_point = {
        "base_angle": math.pi,
        "range": 2.0,
        "x": -2.0,
        "y": 0.0,
    }
    guard = {
        "emergency_stop": True,
        "reason": "hard_stop",
        "near_body_points": (front_point,),
        "raw_points_base": (front_point, clear_rear_point),
        "dynamic_obstacle_near_body_match": True,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_left_side_range": 1.4,
        "min_right_side_range": 1.4,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        # A person moving left requires the robot's right-side passage.
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.8,
    }
    arbiter = ScanGuardArbiter(
        action_spec, config["perception"]["scan_guard"]
    )

    turns = [
        arbiter.arbitrate(ControlCommand([0.5, 0.1]), guard, context)
        for _ in range(4)
    ]
    for decision in turns:
        assert decision.executed_control.v == 0.0
        assert decision.executed_control.omega == pytest.approx(-0.6)
        assert decision.reason == "dynamic_hard_stop_escape"
        assert decision.diagnostics[
            "dynamic_escape_hard_stop_phase"
        ] == "turn_in_place"

    reverse = arbiter.arbitrate(
        ControlCommand([0.5, 0.1]), guard, context
    )
    assert reverse.executed_control.v == pytest.approx(-0.3)
    assert reverse.executed_control.omega == pytest.approx(-0.45)
    assert reverse.diagnostics[
        "dynamic_escape_hard_stop_reverse_authorized"
    ] is True
    assert _immediate_translation_stop_requested(
        reverse.reason, reverse.diagnostics, reverse.executed_control.v
    ) is False
    assert _immediate_translation_stop_requested(
        turns[0].reason, turns[0].diagnostics, turns[0].executed_control.v
    ) is True

    blocked_guard = dict(guard)
    blocked_guard["raw_points_base"] = (
        front_point,
        {
            "base_angle": math.pi,
            "range": 0.45,
            "x": -0.45,
            "y": 0.0,
        },
    )
    blocked_arbiter = ScanGuardArbiter(
        action_spec, config["perception"]["scan_guard"]
    )
    for _ in range(4):
        blocked_arbiter.arbitrate(
            ControlCommand([0.5, 0.1]), blocked_guard, context
        )
    blocked = blocked_arbiter.arbitrate(
        ControlCommand([0.5, 0.1]), blocked_guard, context
    )
    assert blocked.executed_control.v == 0.0
    assert blocked.diagnostics[
        "dynamic_escape_hard_stop_reverse_authorized"
    ] is False
    assert blocked.diagnostics[
        "dynamic_escape_hard_stop_phase"
    ] == "rear_blocked_turn_only"
    assert blocked.diagnostics[
        "dynamic_escape_hard_stop_reverse_remaining"
    ] == 12
    assert blocked.diagnostics[
        "dynamic_escape_hard_stop_rear_blocked_wait_remaining"
    ] == 5

    head_on_guard = dict(guard)
    head_on_guard["min_left_side_range"] = 1.6
    head_on_guard["min_right_side_range"] = 0.6
    head_on = ScanGuardArbiter(
        action_spec, config["perception"]["scan_guard"]
    ).arbitrate(
        ControlCommand([0.5, 0.0]),
        head_on_guard,
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
        },
    )
    assert head_on.executed_control.v == 0.0
    assert head_on.executed_control.omega == pytest.approx(0.6)
    assert head_on.diagnostics[
        "dynamic_escape_geometric_turn_source"
    ] == "measured_side_clearance"


def test_dynamic_hard_stop_transaction_survives_sparse_clear_frames(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    front = {"base_angle": 0.0, "range": 0.44, "x": 0.44, "y": 0.0}
    rear = {"base_angle": math.pi, "range": 2.0, "x": -2.0, "y": 0.0}
    hard_guard = {
        "emergency_stop": True,
        "reason": "hard_stop",
        "raw_points_base": (front, rear),
        "dynamic_obstacle_scan_flow_match": True,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_left_side_range": 1.5,
        "min_right_side_range": 0.8,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.8,
    }
    first = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]), hard_guard, context
    )
    second = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]), hard_guard, context
    )
    sparse_guard = {
        "emergency_stop": False,
        "reason": "front_clear",
        "raw_points_base": (rear,),
        "dynamic_obstacle_bearing_rad": 0.0,
    }
    third = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]), sparse_guard, {}
    )
    fourth = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]), sparse_guard, {}
    )
    fifth = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]), sparse_guard, {}
    )

    assert [first.executed_control.v, second.executed_control.v,
            third.executed_control.v] == [0.0, 0.0, 0.0]
    assert third.diagnostics[
        "dynamic_escape_hard_stop_transaction_held"
    ] is True
    assert fourth.executed_control.v == 0.0
    assert fifth.executed_control.v == pytest.approx(-0.3)
    assert fifth.diagnostics[
        "dynamic_escape_hard_stop_reverse_authorized"
    ] is True


def test_close_crowd_track_switch_cannot_restart_bounded_turn(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    front = {"base_angle": 0.0, "range": 0.40, "x": 0.40, "y": 0.0}
    rear = {"base_angle": math.pi, "range": 2.0, "x": -2.0, "y": 0.0}
    guard = {
        "emergency_stop": True,
        "reason": "hard_stop",
        "raw_points_base": (front, rear),
        "dynamic_obstacle_scan_flow_match": True,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_left_side_range": 1.5,
        "min_right_side_range": 0.8,
    }
    initial = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.8,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.8,
    }
    first = arbiter.arbitrate(ControlCommand([-0.3, 0.6]), guard, initial)
    switched = dict(initial)
    switched.update({
        "probabilistic_obstacle_escape_direction_refreshed": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.8,
        "probabilistic_obstacle_motion_lateral_body_mps": -0.8,
    })
    second = arbiter.arbitrate(
        ControlCommand([-0.3, -0.6]), guard, switched
    )

    assert first.executed_control.omega == pytest.approx(0.6)
    assert second.executed_control.omega == pytest.approx(0.6)
    # A tracker identity/motion flip inside the same frontal close-crowd
    # encounter is not permission to change the already selected free side.
    assert second.diagnostics[
        "dynamic_escape_prediction_direction_refreshed"
    ] is False
    assert second.diagnostics[
        "dynamic_escape_prediction_direction_refresh_requested"
    ] is True
    assert second.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True
    assert second.diagnostics[
        "dynamic_escape_hard_stop_direction_refresh_applied"
    ] is False
    assert second.diagnostics[
        "dynamic_escape_hard_stop_turn_remaining"
    ] == 2


def test_blocked_crowd_stops_after_finite_turn_then_retries_open_rear_once(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    front = {"base_angle": 0.0, "range": 0.40, "x": 0.40, "y": 0.0}
    rear_blocked = {
        "base_angle": math.pi, "range": 0.45, "x": -0.45, "y": 0.0
    }
    base_guard = {
        "emergency_stop": True,
        "reason": "hard_stop",
        "dynamic_obstacle_scan_flow_match": True,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_left_side_range": 0.55,
        "min_right_side_range": 0.55,
    }
    blocked_guard = dict(base_guard)
    blocked_guard["raw_points_base"] = (front, rear_blocked)
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.8,
    }
    decisions = [
        arbiter.arbitrate(
            ControlCommand([-0.3, 0.6]), blocked_guard, context
        )
        for _ in range(16)
    ]
    held = [
        arbiter.arbitrate(
            ControlCommand([-0.3, 0.6]), blocked_guard, context
        )
        for _ in range(4)
    ]

    assert sum(
        item.diagnostics["dynamic_escape_hard_stop_phase"]
        == "turn_in_place"
        for item in decisions
    ) == 4
    assert all(item.executed_control.v == 0.0 for item in decisions)
    assert all(item.executed_control.v == 0.0 for item in held)
    assert all(item.executed_control.omega == 0.0 for item in held)
    assert all(
        item.diagnostics["dynamic_escape_hard_stop_phase"]
        == "completed_fail_closed"
        for item in held
    )
    assert sum(
        item.diagnostics["dynamic_escape_hard_stop_phase"]
        == "bounded_transaction_complete"
        for item in decisions
    ) == 2

    rear_clear = {
        "base_angle": math.pi, "range": 2.0, "x": -2.0, "y": 0.0
    }
    clear_guard = dict(base_guard)
    clear_guard["raw_points_base"] = (front, rear_clear)
    retry = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]), clear_guard, context
    )
    assert retry.executed_control.v == pytest.approx(-0.3)
    assert retry.diagnostics[
        "dynamic_escape_hard_stop_phase"
    ] == "rear_clear_retry_reverse"
    assert retry.diagnostics[
        "dynamic_escape_hard_stop_rear_clear_retry_started"
    ] is True


def test_confirmed_crossing_reversal_rejects_single_leg_swap(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.2,
        "dynamic_obstacle_bearing_rad": -0.45,
        "min_left_side_range": 1.6,
        "min_right_side_range": 0.8,
        "min_front_range": 1.4,
    }
    initial_context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": -0.60,
        "probabilistic_obstacle_motion_lateral_fraction": 0.90,
    }
    initial = arbiter.arbitrate(
        ControlCommand([0.35, 0.0]), guard, initial_context
    )
    assert initial.executed_control.omega == pytest.approx(0.60)

    single_swap = dict(initial_context)
    single_swap.update({
        "probabilistic_obstacle_escape_direction_refreshed": True,
        "probabilistic_obstacle_escape_direction_reversal_confirmation_count": 1,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.60,
    })
    rejected = arbiter.arbitrate(
        ControlCommand([0.35, -0.60]), guard, single_swap
    )
    assert rejected.executed_control.omega == pytest.approx(0.60)
    assert rejected.diagnostics[
        "dynamic_escape_prediction_direction_refresh_confirmed"
    ] is False
    assert rejected.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True

    confirmed = dict(single_swap)
    confirmed.update({
        "probabilistic_obstacle_escape_direction_reversal_confirmation_count": 2,
        # The estimator's second confirmation may be a weak instantaneous
        # sample; its accumulated confirmation is the causal evidence.
        "probabilistic_obstacle_motion_lateral_body_mps": 0.10,
    })
    corrected = arbiter.arbitrate(
        ControlCommand([0.35, -0.60]), guard, confirmed
    )
    assert corrected.executed_control.omega == pytest.approx(-0.60)
    assert corrected.diagnostics[
        "dynamic_escape_prediction_direction_refresh_confirmed"
    ] is True
    assert corrected.diagnostics[
        "dynamic_escape_prediction_direction_refreshed"
    ] is True


def test_completed_hard_stop_moves_forward_from_clear_side_rear_geometry(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    front = {"base_angle": 0.0, "range": 0.40, "x": 0.40, "y": 0.0}
    rear_blocked = {
        "base_angle": math.pi, "range": 0.45, "x": -0.45, "y": 0.0
    }
    blocked = {
        "emergency_stop": True,
        "reason": "near_body_hard_stop",
        "near_body_points": (front,),
        "raw_points_base": (front, rear_blocked),
        "dynamic_obstacle_near_body_match": True,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_front_range": 0.40,
        "min_left_side_range": 0.55,
        "min_right_side_range": 0.55,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.8,
    }
    decisions = [
        arbiter.arbitrate(ControlCommand([0.5, 0.6]), blocked, context)
        for _ in range(10)
    ]
    assert decisions[-1].diagnostics[
        "dynamic_escape_hard_stop_phase"
    ] == "rear_blocked_wait_exhausted"
    assert decisions[-1].diagnostics[
        "dynamic_escape_hard_stop_reverse_remaining"
    ] == 0

    side_point = {
        "base_angle": -1.60,
        "range": 0.30,
        "x": 0.30 * math.cos(-1.60),
        "y": 0.30 * math.sin(-1.60),
    }
    side_rear = dict(blocked)
    side_rear.update({
        "near_body_points": (side_point,),
        "raw_points_base": (side_point, rear_blocked),
        "dynamic_obstacle_bearing_rad": -2.0,
        "min_front_range": 2.0,
    })
    released = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]), side_rear, context
    )
    assert released.reason == "dynamic_hard_stop_side_rear_release"
    assert released.executed_control.values.tolist() == pytest.approx(
        [0.35, 0.0]
    )
    assert released.diagnostics["emergency_stop"] is False
    assert released.diagnostics[
        "dynamic_escape_hard_stop_side_rear_release_applied"
    ] is True
    assert _immediate_translation_stop_requested(
        released.reason,
        released.diagnostics,
        released.executed_control.v,
    ) is False
    path_v, path = _DynamicPathGuardSupervisor().apply(
        2.8,
        0.1,
        1.0,
        5.0,
        0.0,
        released.executed_control.v,
        released.reason,
        proposed_omega=released.executed_control.omega,
        maximum_omega_radps=0.6,
        hazard_active=True,
        rear_only_hazard=True,
    )
    assert path_v == pytest.approx(0.35)
    assert path["reason"] == "dynamic_authority"
    assert path["commanded_omega_override_radps"] is None

    noisy_side = dict(side_rear)
    noisy_side["dynamic_obstacle_bearing_rad"] = -1.40
    first_gap = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]), noisy_side, context
    )
    second_gap = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]), noisy_side, context
    )
    aborted = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]), noisy_side, context
    )
    assert first_gap.executed_control.v == pytest.approx(0.35)
    assert second_gap.executed_control.v == pytest.approx(0.35)
    assert second_gap.diagnostics[
        "dynamic_escape_hard_stop_side_rear_release_abort_count"
    ] == 2
    assert aborted.executed_control.v == 0.0
    assert aborted.diagnostics[
        "dynamic_escape_hard_stop_side_rear_release_latched"
    ] is False


def test_frontal_escape_rejects_noisy_side_reversal_and_locks_coast(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "slow_scale": 0.25,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.3,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_front_range": 1.45,
        "min_left_side_range": 1.6,
        "min_right_side_range": 0.8,
    }
    base_context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
        "probabilistic_obstacle_escape_direction_refreshed": False,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.0,
    }
    first = arbiter.arbitrate(
        ControlCommand([0.35, 0.0]), guard, base_context
    )
    assert first.executed_control.omega == pytest.approx(0.6)

    noisy_reversal = dict(base_context)
    noisy_reversal.update({
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_escape_direction_refreshed": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.13,
    })
    second = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]), guard, noisy_reversal
    )
    assert second.executed_control.omega == pytest.approx(0.6)
    assert second.diagnostics[
        "dynamic_escape_prediction_direction_refresh_requested"
    ] is True
    assert second.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True

    # Finish the finite full-yaw prefix, then verify that a noisy preferred
    # heading cannot bend the coast back toward the other side.
    for _ in range(
        config["perception"]["scan_guard"][
            "dynamic_escape_frontal_commit_steps"
        ] - 2
    ):
        arbiter.arbitrate(
            ControlCommand([0.35, -0.6]), guard, noisy_reversal
        )
    coast = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]), guard, noisy_reversal
    )
    assert coast.executed_control.omega > 0.0
    assert coast.diagnostics["dynamic_escape_coast_direction_locked"] is True

    # A one-frame TTC gate miss must not hand control back to a planner sample
    # that turns across the already selected frontal passage side.
    sparse_guard = dict(guard)
    sparse_guard.update({
        "should_slow_down": False,
        "reason": "front_clear",
        "temporal_scan_ttc_s": 3.1,
    })
    sparse = arbiter.arbitrate(
        ControlCommand([0.29, -0.55]), sparse_guard, noisy_reversal
    )
    assert sparse.executed_control.omega > 0.0
    assert sparse.diagnostics["dynamic_escape_geometric_forward_coast"] is True

    genuine_crossing_reversal = dict(noisy_reversal)
    genuine_crossing_reversal[
        "probabilistic_obstacle_motion_lateral_body_mps"
    ] = -0.55
    still_frontal = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]), guard, genuine_crossing_reversal
    )
    assert still_frontal.executed_control.omega > 0.0
    assert still_frontal.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True

    # The chassis turning about 26 degrees makes a stationary frontal person
    # appear oblique in body coordinates.  Keep the encounter latch until the
    # threat clears or moves behind; accepting this refresh caused the
    # 20260804_225524 cycle-57 wrong-side reversal.
    oblique_guard = dict(guard)
    oblique_guard["dynamic_obstacle_bearing_rad"] = 0.45
    refreshed = arbiter.arbitrate(
        ControlCommand([0.35, -0.6]),
        oblique_guard,
        genuine_crossing_reversal,
    )
    assert refreshed.executed_control.omega > 0.0
    assert refreshed.diagnostics[
        "dynamic_escape_prediction_direction_refreshed"
    ] is False
    assert refreshed.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True
    assert refreshed.diagnostics[
        "dynamic_escape_frontal_encounter_latched"
    ] is True


def test_late_crossing_prediction_replaces_uninformed_clearance_side(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "slow_scale": 0.25,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.5,
        "dynamic_obstacle_bearing_rad": -0.45,
        # This fallback selected left in both 20260804_213537 cycle 38 and
        # 20260804_213648 cycle 32 before crossing motion was available.
        "min_left_side_range": 1.60,
        "min_right_side_range": 0.80,
        "min_front_range": 1.45,
    }
    unavailable = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": None,
        "probabilistic_obstacle_escape_direction_refreshed": False,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": False,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.0,
    }
    initial = arbiter.arbitrate(
        ControlCommand([0.35, 0.0]), guard, unavailable
    )
    assert initial.executed_control.omega == pytest.approx(0.60)
    assert initial.diagnostics[
        "dynamic_escape_geometric_direction_prediction_backed"
    ] is False

    crossing = dict(unavailable)
    crossing.update({
        # The person is moving left in the vehicle frame, therefore the robot
        # must turn right even though this is the planner's first valid motion
        # estimate rather than a later `escape_direction_refreshed` event.
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.38,
    })
    corrected = arbiter.arbitrate(
        ControlCommand([0.35, 0.60]), guard, crossing
    )

    assert corrected.executed_control.v == pytest.approx(0.35)
    assert corrected.executed_control.omega == pytest.approx(-0.60)
    assert corrected.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_available"
    ] is True
    assert corrected.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_applied"
    ] is True
    assert corrected.diagnostics[
        "dynamic_escape_geometric_direction_prediction_backed"
    ] is True


def test_strong_crossing_inside_frontal_cone_uses_predicted_opposite_side(
    tmp_path,
):
    """The 235835 trace entered at 0.23 rad with real lateral motion.

    Bearing alone previously classified that frame as head-on, so wider-side
    clearance overruled the measured crossing direction and turned with the
    pedestrian.  Strong lateral evidence must keep it a crossing.
    """
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.2,
        "dynamic_obstacle_bearing_rad": 0.23,
        # Clearance alone asks for the wrong (+) side.
        "min_left_side_range": 1.60,
        "min_right_side_range": 0.70,
        "min_front_range": 1.10,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.60,
        "probabilistic_obstacle_motion_lateral_fraction": 0.90,
    }

    decision = arbiter.arbitrate(
        ControlCommand([-0.30, 0.60]), guard, context
    )

    assert decision.executed_control.v == pytest.approx(0.35)
    assert decision.executed_control.omega == pytest.approx(-0.60)
    assert decision.diagnostics[
        "dynamic_escape_geometric_turn_source"
    ] == "predicted_relative_motion"
    assert decision.diagnostics[
        "dynamic_escape_frontal_encounter_latched"
    ] is False


def test_persistent_front_retries_same_passage_side_after_bounded_reverse(
    tmp_path,
):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.2,
        "dynamic_obstacle_bearing_rad": 0.45,
        "min_left_side_range": 0.70,
        "min_right_side_range": 1.60,
        "min_front_range": 1.10,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.60,
        "probabilistic_obstacle_motion_lateral_fraction": 0.90,
    }

    first = arbiter.arbitrate(
        ControlCommand([-0.30, 0.60]), guard, context
    )
    assert first.executed_control.omega == pytest.approx(-0.60)
    for _ in range(3):
        arbiter.arbitrate(ControlCommand([-0.30, 0.60]), guard, context)
    for _ in range(6):
        arbiter.arbitrate(ControlCommand([-0.30, 0.60]), guard, context)

    reverse = None
    for _ in range(4):
        reverse = arbiter.arbitrate(
            ControlCommand([-0.30, 0.60]), guard, context
        )
        assert reverse.executed_control.v == pytest.approx(-0.30)
        assert reverse.diagnostics[
            "dynamic_escape_persistent_front_retry_rearmed"
        ] is False

    retry = arbiter.arbitrate(
        ControlCommand([-0.30, 0.60]), guard, context
    )
    assert retry.executed_control.v > 0.0
    assert retry.executed_control.omega == pytest.approx(-0.60)
    assert retry.diagnostics[
        "dynamic_escape_geometric_turn_source"
    ] == "persistent_front_previous_side"
    assert retry.diagnostics[
        "dynamic_escape_persistent_front_retry_rearmed"
    ] is True
    assert retry.diagnostics[
        "dynamic_escape_persistent_front_retry_count"
    ] == 1
    assert retry.diagnostics[
        "dynamic_escape_direction_commit_remaining"
    ] == 3

    for _ in range(3):
        arbiter.arbitrate(ControlCommand([-0.30, 0.60]), guard, context)
    for _ in range(6):
        arbiter.arbitrate(ControlCommand([-0.30, 0.60]), guard, context)
    for _ in range(4):
        bounded_reverse = arbiter.arbitrate(
            ControlCommand([-0.30, 0.60]), guard, context
        )
        assert bounded_reverse.executed_control.v == pytest.approx(-0.30)

    exhausted = arbiter.arbitrate(
        ControlCommand([-0.30, 0.60]), guard, context
    )
    assert exhausted.executed_control.values.tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert exhausted.diagnostics[
        "dynamic_escape_post_retry_reverse_hold_applied"
    ] is True


def test_goal_divergence_rejects_stale_retry_and_takes_lateral_exit(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.2,
        "dynamic_obstacle_bearing_rad": 0.75,
        "dynamic_obstacle_surface_range_m": 0.80,
        "min_left_side_range": 0.70,
        "min_right_side_range": 1.60,
        "min_front_range": 1.10,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.60,
        "probabilistic_obstacle_motion_lateral_fraction": 0.90,
        "physical_goal_bearing_error_rad": 1.0,
    }

    first = arbiter.arbitrate(ControlCommand([-0.30, 0.60]), guard, context)
    assert first.executed_control.omega == pytest.approx(-0.60)
    reverse = []
    for _ in range(4):
        reverse.append(
            arbiter.arbitrate(
                ControlCommand([-0.30, 0.60]), guard, context
            )
        )
    assert reverse[0].diagnostics[
        "dynamic_escape_geometric_goal_release_applied"
    ] is True
    assert all(item.executed_control.v < 0.0 for item in reverse)

    lateral_exit = arbiter.arbitrate(
        ControlCommand([-0.30, 0.60]), guard, context
    )
    assert lateral_exit.executed_control.values.tolist() == pytest.approx(
        [0.20, 0.0]
    )
    assert lateral_exit.diagnostics[
        "dynamic_escape_persistent_front_retry_goal_rejected"
    ] is True
    assert lateral_exit.diagnostics[
        "dynamic_escape_post_retry_side_forward_applied"
    ] is True


def test_prediction_backed_crossing_side_survives_temporal_emergency(tmp_path):
    """Replay the control conflict from 220825 cycle 40.

    A temporal TTC emergency is not a 0.50 m near-body event.  It must not hand
    an already forecast-backed right passage back to a left-turning planner
    sample.
    """
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.5,
        "dynamic_obstacle_bearing_rad": 0.48,
        "min_left_side_range": 1.5,
        "min_right_side_range": 1.5,
        "min_front_range": 1.4,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.95,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.48,
    }
    for _ in range(4):
        decision = arbiter.arbitrate(
            ControlCommand([0.5, 0.6]), guard, context
        )
        assert decision.executed_control.omega < 0.0

    emergency = dict(guard)
    emergency.update({
        "emergency_stop": True,
        "reason": "temporal_collision_risk",
    })
    decision = arbiter.arbitrate(
        ControlCommand([0.5, 0.6]), emergency, context
    )

    assert decision.executed_control.v > 0.0
    assert decision.executed_control.omega < 0.0
    assert decision.diagnostics[
        "dynamic_escape_prediction_backed_temporal_override"
    ] is True


def test_frontal_lateral_noise_cannot_late_flip_clearance_side(tmp_path):
    """Replay the false crossing acquisition from 221026 cycle 41."""
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.5,
        "dynamic_obstacle_bearing_rad": -0.23,
        "min_left_side_range": 1.6,
        "min_right_side_range": 0.7,
        "min_front_range": 1.2,
    }
    initial_context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": False,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.0,
    }
    initial = arbiter.arbitrate(
        ControlCommand([0.5, 0.0]), guard, initial_context
    )
    assert initial.executed_control.omega > 0.0

    noisy_leg = dict(initial_context)
    noisy_leg.update({
        "probabilistic_obstacle_escape_direction_refreshed": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.9,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.99,
    })
    decision = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]), guard, noisy_leg
    )

    assert decision.executed_control.omega > 0.0
    assert decision.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_available"
    ] is False
    assert decision.diagnostics[
        "dynamic_escape_prediction_direction_refresh_requested"
    ] is True
    assert decision.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True

    # Finish the finite side-selection prefix.  A subsequent temporal-only
    # emergency must keep this clearance-selected side instead of accepting an
    # opposite raw planner yaw.  The 0.50 m near-body event still preempts it.
    for _ in range(
        config["perception"]["scan_guard"][
            "dynamic_escape_frontal_commit_steps"
        ] - 2
    ):
        decision = arbiter.arbitrate(
            ControlCommand([0.5, -0.6]), guard, noisy_leg
        )
        assert decision.executed_control.omega > 0.0
    emergency = dict(guard)
    emergency.update({
        "emergency_stop": True,
        "reason": "temporal_collision_risk",
    })
    decision = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]), emergency, noisy_leg
    )
    assert decision.executed_control.omega > 0.0
    assert decision.diagnostics[
        "dynamic_escape_geometric_temporal_override"
    ] is True


def test_frontal_encounter_stays_latched_after_robot_turns(tmp_path):
    """Replay 225524 cycles 54-57 without ego-turn reclassification."""
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.27,
        "dynamic_obstacle_bearing_rad": -0.166,
        "min_left_side_range": 0.70,
        "min_right_side_range": 1.55,
        "min_front_range": 1.42,
    }
    initial_context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": False,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.0,
    }
    initial = arbiter.arbitrate(
        ControlCommand([0.35, 0.53]), guard, initial_context
    )
    assert initial.executed_control.omega == pytest.approx(-0.60)
    assert initial.diagnostics[
        "dynamic_escape_frontal_encounter_latched"
    ] is True

    turned_guard = dict(guard)
    turned_guard["dynamic_obstacle_bearing_rad"] = -0.419
    noisy_relative_motion = dict(initial_context)
    noisy_relative_motion.update({
        "probabilistic_obstacle_escape_direction_refreshed": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.876,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": -0.515,
    })
    decision = arbiter.arbitrate(
        ControlCommand([-0.30, 0.0]), turned_guard, noisy_relative_motion
    )

    assert decision.executed_control.omega == pytest.approx(-0.60)
    assert decision.diagnostics[
        "dynamic_escape_prediction_direction_refresh_rejected"
    ] is True
    assert decision.diagnostics[
        "dynamic_escape_frontal_encounter_latched"
    ] is True

    for _ in range(
        config["perception"]["scan_guard"][
            "dynamic_escape_frontal_commit_steps"
        ] - 2
    ):
        decision = arbiter.arbitrate(
            ControlCommand([-0.30, 0.0]),
            turned_guard,
            noisy_relative_motion,
        )
        assert decision.executed_control.omega < 0.0
    coast = arbiter.arbitrate(
        ControlCommand([-0.30, 0.0]), turned_guard, initial_context
    )
    assert coast.executed_control.omega == pytest.approx(-0.30)
    assert coast.diagnostics[
        "dynamic_escape_geometric_turn_source"
    ] == "frontal_encounter_coast"

    hard_guard = dict(turned_guard)
    hard_guard.update({
        "emergency_stop": True,
        "reason": "hard_stop",
        "dynamic_obstacle_near_body_match": True,
        "dynamic_obstacle_bearing_rad": -0.585,
    })
    hard = arbiter.arbitrate(
        ControlCommand([-0.30, 0.60]), hard_guard, initial_context
    )
    assert hard.executed_control.v == 0.0
    assert hard.executed_control.omega == pytest.approx(-0.60)
    assert hard.diagnostics[
        "dynamic_escape_geometric_turn_source"
    ] == "frontal_encounter_side"


def test_late_crossing_prediction_does_not_flip_after_obstacle_passes_rear(
    tmp_path,
):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "slow_scale": 0.25,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.5,
        "dynamic_obstacle_bearing_rad": -0.45,
        "min_left_side_range": 1.60,
        "min_right_side_range": 0.80,
        "min_front_range": 1.45,
    }
    unavailable = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": None,
        "probabilistic_obstacle_escape_direction_refreshed": False,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": False,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.0,
    }
    initial = arbiter.arbitrate(
        ControlCommand([0.35, 0.0]), guard, unavailable
    )
    assert initial.executed_control.omega == pytest.approx(0.60)

    rear_crossing = dict(unavailable)
    rear_crossing.update({
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.45,
    })
    rear_guard = dict(guard)
    rear_guard["dynamic_obstacle_bearing_rad"] = math.radians(110.0)
    passed = arbiter.arbitrate(
        ControlCommand([0.35, 0.60]), rear_guard, rear_crossing
    )

    assert passed.executed_control.omega == pytest.approx(0.60)
    assert passed.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_forward_sector"
    ] is False
    assert passed.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_applied"
    ] is False


def test_late_crossing_prediction_does_not_flip_head_on_clearance_choice(
    tmp_path,
):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "slow_scale": 0.25,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 1.5,
        "dynamic_obstacle_bearing_rad": 0.05,
        "min_left_side_range": 1.60,
        "min_right_side_range": 0.80,
        "min_front_range": 1.45,
    }
    unavailable = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": None,
        "probabilistic_obstacle_escape_direction_refreshed": False,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": False,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.0,
    }
    initial = arbiter.arbitrate(
        ControlCommand([0.35, 0.0]), guard, unavailable
    )
    assert initial.executed_control.omega == pytest.approx(0.60)

    noisy_leg_motion = dict(unavailable)
    noisy_leg_motion.update({
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.45,
    })
    held = arbiter.arbitrate(
        ControlCommand([0.35, 0.60]), guard, noisy_leg_motion
    )

    assert held.executed_control.omega == pytest.approx(0.60)
    assert held.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_forward_sector"
    ] is True
    assert held.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_oblique_geometry"
    ] is False
    assert held.diagnostics[
        "dynamic_escape_prediction_direction_late_acquisition_applied"
    ] is False


def test_front_speed_governor_caps_active_escape_final_command(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": False,
        "should_slow_down": True,
        "slow_scale": 0.25,
        "reason": "temporal_slowdown",
        "dynamic_obstacle_scan_flow_match": True,
        "temporal_scan_valid": True,
        "temporal_scan_ttc_s": 0.9,
        "dynamic_obstacle_bearing_rad": 0.0,
        "min_front_range": 0.56,
        "min_left_side_range": 1.5,
        "min_right_side_range": 0.8,
    }
    decision = arbiter.arbitrate(
        ControlCommand([0.5, 0.6]),
        guard,
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.8,
            "probabilistic_obstacle_motion_lateral_body_mps": 0.5,
        },
    )
    assert 0.0 < decision.executed_control.v < 0.35
    assert decision.diagnostics[
        "dynamic_escape_frontal_entry_speed_applied"
    ] is True


def test_dynamic_path_authority_never_bypasses_near_body_hard_stop():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.50,
        safety_reason="dynamic_active_escape",
        emergency_stop=False,
    )
    stopped_v, stopped = supervisor.apply(
        0.0, 0.0, 0.0, 5.0, 2.0, 0.50,
        safety_reason="near_body_hard_stop",
        emergency_stop=True,
    )
    assert stopped_v == 0.0
    assert stopped["active"] is True
    assert stopped["bypassed"] is False


def test_temporal_slowdown_keeps_its_vetted_speed_through_path_guard():
    supervisor = _DynamicPathGuardSupervisor()
    slowed_v, slowed = supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.18,
        safety_reason="temporal_slowdown",
        emergency_stop=False,
        proposed_omega=0.3,
        hazard_active=True,
    )
    stopped_v, stopped = supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.18,
        safety_reason="temporal_collision_risk",
        emergency_stop=True,
        proposed_omega=0.3,
        hazard_active=True,
    )

    assert slowed_v == pytest.approx(0.18)
    assert slowed["dynamic_authority"] is True
    assert slowed["reason"] == "dynamic_authority"
    assert stopped_v == 0.0
    assert stopped["dynamic_authority"] is False


def test_live_hazard_slow_command_keeps_avoidance_yaw_through_path_guard():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        1.8,
        1.1,
        1.2,
        5.0,
        2.0,
        0.35,
        safety_reason="dynamic_active_escape",
        proposed_omega=0.6,
        hazard_active=True,
    )
    slowed_v, slowed = supervisor.apply(
        1.8,
        1.1,
        1.2,
        5.0,
        2.0,
        0.30,
        safety_reason="front_obstacle_slow",
        proposed_omega=0.6,
        hazard_active=True,
    )

    assert slowed_v == pytest.approx(0.30)
    assert slowed["reason"] == "dynamic_authority"
    assert slowed["dynamic_authority"] is True
    assert slowed["commanded_omega_override_radps"] is None


def test_rear_pass_through_bypasses_straight_path_stop_and_bridges_one_gap():
    supervisor = _DynamicPathGuardSupervisor()
    first_v, first = supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.35,
        safety_reason="rear_pass_through",
        emergency_stop=False,
        proposed_omega=0.3,
        hazard_active=True,
    )
    gap_v, gap = supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.35,
        safety_reason="front_clear",
        emergency_stop=False,
        proposed_omega=0.3,
        hazard_active=True,
    )
    stopped_v, stopped = supervisor.apply(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.35,
        safety_reason="near_body_hard_stop",
        emergency_stop=True,
        proposed_omega=0.3,
        hazard_active=True,
    )

    assert first_v == pytest.approx(0.35)
    assert first["dynamic_authority"] is True
    assert gap_v == pytest.approx(0.35)
    assert gap["dynamic_authority"] is True
    assert gap["rear_pass_through_hold_active"] is True
    assert stopped_v == 0.0
    assert stopped["dynamic_authority"] is False


def test_direction_reversal_requires_one_zero_cycle_and_preserves_steering():
    forward_v, omega, clear = _direction_reversal_guard(0.50, -0.60, 0.30)
    assert forward_v == pytest.approx(0.50)
    assert omega == pytest.approx(-0.60)
    assert clear["active"] is False

    transition_v, omega, transition = _direction_reversal_guard(
        -0.30, 0.60, forward_v
    )
    assert transition_v == 0.0
    assert omega == pytest.approx(0.60)
    assert transition["active"] is True

    reverse_v, omega, reverse = _direction_reversal_guard(
        -0.30, 0.60, transition_v
    )
    assert reverse_v == pytest.approx(-0.30)
    assert omega == pytest.approx(0.60)
    assert reverse["active"] is False


def test_physical_slew_guard_limits_all_nonemergency_command_steps():
    v, omega, diagnostics = _physical_command_slew_guard(
        -0.30,
        -0.60,
        0.50,
        0.60,
        dt_s=0.10,
        maximum_v_rate_mps2=0.60,
        maximum_omega_rate_radps2=2.0,
    )
    assert v == pytest.approx(0.44)
    assert omega == pytest.approx(0.40)
    assert diagnostics["active"] is True
    assert diagnostics["reason"] == "rate_limited"


def test_physical_slew_guard_brakes_before_hazard_reverse_then_ramps():
    stopped_v, stopped_omega, stopped = _physical_command_slew_guard(
        -0.30,
        -0.60,
        0.50,
        0.60,
        dt_s=0.10,
        maximum_v_rate_mps2=0.60,
        maximum_omega_rate_radps2=2.0,
        hazard_active=True,
    )
    assert stopped_v == 0.0
    assert stopped_omega == pytest.approx(0.40)
    assert stopped["reason"] == "hazard_reversal_brake"

    reverse_v, reverse_omega, ramped = _physical_command_slew_guard(
        -0.30,
        -0.60,
        stopped_v,
        stopped_omega,
        dt_s=0.10,
        maximum_v_rate_mps2=0.60,
        maximum_omega_rate_radps2=2.0,
        hazard_active=True,
    )
    assert reverse_v == pytest.approx(-0.06)
    assert reverse_omega == pytest.approx(0.20)
    assert ramped["reason"] == "rate_limited"


def test_physical_slew_guard_preserves_immediate_hard_stop_authority():
    v, omega, diagnostics = _physical_command_slew_guard(
        0.0,
        -0.60,
        0.50,
        0.60,
        dt_s=0.10,
        maximum_v_rate_mps2=0.60,
        maximum_omega_rate_radps2=2.0,
        immediate_translation_stop=True,
    )
    assert v == 0.0
    assert omega == pytest.approx(0.40)
    assert diagnostics["reason"] == "immediate_translation_stop"


def test_control_mode_interlock_rejects_takeover_and_lost_authority():
    assert _control_mode_interlock_reason(0, False, 0.2) is None
    assert _control_mode_interlock_reason(1, True, 5.0) is None
    assert _control_mode_interlock_reason(3, False, 0.1) == "remote_takeover"
    assert _control_mode_interlock_reason(3, True, 0.1) == "remote_takeover"
    assert _control_mode_interlock_reason(0, True, 0.1) == "can_mode_lost"
    assert _control_mode_interlock_reason(0, False, 1.01) == (
        "can_mode_not_confirmed"
    )
    assert _scout_fault_labels(4) == ("remote_controller_link_lost",)


def test_chassis_preflight_waits_past_initial_tcp_only_status():
    incomplete = SimpleNamespace(
        battery_v=None, control_mode=None, fault=None
    )
    complete = SimpleNamespace(
        battery_v=25.4, control_mode=0, fault=0
    )

    class _Remote:
        def __init__(self):
            self.calls = 0

        def wait_for_status(self, timeout_s):
            return incomplete

        def status(self):
            self.calls += 1
            return complete if self.calls >= 2 else incomplete

    result = _wait_for_complete_chassis_status(_Remote(), timeout_s=0.2)

    assert result is complete


def test_20260804_near_miss_snapshots_are_blocked_before_takeover():
    # Cycle 35 had already turned 1.31 rad away from the rejoin heading while
    # still moving quickly.  The path governor now removes forward authority.
    guarded_v, diagnostics = _path_deviation_guard(
        1.83, 1.14, 1.25, 5.0, 2.0, 0.50
    )
    assert guarded_v == 0.0
    assert "heading_stop" in diagnostics["reason"]

    # A 0.734 m return should reduce speed rather than causing a distant hard
    # stop.  The later 0.365 m near-miss return remains an unconditional stop.
    envelope = _physical_front_envelope(0.5)
    assert envelope["hard_stop_distance_m"] < 0.734
    assert envelope["hard_stop_distance_m"] > 0.365

    # The raw temporal scan already saw a closing consensus.  This fallback
    # does not depend on the mistaken 2.31 m forecast identity and would have
    # stopped three cycles before takeover.
    cycle_57_ttc_s = 0.6810890144050882
    assert cycle_57_ttc_s <= 0.80


def test_physical_front_governor_slows_continuously_before_close_hard_stop(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.2,
        max_omega_radps=0.4,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    decision = arbiter.arbitrate(
        ControlCommand([0.5, 0.2]),
        {
            "emergency_stop": False,
            "should_slow_down": True,
            "slow_scale": 1.0,
            "min_front_range": 0.654,
            "reason": "front_obstacle_slow",
        },
        {},
    )
    assert 0.0 < decision.executed_control.v < 0.5
    assert decision.executed_control.v == pytest.approx(0.404, abs=0.002)
    assert decision.executed_control.omega == pytest.approx(0.2)
    assert decision.diagnostics["physical_front_speed_governor_applied"] is True

    stopped = arbiter.arbitrate(
        ControlCommand([0.5, 0.2]),
        {
            "emergency_stop": True,
            "should_slow_down": False,
            "min_front_range": 0.365,
            "reason": "hard_stop",
        },
        {},
    )
    assert stopped.executed_control.v == 0.0


def test_directional_guard_passes_rear_person_only_for_forward_motion(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    rear_angle = math.radians(165.0)
    rear_point = {
        "base_angle": rear_angle,
        "range": 0.30,
        "x": 0.30 * math.cos(rear_angle),
        "y": 0.30 * math.sin(rear_angle),
    }
    guard = {
        "emergency_stop": True,
        "should_slow_down": False,
        "slow_scale": 0.0,
        "reason": "near_body_hard_stop",
        "near_body_points": (rear_point,),
        "min_front_range": 2.0,
        "dynamic_obstacle_bearing_rad": rear_angle,
        "dynamic_obstacle_near_body_match": True,
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_active_fallback_used": True,
        "probabilistic_obstacle_active_fallback_index": 4,
        "probabilistic_obstacle_maximum_step_probability": 0.1,
        "probabilistic_obstacle_stop_maximum_probability": 0.9,
    }

    forward = arbiter.arbitrate(ControlCommand([0.5, 0.15]), guard, context)
    assert forward.executed_control.v == pytest.approx(0.5)
    assert forward.executed_control.omega == pytest.approx(0.15)
    assert forward.reason == "rear_pass_through"
    assert forward.diagnostics["emergency_stop"] is False
    assert forward.diagnostics["rear_pass_through_active"] is True
    assert forward.diagnostics["dynamic_escape_allowed"] is False

    reverse = arbiter.arbitrate(ControlCommand([-0.3, 0.15]), guard, context)
    assert reverse.executed_control.v == pytest.approx(0.35)
    assert reverse.executed_control.omega == pytest.approx(0.15)
    assert reverse.reason == "rear_pass_through"
    assert reverse.diagnostics["rear_pass_through_active"] is True
    assert reverse.diagnostics[
        "rear_pass_through_force_forward_ready"
    ] is True
    assert reverse.diagnostics["rear_reverse_blocked"] is True


def test_directional_guard_keeps_front_person_fail_closed(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    front_point = {
        "base_angle": 0.0,
        "range": 0.30,
        "x": 0.30,
        "y": 0.0,
    }
    decision = arbiter.arbitrate(
        ControlCommand([0.5, -0.2]),
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "near_body_points": (front_point,),
            "dynamic_obstacle_bearing_rad": 0.0,
        },
        {},
    )
    assert decision.executed_control.v == 0.0
    assert decision.executed_control.omega == pytest.approx(-0.2)
    assert decision.reason == "near_body_hard_stop"
    assert decision.diagnostics["rear_pass_through_active"] is False


def test_rear_pass_force_forward_requires_fresh_front_clearance(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    bearing = math.radians(-145.0)
    rear_point = {
        "base_angle": bearing,
        "range": 0.30,
        "x": 0.30 * math.cos(bearing),
        "y": 0.30 * math.sin(bearing),
    }
    guard = {
        "emergency_stop": True,
        "reason": "near_body_hard_stop",
        "near_body_points": (rear_point,),
        "min_front_range": 0.72,
        "dynamic_obstacle_bearing_rad": bearing,
    }
    blocked = arbiter.arbitrate(
        ControlCommand([-0.3, -0.6]), guard, {}
    )

    assert blocked.executed_control.v == 0.0
    assert blocked.reason == "near_body_hard_stop"
    assert blocked.diagnostics[
        "rear_pass_through_force_forward_ready"
    ] is False


def test_rear_pass_holds_one_turn_side_through_bearing_flips_and_scan_gaps(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )

    def rear_guard(angle_deg):
        angle = math.radians(angle_deg)
        return {
            "emergency_stop": True,
            "reason": "temporal_collision_risk",
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 0.30,
            "temporal_scan_center_angle_rad": angle,
            "min_front_range": 2.0,
        }

    first = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]), rear_guard(-145.0), {}
    )
    opposite_fragment = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]), rear_guard(145.0), {}
    )
    for _ in range(2):
        gap = arbiter.arbitrate(
            ControlCommand([0.5, 0.0]),
            {"emergency_stop": False, "reason": "front_clear"},
            {},
        )
        assert gap.diagnostics[
            "rear_pass_through_direction_clear_streak"
        ] < 4
    after_gap = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]), rear_guard(145.0), {}
    )

    assert first.executed_control.values.tolist() == pytest.approx(
        [0.35, -0.3]
    )
    assert opposite_fragment.executed_control.values.tolist() == pytest.approx(
        [0.35, -0.3]
    )
    assert after_gap.executed_control.values.tolist() == pytest.approx(
        [0.35, -0.3]
    )
    assert first.diagnostics[
        "rear_pass_through_direction_lock_started"
    ] is True
    assert opposite_fragment.diagnostics[
        "rear_pass_through_candidate_turn_sign"
    ] == 1.0
    assert opposite_fragment.diagnostics[
        "rear_pass_through_turn_sign"
    ] == -1.0


def test_rear_pass_goes_straight_when_person_is_centered_behind(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    decision = arbiter.arbitrate(
        ControlCommand([-0.3, 0.6]),
        {
            "emergency_stop": True,
            "reason": "temporal_collision_risk",
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 0.30,
            "temporal_scan_center_angle_rad": math.pi,
            "min_front_range": 2.0,
        },
        {},
    )

    assert decision.reason == "rear_pass_through"
    assert decision.executed_control.values.tolist() == pytest.approx(
        [0.35, 0.0]
    )


def test_dynamic_side_hard_stop_does_not_advance_inside_protected_sector(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    bearing = math.radians(98.0)
    decision = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]),
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "dynamic_obstacle_bearing_rad": bearing,
            "dynamic_obstacle_near_body_match": True,
            "min_left_side_range": 0.31,
            "min_right_side_range": 2.0,
        },
        {"probabilistic_obstacle_active_avoidance_enabled": True},
    )

    assert decision.reason == "dynamic_hard_stop_escape"
    assert decision.executed_control.v == 0.0
    assert decision.diagnostics[
        "dynamic_escape_hard_stop_phase"
    ] == "turn_in_place"


def test_hard_stop_reverse_turn_decays_across_bounded_transaction(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    guard = {
        "emergency_stop": True,
        "reason": "near_body_hard_stop",
        "dynamic_obstacle_bearing_rad": 0.10,
        "dynamic_obstacle_near_body_match": True,
        "raw_points_base": (
            {"base_angle": math.pi, "range": 2.0},
        ),
    }
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_preferred_escape_heading_error_rad": -0.88,
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_motion_lateral_body_mps": 0.60,
        "probabilistic_obstacle_motion_lateral_fraction": 0.90,
        "physical_goal_bearing_error_rad": 0.0,
    }

    for _ in range(4):
        turn = arbiter.arbitrate(ControlCommand([0.5, 0.6]), guard, context)
        assert turn.executed_control.v == 0.0
    first_reverse = arbiter.arbitrate(
        ControlCommand([-0.3, -0.6]), guard, context
    )
    second_reverse = arbiter.arbitrate(
        ControlCommand([-0.3, -0.6]), guard, context
    )

    assert first_reverse.executed_control.v == pytest.approx(-0.30)
    assert abs(first_reverse.executed_control.omega) == pytest.approx(0.45)
    assert abs(second_reverse.executed_control.omega) == pytest.approx(0.4125)
    assert second_reverse.diagnostics[
        "dynamic_escape_hard_stop_reverse_turn_scale"
    ] == pytest.approx(11.0 / 12.0)


def test_path_supervisor_recovers_forward_after_exhausted_reverse():
    supervisor = _DynamicPathGuardSupervisor()
    supervisor.apply(
        0.0,
        0.0,
        math.pi,
        5.0,
        0.0,
        0.20,
        "dynamic_active_escape",
        hazard_active=True,
    )
    supervisor.apply(
        0.0,
        0.0,
        math.pi,
        5.0,
        0.0,
        0.0,
        "dynamic_active_escape",
        hazard_active=True,
        reverse_escape_exhausted=True,
    )

    held_v, held = supervisor.apply(
        0.0,
        0.0,
        math.pi,
        5.0,
        0.0,
        -0.30,
        "front_clear",
        hazard_active=True,
    )
    assert held_v == 0.0
    assert held["reason"] == "dynamic_hazard_reverse_exhausted"

    forward_v, forward = supervisor.apply(
        0.0,
        0.0,
        1.0,
        5.0,
        0.0,
        -0.30,
        "front_clear",
        hazard_active=True,
    )
    assert forward_v > 0.0
    assert forward["reason"] == "dynamic_hazard_goal_rejoin"
    assert forward["commanded_omega_override_radps"] < 0.0


def test_dynamic_zero_translation_turn_budget_includes_prior_temporal_turns(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    temporal = {
        "emergency_stop": True,
        "reason": "temporal_collision_risk",
        "dynamic_obstacle_scan_flow_match": True,
    }
    hard = {
        "emergency_stop": True,
        "reason": "near_body_hard_stop",
        "dynamic_obstacle_bearing_rad": 0.0,
        "dynamic_obstacle_near_body_match": True,
    }
    decisions = [
        arbiter.arbitrate(ControlCommand([0.5, 0.6]), temporal, {})
        for _ in range(2)
    ]
    decisions.extend(
        arbiter.arbitrate(
            ControlCommand([0.5, 0.6]),
            hard,
            {"probabilistic_obstacle_active_avoidance_enabled": True},
        )
        for _ in range(9)
    )

    rotating = [
        decision
        for decision in decisions
        if abs(decision.executed_control.omega) >= 0.10
    ]
    assert len(rotating) == 9
    assert decisions[-1].executed_control.values.tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert decisions[-1].diagnostics[
        "dynamic_escape_zero_translation_turn_suppressed"
    ] is True


def test_recovery_releases_when_zero_translation_turn_budget_is_exhausted(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    guard_config = dict(config["perception"]["scan_guard"])
    guard_config.update({
        "dynamic_recovery_enabled": True,
        "dynamic_recovery_translation_enabled": True,
    })
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]), guard_config
    )
    arbiter._dynamic_escape_seen = True
    arbiter._dynamic_recovery_active = True
    arbiter._dynamic_escape_zero_translation_turn_steps = (
        arbiter.dynamic_escape_max_zero_translation_turn_steps
    )
    clear = {
        "emergency_stop": False,
        "should_slow_down": False,
        "reason": "front_clear",
        "temporal_scan_valid": False,
    }
    context = {
        "target_bearing_error": 1.0,
        "terminal_control_distance": 3.0,
        "probabilistic_obstacle_maximum_step_probability": 0.0,
    }

    released = arbiter.arbitrate(
        ControlCommand([0.4, 0.0]), clear, context
    )
    assert released.executed_control.values.tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert released.reason == "dynamic_recovery_budget_release"
    assert released.diagnostics[
        "dynamic_recovery_zero_turn_budget_release"
    ] is True
    assert arbiter._dynamic_recovery_active is False

    resumed = arbiter.arbitrate(
        ControlCommand([0.4, 0.0]), clear, context
    )
    assert resumed.executed_control.v == pytest.approx(0.4)


def test_directional_guard_ignores_rear_temporal_ttc_for_forward_motion(
        tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    decision = arbiter.arbitrate(
        ControlCommand([0.5, 0.0]),
        {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "reason": "temporal_collision_risk",
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 0.34,
            "temporal_scan_center_angle_rad": math.radians(-150.0),
        },
        {},
    )
    assert decision.executed_control.v == pytest.approx(0.5)
    assert decision.reason == "rear_pass_through"
    assert decision.diagnostics["directional_guard_original_reason"] == (
        "temporal_collision_risk"
    )


def test_front_clear_reactive_window_uses_rear_temporal_pass_through(tmp_path):
    """Replay 225246 cycle 62: scan guard clear, reactive TTC still live."""
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    rear_angle = math.radians(108.0)
    decision = arbiter.arbitrate(
        ControlCommand([0.5, -0.6]),
        {
            "emergency_stop": False,
            "should_slow_down": False,
            "reason": "front_clear",
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 2.72,
            "temporal_scan_center_angle_rad": rear_angle,
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_bearing_rad": math.radians(120.0),
            "min_front_range": 1.85,
        },
        {"probabilistic_obstacle_active_avoidance_enabled": True},
    )

    assert decision.reason == "rear_pass_through"
    assert decision.executed_control.v == pytest.approx(0.5)
    assert abs(decision.executed_control.omega) <= 0.30
    assert decision.diagnostics["rear_pass_through_active"] is True
    assert decision.diagnostics["dynamic_escape_allowed"] is False


def test_directional_guard_never_grants_tracker_only_global_release(tmp_path):
    config = build_pi5_full_config(
        _weight_root(tmp_path),
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    arbiter = ScanGuardArbiter(
        action_spec_from_config(config["action_space"]),
        config["perception"]["scan_guard"],
    )
    decision = arbiter.arbitrate(
        ControlCommand([0.5, 0.0]),
        {
            "emergency_stop": False,
            "reason": "front_clear",
            "dynamic_obstacle_bearing_rad": math.radians(165.0),
        },
        {},
    )
    assert decision.reason == "front_clear"
    assert decision.diagnostics["rear_pass_through_active"] is False
    assert decision.diagnostics["rear_pass_through_sources"] == ()


def test_livox_deskew_compensates_translation_and_rotation():
    adapter = LivoxScanAdapter()
    translation_frame = LivoxPointCloudFrame(
        timestamp_ns=1_000_000_000,
        points=np.asarray(((5.0, 0.0, 1.0), (4.0, 0.0, 1.0))),
        reflectivity=np.asarray((10, 10), dtype=np.uint8),
        tags=np.asarray((0, 0), dtype=np.uint8),
        point_timestamps_ns=np.asarray((0, 1_000_000_000), dtype=np.int64),
    )
    translated, diagnostics = _deskew_livox_frame(
        translation_frame, adapter, v_mps=1.0, omega_radps=0.0,
        maximum_age_s=1.0,
    )
    assert diagnostics["applied"] is True
    assert np.allclose(translated.points[:, :2], ((4.0, 0.0), (4.0, 0.0)))

    rotation_frame = LivoxPointCloudFrame(
        timestamp_ns=1_000_000_000,
        points=np.asarray(((1.0, 0.0, 1.0), (0.0, -1.0, 1.0))),
        reflectivity=np.asarray((10, 10), dtype=np.uint8),
        tags=np.asarray((0, 0), dtype=np.uint8),
        point_timestamps_ns=np.asarray((0, 1_000_000_000), dtype=np.int64),
    )
    rotated, _ = _deskew_livox_frame(
        rotation_frame, adapter, v_mps=0.0, omega_radps=np.pi / 2.0,
        maximum_age_s=1.0,
    )
    assert np.allclose(
        rotated.points[:, :2], ((0.0, -1.0), (0.0, -1.0)), atol=1e-6
    )
