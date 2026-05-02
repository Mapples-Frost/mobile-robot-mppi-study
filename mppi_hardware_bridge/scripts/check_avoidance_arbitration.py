# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mppi_ros_adapter_skeleton import MppiRosAdapterSkeleton
from scan_guard import analyze_scan_front_sector
from scenario_config import load_lab_runtime_config


CONFIG_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "config", "lab_runtime_goal_3_3_safe.yaml")
)


def assert_condition(condition, message):
    if not condition:
        raise AssertionError(message)


def make_adapter(cfg):
    adapter = MppiRosAdapterSkeleton.__new__(MppiRosAdapterSkeleton)
    adapter.cfg = cfg
    adapter.latest_dynamic_obstacle_count = 0
    adapter.avoidance_side = "none"
    adapter.avoidance_side_until = 0.0
    adapter.avoidance_release_reason = "none"
    adapter.obstacle_turn_active = False
    adapter.side_avoid_active = False
    adapter.side_avoid_until = 0.0
    adapter.side_avoid_side = "none"
    adapter.last_smoothed_omega = 0.0
    adapter.last_control_time = None
    adapter.mppi_omega_trust_sign = 0
    adapter.mppi_omega_trust_count = 0
    adapter.front_soft_block_start_time = None
    adapter.previous_loop_start_time = None
    adapter.current_loop_start_time = None
    adapter.current_loop_dt_ms = None
    adapter.realtime_overrun_streak = 0
    adapter.realtime_degraded = False
    adapter.latest_corridor_debug = {}
    adapter.last_obstacle_turn_yaw_error_abs_deg = None
    return adapter


def base_scan_result(reason, min_front_range, raw_points=None):
    if raw_points is None:
        raw_points = []
    return {
        "reason": reason,
        "emergency_stop": False,
        "should_slow_down": False,
        "slow_scale": 1.0,
        "min_front_range": min_front_range,
        "min_side_range": None,
        "min_left_side_range": None,
        "min_right_side_range": None,
        "valid_front_count": 1,
        "valid_side_count": 0,
        "front_points": [],
        "side_points": [],
        "raw_points_base": list(raw_points),
        "side_avoid_mode": "none",
        "side_stop_mode": "none",
        "side_obstacle_side": "none",
        "front_stop_mode": reason,
        "front_stop_distance": None,
        "front_soft_block_distance": None,
        "hard_stop_distance": None,
    }


def make_scan_with_point(angle_rad, point_range, default_range=3.0):
    count = 361
    angle_min = -math.pi
    angle_increment = (2.0 * math.pi) / float(count - 1)
    ranges = [default_range for _ in range(count)]
    index = int(round((angle_rad - angle_min) / angle_increment))
    index = max(0, min(count - 1, index))
    ranges[index] = point_range
    return ranges, angle_min, angle_increment


def analyze_side_scan(cfg, angle_rad, point_range):
    ranges, angle_min, angle_increment = make_scan_with_point(angle_rad, point_range)
    return analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.05,
        range_max=6.0,
        front_stop_distance=cfg.safety.front_stop_distance,
        front_slow_distance=cfg.safety.front_slow_distance,
        front_angle_deg=cfg.safety.front_angle_deg,
        front_angle_offset_deg=0.0,
        side_stop_distance=getattr(cfg.safety, "side_stop_distance", 0.0),
        side_angle_deg=cfg.safety.side_angle_deg,
        near_body_stop_radius=cfg.safety.near_body_stop_radius,
        side_soft_distance=cfg.safety.side_soft_distance,
        side_hard_distance=cfg.safety.side_hard_distance,
        side_release_distance=cfg.safety.side_release_distance,
        side_front_exclusion_angle_deg=cfg.safety.side_front_exclusion_angle_deg,
    )


def test_front_clear_planner_obstacles_do_not_lock(cfg):
    adapter = make_adapter(cfg)
    adapter.obstacle_turn_active = True
    adapter.avoidance_side = "left"
    adapter.avoidance_side_until = time.time() + 10.0
    scan_result = base_scan_result(
        "front_clear",
        1.5,
        [{"x": 1.5, "y": 0.0, "range": 1.5}],
    )
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.09, 0.0),
        scan_result,
        planner_obstacles=2,
        heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
    )
    assert_condition(not debug["obstacle_turn_mode"], "front_clear must not keep obstacle turn active")
    assert_condition(debug["obstacle_turn_direction"] == "none", "front_clear must release left lock")
    assert_condition(abs(control[1]) < 1e-9, "front_clear must not inject omega")


def test_clear_front_turn_candidate_does_not_lock(cfg):
    adapter = make_adapter(cfg)
    adapter.planner_mode = "mppi"
    adapter.avoidance_state = "CLEAR"
    adapter.obstacle_turn_active = True
    adapter.avoidance_side = "left"
    adapter.avoidance_side_until = time.time() + 10.0
    min_front = float(cfg.safety.front_turn_distance) - 0.002
    scan_result = base_scan_result(
        "front_clear",
        min_front,
        [{"x": min_front, "y": 0.03, "range": min_front}],
    )
    scan_result["front_stop_mode"] = "clear"
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.10, 0.14),
        scan_result,
        planner_obstacles=0,
        heading_debug={"yaw_error": math.radians(20.0), "yaw_error_deg": 20.0},
    )
    assert_condition(debug["preemptive_turn_candidate"], "front_turn_distance should only mark candidate")
    assert_condition(not debug["obstacle_turn_mode"], "CLEAR must not enter obstacle_turn_mode")
    assert_condition(
        debug["obstacle_turn_release_reason"] == "release_state",
        "CLEAR should force release_state",
    )
    assert_condition(not debug["avoidance_direction_locked"], "CLEAR must release direction lock")
    assert_condition(abs(control[1] - 0.14) < 1e-9, "CLEAR candidate must preserve MPPI omega")

    _, tracking_debug = adapter._apply_goal_tracking_override(
        (0.0, 0.0, 0.0),
        (3.0, 3.0),
        control,
        cfg,
        {
            "scan_result": scan_result,
            "dynamic_obstacles": 0,
            "dynamic_obstacles_current_scan": 0,
            "planner_obstacles": 0,
            "obstacle_mode": "geometric",
            "proposed_omega": control[1],
            "avoidance_state": "CLEAR",
            "obstacle_turn_mode": debug["obstacle_turn_mode"],
            "side_soft_avoid": False,
        },
    )
    assert_condition(
        tracking_debug["tracking_mode"] != "preemptive_obstacle_turn",
        "CLEAR must not report preemptive obstacle tracking",
    )
    assert_condition(
        tracking_debug["goal_tracking_reason"] != "suppressed_during_obstacle_avoidance",
        "CLEAR must not suppress goal tracking for a front_turn candidate",
    )


def test_front_corridor_turn_keeps_forward_motion(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "APPROACH_SLOW"
    scan_result = base_scan_result(
        "front_obstacle_slow",
        0.70,
        [{"x": 0.70, "y": 0.10, "range": 0.707}],
    )
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.09, 0.0),
        scan_result,
        planner_obstacles=0,
        heading_debug={"yaw_error": 0.2, "yaw_error_deg": 11.5},
    )
    assert_condition(debug["obstacle_turn_mode"], "front corridor obstacle should enable turn")
    assert_condition(abs(control[1]) > 1e-6, "front corridor turn should inject omega")
    assert_condition(control[0] >= cfg.limits.v_max * 0.70, "preemptive turn should keep forward motion")


def test_side_soft_avoid_not_hard_stop(cfg):
    adapter = make_adapter(cfg)
    scan_result = analyze_side_scan(cfg, math.pi / 2.0, 0.25)
    assert_condition(not scan_result["emergency_stop"], "0.25m side obstacle should not hard stop")
    assert_condition(scan_result["reason"] == "side_obstacle_soft", "0.25m side obstacle should be soft")
    control, debug = adapter.apply_side_soft_avoid(
        (0.10, 0.05),
        scan_result,
        obstacle_turn_debug={"obstacle_turn_direction": "left"},
    )
    assert_condition(debug["side_avoid_mode"] == "side_soft_avoid", "side soft mode should apply")
    assert_condition(control[0] > 0.0, "side soft avoid should keep positive v")
    assert_condition(control[1] >= 0.0, "side soft avoid should not reverse MPPI omega by itself")
    assert_condition(
        debug["side_arbitration_mode"] == "sign_preserving_clamp",
        "opposite side soft correction should be sign-preserving",
    )


def test_side_hard_or_near_body_stop(cfg):
    scan_result = analyze_side_scan(cfg, math.pi / 2.0, 0.12)
    assert_condition(scan_result["emergency_stop"], "0.12m side obstacle must stop")
    assert_condition(
        scan_result["reason"] in ("side_hard_stop", "near_body_hard_stop"),
        "0.12m side obstacle must be hard or near-body stop",
    )


def test_front_clear_yaw_growth_releases_left_lock(cfg):
    adapter = make_adapter(cfg)
    adapter.obstacle_turn_active = True
    adapter.avoidance_side = "left"
    adapter.avoidance_side_until = time.time() + 10.0
    scan_result = base_scan_result(
        "front_clear",
        1.5,
        [{"x": 1.5, "y": 0.0, "range": 1.5}],
    )
    _, debug = adapter.apply_preemptive_obstacle_turn(
        (0.09, 0.16),
        scan_result,
        planner_obstacles=1,
        heading_debug={"yaw_error": math.radians(-25.0), "yaw_error_deg": -25.0},
    )
    assert_condition(not debug["obstacle_turn_mode"], "front_clear yaw growth must release obstacle turn")
    assert_condition(
        debug["obstacle_turn_release_reason"] in ("front_clear_release", "yaw_error_release", "release_state"),
        "release reason should explain front clear or yaw error",
    )


def test_mppi_omega_same_direction_is_preserved(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "APPROACH_SLOW"
    scan_result = base_scan_result(
        "front_obstacle_slow",
        0.45,
        [{"x": 0.45, "y": -0.08, "range": 0.457}],
    )
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.08, 0.12),
        scan_result,
        planner_obstacles=2,
        heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
    )
    assert_condition(debug["obstacle_turn_mode"], "front slow should enable obstacle turn bias")
    assert_condition(debug["mppi_omega_preserved"], "same-direction MPPI omega must be preserved")
    assert_condition(not debug["mppi_omega_overridden"], "same-direction MPPI omega must not be overridden")
    assert_condition(debug["omega_source"] in ("mppi", "mppi_biased"), "same direction should only bias")
    assert_condition(control[1] > 0.0, "same-direction left omega should remain left")


def test_mppi_omega_opposite_low_risk_is_not_overridden(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "APPROACH_SLOW"
    adapter.avoidance_state_reason = "front_corridor_obstacle"
    scan_result = base_scan_result(
        "front_clear",
        0.75,
        [{"x": 0.75, "y": -0.05, "range": 0.752}],
    )
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.08, -0.20),
        scan_result,
        planner_obstacles=1,
        heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
    )
    assert_condition(debug["obstacle_turn_mode"], "near corridor obstacle should activate turn")
    assert_condition(not debug["mppi_omega_overridden"], "low-risk opposite MPPI omega must not be forced")
    assert_condition(
        debug["hard_override_gate_reason"] == "risk_below_override_threshold",
        "low risk must be logged in hard override gate",
    )
    assert_condition(control[1] < 0.0, "low-risk opposite omega should preserve MPPI direction")


def test_mppi_omega_opposite_high_risk_can_override_with_reason(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "CREEP_ESCAPE"
    adapter.avoidance_state_reason = "front_soft_block"
    adapter.min_front_range_trend = "decreasing"
    adapter.goal_distance_trend = "stable"
    scan_result = base_scan_result(
        "front_soft_block",
        0.29,
        [{"x": 0.29, "y": -0.02, "range": 0.291}],
    )
    scan_result["front_stop_mode"] = "front_soft_block"
    control = (0.060, -0.12)
    debug = {}
    for _ in range(3):
        control, debug = adapter.apply_preemptive_obstacle_turn(
            (0.060, -0.12),
            scan_result,
            planner_obstacles=2,
            heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
        )
    assert_condition(debug["mppi_omega_overridden"], "high-risk front block may override opposite MPPI omega")
    assert_condition(debug["omega_source"] == "hard_override", "high-risk override must be explicit")
    assert_condition(debug["omega_override_reason"] != "none", "override reason must be logged")
    assert_condition(control[1] > 0.0, "override should turn toward selected safer direction")
    assert_condition(debug["hard_override_allowed"], "sustained high-risk conflict should open hard gate")


def test_hard_override_gate_requires_sustained_conflict(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "CREEP_ESCAPE"
    adapter.avoidance_state_reason = "front_soft_block"
    adapter.min_front_range_trend = "decreasing"
    adapter.goal_distance_trend = "stable"
    scan_result = base_scan_result(
        "front_soft_block",
        0.29,
        [{"x": 0.29, "y": -0.02, "range": 0.291}],
    )
    scan_result["front_stop_mode"] = "front_soft_block"
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.060, -0.45),
        scan_result,
        planner_obstacles=2,
        heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
    )
    assert_condition(not debug["hard_override_allowed"], "single-frame conflict must not open hard gate")
    assert_condition(not debug["mppi_omega_overridden"], "single-frame conflict must preserve MPPI sign")
    assert_condition(control[1] < 0.0, "single-frame conflict should not flip omega")
    assert_condition(debug["arbitration_mode"] in ("clamp", "mppi"), "single-frame conflict should clamp or pass")

    for _ in range(3):
        control, debug = adapter.apply_preemptive_obstacle_turn(
            (0.060, -0.45),
            scan_result,
            planner_obstacles=2,
            heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
        )
    assert_condition(debug["hard_override_allowed"], "sustained conflict should open hard gate")
    assert_condition(debug["mppi_omega_overridden"], "sustained conflict may hard override")


def test_corridor_hysteresis_blocks_single_frame_switch(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_side = "left"
    adapter.avoidance_side_until = time.time() + 0.4
    adapter.corridor_direction_since = time.time() - 0.1
    scan_result = base_scan_result(
        "front_obstacle_slow",
        0.45,
        [{"x": 0.45, "y": 0.12, "range": 0.466}],
    )
    selected = adapter.update_avoidance_side(
        scan_result,
        heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
        force_active=True,
    )
    debug = adapter.latest_corridor_debug
    assert_condition(selected == "left", "direction lock should block one-frame switch")
    assert_condition(debug["switch_blocked_by_lock"], "switch block should be logged")
    assert_condition(debug["corridor_switch_streak"] <= 1, "single-frame switch streak should stay short")


def test_creep_escape_limits_small_radius_turn(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "CREEP_ESCAPE"
    adapter.min_front_range_trend = "stable"
    adapter.goal_distance_trend = "stable"
    scan_result = base_scan_result(
        "front_soft_block",
        0.30,
        [{"x": 0.30, "y": 0.02, "range": 0.301}],
    )
    scan_result["front_stop_mode"] = "front_soft_block"
    control, debug = adapter.apply_front_soft_block_creep((0.12, 0.18), scan_result)
    assert_condition(debug["creep_escape_active"], "front soft block should enter creep escape")
    assert_condition(control[0] >= 0.055, "creep escape should keep usable forward speed")
    assert_condition(abs(control[1]) <= 0.14 + 1e-9, "stable creep should cap omega")
    assert_condition(debug["creep_progress_ok"], "stable front range should be progress-ok")


def test_goal_reacquire_releases_suppression_and_blocks_anti_goal_drive(cfg):
    adapter = make_adapter(cfg)
    adapter.planner_mode = "mppi"
    adapter.avoidance_state = "CREEP_ESCAPE"
    adapter.avoidance_side = "left"
    scan_result = base_scan_result("front_clear", 1.5, [])
    for _ in range(2):
        adapter.update_avoidance_state(
            scan_result,
            {"goal_distance": 1.0, "yaw_error_deg": 130.0},
            planner_obstacles=0,
            dynamic_obstacles_current_scan=0,
        )
    assert_condition(
        adapter.avoidance_state in ("GOAL_REACQUIRE", "CLEAR"),
        "front clear with no obstacles should release avoidance suppression",
    )
    assert_condition(adapter.avoidance_side == "none", "avoidance direction lock should release")
    control, debug = adapter._apply_goal_tracking_override(
        (0.0, 0.0, math.radians(180.0)),
        (3.0, 0.0),
        (0.12, 0.0),
        cfg,
        {
            "scan_result": scan_result,
            "dynamic_obstacles": 0,
            "dynamic_obstacles_current_scan": 0,
            "planner_obstacles": 0,
            "obstacle_mode": "geometric",
            "proposed_omega": 0.0,
            "avoidance_state": adapter.avoidance_state,
            "obstacle_turn_mode": False,
            "side_soft_avoid": False,
        },
    )
    assert_condition(debug["anti_goal_drive_blocked"], "large yaw error should block fast anti-goal drive")
    assert_condition(control[0] <= cfg.limits.v_max * 0.30, "anti-goal drive should be low speed")


def test_obstacle_smoothing_magnitude_clamp(cfg):
    adapter = make_adapter(cfg)
    adapter.rate_hz = 5.0
    adapter.last_control_time = time.time()
    adapter.last_smoothed_omega = 0.0
    control, debug = adapter.smooth_control_for_publish(
        (0.06, 0.12),
        context="creep",
    )
    assert_condition(abs(control[1]) >= 0.10, "obstacle smoothing should preserve useful omega magnitude")
    assert_condition(debug["smoothing_magnitude_clamped"], "magnitude clamp should be logged")


def test_clear_goal_tracking_smoothing_keeps_useful_omega(cfg):
    adapter = make_adapter(cfg)
    adapter.rate_hz = 5.0
    adapter.last_control_time = time.time()
    adapter.last_smoothed_omega = 0.0
    control, debug = adapter.smooth_control_for_publish(
        (0.10, 0.15),
        context="clear",
        goal_tracking_active=True,
        yaw_error_deg=45.0,
    )
    assert_condition(
        debug["smoothing_alpha_effective"] >= 0.25,
        "clear goal tracking should use responsive alpha",
    )
    assert_condition(
        abs(control[1]) >= 0.05,
        "clear goal tracking omega must not be smoothed down to near zero",
    )


def test_release_state_clears_obstacle_turn_lock(cfg):
    adapter = make_adapter(cfg)
    adapter.avoidance_state = "GOAL_REACQUIRE"
    adapter.obstacle_turn_active = True
    adapter.avoidance_side = "left"
    adapter.avoidance_side_until = time.time() + 10.0
    min_front = float(cfg.safety.front_turn_distance) - 0.01
    scan_result = base_scan_result(
        "front_clear",
        min_front,
        [{"x": min_front, "y": -0.02, "range": min_front}],
    )
    scan_result["front_stop_mode"] = "clear"
    control, debug = adapter.apply_preemptive_obstacle_turn(
        (0.10, -0.12),
        scan_result,
        planner_obstacles=0,
        heading_debug={"yaw_error": 0.0, "yaw_error_deg": 0.0},
    )
    assert_condition(not debug["obstacle_turn_mode"], "release_state must clear obstacle_turn_mode")
    assert_condition(not debug["avoidance_direction_locked"], "release_state must clear direction lock")
    assert_condition(debug["corridor_selected_direction"] == "none", "release_state must clear corridor direction")
    assert_condition(abs(control[1] + 0.12) < 1e-9, "release_state must preserve MPPI omega")


def test_goal_tracking_does_not_reverse_mppi_omega(cfg):
    adapter = make_adapter(cfg)
    adapter.planner_mode = "mppi"
    scan_result = base_scan_result(
        "front_clear",
        1.5,
        [{"x": 1.2, "y": 0.0, "range": 1.2}],
    )
    control, debug = adapter._apply_goal_tracking_override(
        (0.0, 0.0, 0.0),
        (3.0, 3.0),
        (0.12, -0.45),
        cfg,
        {
            "scan_result": scan_result,
            "dynamic_obstacles": 1,
            "dynamic_obstacles_current_scan": 1,
            "planner_obstacles": 1,
            "obstacle_mode": "geometric",
            "proposed_omega": -0.45,
            "obstacle_turn_mode": False,
            "side_soft_avoid": False,
        },
    )
    assert_condition(
        control[1] <= 0.0,
        "goal tracking must not reverse a nonzero MPPI omega",
    )
    assert_condition(
        debug["goal_tracking_reverse_blocked"],
        "reverse block should be logged when goal bearing asks for opposite omega",
    )
    assert_condition(
        debug["goal_tracking_reason"] == "mppi_omega_sign_preserved",
        "goal tracking should log MPPI sign preservation",
    )


def test_smoothing_sign_flip_blocked_in_obstacle_context(cfg):
    adapter = make_adapter(cfg)
    scan_result = base_scan_result("front_obstacle_slow", 0.45)
    obstacle_turn_debug = {
        "obstacle_turn_mode": True,
        "avoidance_direction_locked": False,
    }
    side_avoid_debug = {
        "side_avoid_applied": False,
        "side_avoid_mode": "none",
    }
    front_creep_debug = {"planner_creep_mode": False}
    planner_debug = {"recovery_direction": "none"}

    control, debug = adapter.block_smoothing_sign_flip_if_needed(
        (0.12, -0.20),
        (0.12, 0.18),
        scan_result,
        obstacle_turn_debug,
        side_avoid_debug,
        front_creep_debug,
        planner_debug,
    )
    assert_condition(debug["smoothing_sign_flip_blocked"], "negative-to-positive smoothing flip must be blocked")
    assert_condition(debug["smoothing_block_reason"] == "obstacle_context_sign_preserve", "block reason must be logged")
    assert_condition(abs(control[0] - 0.12) < 1e-9, "smoothing sign block must not change v")
    assert_condition(abs(control[1] + 0.20) < 1e-9, "blocked omega should preserve pre-smoothing negative omega")
    assert_condition(abs(adapter.last_smoothed_omega + 0.20) < 1e-9, "smoother omega state must be reset negative")

    control, debug = adapter.block_smoothing_sign_flip_if_needed(
        (0.12, 0.20),
        (0.12, -0.18),
        scan_result,
        obstacle_turn_debug,
        side_avoid_debug,
        front_creep_debug,
        planner_debug,
    )
    assert_condition(debug["smoothing_sign_flip_blocked"], "positive-to-negative smoothing flip must be blocked")
    assert_condition(abs(control[0] - 0.12) < 1e-9, "smoothing sign block must not change v")
    assert_condition(abs(control[1] - 0.20) < 1e-9, "blocked omega should preserve pre-smoothing positive omega")
    assert_condition(abs(adapter.last_smoothed_omega - 0.20) < 1e-9, "smoother omega state must be reset positive")


def main():
    cfg = load_lab_runtime_config(CONFIG_PATH)
    tests = [
        test_front_clear_planner_obstacles_do_not_lock,
        test_clear_front_turn_candidate_does_not_lock,
        test_front_corridor_turn_keeps_forward_motion,
        test_side_soft_avoid_not_hard_stop,
        test_side_hard_or_near_body_stop,
        test_front_clear_yaw_growth_releases_left_lock,
        test_mppi_omega_same_direction_is_preserved,
        test_mppi_omega_opposite_low_risk_is_not_overridden,
        test_hard_override_gate_requires_sustained_conflict,
        test_mppi_omega_opposite_high_risk_can_override_with_reason,
        test_corridor_hysteresis_blocks_single_frame_switch,
        test_creep_escape_limits_small_radius_turn,
        test_goal_reacquire_releases_suppression_and_blocks_anti_goal_drive,
        test_obstacle_smoothing_magnitude_clamp,
        test_clear_goal_tracking_smoothing_keeps_useful_omega,
        test_release_state_clears_obstacle_turn_lock,
        test_goal_tracking_does_not_reverse_mppi_omega,
        test_smoothing_sign_flip_blocked_in_obstacle_context,
    ]
    for test_func in tests:
        test_func(cfg)
        print("PASS {}".format(test_func.__name__))


if __name__ == "__main__":
    main()
