# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from local_obstacle_layer import scan_to_experiment_obstacles_geometric
from mppi_ros_adapter_skeleton import MppiRosAdapterSkeleton
from scan_guard import analyze_scan_front_sector
from scenario_config import load_lab_runtime_config


CONFIG_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "config", "lab_runtime_goal_3_3_safe.yaml")
)


def assert_condition(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeRospy(object):
    def __init__(self):
        self.infos = []
        self.warns = []
        self.errs = []

    def loginfo(self, message):
        self.infos.append(str(message))

    def logwarn(self, message):
        self.warns.append(str(message))

    def logerr(self, message):
        self.errs.append(str(message))


class FakeBridge(object):
    def __init__(self, control=(0.12, -0.12), compute_ms=20.0):
        self.control = control
        self.compute_ms = float(compute_ms)
        self.obstacles = []
        self.called = False
        self.profile_calls = []
        self.effective_num_samples = 350
        self.effective_horizon = 25
        self.degraded_reason = "normal"

    def set_obstacles(self, obstacles):
        self.obstacles = list(obstacles)

    def clear_obstacles(self):
        self.obstacles = []

    def set_runtime_profile(self, profile_name, num_samples=None, horizon=None, reason="none"):
        if num_samples is not None:
            self.effective_num_samples = int(num_samples)
        if horizon is not None:
            self.effective_horizon = int(horizon)
        self.degraded_reason = str(reason)
        self.profile_calls.append(
            (
                str(profile_name),
                self.effective_num_samples,
                self.effective_horizon,
                self.degraded_reason,
            )
        )

    def compute_control(self, current_state_exp, goal_xy):
        self.called = True
        return self.control, {
            "planner_type": "mppi",
            "proposed_control": self.control,
            "raw_mppi_control": self.control,
            "num_samples": self.effective_num_samples,
            "effective_num_samples": self.effective_num_samples,
            "horizon": self.effective_horizon,
            "effective_horizon": self.effective_horizon,
            "degraded_reason": self.degraded_reason,
            "safety_intervened": False,
            "best_cost": 1.0,
            "obstacle_count": len(self.obstacles),
            "planner_compute_ms": self.compute_ms,
            "planner_core_compute_ms": self.compute_ms,
            "goal_xy": goal_xy,
            "current_state_exp": current_state_exp,
        }


def make_scan_result(cfg, reason, min_front_range, raw_points=None):
    if raw_points is None:
        raw_points = []
    return {
        "reason": reason,
        "emergency_stop": False,
        "should_slow_down": reason in ("front_obstacle_slow", "front_soft_block"),
        "slow_scale": 1.0,
        "min_front_range": min_front_range,
        "min_side_range": None,
        "min_left_side_range": None,
        "min_right_side_range": None,
        "valid_front_count": 1,
        "valid_side_count": 0,
        "front_points": list(raw_points),
        "side_points": [],
        "raw_points_base": list(raw_points),
        "side_avoid_mode": "none",
        "side_stop_mode": "none",
        "side_obstacle_side": "none",
        "front_stop_mode": reason,
        "front_stop_distance": cfg.safety.front_stop_distance,
        "front_soft_block_distance": cfg.safety.front_soft_block_distance,
        "hard_stop_distance": cfg.safety.hard_stop_distance,
        "front_slow_min_scale": cfg.safety.front_slow_min_scale,
    }


def make_runtime_adapter(cfg, scan_result, bridge):
    adapter = MppiRosAdapterSkeleton.__new__(MppiRosAdapterSkeleton)
    adapter.rospy = FakeRospy()
    adapter.cfg = cfg
    adapter.rate_hz = 5.0
    adapter.planner_mode = "mppi"
    adapter.mppi_bridge = bridge
    adapter.current_state_exp = (0.0, 0.0, 0.0)
    adapter.latest_odom_time = time.time()
    adapter.latest_scan_result = scan_result
    adapter.latest_scan_time = time.time()
    adapter.latest_scan_msg_fields = None
    adapter.latest_local_obstacle_count = 0
    adapter.latest_dynamic_obstacle_count = 0
    adapter.latest_dynamic_obstacle_current_scan_count = 0
    adapter.latest_dynamic_obstacle_debug = {}
    adapter.goal_xy = (float(cfg.goal.x), float(cfg.goal.y))
    adapter.goal_reached_latched = False
    adapter.front_stop_recovery_active = False
    adapter.front_stop_recovery_direction = "none"
    adapter.front_soft_block_start_time = None
    adapter.obstacle_turn_active = False
    adapter.side_avoid_active = False
    adapter.side_avoid_until = 0.0
    adapter.side_avoid_side = "none"
    adapter.avoidance_side = "none"
    adapter.avoidance_side_until = 0.0
    adapter.avoidance_release_reason = "none"
    adapter.latest_corridor_debug = {}
    adapter.last_obstacle_turn_yaw_error_abs_deg = None
    adapter.mppi_omega_trust_sign = 0
    adapter.mppi_omega_trust_count = 0
    adapter.last_control_time = None
    adapter.last_smoothed_v = 0.0
    adapter.last_smoothed_omega = 0.0
    adapter.previous_loop_start_time = None
    adapter.current_loop_start_time = None
    adapter.current_loop_dt_ms = None
    adapter.realtime_overrun_streak = 0
    adapter.realtime_degraded = False
    adapter.mppi_runtime_profile = "normal"
    adapter.mppi_degraded_reason = "normal"
    adapter.mppi_core_overrun_streak = 0
    adapter.mppi_core_recover_streak = 0
    adapter.enable_publish = False
    adapter.cmd_pub = None
    adapter.debug_pubs = {}
    adapter.debug_types = {}
    adapter.debug_publish_last_time = {}
    adapter.cycle_status_count = 0
    adapter.last_cycle_status_signature = None
    adapter.last_scan_frame_id = "base_footprint"
    adapter.base_frame_id = "base_footprint"
    adapter.latest_scan_effective_offset_deg = 0.0
    adapter.latest_scan_tf_ok = True
    adapter.front_angle_offset_deg = 0.0
    adapter.odom_timeout_sec = 1.0
    adapter.scan_timeout_sec = 1.0
    return adapter


def make_scan_with_point(angle_rad, point_range, default_range=3.0):
    count = 361
    angle_min = -math.pi
    angle_increment = (2.0 * math.pi) / float(count - 1)
    ranges = [default_range for _ in range(count)]
    index = int(round((angle_rad - angle_min) / angle_increment))
    index = max(0, min(count - 1, index))
    ranges[index] = point_range
    return ranges, angle_min, angle_increment


def analyze_front_scan(cfg, point_range):
    ranges, angle_min, angle_increment = make_scan_with_point(0.0, point_range)
    return analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.05,
        range_max=6.0,
        front_stop_distance=cfg.safety.front_stop_distance,
        front_slow_distance=cfg.safety.front_slow_distance,
        front_angle_deg=cfg.safety.front_angle_deg,
        hard_stop_distance=cfg.safety.hard_stop_distance,
        front_soft_block_distance=cfg.safety.front_soft_block_distance,
        front_slow_min_scale=cfg.safety.front_slow_min_scale,
        front_angle_offset_deg=0.0,
        side_stop_distance=getattr(cfg.safety, "side_stop_distance", 0.0),
        side_angle_deg=cfg.safety.side_angle_deg,
        near_body_stop_radius=cfg.safety.near_body_stop_radius,
        side_soft_distance=cfg.safety.side_soft_distance,
        side_hard_distance=cfg.safety.side_hard_distance,
        side_release_distance=cfg.safety.side_release_distance,
        side_front_exclusion_angle_deg=cfg.safety.side_front_exclusion_angle_deg,
    )


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
        hard_stop_distance=cfg.safety.hard_stop_distance,
        front_soft_block_distance=cfg.safety.front_soft_block_distance,
        front_slow_min_scale=cfg.safety.front_slow_min_scale,
        front_angle_offset_deg=0.0,
        side_stop_distance=getattr(cfg.safety, "side_stop_distance", 0.0),
        side_angle_deg=cfg.safety.side_angle_deg,
        near_body_stop_radius=cfg.safety.near_body_stop_radius,
        side_soft_distance=cfg.safety.side_soft_distance,
        side_hard_distance=cfg.safety.side_hard_distance,
        side_release_distance=cfg.safety.side_release_distance,
        side_front_exclusion_angle_deg=cfg.safety.side_front_exclusion_angle_deg,
    )


def make_wall_scan(x_value=1.0):
    ranges = [float("inf")] * 360
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / 360.0
    for index in range(31):
        y_value = -0.45 + 0.90 * float(index) / 30.0
        angle = math.atan2(y_value, x_value)
        scan_index = int(round((angle - angle_min) / angle_increment))
        if scan_index < 0 or scan_index >= len(ranges):
            continue
        ranges[scan_index] = math.hypot(x_value, y_value)
    return ranges, angle_min, angle_increment


def test_front_obstacle_slow_keeps_min_velocity(cfg):
    adapter = make_runtime_adapter(
        cfg,
        make_scan_result(cfg, "front_obstacle_slow", 0.40),
        FakeBridge(),
    )
    control, debug = adapter.apply_soft_avoid_velocity_floor(
        (0.02, 0.0),
        adapter.latest_scan_result,
    )
    assert_condition(control[0] >= cfg.safety.soft_avoid_min_v, "front slow must keep soft min v")
    assert_condition(debug["soft_avoid_reason"] == "front_obstacle_slow", "front slow floor reason missing")


def test_front_soft_block_calls_planner_and_creeps(cfg):
    scan_result = make_scan_result(
        cfg,
        "front_soft_block",
        0.29,
        [{"x": 0.29, "y": -0.02, "range": 0.291}],
    )
    bridge = FakeBridge(control=(0.12, -0.12), compute_ms=25.0)
    adapter = make_runtime_adapter(cfg, scan_result, bridge)
    adapter.run_once()
    assert_condition(bridge.called, "front_soft_block must call planner")
    log_text = "\n".join(adapter.rospy.infos)
    assert_condition("planner_creep_mode=True" in log_text, "front_soft_block must enter planner creep mode")
    assert_condition(
        "planner_called_under_front_block=True" in log_text,
        "front block planner call must be logged",
    )
    assert_condition("planner_skip_reason=none" in log_text, "front_soft_block must not skip planner")
    assert_condition("creep_escape_active=True" in log_text, "front_soft_block must log creep escape")
    assert_condition("creep_v_selected=0.06" in log_text, "creep escape should select about 0.06 m/s")
    assert_condition("final_control=(0.02" not in log_text, "creep escape must not collapse to 0.02 m/s")


def test_hard_stop_and_near_body_remain_hard(cfg):
    hard_scan = analyze_front_scan(cfg, 0.16)
    assert_condition(hard_scan["emergency_stop"], "front hard stop must be emergency")
    assert_condition(
        hard_scan["reason"] in ("hard_stop", "near_body_hard_stop"),
        "front close obstacle must be hard_stop or near_body_hard_stop",
    )

    near_body_scan = analyze_side_scan(cfg, math.pi / 2.0, 0.12)
    assert_condition(near_body_scan["emergency_stop"], "near body side point must be emergency")
    assert_condition(
        near_body_scan["reason"] in ("near_body_hard_stop", "side_hard_stop"),
        "near body or side hard stop reason expected",
    )


def test_side_soft_keeps_min_velocity(cfg):
    adapter = make_runtime_adapter(
        cfg,
        analyze_side_scan(cfg, math.pi / 2.0, 0.24),
        FakeBridge(),
    )
    control, debug = adapter.apply_side_soft_avoid(
        (0.02, 0.0),
        adapter.latest_scan_result,
        obstacle_turn_debug={"obstacle_turn_direction": "none"},
    )
    assert_condition(debug["side_avoid_mode"] == "side_soft_avoid", "side soft should apply")
    assert_condition(control[0] >= cfg.safety.side_soft_min_v, "side soft must keep min controllable v")


def test_goal_tracking_log_does_not_reverse_mppi_omega(cfg):
    adapter = make_runtime_adapter(
        cfg,
        make_scan_result(cfg, "front_clear", 1.5),
        FakeBridge(control=(0.12, -0.45), compute_ms=20.0),
    )
    adapter.run_once()
    log_text = "\n".join(adapter.rospy.infos)
    assert_condition(
        "omega_source=goal_tracking" in log_text,
        "goal tracking intervention should be visible in the log",
    )
    assert_condition(
        "omega_after_tracking=-" in log_text,
        "goal tracking must keep negative MPPI omega negative",
    )
    assert_condition(
        "omega_after_tracking=0.1" not in log_text,
        "goal tracking must not flip MPPI omega to positive",
    )
    assert_condition(
        "mppi_omega_preserved=False" in log_text,
        "log should not claim MPPI omega is preserved after goal tracking limits it",
    )


def test_realtime_degraded_sets_flag_without_disabling_guard(cfg):
    adapter = make_runtime_adapter(
        cfg,
        make_scan_result(cfg, "front_clear", 1.5),
        FakeBridge(),
    )
    adapter.current_loop_start_time = time.time() - 0.25
    adapter.current_loop_dt_ms = 260.0
    adapter.realtime_overrun_streak = 2
    adapter.print_cycle_status(
        state=(0.0, 0.0, 0.0),
        goal=(3.0, 3.0),
        proposed_control=(0.12, 0.0),
        scan_result=adapter.latest_scan_result,
        final_control=(0.12, 0.0),
        note="realtime_test",
        planner_debug={
            "planner_type": "mppi",
            "planner_compute_ms": 190.0,
            "obstacle_count": 4,
            "num_samples": 350,
            "safety_intervened": False,
            "best_cost": 1.0,
        },
    )
    assert_condition(adapter.realtime_degraded, "sustained overrun should enter realtime_degraded")
    assert_condition(cfg.safety.enable_scan_guard, "realtime degraded must not disable scan_guard")
    assert_condition(cfg.mppi.dynamic_obstacle_max_count <= 4, "dynamic obstacle budget must stay <= 4")
    assert_condition(cfg.mppi.geometric_max_obstacle_count <= 4, "geometric obstacle budget must stay <= 4")


def test_mppi_profile_degrades_after_sustained_core_overrun(cfg):
    bridge = FakeBridge(control=(0.12, -0.12), compute_ms=520.0)
    adapter = make_runtime_adapter(
        cfg,
        make_scan_result(cfg, "front_clear", 1.5),
        bridge,
    )
    adapter.run_once()
    assert_condition(
        adapter.mppi_runtime_profile == "realtime_degraded",
        "core compute over threshold should select degraded profile for next frame",
    )
    adapter.run_once()
    assert_condition(
        bridge.profile_calls[-1][1] == cfg.mppi.realtime_degraded_num_samples,
        "next frame should use degraded sample count",
    )
    assert_condition(
        bridge.profile_calls[-1][2] == cfg.mppi.realtime_degraded_horizon,
        "next frame should use degraded horizon",
    )
    log_text = "\n".join(adapter.rospy.infos)
    assert_condition(
        "effective_num_samples={}".format(cfg.mppi.realtime_degraded_num_samples) in log_text,
        "degraded sample count must be logged",
    )


def test_degraded_scan_guard_still_hard_stops(cfg):
    bridge = FakeBridge(control=(0.12, 0.0), compute_ms=20.0)
    adapter = make_runtime_adapter(
        cfg,
        analyze_front_scan(cfg, 0.16),
        bridge,
    )
    adapter.mppi_runtime_profile = "realtime_degraded"
    adapter.mppi_degraded_reason = "test_degraded"
    adapter.run_once()
    assert_condition(not bridge.called, "hard stop must skip planner even while degraded")
    log_text = "\n".join(adapter.rospy.infos)
    assert_condition("final_control=(0.000" in log_text, "hard stop should keep v=0")
    assert_condition(
        "planner_skipped_no_core_update=True" in log_text,
        "planner-not-called cycles should not update core overrun streak",
    )
    assert_condition(cfg.safety.enable_scan_guard, "scan_guard must remain enabled")


def test_degraded_keeps_planner_obstacles(cfg):
    ranges, angle_min, angle_increment = make_wall_scan(x_value=0.45)
    bridge = FakeBridge(control=(0.12, 0.0), compute_ms=20.0)
    adapter = make_runtime_adapter(
        cfg,
        make_scan_result(cfg, "front_clear", 1.5),
        bridge,
    )
    adapter.mppi_runtime_profile = "realtime_degraded"
    adapter.mppi_degraded_reason = "test_degraded"
    adapter.latest_scan_msg_fields = {
        "ranges": ranges,
        "angle_min": angle_min,
        "angle_increment": angle_increment,
        "range_min": 0.05,
        "range_max": 6.0,
        "effective_angle_offset_deg": 0.0,
    }
    adapter.run_once()
    assert_condition(bridge.called, "degraded profile should still call planner")
    assert_condition(len(bridge.obstacles) > 0, "degraded profile must not zero planner obstacles")
    assert_condition(len(bridge.obstacles) <= 4, "degraded profile should keep bounded obstacle budget")


def test_long_obstacle_compression(cfg):
    ranges, angle_min, angle_increment = make_wall_scan()
    obstacles, debug = scan_to_experiment_obstacles_geometric(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.05,
        range_max=6.0,
        current_state_exp=(0.0, 0.0, 0.0),
        max_radius=2.0,
        downsample_step=1,
        obstacle_radius=cfg.safety.local_obstacle_radius,
        max_obstacle_count=cfg.mppi.geometric_max_obstacle_count,
        line_sample_spacing=cfg.mppi.geometric_line_spacing,
        line_max_obstacles=cfg.mppi.geometric_line_max_obstacles,
        obstacle_inflation=cfg.mppi.geometric_obstacle_inflation,
        return_debug=True,
    )
    assert_condition(debug["line_surface_count"] >= 1, "wall scan should detect a line surface")
    assert_condition(debug["compressed_line_obstacles"] <= 3, "line surface should compress to <= 3 circles")
    assert_condition(len(obstacles) <= 4, "planner obstacle budget should stay around 4")
    assert_condition(debug["segment_count"] < 20, "line wall should not explode segment count")


def main():
    cfg = load_lab_runtime_config(CONFIG_PATH)
    tests = [
        test_front_obstacle_slow_keeps_min_velocity,
        test_front_soft_block_calls_planner_and_creeps,
        test_hard_stop_and_near_body_remain_hard,
        test_side_soft_keeps_min_velocity,
        test_goal_tracking_log_does_not_reverse_mppi_omega,
        test_realtime_degraded_sets_flag_without_disabling_guard,
        test_mppi_profile_degrades_after_sustained_core_overrun,
        test_degraded_scan_guard_still_hard_stops,
        test_degraded_keeps_planner_obstacles,
        test_long_obstacle_compression,
    ]
    for test_func in tests:
        test_func(cfg)
        print("PASS {}".format(test_func.__name__))


if __name__ == "__main__":
    main()
