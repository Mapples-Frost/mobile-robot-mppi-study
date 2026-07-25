import copy
from pathlib import Path

import numpy as np

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.integration.gymnasium_adapter import GymnasiumAdapter
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from mobile_robot_mppi.runtime.factories import make_components
from mobile_robot_mppi.planning.rl_driven_mppi import (
    PaperRLDrivenMppiController,
)
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter
from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.core.types import ControlCommand


ROOT = Path(__file__).resolve().parents[2]


def small_legacy_config():
    config = load_yaml(ROOT / "configs/research/legacy_kinematic.yaml")
    config["experiment"]["max_steps"] = 4
    config["planner"]["horizon"] = 4
    config["planner"]["num_samples"] = 8
    return config


def test_runner_writes_reproducible_artifacts(tmp_path):
    result = ExperimentRunner(small_legacy_config(), ROOT, tmp_path, headless=True).run()
    assert result.summary["steps"] == 4
    assert (tmp_path / "config_resolved.yaml").exists()
    assert (tmp_path / "provenance.json").exists()
    assert (tmp_path / "trajectory.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert result.summary["termination_reason"] == "max_steps"
    assert "planner_compute_ms_p95" in result.summary
    assert "planner_deadline_miss_rate" in result.summary
    assert "path_boundary_candidate_feasible_fraction_mean" in result.summary
    assert "path_boundary_fallback_fraction" in result.summary


def test_gym_adapter_preserves_proposed_and_executed_actions():
    config = small_legacy_config()
    components = make_components(config, ROOT)
    environment = GymnasiumAdapter(components, config)
    try:
        observation, info = environment.reset(seed=4)
        assert observation.shape == (5,)
        following, reward, terminated, truncated, info = environment.step(np.asarray((0.2, 0.0)))
        assert following.shape == (5,)
        assert info["proposed_action"].shape == (2,)
        assert info["executed_action"].shape == (2,)
        assert isinstance(reward, float)
    finally:
        environment.close()


def test_runtime_goal_override_changes_task(tmp_path):
    config = small_legacy_config()
    config["task"]["position"] = [0.1, 0.0]
    result = ExperimentRunner(config, ROOT, tmp_path, headless=True).run()
    assert result.summary["final_goal_distance"] < 0.2


def test_waypoint_runner_reports_distance_to_final_waypoint(tmp_path):
    config = small_legacy_config()
    config["experiment"]["max_steps"] = 12
    config["task"] = {
        "type": "waypoints",
        "waypoints": [[0.05, 0.0], [0.15, 0.0]],
        "position_tolerance": 0.06,
    }
    result = ExperimentRunner(config, ROOT, tmp_path, headless=True).run()
    assert result.summary["success"]
    assert result.summary["final_goal_distance"] <= 0.06


def test_soft_block_stops_translation_but_preserves_turning():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec, {"front_soft_block_max_speed": 0.0}
    )
    proposed = ControlCommand(np.asarray((0.25, 0.6)))
    decision = arbiter.arbitrate(
        proposed,
        {
            "emergency_stop": False,
            "should_slow_down": True,
            "slow_scale": 1.0,
            "reason": "front_soft_block",
        },
    )
    np.testing.assert_allclose(decision.executed_control.values, (0.0, 0.6))
    assert decision.overridden


def test_soft_block_creep_is_strictly_capped_and_preserves_turning():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec, {"front_soft_block_max_speed": 0.04}
    )
    decision = arbiter.arbitrate(
        ControlCommand(np.asarray((0.25, -0.5))),
        {"reason": "front_soft_block", "should_slow_down": True, "slow_scale": 1.0},
    )
    np.testing.assert_allclose(decision.executed_control.values, (0.04, -0.5))


def test_dynamic_escape_requires_matched_obstacle_and_lower_risk_than_stop():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.20,
            "dynamic_escape_min_risk_improvement": 0.05,
        },
    )
    proposed = ControlCommand(np.asarray((0.30, 0.40)))
    context = {
        "probabilistic_obstacle_active_fallback_used": True,
        "probabilistic_obstacle_active_fallback_index": 12,
        "probabilistic_obstacle_maximum_step_probability": 0.30,
        "probabilistic_obstacle_stop_maximum_probability": 0.90,
    }
    escaped = arbiter.arbitrate(
        proposed,
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "dynamic_obstacle_near_body_match": True,
        },
        context,
    )
    np.testing.assert_allclose(
        escaped.executed_control.values, (0.20, 0.40)
    )
    assert escaped.reason == "dynamic_active_escape"
    assert escaped.diagnostics["dynamic_escape_allowed"]

    stopped = arbiter.arbitrate(
        proposed,
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "dynamic_obstacle_near_body_match": False,
        },
        context,
    )
    np.testing.assert_allclose(
        stopped.executed_control.values, (0.0, 0.40)
    )
    assert not stopped.diagnostics["dynamic_escape_allowed"]


def test_dynamic_escape_accepts_lower_mass_when_maximum_risk_is_saturated():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.20,
            "dynamic_escape_min_risk_improvement": 0.05,
            "dynamic_escape_min_probability_mass_improvement": 0.05,
            "dynamic_escape_probability_mass_enabled": True,
        },
    )
    decision = arbiter.arbitrate(
        ControlCommand(np.asarray((0.30, 0.40))),
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "dynamic_obstacle_near_body_match": True,
        },
        {
            "probabilistic_obstacle_active_fallback_used": True,
            "probabilistic_obstacle_active_fallback_index": 12,
            "probabilistic_obstacle_maximum_step_probability": 1.0,
            "probabilistic_obstacle_stop_maximum_probability": 1.0,
            "probabilistic_obstacle_probability_mass": 2.0,
            "probabilistic_obstacle_stop_probability_mass": 3.0,
        },
    )

    np.testing.assert_allclose(
        decision.executed_control.values, (0.20, 0.40)
    )
    assert decision.reason == "dynamic_active_escape"
    assert decision.diagnostics["dynamic_escape_probability_mass_fallback"]


def test_dynamic_escape_rejects_small_relative_mass_improvement():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.20,
            "dynamic_escape_min_risk_improvement": 0.05,
            "dynamic_escape_min_probability_mass_improvement": 0.05,
            "dynamic_escape_min_probability_mass_relative_improvement": 0.10,
            "dynamic_escape_probability_mass_enabled": True,
        },
    )
    decision = arbiter.arbitrate(
        ControlCommand(np.asarray((0.20, 0.40))),
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "dynamic_obstacle_near_body_match": True,
        },
        {
            "probabilistic_obstacle_active_fallback_used": True,
            "probabilistic_obstacle_active_fallback_index": 12,
            "probabilistic_obstacle_maximum_step_probability": 1.0,
            "probabilistic_obstacle_stop_maximum_probability": 1.0,
            "probabilistic_obstacle_probability_mass": 18.1,
            "probabilistic_obstacle_stop_probability_mass": 19.3,
        },
    )

    np.testing.assert_allclose(
        decision.executed_control.values, (0.0, 0.40)
    )
    assert not decision.diagnostics["dynamic_escape_allowed"]
    assert not decision.diagnostics[
        "dynamic_escape_probability_mass_fallback"
    ]


def test_dynamic_escape_probability_mass_order_is_opt_in():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.20,
            "dynamic_escape_min_risk_improvement": 0.05,
            "dynamic_escape_min_probability_mass_improvement": 0.05,
        },
    )
    decision = arbiter.arbitrate(
        ControlCommand(np.asarray((0.30, 0.40))),
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "dynamic_obstacle_near_body_match": True,
        },
        {
            "probabilistic_obstacle_active_fallback_used": True,
            "probabilistic_obstacle_active_fallback_index": 12,
            "probabilistic_obstacle_maximum_step_probability": 1.0,
            "probabilistic_obstacle_stop_maximum_probability": 1.0,
            "probabilistic_obstacle_probability_mass": 2.0,
            "probabilistic_obstacle_stop_probability_mass": 3.0,
        },
    )

    np.testing.assert_allclose(
        decision.executed_control.values, (0.0, 0.40)
    )
    assert not decision.diagnostics["dynamic_escape_allowed"]
    assert not decision.diagnostics[
        "dynamic_escape_probability_mass_enabled"
    ]


def test_reactive_dynamic_escape_turns_away_before_near_body_stop():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.20,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
        },
    )
    decision = arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, -0.20))),
        {
            "emergency_stop": False,
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": 0.80,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 0.90,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True
        },
    )
    np.testing.assert_allclose(
        decision.executed_control.values, (0.20, 0.90)
    )
    assert decision.reason == "dynamic_active_escape"
    assert decision.diagnostics["dynamic_escape_reactive"]


def test_reactive_dynamic_escape_can_preserve_vetted_planner_control():
    action_spec = body_velocity_action((-0.35, 0.35), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_reverse_speed": 0.35,
            "dynamic_escape_hold_enabled": True,
            "dynamic_escape_use_vetted_planner_control": True,
        },
    )
    proposed = ControlCommand(np.asarray((0.01, 0.11)))
    context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_maximum_step_probability": 0.0,
    }
    fresh = arbiter.arbitrate(
        proposed,
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": -2.20,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.00,
        },
        context,
    )
    dropout = arbiter.arbitrate(
        ControlCommand(np.asarray((0.02, -0.05))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": False,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 3.0,
        },
        context,
    )

    np.testing.assert_allclose(fresh.executed_control.values, proposed.values)
    np.testing.assert_allclose(
        dropout.executed_control.values, (0.02, -0.05)
    )
    assert fresh.reason == "dynamic_active_escape"
    assert fresh.diagnostics["dynamic_escape_vetted_planner_control"]
    assert not fresh.diagnostics["dynamic_escape_held"]
    assert not dropout.diagnostics["dynamic_escape_allowed"]


def test_reactive_dynamic_escape_hold_bridges_scan_flow_dropout():
    action_spec = body_velocity_action((0.0, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.20,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_hold_enabled": True,
            "dynamic_escape_hold_steps": 3,
            "dynamic_escape_hold_min_probability": 0.20,
        },
    )
    active_context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_maximum_step_probability": 0.90,
    }
    fresh = arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, -0.20))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": 0.80,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 0.90,
        },
        active_context,
    )
    held = arbiter.arbitrate(
        ControlCommand(np.asarray((0.0, 0.0))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": False,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 3.0,
        },
        active_context,
    )

    np.testing.assert_allclose(
        held.executed_control.values,
        fresh.executed_control.values,
    )
    assert held.reason == "dynamic_active_escape"
    assert held.diagnostics["dynamic_escape_held"]
    assert held.diagnostics["dynamic_escape_hold_remaining"] == 2

    released = arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, -0.20))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": False,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 3.0,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.10,
        },
    )
    np.testing.assert_allclose(
        released.executed_control.values, (0.10, -0.20)
    )
    assert not released.diagnostics["dynamic_escape_held"]


def test_reactive_dynamic_escape_reverses_for_rear_half_plane():
    action_spec = body_velocity_action((-0.2, 0.4), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_reverse_speed": 0.20,
        },
    )
    decision = arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, -0.20))),
        {
            "emergency_stop": False,
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": -2.20,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.00,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True
        },
    )
    assert np.isclose(decision.executed_control.values[0], -0.20)
    assert decision.executed_control.values[1] > 0.0
    assert decision.diagnostics["dynamic_escape_reverse"]


def test_dynamic_recovery_waits_for_clear_hold_then_aligns_to_goal():
    action_spec = body_velocity_action((-0.35, 0.35), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_reverse_speed": 0.35,
            "dynamic_recovery_enabled": True,
            "dynamic_recovery_entry_probability": 0.10,
            "dynamic_recovery_abort_probability": 0.15,
            "dynamic_recovery_clear_hold_steps": 2,
            "dynamic_recovery_heading_tolerance_rad": 0.30,
            "dynamic_recovery_min_speed": 0.30,
            "dynamic_recovery_turn_gain": 1.50,
            "dynamic_recovery_release_steps": 3,
        },
    )
    escaped = arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, 0.0))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": 0.80,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.00,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.18,
        },
    )
    assert escaped.reason == "dynamic_active_escape"

    clear_guard = {
        "reason": "front_clear",
        "temporal_scan_valid": False,
    }
    recovery_context = {
        "probabilistic_obstacle_active_avoidance_enabled": True,
        "probabilistic_obstacle_maximum_step_probability": 0.05,
        "terminal_bearing_error": 0.80,
        "terminal_control_distance": 4.0,
    }
    waiting = arbiter.arbitrate(
        ControlCommand(np.asarray((0.0, 0.0))),
        clear_guard,
        recovery_context,
    )
    assert waiting.diagnostics["dynamic_recovery_pending"]
    assert not waiting.diagnostics["dynamic_recovery_active"]

    aligning = arbiter.arbitrate(
        ControlCommand(np.asarray((0.0, 0.0))),
        clear_guard,
        recovery_context,
    )
    np.testing.assert_allclose(
        aligning.executed_control.values, (0.0, 0.9)
    )
    assert aligning.reason == "dynamic_recovery_align"
    assert aligning.diagnostics["dynamic_recovery_active"]


def test_dynamic_recovery_does_not_override_vetted_temporal_escape():
    action_spec = body_velocity_action((-0.35, 0.35), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_recovery_enabled": True,
            "dynamic_recovery_entry_probability": 0.10,
            "dynamic_recovery_abort_probability": 0.15,
            "dynamic_recovery_clear_hold_steps": 1,
            "dynamic_recovery_heading_tolerance_rad": 0.30,
            "dynamic_recovery_minimum_heading_error_rad": 0.80,
            "dynamic_recovery_translation_enabled": False,
        },
    )
    arbiter._dynamic_escape_seen = True
    proposed = ControlCommand(np.asarray((0.35, 0.90)))

    decision = arbiter.arbitrate(
        proposed,
        {
            "reason": "front_clear",
            "temporal_scan_valid": False,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.05,
            "probabilistic_obstacle_temporal_emergency_vetted": True,
            "probabilistic_obstacle_emergency_candidate_selected": True,
            "target_bearing_error": -1.20,
            "terminal_control_distance": 4.0,
        },
    )

    np.testing.assert_allclose(
        decision.executed_control.values, proposed.values
    )
    assert decision.diagnostics["planner_temporal_escape_active"]
    assert not decision.diagnostics["dynamic_recovery_active"]


def test_dynamic_recovery_advances_when_aligned_and_aborts_on_new_risk():
    action_spec = body_velocity_action((-0.35, 0.35), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_reverse_speed": 0.35,
            "dynamic_recovery_enabled": True,
            "dynamic_recovery_entry_probability": 0.10,
            "dynamic_recovery_abort_probability": 0.15,
            "dynamic_recovery_clear_hold_steps": 1,
            "dynamic_recovery_heading_tolerance_rad": 0.30,
            "dynamic_recovery_min_speed": 0.30,
            "dynamic_recovery_turn_gain": 1.50,
            "dynamic_recovery_release_steps": 3,
        },
    )
    arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, 0.0))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": -2.20,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.00,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.18,
        },
    )
    advancing = arbiter.arbitrate(
        ControlCommand(np.asarray((0.0, -0.2))),
        {"reason": "front_clear", "temporal_scan_valid": False},
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.05,
            "terminal_bearing_error": 0.10,
            "terminal_control_distance": 4.0,
        },
    )
    np.testing.assert_allclose(
        advancing.executed_control.values, (0.30, 0.15)
    )
    assert advancing.reason == "dynamic_recovery_advance"

    aborted = arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, 0.20))),
        {"reason": "front_clear", "temporal_scan_valid": False},
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.16,
            "terminal_bearing_error": 0.05,
            "terminal_control_distance": 3.0,
        },
    )
    np.testing.assert_allclose(
        aborted.executed_control.values, (0.10, 0.20)
    )
    assert aborted.diagnostics["dynamic_recovery_mode"] == "aborted"
    assert not aborted.diagnostics["dynamic_recovery_active"]


def test_rotation_only_recovery_never_forces_unevaluated_translation():
    action_spec = body_velocity_action((-0.35, 0.35), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_reverse_speed": 0.35,
            "dynamic_recovery_enabled": True,
            "dynamic_recovery_entry_probability": 0.10,
            "dynamic_recovery_abort_probability": 0.15,
            "dynamic_recovery_clear_hold_steps": 1,
            "dynamic_recovery_heading_tolerance_rad": 0.20,
            "dynamic_recovery_translation_enabled": False,
            "dynamic_recovery_turn_gain": 1.50,
        },
    )
    arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, 0.0))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": 0.80,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.00,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.18,
        },
    )
    aligning = arbiter.arbitrate(
        ControlCommand(np.asarray((0.25, -0.20))),
        {"reason": "front_clear", "temporal_scan_valid": False},
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.05,
            "terminal_bearing_error": 0.60,
            "terminal_control_distance": 4.0,
        },
    )
    np.testing.assert_allclose(
        aligning.executed_control.values, (0.0, 0.9)
    )
    assert aligning.reason == "dynamic_recovery_align"

    released = arbiter.arbitrate(
        ControlCommand(np.asarray((0.0, 0.0))),
        {"reason": "front_clear", "temporal_scan_valid": False},
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.05,
            "terminal_bearing_error": 0.10,
            "terminal_control_distance": 4.0,
        },
    )
    np.testing.assert_allclose(
        released.executed_control.values, (0.0, 0.0)
    )
    assert (
        released.diagnostics["dynamic_recovery_mode"]
        == "aligned_release"
    )
    assert not released.diagnostics["dynamic_recovery_active"]
    assert not released.diagnostics["dynamic_recovery_pending"]


def test_rotation_only_recovery_does_not_perturb_small_heading_error():
    action_spec = body_velocity_action((-0.35, 0.35), 0.9)
    arbiter = ScanGuardArbiter(
        action_spec,
        {
            "dynamic_escape_enabled": True,
            "dynamic_escape_max_speed": 0.35,
            "dynamic_escape_reactive_enabled": True,
            "dynamic_escape_trigger_ttc_s": 1.50,
            "dynamic_escape_turn_gain": 1.50,
            "dynamic_escape_min_speed": 0.20,
            "dynamic_escape_reverse_speed": 0.35,
            "dynamic_recovery_enabled": True,
            "dynamic_recovery_entry_probability": 0.10,
            "dynamic_recovery_abort_probability": 0.15,
            "dynamic_recovery_clear_hold_steps": 1,
            "dynamic_recovery_heading_tolerance_rad": 0.20,
            "dynamic_recovery_minimum_heading_error_rad": 0.80,
            "dynamic_recovery_translation_enabled": False,
        },
    )
    arbiter.arbitrate(
        ControlCommand(np.asarray((0.10, 0.0))),
        {
            "reason": "front_clear",
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": 0.80,
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.00,
        },
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.18,
        },
    )
    unchanged = arbiter.arbitrate(
        ControlCommand(np.asarray((0.25, -0.20))),
        {"reason": "front_clear", "temporal_scan_valid": False},
        {
            "probabilistic_obstacle_active_avoidance_enabled": True,
            "probabilistic_obstacle_maximum_step_probability": 0.05,
            "target_bearing_error": 0.34,
            "terminal_control_distance": 4.0,
        },
    )
    np.testing.assert_allclose(
        unchanged.executed_control.values, (0.25, -0.20)
    )
    assert (
        unchanged.diagnostics["dynamic_recovery_mode"]
        == "not_needed"
    )
    assert not unchanged.diagnostics["dynamic_recovery_pending"]


def test_strong_mujoco_guard_envelope_exceeds_collision_radius():
    config = load_yaml(ROOT / "configs/research/mujoco_strong_mppi_baseline.yaml")
    radius = float(config["plant"]["robot"]["collision_radius"])
    guard = config["perception"]["scan_guard"]
    assert float(guard["hard_stop_distance"]) > radius
    assert float(guard["side_stop_distance"]) > radius
    assert float(guard["near_body_stop_radius"]) > radius


def test_paper_rl_driven_factory_is_explicit_and_opt_in():
    config = small_legacy_config()
    config["planner"].update({
        "optimizer": "paper_rl_driven",
        "sampling_prior": "paper_direct_rl",
        "importance_sampling_correction": False,
        "paper_rl_driven": {
            "iterations": 2,
            "guided_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    })
    config["rl"] = {"enabled": True}

    class DirectPolicy:
        def propose(self, *args, **kwargs):
            raise AssertionError("factory construction must not run policy")

        def action_distribution(self, *args, **kwargs):
            raise AssertionError("factory construction must not run policy")

        def sample_actions(self, *args, **kwargs):
            raise AssertionError("factory construction must not run policy")

        def terminal_value(self, *args, **kwargs):
            raise AssertionError("factory construction must not run policy")

    components = make_components(config, ROOT, rl_policy=DirectPolicy())
    try:
        assert isinstance(
            components["controller"], PaperRLDrivenMppiController
        )
    finally:
        components["plant"].close()
