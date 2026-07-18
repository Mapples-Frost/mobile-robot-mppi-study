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
