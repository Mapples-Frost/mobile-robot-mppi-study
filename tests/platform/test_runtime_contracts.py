import copy
from pathlib import Path

import numpy as np

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.integration.gymnasium_adapter import GymnasiumAdapter
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from mobile_robot_mppi.runtime.factories import make_components


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
