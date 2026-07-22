from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l261_screen_is_small_bounded_rl_only_and_isolated():
    config = load_yaml(ROOT / "configs/rl/l261_value_stability_3k.yaml")
    reward = config["rl"]["reward"]
    training = config["rl"]["training"]
    assert training["seed"] == 20262601
    assert training["validation_seed_base"] == 20263601
    assert training["total_steps"] == 3000
    assert training["update_after"] == 500
    assert training["actor_update_after"] == 1000
    assert training["evaluation_interval"] == 1500
    assert training["checkpoint_interval"] == 1500
    assert training["bc_anchor"]["enabled"] is False
    assert training["device"] == "cuda"
    assert len(training["scene_configs"]) == 6
    assert len(training["validation_scene_configs"]) == 3
    assert all("l261_value_stability/train" in p for p in training["scene_configs"])
    assert all("l261_value_stability/validation" in p for p in training["validation_scene_configs"])
    phases = training["curriculum"]["phases"]
    assert [phase["until_step"] for phase in phases] == [
        500, 1000, 1500, 2000, 2500, 3000,
    ]
    assert reward["path_progress_step_cap"] == 0.10
    assert reward["path_progress_hard_corridor"] is True
    assert reward["cross_track_error_cap"] == 1.50
    assert reward["reward_scale"] == 0.10
    assert config["rl"]["sac"]["critic_distribution"] == "quantile"
    assert config["rl"]["sac"]["actor_group_robust_enabled"] is True


def test_l261_scene_overlays_bound_projection_and_preserve_geometry():
    paths = sorted((ROOT / "configs/research/l261_value_stability").rglob("*.yaml"))
    assert len(paths) == 9
    for path in paths:
        config = load_yaml(path)
        task = config["task"]
        assert task["projection_backtrack_distance"] == 0.15
        assert task["projection_forward_distance"] == 0.25
        assert task["completion_corridor"] == 0.75
        assert len(task["points"]) >= 2
        assert "obstacles" in config["scene"]
        assert "initial_state" in config["experiment"]
