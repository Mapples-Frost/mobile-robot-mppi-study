from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l262_delays_learning_until_six_scene_support_is_complete():
    config = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    training = config["rl"]["training"]

    assert training["seed"] == 20262611
    assert training["total_steps"] == 6000
    assert training["replay_require_all_scenes"] is True
    assert training["actor_update_after"] == 3000
    assert training["evaluation_interval"] == 3000
    assert training["bc_anchor"]["enabled"] is False

    phases = training["curriculum"]["phases"]
    assert len(phases) == 12
    assert [phase["until_step"] for phase in phases] == list(range(500, 6001, 500))
    assert all(sum(phase["scene_weights"]) == 1 for phase in phases)
    assert [phase["scene_weights"].index(1) for phase in phases] == list(range(6)) * 2


def test_l262_changes_no_l261_model_reward_or_map_contract():
    l261 = load_yaml(ROOT / "configs/rl/l261_value_stability_3k.yaml")
    l262 = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")

    for key in ("reward", "observation", "prior", "sac", "residual_context"):
        assert l262["rl"][key] == l261["rl"][key]
    for key in ("scene_configs", "validation_scene_configs", "initial_state_curriculum"):
        assert l262["rl"]["training"][key] == l261["rl"]["training"][key]
