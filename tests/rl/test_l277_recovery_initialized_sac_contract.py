from pathlib import Path

import yaml

from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l277_protocol_freezes_small_budget_and_gate():
    text = (ROOT / "docs/experiments/post_l217/l277_recovery_initialized_small_budget_sac_protocol.md").read_text(encoding="utf-8")
    assert "exactly 6,000" in text
    assert "Actor updates remain frozen through step 3,000" in text
    assert "final Hairpin/S-Chicane/Infinity" in text
    assert "Ordinary resume is forbidden" in text


def test_l277_training_contract_inherits_frozen_l262_method():
    config = load_yaml(ROOT / "configs/rl/l277_recovery_initialized_sac_6k.yaml")
    training = config["rl"]["training"]
    assert training["total_steps"] == 6000
    assert training["actor_update_after"] == 3000
    assert training["replay_require_all_scenes"] is True
    assert training["evaluation_interval"] == 3000
    assert training["checkpoint_interval"] == 3000
    assert training["bc_anchor"]["enabled"] is False
    assert len(training["scene_configs"]) == 6
    assert len(training["validation_scene_configs"]) == 3


def test_l277_paired_seed_and_initialization_grid_is_fixed():
    gate = yaml.safe_load((ROOT / "configs/rl/l277_recovery_initialized_sac_gate.yaml").read_text(encoding="utf-8"))
    assert [row["seed"] for row in gate["runs"]] == [20263311, 20263312, 20263313]
    assert [row["validation_seed_base"] for row in gate["runs"]] == [20264311, 20264312, 20264313]
    assert len({row["initialization"] for row in gate["runs"]}) == 3
    assert all(len(row["initialization_sha256"]) == 64 for row in gate["runs"])
    assert gate["training"]["expected_validation_steps"] == [0, 3000, 6000]
    assert gate["training"]["full_agent_initialization"] is True
    assert gate["training"]["import_replay"] is False
    assert gate["training"]["reset_counters"] is True


def test_l277_gate_is_fail_closed_and_does_not_select_checkpoints():
    gate = yaml.safe_load((ROOT / "configs/rl/l277_recovery_initialized_sac_gate.yaml").read_text(encoding="utf-8"))
    frozen = gate["gate"]
    assert frozen["maximum_per_seed_mean_completion_regression"] == 0.02
    assert frozen["minimum_seeds_with_cte_or_goal_improvement"] == 2
    assert frozen["minimum_scenes_with_cte_improvement"] == 2
    assert frozen["maximum_collision_increase_per_seed"] == 0
    assert frozen["maximum_success_loss_per_seed"] == 0

