from pathlib import Path

import pytest

from experiments.rl.run_cross_layer_factorial import (
    _condition_config,
    _domain_entries,
    _protected_previous_seeds,
    _scene_entries,
)
from experiments.rl.summarize_correction_support_gate import (
    _aggregate,
    _effect_rows,
    _oracle_summary,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/correction_support_gate_l76.yaml"


def _episode(condition, success, distance, support, effective):
    return {
        "model_block": 0,
        "scene": "narrow_corridor",
        "episode_seed": 7,
        "condition": condition,
        "success": int(success),
        "collision": 0,
        "final_goal_distance": float(distance),
        "minimum_clearance": 0.25,
        "planner_compute_ms_mean": 10.0,
        "rl_gate_alpha_mean": 0.5,
        "rl_correction_support_confidence_mean": float(support),
        "rl_correction_effective_gate_alpha_mean": float(effective),
        "rl_correction_advantage_gate_alpha_mean": 0.5,
        "rl_applied_correction_abs_mean": 0.01,
        "rl_ood_score_mean": 3.2,
    }


def test_l76_is_fresh_complete_and_uses_locked_thresholds():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == [
        "complexity_bc_icode",
        "gated_lcb_icode",
        "support_gated_lcb_icode",
    ]
    assert 3 * 3 * 1 * 5 * 3 == 135
    assert len(design["model_blocks"]) == 3
    assert len(design["development_episode_seeds"]) == 5
    selected = set(design["development_episode_seeds"])
    selected.update(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert not selected.intersection(protected)
    gate = config["rl"]["gate"]
    assert gate["correction_support_soft_threshold"] == 3.0
    assert gate["correction_support_hard_threshold"] == 7.0
    for block in design["model_blocks"]:
        checkpoints = block["condition_checkpoints"]
        assert checkpoints["complexity_bc_icode"].endswith("initial.pt")
        assert checkpoints["gated_lcb_icode"] == checkpoints[
            "support_gated_lcb_icode"
        ]


def test_l76_support_condition_changes_only_correction_support_authority():
    base = load_yaml(CONFIG)
    design = base["rl"]["cross_layer_factorial"]
    scene = _scene_entries(design)[1]["path"]
    domain = _domain_entries(design)[0]
    always = _condition_config(
        base,
        scene,
        domain,
        "gated_lcb_icode",
        ROOT / "trained.pt",
        ROOT / "icode.pt",
        7,
    )
    support = _condition_config(
        base,
        scene,
        domain,
        "support_gated_lcb_icode",
        ROOT / "trained.pt",
        ROOT / "icode.pt",
        7,
    )
    assert always["planner"] == support["planner"]
    assert always["plant"] == support["plant"]
    assert always["sensors"] == support["sensors"]
    assert always["perception"] == support["perception"]
    assert always["rl"]["checkpoint"] == support["rl"]["checkpoint"]
    assert always["rl"]["gate"]["mode"] == "complexity"
    assert support["rl"]["gate"]["mode"] == "complexity"
    assert always["rl"]["gate"]["correction_support_gate_enabled"] is False
    assert support["rl"]["gate"]["correction_support_gate_enabled"] is True
    assert support["rl"]["gate"][
        "correction_support_soft_threshold"
    ] == 3.0
    assert support["rl"]["gate"][
        "correction_support_hard_threshold"
    ] == 7.0


def test_l76_effect_statistics_pair_by_episode_not_control_step():
    episodes = [
        _episode("complexity_bc_icode", False, 0.40, 1.0, 0.0),
        _episode("gated_lcb_icode", True, 0.15, 1.0, 0.5),
        _episode("support_gated_lcb_icode", True, 0.10, 0.6, 0.3),
    ]
    effects = _effect_rows(
        episodes, "complexity_bc_icode", "support_gated_lcb_icode"
    )
    assert len(effects) == 1
    assert effects[0]["success_gain"] == 1
    assert effects[0]["goal_distance_improvement_m"] == pytest.approx(0.30)
    summary = _aggregate(effects, "test")
    assert summary["net_success_gain"] == 1
    assert summary["mean_support_confidence"] == pytest.approx(0.6)
    assert summary["mean_effective_correction_alpha"] == pytest.approx(0.3)
    oracle, selected = _oracle_summary(
        episodes,
        [
            "complexity_bc_icode",
            "gated_lcb_icode",
            "support_gated_lcb_icode",
        ],
    )
    assert oracle["successes"] == 1
    assert oracle["selection_counts"]["support_gated_lcb_icode"] == 1
    assert selected[0]["final_goal_distance"] == pytest.approx(0.10)
