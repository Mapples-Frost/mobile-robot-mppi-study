from pathlib import Path

import pytest

from experiments.rl.run_cross_layer_factorial import (
    _condition_config,
    _domain_entries,
    _protected_previous_seeds,
    _scene_entries,
)
from experiments.rl.summarize_progress_competence_activation import (
    _aggregate,
    _effect_rows,
    _oracle_summary,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/progress_competence_activation_l75.yaml"


def _episode(condition, success, distance, alpha, stagnation=0.0):
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
        "rl_gate_alpha_mean": float(alpha),
        "rl_baseline_stagnation_activation_mean": float(stagnation),
        "rl_learned_inference_skip_fraction": float(alpha <= 0.0),
    }


def test_l75_is_a_fresh_paired_three_block_activation_design():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == [
        "complexity_bc_icode",
        "gated_lcb_icode",
        "progress_gated_lcb_icode",
    ]
    assert len(design["model_blocks"]) == 3
    assert len(design["scenes"]) == 3
    assert len(design["physics_domains"]) == 1
    assert len(design["development_episode_seeds"]) == 5
    assert len(design["sealed_confirmation_episode_seeds"]) == 5
    assert 3 * 3 * 1 * 5 * 3 == 135
    selected = set(design["development_episode_seeds"])
    selected.update(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert not selected.intersection(protected)
    for block in design["model_blocks"]:
        checkpoints = block["condition_checkpoints"]
        assert checkpoints["complexity_bc_icode"].endswith("initial.pt")
        assert checkpoints["gated_lcb_icode"].endswith(
            "step_000030000.pt"
        )
        assert checkpoints["progress_gated_lcb_icode"] == checkpoints[
            "gated_lcb_icode"
        ]


def test_l75_changes_only_online_rl_activation_between_sac_conditions():
    base = load_yaml(CONFIG)
    design = base["rl"]["cross_layer_factorial"]
    scene = _scene_entries(design)[1]["path"]
    domain = _domain_entries(design)[0]
    common = (
        base,
        scene,
        domain,
        None,
        ROOT / "fake_icode.pt",
        7,
    )
    always = _condition_config(
        common[0],
        common[1],
        common[2],
        "gated_lcb_icode",
        ROOT / "trained.pt",
        common[4],
        common[5],
    )
    progress = _condition_config(
        common[0],
        common[1],
        common[2],
        "progress_gated_lcb_icode",
        ROOT / "trained.pt",
        common[4],
        common[5],
    )
    assert always["planner"] == progress["planner"]
    assert always["plant"] == progress["plant"]
    assert always["sensors"] == progress["sensors"]
    assert always["perception"] == progress["perception"]
    assert always["memory"]["enable"] is False
    assert progress["memory"]["enable"] is False
    assert always["rl"]["checkpoint"] == progress["rl"]["checkpoint"]
    assert always["rl"]["gate"]["mode"] == "complexity"
    assert progress["rl"]["gate"]["mode"] == "progress_complexity"
    assert progress["rl"]["gate"]["progress"] == {
        "observation_window_s": 2.0,
        "minimum_window_coverage_s": 1.8,
        "full_activation_progress_m": 0.04,
        "zero_activation_progress_m": 0.20,
        "activation_hold_s": 1.0,
    }


def test_l75_effect_and_oracle_statistics_are_episode_paired():
    episodes = [
        _episode("complexity_bc_icode", False, 0.40, 0.0),
        _episode("gated_lcb_icode", True, 0.15, 0.6, 0.0),
        _episode(
            "progress_gated_lcb_icode",
            True,
            0.10,
            0.2,
            stagnation=0.5,
        ),
    ]
    effects = _effect_rows(
        episodes,
        "complexity_bc_icode",
        "progress_gated_lcb_icode",
    )
    assert len(effects) == 1
    assert effects[0]["success_gain"] == 1
    assert effects[0]["success_loss"] == 0
    assert effects[0]["goal_distance_improvement_m"] == pytest.approx(0.30)
    summary = _aggregate(effects, "test")
    assert summary["net_success_gain"] == 1
    assert summary["mean_stagnation_activation"] == pytest.approx(0.5)
    oracle, selected = _oracle_summary(
        episodes,
        [
            "complexity_bc_icode",
            "gated_lcb_icode",
            "progress_gated_lcb_icode",
        ],
    )
    assert oracle["successes"] == 1
    assert oracle["selection_counts"]["progress_gated_lcb_icode"] == 1
    assert selected[0]["final_goal_distance"] == pytest.approx(0.10)
