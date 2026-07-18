from pathlib import Path

import pytest

from experiments.rl.run_cross_layer_factorial import (
    _condition_config,
    _domain_entries,
    _scene_entries,
)
from experiments.rl.summarize_static_rl_contribution import CONTRASTS
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/static_rl_contribution_l69.yaml"


def test_l69_is_a_complete_attribution_design_with_fresh_seeds():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == [
        "traditional_icode",
        "frozen_bc_icode",
        "complexity_bc_icode",
        "gated_lcb_icode",
    ]
    assert len(design["model_blocks"]) == 3
    assert len(design["development_episode_seeds"]) == 5
    assert len(design["sealed_confirmation_episode_seeds"]) == 10
    assert not set(design["development_episode_seeds"]).intersection(
        design["sealed_confirmation_episode_seeds"]
    )
    assert set(CONTRASTS) == {
        "rl_vs_gated_bc",
        "rl_vs_traditional",
        "gated_bc_vs_traditional",
        "frozen_bc_vs_traditional",
    }


@pytest.mark.parametrize(
    "condition,mode,correction_mode",
    [
        ("frozen_bc_icode", "none", "base"),
        ("complexity_bc_icode", "complexity", "base"),
        ("gated_lcb_icode", "complexity", "lcb"),
    ],
)
def test_l69_learned_conditions_change_only_declared_gate_authority(
    condition, mode, correction_mode
):
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    resolved = _condition_config(
        config,
        _scene_entries(design)[1]["path"],
        _domain_entries(design)[0],
        condition,
        ROOT / "fake_rl.pt",
        ROOT / "fake_icode.pt",
        21860871,
    )
    assert resolved["planner"]["prediction_mode"] == "icode_residual"
    assert resolved["planner"]["sampling_prior"] == "rl"
    assert resolved["rl"]["gate"]["mode"] == mode
    assert resolved["rl"]["gate"][
        "correction_advantage_gate_mode"
    ] == correction_mode
    assert resolved["memory"]["enable"] is False
    assert resolved["sensors"]["pose_source"] == "ground_truth"
    assert resolved["plant"]["robot"]["chassis_mass"] == pytest.approx(13.0)
    assert resolved["planner"]["num_samples"] == 100

