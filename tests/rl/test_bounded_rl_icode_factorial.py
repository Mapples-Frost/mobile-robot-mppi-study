from pathlib import Path

import pytest

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.run_cross_layer_factorial import (
    _condition_config,
    _domain_entries,
    _protected_previous_seeds,
    _scene_entries,
)
from experiments.rl.summarize_bounded_rl_icode_factorial import _paired_effects


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "rl" / "bounded_rl_icode_factorial_l37.yaml"


@pytest.mark.parametrize(
    "condition,prediction",
    (
        ("bounded_rl_nominal", "nominal"),
        ("bounded_rl_icode", "icode_residual"),
    ),
)
def test_l37_bounded_rl_is_fixed_quarter_authority_with_shared_safety(
    condition, prediction
):
    base = load_yaml(CONFIG)
    design = base["rl"]["cross_layer_factorial"]
    config = _condition_config(
        base,
        _scene_entries(design)[0]["path"],
        _domain_entries(design)[0],
        condition,
        ROOT / "fake_rl.pt",
        ROOT / "fake_icode.pt",
        7,
    )
    assert config["planner"]["sampling_prior"] == "rl"
    assert config["planner"]["prediction_mode"] == prediction
    assert config["rl"]["gate"]["mode"] == "fixed"
    assert config["rl"]["gate"]["fixed_alpha"] == pytest.approx(0.25)
    assert config["rl"]["gate"]["correction_advantage_gate_mode"] == "none"
    assert config["perception"]["temporal_scan_guard"]["enabled"] is True
    assert config["perception"]["temporal_scan_guard"]["safety_enabled"] is True
    assert config["memory"]["enable"] is False


def test_l37_uses_fresh_development_and_sealed_seeds():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    selected = set(design["development_episode_seeds"])
    selected.update(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert not selected.intersection(protected)
    assert len(design["model_blocks"]) == 3
    assert len(design["scenes"]) == 3
    assert len(design["conditions"]) == 4
    assert 3 * 3 * 1 * 5 * 4 == 180


def test_l37_interaction_is_difference_in_differences():
    outcomes = {
        "traditional_nominal": (0, 1, 2.0),
        "traditional_icode": (1, 0, 1.5),
        "bounded_rl_nominal": (1, 0, 1.2),
        "bounded_rl_icode": (1, 0, 0.9),
    }
    rows = []
    for condition, (success, collision, distance) in outcomes.items():
        rows.append({
            "model_block": 0,
            "scene": "s",
            "physics_domain": "p",
            "episode_seed": 1,
            "condition": condition,
            "success": success,
            "collision": collision,
            "final_goal_distance": distance,
        })
    interaction = next(
        row for row in _paired_effects(rows) if row["effect"] == "interaction"
    )
    assert interaction["success_difference"] == -1
    assert interaction["collision_difference"] == 1
    assert interaction["goal_distance_improvement_m"] == pytest.approx(-0.2)

