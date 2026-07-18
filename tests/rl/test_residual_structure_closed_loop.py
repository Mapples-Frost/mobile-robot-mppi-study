from pathlib import Path

from experiments.rl.run_cross_layer_factorial import _protected_previous_seeds
from experiments.rl.summarize_residual_structure_closed_loop import (
    CONDITIONS,
    CONTRASTS,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/residual_structure_closed_loop_l61.yaml"
CONFIRMATION_CONFIG = ROOT / "configs/rl/residual_structure_confirmation_l62.yaml"


def test_l61_is_complete_parameter_matched_three_way_design():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    assert tuple(design["conditions"]) == CONDITIONS
    assert set(CONTRASTS) == {
        "mlp_vs_nominal", "icode_vs_nominal", "icode_vs_mlp"
    }
    assert len(design["model_blocks"]) == 3
    assert len(design["scenes"]) == 4
    assert len(design["physics_domains"]) == 1
    assert len(design["development_episode_seeds"]) == 5
    assert 3 * 4 * 1 * 5 * 3 == 180
    for block in design["model_blocks"]:
        assert block["mlp_seed"] == block["icode_seed"]
        assert "l60_mlp" in block["mlp_checkpoint"]
        assert "l57_icode" in block["icode_checkpoint"]


def test_l61_development_and_sealed_seeds_are_fresh():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    development = set(design["development_episode_seeds"])
    sealed = set(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert development.isdisjoint(sealed)
    assert development.isdisjoint(protected)
    assert sealed.isdisjoint(protected)


def test_l61_keeps_rl_and_memory_out_of_structure_ablation():
    config = load_yaml(CONFIG)
    assert config["memory"]["enable"] is False
    assert config["rl"]["enabled"] is False
    assert config["planner"]["sampling_prior"] == "goal_warm_start"


def test_l62_unlocks_exactly_the_predeclared_l61_confirmation_seeds():
    development = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    confirmation = load_yaml(CONFIRMATION_CONFIG)["rl"]["cross_layer_factorial"]
    selected = set(confirmation["development_episode_seeds"])
    sealed = set(confirmation["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(confirmation["protected_config_paths"])
    assert confirmation["confirmation_mode"] is True
    assert selected == sealed == set(development["sealed_confirmation_episode_seeds"])
    assert selected.isdisjoint(protected)
    assert set(development["development_episode_seeds"]).issubset(protected)
