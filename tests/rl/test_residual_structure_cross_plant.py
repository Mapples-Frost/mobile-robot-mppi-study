from pathlib import Path

from experiments.rl.run_cross_layer_factorial import _protected_previous_seeds
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/residual_structure_cross_plant_l64.yaml"
CONFIRMATION_CONFIG = ROOT / "configs/rl/residual_structure_cross_plant_confirmation_l65.yaml"


def test_l64_is_parameter_matched_cross_plant_development_design():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == [
        "traditional_nominal", "traditional_mlp", "traditional_icode"
    ]
    assert config["rl"]["enabled"] is False
    assert config["memory"]["enable"] is False
    assert design["confirmation_mode"] is False
    assert design["planner_known_command_delay"] is True
    assert len(design["model_blocks"]) == 3
    assert all(block["mlp_seed"] == block["icode_seed"] for block in design["model_blocks"])


def test_l64_uses_six_frozen_domains_and_two_paths():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    assert [item["name"] for item in design["physics_domains"]] == [
        "train_anchor", "mass_light", "friction_high", "torque_weak_v2",
        "delay_long", "combined_moderate_b",
    ]
    assert len(design["scenes"]) == 2
    assert len(design["development_episode_seeds"]) == 3
    assert 3 * 2 * 6 * 3 * 3 == 324


def test_l64_development_seeds_do_not_touch_protected_or_sealed_seeds():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    development = set(design["development_episode_seeds"])
    sealed = set(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert development.isdisjoint(sealed)
    assert development.isdisjoint(protected)
    assert sealed.isdisjoint(protected)


def test_l64_gate_requires_cross_domain_unseen_and_combined_support():
    gate = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]["cross_plant_structure_gate"]
    assert gate["minimum_positive_icode_vs_mlp_shifted_domains"] == 4
    assert gate["minimum_positive_unseen_icode_vs_mlp_shifted_blocks"] == 2
    assert gate["minimum_positive_combined_icode_vs_mlp_blocks"] == 2
    assert gate["minimum_icode_vs_mlp_shifted_ci95_lower_m"] == 0.0


def test_l65_unlocks_only_the_predeclared_l64_confirmation_seeds():
    development = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    confirmation = load_yaml(CONFIRMATION_CONFIG)["rl"]["cross_layer_factorial"]
    selected = set(confirmation["development_episode_seeds"])
    sealed = set(confirmation["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(confirmation["protected_config_paths"])

    assert confirmation["confirmation_mode"] is True
    assert selected == sealed == set(development["sealed_confirmation_episode_seeds"])
    assert selected.isdisjoint(protected)
    assert set(development["development_episode_seeds"]).issubset(protected)
    assert 3 * 2 * 6 * 5 * 3 == 540

