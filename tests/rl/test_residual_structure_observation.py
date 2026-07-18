from pathlib import Path

from experiments.rl.run_cross_layer_factorial import _protected_previous_seeds
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/residual_structure_observation_domains_l67.yaml"
CONFIRMATION_CONFIG = ROOT / "configs/rl/residual_structure_observation_confirmation_l68.yaml"


def test_l67_is_fixed_plant_parameter_matched_observation_design():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == [
        "traditional_nominal", "traditional_mlp", "traditional_icode"
    ]
    assert config["rl"]["enabled"] is False
    assert config["memory"]["enable"] is False
    assert len(design["model_blocks"]) == 3
    assert len(design["scenes"]) == 2
    assert len(design["physics_domains"]) == 4
    assert all(not item["plant_override"] for item in design["physics_domains"])
    assert all(block["mlp_seed"] == block["icode_seed"] for block in design["model_blocks"])
    assert 3 * 2 * 4 * 3 * 3 == 216


def test_l67_primary_and_stress_domains_are_explicitly_separated():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    gate = design["observation_structure_gate"]
    assert gate["primary_domains"] == [
        "clean_ground_truth", "latency_100ms", "combined_medium_100ms"
    ]
    assert gate["shifted_domains"] == ["latency_100ms", "combined_medium_100ms"]
    assert gate["stress_domain"] == "raw_wheel_odometry"
    assert gate["minimum_positive_icode_vs_mlp_shifted_domains"] == 2


def test_l67_seeds_are_fresh_and_confirmation_is_still_sealed():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    development = set(design["development_episode_seeds"])
    sealed = set(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert development.isdisjoint(sealed)
    assert development.isdisjoint(protected)
    assert sealed.isdisjoint(protected)


def test_l68_unlocks_only_predeclared_observation_confirmation_seeds():
    development = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    confirmation = load_yaml(CONFIRMATION_CONFIG)["rl"]["cross_layer_factorial"]
    selected = set(confirmation["development_episode_seeds"])
    sealed = set(confirmation["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(confirmation["protected_config_paths"])
    assert confirmation["confirmation_mode"] is True
    assert selected == sealed == set(development["sealed_confirmation_episode_seeds"])
    assert selected.isdisjoint(protected)
    assert set(development["development_episode_seeds"]).issubset(protected)
    assert 3 * 2 * 4 * 5 * 3 == 360

