from pathlib import Path

from experiments.rl.run_cross_layer_factorial import _protected_previous_seeds
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/cross_plant_nominal_calibration_l63.yaml"


def test_l63_calibration_is_nominal_only_and_complete():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == ["traditional_nominal"]
    assert config["rl"]["enabled"] is False
    assert config["memory"]["enable"] is False
    assert len(design["scenes"]) == 2
    assert len(design["physics_domains"]) == 10
    assert len(design["development_episode_seeds"]) == 3
    assert 2 * 10 * 3 == 60


def test_l63_calibration_covers_each_predeclared_physics_group():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    calibration = config["cross_plant_calibration"]
    domains = {item["name"]: item for item in design["physics_domains"]}
    assert set(calibration["groups"]) == {
        "mass", "friction", "torque", "delay", "combined"
    }
    for candidates in calibration["groups"].values():
        assert candidates
        assert set(candidates).issubset(domains)
    assert domains[calibration["anchor_domain"]]["plant_override"] == {}


def test_l63_calibration_seeds_are_fresh_and_confirmation_stays_sealed():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    development = set(design["development_episode_seeds"])
    sealed = set(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert development.isdisjoint(sealed)
    assert development.isdisjoint(protected)
    assert sealed.isdisjoint(protected)


def test_l63_v2_retains_model_blind_design_and_group_specific_delay_floor():
    config = load_yaml(
        ROOT / "configs/rl/cross_plant_nominal_calibration_l63_v2.yaml"
    )
    design = config["rl"]["cross_layer_factorial"]
    gate = config["cross_plant_calibration"]

    assert design["conditions"] == ["traditional_nominal"]
    assert len(design["physics_domains"]) == 8
    assert gate["minimum_relative_cross_track_shift"] == 0.03
    assert gate["minimum_relative_cross_track_shift_by_group"] == {"delay": 0.01}
    assert set(design["development_episode_seeds"]) == {
        21860804, 21860805, 21860806
    }
    assert set(design["development_episode_seeds"]).isdisjoint(
        set(load_yaml(CONFIG)["rl"]["cross_layer_factorial"]["development_episode_seeds"])
    )


def test_l63_v3_freezes_single_factors_and_only_recalibrates_combined_domain():
    v2 = load_yaml(ROOT / "configs/rl/cross_plant_nominal_calibration_l63_v2.yaml")
    v3 = load_yaml(ROOT / "configs/rl/cross_plant_nominal_calibration_l63_v3.yaml")
    v2_domains = {
        item["name"]: item["plant_override"]
        for item in v2["rl"]["cross_layer_factorial"]["physics_domains"]
    }
    design = v3["rl"]["cross_layer_factorial"]
    v3_domains = {item["name"]: item["plant_override"] for item in design["physics_domains"]}

    for name in ("train_anchor", "mass_light", "friction_high", "torque_weak_v2", "delay_long"):
        assert v3_domains[name] == v2_domains[name]
    assert len(design["physics_domains"]) == 7
    assert set(design["development_episode_seeds"]) == {21860807, 21860808, 21860809}
    assert set(v3["cross_plant_calibration"]["groups"]["combined"]) == {
        "combined_moderate_a", "combined_moderate_b"
    }
