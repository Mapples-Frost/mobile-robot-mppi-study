from pathlib import Path

from experiments.rl.run_cross_layer_factorial import (
    _domain_entries,
    _protected_previous_seeds,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/observation_domain_nominal_calibration_l66.yaml"
V2_CONFIG = ROOT / "configs/rl/observation_domain_nominal_calibration_l66_v2.yaml"


def test_l66_is_nominal_only_fixed_plant_observation_calibration():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == ["traditional_nominal"]
    assert config["rl"]["enabled"] is False
    assert config["memory"]["enable"] is False
    assert len(design["scenes"]) == 2
    assert len(design["physics_domains"]) == 8
    assert all(not item["plant_override"] for item in design["physics_domains"])
    assert 2 * 8 * 3 == 48


def test_l66_observation_domains_cover_noise_latency_combined_and_odom_stress():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    domains = {item["name"]: item for item in _domain_entries(design)}
    gate = config["observation_domain_calibration"]
    assert set(gate["groups"]) == {"noise", "latency", "combined"}
    assert domains["clean_ground_truth"]["sensor_override"] == {}
    assert domains["latency_100ms"]["sensor_override"]["latency"] == 0.10
    assert domains["raw_wheel_odometry"]["sensor_override"]["pose_source"] == "wheel_odometry"
    assert "raw_wheel_odometry" not in {
        name for candidates in gate["groups"].values() for name in candidates
    }


def test_l66_development_and_sealed_seeds_are_fresh():
    design = load_yaml(CONFIG)["rl"]["cross_layer_factorial"]
    development = set(design["development_episode_seeds"])
    sealed = set(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert development.isdisjoint(sealed)
    assert development.isdisjoint(protected)
    assert sealed.isdisjoint(protected)


def test_l66_v2_changes_only_noise_candidates_and_uses_fresh_seeds():
    v1 = load_yaml(CONFIG)
    v2 = load_yaml(V2_CONFIG)
    v1_domains = {
        item["name"]: item["sensor_override"]
        for item in v1["rl"]["cross_layer_factorial"]["physics_domains"]
    }
    design = v2["rl"]["cross_layer_factorial"]
    v2_domains = {item["name"]: item["sensor_override"] for item in design["physics_domains"]}
    for name in ("clean_ground_truth", "latency_100ms", "combined_medium_100ms", "raw_wheel_odometry"):
        assert v2_domains[name] == v1_domains[name]
    assert set(v2["observation_domain_calibration"]["groups"]["noise"]) == {
        "noise_high", "noise_very_high"
    }
    assert set(design["development_episode_seeds"]) == {21860844, 21860845, 21860846}
    assert set(design["development_episode_seeds"]).isdisjoint(
        v1["rl"]["cross_layer_factorial"]["development_episode_seeds"]
    )

