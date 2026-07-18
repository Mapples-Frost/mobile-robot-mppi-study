import json
from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.train_rl_sampling_prior import (
    _scene_configs,
    _write_config_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "rl" / "sac_mppi_dynamic_history_l33.yaml"


def test_dynamic_history_training_uses_disjoint_motion_paths_and_three_frames():
    config = load_yaml(CONFIG)
    training = config["rl"]["training"]
    assert config["rl"]["observation"]["history_frames"] == 3
    assert set(training["scene_configs"]).isdisjoint(
        training["validation_scene_configs"]
    )
    assert training["validation_domain_roles"] == ["unseen"]


def test_training_cli_applies_only_explicit_temporal_perception_overlay():
    config = load_yaml(CONFIG)
    scene = _scene_configs(
        config, config["rl"]["training"]["scene_configs"][:1]
    )[0]
    temporal = scene["perception"]["temporal_scan_guard"]
    assert temporal["enabled"] is True
    assert temporal["safety_enabled"] is True
    assert temporal["maximum_absolute_rate_mps"] == 1.5
    assert scene["perception"]["scan_guard"]["hard_stop_distance"] == 0.35


def test_every_training_motion_family_declares_seeded_randomization():
    config = load_yaml(CONFIG)
    training = config["rl"]["training"]
    for relative in training["scene_configs"] + training["validation_scene_configs"]:
        scene = load_yaml(ROOT / relative)
        motion = scene["scene"]["obstacles"][0]["motion"]
        assert motion["phase_jitter_s"] > 0.0
        assert motion["period_scale_range"][0] < motion["period_scale_range"][1]
        assert motion["endpoint_jitter_m"] > 0.0


def test_training_cli_writes_resolved_config_and_environment_snapshot(tmp_path):
    config = load_yaml(CONFIG)
    training = _scene_configs(config, config["rl"]["training"]["scene_configs"][:1])
    validation = _scene_configs(
        config, config["rl"]["training"]["validation_scene_configs"][:1]
    )
    target = _write_config_snapshot(
        tmp_path,
        CONFIG,
        config,
        training,
        validation,
        cli_overrides={"seed": 7},
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["resolved_config"]["rl"]["observation"]["history_frames"] == 3
    assert len(payload["training_environments"]) == 1
    assert len(payload["validation_environments"]) == 1
    assert payload["cli_overrides"]["seed"] == 7
