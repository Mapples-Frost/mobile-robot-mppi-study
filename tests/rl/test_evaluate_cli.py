from pathlib import Path

import pytest

from experiments.rl.evaluate_rl_sampling_prior import (
    _apply_physics_domain,
    _apply_scene_config,
    _require_explicit_scene_for_training_config,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_explicit_evaluation_scene_preserves_rl_setup_and_replaces_geometry():
    training = load_yaml(ROOT / "configs/rl/sac_mppi_utrap_bc_l13.yaml")
    result = _apply_scene_config(
        training, ROOT / "configs/research/mujoco_u_trap_long_board.yaml"
    )

    assert result["scene"]["name"] == "u_trap_long_board"
    assert result["rl"]["policy_id"] == "sac_mppi_utrap_bc_l13_v1"
    assert result["planner"]["sampling_prior"] == "rl"
    assert result["memory"]["enable"] is False


def test_omitted_evaluation_scene_preserves_legacy_behavior():
    base = {"scene": {"name": "legacy"}}
    assert _apply_scene_config(base, None) is base


def test_training_config_fails_closed_when_evaluation_scene_is_implicit():
    training = load_yaml(ROOT / "configs/rl/sac_mppi_utrap_bc_l13.yaml")
    with pytest.raises(ValueError, match="--scene-config"):
        _require_explicit_scene_for_training_config(training, None)
    _require_explicit_scene_for_training_config(
        training, ROOT / "configs/research/mujoco_u_trap_long_board.yaml"
    )
    _require_explicit_scene_for_training_config(
        training, None, allow_embedded_scene=True
    )


def test_scene_config_only_training_marker_also_fails_closed():
    training = {
        "rl": {"training": {"scene_configs": ["scene.yaml"]}},
        "scene": {"name": "embedded"},
    }
    with pytest.raises(ValueError, match="--scene-config"):
        _require_explicit_scene_for_training_config(training, None)


def test_dynamic_scene_applies_only_explicit_training_perception_overlay():
    training = load_yaml(
        ROOT / "configs/rl/sac_mppi_dynamic_history_bounded_l34.yaml"
    )
    result = _apply_scene_config(
        training,
        ROOT / "configs/research/mujoco_dynamic_validation_diagonal_l33.yaml",
    )
    assert result["perception"]["temporal_scan_guard"]["safety_enabled"] is True
    assert result["perception"]["scan_guard"]["hard_stop_distance"] == 0.35


def test_explicit_physics_domain_applies_true_plant_override_and_provenance():
    base = load_yaml(
        ROOT / "configs/research/mujoco_dynamic_validation_diagonal_l33.yaml"
    )
    result = _apply_physics_domain(
        base,
        ROOT / "configs/research/mujoco_physics_domains.yaml",
        "combined_unseen",
    )
    assert result["scene"]["name"].endswith("__combined_unseen")
    assert result["experiment"]["physics_domain"] == "combined_unseen"
    assert result["experiment"]["physics_domain_role"] == "unseen"
    assert result["plant"] != base["plant"]


def test_physics_domain_selection_fails_closed_on_partial_arguments():
    with pytest.raises(ValueError, match="provided together"):
        _apply_physics_domain({}, "domains.yaml", None)
