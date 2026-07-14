from pathlib import Path

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
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
