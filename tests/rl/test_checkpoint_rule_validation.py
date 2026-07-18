from pathlib import Path

from experiments.rl.run_cross_layer_factorial import (
    _condition_config,
    _domain_entries,
    _protected_previous_seeds,
    _scene_entries,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/training_internal_checkpoint_rule_l77.yaml"


def test_l77_uses_preexisting_l72_checkpoint_choices_and_fresh_seeds():
    config = load_yaml(CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == [
        "complexity_bc_icode",
        "gated_lcb_icode",
    ]
    assert [row["selected_step"] for row in design["model_blocks"]] == [
        25000,
        25000,
        30000,
    ]
    assert 3 * 3 * 1 * 5 * 2 == 90
    selected = set(design["development_episode_seeds"])
    selected.update(design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert not selected.intersection(protected)
    for row in design["model_blocks"]:
        checkpoint = row["condition_checkpoints"]["gated_lcb_icode"]
        assert checkpoint.endswith(
            "step_%09d.pt" % int(row["selected_step"])
        )
        assert Path(checkpoint).exists()


def test_l77_candidate_has_no_progress_or_support_deployment_gate():
    base = load_yaml(CONFIG)
    design = base["rl"]["cross_layer_factorial"]
    row = design["model_blocks"][0]
    config = _condition_config(
        base,
        _scene_entries(design)[0]["path"],
        _domain_entries(design)[0],
        "gated_lcb_icode",
        ROOT / row["condition_checkpoints"]["gated_lcb_icode"],
        ROOT / row["icode_checkpoint"],
        design["development_episode_seeds"][0],
    )
    gate = config["rl"]["gate"]
    assert gate["mode"] == "complexity"
    assert gate["correction_advantage_gate_mode"] == "lcb"
    assert gate["correction_support_gate_enabled"] is False
