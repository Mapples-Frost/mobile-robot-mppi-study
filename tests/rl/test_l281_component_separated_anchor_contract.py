from pathlib import Path

import yaml

from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l281_is_the_frozen_component_separated_intervention():
    config = yaml.safe_load(
        (ROOT / "configs/rl/l281_component_separated_anchor_sac_6k.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["include"] == "l279_recovery_retention_anchor_sac_6k.yaml"
    anchor = config["rl"]["training"]["bc_anchor"]
    assert anchor == {
        "source_kind_balanced": True,
        "recovery_action_mean_weights": [1.0, 1.0],
        "source_action_mean_weights": [0.0, 1.0],
    }


def test_l281_protocol_freezes_pairing_and_forbids_final_maps():
    text = (
        ROOT
        / "docs/experiments/post_l217/l281_component_separated_anchor_protocol.md"
    ).read_text(encoding="utf-8").lower()
    assert "only change" in text
    assert "exactly 128 recovery and 128 source" in text
    assert "final hairpin/s-chicane/infinity" in text
    assert "never authorizes final-map evaluation directly" in text


def test_l281_resolved_config_preserves_l279_and_adds_only_component_routing():
    config = load_yaml(
        ROOT / "configs/rl/l281_component_separated_anchor_sac_6k.yaml"
    )
    anchor = config["rl"]["training"]["bc_anchor"]
    assert anchor["enabled"] is True
    assert anchor["dataset_format"] == "recovery_retention_v1"
    assert anchor["batch_size"] == 256
    assert anchor["mean_weight"] == 2.0
    assert anchor["log_std_weight"] == 0.001
    assert anchor["source_kind_balanced"] is True
    assert anchor["recovery_action_mean_weights"] == [1.0, 1.0]
    assert anchor["source_action_mean_weights"] == [0.0, 1.0]


def test_l281_gate_locks_paired_seeds_and_component_weights():
    config = yaml.safe_load(
        (ROOT / "configs/rl/l281_component_separated_anchor_gate.yaml")
        .read_text(encoding="utf-8")
    )
    assert [row["seed"] for row in config["runs"]] == [
        20263311, 20263312, 20263313
    ]
    assert config["training"]["source_kind_balanced"] is True
    assert config["training"]["recovery_action_mean_weights"] == [1.0, 1.0]
    assert config["training"]["source_action_mean_weights"] == [0.0, 1.0]
    assert config["gate"]["minimum_scenes_with_cte_improvement"] == 2
