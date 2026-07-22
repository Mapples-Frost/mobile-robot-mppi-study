from pathlib import Path

import yaml


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
