from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l282_contract_is_evaluation_only_and_complete():
    config = yaml.safe_load(
        (ROOT / "configs/rl/l282_recovery_component_counterfactual.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["diagnostic_only"] is True
    assert config["actor_training_authorized"] is False
    assert config["split"] == "test"
    assert config["expected_chains"] == 18
    assert len(config["checkpoints"]) == 3
    assert config["arms"] == [
        "teacher_both",
        "actor_both",
        "teacher_v_actor_omega",
        "actor_v_teacher_omega",
    ]


def test_l282_protocol_freezes_coverage_attribution_and_forbidden_maps():
    text = (
        ROOT
        / "docs/experiments/post_l217/l282_recovery_component_counterfactual_protocol.md"
    ).read_text(encoding="utf-8").lower()
    assert "216 rollouts" in text
    assert "minimum" not in text or "50%" in text
    assert "final-map evaluation directly" in text
    assert "hairpin/s-chicane/infinity" in text
