from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l280_is_evaluation_only_and_fully_paired():
    config = yaml.safe_load((ROOT / "configs/rl/l280_anchor_gradient_alignment_diagnosis.yaml").read_text())
    assert config["windows_native_only"] is True
    assert [row["seed"] for row in config["checkpoints"]] == [20263311, 20263312, 20263313]
    assert all(set(row) == {"seed", "step3000", "step6000"} for row in config["checkpoints"])
    assert config["partitions"] == ["trunk", "v_mean", "omega_mean", "log_std", "all"]


def test_l280_freezes_batches_and_conflict_decision():
    config = yaml.safe_load((ROOT / "configs/rl/l280_anchor_gradient_alignment_diagnosis.yaml").read_text())
    assert config["batch_size"] == 256
    assert config["diagnostic_seed"] == 20268001
    assert config["gate"]["minimum_conflicted_scenes"] == 4
    assert config["gate"]["minimum_conflicted_seeds"] == 2
    assert config["gate"]["minimum_dominance_norm_ratio"] == 4.0


def test_l280_protocol_does_not_authorize_training():
    text = (ROOT / "docs/experiments/post_l217/l280_anchor_gradient_alignment_diagnosis_protocol.md").read_text().lower()
    assert "performs no optimizer step" in text
    assert "does not authorize actor training" in text
    assert "final maps" in text
