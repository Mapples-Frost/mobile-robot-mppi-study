from pathlib import Path

import yaml

from experiments.rl.generate_l284_closed_loop_rollin_anchor import (
    _episode,
    _rollin,
)


ROOT = Path(__file__).resolve().parents[2]


def test_l284_rollin_dataset_is_train_only_and_exactly_balanced():
    config = yaml.safe_load(
        (ROOT / "configs/rl/l284_closed_loop_rollin_anchor_dataset.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["split_counts"] == {"train": 72, "validation": 18, "test": 18}
    assert sum(row["samples_per_chain"] for row in config["rollin_checkpoints"]) == 128
    assert config["train"] == {
        "recovery_samples": 9216,
        "source_samples": 9216,
        "total_samples": 18432,
    }


def test_l284_changes_only_anchor_dataset_over_l281_training_config():
    base = yaml.safe_load(
        (ROOT / "configs/rl/l284_closed_loop_rollin_joint_anchor_sac_6k.yaml")
        .read_text(encoding="utf-8")
    )
    assert base["include"] == "l281_component_separated_anchor_sac_6k.yaml"
    assert set(base["rl"]["training"]["bc_anchor"]) == {"dataset_dir"}


def test_l284_protocol_forbids_outcome_selection_and_final_maps():
    text = (
        ROOT / "docs/experiments/post_l217/"
        "l284_closed_loop_rollin_joint_anchor_protocol.md"
    ).read_text(encoding="utf-8").lower()
    assert "43/43/42" in text
    assert "never enter" in text
    assert "no seed, checkpoint, chain, threshold, or scene" in text
    assert "final hairpin/s-chicane/infinity" in text


def test_l284_generator_exports_rollin_and_schema_helpers():
    assert callable(_rollin)
    row = _episode([[0.0] * 69], [[0.2, -0.3]], 0, 7)
    assert row["observation"].shape == (1, 69)
    assert row["teacher_action"].shape == (1, 2)
    assert row["source_kind"].tolist() == [0]
