from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l278_protocol_is_evaluation_only_and_leakage_safe():
    text = (ROOT / "docs/experiments/post_l217/l278_recovery_retention_diagnosis_protocol.md").read_text(encoding="utf-8")
    assert "evaluation-only" in text
    assert "performs no training" in text
    assert "final Hairpin/S-Chicane/Infinity" in text
    assert "step 3k and 6k" in text


def test_l278_checkpoint_grid_and_hashes_are_frozen():
    config = yaml.safe_load((ROOT / "configs/rl/l278_recovery_retention_diagnosis.yaml").read_text(encoding="utf-8"))
    assert [row["seed"] for row in config["checkpoints"]] == [20263311, 20263312, 20263313]
    assert config["split_counts"] == {"validation": 18, "test": 18}
    for row in config["checkpoints"]:
        assert set(row) == {"seed", "initial", "step3000", "step6000"}
        for name in ("initial", "step3000", "step6000"):
            assert len(row[name]["sha256"]) == 64


def test_l278_gate_thresholds_are_fixed_before_evaluation():
    config = yaml.safe_load((ROOT / "configs/rl/l278_recovery_retention_diagnosis.yaml").read_text(encoding="utf-8"))
    gate = config["gate"]
    assert gate["maximum_median_relative_test_rmse_increase"] == 0.20
    assert gate["maximum_median_test_return_loss"] == 0.50
    assert gate["maximum_median_test_reentry_loss"] == 0.10
    assert gate["minimum_scenes_with_nonnegative_test_return_change"] == 4
    assert gate["maximum_collision_boundary_failure_increase"] == 0

