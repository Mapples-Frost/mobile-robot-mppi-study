from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l279_is_paired_single_variable_anchor_intervention():
    base = yaml.safe_load((ROOT / "configs/rl/l279_recovery_retention_anchor_sac_6k.yaml").read_text())
    anchor = base["rl"]["training"]["bc_anchor"]
    assert base["include"] == "l277_recovery_initialized_sac_6k.yaml"
    assert anchor == {
        "enabled": True,
        "dataset_dir": "results/research_platform/rl/l279_recovery_retention_anchor_dataset",
        "dataset_format": "recovery_retention_v1",
        "batch_size": 256,
        "mean_weight": 2.0,
        "log_std_weight": 0.001,
        "target_log_std": -2.0,
    }


def test_l279_dataset_is_balanced_and_train_split_only():
    config = yaml.safe_load((ROOT / "configs/rl/l279_recovery_retention_anchor_dataset.yaml").read_text())
    assert config["split_counts"] == {"train": 72, "validation": 18, "test": 18}
    assert config["train"]["recovery_samples"] == config["train"]["source_samples"] == 9216
    assert config["train"]["total_samples"] == 18432
    assert "source_kind" in config["student_fields"]
    assert "return" not in config["student_fields"]
    assert "privileged" not in config["student_fields"]


def test_l279_gate_locks_pairing_and_retention_thresholds():
    config = yaml.safe_load((ROOT / "configs/rl/l279_recovery_retention_anchor_gate.yaml").read_text())
    assert [row["seed"] for row in config["runs"]] == [20263311, 20263312, 20263313]
    assert config["training"]["expected_anchor_update_steps"] == 3001
    assert config["gate"]["maximum_relative_test_rmse_increase"] == 0.20
    assert config["gate"]["minimum_relative_test_rmse_improvement_vs_l277"] == 0.20
    assert config["gate"]["minimum_scenes_with_cte_improvement"] == 2


def test_l279_protocol_forbids_final_maps_and_selection():
    text = (ROOT / "docs/experiments/post_l217/l279_recovery_retention_anchor_sac_protocol.md").read_text()
    lowered = text.lower()
    assert "final hairpin/s-chicane/infinity" in lowered
    assert "cannot be selected" in lowered
    assert "never authorizes" in lowered
