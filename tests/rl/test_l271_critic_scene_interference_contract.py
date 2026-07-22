from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l271_is_read_only_and_hash_pinned():
    config = yaml.safe_load((
        ROOT / "configs/rl/l271_critic_scene_interference_diagnosis.yaml"
    ).read_text(encoding="utf-8"))["l271"]
    assert config["conditional_training"] is False
    assert len(config["l268"]["treatment_checkpoints"]) == 3
    assert all(len(row["sha256"]) == 64 for row in config["l268"]["treatment_checkpoints"])
    assert config["l270"]["required_decision"] == (
        "close_conflicts_but_privileged_probe_unresolved"
    )


def test_l271_balanced_null_and_gate_are_frozen():
    config = yaml.safe_load((
        ROOT / "configs/rl/l271_critic_scene_interference_diagnosis.yaml"
    ).read_text(encoding="utf-8"))["l271"]
    audit = config["audit"]
    assert audit["scene_count"] == 6
    assert audit["rows_per_scene"] == 128
    assert audit["null_partitions"] == 24
    assert "input_action_columns" in audit["layers"]
    assert config["gate"]["minimum_seed_blocks_both_extreme"] == 2
    assert config["next_if_pass"].startswith("preregister_")


def test_l271_forbids_l258_and_final_tracking_scenes():
    config = yaml.safe_load((
        ROOT / "configs/rl/l271_critic_scene_interference_diagnosis.yaml"
    ).read_text(encoding="utf-8"))["l271"]
    forbidden = set(config["forbidden_tokens"])
    assert {"l258_tracking_curriculum", "l258_heldout_tracking_gate"} <= forbidden
    assert {"hairpin_v1", "s_chicane_v1", "infinity_v1"} <= forbidden
