import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l276_recovery_sequence_imitation_initialization.yaml"
PROTOCOL = ROOT / "docs/experiments/post_l217/l276_recovery_sequence_imitation_initialization_protocol.md"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l276"]


def test_l276_inputs_are_pinned_and_sac_is_fail_closed():
    config = _config()
    assert config["protocol"] == "L276"
    assert config["initialization_gate_only"] is True
    assert config["sac_training_authorized"] is False
    assert _sha256(ROOT / config["source_checkpoint"]) == config["source_checkpoint_sha256"]
    assert _sha256(ROOT / config["recovery_dataset"] / "manifest.json") == (
        config["recovery_dataset_manifest_sha256"]
    )
    assert _sha256(ROOT / config["l275_summary"]) == config["l275_summary_sha256"]


def test_l276_uses_frozen_chain_split_and_balanced_anchor():
    config = _config()
    assert config["split_counts"] == {"train": 72, "validation": 18, "test": 18}
    training = config["training"]
    assert len(set(training["seeds"])) == 3
    assert training["updates"] == 3000
    assert training["selected_checkpoint_update"] == 3000
    assert training["recovery_batch_size"] == training["source_anchor_batch_size"] == 128
    assert config["evaluation"]["test_after_all_seeds"] is True


def test_l276_gate_preserves_nominal_actor_and_nonactor_state():
    gate = _config()["gate"]
    assert gate["minimum_median_relative_test_rmse_improvement"] == 0.20
    assert gate["minimum_test_reentry_fraction_gain"] == 0.15
    assert gate["minimum_test_scenes_with_return_improvement"] == 4
    assert gate["maximum_source_replay_mean_absolute_action_drift"] == 0.10
    assert gate["require_frozen_nonactor_hashes"] is True
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "does not authorize SAC" in protocol


def test_l276_forbids_final_and_invalid_artifacts():
    config = _config()
    searchable = str({key: value for key, value in config.items() if key != "forbidden_tokens"}).lower()
    assert not any(token.lower() in searchable for token in config["forbidden_tokens"])

