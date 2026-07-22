import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l275_recovery_commitment_continuation_diagnosis.yaml"
PROTOCOL = ROOT / "docs/experiments/post_l217/l275_recovery_commitment_continuation_diagnosis_protocol.md"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l275"]


def test_l275_is_hash_pinned_diagnostic_only():
    config = _config()
    assert config["protocol"] == "L275"
    assert config["diagnostic_only"] is True
    assert config["actor_training_authorized"] is False
    for path_key, hash_key in (
        ("source_checkpoint", "source_checkpoint_sha256"),
        ("l268_dataset", "l268_dataset_manifest_sha256"),
        ("l268_heldout_states", "l268_heldout_states_sha256"),
        ("l269_summary", "l269_summary_sha256"),
        ("l274_summary", "l274_summary_sha256"),
    ):
        path = ROOT / config[path_key]
        if path.is_dir():
            path = path / "manifest.json"
        assert _sha256(path) == config[hash_key]


def test_l275_changes_only_recovery_commitment():
    config = _config()
    assert config["commitments"] == [1, 5, 10, 20, 40, "full_chain"]
    assert config["evaluation_horizon"] == "accepted_chain_length"
    assert config["intervention_continuation"] == "recorded_recovery_then_source_actor"
    assert config["baseline_continuation"] == "source_actor"
    assert config["expected_states"] == 36
    assert config["expected_scenes"] == 6


def test_l275_gate_is_fail_closed_and_forbids_final_data():
    config = _config()
    gate = config["gate"]
    assert gate["minimum_full_chain_positive_fraction"] == 0.75
    assert gate["minimum_positive_fraction_gain_full_vs_one"] == 0.20
    assert gate["minimum_late_first_positive_fraction"] == 0.25
    searchable = str({k: v for k, v in config.items() if k != "forbidden_tokens"}).lower()
    assert not any(token.lower() in searchable for token in config["forbidden_tokens"])
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "does not authorize Actor training" in protocol

