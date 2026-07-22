import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l269_long_horizon_credit_probe.yaml"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l269"]


def test_l269_inputs_are_hash_pinned_and_final_maps_forbidden():
    config = _config()
    assert _sha256(ROOT / config["source_checkpoint"]) == (
        config["source_checkpoint_sha256"]
    )
    assert _sha256(ROOT / config["l263"]["directory"] / "state_manifest.json") == (
        config["l263"]["state_manifest_sha256"]
    )
    assert _sha256(ROOT / config["l268"]["dataset"] / "manifest.json") == (
        config["l268"]["dataset_manifest_sha256"]
    )
    assert _sha256(
        ROOT / config["l268"]["heldout_diagnostic"] / "summary.json"
    ) == config["l268"]["heldout_summary_sha256"]
    references = "\n".join((
        config["source_checkpoint"],
        config["l263"]["directory"],
        config["l268"]["dataset"],
        config["l268"]["heldout_diagnostic"],
    )).lower()
    assert not any(token.lower() in references for token in config["forbidden_tokens"])


def test_l269_horizon_gate_is_fail_closed():
    config = _config()
    diagnostic = config["horizon_diagnostic"]
    assert diagnostic["horizons"] == [1, 5, 10, 20, 40]
    assert diagnostic["candidate_continuation"] == "actor_follow"
    assert diagnostic["gate"]["minimum_h40_recovery_advantage_fraction"] == 0.60
    assert diagnostic["gate"]["minimum_late_first_positive_fraction"] == 0.20
    assert config["critic_probe"]["enabled_only_after_horizon_gate"] is True


def test_l269_probe_is_paired_and_actor_forbidden():
    config = _config()
    probe = config["critic_probe"]
    assert [row["name"] for row in probe["arms"]] == [
        "one_step", "five_step", "ten_step", "td_lambda"
    ]
    assert len(set(probe["seeds"])) == 3
    assert probe["updates"] == 6000
    assert probe["replay"]["total_start_transitions"] == 6000
    assert probe["replay"]["transitions_per_scene"] == 1000
    assert probe["selection_order"] == ["five_step", "ten_step", "td_lambda"]
    assert config["conditional_actor_training"] is False
