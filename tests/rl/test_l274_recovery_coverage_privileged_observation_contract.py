from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l274_recovery_coverage_privileged_observation_diagnosis.yaml"
PROTOCOL = ROOT / "docs/experiments/post_l217/l274_recovery_coverage_privileged_observation_diagnosis_protocol.md"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l274"]


def test_l274_is_diagnostic_only_and_cross_fitted_by_state():
    config = _config()
    assert config["protocol"] == "L274"
    assert config["actor_training_authorized"] is False
    assert config["cross_fit"]["folds"] == 3
    assert config["cross_fit"]["heldout_states_per_fold"] == 12
    assert config["cross_fit"]["each_recovery_state_evaluated_once"] is True
    assert config["evaluation"]["unit"] == "paired_seed_crossfit_aggregate"


def test_l274_is_a_frozen_two_by_two_probe():
    arms = {row["name"]: row for row in _config()["design"]["arms"]}
    assert set(arms) == {
        "source_69d", "source_privileged75d",
        "recovery_augmented_69d", "recovery_augmented_privileged75d",
    }
    assert not arms["source_69d"]["recovery_training"]
    assert not arms["source_69d"]["privileged_features"]
    assert arms["source_privileged75d"]["privileged_features"]
    assert arms["recovery_augmented_69d"]["recovery_training"]
    assert arms["recovery_augmented_privileged75d"]["recovery_training"]
    assert arms["recovery_augmented_privileged75d"]["privileged_features"]


def test_l274_privileged_fields_are_probe_only_and_fixed():
    config = _config()
    assert config["features"]["base_observation_dim"] == 69
    assert config["features"]["privileged_dim"] == 6
    assert len(config["features"]["privileged_path_fields"]) == 6
    text = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    assert "intentionally privileged probes" in text
    assert "does not authorize adding them" in text


def test_l274_forbids_final_and_invalid_artifacts():
    config = _config()
    searchable = str({key: value for key, value in config.items() if key != "forbidden_tokens"}).lower()
    assert not any(token.lower() in searchable for token in config["forbidden_tokens"])

