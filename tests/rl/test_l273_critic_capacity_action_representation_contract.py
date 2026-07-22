from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l273_critic_capacity_action_representation_diagnosis.yaml"
PROTOCOL = ROOT / "docs/experiments/post_l217/l273_critic_capacity_action_representation_diagnosis_protocol.md"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l273"]


def test_l273_is_diagnostic_only_and_uses_state_level_splits():
    config = _config()
    assert config["protocol"] == "L273"
    assert config["actor_training_authorized"] is False
    assert config["oracle_fit"]["expected_train_states"] == 18
    assert config["oracle_fit"]["expected_validation_states"] == 8
    assert config["external_evaluation"]["expected_states"] == 36
    assert config["evaluation"]["inference_unit"] == "paired_seed"


def test_l273_is_a_frozen_paired_factorial_probe():
    design = _config()["design"]
    assert design["device"] == "cuda"
    assert len(design["paired_seeds"]) == 3
    assert design["updates"] == 3000
    assert design["checkpoint_updates"] == [1000, 3000]
    arms = {row["name"]: row for row in design["arms"]}
    assert set(arms) == {
        "baseline_raw", "capacity_wide_raw",
        "baseline_polynomial", "capacity_wide_polynomial",
    }
    assert arms["baseline_raw"]["hidden_sizes"] == [256, 256]
    assert arms["capacity_wide_raw"]["hidden_sizes"] == [512, 512]
    assert arms["baseline_polynomial"]["action_encoding"] == "polynomial_degree3"


def test_l273_has_no_forbidden_artifact_reference():
    config = _config()
    searchable = str({key: value for key, value in config.items() if key != "forbidden_tokens"}).lower()
    assert not any(token.lower() in searchable for token in config["forbidden_tokens"])


def test_l273_protocol_records_oracle_non_deployment_boundary():
    text = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    assert "not policy training" in text
    assert "never supplied to an Actor" in text
    assert "No result authorizes Actor training automatically" in text
