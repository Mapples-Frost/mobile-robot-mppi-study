from pathlib import Path

import yaml

from experiments.rl.run_l283_closed_loop_component_counterfactual import (
    _closed_loop_teacher_action,
    _new_teacher,
)


ROOT = Path(__file__).resolve().parents[2]


def test_l283_is_evaluation_only_and_reuses_all_test_chains():
    config = yaml.safe_load(
        (ROOT / "configs/rl/l283_closed_loop_component_counterfactual.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["protocol"] == "L283"
    assert config["diagnostic_only"] is True
    assert config["actor_training_authorized"] is False
    assert config["split"] == "test"
    assert config["expected_chains"] == 18
    assert len(config["checkpoints"]) == 3


def test_l283_protocol_freezes_actual_state_teacher_and_attribution():
    text = (
        ROOT / "docs/experiments/post_l217/"
        "l283_closed_loop_component_counterfactual_protocol.md"
    ).read_text(encoding="utf-8").lower()
    assert "actual hybrid-rollout state" in text
    assert "216 rollouts" in text
    assert "50%" in text
    assert "never authorizes" in text


def test_l283_exports_closed_loop_teacher_helpers():
    assert callable(_new_teacher)
    assert callable(_closed_loop_teacher_action)
