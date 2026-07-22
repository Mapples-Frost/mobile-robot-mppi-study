from pathlib import Path

import numpy as np
import yaml

from experiments.rl.run_l282_recovery_component_counterfactual import (
    _compose_action,
    _decision,
)


ROOT = Path(__file__).resolve().parents[2]


def test_l282_contract_is_evaluation_only_and_complete():
    config = yaml.safe_load(
        (ROOT / "configs/rl/l282_recovery_component_counterfactual.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["diagnostic_only"] is True
    assert config["actor_training_authorized"] is False
    assert config["split"] == "test"
    assert config["expected_chains"] == 18
    assert len(config["checkpoints"]) == 3
    assert config["arms"] == [
        "teacher_both",
        "actor_both",
        "teacher_v_actor_omega",
        "actor_v_teacher_omega",
    ]


def test_l282_protocol_freezes_coverage_attribution_and_forbidden_maps():
    text = (
        ROOT
        / "docs/experiments/post_l217/l282_recovery_component_counterfactual_protocol.md"
    ).read_text(encoding="utf-8").lower()
    assert "216 rollouts" in text
    assert "minimum" not in text or "50%" in text
    assert "final-map evaluation directly" in text
    assert "hairpin/s-chicane/infinity" in text


def test_l282_component_actions_preserve_the_frozen_component_semantics():
    teacher = np.asarray((0.2, -0.7), dtype=np.float32)
    actor = np.asarray((0.9, 0.4), dtype=np.float32)
    np.testing.assert_allclose(
        _compose_action("teacher_v_actor_omega", teacher, actor), (0.2, 0.4)
    )
    np.testing.assert_allclose(
        _compose_action("actor_v_teacher_omega", teacher, actor), (0.9, -0.7)
    )


def test_l282_decision_requires_scene_and_seed_coverage():
    rows = []
    for seed in (1, 2, 3):
        for scene_index in range(6):
            rows.append({
                "seed": seed,
                "scene": f"scene_{scene_index}",
                "teacher_gap": 10.0,
                "teacher_velocity_gain": 7.0 if scene_index < 4 else 0.0,
                "teacher_steering_gain": 1.0,
            })
    decision, metrics = _decision(rows, {
        "minimum_recovered_loss_fraction": 0.5,
        "minimum_attributed_scenes": 4,
        "minimum_attributed_seeds": 2,
    })
    assert decision == "velocity_component_bottleneck"
    assert metrics["velocity_attributed_seed_count"] == 3
