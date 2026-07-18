import numpy as np
import pytest

from experiments.rl.collect_correction_counterfactual_branches import (
    _branch_steps,
    _episode_seeds,
)
from mobile_robot_mppi.rl.risk_dataset import (
    CounterfactualLabelConfig,
    RiskDatasetSufficiencyConfig,
    assess_risk_dataset_sufficiency,
    burst_accounting_valid,
    classify_counterfactual,
    feature_vector,
    validate_episode_split,
    validate_sealed_episode_split,
)


def _summary(
    reward=1.0,
    distance=1.0,
    success=False,
    collision=False,
    clearance=0.5,
    overrides=0,
):
    return {
        "return": reward,
        "goal_distance": distance,
        "success": success,
        "collision": collision,
        "minimum_clearance": clearance,
        "safety_overrides": overrides,
    }


def test_counterfactual_labels_preserve_continuous_targets():
    base = _summary(reward=2.0, distance=1.0)
    harmful = classify_counterfactual(
        base, _summary(reward=1.0, distance=1.1)
    )
    assert harmful["label"] == "harmful"
    assert harmful["return_delta"] == pytest.approx(-1.0)
    assert harmful["goal_distance_improvement"] == pytest.approx(-0.1)

    beneficial = classify_counterfactual(
        base, _summary(reward=3.0, distance=0.9)
    )
    assert beneficial["label"] == "beneficial"
    assert beneficial["label_index"] == 2

    neutral = classify_counterfactual(
        base, _summary(reward=2.1, distance=0.99)
    )
    assert neutral["label"] == "neutral"


def test_collision_or_success_regression_has_priority():
    collision = classify_counterfactual(
        _summary(success=False),
        _summary(reward=100.0, distance=0.0, collision=True),
    )
    assert collision["label"] == "harmful"
    assert collision["collision_regression"] is True

    success_loss = classify_counterfactual(
        _summary(success=True),
        _summary(reward=100.0, distance=0.0, success=False),
    )
    assert success_loss["label"] == "harmful"
    assert success_loss["success_loss"] is True


@pytest.mark.parametrize(
    "values,match",
    [
        ({"harmful_return_delta": 0.0}, "negative"),
        ({"harmful_distance_improvement": 0.0}, "negative"),
        ({"beneficial_return_delta": 0.0}, "positive"),
        ({"beneficial_distance_improvement": float("nan")}, "finite"),
    ],
)
def test_label_config_fails_closed(values, match):
    with pytest.raises(ValueError, match=match):
        CounterfactualLabelConfig.from_mapping(values).validate()


def test_episode_split_and_branch_steps_are_group_safe():
    validate_episode_split((1, 2), (3, 4))
    with pytest.raises(ValueError, match="disjoint"):
        validate_episode_split((1, 2), (2, 3))
    assert _branch_steps("10,30,50") == [10, 30, 50]
    with pytest.raises(ValueError, match="increasing"):
        _branch_steps("30,10")
    assert _episode_seeds("5,6") == [5, 6]
    assert _episode_seeds([5, 6]) == [5, 6]

    validate_sealed_episode_split((1, 2), (3, 4), (5, 6))
    with pytest.raises(ValueError, match="sealed"):
        validate_sealed_episode_split((1, 2), (3, 4), (4, 5))


def test_burst_accounting_requires_first_acceptance_and_conserved_counts():
    row = {
        "candidate_intervention_steps_requested": 10,
        "candidate_intervention_steps_run": 10,
        "candidate_intervention_accepted_steps": 6,
        "candidate_intervention_rejected_steps": 4,
        "candidate_intervention_first_gate_accepted": True,
        "candidate_intervention_accept_fraction": 0.6,
        "candidate_intervention_correction_abs_mean": 0.1,
        "candidate_intervention_correction_abs_max": 0.3,
    }
    assert burst_accounting_valid(row) is True
    row["candidate_intervention_rejected_steps"] = 3
    assert burst_accounting_valid(row) is False
    row["candidate_intervention_rejected_steps"] = 4
    row["candidate_intervention_first_gate_accepted"] = "False"
    assert burst_accounting_valid(row) is False


def test_feature_vector_has_fixed_leakage_free_schema():
    diagnostics = {
        "online_advantage_q1": 1.0,
        "online_advantage_q2": 2.0,
        "online_advantage_mean": 1.5,
        "online_advantage_half_disagreement": 0.5,
        "online_consensus_lcb": 0.5,
        "target_advantage_q1": 3.0,
        "target_advantage_q2": 4.0,
        "target_advantage_mean": 3.5,
        "target_advantage_half_disagreement": 0.5,
        "target_consensus_lcb": 2.5,
        "raw_applied_correction_abs_mean": 0.1,
        "raw_applied_correction_abs_max": 0.2,
    }
    vector, schema = feature_vector(
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((0.0, 0.5)),
        np.asarray((0.25, 0.0)),
        diagnostics,
        10,
    )
    assert schema["feature_dim"] == 22
    assert vector.shape == (22,)
    np.testing.assert_allclose(vector[7:9], (0.25, -0.5))


def test_risk_dataset_sufficiency_requires_both_classes_and_all_seeds():
    summary = {
        "quality_passed": True,
        "train_samples": 100,
        "validation_samples": 20,
        "label_counts": {
            "train": {"harmful": 8, "beneficial": 9, "neutral": 83},
            "validation": {"harmful": 2, "beneficial": 2, "neutral": 16},
        },
        "train_training_seeds": [1, 2, 3],
        "validation_training_seeds": [1, 2, 3],
    }
    passed = assess_risk_dataset_sufficiency(summary)
    assert passed["train_risk_model"] is True

    summary["label_counts"]["validation"]["harmful"] = 0
    failed = assess_risk_dataset_sufficiency(summary)
    assert failed["train_risk_model"] is False
    assert failed["checks"]["enough_validation_harmful"] is False


def test_risk_dataset_sufficiency_config_rejects_invalid_counts():
    with pytest.raises(ValueError, match="counts"):
        RiskDatasetSufficiencyConfig(minimum_train_samples=-1).validate()
