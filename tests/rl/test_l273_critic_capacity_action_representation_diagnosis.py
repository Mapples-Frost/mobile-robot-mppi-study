import numpy as np

from experiments.rl.run_l273_critic_capacity_action_representation_diagnosis import (
    StateActionSampler,
    _action_features,
    _decision,
    _standardize_by_state,
)


def test_polynomial_action_features_are_fixed_and_complete():
    actions = np.asarray([[2.0, -3.0]], dtype=np.float32)
    features = _action_features(actions, "polynomial_degree3")
    assert features.shape == (1, 9)
    assert np.allclose(features[0], [2, -3, 4, 9, -6, 8, -27, -12, 18])
    assert np.array_equal(_action_features(actions, "raw"), actions)


def test_state_standardization_removes_only_within_state_offset_and_scale():
    rows = [
        {"state_id": "a", "return": 1.0},
        {"state_id": "a", "return": 3.0},
        {"state_id": "b", "return": 10.0},
        {"state_id": "b", "return": 14.0},
    ]
    standardized = _standardize_by_state(rows, 1e-6)
    for state in ("a", "b"):
        values = [row["target"] for row in standardized if row["state_id"] == state]
        assert np.isclose(np.mean(values), 0.0)
        assert np.isclose(np.std(values), 1.0)


def test_sampler_selects_states_before_actions_deterministically():
    rows = [
        {"state_id": state, "action_id": action}
        for state in ("a", "b") for action in range(3)
    ]
    first = StateActionSampler(rows, 9).sample(20)
    second = StateActionSampler(rows, 9).sample(20)
    assert [(row["state_id"], row["action_id"]) for row in first] == [
        (row["state_id"], row["action_id"]) for row in second
    ]


def _result(seed, arm, train, validation, external, scenes):
    return {
        "seed": seed,
        "arm": arm,
        "train_mean_state_spearman": train,
        "validation_mean_state_spearman": validation,
        "external_recovery_forward_accuracy": external,
        "external_scene_pair_accuracy": scenes,
    }


def test_decision_prioritizes_action_representation_when_frozen_gate_passes():
    results = []
    baseline_scenes = {"s%d" % value: 0.50 for value in range(6)}
    improved_scenes = {"s%d" % value: 0.75 for value in range(6)}
    for seed in (1, 2, 3):
        results.extend([
            _result(seed, "baseline_raw", 0.50, 0.05, 0.50, baseline_scenes),
            _result(seed, "baseline_polynomial", 0.75, 0.25, 0.75, improved_scenes),
            _result(seed, "capacity_wide_raw", 0.75, 0.25, 0.75, improved_scenes),
            _result(seed, "capacity_wide_polynomial", 0.80, 0.30, 0.80, improved_scenes),
        ])
    thresholds = {
        "minimum_train_spearman_for_sufficiency": 0.80,
        "minimum_validation_spearman_for_sufficiency": 0.20,
        "minimum_external_pair_accuracy_for_sufficiency": 0.65,
        "minimum_paired_median_train_spearman_gain": 0.15,
        "minimum_paired_median_validation_spearman_gain": 0.10,
        "minimum_paired_median_external_pair_gain": 0.10,
        "minimum_consistent_seed_blocks": 2,
        "minimum_external_scenes_improved": 4,
        "maximum_external_scene_pair_decrease": 0.166667,
    }
    decision, selected, blocks, comparisons = _decision(results, {"decision": thresholds})
    assert decision == "action_representation_limited"
    assert selected == "baseline_polynomial"
    assert blocks == 0
    assert all(row["gate_pass"] for row in comparisons)
