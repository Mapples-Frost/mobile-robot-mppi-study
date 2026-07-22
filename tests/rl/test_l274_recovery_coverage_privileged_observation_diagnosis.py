import numpy as np

from experiments.rl.run_l274_recovery_coverage_privileged_observation_diagnosis import (
    _assign_folds,
    _decision,
    _privileged_vector,
)


def test_fold_assignment_holds_two_states_per_scene_per_fold():
    states = []
    for scene in ("a", "b"):
        for index in range(6):
            states.append({
                "state_id": "%s%d" % (scene, index), "scene": scene,
                "severity": index // 2, "side": index % 2, "chain_id": index,
            })
    assignment = _assign_folds(states, 3)
    for scene in ("a", "b"):
        assert sorted(assignment["%s%d" % (scene, index)] for index in range(6)) == [0, 0, 1, 1, 2, 2]


def test_privileged_vector_uses_path_local_fractions_only():
    state = {"path_state": {
        "signed_cross_track_error": -0.4, "cross_track_error": 0.4,
        "heading_error": -0.2, "curvature": 0.5,
        "progress": 3.0, "remaining": 1.0,
    }}
    assert np.allclose(_privileged_vector(state), [-0.4, 0.4, -0.2, 0.5, 0.75, 0.25])


def _result(seed, arm, accuracy, spearman, scenes):
    return {
        "seed": seed, "arm": arm,
        "recovery_forward_accuracy": accuracy,
        "three_action_spearman": spearman,
        "scene_pair_accuracy": scenes,
    }


def test_decision_identifies_recovery_coverage_when_only_coverage_passes():
    results = []
    base = {"s%d" % value: 0.5 for value in range(6)}
    better = {"s%d" % value: 0.75 for value in range(6)}
    for seed in (1, 2, 3):
        results.extend([
            _result(seed, "source_69d", 0.50, 0.00, base),
            _result(seed, "source_privileged75d", 0.52, 0.02, base),
            _result(seed, "recovery_augmented_69d", 0.75, 0.25, better),
            _result(seed, "recovery_augmented_privileged75d", 0.76, 0.26, better),
        ])
    section = {"gate": {
        "minimum_aggregate_pair_accuracy": 0.65,
        "minimum_paired_median_pair_gain": 0.10,
        "minimum_paired_median_spearman_gain": 0.10,
        "minimum_consistent_seed_blocks": 2,
        "minimum_scenes_improved": 4,
        "maximum_scene_pair_decrease": 0.166667,
    }}
    status, selected, comparisons = _decision(results, section)
    assert status == "recovery_state_coverage_limited"
    assert selected == "recovery_augmented_69d"
    assert [row["gate_pass"] for row in comparisons] == [False, True, True]
