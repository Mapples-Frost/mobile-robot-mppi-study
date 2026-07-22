import numpy as np

from experiments.rl.run_l276_recovery_sequence_imitation_initialization import (
    _gate,
    _sample_recovery,
)


GATE = {
    "minimum_median_relative_test_rmse_improvement": 0.20,
    "minimum_median_validation_return_gain": 0.0,
    "minimum_median_test_return_gain": 0.0,
    "minimum_test_reentry_fraction_gain": 0.15,
    "minimum_test_scenes_with_return_improvement": 4,
    "maximum_source_replay_mean_absolute_action_drift": 0.10,
    "maximum_collision_boundary_failure_increase": 0.0,
}


def test_l276_scene_then_chain_sampler_is_deterministic_and_bounded():
    chain = lambda value: {
        "observations": np.full((3, 69), value, dtype=np.float32),
        "actions": np.full((3, 2), 0.1 * value, dtype=np.float32),
    }
    groups = {"a": [chain(1), chain(2)], "b": [chain(3), chain(4)]}
    left = _sample_recovery(groups, np.random.RandomState(7), 32)
    right = _sample_recovery(groups, np.random.RandomState(7), 32)
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    assert left[0].shape == (32, 69)
    assert np.max(np.abs(left[1])) <= 1.0


def _passing_rows():
    seed_metrics = []
    rollout_rows = []
    for seed in (1, 2, 3):
        seed_metrics.append({
            "relative_test_rmse_improvement": 0.30,
            "median_validation_return_gain": 1.0,
            "median_test_return_gain": 1.0,
            "test_reentry_fraction_gain": 0.20,
            "source_replay_mean_absolute_action_drift": 0.05,
        })
        for scene in range(6):
            rollout_rows.extend((
                {
                    "seed": seed, "split": "test", "scene": str(scene),
                    "policy": "source_actor", "return_gain_vs_source": 0.0,
                    "collision": False, "boundary_violation": False,
                },
                {
                    "seed": seed, "split": "test", "scene": str(scene),
                    "policy": "l276_initialized", "return_gain_vs_source": 1.0,
                    "collision": False, "boundary_violation": False,
                },
            ))
    return seed_metrics, rollout_rows


def test_l276_gate_recognizes_safe_recovery_initialization():
    seed_metrics, rollout_rows = _passing_rows()
    checks, metrics = _gate(seed_metrics, rollout_rows, True, True, GATE)
    assert all(checks.values())
    assert metrics["test_scenes_with_return_improvement"] == 6


def test_l276_gate_fails_closed_on_drift_and_nonactor_change():
    seed_metrics, rollout_rows = _passing_rows()
    seed_metrics[0]["source_replay_mean_absolute_action_drift"] = 0.2
    checks, _ = _gate(seed_metrics, rollout_rows, False, True, GATE)
    assert not checks["source_replay_drift"]
    assert not checks["frozen_nonactor_hashes"]

