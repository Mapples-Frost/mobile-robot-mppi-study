import numpy as np
import pytest

from experiments.rl.run_control_sequence_ranking_diagnostic import (
    _array_sha256,
    _ranking_metrics,
    _spearman,
)


def test_spearman_recovers_perfect_and_reversed_ordering():
    observed = np.asarray((4.0, 1.0, 3.0, 2.0))
    assert _spearman(observed, observed) == pytest.approx(1.0)
    assert _spearman(-observed, observed) == pytest.approx(-1.0)


def test_ranking_metrics_reward_correct_elites_and_argmin():
    observed = np.linspace(0.0, 19.0, 20)
    perfect = _ranking_metrics(observed, observed, temperature=2.0)
    shifted = _ranking_metrics(np.roll(observed, 5), observed, temperature=2.0)

    assert perfect["elite_recall"] == 1.0
    assert perfect["normalized_regret"] == 0.0
    assert perfect["top1_hit"] == 1
    assert perfect["weight_js_divergence"] == 0.0
    assert shifted["elite_recall"] < perfect["elite_recall"]
    assert shifted["normalized_regret"] > perfect["normalized_regret"]


def test_candidate_hash_includes_values_shape_and_dtype():
    values = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    assert _array_sha256(values) == _array_sha256(values.copy())
    assert _array_sha256(values) != _array_sha256(values + 1.0)
    assert _array_sha256(values) != _array_sha256(values.reshape(3, 2, 2))
    assert _array_sha256(values) != _array_sha256(values.astype(np.float32))
