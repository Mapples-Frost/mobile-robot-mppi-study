import numpy as np
import pytest

from mobile_robot_mppi.rl.reliability import (
    HybridSamplingReliability,
    decreasing_linear_confidence,
)


class _Residual:
    innovation_samples = 4
    innovation_error_ema = 0.15

    def disagreement(self, states, controls):
        del controls
        return np.linspace(0.01, 0.05, len(states))

    def support_confidence(self, states, controls):
        del controls
        return np.linspace(0.9, 0.7, len(states))


def _config():
    return {
        "enabled": True,
        "ensemble_disagreement_soft": 0.02,
        "ensemble_disagreement_hard": 0.10,
        "innovation_error_soft": 0.10,
        "innovation_error_hard": 0.40,
        "innovation_minimum_samples": 3,
        "actor_ood_soft": 3.0,
        "actor_ood_hard": 7.0,
        "medium_confidence": 0.33,
        "high_confidence": 0.67,
        "low_guided_fraction": 0.0,
        "medium_guided_fraction": 0.3,
        "high_guided_fraction": 0.6,
    }


def test_decreasing_linear_confidence_has_frozen_endpoints():
    np.testing.assert_allclose(
        decreasing_linear_confidence(
            np.asarray((1.0, 3.0, 5.0, 7.0, 9.0)), 3.0, 7.0
        ),
        (1.0, 1.0, 0.5, 0.0, 0.0),
    )


def test_cross_layer_reliability_uses_worst_rollout_signals():
    evaluator = HybridSamplingReliability(_config())
    result = evaluator.evaluate(
        _Residual(),
        np.zeros((5, 5)),
        np.zeros((5, 2)),
        np.full(5, 3.5),
    )

    assert result["reliability_level"] == "medium"
    assert result["guided_fraction"] == pytest.approx(0.3)
    assert result["ensemble_disagreement_max"] == pytest.approx(0.05)
    assert result["residual_support_confidence_min"] == pytest.approx(0.7)
    assert result["innovation_ready"]


def test_low_reliability_removes_guided_authority():
    evaluator = HybridSamplingReliability(_config())
    result = evaluator.evaluate(
        _Residual(),
        np.zeros((2, 5)),
        np.zeros((2, 2)),
        np.full(2, 8.0),
    )

    assert result["reliability_level"] == "low"
    assert result["reliability_authority"] == 0.0
    assert result["guided_fraction"] == 0.0
