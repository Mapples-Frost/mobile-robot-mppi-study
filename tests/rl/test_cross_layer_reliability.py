import numpy as np
import pytest

from mobile_robot_mppi.rl.reliability import (
    HybridSamplingReliability,
    SourceRelativeCompetence,
    decreasing_linear_confidence,
    fuse_hybrid_confidence,
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


def test_innovation_anchor_ignores_soft_support_but_keeps_hard_actor_veto():
    dynamics, actor_factor = fuse_hybrid_confidence(
        disagreement_confidence=0.1,
        innovation_confidence=0.8,
        support_confidence=0.2,
        actor_confidence=0.4,
        innovation_ready=True,
        mode="innovation_anchor",
    )
    assert dynamics == 0.8
    assert actor_factor == 1.0

    _, actor_factor = fuse_hybrid_confidence(
        disagreement_confidence=0.9,
        innovation_confidence=0.8,
        support_confidence=0.9,
        actor_confidence=0.0,
        innovation_ready=True,
        mode="innovation_anchor",
    )
    assert actor_factor == 0.0


def test_innovation_anchor_uses_conservative_fallback_before_ready():
    dynamics, actor_factor = fuse_hybrid_confidence(
        disagreement_confidence=0.6,
        innovation_confidence=1.0,
        support_confidence=0.3,
        actor_confidence=1.0,
        innovation_ready=False,
        mode="innovation_anchor",
    )
    assert dynamics == 0.3
    assert actor_factor == 1.0


def test_source_relative_competence_rewards_equal_or_better_elite_yield():
    config = _config()
    config.update({
        "source_competence_enabled": True,
        "source_competence_initial": 0.5,
        "source_competence_decay": 0.0,
    })
    tracker = SourceRelativeCompetence(config)
    equal = tracker.update(10, 20, 100, 200)
    assert equal["updated"] is True
    assert equal["raw_confidence"] == pytest.approx(1.0, abs=0.02)
    assert equal["confidence"] == equal["raw_confidence"]


def test_source_relative_competence_reduces_poor_guided_source_and_resets():
    config = _config()
    config.update({
        "source_competence_enabled": True,
        "source_competence_initial": 0.5,
        "source_competence_decay": 0.5,
    })
    tracker = SourceRelativeCompetence(config)
    result = tracker.update(2, 20, 100, 100)
    assert result["raw_confidence"] < 0.2
    assert result["confidence"] < 0.5
    assert result["updates"] == 1
    tracker.reset()
    assert tracker.confidence == 0.5
    assert tracker.updates == 0


def test_source_relative_competence_retains_state_without_both_sources():
    tracker = SourceRelativeCompetence(_config())
    before = tracker.confidence
    result = tracker.update(0, 5, 0, 100)
    assert result["updated"] is False
    assert tracker.confidence == before


def test_actor_competence_multiplies_dynamics_authority():
    config = _config()
    config["fusion_mode"] = "innovation_anchor"
    evaluator = HybridSamplingReliability(config)
    result = evaluator.evaluate(
        _Residual(),
        np.zeros((5, 5)),
        np.zeros((5, 2)),
        np.full(5, 3.5),
        actor_competence_confidence=0.5,
    )
    assert result["dynamics_confidence"] == pytest.approx(5.0 / 6.0)
    assert result["actor_support_authority_factor"] == 1.0
    assert result["actor_authority_factor"] == 0.5
    assert result["reliability_authority"] == pytest.approx(5.0 / 12.0)
    assert result["guided_fraction"] == 0.3
