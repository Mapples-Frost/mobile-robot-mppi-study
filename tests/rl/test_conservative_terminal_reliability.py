import numpy as np
import pytest

from mobile_robot_mppi.rl.reliability import (
    ConservativeTerminalReliability,
)


class ResidualSignals:
    innovation_samples = 4
    innovation_error_ema = 0.15

    def disagreement(self, states, controls):
        del controls
        return np.asarray(states)[:, 0]

    def support_confidence(self, states, controls):
        del controls
        return np.asarray(states)[:, 1]


def _evaluator():
    return ConservativeTerminalReliability({
        "enabled": True,
        "ensemble_disagreement_soft": 0.1,
        "ensemble_disagreement_hard": 0.3,
        "innovation_error_soft": 0.1,
        "innovation_error_hard": 0.3,
        "innovation_minimum_samples": 3,
        "critic_ood_soft": 2.0,
        "critic_ood_hard": 6.0,
        "critic_disagreement_soft": 0.5,
        "critic_disagreement_hard": 1.5,
        "uncertainty_penalty_weight": 2.0,
    })


def test_candidate_authority_is_product_of_dynamics_and_critic_confidence():
    result = _evaluator().evaluate(
        ResidualSignals(),
        np.asarray(((0.1, 1.0), (0.3, 1.0))),
        np.zeros((2, 1)),
        critic_ood_scores=np.asarray((2.0, 6.0)),
        critic_disagreement=np.asarray((0.5, 1.5)),
    )

    # Completed-transition innovation has confidence 0.75.
    np.testing.assert_allclose(
        result["dynamics_confidence"], (0.75, 0.0)
    )
    np.testing.assert_allclose(result["critic_confidence"], (1.0, 0.0))
    np.testing.assert_allclose(result["authority"], (0.75, 0.0))
    np.testing.assert_allclose(result["dynamics_uncertainty"], (0.0, 1.0))


def test_terminal_reliability_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="soft < hard"):
        ConservativeTerminalReliability({
            "critic_disagreement_soft": 2.0,
            "critic_disagreement_hard": 1.0,
        })


def test_unvalidated_critic_signals_can_be_excluded_explicitly():
    evaluator = ConservativeTerminalReliability({
        "enabled": True,
        "ensemble_disagreement_soft": 0.1,
        "ensemble_disagreement_hard": 0.3,
        "innovation_error_soft": 0.1,
        "innovation_error_hard": 0.3,
        "use_critic_support": False,
        "use_critic_disagreement": False,
    })
    result = evaluator.evaluate(
        ResidualSignals(),
        np.asarray(((0.1, 1.0),)),
        np.zeros((1, 1)),
        critic_ood_scores=np.asarray((100.0,)),
        critic_disagreement=np.asarray((100.0,)),
    )

    np.testing.assert_allclose(result["critic_confidence"], (1.0,))
