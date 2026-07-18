import numpy as np
import pytest

from mobile_robot_mppi.learning.models import PlatformResidualEnsemble


class _Member:
    state_dim = 3
    control_dim = 2

    def __init__(self, offset, support):
        self.offset = float(offset)
        self.support = float(support)

    def derivative(self, state, control, time=None):
        del control, time
        state = np.asarray(state, dtype=np.float64)
        return np.zeros_like(state) + self.offset

    def support_confidence(self, state, control):
        del control
        return np.full(np.asarray(state).shape[:-1], self.support)


def test_residual_ensemble_returns_mean_and_normalized_disagreement():
    ensemble = PlatformResidualEnsemble(
        [_Member(0.0, 0.8), _Member(2.0, 0.6)],
        state_scales=(1.0, 2.0, 4.0),
    )
    states = np.zeros((5, 3))
    controls = np.zeros((5, 2))

    np.testing.assert_allclose(
        ensemble.derivative(states, controls), np.ones((5, 3))
    )
    expected = np.sqrt(np.mean(np.asarray((1.0, 0.5, 0.25)) ** 2))
    np.testing.assert_allclose(
        ensemble.disagreement(states, controls),
        np.full(5, expected),
    )
    np.testing.assert_allclose(
        ensemble.support_confidence(states, controls),
        np.full(5, 0.6),
    )


def test_residual_ensemble_innovation_uses_completed_residual_error():
    ensemble = PlatformResidualEnsemble(
        [_Member(0.0, 1.0), _Member(0.0, 1.0)],
        state_scales=(1.0, 2.0, 4.0),
        innovation_decay=0.5,
    )

    first = ensemble.observe_prediction_errors(
        np.full(3, 100.0), np.asarray((1.0, 2.0, 4.0))
    )
    second = ensemble.observe_prediction_errors(
        np.zeros(3), np.zeros(3)
    )

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(0.5)
    assert ensemble.diagnostics()["residual_ensemble_innovation_samples"] == 2


def test_residual_ensemble_keeps_derivative_and_state_error_scales_separate():
    ensemble = PlatformResidualEnsemble(
        [_Member(0.0, 1.0), _Member(2.0, 1.0)],
        disagreement_scales=(1.0, 2.0, 4.0),
        innovation_scales=(2.0, 2.0, 2.0),
        innovation_decay=0.0,
    )

    expected_disagreement = np.sqrt(
        np.mean(np.asarray((1.0, 0.5, 0.25)) ** 2)
    )
    assert ensemble.disagreement(np.zeros(3), np.zeros(2)) == pytest.approx(
        expected_disagreement
    )
    assert ensemble.observe_prediction_errors(
        np.zeros(3), np.asarray((2.0, 2.0, 2.0))
    ) == pytest.approx(1.0)


def test_residual_ensemble_rejects_misaligned_members():
    class _Wrong(_Member):
        state_dim = 4

    with pytest.raises(ValueError, match="dimensions differ"):
        PlatformResidualEnsemble([_Member(0.0, 1.0), _Wrong(0.0, 1.0)])
