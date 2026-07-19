import numpy as np

from mobile_robot_mppi.core.spaces import dynamic_unicycle_state
from mobile_robot_mppi.learning.models import PlatformResidualEnsemble
from mobile_robot_mppi.rl.residual_context import ResidualContextEncoder


class _Member:
    state_dim = 5
    control_dim = 2

    def __init__(self, residual, support=1.0):
        self.residual = np.asarray(residual, dtype=np.float64)
        self.support = float(support)

    def derivative(self, state, control, time=None):
        del control, time
        state = np.asarray(state, dtype=np.float64)
        return np.broadcast_to(self.residual, state.shape)

    def support_confidence(self, state, control):
        del control
        return np.full(np.asarray(state).shape[:-1], self.support)


class _Combined:
    state_dim = 5
    control_dim = 2

    def __init__(self, residual):
        self.residual = residual


def test_residual_context_is_scaled_causal_and_batched():
    ensemble = PlatformResidualEnsemble(
        [
            _Member((0.0, 0.0, 0.0, 0.20, -0.30), support=0.9),
            _Member((0.0, 0.0, 0.0, 0.30, -0.50), support=0.7),
        ],
        disagreement_scales=(1.0, 1.0, 1.0, 0.25, 0.50),
        innovation_scales=(1.0, 1.0, 1.0, 0.25, 0.50),
        innovation_decay=0.5,
    )
    context = ResidualContextEncoder(
        _Combined(ensemble),
        dynamic_unicycle_state(),
        {
            "enabled": True,
            "residual_scales": (0.25, 0.50),
            "innovation_scales": (0.25, 0.50),
        },
    )
    states = np.zeros((3, 5))
    controls = np.zeros((3, 2))

    cold = context.features(states, controls)
    assert cold.shape == (3, 7)
    np.testing.assert_allclose(
        cold[:, :2], np.tile((1.0, -0.8), (states.shape[0], 1))
    )
    np.testing.assert_allclose(cold[:, 2:4], 0.0)
    np.testing.assert_allclose(cold[:, 5], 0.7)
    np.testing.assert_allclose(cold[:, 6], 0.0)

    # Only this already-completed prediction error can affect the next call.
    ensemble.observe_prediction_errors(
        np.zeros(5), np.asarray((0.0, 0.0, 0.0, 0.125, -0.25))
    )
    warm = context.features(states, controls)
    np.testing.assert_allclose(
        warm[:, 2:4], np.tile((0.5, -0.5), (states.shape[0], 1))
    )
    np.testing.assert_allclose(warm[:, 6], 1.0)


def test_residual_context_reset_removes_cross_episode_innovation():
    ensemble = PlatformResidualEnsemble(
        [_Member(np.zeros(5)), _Member(np.zeros(5))],
        state_scales=np.ones(5),
    )
    context = ResidualContextEncoder(
        _Combined(ensemble), dynamic_unicycle_state(), {"enabled": True}
    )
    ensemble.observe_prediction_errors(np.zeros(5), np.ones(5))
    assert context.features(np.zeros(5), np.zeros(2))[-1] == 1.0
    context.reset()
    values = context.features(np.zeros(5), np.zeros(2))
    np.testing.assert_allclose(values[2:4], 0.0)
    assert values[-1] == 0.0
