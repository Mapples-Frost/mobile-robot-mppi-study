import numpy as np
import pytest

from src.dynamics import (
    CombinedDynamics,
    DisturbanceConfig,
    DisturbedUnicycle,
    NominalUnicycle,
)
from src.dynamics.residual import OracleResidual, ZeroResidual


def _instantaneous_true_model():
    return DisturbedUnicycle(
        DisturbanceConfig(
            velocity_gain=1.2,
            yaw_gain=0.8,
            yaw_bias=0.15,
            world_disturbance_amplitude=(0.2, -0.3, 0.1),
            world_disturbance_frequency=(0.5, 0.75, 1.25),
            world_disturbance_phase=(0.1, -0.2, 0.4),
            state_disturbance_gain=(0.05, -0.1, 0.2),
        )
    )


def test_oracle_residual_is_exact_true_minus_nominal_identity():
    nominal = NominalUnicycle()
    true_model = _instantaneous_true_model()
    oracle = OracleResidual(true_model, nominal)
    state = np.asarray([1.5, -0.4, 0.7])
    control = np.asarray([1.1, -0.3])
    time = 0.9

    true_derivative = true_model.derivative(state, control, time=time)
    nominal_derivative = nominal.derivative(state, control, time=time)
    residual = oracle.derivative(state, control, time=time)

    np.testing.assert_array_equal(residual, true_derivative - nominal_derivative)
    np.testing.assert_allclose(nominal_derivative + residual, true_derivative)


def test_zero_residual_returns_additive_identity():
    residual = ZeroResidual()

    actual = residual.derivative([1.0, -2.0, 0.3], [0.5, -0.1], time=2.0)

    np.testing.assert_array_equal(actual, np.zeros(3))


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_combined_zero_residual_matches_nominal_derivative_and_step(method):
    nominal = NominalUnicycle()
    combined = CombinedDynamics(nominal, ZeroResidual())
    state = np.asarray([0.6, -0.2, -0.45])
    control = np.asarray([0.9, 0.25])

    np.testing.assert_array_equal(
        combined.derivative(state, control, time=0.3),
        nominal.derivative(state, control, time=0.3),
    )
    np.testing.assert_array_equal(
        combined.step(state, control, 0.05, method=method, time=0.3),
        nominal.step(state, control, 0.05, method=method, time=0.3),
    )


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_combined_oracle_matches_instantaneous_true_dynamics(method):
    nominal = NominalUnicycle()
    true_model = _instantaneous_true_model()
    combined = CombinedDynamics(nominal, OracleResidual(true_model, nominal))
    state = np.asarray([0.6, -0.2, -0.45])
    control = np.asarray([0.9, 0.25])
    time = 0.3

    np.testing.assert_allclose(
        combined.derivative(state, control, time=time),
        true_model.derivative(state, control, time=time),
    )
    np.testing.assert_allclose(
        combined.step(state, control, 0.05, method=method, time=time),
        true_model.step(state, control, 0.05, method=method, time=time),
    )
