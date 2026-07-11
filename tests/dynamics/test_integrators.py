import numpy as np
import pytest

from src.dynamics import integrate_step


class ConstantDynamics:
    state_dim = 3
    control_dim = 2
    angle_indices = ()

    def __init__(self, value=(1.5, -2.0, 0.25)):
        self.value = np.asarray(value, dtype=np.float64)

    def derivative(self, state, control, time=None):
        return self.value.copy()


class TimeRecordingDynamics(ConstantDynamics):
    def __init__(self):
        super().__init__()
        self.times = []

    def derivative(self, state, control, time=None):
        self.times.append(time)
        return np.asarray([time, 0.0, 0.0], dtype=np.float64)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_constant_derivative_matches_analytic_solution(method):
    dynamics = ConstantDynamics()
    state = np.asarray([0.5, -1.0, 2.0])
    control = np.asarray([10.0, -4.0])
    dt = 0.2

    actual = integrate_step(dynamics, state, control, dt, method=method)

    np.testing.assert_allclose(actual, state + dt * dynamics.value)


def test_rk4_uses_start_half_half_end_times():
    dynamics = TimeRecordingDynamics()
    state = np.asarray([1.0, 2.0, 3.0])
    t0 = 1.25
    dt = 0.4

    actual = integrate_step(
        dynamics, state, np.zeros(2), dt, method="rk4", time=t0
    )

    np.testing.assert_allclose(dynamics.times, [t0, t0 + dt / 2, t0 + dt / 2, t0 + dt])
    np.testing.assert_allclose(actual, [state[0] + t0 * dt + dt**2 / 2, 2.0, 3.0])


@pytest.mark.parametrize("dt", [0.0, -1e-9, -1.0])
def test_nonpositive_dt_is_rejected(dt):
    with pytest.raises(ValueError, match="dt|greater than zero|positive"):
        integrate_step(ConstantDynamics(), np.zeros(3), np.zeros(2), dt)


@pytest.mark.parametrize("dt", [np.nan, np.inf, -np.inf])
def test_nonfinite_dt_is_rejected(dt):
    with pytest.raises(ValueError, match="dt|finite"):
        integrate_step(ConstantDynamics(), np.zeros(3), np.zeros(2), dt)


def test_unknown_integration_method_is_rejected():
    with pytest.raises(ValueError, match="unknown integration method"):
        integrate_step(
            ConstantDynamics(), np.zeros(3), np.zeros(2), 0.1, method="rk45"
        )


@pytest.mark.parametrize(
    "state, control",
    [
        (np.zeros(2), np.zeros(2)),
        (np.zeros((1, 3)), np.zeros(2)),
        (np.zeros(3), np.zeros(1)),
        (np.zeros(3), np.zeros((1, 2))),
    ],
)
def test_state_and_control_shapes_are_validated(state, control):
    with pytest.raises(ValueError, match="shape"):
        integrate_step(ConstantDynamics(), state, control, 0.1)


@pytest.mark.parametrize(
    "state, control",
    [
        ([np.nan, 0.0, 0.0], [0.0, 0.0]),
        ([np.inf, 0.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0, 0.0], [np.nan, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, -np.inf]),
    ],
)
def test_nonfinite_state_and_control_are_rejected(state, control):
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        integrate_step(ConstantDynamics(), state, control, 0.1)


@pytest.mark.parametrize(
    "derivative",
    [
        np.zeros(2),
        np.asarray([0.0, np.nan, 0.0]),
        np.asarray([0.0, 0.0, np.inf]),
    ],
)
def test_invalid_derivative_shape_and_values_are_rejected(derivative):
    with pytest.raises(ValueError, match="shape|NaN|Inf|finite"):
        integrate_step(
            ConstantDynamics(derivative), np.zeros(3), np.zeros(2), 0.1
        )
