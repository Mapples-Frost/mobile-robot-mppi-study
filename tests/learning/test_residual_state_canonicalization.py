import numpy as np

from mobile_robot_mppi.learning.models import (
    CanonicalizedStateResidualDynamics,
)


class _DiagnosticResidual:
    state_dim = 5
    control_dim = 2

    def __init__(self):
        self.last_state = None
        self.innovation_samples = 3

    def derivative(self, state, control, time=None):
        del control, time
        self.last_state = np.asarray(state, dtype=np.float64).copy()
        return self.last_state

    def member_derivatives(self, state, control, time=None):
        value = self.derivative(state, control, time)
        return np.stack((value, value), axis=0)

    def disagreement(self, state, control, time=None):
        self.derivative(state, control, time)
        return np.zeros(np.asarray(state).shape[:-1])

    def support_confidence(self, state, control):
        self.derivative(state, control)
        return np.ones(np.asarray(state).shape[:-1])


def test_canonicalization_applies_to_prediction_and_diagnostics():
    base = _DiagnosticResidual()
    model = CanonicalizedStateResidualDynamics(
        base, state_indices=(0, 1), canonical_values=(3.25, 0.31)
    )
    states = np.asarray([
        [0.0, 0.0, 0.2, 0.3, 0.4],
        [6.0, 5.0, -0.2, 0.1, -0.4],
    ])
    controls = np.zeros((2, 2))
    output = model.derivative(states, controls)
    assert np.allclose(output[:, :2], [[3.25, 0.31], [3.25, 0.31]])
    assert np.allclose(states[:, :2], [[0.0, 0.0], [6.0, 5.0]])
    assert np.allclose(model.support_confidence(states, controls), 1.0)
    assert np.allclose(base.last_state[:, :2], [[3.25, 0.31], [3.25, 0.31]])


def test_canonicalization_rejects_invalid_contracts():
    base = _DiagnosticResidual()
    for indices, values in (((0, 0), (1.0, 2.0)), ((5,), (1.0,)), ((0,), ())):
        try:
            CanonicalizedStateResidualDynamics(base, indices, values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid canonicalization contract accepted")
