import numpy as np
import pytest

from mobile_robot_mppi.learning.models import InnovationGatedResidualDynamics


class _ConstantResidual:
    state_dim = 5
    control_dim = 2
    model = None

    def derivative(self, state, control, time=None):
        del control, time
        value = np.zeros_like(np.asarray(state, dtype=np.float64))
        value[..., 3] = 1.0
        return value


def _gate(**overrides):
    values = {
        "state_indices": (3, 4),
        "state_scales": (1.0, 1.0),
        "forgetting_factor": 0.5,
        "minimum_samples": 3,
        "confidence_z": 0.0,
        "off_threshold": 0.0,
        "on_threshold": 0.5,
        "rise_rate": 1.0,
        "fall_rate": 1.0,
    }
    values.update(overrides)
    return InnovationGatedResidualDynamics(_ConstantResidual(), **values)


def test_reliability_gate_fails_closed_then_activates_on_paired_improvement():
    gate = _gate()
    state = np.zeros(5)
    np.testing.assert_array_equal(gate.derivative(state, np.zeros(2)), state)
    for _ in range(2):
        gate.observe_prediction_errors(
            np.asarray((0.0, 0.0, 0.0, 1.0, 0.0)), np.zeros(5)
        )
        assert gate.alpha == 0.0
    gate.observe_prediction_errors(
        np.asarray((0.0, 0.0, 0.0, 1.0, 0.0)), np.zeros(5)
    )
    assert gate.alpha == 1.0
    assert gate.diagnostics()["residual_reliability_samples"] == 3
    assert gate.diagnostics()["residual_reliability_lcb"] > 0.0


def test_reliability_gate_deactivates_when_residual_becomes_worse_and_resets():
    gate = _gate(minimum_samples=1)
    good_nominal = np.asarray((0.0, 0.0, 0.0, 1.0, 0.0))
    bad_residual = np.asarray((0.0, 0.0, 0.0, 2.0, 0.0))
    gate.observe_prediction_errors(good_nominal, np.zeros(5))
    assert gate.alpha == 1.0
    for _ in range(6):
        gate.observe_prediction_errors(np.zeros(5), bad_residual)
    assert gate.alpha == 0.0
    gate.reset()
    assert gate.alpha == 0.0
    assert gate.diagnostics()["residual_reliability_samples"] == 0


@pytest.mark.parametrize(
    "overrides",
    (
        {"state_indices": (), "state_scales": ()},
        {"state_indices": (5,), "state_scales": (1.0,)},
        {"state_indices": (3,), "state_scales": (0.0,)},
        {"forgetting_factor": 0.0},
        {"minimum_samples": 0},
        {"on_threshold": 0.0},
        {"rise_rate": 0.0},
    ),
)
def test_invalid_reliability_gate_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        _gate(**overrides)


def test_actuation_context_multiplies_evidence_and_can_fail_closed():
    active = _gate(
        minimum_samples=1, on_threshold=0.1,
        context_value=1.0, context_off_threshold=0.5,
        context_on_threshold=0.8,
    )
    inactive = _gate(
        minimum_samples=1, on_threshold=0.1,
        context_value=0.4, context_off_threshold=0.5,
        context_on_threshold=0.8,
    )
    nominal_error = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0))
    residual_error = np.zeros(5)
    assert active.observe_prediction_errors(nominal_error, residual_error) == 1.0
    assert inactive.observe_prediction_errors(nominal_error, residual_error) == 0.0
    assert inactive.diagnostics()["residual_reliability_evidence_alpha"] == 1.0
    assert inactive.diagnostics()["residual_reliability_context_alpha"] == 0.0
