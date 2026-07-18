import numpy as np
import pytest
import torch

from mobile_robot_mppi.learning.models import NormalizedSupportGatedResidualDynamics


class _Model:
    angle_indices = (2,)
    feature_mean = torch.zeros(6)
    feature_scale = torch.ones(6)
    control_mean = torch.zeros(2)
    control_scale = torch.ones(2)


class _Residual:
    state_dim = 5
    control_dim = 2
    model = _Model()

    def derivative(self, state, control, time=None):
        del control, time
        return np.ones_like(np.asarray(state, dtype=np.float64))


def test_support_gate_is_one_inside_and_zero_beyond_hard_limit():
    gated = NormalizedSupportGatedResidualDynamics(_Residual(), 3.0, 5.0)
    inside = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
    outside = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0])
    control = np.zeros(2)
    assert gated.confidence(inside, control) == 1.0
    assert gated.confidence(outside, control) == 0.0
    assert np.allclose(gated.derivative(outside, control), 0.0)


def test_support_gate_interpolates_linearly_for_batches():
    gated = NormalizedSupportGatedResidualDynamics(_Residual(), 3.0, 5.0)
    states = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0]])
    confidence = gated.confidence(states, np.zeros((2, 2)))
    assert np.allclose(confidence, [1.0, 0.5])


def test_support_gate_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="soft_z"):
        NormalizedSupportGatedResidualDynamics(_Residual(), 5.0, 5.0)
