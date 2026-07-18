import numpy as np
import pytest

from mobile_robot_mppi.learning.models import ResidualComponentMaskedDynamics


class _Residual:
    state_dim = 5
    control_dim = 2
    model = object()

    def derivative(self, state, control, time=None):
        del control, time
        state = np.asarray(state)
        return np.ones(state.shape[:-1] + (5,), dtype=np.float64)


def test_component_mask_preserves_batch_shape_and_zeros_known_kinematics():
    masked = ResidualComponentMaskedDynamics(_Residual(), [0, 0, 0, 1, 1])
    value = masked.derivative(np.zeros((7, 5)), np.zeros((7, 2)))
    np.testing.assert_array_equal(value[:, :3], 0.0)
    np.testing.assert_array_equal(value[:, 3:], 1.0)


@pytest.mark.parametrize("mask", ([0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 2]))
def test_component_mask_rejects_invalid_masks(mask):
    with pytest.raises(ValueError):
        ResidualComponentMaskedDynamics(_Residual(), mask)

