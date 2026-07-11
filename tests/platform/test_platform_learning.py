import numpy as np
import torch

from mobile_robot_mppi.learning.models import ResidualNetwork
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, ResidualPrediction
from mobile_robot_mppi.planning.mppi import integrate_batch
from mobile_robot_mppi.core.spaces import dynamic_unicycle_state


def statistics():
    return {
        "feature_mean": np.zeros(6, dtype=np.float32),
        "feature_scale": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(2, dtype=np.float32),
        "control_scale": np.ones(2, dtype=np.float32),
        "residual_mean": np.zeros(5, dtype=np.float32),
        "residual_scale": np.ones(5, dtype=np.float32),
    }


def test_dynamic_five_icode_is_control_affine_and_batched():
    model = ResidualNetwork(
        5, 2,
        {"type": "icode_residual", "angle_indices": [2], "hidden_sizes": [16], "activation": "softplus"},
        statistics(),
    )
    state = torch.zeros((7, 5))
    control = torch.zeros((7, 2))
    residual, parts = model.components(state, control)
    assert residual.shape == (7, 5)
    assert parts["gain_normalized"].shape == (7, 5, 2)
    assert model.parameter_count() > 0


def test_five_state_residual_enters_rk4_rollout():
    class ZeroResidual:
        state_dim = 5
        control_dim = 2

        def derivative(self, state, control, time=None):
            del control, time
            return np.zeros_like(state)

    dynamics = ResidualPrediction(DynamicUnicyclePrediction(), ZeroResidual())
    state = np.zeros((3, 5))
    control = np.asarray(((0.2, 0.1),) * 3)
    following = integrate_batch(dynamics, state, control, 0.1, dynamic_unicycle_state(), "rk4")
    assert following.shape == (3, 5)
    assert np.all(following[:, 3] > 0.0)
