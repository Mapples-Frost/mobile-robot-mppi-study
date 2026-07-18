import numpy as np
import pytest
import torch

from mobile_robot_mppi.learning.trainer import rollout_step_weights, state_mse


def test_state_mse_supports_positive_component_weights():
    predicted = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.zeros_like(predicted)
    unweighted = state_mse(predicted, target)
    weighted = state_mse(predicted, target, [4.0, 1.0, 1.0, 1.0, 1.0])
    assert weighted > unweighted


def test_linear_rollout_weights_reach_frozen_terminal_ratio():
    values = rollout_step_weights(
        {"rollout_step_weighting": "linear", "rollout_terminal_weight": 4.0}, 36
    )
    assert values.shape == (36,)
    assert np.isclose(values[0], 1.0)
    assert np.isclose(values[-1], 4.0)


def test_invalid_state_weights_fail_closed():
    with pytest.raises(ValueError, match="positive"):
        state_mse(torch.zeros((1, 5)), torch.zeros((1, 5)), [1, 1, 0, 1, 1])
