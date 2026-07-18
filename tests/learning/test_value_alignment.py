import numpy as np
import torch

from mobile_robot_mppi.learning.models import ResidualNetwork
from mobile_robot_mppi.learning.value_alignment import (
    FrozenDirectSACValue,
    ValueAlignedResidualObjective,
)
from mobile_robot_mppi.rl.observation import ObservationEncoderConfig
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


def _value_model():
    config = SACConfig(
        hidden_sizes=(16,),
        critic_distribution="quantile",
        critic_num_quantiles=5,
    )
    agent = SACAgent(16, 2, config, seed=3)
    normalizer = {
        "count": 2,
        "mean": np.zeros(16, dtype=np.float32),
        "m2": np.ones(16, dtype=np.float32),
        "min_std": 0.05,
        "clip": 10.0,
    }
    encoder = ObservationEncoderConfig(
        lidar_sectors=4,
        include_previous_action=True,
        include_safety_state=True,
    )
    return FrozenDirectSACValue(agent, encoder, normalizer)


def _residual_model():
    statistics = {
        "feature_mean": np.zeros(6, dtype=np.float32),
        "feature_scale": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(2, dtype=np.float32),
        "control_scale": np.ones(2, dtype=np.float32),
        "residual_mean": np.zeros(5, dtype=np.float32),
        "residual_scale": np.ones(5, dtype=np.float32),
    }
    return ResidualNetwork(
        5,
        2,
        {
            "type": "icode_residual",
            "angle_indices": [2],
            "hidden_sizes": [16],
            "activation": "softplus",
            "residual_output_mask": [0, 0, 0, 1, 1],
        },
        statistics,
    )


def test_frozen_value_reencodes_physical_state_and_preserves_context():
    value = _value_model()
    raw = torch.zeros((2, 16), dtype=torch.float32)
    raw[:, 8:] = torch.arange(8, dtype=torch.float32)
    state = torch.tensor(
        [[0.0, 0.0, 0.0, 0.2, -0.1], [1.0, 2.0, 0.5, 0.1, 0.2]]
    )
    target = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
    encoded = value.raw_observation_for_state(state, raw, target)
    torch.testing.assert_close(encoded[:, 8:], raw[:, 8:])
    torch.testing.assert_close(encoded[0, :4], torch.tensor([0.2, 0.0, 0.2, 0.0]))
    assert encoded.shape == raw.shape
    assert torch.isfinite(value.value_from_state(state, raw, target)).all()


def test_value_alignment_gradient_reaches_residual_but_not_actor_or_critic():
    model = _residual_model()
    value = _value_model()
    objective = ValueAlignedResidualObjective(
        model,
        value,
        {
            "velocity_time_constant": 0.18,
            "yaw_time_constant": 0.12,
            "integrator": "rk4",
        },
        derivative_weight=0.0,
        one_step_weight=0.0,
        multistep_weight=0.0,
        value_weight=1.0,
        anchor_weight=0.0,
        value_scale=1.0,
        horizon_weights=(1.0, 2.0),
    )
    batch_size, horizon = 3, 2
    initial = torch.zeros((batch_size, 5))
    controls = torch.tensor([[[0.2, 0.1], [0.2, -0.1]]] * batch_size)
    states_t = torch.zeros((batch_size, horizon, 5))
    targets = torch.zeros((batch_size, horizon, 5))
    targets[:, :, 0] = torch.tensor([0.01, 0.03])
    raw = torch.zeros((batch_size, horizon, 16))
    raw[..., 8:] = 0.5
    target_positions = torch.ones((batch_size, horizon, 2))
    output = objective(
        model,
        {
            "initial_state": initial,
            "states_t": states_t,
            "controls": controls,
            "dt": torch.full((batch_size, horizon), 0.1),
            "target_states": targets,
            "residual_targets": torch.zeros_like(targets),
            "raw_observations": raw,
            "target_positions": target_positions,
        },
    )
    output["total"].backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.parameters()
    )
    assert all(parameter.grad is None for parameter in value.parameters())
    assert torch.isfinite(output["value_rmse"])


def test_anchor_loss_is_zero_for_unchanged_checkpoint():
    model = _residual_model()
    value = _value_model()
    objective = ValueAlignedResidualObjective(
        model,
        value,
        {
            "velocity_time_constant": 0.18,
            "yaw_time_constant": 0.12,
            "integrator": "rk4",
        },
    )
    batch = {
        "initial_state": torch.zeros((2, 5)),
        "states_t": torch.zeros((2, 1, 5)),
        "controls": torch.zeros((2, 1, 2)),
        "dt": torch.full((2, 1), 0.1),
        "target_states": torch.zeros((2, 1, 5)),
        "residual_targets": torch.zeros((2, 1, 5)),
        "raw_observations": torch.zeros((2, 1, 16)),
        "target_positions": torch.ones((2, 1, 2)),
    }
    output = objective(model, batch)
    torch.testing.assert_close(output["anchor"], torch.zeros(()))
