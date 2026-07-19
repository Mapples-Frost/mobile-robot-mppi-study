import numpy as np
import torch

from experiments.icode.train_value_aligned_icode import (
    _calibrate_value_competence,
    _episode_bootstrap_windows,
)
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


def _path_value_model():
    config = SACConfig(
        hidden_sizes=(16,),
        critic_distribution="quantile",
        critic_num_quantiles=5,
    )
    agent = SACAgent(22, 2, config, seed=4)
    normalizer = {
        "count": 2,
        "mean": np.zeros(22, dtype=np.float32),
        "m2": np.ones(22, dtype=np.float32),
        "min_std": 0.05,
        "clip": 10.0,
    }
    encoder = ObservationEncoderConfig(
        lidar_sectors=4,
        include_previous_action=True,
        include_safety_state=True,
        include_path_context=True,
        path_cross_track_scale=0.75,
        path_remaining_scale=7.0,
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


def test_path_conditioned_value_reencodes_local_route_features():
    value = _path_value_model()
    raw = torch.zeros((1, 22), dtype=torch.float32)
    reference = torch.tensor([[1.0, 2.0, 0.2, 0.1, 0.0]])
    predicted = reference.clone()
    predicted[:, 1] += 0.15
    predicted[:, 2] += 0.1
    target = torch.tensor([[3.0, 2.0]])
    path = torch.tensor([[0.2, 0.0, 1.0, 0.1, 0.5, 1.0]])
    encoded = value.raw_observation_for_state(
        predicted,
        raw,
        target,
        reference_state=reference,
        path_context_template=path,
    )
    path_start = 8 + 2 + 1
    assert encoded[0, path_start] > path[0, 0]
    torch.testing.assert_close(
        encoded[0, path_start + 1], torch.sin(torch.tensor(0.1))
    )
    torch.testing.assert_close(
        encoded[0, path_start + 2], torch.cos(torch.tensor(0.1))
    )
    torch.testing.assert_close(
        encoded[0, path_start + 3], path[0, 3]
    )
    assert encoded[0, path_start + 5] == 1.0


def test_path_conditioned_value_requires_explicit_route_context():
    value = _path_value_model()
    state = torch.zeros((1, 5))
    raw = torch.zeros((1, 22))
    target = torch.ones((1, 2))
    with np.testing.assert_raises_regex(
        ValueError, "path-conditioned value requires"
    ):
        value.value_from_state(state, raw, target)


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
    assert torch.isfinite(output["value_ranking"])
    assert 0.0 <= float(output["value_ranking_pair_fraction"]) <= 1.0


def test_pairwise_value_ranking_is_finite_and_backpropagates_when_informative():
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
        value_weight=0.0,
        value_ranking_weight=1.0,
        value_ranking_margin=0.0,
        anchor_weight=0.0,
    )
    batch_size, horizon = 4, 2
    raw = torch.zeros((batch_size, horizon, 16))
    raw[:, :, 0] = torch.tensor((-1.0, -0.2, 0.4, 1.0))[:, None]
    batch = {
        "initial_state": torch.zeros((batch_size, 5)),
        "states_t": torch.zeros((batch_size, horizon, 5)),
        "controls": torch.tensor([[[0.2, 0.1], [0.3, -0.1]]] * batch_size),
        "dt": torch.full((batch_size, horizon), 0.1),
        "target_states": torch.zeros((batch_size, horizon, 5)),
        "residual_targets": torch.zeros((batch_size, horizon, 5)),
        "raw_observations": raw,
        "target_positions": torch.tensor([[[1.0, 0.0], [1.2, 0.2]]] * batch_size),
    }

    output = objective(model, batch)
    output["total"].backward()

    assert torch.isfinite(output["value_ranking"])
    assert torch.isfinite(output["value_ranking_accuracy"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_critic_competence_removes_low_value_sample_authority():
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
        competence_off_value=1e6,
        competence_on_value=1e6 + 1.0,
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

    torch.testing.assert_close(
        output["competence_mean"], torch.zeros(())
    )
    torch.testing.assert_close(output["value"], torch.zeros(()))


def test_competence_calibration_uses_episode_outcome_groups():
    class _Value:
        @staticmethod
        def value_from_raw(raw):
            return raw[:, 0]

    dataset = {
        "episode_id": np.asarray(["fail", "fail", "win", "win"]),
        "raw_observation_t_plus_1": np.asarray(
            [[1.0], [3.0], [7.0], [11.0]], dtype=np.float32
        ),
    }
    result = _calibrate_value_competence(
        _Value(),
        dataset,
        {"fail": False, "win": True},
        torch.device("cpu"),
    )

    assert result["off_value"] == 2.0
    assert result["on_value"] == 9.0
    assert result["success_episodes"] == 1
    assert result["failure_episodes"] == 1


def test_episode_bootstrap_resamples_whole_episode_window_clusters():
    dataset = {
        "episode_id": np.asarray(
            ["a", "a", "a", "b", "b", "c", "c", "c"]
        )
    }
    starts = np.arange(8)

    selected, manifest = _episode_bootstrap_windows(
        dataset, starts, seed=17
    )

    assert manifest["independent_unit"] == "episode"
    assert manifest["draw_count"] == 3
    for episode_id, count in manifest["selection_counts"].items():
        expected = int(np.sum(dataset["episode_id"] == episode_id)) * count
        assert int(np.sum(dataset["episode_id"][selected] == episode_id)) == expected
