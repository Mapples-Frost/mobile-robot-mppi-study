import numpy as np
import pytest

from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig
from mobile_robot_mppi.rl.trainer import (
    BehaviorCloningAnchor,
    BehaviorCloningAnchorConfig,
)


def _anchor(seed=17):
    anchor = object.__new__(BehaviorCloningAnchor)
    anchor.enabled = True
    anchor.config = BehaviorCloningAnchorConfig(
        enabled=True,
        dataset_dir="unused",
        dataset_format="recovery_retention_v1",
        batch_size=8,
        source_kind_balanced=True,
        recovery_action_mean_weights=(1.0, 1.0),
        source_action_mean_weights=(0.0, 1.0),
    )
    anchor.observations = np.arange(48, dtype=np.float32).reshape(16, 3)
    anchor.actions = np.zeros((16, 2), dtype=np.float32)
    anchor.source_kinds = np.asarray([0] * 8 + [1] * 8, dtype=np.int8)
    anchor.rng = np.random.RandomState(seed)
    return anchor


def test_l281_balanced_sampler_is_exact_and_resume_deterministic():
    anchor = _anchor()
    normalizer = RunningNormalizer(3)
    state = anchor.rng.get_state()
    first = anchor.sample(normalizer)
    anchor.rng.set_state(state)
    second = anchor.sample(normalizer)
    np.testing.assert_array_equal(first["source_kinds"], second["source_kinds"])
    np.testing.assert_array_equal(first["observations"], second["observations"])
    assert np.sum(first["source_kinds"] == 0) == 4
    assert np.sum(first["source_kinds"] == 1) == 4
    np.testing.assert_array_equal(
        first["action_mean_weights"][first["source_kinds"] == 0],
        np.tile((1.0, 1.0), (4, 1)),
    )
    np.testing.assert_array_equal(
        first["action_mean_weights"][first["source_kinds"] == 1],
        np.tile((0.0, 1.0), (4, 1)),
    )


def test_l281_weighted_bc_loss_excludes_masked_velocity_component():
    agent = SACAgent(3, 2, SACConfig(hidden_sizes=(8,)), device="cpu", seed=5)
    for parameter in agent.actor.parameters():
        parameter.data.zero_()
    observations = np.zeros((2, 3), dtype=np.float32)
    actions = np.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=np.float32)
    full, _, _ = agent._behavior_cloning_losses(observations, actions, -2.0)
    masked, _, _ = agent._behavior_cloning_losses(
        observations,
        actions,
        -2.0,
        np.asarray(((0.0, 1.0), (0.0, 1.0)), dtype=np.float32),
    )
    assert float(full.detach()) == pytest.approx(0.5)
    assert float(masked.detach()) == pytest.approx(0.0)

    angular_actions = np.asarray(((0.0, 1.0), (0.0, 1.0)), dtype=np.float32)
    angular_full, _, _ = agent._behavior_cloning_losses(
        observations, angular_actions, -2.0
    )
    angular_masked, _, _ = agent._behavior_cloning_losses(
        observations,
        angular_actions,
        -2.0,
        np.asarray(((0.0, 1.0), (0.0, 1.0)), dtype=np.float32),
    )
    assert float(angular_masked.detach()) == pytest.approx(
        float(angular_full.detach())
    )


def test_l281_component_weights_fail_closed_without_balancing():
    config = BehaviorCloningAnchorConfig(
        enabled=True,
        dataset_dir="unused",
        dataset_format="recovery_retention_v1",
        recovery_action_mean_weights=(1.0, 1.0),
        source_action_mean_weights=(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="source-kind balancing"):
        config.validate()
