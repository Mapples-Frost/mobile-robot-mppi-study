import copy

import numpy as np
import pytest
import torch

from mobile_robot_mppi.rl.parameterization import (
    PriorParameterizationConfig,
    decode_local_subgoal_action,
    encode_local_subgoal_action,
)
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


def _snapshot(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _assert_unchanged(module, snapshot):
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), snapshot[name])


def test_behavior_cloning_reduces_mean_error_without_touching_critics_or_alpha():
    torch.set_num_threads(1)
    agent = SACAgent(
        3,
        2,
        SACConfig(hidden_sizes=(16, 16), actor_lr=3e-3),
        device="cpu",
        seed=12,
    )
    rng = np.random.RandomState(4)
    observations = rng.normal(size=(64, 3)).astype(np.float32)
    targets = np.tanh(np.stack((
        observations[:, 0] - 0.5 * observations[:, 1],
        observations[:, 2] + 0.25 * observations[:, 0],
    ), axis=1)).astype(np.float32)
    with torch.no_grad():
        initial = agent.actor.mean_action(torch.as_tensor(observations)).numpy()
    initial_mse = float(np.mean(np.square(initial - targets)))
    critic1 = _snapshot(agent.critic1)
    critic2 = _snapshot(agent.critic2)
    targets1 = _snapshot(agent.target_critic1)
    alpha = agent.log_alpha.detach().cpu().clone()

    for _ in range(200):
        metrics = agent.behavior_cloning_update(
            observations, targets, log_std_weight=1e-3, target_log_std=-2.0
        )

    with torch.no_grad():
        final = agent.actor.mean_action(torch.as_tensor(observations)).numpy()
    final_mse = float(np.mean(np.square(final - targets)))
    assert final_mse < 0.15 * initial_mse
    assert metrics["bc_mean_rmse"] >= 0.0
    assert agent.bc_update_steps == 200
    _assert_unchanged(agent.critic1, critic1)
    _assert_unchanged(agent.critic2, critic2)
    _assert_unchanged(agent.target_critic1, targets1)
    torch.testing.assert_close(agent.log_alpha.detach().cpu(), alpha)


@pytest.mark.parametrize(
    "observations,actions,match",
    [
        (np.zeros((2, 4)), np.zeros((2, 2)), "observations"),
        (np.zeros((2, 3)), np.zeros((2, 3)), "expert actions"),
        (np.full((2, 3), np.nan), np.zeros((2, 2)), "observations"),
        (np.zeros((2, 3)), np.full((2, 2), 1.2), r"\[-1, 1\]"),
    ],
)
def test_behavior_cloning_rejects_invalid_batches(observations, actions, match):
    agent = SACAgent(3, 2, SACConfig(hidden_sizes=(8,)), device="cpu")
    with pytest.raises(ValueError, match=match):
        agent.behavior_cloning_update(observations, actions)


def test_behavior_cloning_state_roundtrip_and_legacy_default():
    agent = SACAgent(3, 2, SACConfig(hidden_sizes=(8,)), device="cpu", seed=2)
    agent.behavior_cloning_update(
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    restored = SACAgent(3, 2, SACConfig(hidden_sizes=(8,)), device="cpu", seed=3)
    restored.load_state_dict(agent.state_dict(), load_optimizers=True)
    assert restored.bc_update_steps == 1
    legacy = copy.deepcopy(agent.state_dict())
    legacy.pop("bc_update_steps")
    restored.load_state_dict(legacy, load_optimizers=False)
    assert restored.bc_update_steps == 0


def test_sac_update_can_apply_behavior_anchor_and_records_it():
    torch.set_num_threads(1)
    agent = SACAgent(
        3, 2, SACConfig(hidden_sizes=(8, 8)), device="cpu", seed=5
    )
    batch = {
        "observations": np.zeros((8, 3), dtype=np.float32),
        "actions": np.zeros((8, 2), dtype=np.float32),
        "rewards": np.zeros((8, 1), dtype=np.float32),
        "next_observations": np.zeros((8, 3), dtype=np.float32),
        "dones": np.zeros((8, 1), dtype=np.float32),
    }
    behavior = {
        "observations": np.ones((4, 3), dtype=np.float32),
        "actions": np.asarray([[0.5, -0.5]] * 4, dtype=np.float32),
    }

    metrics = agent.update(
        batch,
        behavior_batch=behavior,
        behavior_cloning_weight=5.0,
        behavior_log_std_weight=1e-3,
        behavior_target_log_std=-2.0,
    )

    assert metrics["bc_anchor_mean_mse"] > 0.0
    assert metrics["bc_anchor_log_std_loss"] >= 0.0
    assert agent.bc_anchor_update_steps == 1
    state = agent.state_dict()
    assert state["bc_anchor_update_steps"] == 1
    with pytest.raises(ValueError, match="require behavior_batch"):
        agent.update(batch, behavior_cloning_weight=1.0)


def test_critic_burn_in_keeps_actor_and_entropy_temperature_frozen():
    agent = SACAgent(
        3, 2, SACConfig(hidden_sizes=(8, 8)), device="cpu", seed=10
    )
    actor = _snapshot(agent.actor)
    critic = _snapshot(agent.critic1)
    alpha = agent.log_alpha.detach().cpu().clone()
    batch = {
        "observations": np.zeros((8, 3), dtype=np.float32),
        "actions": np.zeros((8, 2), dtype=np.float32),
        "rewards": np.ones((8, 1), dtype=np.float32),
        "next_observations": np.zeros((8, 3), dtype=np.float32),
        "dones": np.zeros((8, 1), dtype=np.float32),
    }

    metrics = agent.update(batch, update_actor=False)

    _assert_unchanged(agent.actor, actor)
    torch.testing.assert_close(agent.log_alpha.detach().cpu(), alpha)
    assert any(
        not torch.equal(value.detach().cpu(), critic[name])
        for name, value in agent.critic1.state_dict().items()
    )
    assert metrics["actor_update_applied"] == 0.0
    assert metrics["actor_gradient_norm"] == 0.0


def test_local_subgoal_action_roundtrip_boundaries_and_angle_wrap():
    config = PriorParameterizationConfig(
        kind="local_subgoal",
        learn_covariance=False,
        subgoal_min_distance=0.25,
        subgoal_max_distance=1.25,
        subgoal_max_bearing=np.pi,
    )
    action = encode_local_subgoal_action(config, 0.75, 0.4)
    distance, bearing = decode_local_subgoal_action(config, action)
    assert distance == pytest.approx(0.75)
    assert bearing == pytest.approx(0.4, abs=1e-6)
    np.testing.assert_allclose(
        encode_local_subgoal_action(config, -10.0, 0.0), [-1.0, 0.0]
    )
    np.testing.assert_allclose(
        encode_local_subgoal_action(config, 10.0, 0.0), [1.0, 0.0]
    )
    wrapped = encode_local_subgoal_action(config, 0.75, 2.0 * np.pi + 0.3)
    assert wrapped[1] == pytest.approx(0.3 / np.pi, abs=1e-6)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        decode_local_subgoal_action(config, [1.1, 0.0])


def test_local_subgoal_conversion_rejects_nonlocal_or_covariance_actions():
    with pytest.raises(ValueError, match="kind=local_subgoal"):
        encode_local_subgoal_action(PriorParameterizationConfig(), 0.5, 0.0)
    with pytest.raises(ValueError, match="learn_covariance=false"):
        encode_local_subgoal_action(
            PriorParameterizationConfig(
                kind="local_subgoal", learn_covariance=True
            ),
            0.5,
            0.0,
        )
