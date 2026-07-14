import numpy as np
import pytest
import torch

from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint, save_sac_checkpoint
from mobile_robot_mppi.rl.observation import ObservationEncoderConfig, RunningNormalizer
from mobile_robot_mppi.rl.parameterization import PriorParameterizationConfig
from mobile_robot_mppi.rl.replay import ReplayBuffer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig
from mobile_robot_mppi.rl.trainer import (
    TrainingConfig,
    _validation_episode_seed,
)


def test_replay_buffer_sampling_and_resume():
    replay = ReplayBuffer(16, 3, 2, seed=4)
    for index in range(8):
        replay.add(
            np.full(3, index),
            np.full(2, 0.1 * index),
            float(index),
            np.full(3, index + 1),
            index == 7,
            group=index % 2,
        )
    batch = replay.sample(4)
    assert batch["observations"].shape == (4, 3)
    restored = ReplayBuffer.from_state_dict(replay.state_dict())
    assert restored.size == replay.size
    np.testing.assert_allclose(restored.observations[:8], replay.observations[:8])
    np.testing.assert_array_equal(restored.groups[:8], replay.groups[:8])


def test_scene_balanced_replay_is_equal_despite_imbalanced_storage():
    replay = ReplayBuffer(32, 2, 1, seed=9)
    for index in range(16):
        group = 0 if index < 12 else 1
        replay.add(
            np.asarray((index, index + 1)),
            np.asarray((0.0,)),
            0.0,
            np.asarray((index + 1, index + 2)),
            False,
            group=group,
        )
    assert replay.group_counts() == {0: 12, 1: 4}
    batch = replay.sample(10, strategy="scene_balanced")
    groups, counts = np.unique(batch["groups"], return_counts=True)
    assert dict(zip(groups.tolist(), counts.tolist())) == {0: 5, 1: 5}
    with pytest.raises(ValueError, match="strategy"):
        replay.sample(4, strategy="unknown")


def test_legacy_replay_checkpoint_restores_as_single_group():
    replay = ReplayBuffer(8, 2, 1, seed=3)
    replay.add(np.zeros(2), np.zeros(1), 0.0, np.ones(2), False)
    legacy = replay.state_dict()
    legacy.pop("groups")
    legacy.pop("outcomes")
    legacy.pop("transition_ids")
    legacy.pop("next_transition_id")
    restored = ReplayBuffer.from_state_dict(legacy)
    assert restored.group_counts() == {0: 1}
    assert restored.outcome_counts() == {0: 1}


def test_outcome_balanced_replay_preserves_rare_success_transitions():
    replay = ReplayBuffer(64, 2, 1, seed=14)
    failure_handles = []
    success_handles = []
    for index in range(40):
        handle = replay.add(
            np.asarray((index, index + 1)),
            np.asarray((0.0,)),
            0.0,
            np.asarray((index + 1, index + 2)),
            False,
        )
        (success_handles if index >= 36 else failure_handles).append(handle)
    assert replay.mark_episode_outcome(failure_handles, False) == 36
    assert replay.mark_episode_outcome(success_handles, True) == 4
    assert replay.outcome_counts() == {0: 36, 1: 4}

    batch = replay.sample(
        20, strategy="outcome_balanced", success_fraction=0.4
    )
    assert int(np.sum(batch["outcomes"] == 1)) == 8
    assert int(np.sum(batch["outcomes"] == 0)) == 12


def test_episode_outcome_handle_does_not_relabel_overwritten_transition():
    replay = ReplayBuffer(2, 1, 1, seed=2)
    stale = replay.add(np.zeros(1), np.zeros(1), 0.0, np.ones(1), False)
    replay.add(np.ones(1), np.zeros(1), 0.0, np.ones(1), False)
    replay.add(np.full(1, 2.0), np.zeros(1), 0.0, np.ones(1), False)

    assert replay.mark_episode_outcome([stale], True) == 0
    assert replay.outcome_counts() == {-1: 2}


def test_training_config_validates_replay_strategy():
    TrainingConfig(replay_sampling="scene_balanced").validate()
    TrainingConfig(
        replay_sampling="outcome_balanced", replay_success_fraction=0.4
    ).validate()
    with pytest.raises(ValueError, match="replay_sampling"):
        TrainingConfig(replay_sampling="prioritized_magic").validate()
    with pytest.raises(ValueError, match="replay_success_fraction"):
        TrainingConfig(replay_success_fraction=1.1).validate()
    with pytest.raises(ValueError, match="validation_seed_base"):
        TrainingConfig(validation_seed_base=-1).validate()
    TrainingConfig(normalizer_update="frozen").validate()
    with pytest.raises(ValueError, match="normalizer_update"):
        TrainingConfig(normalizer_update="sometimes").validate()
    TrainingConfig(warmup_policy="actor").validate()
    with pytest.raises(ValueError, match="warmup_policy"):
        TrainingConfig(warmup_policy="expert_oracle").validate()


def test_fixed_validation_seed_is_independent_of_training_seed():
    first = TrainingConfig(seed=11, validation_seed_base=7000)
    second = TrainingConfig(seed=99, validation_seed_base=7000)
    assert _validation_episode_seed(first, 0, 0) == 7000
    assert _validation_episode_seed(second, 0, 0) == 7000
    assert _validation_episode_seed(first, 2, 3) == 9003


def test_missing_validation_seed_base_preserves_legacy_schedule():
    config = TrainingConfig(seed=123)
    assert _validation_episode_seed(config, 1, 4) == 101127


def test_sac_update_is_finite_and_checkpoint_is_complete(tmp_path):
    torch.set_num_threads(1)
    config = SACConfig(hidden_sizes=(16, 16), gradient_clip_norm=5.0)
    agent = SACAgent(5, 4, config, device="cpu", seed=2)
    rng = np.random.RandomState(3)
    batch = {
        "observations": rng.normal(size=(8, 5)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(8, 4)).astype(np.float32),
        "rewards": rng.normal(size=(8, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(8, 5)).astype(np.float32),
        "dones": np.zeros((8, 1), dtype=np.float32),
    }
    metrics = agent.update(batch)
    assert np.isfinite(tuple(metrics.values())).all()
    action, diagnostics = agent.select_action(np.zeros(5, dtype=np.float32), deterministic=True)
    assert action.shape == (4,)
    assert np.all(np.abs(action) <= 1.0)
    assert diagnostics["alpha"] > 0.0
    normalizer = RunningNormalizer(5)
    normalizer.update(batch["observations"])
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    checkpoint = save_sac_checkpoint(
        tmp_path / "policy.pt",
        agent,
        normalizer,
        ObservationEncoderConfig(lidar_sectors=1, include_previous_action=False, include_safety_state=False),
        PriorParameterizationConfig(num_knots=2),
        action_spec,
        {"experiment": {"name": "unit"}},
        tmp_path,
        {"global_step": 8, "episodes": 1},
    )
    payload = load_sac_checkpoint(checkpoint)
    assert payload["agent"]["update_steps"] == 1
    assert payload["normalizer"]["count"] == 8
    assert payload["training_state"]["global_step"] == 8


def test_automatic_entropy_tuning_respects_configured_alpha_floor():
    torch.set_num_threads(1)
    agent = SACAgent(
        3,
        2,
        SACConfig(hidden_sizes=(8,), minimum_alpha=0.05),
        device="cpu",
        seed=7,
    )
    with torch.no_grad():
        agent.log_alpha.fill_(float(np.log(1e-4)))
    batch = {
        "observations": np.zeros((4, 3), dtype=np.float32),
        "actions": np.zeros((4, 2), dtype=np.float32),
        "rewards": np.zeros((4, 1), dtype=np.float32),
        "next_observations": np.zeros((4, 3), dtype=np.float32),
        "dones": np.zeros((4, 1), dtype=np.float32),
    }
    agent.update(batch)
    assert float(agent.alpha.detach().cpu()) >= 0.05 - 1e-7
