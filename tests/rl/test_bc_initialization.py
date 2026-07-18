import numpy as np
import pytest
import torch
from types import SimpleNamespace

from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.rl.checkpointing import save_sac_checkpoint
from mobile_robot_mppi.rl.observation import (
    ObservationEncoderConfig,
    RunningNormalizer,
)
from mobile_robot_mppi.rl.parameterization import PriorParameterizationConfig
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig
from mobile_robot_mppi.rl.trainer import SACTrainer


def _parameter_snapshot(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _assert_snapshot_equal(module, snapshot):
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), snapshot[name])


def _save_source(tmp_path, encoder=None):
    config = SACConfig(hidden_sizes=(8, 8), activation="relu")
    source = SACAgent(5, 2, config, device="cpu", seed=8)
    with torch.no_grad():
        for parameter in source.actor.parameters():
            parameter.add_(0.25)
    normalizer = RunningNormalizer(5)
    normalizer.update(np.arange(20, dtype=np.float32).reshape(4, 5))
    encoder = encoder or ObservationEncoderConfig(
        lidar_sectors=1,
        include_previous_action=False,
        include_safety_state=False,
    )
    prior = PriorParameterizationConfig(
        kind="local_subgoal", num_knots=2, learn_covariance=False
    )
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    checkpoint = save_sac_checkpoint(
        tmp_path / "bc.pt",
        source,
        normalizer,
        encoder,
        prior,
        action_spec,
        {"experiment": {"name": "bc_unit"}},
        tmp_path,
        {"phase": "behavior_cloning", "bc_epoch": 3, "global_step": 0},
    )
    return checkpoint, source, normalizer, encoder, prior, action_spec


def _target_trainer(source_items, sac_config=None):
    _, _, _, encoder, prior, action_spec = source_items
    trainer = SACTrainer.__new__(SACTrainer)
    trainer.agent = SACAgent(
        5,
        2,
        sac_config or SACConfig(hidden_sizes=(8, 8), activation="relu"),
        device="cpu",
        seed=99,
    )
    trainer.normalizer = RunningNormalizer(5)
    trainer.encoder_config = encoder
    trainer.parameterization_config = prior
    trainer.action_spec = action_spec
    trainer.actor_initialization = None
    trainer.bc_anchor = SimpleNamespace(enabled=False)
    trainer._write_run_metadata = lambda: None
    return trainer


def test_actor_only_initialization_does_not_import_sac_state(tmp_path):
    source_items = _save_source(tmp_path)
    checkpoint, source, source_normalizer, _, _, _ = source_items
    trainer = _target_trainer(source_items)
    critic_before = _parameter_snapshot(trainer.agent.critic1)
    target_before = _parameter_snapshot(trainer.agent.target_critic1)
    alpha_before = trainer.agent.log_alpha.detach().cpu().clone()

    trainer.initialize_actor_from(checkpoint)

    for name, value in trainer.agent.actor.state_dict().items():
        torch.testing.assert_close(value.cpu(), source.actor.state_dict()[name].cpu())
    _assert_snapshot_equal(trainer.agent.critic1, critic_before)
    _assert_snapshot_equal(trainer.agent.target_critic1, target_before)
    torch.testing.assert_close(trainer.agent.log_alpha.detach().cpu(), alpha_before)
    assert trainer.agent.actor_optimizer.state_dict()["state"] == {}
    assert trainer.agent.update_steps == 0
    assert trainer.normalizer.count == source_normalizer.count
    np.testing.assert_allclose(trainer.normalizer.mean, source_normalizer.mean)
    assert trainer.actor_initialization["mode"] == "actor_and_normalizer_only"
    assert trainer.actor_initialization["source_phase"] == "behavior_cloning"


def test_actor_only_initialization_fails_closed_on_contract_mismatch(tmp_path):
    source_items = _save_source(tmp_path)
    trainer = _target_trainer(source_items)
    trainer.encoder_config = ObservationEncoderConfig(
        lidar_sectors=2,
        include_previous_action=False,
        include_safety_state=False,
    )

    with pytest.raises(ValueError, match="encoder_config"):
        trainer.initialize_actor_from(source_items[0])


def test_correction_initialization_freezes_bc_and_keeps_zero_residual(tmp_path):
    source_items = _save_source(tmp_path)
    checkpoint, source, source_normalizer, _, _, _ = source_items
    trainer = _target_trainer(
        source_items,
        SACConfig(
            hidden_sizes=(8, 8),
            activation="relu",
            policy_mode="frozen_bc_correction",
            correction_scale=(0.2, 0.1),
            correction_gate_alpha=0.5,
        ),
    )
    correction_before = _parameter_snapshot(trainer.agent.actor)
    critic_before = _parameter_snapshot(trainer.agent.critic1)

    trainer.initialize_actor_from(checkpoint)

    for name, value in trainer.agent.base_actor.state_dict().items():
        torch.testing.assert_close(
            value.cpu(), source.actor.state_dict()[name].cpu(), rtol=0.0, atol=0.0
        )
    assert all(
        not parameter.requires_grad
        for parameter in trainer.agent.base_actor.parameters()
    )
    _assert_snapshot_equal(trainer.agent.critic1, critic_before)
    assert trainer.agent.actor_optimizer.state_dict()["state"] == {}
    assert trainer.normalizer.count == source_normalizer.count
    assert trainer.actor_initialization["mode"] == (
        "frozen_bc_base_and_normalizer_only"
    )

    observation = np.linspace(-0.3, 0.3, 5, dtype=np.float32)
    expected, _ = source.select_action(observation, deterministic=True)
    actual, diagnostics = trainer.agent.select_action(
        observation, deterministic=True
    )
    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["applied_correction_abs_max"] == 0.0

    # Initialization deliberately resets only the correction output layer;
    # hidden features remain the fresh target run's seeded initialization.
    for name, value in trainer.agent.actor.state_dict().items():
        if name not in ("network.4.weight", "network.4.bias"):
            torch.testing.assert_close(value.cpu(), correction_before[name])
