import numpy as np
import torch

from test_bc_initialization import _save_source, _target_trainer
from mobile_robot_mppi.rl.sac import SACConfig


def _assert_module_equal(actual, expected):
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected.state_dict()[name].cpu())


def test_full_agent_initialization_imports_parameters_but_not_optimizers(tmp_path):
    source_items = _save_source(tmp_path)
    checkpoint, source, source_normalizer, _, _, _ = source_items
    trainer = _target_trainer(source_items)

    trainer.initialize_agent_from(checkpoint)

    _assert_module_equal(trainer.agent.actor, source.actor)
    _assert_module_equal(trainer.agent.critic1, source.critic1)
    _assert_module_equal(trainer.agent.critic2, source.critic2)
    _assert_module_equal(trainer.agent.target_critic1, source.target_critic1)
    _assert_module_equal(trainer.agent.target_critic2, source.target_critic2)
    torch.testing.assert_close(trainer.agent.log_alpha.cpu(), source.log_alpha.cpu())
    assert trainer.agent.actor_optimizer.state_dict()["state"] == {}
    assert trainer.agent.critic1_optimizer.state_dict()["state"] == {}
    assert trainer.agent.critic2_optimizer.state_dict()["state"] == {}
    assert trainer.agent.alpha_optimizer.state_dict()["state"] == {}
    assert trainer.agent.update_steps == 0
    assert trainer.agent.bc_update_steps == 0
    assert trainer.normalizer.count == source_normalizer.count
    np.testing.assert_allclose(trainer.normalizer.mean, source_normalizer.mean)
    assert trainer.actor_initialization["mode"] == (
        "full_agent_parameters_and_normalizer_only"
    )
    assert trainer.actor_initialization["optimizer_state_imported"] is False
    assert trainer.actor_initialization["replay_imported"] is False
    assert trainer.actor_initialization["training_counters_reset"] is True


def test_full_agent_initialization_fails_closed_on_model_contract(tmp_path):
    source_items = _save_source(tmp_path)
    trainer = _target_trainer(
        source_items,
        SACConfig(hidden_sizes=(8, 8), activation="relu", actor_lr=6e-4),
    )

    try:
        trainer.initialize_agent_from(source_items[0])
    except ValueError as error:
        assert "SAC config mismatch" in str(error)
    else:
        raise AssertionError("full-agent mismatch did not fail closed")
