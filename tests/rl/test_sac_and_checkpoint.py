import copy

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint, save_sac_checkpoint
from mobile_robot_mppi.rl.observation import ObservationEncoderConfig, RunningNormalizer
from mobile_robot_mppi.rl.parameterization import PriorParameterizationConfig
from mobile_robot_mppi.rl.replay import ReplayBuffer
from mobile_robot_mppi.rl.sac import (
    SACAgent,
    SACConfig,
    group_robust_mean,
    lower_tail_cvar,
    quantile_huber_loss,
)
from mobile_robot_mppi.rl.trainer import (
    BehaviorCloningAnchorConfig,
    CheckpointSelectionConfig,
    PrecisionConstraintConfig,
    PrecisionConstraintController,
    PrecisionConstrainedCheckpointSelector,
    SACTrainer,
    TrainingConfig,
    ValidationCheckpointSelector,
    _validation_episode_seed,
)


class _MetadataStub:
    def __init__(self, value):
        self.value = value

    def metadata(self):
        return self.value


class _AnchorStub:
    enabled = False
    manifest_fingerprint = None

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def metadata(self):
        return {
            "enabled": False,
            "dataset_dir": None,
            "dataset_manifest_sha256": None,
            "batch_size": 256,
            "mean_weight": 1.0,
            "log_std_weight": 0.0,
            "target_log_std": -2.0,
        }


def _checkpoint_trainer(tmp_path, save_replay=True, actor_update_after=0):
    trainer = SACTrainer.__new__(SACTrainer)
    trainer.output_dir = tmp_path
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.project_root = tmp_path
    trainer.resolved_config = {"experiment": {"name": "resume_unit"}}
    trainer.training_config = TrainingConfig(
        total_steps=20,
        batch_size=2,
        replay_capacity=8,
        actor_update_after=actor_update_after,
        save_replay_buffer=save_replay,
        device="cpu",
        seed=7,
    )
    trainer.checkpoint_selection_config = CheckpointSelectionConfig()
    trainer.checkpoint_selector = None
    trainer.precision_constraint = PrecisionConstraintController({})
    trainer.agent = SACAgent(
        3, 2, SACConfig(hidden_sizes=(8,)), device="cpu", seed=7
    )
    trainer.normalizer = RunningNormalizer(3)
    trainer.normalizer.update(np.asarray(((0.0, 1.0, 2.0),)))
    trainer.encoder_config = ObservationEncoderConfig(
        lidar_sectors=1,
        include_previous_action=False,
        include_safety_state=False,
    )
    trainer.parameterization_config = PriorParameterizationConfig(
        kind="local_subgoal", num_knots=2, learn_covariance=False
    )
    trainer.action_spec = body_velocity_action((0.0, 0.4), 1.0)
    scene = {"scene": {"name": "unit"}, "experiment": {"seed": 7}}
    trainer.pool = SimpleNamespace(
        configs=[scene], rng=np.random.RandomState(8)
    )
    trainer.validation_configs = [scene]
    trainer.curriculum = _MetadataStub({"enabled": False})
    trainer.initial_state_curriculum = _MetadataStub({"enabled": False})
    trainer.bc_anchor = _AnchorStub(seed=9)
    trainer.replay = ReplayBuffer(8, 3, 2, seed=10)
    trainer.replay.add(
        np.zeros(3), np.zeros(2), 0.0, np.ones(3), False
    )
    trainer.global_step = 4
    trainer.episodes = 1
    trainer.interrupted_episodes = 0
    trainer.best_validation_score = 3.0
    trainer.actor_initialization = None
    trainer.episode_records = []
    trainer.update_records = []
    trainer.validation_records = []
    trainer.rng = np.random.RandomState(11)
    trainer._episode_in_progress = True
    trainer.resume_provenance = None
    trainer._write_run_metadata = lambda: None
    return trainer


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
            constraint_cost=0.01 * index,
        )
    batch = replay.sample(4)
    assert batch["observations"].shape == (4, 3)
    restored = ReplayBuffer.from_state_dict(replay.state_dict())
    assert restored.size == replay.size
    np.testing.assert_allclose(restored.observations[:8], replay.observations[:8])
    np.testing.assert_array_equal(restored.groups[:8], replay.groups[:8])
    np.testing.assert_allclose(
        restored.constraint_costs[:8], replay.constraint_costs[:8]
    )


def test_legacy_replay_checkpoint_restores_zero_constraint_cost():
    replay = ReplayBuffer(8, 2, 1, seed=13)
    replay.add(
        np.zeros(2), np.zeros(1), 1.0, np.ones(2), False,
        constraint_cost=0.25,
    )
    legacy = replay.state_dict()
    legacy.pop("constraint_costs")
    restored = ReplayBuffer.from_state_dict(legacy)
    np.testing.assert_array_equal(restored.constraint_costs[:1], [[0.0]])


def test_replay_constraint_cost_fails_closed():
    replay = ReplayBuffer(8, 2, 1, seed=13)
    with pytest.raises(ValueError, match="non-negative"):
        replay.add(
            np.zeros(2), np.zeros(1), 0.0, np.ones(2), False,
            constraint_cost=-0.1,
        )


def test_precision_constraint_shapes_replay_and_updates_dual():
    controller = PrecisionConstraintController(PrecisionConstraintConfig(
        enabled=True,
        target_cross_track_rmse_m=0.04,
        dual_lr=100.0,
        initial_multiplier=2.0,
        maximum_multiplier=5.0,
    ))
    batch = {
        "rewards": np.asarray(((1.0,), (2.0,)), dtype=np.float32),
        "constraint_costs": np.asarray(((0.01,), (0.02,)), dtype=np.float32),
    }
    shaped = controller.shape_batch(batch)
    np.testing.assert_allclose(shaped["rewards"], ((0.98,), (1.96,)))
    diagnostics = controller.update(0.0036)
    assert diagnostics["precision_constraint_violation"] == pytest.approx(0.002)
    assert controller.multiplier == pytest.approx(2.2)
    controller.update(0.0)
    assert controller.multiplier == pytest.approx(2.04)
    restored = PrecisionConstraintController(controller.config)
    restored.load_state_dict(controller.state_dict())
    assert restored.multiplier == pytest.approx(controller.multiplier)
    assert restored.updates == 2


def test_disabled_precision_constraint_preserves_rewards_exactly():
    controller = PrecisionConstraintController({"enabled": False})
    rewards = np.asarray(((1.25,),), dtype=np.float32)
    shaped = controller.shape_batch({
        "rewards": rewards,
        "constraint_costs": np.asarray(((999.0,),), dtype=np.float32),
    })
    assert shaped["rewards"] is rewards
    controller.update(1.0)
    assert controller.multiplier == 0.0


def test_precision_checkpoint_selector_prefers_feasible_then_return():
    selector = PrecisionConstrainedCheckpointSelector({
        "mode": "precision_constrained",
        "maximum_mean_cross_track_rmse": 0.04,
    })
    base = {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "mean_return": 10.0,
        "mean_cross_track_rmse": 0.06,
    }
    first = selector.consider([], base, 0)
    assert first["selection_selected"]
    assert not first["selection_eligible"]
    less_bad = dict(base, mean_cross_track_rmse=0.05, mean_return=9.0)
    assert selector.consider([], less_bad, 100)["selection_selected"]
    feasible = dict(base, mean_cross_track_rmse=0.039, mean_return=8.0)
    decision = selector.consider([], feasible, 200)
    assert decision["selection_selected"]
    assert decision["selection_best_feasible"]
    faster = dict(feasible, mean_return=12.0, mean_cross_track_rmse=0.04)
    assert selector.consider([], faster, 300)["selection_selected"]
    infeasible_fast = dict(base, mean_cross_track_rmse=0.041, mean_return=100.0)
    assert not selector.consider([], infeasible_fast, 400)["selection_selected"]


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


def test_scene_outcome_balanced_replay_balances_both_axes():
    replay = ReplayBuffer(128, 2, 1, seed=17)
    for group, success_count, failure_count in ((0, 4, 20), (1, 12, 4)):
        success_handles = []
        failure_handles = []
        for index in range(success_count + failure_count):
            handle = replay.add(
                np.asarray((group, index), dtype=np.float32),
                np.asarray((0.0,), dtype=np.float32),
                0.0,
                np.asarray((group, index + 1), dtype=np.float32),
                False,
                group=group,
            )
            (success_handles if index < success_count else failure_handles).append(handle)
        replay.mark_episode_outcome(success_handles, True)
        replay.mark_episode_outcome(failure_handles, False)
    batch = replay.sample(
        24, strategy="scene_outcome_balanced", success_fraction=0.5
    )
    for group in (0, 1):
        selected = batch["groups"] == group
        assert int(np.sum(selected)) == 12
        assert int(np.sum(batch["outcomes"][selected] == 1)) == 6
        assert int(np.sum(batch["outcomes"][selected] == 0)) == 6


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
    TrainingConfig(
        replay_sampling="scene_outcome_balanced", replay_success_fraction=0.5
    ).validate()
    with pytest.raises(ValueError, match="replay_sampling"):
        TrainingConfig(replay_sampling="prioritized_magic").validate()
    with pytest.raises(ValueError, match="replay_success_fraction"):
        TrainingConfig(replay_success_fraction=1.1).validate()
    with pytest.raises(ValueError, match="validation_seed_base"):
        TrainingConfig(validation_seed_base=-1).validate()
    TrainingConfig(validation_initial_state_noise=(0.0, 0.0, 0.0)).validate()
    with pytest.raises(ValueError, match="validation_initial_state_noise"):
        TrainingConfig(validation_initial_state_noise=(0.0, -0.1, 0.0)).validate()
    TrainingConfig(normalizer_update="frozen").validate()
    with pytest.raises(ValueError, match="normalizer_update"):
        TrainingConfig(normalizer_update="sometimes").validate()
    TrainingConfig(warmup_policy="actor").validate()
    with pytest.raises(ValueError, match="warmup_policy"):
        TrainingConfig(warmup_policy="expert_oracle").validate()
    with pytest.raises(ValueError, match="warmup/update_after"):
        TrainingConfig(actor_update_after=-1).validate()


def test_behavior_cloning_anchor_config_fails_closed():
    BehaviorCloningAnchorConfig(
        enabled=True, dataset_dir="dataset", mean_weight=5.0
    ).validate()
    with pytest.raises(ValueError, match="dataset_dir"):
        BehaviorCloningAnchorConfig(enabled=True).validate()
    with pytest.raises(ValueError, match="mean_weight"):
        BehaviorCloningAnchorConfig(
            enabled=True, dataset_dir="dataset", mean_weight=0.0
        ).validate()


def _validation_rows(success, collision, distance, returns=None):
    if returns is None:
        returns = [-value for value in distance]
    return [
        {
            "scene": "unit_scene",
            "seed": 8000 + index,
            "success": bool(success[index]),
            "collision": bool(collision[index]),
            "goal_distance": float(distance[index]),
            "return": float(returns[index]),
        }
        for index in range(len(success))
    ]


def _validation_summary(rows):
    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "mean_goal_distance": float(
            np.mean([row["goal_distance"] for row in rows])
        ),
        "mean_return": float(np.mean([row["return"] for row in rows])),
    }


def test_paired_checkpoint_selector_rejects_success_and_collision_regressions():
    selector = ValidationCheckpointSelector(CheckpointSelectionConfig(
        mode="initial_noninferiority",
        minimum_mean_goal_distance_improvement=0.005,
    ))
    reference = _validation_rows(
        success=[True, True, False],
        collision=[False, False, False],
        distance=[0.10, 0.10, 0.40],
    )
    initial = selector.consider(reference, _validation_summary(reference), 0)
    assert initial["selection_selected"] is True

    lost_success = _validation_rows(
        success=[False, True, True],
        collision=[False, False, False],
        distance=[0.05, 0.05, 0.05],
    )
    decision = selector.consider(
        lost_success, _validation_summary(lost_success), 5000
    )
    assert decision["selection_selected"] is False
    assert decision["selection_success_losses"] == 1
    assert decision["selection_reason"] == "noninferiority_failed"

    new_collision = _validation_rows(
        success=[True, True, False],
        collision=[False, True, False],
        distance=[0.05, 0.05, 0.20],
    )
    decision = selector.consider(
        new_collision, _validation_summary(new_collision), 10000
    )
    assert decision["selection_selected"] is False
    assert decision["selection_collision_regressions"] == 1


def test_paired_checkpoint_selector_requires_meaningful_paired_improvement():
    selector = ValidationCheckpointSelector(CheckpointSelectionConfig(
        mode="initial_noninferiority",
        minimum_mean_goal_distance_improvement=0.005,
    ))
    reference = _validation_rows(
        success=[True, True, False],
        collision=[False, False, False],
        distance=[0.10, 0.10, 0.40],
    )
    selector.consider(reference, _validation_summary(reference), 0)

    negligible = _validation_rows(
        success=[True, True, False],
        collision=[False, False, False],
        distance=[0.098, 0.098, 0.398],
    )
    decision = selector.consider(
        negligible, _validation_summary(negligible), 5000
    )
    assert decision["selection_eligible"] is True
    assert decision["selection_meaningful_improvement"] is False
    assert decision["selection_selected"] is False

    improved = _validation_rows(
        success=[True, True, False],
        collision=[False, False, False],
        distance=[0.09, 0.09, 0.39],
    )
    decision = selector.consider(
        improved, _validation_summary(improved), 10000
    )
    assert decision["selection_selected"] is True
    assert decision["selection_best_global_step"] == 10000


def test_paired_checkpoint_selector_state_round_trip_and_seed_drift_guard():
    config = CheckpointSelectionConfig(mode="initial_noninferiority")
    selector = ValidationCheckpointSelector(config)
    reference = _validation_rows(
        success=[True, False],
        collision=[False, False],
        distance=[0.10, 0.30],
    )
    selector.consider(reference, _validation_summary(reference), 0)
    restored = ValidationCheckpointSelector(config)
    restored.load_state_dict(selector.state_dict())
    assert restored.state_dict() == selector.state_dict()

    changed_seed_set = copy.deepcopy(reference)
    changed_seed_set[1]["seed"] += 1
    with pytest.raises(ValueError, match="scene/seed set changed"):
        restored.consider(
            changed_seed_set, _validation_summary(changed_seed_set), 5000
        )

    with pytest.raises(ValueError, match="config does not match"):
        ValidationCheckpointSelector(CheckpointSelectionConfig(
            mode="initial_noninferiority",
            minimum_mean_goal_distance_improvement=0.01,
        )).load_state_dict(selector.state_dict())


def test_checkpoint_selection_config_fails_closed():
    CheckpointSelectionConfig(mode="initial_noninferiority").validate()
    with pytest.raises(ValueError, match="mode"):
        CheckpointSelectionConfig(mode="best_effort").validate()
    with pytest.raises(ValueError, match="non-negative integers"):
        CheckpointSelectionConfig(
            mode="initial_noninferiority", maximum_success_losses=0.5
        ).validate()
    with pytest.raises(ValueError, match="distance thresholds"):
        CheckpointSelectionConfig(
            mode="initial_noninferiority",
            maximum_mean_goal_distance_increase=-0.1,
        ).validate()


def test_fixed_validation_seed_is_independent_of_training_seed():
    first = TrainingConfig(seed=11, validation_seed_base=7000)
    second = TrainingConfig(seed=99, validation_seed_base=7000)
    assert _validation_episode_seed(first, 0, 0) == 7000
    assert _validation_episode_seed(second, 0, 0) == 7000
    assert _validation_episode_seed(first, 2, 3) == 9003


def test_missing_validation_seed_base_preserves_legacy_schedule():
    config = TrainingConfig(seed=123)
    assert _validation_episode_seed(config, 1, 4) == 101127


def test_resume_log_loader_truncates_to_checkpoint_and_csv_accepts_new_fields(
    tmp_path,
):
    trainer = SACTrainer.__new__(SACTrainer)
    path = tmp_path / "updates.csv"
    trainer._save_csv(path, [
        {"global_step": 1, "loss": 2.0},
        {"global_step": 2, "loss": 1.0, "new_metric": 3.0},
        {"global_step": 3, "loss": 0.5, "new_metric": 4.0},
    ])

    restored = trainer._load_csv_until(path, 2)

    assert [int(row["global_step"]) for row in restored] == [1, 2]
    assert restored[1]["new_metric"] == "3.0"


def test_trainer_resume_restores_replay_torch_rng_and_records_episode_restart(
    tmp_path,
):
    source = _checkpoint_trainer(tmp_path / "source")
    torch.manual_seed(1234)
    checkpoint = source.save("resume.pt")
    expected_random = torch.rand(5)

    target = _checkpoint_trainer(tmp_path / "target")
    torch.manual_seed(9999)
    target.resume(checkpoint)

    torch.testing.assert_close(torch.rand(5), expected_random)
    assert target.replay.size == source.replay.size
    assert target.global_step == source.global_step
    assert target.interrupted_episodes == 1
    assert target.resume_provenance["episode_restart_applied"] is True
    assert target.resume_provenance["legacy_override"] is False


def test_trainer_resume_fails_closed_without_replay_or_on_contract_drift(
    tmp_path,
):
    no_replay = _checkpoint_trainer(
        tmp_path / "no_replay", save_replay=False
    )
    checkpoint = no_replay.save("resume.pt")
    target = _checkpoint_trainer(tmp_path / "target", save_replay=False)
    with pytest.raises(ValueError, match="requires checkpoint replay data"):
        target.resume(checkpoint)

    source = _checkpoint_trainer(tmp_path / "source", actor_update_after=0)
    checkpoint = source.save("resume.pt")
    changed = _checkpoint_trainer(tmp_path / "changed", actor_update_after=12)
    with pytest.raises(ValueError, match="contract does not match"):
        changed.resume(checkpoint)


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


def _frozen_correction_agent(seed=17, correction_penalty_weight=0.0):
    base_config = SACConfig(hidden_sizes=(16, 16), activation="relu")
    base = SACAgent(5, 2, base_config, device="cpu", seed=seed)
    correction_config = SACConfig(
        hidden_sizes=(16, 16),
        activation="relu",
        policy_mode="frozen_bc_correction",
        correction_scale=(0.20, 0.10),
        correction_gate_alpha=0.5,
        correction_initial_log_std=-2.5,
        correction_penalty_weight=correction_penalty_weight,
    )
    correction = SACAgent(
        5, 2, correction_config, device="cpu", seed=seed + 1
    )
    correction.initialize_frozen_base_actor(base.actor.state_dict())
    return base, correction


def test_frozen_correction_zero_mean_exactly_recovers_bc_and_is_bounded():
    base, correction = _frozen_correction_agent()
    observation = np.asarray((0.3, -0.2, 0.1, 0.4, -0.5), dtype=np.float32)

    expected, _ = base.select_action(observation, deterministic=True)
    actual, diagnostics = correction.select_action(
        observation, deterministic=True
    )

    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["policy_mode"] == "frozen_bc_correction"
    assert diagnostics["applied_correction_abs_max"] == 0.0
    assert all(
        not parameter.requires_grad
        for parameter in correction.base_actor.parameters()
    )

    base_tensor = torch.as_tensor(expected).unsqueeze(0)
    positive, positive_delta, _ = correction._compose_correction(
        base_tensor, torch.ones_like(base_tensor)
    )
    negative, negative_delta, _ = correction._compose_correction(
        base_tensor, -torch.ones_like(base_tensor)
    )
    assert torch.all(positive <= 1.0) and torch.all(positive >= -1.0)
    assert torch.all(negative <= 1.0) and torch.all(negative >= -1.0)
    maximum = torch.as_tensor((0.10, 0.05)).unsqueeze(0)
    assert torch.all(positive_delta.abs() <= maximum + 1e-7)
    assert torch.all(negative_delta.abs() <= maximum + 1e-7)


def test_frozen_correction_exposes_composed_mppi_gaussian():
    _, correction = _frozen_correction_agent(seed=19)
    observations = np.linspace(-0.4, 0.4, 15, dtype=np.float32).reshape(3, 5)
    expected = correction.select_action_batch(observations, deterministic=True)

    pre_tanh_mean, log_std = correction.policy_gaussian_parameters_batch(
        observations
    )

    np.testing.assert_allclose(
        np.tanh(pre_tanh_mean), expected, rtol=1e-6, atol=1e-6
    )
    assert pre_tanh_mean.shape == expected.shape
    assert log_std.shape == expected.shape
    assert np.isfinite(log_std).all()
    assert np.all(np.exp(log_std) > 0.0)
    with torch.no_grad():
        tensor = torch.as_tensor(observations)
        base_pre, base_log_std = correction.base_actor.distribution(tensor)
        base_mean = torch.tanh(base_pre).numpy()
    expected_post_tanh_std = (
        (1.0 - base_mean ** 2) * np.exp(base_log_std.numpy())
    )
    actual_post_tanh_std = (
        (1.0 - np.tanh(pre_tanh_mean) ** 2) * np.exp(log_std)
    )
    np.testing.assert_allclose(
        actual_post_tanh_std, expected_post_tanh_std, rtol=1e-5, atol=1e-6
    )


def test_base_gate_exactly_recovers_frozen_actor_without_critic_authority():
    base, correction = _frozen_correction_agent(seed=23)
    observation = np.asarray((0.2, -0.1, 0.4, 0.3, -0.2), dtype=np.float32)
    with torch.no_grad():
        correction.actor.network[-1].bias[:2].fill_(1.0)
    candidate, _ = correction.select_action(observation, deterministic=True)
    expected, _ = base.select_action(observation, deterministic=True)
    actual, diagnostics = correction.filter_correction_by_advantage(
        observation,
        candidate,
        gate_mode="base",
        compute_unselected_diagnostics=False,
    )

    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["correction_advantage_gate_alpha"] == 0.0
    assert diagnostics["correction_advantage_gate_mode"] == "base"


def test_frozen_correction_update_and_checkpoint_never_modify_bc_base():
    _, correction = _frozen_correction_agent(seed=31)
    base_before = {
        name: value.detach().clone()
        for name, value in correction.base_actor.state_dict().items()
    }
    actor_before = {
        name: value.detach().clone()
        for name, value in correction.actor.state_dict().items()
    }
    rng = np.random.RandomState(32)
    batch = {
        "observations": rng.normal(size=(16, 5)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(16, 2)).astype(np.float32),
        "rewards": rng.normal(size=(16, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(16, 5)).astype(np.float32),
        "dones": np.zeros((16, 1), dtype=np.float32),
    }

    metrics = correction.update(batch)

    assert np.isfinite(tuple(metrics.values())).all()
    for name, value in correction.base_actor.state_dict().items():
        torch.testing.assert_close(value, base_before[name], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in correction.actor.state_dict().items()
    )

    state = correction.state_dict()
    restored = SACAgent(
        5, 2, correction.config, device="cpu", seed=999
    )
    restored.load_state_dict(state, load_optimizers=True)
    observation = rng.normal(size=5).astype(np.float32)
    expected, _ = correction.select_action(observation, deterministic=True)
    actual, _ = restored.select_action(observation, deterministic=True)
    np.testing.assert_array_equal(actual, expected)
    assert restored.base_actor_initialized is True
    assert all(
        not parameter.requires_grad for parameter in restored.base_actor.parameters()
    )
    assert (
        restored.frozen_base_actor_sha256()
        == correction.frozen_base_actor_sha256()
    )

    tampered = copy.deepcopy(state)
    first_name = next(iter(tampered["base_actor"]))
    tampered["base_actor"][first_name].reshape(-1)[0] += 1.0
    rejected = SACAgent(5, 2, correction.config, device="cpu", seed=1000)
    with pytest.raises(ValueError, match="checksum"):
        rejected.load_state_dict(tampered, load_optimizers=False)


def test_frozen_correction_contract_fails_closed():
    with pytest.raises(ValueError, match="policy_mode"):
        SACAgent(3, 2, SACConfig(policy_mode="magic"), device="cpu")
    with pytest.raises(ValueError, match="correction_scale"):
        SACAgent(
            3,
            2,
            SACConfig(
                policy_mode="frozen_bc_correction",
                correction_scale=(0.2, 0.1, 0.3),
            ),
            device="cpu",
        )
    with pytest.raises(ValueError, match="correction penalty"):
        SACAgent(
            3,
            2,
            SACConfig(policy_mode="direct", correction_penalty_weight=0.1),
            device="cpu",
        )

    agent = SACAgent(
        3,
        2,
        SACConfig(policy_mode="frozen_bc_correction"),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="requires a BC initialization"):
        agent.select_action(np.zeros(3, dtype=np.float32), deterministic=True)
    with pytest.raises(ValueError, match="trains a direct actor"):
        agent.behavior_cloning_update(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )


def test_frozen_correction_penalty_is_normalized_and_reported():
    _, agent = _frozen_correction_agent(
        seed=43, correction_penalty_weight=0.25
    )
    rng = np.random.RandomState(44)
    batch = {
        "observations": rng.normal(size=(8, 5)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(8, 2)).astype(np.float32),
        "rewards": rng.normal(size=(8, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(8, 5)).astype(np.float32),
        "dones": np.zeros((8, 1), dtype=np.float32),
    }

    metrics = agent.update(batch)

    assert metrics["correction_penalty"] >= 0.0
    assert metrics["weighted_correction_penalty"] == pytest.approx(
        0.25 * metrics["correction_penalty"]
    )


def test_group_robust_mean_emphasizes_worst_scene_and_keeps_gradients():
    losses = torch.tensor(
        [[1.0], [1.0], [3.0], [3.0], [2.0], [2.0]],
        requires_grad=True,
    )
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    smooth, group_losses = group_robust_mean(losses, groups, 0.1)
    exact, _ = group_robust_mean(losses, groups, 0.0)
    assert group_losses.tolist() == pytest.approx([1.0, 3.0, 2.0])
    assert losses.mean() < smooth <= exact
    assert exact.item() == pytest.approx(3.0)
    smooth.backward()
    assert torch.isfinite(losses.grad).all()
    assert losses.grad[2].item() > losses.grad[0].item()


def test_group_robust_sac_update_requires_and_reports_scene_groups():
    base = SACConfig(hidden_sizes=(8,), activation="relu")
    config = SACConfig(
        hidden_sizes=(8,),
        activation="relu",
        policy_mode="frozen_bc_correction",
        correction_scale=(0.2, 0.1),
        actor_group_robust_enabled=True,
        actor_group_robust_temperature=0.1,
    )
    agent = SACAgent(3, 2, config, device="cpu", seed=47)
    source = SACAgent(3, 2, base, device="cpu", seed=48)
    agent.initialize_frozen_base_actor(source.actor.state_dict())
    rng = np.random.RandomState(49)
    batch = {
        "observations": rng.normal(size=(9, 3)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(9, 2)).astype(np.float32),
        "rewards": rng.normal(size=(9, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(9, 3)).astype(np.float32),
        "dones": np.zeros((9, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="group labels"):
        agent.update(batch)
    batch["groups"] = np.arange(len(batch["rewards"])) % 3
    metrics = agent.update(batch)
    assert metrics["actor_group_robust_enabled"] == 1.0
    assert metrics["actor_group_count"] == 3.0
    assert metrics["actor_group_loss_max"] >= metrics["actor_group_loss_min"]


def test_quantile_huber_and_lower_tail_cvar_are_finite_and_differentiable():
    predictions = torch.zeros((2, 5), requires_grad=True)
    targets = torch.ones((2, 5))
    loss = quantile_huber_loss(predictions, targets, kappa=1.0)
    assert loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(predictions.grad).all()

    values = torch.tensor([[4.0, -3.0, 2.0, -1.0, 7.0]])
    assert lower_tail_cvar(values, 0.4).item() == pytest.approx(-2.0)
    assert lower_tail_cvar(values, 1.0).item() == pytest.approx(1.8)


def test_quantile_cvar_sac_update_and_checkpoint_roundtrip():
    config = SACConfig(
        hidden_sizes=(8,),
        activation="relu",
        policy_mode="frozen_bc_correction",
        correction_scale=(0.2, 0.1),
        critic_distribution="quantile",
        critic_num_quantiles=7,
        actor_cvar_fraction=0.3,
    )
    base = SACAgent(
        3,
        2,
        SACConfig(hidden_sizes=(8,), activation="relu"),
        device="cpu",
        seed=51,
    )
    agent = SACAgent(3, 2, config, device="cpu", seed=52)
    agent.initialize_frozen_base_actor(base.actor.state_dict())
    rng = np.random.RandomState(53)
    batch = {
        "observations": rng.normal(size=(9, 3)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(9, 2)).astype(np.float32),
        "rewards": rng.normal(size=(9, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(9, 3)).astype(np.float32),
        "dones": np.zeros((9, 1), dtype=np.float32),
    }
    metrics = agent.update(batch)
    values = agent.critic1(
        torch.as_tensor(batch["observations"]),
        torch.as_tensor(batch["actions"]),
    )
    assert values.shape == (9, 7)
    assert metrics["critic_quantile_enabled"] == 1.0
    assert metrics["critic_quantile_count"] == 7.0
    assert metrics["critic_cvar_fraction"] == pytest.approx(0.3)
    assert metrics["critic_quantile_spread_mean"] >= 0.0
    assert np.isfinite(tuple(metrics.values())).all()

    restored = SACAgent(3, 2, config, device="cpu", seed=54)
    restored.load_state_dict(agent.state_dict())
    observation = np.zeros(3, dtype=np.float32)
    expected, _ = agent.select_action(observation, deterministic=True)
    actual, _ = restored.select_action(observation, deterministic=True)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_quantile_cvar_advantage_can_reject_positive_mean_tail_risk():
    class TailRiskQ(torch.nn.Module):
        def forward(self, observation, action):
            del observation
            scale = action[:, :1]
            return torch.cat(
                (-4.0 * scale, 2.0 * scale, 2.0 * scale,
                 2.0 * scale, 2.0 * scale),
                dim=-1,
            )

    config = SACConfig(
        hidden_sizes=(8,),
        policy_mode="frozen_bc_correction",
        correction_scale=(0.2, 0.1),
        critic_distribution="quantile",
        critic_num_quantiles=5,
        actor_cvar_fraction=0.2,
    )
    source = SACAgent(
        3, 2, SACConfig(hidden_sizes=(8,)), device="cpu", seed=55
    )
    agent = SACAgent(3, 2, config, device="cpu", seed=56)
    agent.initialize_frozen_base_actor(source.actor.state_dict())
    with torch.no_grad():
        for parameter in agent.base_actor.parameters():
            parameter.zero_()
    agent.critic1 = TailRiskQ()
    agent.critic2 = TailRiskQ()
    candidate = np.asarray((0.5, 0.0), dtype=np.float32)
    selected, diagnostics = agent.filter_correction_by_advantage(
        np.zeros(3, dtype=np.float32),
        candidate,
        gate_mode="hard",
        critic_source="online",
    )
    np.testing.assert_allclose(selected, np.zeros(2), atol=1e-7)
    assert diagnostics["online_conservative_advantage"] < 0.0
    assert diagnostics["critic_quantile_count"] == 5.0
    assert diagnostics["critic_cvar_fraction"] == pytest.approx(0.2)


def test_correction_advantage_uses_paired_twin_differences_and_recovers_bc():
    class ActionQ(torch.nn.Module):
        def __init__(self, slope):
            super().__init__()
            self.slope = float(slope)

        def forward(self, observation, action):
            del observation
            return self.slope * action[:, :1]

    _, agent = _frozen_correction_agent(seed=47)
    with torch.no_grad():
        for parameter in agent.base_actor.parameters():
            parameter.zero_()
    agent.critic1 = ActionQ(1.0)
    agent.critic2 = ActionQ(2.0)
    agent.target_critic1 = ActionQ(-1.0)
    agent.target_critic2 = ActionQ(-2.0)
    observation = np.zeros(agent.observation_dim, dtype=np.float32)
    candidate = np.asarray((0.5, 0.0), dtype=np.float32)

    accepted, online = agent.filter_correction_by_advantage(
        observation,
        candidate,
        gate_mode="hard",
        critic_source="online",
        threshold=0.0,
    )
    np.testing.assert_allclose(accepted, candidate)
    assert online["online_advantage_q1"] == pytest.approx(0.5)
    assert online["online_advantage_q2"] == pytest.approx(1.0)
    assert online["online_conservative_advantage"] == pytest.approx(0.5)
    assert online["correction_advantage_gate_alpha"] == 1.0

    rejected, target = agent.filter_correction_by_advantage(
        observation,
        candidate,
        gate_mode="hard",
        critic_source="target",
        threshold=0.0,
    )
    np.testing.assert_allclose(rejected, np.zeros(2), atol=0.0)
    assert target["target_advantage_q1"] == pytest.approx(-0.5)
    assert target["target_advantage_q2"] == pytest.approx(-1.0)
    assert target["target_conservative_advantage"] == pytest.approx(-1.0)
    assert target["correction_advantage_gate_alpha"] == 0.0
    assert target["applied_correction_abs_max"] == 0.0
    assert target["raw_applied_correction_abs_max"] == pytest.approx(0.5)


def test_correction_advantage_none_is_exact_behavioral_regression():
    _, agent = _frozen_correction_agent(seed=53)
    observation = np.linspace(
        -0.5, 0.5, agent.observation_dim, dtype=np.float32
    )
    candidate, _ = agent.select_action(observation, deterministic=True)
    unchanged, diagnostics = agent.filter_correction_by_advantage(
        observation,
        candidate,
        gate_mode="none",
        critic_source="online",
        threshold=0.0,
    )
    np.testing.assert_array_equal(unchanged, candidate)
    assert diagnostics["correction_advantage_gate_alpha"] == 1.0


def test_correction_consensus_lcb_is_scale_invariant_and_beta_one_is_minimum():
    class ActionQ(torch.nn.Module):
        def __init__(self, slope):
            super().__init__()
            self.slope = float(slope)

        def forward(self, observation, action):
            del observation
            return self.slope * action[:, :1]

    _, agent = _frozen_correction_agent(seed=59)
    with torch.no_grad():
        for parameter in agent.base_actor.parameters():
            parameter.zero_()
    observation = np.zeros(agent.observation_dim, dtype=np.float32)
    candidate = np.asarray((0.5, 0.0), dtype=np.float32)

    for scale in (1.0, 10.0):
        agent.target_critic1 = ActionQ(2.0 * scale)
        agent.target_critic2 = ActionQ(0.5 * scale)
        accepted, beta_one = agent.filter_correction_by_advantage(
            observation,
            candidate,
            gate_mode="lcb",
            critic_source="target",
            uncertainty_multiplier=1.0,
        )
        np.testing.assert_allclose(accepted, candidate)
        assert beta_one["target_consensus_lcb"] == pytest.approx(
            beta_one["target_conservative_advantage"]
        )
        assert beta_one["correction_advantage_gate_alpha"] == 1.0

        rejected, beta_two = agent.filter_correction_by_advantage(
            observation,
            candidate,
            gate_mode="lcb",
            critic_source="target",
            uncertainty_multiplier=2.0,
        )
        np.testing.assert_allclose(rejected, np.zeros(2), atol=0.0)
        assert beta_two["target_consensus_lcb"] < 0.0
        assert beta_two["correction_advantage_gate_alpha"] == 0.0


@pytest.mark.parametrize("value", [0.5, float("nan"), float("inf")])
def test_correction_consensus_multiplier_fails_closed(value):
    _, agent = _frozen_correction_agent(seed=61)
    observation = np.zeros(agent.observation_dim, dtype=np.float32)
    candidate = np.zeros(agent.action_dim, dtype=np.float32)
    with pytest.raises(ValueError, match="uncertainty multiplier"):
        agent.filter_correction_by_advantage(
            observation,
            candidate,
            gate_mode="lcb",
            critic_source="target",
            uncertainty_multiplier=value,
        )


def test_frozen_base_action_is_public_exact_and_validated():
    _, agent = _frozen_correction_agent(seed=67)
    observation = np.linspace(
        -0.2, 0.2, agent.observation_dim, dtype=np.float32
    )
    action = agent.frozen_base_action(observation)
    with torch.no_grad():
        expected = agent.base_actor.mean_action(
            torch.as_tensor(observation).unsqueeze(0)
        )[0].cpu().numpy()
    np.testing.assert_array_equal(action, expected)
    with pytest.raises(ValueError, match="observation"):
        agent.frozen_base_action(observation[:-1])


def test_selected_critic_only_and_base_reuse_preserve_target_gate_exactly():
    class CountingActionQ(torch.nn.Module):
        def __init__(self, slope):
            super().__init__()
            self.slope = float(slope)
            self.calls = 0

        def forward(self, observation, action):
            del observation
            self.calls += 1
            return self.slope * action[:, :1]

    _, agent = _frozen_correction_agent(seed=71)
    online1 = CountingActionQ(3.0)
    online2 = CountingActionQ(4.0)
    target1 = CountingActionQ(2.0)
    target2 = CountingActionQ(0.5)
    agent.critic1 = online1
    agent.critic2 = online2
    agent.target_critic1 = target1
    agent.target_critic2 = target2
    observation = np.linspace(
        -0.2, 0.2, agent.observation_dim, dtype=np.float32
    )
    candidate, internal = agent.select_action(
        observation, deterministic=True, include_internal=True
    )
    reference, reference_diagnostics = agent.filter_correction_by_advantage(
        observation,
        candidate,
        gate_mode="lcb",
        critic_source="target",
        uncertainty_multiplier=2.0,
    )
    assert online1.calls == 2
    assert online2.calls == 2
    online1.calls = online2.calls = 0
    target1.calls = target2.calls = 0

    optimized, optimized_diagnostics = agent.filter_correction_by_advantage(
        observation,
        candidate,
        gate_mode="lcb",
        critic_source="target",
        uncertainty_multiplier=2.0,
        compute_unselected_diagnostics=False,
        base_action=internal["_base_action"],
    )
    np.testing.assert_array_equal(optimized, reference)
    assert optimized_diagnostics["selected_consensus_lcb"] == (
        reference_diagnostics["selected_consensus_lcb"]
    )
    assert optimized_diagnostics["target_q1_base"] == (
        reference_diagnostics["target_q1_base"]
    )
    assert optimized_diagnostics["correction_advantage_gate_alpha"] == (
        reference_diagnostics["correction_advantage_gate_alpha"]
    )
    assert optimized_diagnostics[
        "unselected_critic_diagnostics_computed"
    ] is False
    assert "online_q1_base" not in optimized_diagnostics
    assert online1.calls == 0
    assert online2.calls == 0
    assert target1.calls == 2
    assert target2.calls == 2


def test_batched_policy_and_expected_quantile_q_are_finite():
    config = SACConfig(
        hidden_sizes=(8,),
        critic_distribution="quantile",
        critic_num_quantiles=7,
        actor_cvar_fraction=0.25,
    )
    agent = SACAgent(3, 2, config, device="cpu", seed=91)
    observations = np.asarray(
        ((0.0, 0.1, 0.2), (0.3, -0.2, 0.5), (-0.1, 0.4, 0.7)),
        dtype=np.float32,
    )
    actions = agent.select_action_batch(observations, deterministic=True)
    values = agent.expected_twin_q(
        observations, actions, critic_source="target"
    )

    assert actions.shape == (3, 2)
    for key in ("q1", "q2", "minimum", "mean", "disagreement"):
        assert values[key].shape == (3,)
        assert np.isfinite(values[key]).all()
    assert np.all(values["minimum"] <= values["mean"] + 1e-12)
    assert values["critic_source"] == "target"
