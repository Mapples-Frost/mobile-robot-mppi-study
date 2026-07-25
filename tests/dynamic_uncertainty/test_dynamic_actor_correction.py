from pathlib import Path

import numpy as np
import torch

from experiments.dynamic_uncertainty.collect_dynamic_actor_teacher import (
    configure_teacher_job,
)
from experiments.dynamic_uncertainty.collect_dynamic_actor_rollin import (
    configure_rollin_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_combined_safety_probe import (
    configure_combined_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_exact_fallback_probe import (
    STAGE5_PROTOCOL,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    _mapping,
    _resolve,
)
from experiments.dynamic_uncertainty.train_dynamic_actor_correction import (
    _metrics,
    _normalized_to_physical,
    _physical_to_normalized,
    _target_log_std_values,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


ROOT = Path(__file__).resolve().parents[2]


def test_dynamic_actor_teacher_is_rl_free_and_enables_motion_safety():
    protocol = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(protocol["base_config"]))
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    config = configure_teacher_job(
        base, stage3, stage4, 730100101, maximum_steps=123
    )

    assert config["rl"]["enabled"] is False
    assert config["planner"].get("optimizer", "standard") == "standard"
    assert config["experiment"]["max_steps"] == 123
    assert config["experiment"]["sealed_seeds_opened"] is False
    assert config["planner"][
        "probabilistic_obstacle_stopping_feasibility_enabled"
    ] is True
    assert config["planner"][
        "probabilistic_obstacle_emergency_candidates_enabled"
    ] is True
    assert config["planner"][
        "probabilistic_obstacle_emergency_candidate_prefix_steps"
    ] == 3
    assert config["perception"]["scan_guard"][
        "dynamic_escape_hold_enabled"
    ] is True


def test_combined_safety_preserves_explicit_emergency_prefix_protocol():
    protocol = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(protocol["base_config"]))
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    base["planner"][
        "probabilistic_obstacle_emergency_candidate_prefix_steps"
    ] = 15

    config = configure_combined_job(
        base, stage3, stage4, "combined_veto", 730100101
    )

    assert config["planner"][
        "probabilistic_obstacle_emergency_candidate_prefix_steps"
    ] == 15


def test_dynamic_actor_action_normalization_round_trip():
    lower = np.asarray((0.0, -0.9), dtype=np.float32)
    upper = np.asarray((0.35, 0.9), dtype=np.float32)
    physical = np.asarray(
        ((0.0, -0.9), (0.12, 0.2), (0.35, 0.9)), dtype=np.float32
    )

    normalized = _physical_to_normalized(physical, lower, upper)
    restored = _normalized_to_physical(normalized, lower, upper)

    np.testing.assert_allclose(restored, physical, atol=1.0e-7)
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)


def test_dynamic_actor_metrics_report_collision_exposure_fit():
    source = np.zeros((3, 2), dtype=np.float32)
    target = np.asarray(((1.0, -0.5), (0.5, 0.0), (0.0, 0.0)))
    candidate = target.copy()
    arrays = {"low_risk_opportunity": np.ones(3, dtype=bool)}
    selected = np.ones(3, dtype=bool)
    critical = np.asarray((True, False, False))

    result = _metrics(
        source,
        candidate,
        target,
        arrays,
        selected,
        np.asarray((-0.35, -0.9)),
        np.asarray((0.35, 0.9)),
        np.zeros(3, dtype=bool),
        critical,
    )

    assert result["critical_samples"] == 1
    assert result["source_critical_physical_teacher_rmse"] > 0.0
    assert result["candidate_critical_physical_teacher_rmse"] == 0.0


def test_dynamic_actor_target_log_std_supports_per_action_exploration():
    np.testing.assert_allclose(
        _target_log_std_values((-2.5, -0.35), 2), (-2.5, -0.35)
    )
    np.testing.assert_allclose(
        _target_log_std_values(-1.0, 2), (-1.0, -1.0)
    )


def test_dynamic_actor_rollin_keeps_shadow_authority_and_development_scope():
    protocol = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(protocol["base_config"]))
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    checkpoint = ROOT / (
        "research_artifacts/dynamic_actor_correction_development_v2/"
        "checkpoints/best.pt"
    )

    config = configure_rollin_job(
        base, stage3, stage4, 730100118, checkpoint, maximum_steps=321
    )

    gate = config["planner"]["paper_rl_driven"][
        "proposal_advantage_gate"
    ]
    assert config["rl"]["enabled"] is True
    assert Path(config["rl"]["checkpoint"]).resolve() == checkpoint.resolve()
    assert gate["enabled"] is True
    assert gate["mode"] == "shadow"
    assert config["planner"]["paper_rl_driven"][
        "standard_fallback_on_advantage_veto"
    ] is False
    assert config["experiment"]["max_steps"] == 321
    assert config["experiment"]["sealed_seeds_opened"] is False


def test_bidirectional_migration_preserves_base_physics_and_allows_reverse():
    base = SACAgent(
        5, 2, SACConfig(hidden_sizes=(8, 8)), device="cpu", seed=3
    )
    migrated = SACAgent(
        5,
        2,
        SACConfig(
            hidden_sizes=(8, 8),
            policy_mode="frozen_bc_correction",
            correction_scale=(2.0, 1.0),
            correction_base_action_scale=(0.5, 1.0),
            correction_base_action_offset=(0.5, 0.0),
        ),
        device="cpu",
        seed=4,
    )
    migrated.initialize_frozen_base_actor(base.actor.state_dict())
    observation = np.linspace(-0.4, 0.4, 5, dtype=np.float32)
    old_latent, _ = base.select_action(observation, deterministic=True)
    zero_correction, _ = migrated.select_action(
        observation, deterministic=True
    )
    old_physical = _normalized_to_physical(
        old_latent, np.asarray((0.0, -0.9)), np.asarray((0.35, 0.9))
    )
    migrated_physical = _normalized_to_physical(
        zero_correction,
        np.asarray((-0.35, -0.9)),
        np.asarray((0.35, 0.9)),
    )
    np.testing.assert_allclose(migrated_physical, old_physical, atol=1e-7)

    with torch.no_grad():
        migrated.actor.network[-1].bias[0] = -8.0
    reverse_latent, _ = migrated.select_action(
        observation, deterministic=True
    )
    reverse_physical = _normalized_to_physical(
        reverse_latent,
        np.asarray((-0.35, -0.9)),
        np.asarray((0.35, 0.9)),
    )
    assert reverse_physical[0] < -0.30


def test_expanded_correction_observation_preserves_frozen_base_action():
    base = SACAgent(
        5, 2, SACConfig(hidden_sizes=(8, 8)), device="cpu", seed=13
    )
    migrated = SACAgent(
        9,
        2,
        SACConfig(
            hidden_sizes=(8, 8),
            policy_mode="frozen_bc_correction",
            correction_scale=(2.0, 1.0),
            correction_base_action_scale=(0.5, 1.0),
            correction_base_action_offset=(0.5, 0.0),
            correction_base_observation_dim=5,
        ),
        device="cpu",
        seed=14,
    )
    migrated.initialize_frozen_base_actor(base.actor.state_dict())
    old_observation = np.linspace(-0.4, 0.4, 5, dtype=np.float32)
    expanded = np.concatenate((
        old_observation,
        np.asarray((-1.0, -0.5, 0.5, 1.0), dtype=np.float32),
    ))
    old_action, _ = base.select_action(old_observation, deterministic=True)
    new_action, _ = migrated.select_action(expanded, deterministic=True)
    old_physical = _normalized_to_physical(
        old_action, np.asarray((0.0, -0.9)), np.asarray((0.35, 0.9))
    )
    new_physical = _normalized_to_physical(
        new_action, np.asarray((-0.35, -0.9)), np.asarray((0.35, 0.9))
    )
    np.testing.assert_allclose(new_physical, old_physical, atol=1e-7)
