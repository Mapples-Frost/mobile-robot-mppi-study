import numpy as np

from mobile_robot_mppi.core.references import (
    PointGoal,
    PolylineReference,
    ReferenceTarget,
)
from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    dynamic_unicycle_state,
)
from mobile_robot_mppi.core.types import (
    Pose2D,
    RobotObservation,
    Twist2D,
)
from mobile_robot_mppi.obstacles.collision_risk import (
    GaussianMixtureObstacleForecast,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.planning.rl_driven_mppi import (
    PaperRLDrivenMppiController,
)
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior


class AuditableDirectPolicy:
    def __init__(self):
        self.fallback = GoalWarmStartPrior()
        self.distribution_calls = 0
        self.sample_calls = 0
        self.terminal_calls = 0

    def reset(self):
        self.distribution_calls = 0
        self.sample_calls = 0
        self.terminal_calls = 0

    def propose(self, observation, reference, horizon, action_spec):
        return self.fallback.propose(
            observation, reference, horizon, action_spec
        )

    def action_distribution(
        self,
        states,
        previous_controls,
        observation,
        reference,
        state_spec,
        time_offset,
    ):
        del previous_controls, observation, reference, state_spec, time_offset
        self.distribution_calls += 1
        batch = np.asarray(states).shape[0]
        return {
            "physical_mean": np.tile((0.22, 0.04), (batch, 1)),
            "physical_std": np.tile((0.06, 0.16), (batch, 1)),
        }

    def sample_actions(
        self,
        states,
        previous_controls,
        observation,
        reference,
        state_spec,
        time_offset,
        rng,
    ):
        self.sample_calls += 1
        distribution = self.action_distribution(
            states,
            previous_controls,
            observation,
            reference,
            state_spec,
            time_offset,
        )
        values = distribution["physical_mean"] + rng.normal(
            size=distribution["physical_mean"].shape
        ) * distribution["physical_std"]
        return values, distribution

    def terminal_value(
        self,
        terminal_states,
        terminal_controls,
        observation,
        reference,
        state_spec,
        horizon_dt,
        critic_source="target",
    ):
        del (
            terminal_controls,
            observation,
            reference,
            horizon_dt,
        )
        self.terminal_calls += 1
        returns = 2.0 + terminal_states[:, state_spec.index("x")]
        return returns, {
            "terminal_q_mean": float(np.mean(returns)),
            "terminal_q_disagreement_mean": 0.0,
            "terminal_critic_source": critic_source,
        }


class JointBatchedDirectPolicy(AuditableDirectPolicy):
    def sample_from_distribution(self, distribution, rng):
        return (
            distribution["physical_mean"]
            + rng.normal(size=distribution["physical_mean"].shape)
            * distribution["physical_std"]
        )


class EncodedJointBatchedDirectPolicy(JointBatchedDirectPolicy):
    def _encoded_batch(
        self,
        states,
        previous_controls,
        observation,
        reference,
        state_spec,
        time_offset,
    ):
        del (
            previous_controls,
            observation,
            reference,
            state_spec,
            time_offset,
        )
        values = np.zeros((len(states), 4), dtype=np.float32)
        return values, values


class FixedSupervisedManeuverPolicy:
    heads = 3
    horizon = 5
    action_dim = 2

    def propose(self, raw_observation, action_spec):
        assert np.asarray(raw_observation).shape == (4,)
        values = np.zeros((self.heads, self.horizon, self.action_dim))
        values[0, :, 0] = 0.15
        values[0, :, 1] = 0.5
        values[1, :, 0] = 0.20
        values[1, :, 1] = -0.5
        values[2, :, 0] = 0.05
        return np.clip(values, action_spec.lower, action_spec.upper)


class FixedMeanDirectPolicy(AuditableDirectPolicy):
    def __init__(self, mean):
        super().__init__()
        self.mean = np.asarray(mean, dtype=np.float64)

    def action_distribution(self, *args, **kwargs):
        result = super().action_distribution(*args, **kwargs)
        batch = result["physical_mean"].shape[0]
        result["physical_mean"] = np.tile(self.mean, (batch, 1))
        return result


class ReliabilityDirectPolicy(JointBatchedDirectPolicy):
    def __init__(self, ood_score, critic_disagreement=0.0):
        super().__init__()
        self.ood_score = float(ood_score)
        self.critic_disagreement = float(critic_disagreement)

    def action_distribution(self, *args, **kwargs):
        result = super().action_distribution(*args, **kwargs)
        batch = result["physical_mean"].shape[0]
        result["raw_observation"] = np.zeros((batch, 3))
        return result

    def support_ood_scores(self, raw_observations):
        return np.full(len(raw_observations), self.ood_score)

    def terminal_value_details(self, *args, **kwargs):
        returns, diagnostics = self.terminal_value(*args, **kwargs)
        return {
            "returns": returns,
            "critic_disagreement": np.full(
                len(returns), self.critic_disagreement
            ),
            "critic_ood_scores": np.full(
                len(returns), self.ood_score
            ),
            "diagnostics": diagnostics,
        }


class ReliabilityResidual:
    state_dim = 5
    control_dim = 2
    innovation_samples = 0
    innovation_error_ema = 0.0

    def derivative(self, state, control, time=None):
        del control, time
        return np.zeros_like(np.asarray(state, dtype=np.float64))

    def disagreement(self, states, controls):
        del controls
        return np.zeros(len(states))

    def support_confidence(self, states, controls):
        del controls
        return np.ones(len(states))


def _observation():
    return RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
    )


def _controller(policy):
    return PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 3,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.5,
        },
    )


def test_supervised_maneuver_actor_replaces_guided_rows_without_more_rollouts():
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            path_preview_enabled=True,
            path_boundary_enabled=True,
            path_boundary_violation_penalty=10000.0,
            path_boundary_candidate_filter_enabled=True,
            seed=20260718,
        ),
        sampling_prior=EncodedJointBatchedDirectPolicy(),
        maneuver_proposal_policy=FixedSupervisedManeuverPolicy(),
        paper_rl_driven_config={
            "iterations": 3,
            "guided_fraction": 0.0,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )

    reference = PolylineReference(
        [(0.0, 0.0), (1.0, 0.0)],
        corridor_half_width=1.0,
        footprint_radius=0.2,
    )
    result = controller.plan(_observation(), reference)
    diagnostics = result.diagnostics

    assert diagnostics["supervised_maneuver_actor_enabled"]
    assert diagnostics["supervised_proposal_count"] == 3
    assert diagnostics["supervised_replaced_guided_count"] == 3
    assert diagnostics["supervised_added_rollout_count"] == 0
    assert diagnostics["paper_guided_unique_sequences"] == 4
    assert diagnostics["paper_gaussian_samples_per_iteration"] == 16
    assert diagnostics["paper_total_rollouts"] == 60
    assert diagnostics["supervised_risk_feasible_count"] == 9
    assert [
        diagnostics[f"supervised_head_{head}_source_id"]
        for head in range(3)
    ] == [
        "supervised_head_0",
        "supervised_head_1",
        "supervised_head_2",
    ]
    assert [
        diagnostics[f"supervised_head_{head}_insertion_index"]
        for head in range(3)
    ] == [1, 2, 3]
    assert [
        diagnostics[f"supervised_head_{head}_behavior_label"]
        for head in range(3)
    ] == ["left", "right", "yield"]
    assert [
        diagnostics[f"supervised_head_{head}_proposal_count"]
        for head in range(3)
    ] == [1, 1, 1]
    assert sum(
        diagnostics[f"supervised_head_{head}_elite_count"]
        for head in range(3)
    ) == diagnostics["supervised_elite_count"]


def test_supervised_allocation_floor_is_inert_when_actor_is_disabled():
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            path_preview_enabled=True,
            path_boundary_enabled=True,
            path_boundary_violation_penalty=10000.0,
            path_boundary_candidate_filter_enabled=True,
            seed=20260718,
        ),
        sampling_prior=EncodedJointBatchedDirectPolicy(),
        paper_rl_driven_config={
            "iterations": 3,
            "guided_fraction": 0.0,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )

    reference = PolylineReference(
        [(0.0, 0.0), (1.0, 0.0)],
        corridor_half_width=1.0,
        footprint_radius=0.2,
    )
    result = controller.plan(_observation(), reference)
    diagnostics = result.diagnostics

    assert not diagnostics["supervised_maneuver_actor_enabled"]
    assert diagnostics["supervised_proposal_count"] == 0
    assert diagnostics["paper_guided_unique_sequences"] == 0
    assert diagnostics["paper_gaussian_samples_per_iteration"] == 20
    assert diagnostics["paper_total_rollouts"] == 60


def test_paper_optimizer_applies_shared_boundary_candidate_filter(monkeypatch):
    policy = AuditableDirectPolicy()
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            path_preview_enabled=True,
            path_boundary_enabled=True,
            path_boundary_violation_penalty=10000.0,
            path_boundary_candidate_filter_enabled=True,
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    reference = PolylineReference(
        [(0.0, 0.0), (5.0, 0.0)],
        corridor_half_width=0.5,
        footprint_radius=0.2,
    )

    margin_calls = 0

    def synthetic_margins(trajectories, _reference, **kwargs):
        nonlocal margin_calls
        del kwargs
        margin_calls += 1
        count = np.asarray(trajectories).shape[0]
        values = np.full((count, controller.config.horizon + 1), 0.1)
        if count > 1:
            values[-1, 1:] = -0.1
        return values

    monkeypatch.setattr(
        controller, "_path_boundary_margins", synthetic_margins
    )

    result = controller.plan(_observation(), reference)

    assert result.diagnostics["path_boundary_candidate_filter_enabled"]
    assert result.diagnostics[
        "path_boundary_candidate_feasible_fraction"
    ] == 0.95
    assert result.diagnostics[
        "path_boundary_candidate_feasible_fraction_min"
    ] == 0.95
    assert not result.diagnostics["path_boundary_no_feasible_candidates"]
    assert result.diagnostics["path_boundary_weighted_update_feasible"]
    # Two optimizer iterations plus one final weighted sequence. The shared
    # per-iteration margins serve both soft cost and hard filtering.
    assert margin_calls == 3


def test_paper_optimizer_reuses_candidate_risk_for_cost_and_filter(monkeypatch):
    policy = AuditableDirectPolicy()
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    forecast = GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=np.full((5, 1, 2), 20.0),
        component_covariances=np.repeat(
            (0.01 * np.eye(2))[None, None, :, :], 5, axis=0
        ),
        component_weights=np.ones((5, 1)),
        radius_m=0.2,
        source="synthetic_test",
    )
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (forecast,)},
    )
    original = controller._probabilistic_collision_risk
    risk_calls = 0

    def counted_risk(*args, **kwargs):
        nonlocal risk_calls
        risk_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", counted_risk
    )

    result = controller.plan(observation, PointGoal(1.0, 0.0))

    assert not result.diagnostics["probabilistic_obstacle_hard_violation"]
    assert result.diagnostics[
        "paper_guided_boundary_feasible_fraction"
    ] == 1.0
    assert result.diagnostics[
        "paper_guided_static_feasible_fraction"
    ] == 1.0
    assert result.diagnostics[
        "paper_guided_risk_feasible_fraction"
    ] == 1.0
    assert result.diagnostics[
        "paper_guided_boundary_first_failure_step_mean"
    ] == -1.0
    assert result.diagnostics[
        "paper_guided_static_first_failure_step_mean"
    ] == -1.0
    assert result.diagnostics[
        "paper_guided_risk_first_failure_step_mean"
    ] == -1.0
    # Two optimizer iterations plus one final weighted sequence. The shared
    # per-iteration evaluation serves both soft cost and hard filtering.
    assert risk_calls == 3


def test_post_center_commit_keeps_six_direction_emergency_lattice_available(
    monkeypatch,
):
    policy = AuditableDirectPolicy()
    horizon = 5
    sample_count = 20
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((-0.35, 0.60), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=sample_count,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=True,
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    forecast = GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=np.full((horizon, 1, 2), 20.0),
        component_covariances=np.repeat(
            (0.01 * np.eye(2))[None, None, :, :], horizon, axis=0
        ),
        component_weights=np.ones((horizon, 1)),
        radius_m=0.2,
        source="synthetic_test",
    )
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (forecast,)},
    )
    traversal_sequence = np.zeros((horizon, 2), dtype=np.float64)
    traversal_sequence[:, 0] = 0.60
    monkeypatch.setattr(
        controller,
        "_probabilistic_emergency_context",
        lambda *_args, **_kwargs: {"triggered": False},
    )
    monkeypatch.setattr(
        controller,
        "_probabilistic_traversal_window_context",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "candidate_requested": True,
            "commit_active": True,
            "commit_started": True,
            "current_progress": 2.30,
            "crossing_progress": 2.20,
            "clear_progress": 2.90,
            "sequence": traversal_sequence.copy(),
        },
    )
    original_guard = controller._apply_probabilistic_obstacle_action_guard
    observed = {"calls": 0}

    def inspected_guard(*args, **kwargs):
        samples = np.asarray(args[4])
        emergency_mask = np.asarray(
            kwargs["emergency_candidate_mask"], dtype=bool
        )
        traversal_index = int(kwargs["traversal_candidate_index"])
        emergency_actions = samples[emergency_mask, 0, :]
        observed["calls"] += 1
        observed["count"] = int(np.sum(emergency_mask))
        observed["unique_actions"] = int(
            np.unique(emergency_actions, axis=0).shape[0]
        )
        observed["traversal_is_emergency"] = bool(
            emergency_mask[traversal_index]
        )
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(
        controller,
        "_apply_probabilistic_obstacle_action_guard",
        inspected_guard,
    )

    result = controller.plan(
        observation,
        PolylineReference(((0.0, 0.0), (5.0, 0.0))),
    )

    assert observed == {
        "calls": 1,
        "count": 6,
        "unique_actions": 6,
        "traversal_is_emergency": False,
    }
    assert result.diagnostics[
        "probabilistic_obstacle_emergency_candidate_count"
    ] == 6


def test_rearm_raw_closing_keeps_six_direction_lattice_after_intent_expiry(
    monkeypatch,
):
    policy = AuditableDirectPolicy()
    horizon = 5
    sample_count = 20
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((-0.35, 0.60), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=sample_count,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled=True,
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    forecast = GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=np.full((horizon, 1, 2), 20.0),
        component_covariances=np.repeat(
            (0.01 * np.eye(2))[None, None, :, :], horizon, axis=0
        ),
        component_weights=np.ones((horizon, 1)),
        radius_m=0.2,
        source="synthetic_test",
    )
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (forecast,)},
    )
    hold_sequence = np.zeros((horizon, 2), dtype=np.float64)
    monkeypatch.setattr(
        controller,
        "_probabilistic_emergency_context",
        lambda *_args, **_kwargs: {
            "triggered": False,
            "raw_triggered": True,
            "closing_observed": True,
            "away_heading_error_rad": np.pi,
            "ttc_s": 0.8,
        },
    )
    monkeypatch.setattr(
        controller,
        "_probabilistic_traversal_window_context",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "candidate_requested": True,
            "commit_active": False,
            "commit_started": False,
            "retreat_requested": False,
            "rearm_pending": True,
            "sequence": hold_sequence.copy(),
        },
    )
    original_guard = controller._apply_probabilistic_obstacle_action_guard
    observed = {"calls": 0}

    def inspected_guard(*args, **kwargs):
        samples = np.asarray(args[4])
        emergency_mask = np.asarray(
            kwargs["emergency_candidate_mask"], dtype=bool
        )
        traversal_index = int(kwargs["traversal_candidate_index"])
        emergency_actions = samples[emergency_mask, 0, :]
        observed["calls"] += 1
        observed["count"] = int(np.sum(emergency_mask))
        observed["unique_actions"] = int(
            np.unique(emergency_actions, axis=0).shape[0]
        )
        observed["traversal_is_emergency"] = bool(
            emergency_mask[traversal_index]
        )
        observed["raw_triggered"] = bool(
            kwargs["traversal_context"][
                "temporal_emergency_raw_triggered"
            ]
        )
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(
        controller,
        "_apply_probabilistic_obstacle_action_guard",
        inspected_guard,
    )

    result = controller.plan(
        observation,
        PolylineReference(((0.0, 0.0), (5.0, 0.0))),
    )

    assert observed == {
        "calls": 1,
        "count": 6,
        "unique_actions": 6,
        "traversal_is_emergency": False,
        "raw_triggered": True,
    }
    assert result.diagnostics[
        "probabilistic_obstacle_emergency_candidate_count"
    ] == 6


def test_temporal_retreat_raw_closing_keeps_lattice_after_intent_expiry(
    monkeypatch,
):
    policy = AuditableDirectPolicy()
    horizon = 5
    sample_count = 20
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((-0.35, 0.60), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=sample_count,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_midpoint_retreat_reverse_filter_enabled=True,
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    forecast = GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=np.full((horizon, 1, 2), 20.0),
        component_covariances=np.repeat(
            (0.01 * np.eye(2))[None, None, :, :], horizon, axis=0
        ),
        component_weights=np.ones((horizon, 1)),
        radius_m=0.2,
        source="synthetic_test",
    )
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (forecast,)},
    )
    retreat_sequence = np.zeros((horizon, 2), dtype=np.float64)
    retreat_sequence[:, 0] = -0.35
    monkeypatch.setattr(
        controller,
        "_probabilistic_emergency_context",
        lambda *_args, **_kwargs: {
            "triggered": False,
            "raw_triggered": True,
            "closing_observed": True,
            "away_heading_error_rad": 0.0,
            "ttc_s": 0.8,
        },
    )
    monkeypatch.setattr(
        controller,
        "_probabilistic_traversal_window_context",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "candidate_requested": True,
            "commit_active": False,
            "commit_started": False,
            "retreat_requested": True,
            "retreat_temporal_lattice_requested": True,
            "rearm_pending": False,
            "sequence": retreat_sequence.copy(),
        },
    )
    original_guard = controller._apply_probabilistic_obstacle_action_guard
    observed = {"calls": 0}

    def inspected_guard(*args, **kwargs):
        samples = np.asarray(args[4])
        emergency_mask = np.asarray(
            kwargs["emergency_candidate_mask"], dtype=bool
        )
        emergency_actions = samples[emergency_mask, 0, :]
        observed["calls"] += 1
        observed["count"] = int(np.sum(emergency_mask))
        observed["unique_actions"] = int(
            np.unique(emergency_actions, axis=0).shape[0]
        )
        observed["raw_binding"] = bool(
            kwargs["traversal_context"][
                "temporal_retreat_raw_lattice_requested"
            ]
        )
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(
        controller,
        "_apply_probabilistic_obstacle_action_guard",
        inspected_guard,
    )

    result = controller.plan(
        observation,
        PolylineReference(((0.0, 0.0), (5.0, 0.0))),
    )

    assert observed == {
        "calls": 1,
        "count": 6,
        "unique_actions": 6,
        "raw_binding": True,
    }
    assert result.diagnostics[
        "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested"
    ]
    assert result.diagnostics[
        "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered"
    ]
    assert result.control_sequence[0, 0] < 0.0


def test_exit_deadline_retreat_keeps_latched_lattice_after_intent_expiry(
    monkeypatch,
):
    policy = AuditableDirectPolicy()
    horizon = 5
    sample_count = 20
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((-0.35, 0.60), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=sample_count,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_post_intent_all_hard_forward_filter_enabled=True,
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    controller._start_probabilistic_traversal_exit_deadline_retreat_escape()
    state = np.zeros(controller.state_spec.dimension, dtype=np.float64)
    assert controller._latch_probabilistic_traversal_exit_deadline_retreat_escape(
        (-0.35, 0.90), state
    )
    controller._probabilistic_emergency_intent_remaining = 0
    controller._probabilistic_emergency_latched_pattern = None
    forecast = GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=np.full((horizon, 1, 2), 20.0),
        component_covariances=np.repeat(
            (0.01 * np.eye(2))[None, None, :, :], horizon, axis=0
        ),
        component_weights=np.ones((horizon, 1)),
        radius_m=0.2,
        source="synthetic_test",
    )
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (forecast,)},
    )
    traversal_sequence = np.zeros((horizon, 2), dtype=np.float64)
    traversal_sequence[:, 0] = -0.35
    monkeypatch.setattr(
        controller,
        "_probabilistic_emergency_context",
        lambda *_args, **_kwargs: {
            "triggered": False,
            "raw_triggered": True,
        },
    )
    monkeypatch.setattr(
        controller,
        "_probabilistic_traversal_window_context",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "candidate_requested": True,
            "commit_active": False,
            "commit_started": False,
            "retreat_requested": True,
            "retreat_temporal_lattice_requested": True,
            "rearm_pending": False,
            "sequence": traversal_sequence.copy(),
        },
    )
    original_guard = controller._apply_probabilistic_obstacle_action_guard
    observed = {"calls": 0}

    def inspected_guard(*args, **kwargs):
        samples = np.asarray(args[4])
        emergency_mask = np.asarray(
            kwargs["emergency_candidate_mask"], dtype=bool
        )
        emergency_actions = samples[emergency_mask, 0, :]
        observed["calls"] += 1
        observed["count"] = int(np.sum(emergency_mask))
        observed["unique_actions"] = int(
            np.unique(emergency_actions, axis=0).shape[0]
        )
        observed["first_v"] = float(emergency_actions[0, 0])
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(
        controller,
        "_apply_probabilistic_obstacle_action_guard",
        inspected_guard,
    )

    result = controller.plan(
        observation,
        PolylineReference(((0.0, 0.0), (5.0, 0.0))),
    )

    assert observed == {
        "calls": 1,
        "count": 6,
        "unique_actions": 6,
        "first_v": -0.35,
    }
    assert result.control_sequence[0, 0] < 0.0
    assert result.diagnostics[
        "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active"
    ]
    assert result.diagnostics[
        "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused"
    ]
    assert result.diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "exit_deadline_retreat_emergency_candidate"
    )


def _reliable_controller(
    policy,
    conservative_terminal=False,
    terminal_guidance=False,
):
    from mobile_robot_mppi.planning.dynamics import ResidualPrediction

    return PaperRLDrivenMppiController(
        ResidualPrediction(
            DynamicUnicyclePrediction(), ReliabilityResidual()
        ),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            seed=20260718,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.3,
            "elite_fraction": 0.25,
            "terminal_value_weight": (
                0.5 if conservative_terminal else 0.0
            ),
            "terminal_guidance_radius": (
                0.5 if terminal_guidance else 0.0
            ),
            "terminal_guided_fraction_floor": (
                0.3 if terminal_guidance else 0.0
            ),
            "reliability": {
                "enabled": True,
                "ensemble_disagreement_soft": 0.02,
                "ensemble_disagreement_hard": 0.10,
                "innovation_error_soft": 0.1,
                "innovation_error_hard": 0.4,
                "actor_ood_soft": 3.0,
                "actor_ood_hard": 7.0,
                "medium_confidence": 0.33,
                "high_confidence": 0.67,
                "low_guided_fraction": 0.0,
                "medium_guided_fraction": 0.3,
                "high_guided_fraction": 0.6,
            },
            "conservative_terminal": {
                "enabled": bool(conservative_terminal),
                "ensemble_disagreement_soft": 0.02,
                "ensemble_disagreement_hard": 0.10,
                "innovation_error_soft": 0.1,
                "innovation_error_hard": 0.4,
                "critic_ood_soft": 3.0,
                "critic_ood_hard": 7.0,
                "critic_disagreement_soft": 0.5,
                "critic_disagreement_hard": 2.0,
                "uncertainty_penalty_weight": 2.0,
            },
        },
    )


def test_paper_guided_set_is_generated_once_and_reused_each_iteration():
    policy = AuditableDirectPolicy()
    result = _controller(policy).plan(
        _observation(), PointGoal(1.0, 0.0)
    )
    diagnostics = result.diagnostics

    assert diagnostics["optimizer"] == "paper_rl_driven"
    assert diagnostics["paper_faithful_gate1"]
    assert diagnostics["paper_guided_unique_sequences"] == 5
    assert diagnostics["paper_guided_reuses"] == 15
    assert diagnostics["paper_guided_generation_calls"] == 1
    assert diagnostics["paper_total_rollouts"] == 60
    # One stochastic Actor query per horizon step, not per MPPI iteration.
    assert policy.sample_calls == 5
    assert policy.terminal_calls == 3
    assert result.control_sequence.shape == (5, 2)
    assert result.predicted_trajectory.shape == (6, 5)


def test_actor_mean_and_guided_rollouts_share_one_batched_query_per_step():
    policy = JointBatchedDirectPolicy()
    result = _controller(policy).plan(
        _observation(), PointGoal(1.0, 0.0)
    )

    assert result.diagnostics["paper_actor_joint_batched"]
    assert policy.distribution_calls == 5
    assert policy.sample_calls == 0
    assert result.diagnostics["paper_guided_generation_calls"] == 1
    assert result.diagnostics["paper_guided_reuses"] == 15


def test_joint_actor_batch_preserves_unbatched_sampling_result():
    fallback = _controller(AuditableDirectPolicy()).plan(
        _observation(), PointGoal(1.0, 0.0)
    )
    batched = _controller(JointBatchedDirectPolicy()).plan(
        _observation(), PointGoal(1.0, 0.0)
    )

    np.testing.assert_array_equal(
        batched.control_sequence, fallback.control_sequence
    )
    np.testing.assert_array_equal(
        batched.predicted_trajectory, fallback.predicted_trajectory
    )


def test_paper_terminal_return_is_explicitly_converted_to_mppi_cost():
    policy = AuditableDirectPolicy()
    result = _controller(policy).plan(
        _observation(), PointGoal(1.0, 0.0)
    )

    assert result.diagnostics["terminal_value_enabled"]
    assert result.diagnostics["terminal_value_sign"] == "cost=-weight*return"
    assert result.diagnostics["terminal_value_cost_mean"] < 0.0


def test_paper_controller_is_deterministic_for_fixed_seed():
    first = _controller(AuditableDirectPolicy()).plan(
        _observation(), PointGoal(1.0, 0.0)
    )
    second = _controller(AuditableDirectPolicy()).plan(
        _observation(), PointGoal(1.0, 0.0)
    )

    np.testing.assert_array_equal(
        first.control_sequence, second.control_sequence
    )
    np.testing.assert_array_equal(
        first.predicted_trajectory, second.predicted_trajectory
    )


def test_paper_terminal_heading_gate_preserves_rotate_in_place():
    policy = JointBatchedDirectPolicy()
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            terminal_translation_speed_limit=0.11,
            terminal_translation_heading_gate_rad=0.40,
            terminal_alignment_yaw_gain=1.0,
            seed=20260719,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    misaligned = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, np.pi / 2.0),
        twist=Twist2D(0.2, 0.0),
    )

    result = controller.plan(misaligned, PointGoal(2.0, 0.0))

    assert result.proposed_control.values[0] == 0.0
    assert result.control_sequence[0, 0] == 0.0
    assert result.proposed_control.values[1] < 0.0
    assert result.diagnostics["terminal_heading_gate_active"]
    assert result.diagnostics["terminal_translation_scale"] == 0.0
    assert result.diagnostics["terminal_alignment_active"]
    assert result.diagnostics["terminal_bearing_error"] < 0.0
    assert result.diagnostics["terminal_alignment_omega"] < 0.0


def test_paper_reports_target_bearing_without_terminal_heading_gate():
    policy = JointBatchedDirectPolicy()
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            seed=20260719,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    misaligned = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, np.pi / 2.0),
        twist=Twist2D(0.2, 0.0),
    )

    result = controller.plan(misaligned, PointGoal(2.0, 0.0))

    np.testing.assert_allclose(
        result.diagnostics["target_bearing_error"], -np.pi / 2.0
    )
    assert not result.diagnostics["terminal_heading_gate_active"]


def test_paper_terminal_control_radius_preserves_actor_far_from_goal():
    policy = JointBatchedDirectPolicy()
    controller = PaperRLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=5,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            terminal_translation_speed_limit=0.11,
            terminal_translation_heading_gate_rad=0.40,
            terminal_alignment_yaw_gain=1.0,
            terminal_control_radius=0.8,
            seed=20260719,
        ),
        sampling_prior=policy,
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )
    misaligned = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, np.pi / 2.0),
        twist=Twist2D(0.2, 0.0),
    )

    far = controller.plan(misaligned, PointGoal(2.0, 0.0))
    assert not far.diagnostics["terminal_control_region_active"]
    assert not far.diagnostics["terminal_heading_gate_active"]
    assert not far.diagnostics["terminal_speed_limit_active"]
    assert not far.diagnostics["terminal_alignment_active"]

    near = controller.plan(misaligned, PointGoal(0.7, 0.0))
    assert near.diagnostics["terminal_control_region_active"]
    assert near.diagnostics["terminal_heading_gate_active"]
    assert near.diagnostics["terminal_speed_limit_active"]
    assert near.diagnostics["terminal_alignment_active"]
    assert near.proposed_control.values[0] == 0.0
    assert near.proposed_control.values[1] < 0.0


def test_paper_terminal_action_constraints_are_opt_in():
    result = _controller(JointBatchedDirectPolicy()).plan(
        _observation(), PointGoal(1.0, 0.0)
    )

    assert not result.diagnostics["terminal_heading_gate_active"]
    assert not result.diagnostics["terminal_alignment_active"]
    assert result.diagnostics["terminal_translation_scale"] == 1.0


def test_reliability_hss_applies_authority_on_next_control_cycle():
    high = _reliable_controller(ReliabilityDirectPolicy(0.0))
    first = high.plan(_observation(), PointGoal(1.0, 0.0))
    second = high.plan(_observation(), PointGoal(1.0, 0.0))

    assert first.diagnostics["paper_guided_unique_sequences"] == 6
    assert first.diagnostics["reliability_level"] == "high"
    assert first.diagnostics["reliability_guided_fraction_next"] == 0.6
    assert first.diagnostics["reliability_proposal_authority"] == 1.0
    assert first.diagnostics["reliability_proposal_fallback_fraction"] == 0.0
    assert second.diagnostics["paper_guided_unique_sequences"] == 12
    assert second.diagnostics["reliability_guided_fraction_applied"] == 0.6
    assert second.diagnostics["paper_guided_cost_observed"]
    assert second.diagnostics["paper_gaussian_cost_observed"]
    assert second.diagnostics["paper_guided_opportunity_count"] > 0
    assert second.diagnostics["paper_gaussian_opportunity_count"] > 0
    assert np.isfinite(
        second.diagnostics["paper_guided_minus_gaussian_cost_mean"]
    )
    assert np.isfinite(
        second.diagnostics["paper_actor_baseline_mean_abs_delta"]
    )

    low = _reliable_controller(ReliabilityDirectPolicy(8.0))
    low.plan(_observation(), PointGoal(1.0, 0.0))
    suppressed = low.plan(_observation(), PointGoal(1.0, 0.0))
    assert suppressed.diagnostics["paper_guided_unique_sequences"] == 0
    assert not suppressed.diagnostics["paper_guided_cost_observed"]
    assert suppressed.diagnostics["paper_guided_opportunity_count"] == 0
    assert suppressed.diagnostics["reliability_guided_fraction_applied"] == 0.0
    assert suppressed.diagnostics["reliability_proposal_authority"] == 0.0
    assert suppressed.diagnostics["reliability_proposal_fallback_fraction"] == 1.0


def _proposal_advantage_controller(
    mode="episode_latched_veto",
    standard_fallback=False,
):
    base = _reliable_controller(ReliabilityDirectPolicy(0.0))
    config = dict(base.paper_rl_driven_config.__dict__)
    config["proposal_advantage_gate"] = {
        "enabled": True,
        "mode": mode,
        "relative_disadvantage_margin": 0.0,
        "consecutive_disadvantages": 3,
    }
    config["standard_fallback_on_advantage_veto"] = bool(
        standard_fallback
    )
    return PaperRLDrivenMppiController(
        base.dynamics,
        base.state_spec,
        base.action_spec,
        base.config,
        sampling_prior=ReliabilityDirectPolicy(0.0),
        paper_rl_driven_config=config,
    )


def test_proposal_advantage_veto_removes_guided_share_centre_and_variance():
    controller = _proposal_advantage_controller()
    for _ in range(3):
        controller.proposal_advantage_gate.update(
            110.0, 100.0, observed=True
        )

    result = controller.plan(_observation(), PointGoal(1.0, 0.0))
    diagnostics = result.diagnostics

    assert diagnostics["paper_guided_unique_sequences"] == 0
    assert diagnostics["paper_gaussian_samples_per_iteration"] == 20
    assert diagnostics["reliability_guided_fraction_applied"] == 0.0
    assert diagnostics["reliability_guided_fraction_next"] == 0.0
    assert diagnostics["reliability_proposal_authority"] == 0.0
    assert diagnostics["reliability_proposal_fallback_fraction"] == 1.0
    assert diagnostics[
        "reliability_proposal_advantage_authority_applied"
    ] == 0.0
    assert diagnostics["reliability_proposal_advantage_latched"]

    controller.reset(seed=20260718)
    reset = controller.plan(_observation(), PointGoal(1.0, 0.0))
    assert reset.diagnostics["paper_guided_unique_sequences"] > 0
    assert reset.diagnostics[
        "reliability_proposal_advantage_authority_applied"
    ] == 1.0


def test_proposal_advantage_shadow_never_changes_actor_authority():
    controller = _proposal_advantage_controller(mode="shadow")
    for _ in range(3):
        controller.proposal_advantage_gate.update(
            110.0, 100.0, observed=True
        )

    result = controller.plan(_observation(), PointGoal(1.0, 0.0))
    diagnostics = result.diagnostics

    assert diagnostics["paper_guided_unique_sequences"] > 0
    assert diagnostics["reliability_proposal_authority"] > 0.0
    assert diagnostics["reliability_proposal_advantage_gate_shadow"]
    assert diagnostics[
        "reliability_proposal_advantage_would_authority_next"
    ] == 0.0
    assert diagnostics[
        "reliability_proposal_advantage_authority_applied"
    ] == 1.0


def test_same_cycle_cost_filter_excludes_disadvantaged_guided_elites(
    monkeypatch,
):
    controller = _controller(AuditableDirectPolicy())
    config = dict(controller.paper_rl_driven_config.__dict__)
    config["same_cycle_guided_cost_filter"] = True
    controller = PaperRLDrivenMppiController(
        controller.dynamics,
        controller.state_spec,
        controller.action_spec,
        controller.config,
        sampling_prior=AuditableDirectPolicy(),
        paper_rl_driven_config=config,
    )

    def synthetic_cost(trajectories, controls, *args, **kwargs):
        del trajectories, args, kwargs
        costs = np.zeros(len(controls), dtype=np.float64)
        costs[:5] = 100.0
        return costs

    monkeypatch.setattr(controller, "_cost", synthetic_cost)
    result = controller.plan(_observation(), PointGoal(1.0, 0.0))

    assert result.diagnostics[
        "paper_same_cycle_guided_cost_filter_iterations"
    ] == 3
    assert result.diagnostics[
        "paper_same_cycle_guided_filtered_candidates"
    ] == 15
    assert result.diagnostics["paper_guided_elite_count"] == 0


def test_same_cycle_filter_keeps_gaussian_control_actor_independent(
    monkeypatch,
):
    def make_controller(mean):
        base = _controller(FixedMeanDirectPolicy(mean))
        config = dict(base.paper_rl_driven_config.__dict__)
        config["same_cycle_guided_cost_filter"] = True
        return PaperRLDrivenMppiController(
            base.dynamics,
            base.state_spec,
            base.action_spec,
            base.config,
            sampling_prior=FixedMeanDirectPolicy(mean),
            paper_rl_driven_config=config,
        )

    def synthetic_cost(trajectories, controls, *args, **kwargs):
        del trajectories, args, kwargs
        costs = np.sum(np.asarray(controls) ** 2, axis=(1, 2))
        costs[:5] += 1000.0
        return costs

    left = make_controller((0.05, -0.80))
    right = make_controller((0.45, 0.80))
    monkeypatch.setattr(left, "_cost", synthetic_cost)
    monkeypatch.setattr(right, "_cost", synthetic_cost)

    left_result = left.plan(_observation(), PointGoal(1.0, 0.0))
    right_result = right.plan(_observation(), PointGoal(1.0, 0.0))

    np.testing.assert_allclose(
        left_result.control_sequence,
        right_result.control_sequence,
        rtol=0.0,
        atol=1.0e-12,
    )
    for result in (left_result, right_result):
        assert result.diagnostics["paper_guided_elite_count"] == 0
        assert result.diagnostics[
            "paper_same_cycle_gaussian_actor_isolated"
        ]
        assert result.diagnostics["reliability_proposal_authority"] == 0.0
        assert result.diagnostics[
            "reliability_requested_proposal_authority"
        ] > 0.0


def test_proposal_advantage_veto_can_fall_back_to_exact_standard_mppi():
    policy = ReliabilityDirectPolicy(0.0)
    paper = _proposal_advantage_controller(standard_fallback=True)
    paper.sampling_prior = policy
    standard = MppiController(
        paper.dynamics,
        paper.state_spec,
        paper.action_spec,
        MppiConfig(
            horizon=paper.config.horizon,
            num_samples=(
                paper.config.num_samples
                * paper.paper_rl_driven_config.iterations
            ),
            dt=paper.config.dt,
            noise_sigma=paper.config.noise_sigma,
            importance_sampling_correction=True,
            seed=paper.config.seed,
        ),
        sampling_prior=GoalWarmStartPrior(),
    )
    paper.reset(seed=20260718)
    standard.reset(seed=20260718)
    for _ in range(3):
        paper.proposal_advantage_gate.update(
            110.0, 100.0, observed=True
        )

    for timestamp in (0.0, 0.1):
        observation = RobotObservation(
            timestamp=timestamp,
            pose=Pose2D(0.0, 0.0, 0.0),
            twist=Twist2D(0.0, 0.0),
        )
        paper_result = paper.plan(observation, PointGoal(1.0, 0.0))
        standard_result = standard.plan(observation, PointGoal(1.0, 0.0))

        np.testing.assert_array_equal(
            paper_result.proposed_control.values,
            standard_result.proposed_control.values,
        )
        np.testing.assert_array_equal(
            paper_result.control_sequence,
            standard_result.control_sequence,
        )
        np.testing.assert_array_equal(
            paper_result.predicted_trajectory,
            standard_result.predicted_trajectory,
        )
        assert paper_result.diagnostics["paper_standard_fallback_active"]
        assert paper_result.diagnostics["paper_total_rollouts"] == (
            paper.config.num_samples
            * paper.paper_rl_driven_config.iterations
        )
        assert paper_result.diagnostics[
            "reliability_proposal_advantage_authority_applied"
        ] == 0.0

    assert policy.distribution_calls == 0
    assert policy.sample_calls == 0
    assert policy.terminal_calls == 0


def test_standard_advantage_fallback_requires_active_veto():
    base = _reliable_controller(ReliabilityDirectPolicy(0.0))
    config = dict(base.paper_rl_driven_config.__dict__)
    config["standard_fallback_on_advantage_veto"] = True
    config["proposal_advantage_gate"] = {
        "enabled": True,
        "mode": "shadow",
        "consecutive_disadvantages": 3,
    }

    with np.testing.assert_raises_regex(
        ValueError,
        "standard fallback requires an active episode-latched",
    ):
        PaperRLDrivenMppiController(
            base.dynamics,
            base.state_spec,
            base.action_spec,
            base.config,
            sampling_prior=ReliabilityDirectPolicy(0.0),
            paper_rl_driven_config=config,
        )


def test_terminal_guidance_floor_preserves_completion_candidate_share():
    low = _reliable_controller(
        ReliabilityDirectPolicy(8.0),
        terminal_guidance=True,
    )
    far_target = PointGoal(1.0, 0.0)
    low.plan(_observation(), far_target)
    far = low.plan(_observation(), far_target)
    assert far.diagnostics["terminal_guidance_floor_enabled"]
    assert not far.diagnostics["terminal_guidance_floor_active"]
    assert far.diagnostics["reliability_guided_fraction_applied"] == 0.0
    assert far.diagnostics["paper_guided_unique_sequences"] == 0

    low.reset(seed=20260718)
    near_target = PointGoal(0.4, 0.0)
    low.plan(_observation(), near_target)
    near = low.plan(_observation(), near_target)
    assert near.diagnostics["terminal_guidance_floor_active"]
    assert near.diagnostics["reliability_guided_fraction_raw_applied"] == 0.3
    assert near.diagnostics["reliability_guided_fraction_raw_next"] == 0.0
    assert near.diagnostics["reliability_guided_fraction_applied"] == 0.3
    assert near.diagnostics["reliability_guided_fraction_next"] == 0.3
    assert near.diagnostics["paper_guided_unique_sequences"] == 6


def test_terminal_guidance_floor_does_not_override_tracking_lookahead():
    controller = _reliable_controller(
        ReliabilityDirectPolicy(8.0),
        terminal_guidance=True,
    )
    tracking = ReferenceTarget(
        Pose2D(0.4, 0.0, 0.0),
        position_tolerance=0.0,
        is_terminal=False,
        phase="tracking",
    )
    floor, diagnostics = controller._completion_preserving_guidance(
        np.zeros(5), tracking
    )
    assert floor == 0.0
    assert not diagnostics["terminal_guidance_terminal_phase"]
    assert not diagnostics["terminal_guidance_floor_active"]

    approach = ReferenceTarget(
        Pose2D(0.4, 0.0, 0.0),
        position_tolerance=0.0,
        is_terminal=False,
        phase="terminal_approach",
    )
    floor, diagnostics = controller._completion_preserving_guidance(
        np.zeros(5), approach
    )
    assert floor == 0.3
    assert diagnostics["terminal_guidance_terminal_phase"]
    assert diagnostics["terminal_guidance_floor_active"]


def test_completion_handover_is_continuous_and_terminal_only():
    controller = _reliable_controller(ReliabilityDirectPolicy(0.0))
    config = dict(controller.paper_rl_driven_config.__dict__)
    config.update({
        "completion_handover_full_fallback_distance": 0.4,
        "completion_handover_full_rl_distance": 0.8,
    })
    controller = PaperRLDrivenMppiController(
        controller.dynamics,
        controller.state_spec,
        controller.action_spec,
        controller.config,
        sampling_prior=ReliabilityDirectPolicy(0.0),
        paper_rl_driven_config=config,
    )

    near = ReferenceTarget(
        Pose2D(0.3, 0.0, 0.0),
        position_tolerance=0.2,
        is_terminal=True,
        phase="terminal",
    )
    authority, diagnostics = controller._completion_handover(
        np.zeros(5), near
    )
    assert authority == 0.0
    assert diagnostics["completion_handover_enabled"]

    middle = ReferenceTarget(
        Pose2D(0.6, 0.0, 0.0),
        position_tolerance=0.2,
        is_terminal=True,
        phase="terminal",
    )
    authority, _ = controller._completion_handover(np.zeros(5), middle)
    assert np.isclose(authority, 0.5)

    far = ReferenceTarget(
        Pose2D(1.0, 0.0, 0.0),
        position_tolerance=0.2,
        is_terminal=True,
        phase="terminal",
    )
    authority, _ = controller._completion_handover(np.zeros(5), far)
    assert authority == 1.0

    tracking = ReferenceTarget(
        Pose2D(0.3, 0.0, 0.0),
        position_tolerance=0.0,
        is_terminal=False,
        phase="tracking",
    )
    authority, diagnostics = controller._completion_handover(
        np.zeros(5), tracking
    )
    assert authority == 1.0
    assert not diagnostics["completion_handover_enabled"]


def test_completion_handover_attenuates_guidance_and_terminal_value():
    controller = _reliable_controller(ReliabilityDirectPolicy(0.0))
    config = dict(controller.paper_rl_driven_config.__dict__)
    config.update({
        "terminal_value_weight": 0.5,
        "completion_handover_full_fallback_distance": 0.4,
        "completion_handover_full_rl_distance": 0.8,
    })
    controller = PaperRLDrivenMppiController(
        controller.dynamics,
        controller.state_spec,
        controller.action_spec,
        controller.config,
        sampling_prior=ReliabilityDirectPolicy(0.0),
        paper_rl_driven_config=config,
    )
    result = controller.plan(_observation(), PointGoal(0.3, 0.0))

    assert result.diagnostics["completion_handover_authority"] == 0.0
    assert result.diagnostics["paper_guided_unique_sequences"] == 0
    assert result.diagnostics["terminal_value_completion_authority"] == 0.0
    assert result.diagnostics["terminal_value_completion_handover_enabled"]


def test_counterfactual_proposal_gate_bounds_route_regressing_actor():
    base = _reliable_controller(ReliabilityDirectPolicy(0.0))
    config = dict(base.paper_rl_driven_config.__dict__)
    config.update({
        "counterfactual_proposal_gate_enabled": True,
        "counterfactual_progress_soft_m": 0.0,
        "counterfactual_progress_hard_m": -0.15,
        "counterfactual_cross_track_weight": 0.5,
    })
    controller = PaperRLDrivenMppiController(
        base.dynamics,
        base.state_spec,
        base.action_spec,
        base.config,
        sampling_prior=ReliabilityDirectPolicy(0.0),
        paper_rl_driven_config=config,
    )
    reference = PolylineReference(((0.0, 0.0), (2.0, 0.0)))
    baseline = np.tile((0.5, 0.0), (controller.config.horizon, 1))
    actor_terminal = np.asarray((0.10, 0.40, 0.0, 0.0, 0.0))

    authority, diagnostics = controller._counterfactual_proposal_gate(
        np.zeros(5), actor_terminal, baseline, reference
    )

    assert authority == 0.0
    assert diagnostics["reliability_counterfactual_enabled"]
    assert diagnostics["reliability_counterfactual_advantage"] < -0.15
    assert (
        diagnostics["reliability_counterfactual_baseline_progress"]
        > diagnostics["reliability_counterfactual_actor_progress"]
    )


def test_counterfactual_proposal_gate_is_polyline_only():
    controller = _reliable_controller(ReliabilityDirectPolicy(0.0))
    authority, diagnostics = controller._counterfactual_proposal_gate(
        np.zeros(5), np.zeros(5), np.zeros((5, 2)), PointGoal(1.0, 0.0)
    )
    assert authority == 1.0
    assert not diagnostics["reliability_counterfactual_enabled"]


def test_counterfactual_proposal_gate_cannot_leak_into_simple_combination():
    controller = _controller(ReliabilityDirectPolicy(0.0))
    config = dict(controller.paper_rl_driven_config.__dict__)
    config["counterfactual_proposal_gate_enabled"] = True
    controller = PaperRLDrivenMppiController(
        controller.dynamics,
        controller.state_spec,
        controller.action_spec,
        controller.config,
        sampling_prior=ReliabilityDirectPolicy(0.0),
        paper_rl_driven_config=config,
    )
    reference = PolylineReference(((0.0, 0.0), (2.0, 0.0)))
    authority, diagnostics = controller._counterfactual_proposal_gate(
        np.zeros(5), np.zeros(5), np.zeros((5, 2)), reference
    )
    assert authority == 1.0
    assert not diagnostics["reliability_counterfactual_enabled"]


def test_paper_diagnostics_preserve_reference_phase():
    controller = _reliable_controller(
        ReliabilityDirectPolicy(8.0),
        terminal_guidance=True,
    )
    target = PointGoal(0.4, 0.0, position_tolerance=0.1)
    result = controller.plan(_observation(), target)
    assert result.diagnostics["target_phase"] == "terminal"
    assert result.diagnostics["target_is_terminal"]
    assert result.diagnostics["target_x"] == 0.4


def test_conservative_terminal_uses_candidate_confidence_and_safe_fallback():
    trusted = _reliable_controller(
        ReliabilityDirectPolicy(0.0, critic_disagreement=0.0),
        conservative_terminal=True,
    ).plan(_observation(), PointGoal(1.0, 0.0))
    untrusted = _reliable_controller(
        ReliabilityDirectPolicy(8.0, critic_disagreement=3.0),
        conservative_terminal=True,
    ).plan(_observation(), PointGoal(1.0, 0.0))

    assert trusted.diagnostics["terminal_value_conservative_enabled"]
    assert trusted.diagnostics["terminal_value_authority_mean"] == 1.0
    assert trusted.diagnostics["terminal_value_cost_mean"] < 0.0
    assert untrusted.diagnostics["terminal_value_authority_mean"] == 0.0
    # The incremental critic contribution disappears. The unchanged MPPI
    # geometric terminal remains the safe fallback in the base cost.
    assert untrusted.diagnostics["terminal_value_cost_mean"] == 0.0
    assert (
        untrusted.diagnostics["terminal_value_safe_fallback"]
        == "existing_mppi_geometric_terminal"
    )


def test_conservative_terminal_authority_is_capped_by_causal_dynamics_confidence():
    controller = _reliable_controller(
        ReliabilityDirectPolicy(0.0, critic_disagreement=0.0),
        conservative_terminal=True,
    )
    trajectories = np.zeros((2, controller.config.horizon + 1, 5))
    controls = np.zeros((2, controller.config.horizon, 2))

    costs, diagnostics = controller._paper_terminal_cost(
        trajectories,
        controls,
        _observation(),
        PointGoal(1.0, 0.0),
        causal_dynamics_confidence=0.0,
    )

    np.testing.assert_allclose(costs, 0.0)
    assert diagnostics["terminal_value_raw_authority_mean"] == 1.0
    assert diagnostics["terminal_value_authority_mean"] == 0.0
    assert diagnostics["terminal_value_causal_dynamics_cap"] == 0.0
    assert diagnostics["terminal_value_causal_cap_active_fraction"] == 1.0
