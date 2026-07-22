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
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig
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

    def synthetic_margins(trajectories, _reference, **kwargs):
        del kwargs
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
