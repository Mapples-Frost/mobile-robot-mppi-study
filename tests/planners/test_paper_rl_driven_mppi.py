import numpy as np

from mobile_robot_mppi.core.references import PointGoal
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


def _reliable_controller(policy, conservative_terminal=False):
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


def test_reliability_hss_applies_authority_on_next_control_cycle():
    high = _reliable_controller(ReliabilityDirectPolicy(0.0))
    first = high.plan(_observation(), PointGoal(1.0, 0.0))
    second = high.plan(_observation(), PointGoal(1.0, 0.0))

    assert first.diagnostics["paper_guided_unique_sequences"] == 6
    assert first.diagnostics["reliability_level"] == "high"
    assert first.diagnostics["reliability_guided_fraction_next"] == 0.6
    assert second.diagnostics["paper_guided_unique_sequences"] == 12
    assert second.diagnostics["reliability_guided_fraction_applied"] == 0.6

    low = _reliable_controller(ReliabilityDirectPolicy(8.0))
    low.plan(_observation(), PointGoal(1.0, 0.0))
    suppressed = low.plan(_observation(), PointGoal(1.0, 0.0))
    assert suppressed.diagnostics["paper_guided_unique_sequences"] == 0
    assert suppressed.diagnostics["reliability_guided_fraction_applied"] == 0.0


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
