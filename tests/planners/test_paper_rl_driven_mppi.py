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
