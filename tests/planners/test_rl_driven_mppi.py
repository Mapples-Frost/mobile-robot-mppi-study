import numpy as np

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import body_velocity_action, dynamic_unicycle_state
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.planning.rl_driven_mppi import RLDrivenMppiController
from mobile_robot_mppi.policies.priors import (
    PriorOutput,
    ProposalDistribution,
)


class AuditablePrior:
    def __init__(self, terminal_scale=1.0):
        self.terminal_scale = float(terminal_scale)
        self.terminal_calls = 0

    def reset(self):
        self.terminal_calls = 0

    def propose(self, observation, reference, horizon, action_spec):
        del observation, reference
        base = np.tile((0.10, 0.0), (horizon, 1)).astype(np.float64)
        learned = np.tile((0.32, 0.15), (horizon, 1)).astype(np.float64)
        return PriorOutput(
            base,
            metadata={"type": "auditable"},
            proposals=(
                ProposalDistribution("rl", learned),
                ProposalDistribution("base", base),
            ),
        )

    def terminal_value(
        self,
        terminal_states,
        terminal_controls,
        observation,
        target,
        state_spec,
        horizon_dt,
        critic_source="target",
    ):
        del terminal_controls, observation, target, horizon_dt
        self.terminal_calls += 1
        values = self.terminal_scale * terminal_states[:, state_spec.index("x")]
        return values, {
            "terminal_q_mean": float(np.mean(values)),
            "terminal_q_disagreement_mean": 0.0,
            "terminal_critic_source": critic_source,
        }


def _observation():
    return RobotObservation(
        0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(0.0, 0.0)
    )


def _controller(prior, terminal_weight=0.0, samples=31):
    action = body_velocity_action((0.0, 0.5), 1.0)
    config = MppiConfig(
        horizon=6,
        num_samples=samples,
        dt=0.1,
        noise_sigma=(0.08, 0.20),
        seed=41,
    )
    return RLDrivenMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        action,
        config,
        sampling_prior=prior,
        rl_driven_config={
            "iterations": 2,
            "rl_fraction": 0.30,
            "shifted_fraction": 0.40,
            "base_fraction": 0.30,
            "elite_fraction": 0.20,
            "terminal_value_weight": terminal_weight,
        },
    )


def test_hybrid_sources_preserve_exact_total_rollout_budget():
    controller = _controller(AuditablePrior(), samples=31)
    result = controller.plan(_observation(), PointGoal(1.0, 0.0))
    diagnostics = result.diagnostics

    assert diagnostics["optimizer"] == "rl_driven"
    assert diagnostics["rl_driven_total_rollouts"] == 31
    assert diagnostics["rl_driven_iteration_budgets"] == [16, 15]
    assert (
        diagnostics["rl_source_samples"]
        + diagnostics["shifted_source_samples"]
        + diagnostics["base_source_samples"]
        == 31
    )
    assert (
        diagnostics["rl_elite_count"]
        + diagnostics["shifted_elite_count"]
        + diagnostics["base_elite_count"]
        == 7
    )
    assert result.control_sequence.shape == (6, 2)
    assert result.predicted_trajectory.shape == (7, 5)
    assert np.isfinite(result.control_sequence).all()


def test_terminal_critic_is_strictly_opt_in_and_auditable():
    disabled_prior = AuditablePrior()
    disabled = _controller(disabled_prior, terminal_weight=0.0)
    disabled_result = disabled.plan(_observation(), PointGoal(1.0, 0.0))
    assert disabled_prior.terminal_calls == 0
    assert not disabled_result.diagnostics["terminal_value_enabled"]

    enabled_prior = AuditablePrior()
    enabled = _controller(enabled_prior, terminal_weight=0.5)
    enabled_result = enabled.plan(_observation(), PointGoal(1.0, 0.0))
    assert enabled_prior.terminal_calls == 2
    assert enabled_result.diagnostics["terminal_value_enabled"]
    assert enabled_result.diagnostics["terminal_value_weight"] == 0.5
    assert np.isfinite(enabled_result.diagnostics["terminal_q_mean"])


def test_standard_mppi_ignores_optional_proposal_families():
    action = body_velocity_action((0.0, 0.5), 1.0)
    config = MppiConfig(
        horizon=5,
        num_samples=24,
        dt=0.1,
        noise_sigma=(0.08, 0.20),
        seed=73,
    )
    mean = np.tile((0.12, 0.0), (5, 1)).astype(np.float64)

    class Plain:
        def propose(self, *args):
            return PriorOutput(mean.copy(), metadata={"type": "plain"})

    class WithUnusedProposals:
        def propose(self, *args):
            return PriorOutput(
                mean.copy(),
                metadata={"type": "plain"},
                proposals=(
                    ProposalDistribution("rl", np.full_like(mean, 0.49)),
                    ProposalDistribution("base", np.zeros_like(mean)),
                ),
            )

    first = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config,
        sampling_prior=Plain(),
    ).plan(_observation(), PointGoal(1.0, 0.0))
    second = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config,
        sampling_prior=WithUnusedProposals(),
    ).plan(_observation(), PointGoal(1.0, 0.0))

    np.testing.assert_array_equal(first.control_sequence, second.control_sequence)
    np.testing.assert_array_equal(
        first.predicted_trajectory, second.predicted_trajectory
    )
