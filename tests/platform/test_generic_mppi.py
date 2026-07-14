import numpy as np

from mobile_robot_mppi.core.references import PointGoal, WaypointReference
from mobile_robot_mppi.core.spaces import ActionSpec, body_velocity_action, dynamic_unicycle_state, unicycle_state
from mobile_robot_mppi.core.types import (
    ControlCommand,
    Pose2D,
    RobotObservation,
    SafetyDecision,
    Twist2D,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, LegacyUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior, PriorOutput, RLPolicyPrior


def observation():
    return RobotObservation(0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(0.0, 0.0))


def test_three_state_and_five_state_mppi_share_controller():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(horizon=4, num_samples=8, dt=0.1, noise_sigma=(0.05, 0.1), seed=3)
    controllers = (
        MppiController(LegacyUnicyclePrediction(), unicycle_state(), action, config),
        MppiController(DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config),
    )
    for controller in controllers:
        result = controller.plan(observation(), PointGoal(1.0, 0.5))
        assert result.control_sequence.shape == (4, 2)
        assert result.predicted_trajectory.shape[0] == 5
        assert np.isfinite(result.predicted_trajectory).all()


def test_rl_prior_is_framework_independent():
    action = body_velocity_action((0.0, 0.4), 1.0)

    def policy(obs, reference, horizon, spec):
        del obs, reference
        return np.zeros((horizon, spec.dimension))

    prior = RLPolicyPrior(policy, "unit_policy")
    output = prior.propose(observation(), PointGoal(1.0, 0.0), 6, action)
    assert output.mean.shape == (6, 2)
    assert output.metadata["policy_id"] == "unit_policy"


def test_mppi_rollout_supports_more_than_two_control_dimensions():
    class ThreeControlDynamics:
        state_dim = 3
        control_dim = 3

        def derivative(self, state, control, time=None):
            del time
            state = np.asarray(state)
            control = np.asarray(control)
            return np.stack((
                control[..., 0] * np.cos(state[..., 2]),
                control[..., 0] * np.sin(state[..., 2]),
                control[..., 1] + 0.1 * control[..., 2],
            ), axis=-1)

    action = ActionSpec(
        ("v_cmd", "omega_cmd", "aux"),
        np.asarray((0.0, -1.0, -0.5)),
        np.asarray((0.4, 1.0, 0.5)),
    )
    config = MppiConfig(horizon=3, num_samples=6, dt=0.1, noise_sigma=(0.05, 0.1, 0.1))
    controller = MppiController(ThreeControlDynamics(), unicycle_state(), action, config)
    result = controller.plan(observation(), PointGoal(1.0, 0.0))
    assert result.control_sequence.shape == (3, 3)


def test_importance_sampling_cost_matches_gaussian_mppi_correction():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=2,
        num_samples=2,
        dt=0.1,
        temperature=3.0,
        noise_sigma=(0.1, 0.2),
        importance_sampling_correction=True,
    )
    controller = MppiController(LegacyUnicyclePrediction(), unicycle_state(), action, config)
    nominal = np.asarray(((0.2, 0.1), (0.3, -0.2)))
    perturbations = np.asarray((
        ((0.01, 0.02), (-0.03, 0.04)),
        ((-0.02, 0.01), (0.05, -0.01)),
    ))
    covariance = np.diag((0.01, 0.04))
    expected = 3.0 * config.control_weight * np.einsum(
        "ha,ab,kha->k", nominal, np.linalg.inv(covariance), perturbations
    )
    actual = controller._importance_sampling_cost(nominal, perturbations, covariance)
    np.testing.assert_allclose(actual, expected)


def test_invalid_prior_covariance_is_rejected():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(horizon=2, num_samples=2, dt=0.1, noise_sigma=(0.1, 0.2))
    controller = MppiController(LegacyUnicyclePrediction(), unicycle_state(), action, config)
    prior = PriorOutput(np.zeros((2, 2)), np.asarray(((1.0, 2.0), (0.0, 1.0))), {})
    with np.testing.assert_raises_regex(ValueError, "symmetric"):
        controller._sampling_covariance(prior)


def test_safety_stop_feedback_clears_translation_but_preserves_turning_plan():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=6,
        num_samples=8,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        safety_recovery_prefix_steps=3,
    )
    controller = MppiController(LegacyUnicyclePrediction(), unicycle_state(), action, config)
    controller.previous_sequence = np.tile(np.asarray((0.3, 0.4)), (6, 1))
    decision = SafetyDecision(
        proposed_control=ControlCommand(np.asarray((0.3, 0.4))),
        executed_control=ControlCommand(np.asarray((0.0, 0.4))),
        overridden=True,
        reason="front_emergency_stop",
    )
    controller.observe_safety_decision(decision)
    np.testing.assert_allclose(controller.previous_action, (0.0, 0.4))
    np.testing.assert_allclose(controller.previous_sequence[:3, 0], 0.0)
    np.testing.assert_allclose(controller.previous_sequence[:, 1], 0.4)
    assert controller._safety_blocked


def test_terminal_velocity_cost_penalizes_arrival_with_motion():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=2,
        num_samples=2,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        goal_running_weight=0.0,
        goal_terminal_weight=0.0,
        terminal_velocity_weight=3.0,
        terminal_yaw_rate_weight=2.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    trajectories = np.zeros((2, 3, 5), dtype=np.float64)
    trajectories[1, -1, 3:] = (2.0, 1.0)
    controls = np.zeros((2, 2, 2), dtype=np.float64)
    target = PointGoal(0.0, 0.0).target_at(0.0, np.zeros(5))
    costs = controller._cost(trajectories, controls, target, ())
    assert costs[0] == 0.0
    assert costs[1] == 14.0


def test_goal_prior_turns_in_place_when_target_is_far_off_heading():
    action = body_velocity_action((0.0, 0.4), 1.0)
    prior = GoalWarmStartPrior(
        v_gain=0.8, yaw_gain=1.2, translation_heading_gate_rad=0.9
    )
    output = prior.propose(observation(), PointGoal(0.0, 1.0), 4, action)
    np.testing.assert_allclose(output.mean[:, 0], 0.0)
    assert np.all(output.mean[:, 1] > 0.0)
    assert output.metadata["translation_alignment"] == 0.0


def test_goal_prior_restores_translation_smoothly_when_aligned():
    action = body_velocity_action((0.0, 0.4), 1.0)
    prior = GoalWarmStartPrior(
        v_gain=0.8, yaw_gain=1.2, translation_heading_gate_rad=0.9
    )
    output = prior.propose(observation(), PointGoal(1.0, 0.0), 4, action)
    np.testing.assert_allclose(output.mean[:, 0], 0.4)
    np.testing.assert_allclose(output.mean[:, 1], 0.0)
    assert output.metadata["translation_alignment"] == 1.0


def test_terminal_heading_gate_does_not_freeze_intermediate_waypoint():
    action = body_velocity_action((0.0, 0.4), 1.0)
    prior = GoalWarmStartPrior(
        v_gain=0.8,
        yaw_gain=1.2,
        translation_heading_gate_rad=0.9,
        translation_heading_gate_terminal_only=True,
    )
    reference = WaypointReference(((0.5, 0.866), (1.0, 1.0)), tolerance=0.1)
    output = prior.propose(observation(), reference, 4, action)
    assert np.all(output.mean[:, 0] > 0.0)
    assert not output.metadata["translation_heading_gate_active"]
    assert not output.metadata["target_is_terminal"]
