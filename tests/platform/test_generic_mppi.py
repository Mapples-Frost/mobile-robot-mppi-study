import numpy as np
import pytest

from mobile_robot_mppi.core.references import PointGoal, PolylineReference, WaypointReference
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
from mobile_robot_mppi.policies.priors import (
    FixedCovariancePrior,
    GoalWarmStartPrior,
    PriorOutput,
    RLPolicyPrior,
)


def observation():
    return RobotObservation(0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(0.0, 0.0))


def test_fixed_covariance_prior_preserves_mean_and_scales_standard_deviation():
    action = body_velocity_action((0.0, 0.7), 1.0)
    baseline = GoalWarmStartPrior()
    wrapped = FixedCovariancePrior(
        baseline, noise_sigma=(0.2, 0.4), scale=(1.75, 0.5)
    )
    expected = baseline.propose(observation(), PointGoal(1.0, 0.5), 5, action)
    actual = wrapped.propose(observation(), PointGoal(1.0, 0.5), 5, action)

    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_allclose(
        actual.covariance,
        np.diag(np.square([0.2 * 1.75, 0.4 * 0.5])),
    )
    assert actual.metadata["type"] == "fixed_covariance"
    assert actual.metadata["covariance_scale"] == [1.75, 0.5]


def test_fixed_covariance_prior_rejects_invalid_scale():
    with pytest.raises(ValueError, match="must be positive"):
        FixedCovariancePrior(GoalWarmStartPrior(), (0.2, 0.4), (1.0, 0.0))


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


def test_terminal_reference_caps_all_mppi_translation_samples():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=5,
        num_samples=32,
        dt=0.1,
        noise_sigma=(0.2, 0.3),
        terminal_translation_speed_limit=0.11,
        seed=13,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    result = controller.plan(observation(), PointGoal(2.0, 0.0))
    assert np.max(result.control_sequence[:, 0]) <= 0.11 + 1e-12
    assert result.diagnostics["terminal_speed_limit_active"]


def test_terminal_heading_gate_stops_translation_but_preserves_turning():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=5,
        num_samples=32,
        dt=0.1,
        noise_sigma=(0.2, 0.3),
        terminal_translation_heading_gate_rad=0.4,
        terminal_alignment_yaw_gain=1.0,
        seed=23,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    misaligned = RobotObservation(
        0.0, Pose2D(0.0, 0.0, np.pi / 2.0), Twist2D(0.2, 0.0)
    )
    result = controller.plan(misaligned, PointGoal(2.0, 0.0))

    assert result.control_sequence[0, 0] == 0.0
    assert result.proposed_control.values[0] == 0.0
    assert result.diagnostics["terminal_heading_gate_active"]
    assert result.diagnostics["terminal_translation_scale"] == 0.0
    assert result.diagnostics["terminal_alignment_active"]
    assert result.proposed_control.values[1] < 0.0


def test_terminal_control_radius_limits_alignment_to_goal_neighbourhood():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=5,
        num_samples=32,
        dt=0.1,
        noise_sigma=(0.2, 0.3),
        terminal_translation_speed_limit=0.11,
        terminal_translation_heading_gate_rad=0.4,
        terminal_alignment_yaw_gain=1.0,
        terminal_control_radius=0.8,
        seed=23,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    misaligned = RobotObservation(
        0.0, Pose2D(0.0, 0.0, np.pi / 2.0), Twist2D(0.2, 0.0)
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


def test_terminal_heading_gate_is_backward_compatible_when_disabled():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=5,
        num_samples=32,
        dt=0.1,
        noise_sigma=(0.2, 0.3),
        seed=23,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    misaligned = RobotObservation(
        0.0, Pose2D(0.0, 0.0, np.pi / 2.0), Twist2D(0.2, 0.0)
    )
    result = controller.plan(misaligned, PointGoal(2.0, 0.0))

    assert not result.diagnostics["terminal_heading_gate_active"]
    assert not result.diagnostics["terminal_alignment_active"]
    assert result.diagnostics["terminal_translation_scale"] == 1.0
    assert np.isclose(
        result.diagnostics["target_bearing_error"], -np.pi / 2.0
    )


def test_terminal_bearing_cost_prefers_heading_toward_goal():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=1,
        num_samples=2,
        dt=0.1,
        noise_sigma=(0.2, 0.3),
        goal_running_weight=0.0,
        goal_terminal_weight=0.0,
        terminal_bearing_weight=2.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    trajectories = np.zeros((2, 2, 5), dtype=np.float64)
    trajectories[:, :, 0] = 0.5
    trajectories[1, -1, 2] = np.pi / 2.0
    controls = np.zeros((2, 1, 2), dtype=np.float64)
    target = PointGoal(1.0, 0.0).target_at(0.0, trajectories[0, 0])

    costs = controller._cost(trajectories, controls, target, ())

    np.testing.assert_allclose(costs[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(costs[1], 2.0 * (np.pi / 2.0) ** 2)


def test_path_preview_cost_tracks_future_polyline_poses_not_one_static_target():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=2,
        num_samples=2,
        dt=1.0,
        noise_sigma=(0.2, 0.3),
        goal_running_weight=1.0,
        goal_terminal_weight=5.0,
        path_preview_enabled=True,
        path_preview_speed_mps=1.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), lookahead_distance=0.2
    )
    target = reference.target_at(0.0, np.zeros(5))
    trajectories = np.zeros((2, 3, 5), dtype=np.float64)
    # Candidate 0 follows the time-indexed future route. Candidate 1 remains
    # at the old single lookahead point and would win under the legacy cost.
    trajectories[0, 1, :2] = (1.0, 0.2)
    trajectories[0, 2, :2] = (1.0, 1.0)
    trajectories[1, 1:, :2] = (0.2, 0.0)
    controls = np.zeros((2, 2, 2), dtype=np.float64)

    costs = controller._cost(
        trajectories, controls, target, (), reference=reference
    )

    assert costs[0] < costs[1]


def test_path_preview_disabled_preserves_legacy_static_target_cost():
    action = body_velocity_action((0.0, 0.7), 1.0)
    config = MppiConfig(
        horizon=2,
        num_samples=1,
        dt=0.1,
        noise_sigma=(0.2, 0.3),
        path_preview_enabled=False,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    reference = PolylineReference(((0.0, 0.0), (1.0, 0.0)))
    target = reference.target_at(0.0, np.zeros(5))
    trajectories = np.zeros((1, 3, 5), dtype=np.float64)
    controls = np.zeros((1, 2, 2), dtype=np.float64)

    with_reference = controller._cost(
        trajectories, controls, target, (), reference=reference
    )
    legacy = controller._cost(trajectories, controls, target, ())

    np.testing.assert_array_equal(with_reference, legacy)


def test_mppi_reset_accepts_episode_seed_and_preserves_legacy_default():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=4,
        num_samples=8,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        seed=3,
    )
    controller = MppiController(
        LegacyUnicyclePrediction(), unicycle_state(), action, config
    )

    controller.reset(seed=91)
    episode_draw = controller.rng.normal(size=8)
    controller.reset(seed=91)
    np.testing.assert_array_equal(controller.rng.normal(size=8), episode_draw)

    controller.reset()
    configured_draw = controller.rng.normal(size=8)
    np.testing.assert_array_equal(
        configured_draw, np.random.RandomState(3).normal(size=8)
    )
    assert not np.array_equal(episode_draw, configured_draw)

    with np.testing.assert_raises_regex(ValueError, "reset seed"):
        controller.reset(seed=-1)


def test_mppi_preview_is_repeatable_and_does_not_advance_controller():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=4,
        num_samples=16,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        seed=17,
    )
    controller = MppiController(
        LegacyUnicyclePrediction(), unicycle_state(), action, config
    )
    controller.previous_sequence = np.tile((0.2, 0.1), (4, 1))
    controller.previous_action = np.asarray((0.1, -0.1), dtype=np.float64)
    rng_before = controller.rng.get_state()
    sequence_before = controller.previous_sequence.copy()
    action_before = controller.previous_action.copy()

    first = controller.preview_plan(observation(), PointGoal(1.0, 0.5))
    second = controller.preview_plan(observation(), PointGoal(1.0, 0.5))
    np.testing.assert_array_equal(first.control_sequence, second.control_sequence)
    np.testing.assert_array_equal(
        first.predicted_trajectory, second.predicted_trajectory
    )
    np.testing.assert_array_equal(controller.previous_sequence, sequence_before)
    np.testing.assert_array_equal(controller.previous_action, action_before)
    rng_after = controller.rng.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    assert first.diagnostics["preview"] is True

    actual = controller.plan(observation(), PointGoal(1.0, 0.5))
    np.testing.assert_array_equal(first.control_sequence, actual.control_sequence)
    np.testing.assert_array_equal(
        first.predicted_trajectory, actual.predicted_trajectory
    )
    assert actual.diagnostics["preview"] is False


def test_optional_mppi_component_profile_is_finite_and_nonnegative():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=4,
        num_samples=16,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        profile_components=True,
        seed=19,
    )
    controller = MppiController(
        LegacyUnicyclePrediction(), unicycle_state(), action, config
    )
    result = controller.plan(observation(), PointGoal(1.0, 0.5))
    fields = (
        "profile_planner_state_reference_ms",
        "profile_planner_prior_ms",
        "profile_mppi_sampling_ms",
        "profile_mppi_batch_rollout_ms",
        "profile_mppi_cost_ms",
        "profile_mppi_weighting_update_ms",
        "profile_mppi_final_rollout_ms",
        "profile_mppi_solve_total_ms",
    )
    for name in fields:
        assert np.isfinite(result.diagnostics[name])
        assert result.diagnostics[name] >= 0.0
    assert result.diagnostics["profile_mppi_solve_total_ms"] > 0.0


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


def test_public_sequence_evaluation_matches_existing_rollout_and_cost_contract():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=3,
        num_samples=8,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        seed=31,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    controller.previous_action = np.asarray((0.1, -0.2), dtype=np.float64)
    controls = np.asarray((
        ((0.2, 0.1), (0.3, -0.1), (0.2, 0.0)),
        ((0.1, -0.3), (0.0, 0.4), (0.2, 0.2)),
    ))
    initial_state = np.zeros(5, dtype=np.float64)
    target = PointGoal(1.0, 0.5).target_at(0.0, initial_state)
    action_before = controller.previous_action.copy()

    evaluation = controller.evaluate_control_sequences(
        initial_state, controls, target, ()
    )
    expected_trajectories = controller.rollout(initial_state, controls)
    expected_costs = controller._cost(
        expected_trajectories, controls, target, ()
    )

    np.testing.assert_array_equal(
        evaluation.trajectories, expected_trajectories
    )
    np.testing.assert_array_equal(evaluation.costs, expected_costs)
    np.testing.assert_array_equal(controller.previous_action, action_before)
    assert evaluation.trajectories.shape == (2, 4, 5)
    assert evaluation.costs.shape == (2,)


def test_public_sequence_evaluation_rejects_invalid_or_unbounded_candidates():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=3,
        num_samples=8,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
    )
    controller = MppiController(
        DynamicUnicyclePrediction(), dynamic_unicycle_state(), action, config
    )
    target = PointGoal(1.0, 0.0).target_at(0.0, np.zeros(5))

    with np.testing.assert_raises_regex(ValueError, "shape"):
        controller.evaluate_control_sequences(
            np.zeros(5), np.zeros((2, 2, 2)), target
        )
    outside = np.zeros((2, 3, 2), dtype=np.float64)
    outside[0, 0, 0] = 0.5
    with np.testing.assert_raises_regex(ValueError, "bounds"):
        controller.evaluate_control_sequences(np.zeros(5), outside, target)


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


def test_public_importance_weights_match_manual_softmax_and_are_normalized():
    action = body_velocity_action((0.0, 0.4), 1.0)
    config = MppiConfig(
        horizon=2,
        num_samples=3,
        dt=0.1,
        temperature=2.0,
        noise_sigma=(0.1, 0.2),
        importance_sampling_correction=False,
    )
    controller = MppiController(
        LegacyUnicyclePrediction(), unicycle_state(), action, config
    )
    nominal = np.asarray(((0.2, 0.0), (0.2, 0.0)))
    candidates = np.asarray((
        nominal,
        ((0.1, 0.1), (0.2, 0.1)),
        ((0.3, -0.1), (0.1, -0.1)),
    ))
    costs = np.asarray((3.0, 1.0, 2.0))
    result = controller.importance_weights(nominal, candidates, costs)
    expected = np.exp(-(costs - costs.min()) / 2.0)
    expected /= expected.sum()

    np.testing.assert_allclose(result.adjusted_costs, costs)
    np.testing.assert_allclose(result.weights, expected)
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.effective_sample_size == pytest.approx(
        1.0 / np.sum(expected ** 2)
    )


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


def test_goal_prior_caps_translation_only_for_terminal_target():
    action = body_velocity_action((0.0, 0.7), 1.0)
    prior = GoalWarmStartPrior(
        v_gain=1.0, yaw_gain=1.2, terminal_max_speed=0.12
    )
    terminal = prior.propose(observation(), PointGoal(2.0, 0.0), 4, action)
    np.testing.assert_allclose(terminal.mean[:, 0], 0.12)
    assert terminal.metadata["terminal_speed_cap_active"]

    reference = WaypointReference(((1.0, 0.0), (2.0, 0.0)), tolerance=0.1)
    tracking = prior.propose(observation(), reference, 4, action)
    np.testing.assert_allclose(tracking.mean[:, 0], 0.7)
    assert not tracking.metadata["terminal_speed_cap_active"]


def test_goal_prior_path_rollout_builds_a_structured_bend_sequence():
    action = body_velocity_action((0.0, 0.7), 1.25)
    prior = GoalWarmStartPrior(
        v_gain=1.0,
        yaw_gain=1.5,
        path_rollout_enabled=True,
        rollout_dt=0.1,
    )
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0)),
        lookahead_distance=0.45,
    )
    obs = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.75, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        local_obstacles=(),
    )

    output = prior.propose(obs, reference, 20, action)

    assert output.metadata["type"] == "path_rollout_warm_start"
    assert output.metadata["path_rollout_enabled"]
    assert output.mean.shape == (20, 2)
    assert np.isfinite(output.mean).all()
    # A horizon-aware prior must evolve through the bend instead of repeating
    # one current-target command over all 20 controls.
    assert np.max(np.ptp(output.mean, axis=0)) > 0.1
    assert np.max(output.mean[:, 1]) > 0.1
