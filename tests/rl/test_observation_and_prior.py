import numpy as np
import pytest
from types import SimpleNamespace

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.core.types import (
    ControlCommand,
    LaserScan,
    Pose2D,
    RobotObservation,
    Twist2D,
)
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.rl.observation import (
    ObservationEncoder,
    ObservationEncoderConfig,
    RunningNormalizer,
)
from mobile_robot_mppi.rl.parameterization import (
    PriorParameterization,
    PriorParameterizationConfig,
)
from mobile_robot_mppi.rl.environment import _resolved_prior_mapping
from mobile_robot_mppi.rl.prior import (
    ExternalActionPrior,
    GateConfig,
    TorchSACPrior,
    _distance_gate_alpha,
)


def _observation(scan=True):
    laser = None
    if scan:
        laser = LaserScan(
            ranges=np.linspace(0.5, 4.0, 12),
            angle_min=-np.pi,
            angle_increment=2.0 * np.pi / 11.0,
            range_min=0.05,
            range_max=4.0,
            timestamp=0.0,
        )
    return RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, np.pi / 2.0),
        Twist2D(0.1, -0.2),
        scan=laser,
    )


def test_observation_encoder_uses_robot_relative_goal_and_lidar_sectors():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    config = ObservationEncoderConfig(lidar_sectors=4)
    encoder = ObservationEncoder(config, action_spec)
    encoded = encoder.encode(
        _observation(),
        PointGoal(0.0, 2.0),
        previous_action=np.asarray((0.2, 0.0)),
    )
    assert encoded.shape == (encoder.dimension,)
    assert np.isfinite(encoded).all()
    # Facing +y makes the +y world-frame goal lie directly ahead in body x.
    assert encoded[0] > 0.0
    assert abs(encoded[1]) < 1e-6
    assert encoded[-1] == 1.0


def test_missing_scan_is_explicit_not_nan():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder({"lidar_sectors": 5}, action_spec)
    encoded = encoder.encode(_observation(scan=False), PointGoal(1.0, 0.0))
    np.testing.assert_allclose(encoded[-6:-1], 1.0)
    assert encoded[-1] == 0.0


def test_observation_history_stacks_oldest_to_newest_and_resets():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder(
        {"lidar_sectors": 2, "history_frames": 3}, action_spec
    )
    first = encoder.encode(_observation(), PointGoal(0.0, 2.0))
    frames = first.reshape(3, encoder.frame_dimension)
    np.testing.assert_allclose(frames[0], frames[1])
    np.testing.assert_allclose(frames[1], frames[2])

    moved = RobotObservation(
        0.1,
        Pose2D(0.2, 0.0, np.pi / 2.0),
        Twist2D(0.1, -0.2),
        scan=_observation().scan,
    )
    second = encoder.encode(moved, PointGoal(0.0, 2.0))
    frames = second.reshape(3, encoder.frame_dimension)
    np.testing.assert_allclose(frames[0], frames[1])
    assert not np.allclose(frames[1], frames[2])

    encoder.reset()
    reset = encoder.encode(moved, PointGoal(0.0, 2.0))
    frames = reset.reshape(3, encoder.frame_dimension)
    np.testing.assert_allclose(frames[0], frames[2])


def test_running_normalizer_round_trip_and_ood_score():
    normalizer = RunningNormalizer(3)
    normalizer.update(np.asarray(((0.0, 1.0, 2.0), (1.0, 2.0, 3.0))))
    restored = RunningNormalizer.from_state_dict(normalizer.state_dict())
    np.testing.assert_allclose(restored.mean, normalizer.mean)
    assert restored.ood_score(np.asarray((20.0, 2.0, 3.0))) > 5.0


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("count", -1, "count cannot be negative"),
        ("mean", np.asarray((0.0, np.nan, 0.0)), "must be finite"),
        ("m2", np.asarray((0.0, -1.0, 0.0)), "variance cannot be negative"),
    ],
)
def test_running_normalizer_rejects_corrupt_checkpoint(field, value, match):
    state = RunningNormalizer(3).state_dict()
    state[field] = value
    with pytest.raises(ValueError, match=match):
        RunningNormalizer.from_state_dict(state)


def test_delta_prior_decodes_smooth_bounded_sequence():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    config = PriorParameterizationConfig(num_knots=3, mode="delta", delta_scale=0.5)
    decoder = PriorParameterization(config, action_spec, (0.1, 0.2))
    baseline = np.tile(np.asarray((0.2, 0.0)), (7, 1))
    parameters = np.asarray((
        -1.0, -1.0,
        0.0, 0.0,
        1.0, 1.0,
    ))
    mean, covariance, metadata = decoder.decode(parameters, 7, baseline)
    assert mean.shape == (7, 2)
    assert covariance is None
    assert np.all(mean >= action_spec.lower)
    assert np.all(mean <= action_spec.upper)
    assert metadata["num_knots"] == 3
    assert np.all(np.diff(mean[:, 1]) >= -1e-12)


def test_covariance_only_prior_preserves_mean_and_decodes_two_scales():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        PriorParameterizationConfig(
            kind="covariance_only",
            learn_covariance=True,
            covariance_min_scale=0.5,
            covariance_max_scale=2.0,
        ),
        action_spec,
        (0.1, 0.4),
    )
    baseline = np.tile(np.asarray((0.2, -0.1)), (7, 1))
    mean, covariance, metadata = decoder.decode(
        np.asarray((-1.0, 1.0)), 7, baseline
    )

    assert decoder.parameter_dimension == 2
    np.testing.assert_array_equal(mean, baseline)
    np.testing.assert_allclose(covariance, np.diag((0.05 ** 2, 0.8 ** 2)))
    np.testing.assert_allclose(metadata["covariance_scale"], (0.5, 2.0))
    assert metadata["parameterization"] == "covariance_only"


def test_anchored_covariance_zero_action_is_exact_strong_baseline():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        PriorParameterizationConfig(
            kind="covariance_only",
            learn_covariance=True,
            covariance_min_scale=0.5,
            covariance_max_scale=2.0,
            covariance_anchor_scale=(1.75, 0.75),
            covariance_residual_log_bound=np.log(1.25),
        ),
        action_spec,
        (0.1, 0.4),
    )
    baseline = np.tile(np.asarray((0.2, -0.1)), (7, 1))
    mean, covariance, metadata = decoder.decode(
        np.zeros(2), 7, baseline
    )
    np.testing.assert_array_equal(mean, baseline)
    np.testing.assert_allclose(
        covariance,
        np.diag(((0.1 * 1.75) ** 2, (0.4 * 0.75) ** 2)),
    )
    np.testing.assert_allclose(metadata["covariance_scale"], (1.75, 0.75))

    _, low, low_metadata = decoder.decode(-np.ones(2), 7, baseline)
    _, high, high_metadata = decoder.decode(np.ones(2), 7, baseline)
    np.testing.assert_allclose(low_metadata["covariance_scale"], (1.4, 0.6))
    np.testing.assert_allclose(high_metadata["covariance_scale"], (2.0, 0.9375))
    assert np.all(np.diag(low) > 0.0)
    assert np.all(np.diag(high) > np.diag(low))


def test_anchored_covariance_zero_gate_falls_back_to_anchor():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        {
            "kind": "covariance_only",
            "learn_covariance": True,
            "covariance_min_scale": 0.5,
            "covariance_max_scale": 2.0,
            "covariance_anchor_scale": [1.75, 0.75],
        },
        action_spec,
        (0.1, 0.4),
    )
    _, learned, _ = decoder.decode(
        np.ones(2), 4, np.zeros((4, 2))
    )
    np.testing.assert_allclose(
        decoder.blend_covariance(learned, 0.0),
        decoder.anchor_covariance,
    )
    with pytest.raises(ValueError, match="action dimension"):
        PriorParameterization(
            {
                "kind": "covariance_only",
                "learn_covariance": True,
                "covariance_anchor_scale": [1.0],
            },
            action_spec,
            (0.1, 0.4),
        )


def test_covariance_only_requires_covariance_and_gate_can_exactly_disable_it():
    with pytest.raises(ValueError, match="requires learn_covariance"):
        PriorParameterizationConfig(kind="covariance_only").validate()

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        {
            "kind": "covariance_only",
            "learn_covariance": True,
            "covariance_min_scale": 0.5,
            "covariance_max_scale": 2.0,
        },
        action_spec,
        (0.1, 0.4),
    )
    _, learned, _ = decoder.decode(
        np.asarray((1.0, 1.0)), 4, np.zeros((4, 2))
    )
    assert decoder.blend_covariance(learned, 0.0) is None
    np.testing.assert_allclose(
        decoder.blend_covariance(learned, 0.5),
        0.5 * decoder.base_covariance + 0.5 * learned,
    )


def test_external_covariance_only_prior_zero_gate_is_exact_baseline():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        {
            "kind": "covariance_only",
            "learn_covariance": True,
            "covariance_min_scale": 0.5,
            "covariance_max_scale": 2.0,
        },
        action_spec,
        (0.1, 0.4),
    )
    prior = ExternalActionPrior(decoder, gate_alpha=0.0)
    prior.set_parameters(np.asarray((1.0, 1.0)))
    observation = _observation()
    reference = PointGoal(1.0, 0.0)
    expected = GoalWarmStartPrior().propose(
        observation, reference, 5, action_spec
    )
    actual = prior.propose(observation, reference, 5, action_spec)

    np.testing.assert_array_equal(actual.mean, expected.mean)
    assert actual.covariance is None
    assert actual.metadata["covariance_gate_alpha"] == 0.0


def test_local_subgoal_prior_decodes_direction_into_bounded_sequence():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        PriorParameterizationConfig(
            kind="local_subgoal",
            subgoal_min_distance=0.5,
            subgoal_max_distance=1.5,
            subgoal_max_bearing=np.pi / 2.0,
            subgoal_control_dt=0.1,
        ),
        action_spec,
        (0.1, 0.2),
    )
    baseline = np.tile(np.asarray((0.2, 0.0)), (12, 1))
    straight, covariance, metadata = decoder.decode(
        np.asarray((-1.0, 0.0)), 12, baseline
    )
    assert decoder.parameter_dimension == 2
    assert covariance is None
    assert straight[0, action_spec.index("v_cmd")] > 0.0
    assert abs(straight[0, action_spec.index("omega_cmd")]) < 1e-12
    assert metadata["kind"] == "local_subgoal"
    assert metadata["subgoal_distance"] == 0.5

    left, _, metadata = decoder.decode(np.asarray((-1.0, 1.0)), 12, baseline)
    assert left[0, action_spec.index("v_cmd")] == 0.0
    assert left[0, action_spec.index("omega_cmd")] > 0.0
    assert metadata["subgoal_bearing"] == np.pi / 2.0
    assert np.all(left >= action_spec.lower)
    assert np.all(left <= action_spec.upper)


def test_dynamic_local_subgoal_uses_measured_twist_and_previous_control():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        PriorParameterizationConfig(
            kind="local_subgoal",
            subgoal_decoder="dynamic_first_order",
            subgoal_min_distance=0.5,
            subgoal_max_distance=1.5,
            subgoal_control_dt=0.1,
            subgoal_command_delay=0.04,
        ),
        action_spec,
        (0.1, 0.2),
    )
    baseline = np.zeros((12, 2), dtype=np.float64)
    action = np.asarray((-1.0, 0.35), dtype=np.float64)
    stationary, _, stationary_metadata = decoder.decode(
        action,
        12,
        baseline,
        current_twist=np.asarray((0.0, 0.0)),
        previous_control=np.asarray((0.0, 0.0)),
    )
    moving, _, moving_metadata = decoder.decode(
        action,
        12,
        baseline,
        current_twist=np.asarray((0.35, -0.4)),
        previous_control=np.asarray((0.25, -0.2)),
    )
    assert stationary.shape == moving.shape == (12, 2)
    assert not np.allclose(stationary, moving)
    assert moving_metadata["subgoal_decoder"] == "dynamic_first_order"
    assert moving_metadata["initial_v"] == 0.35
    assert moving_metadata["initial_omega"] == -0.4
    assert moving_metadata["previous_v_cmd"] == 0.25
    assert moving_metadata["command_delay"] == 0.04
    assert stationary_metadata["predicted_terminal_distance"] >= 0.0


def test_dynamic_local_subgoal_requires_explicit_motion_context():
    decoder = PriorParameterization(
        {"kind": "local_subgoal", "subgoal_decoder": "dynamic_first_order"},
        body_velocity_action((0.0, 0.4), 1.0),
    )
    with pytest.raises(ValueError, match="current"):
        decoder.decode(np.zeros(2), 5, np.zeros((5, 2)))


def test_dynamic_decoder_first_order_response_matches_analytic_speed():
    decoder = PriorParameterization(
        {
            "kind": "local_subgoal",
            "subgoal_decoder": "dynamic_first_order",
            "subgoal_velocity_time_constant": 0.2,
            "subgoal_yaw_time_constant": 0.1,
            "subgoal_max_integration_step": 0.01,
        },
        body_velocity_action((0.0, 0.4), 1.0),
    )
    state = decoder._advance_first_order(
        np.zeros(5), np.asarray((0.4, 0.0)), 0.2
    )
    expected_v = 0.4 * (1.0 - np.exp(-1.0))
    expected_x = 0.4 * (0.2 - 0.2 * (1.0 - np.exp(-1.0)))
    assert state[3] == pytest.approx(expected_v, abs=1e-12)
    assert state[0] == pytest.approx(expected_x, abs=1e-12)


def test_dynamic_decoder_timing_inherits_resolved_plant_values():
    values = _resolved_prior_mapping(
        {
            "kind": "local_subgoal",
            "subgoal_decoder": "dynamic_first_order",
            "subgoal_control_dt": None,
            "subgoal_velocity_time_constant": None,
            "subgoal_yaw_time_constant": None,
            "subgoal_command_delay": None,
        },
        {
            "nominal_velocity_time_constant": 0.24,
            "nominal_yaw_time_constant": 0.16,
            "actuator": {"command_delay": 0.06},
        },
        0.12,
    )
    assert values["subgoal_control_dt"] == 0.12
    assert values["subgoal_velocity_time_constant"] == 0.24
    assert values["subgoal_yaw_time_constant"] == 0.16
    assert values["subgoal_command_delay"] == 0.06


@pytest.mark.parametrize(
    "values",
    [
        {"kind": "local_subgoal", "subgoal_max_distance": float("nan")},
        {"kind": "local_subgoal", "subgoal_yaw_gain": float("inf")},
    ],
)
def test_prior_parameterization_rejects_non_finite_config(values):
    with pytest.raises(ValueError, match="must be finite"):
        PriorParameterization(values, body_velocity_action((0.0, 0.4), 1.0))


def test_observation_encoder_rejects_non_finite_scale():
    with pytest.raises(ValueError, match="must be positive"):
        ObservationEncoder(
            {"lidar_max_range": float("nan")},
            body_velocity_action((0.0, 0.4), 1.0),
        )


def test_external_rl_action_is_a_prior_not_a_direct_command():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    prior = ExternalActionPrior(decoder, GoalWarmStartPrior(), gate_alpha=0.5)
    prior.set_parameters(np.zeros(decoder.parameter_dimension))
    output = prior.propose(_observation(), PointGoal(1.0, 0.0), 6, action_spec)
    assert output.mean.shape == (6, 2)
    assert output.metadata["type"] == "rl_external_training"
    assert output.metadata["gate_alpha"] == 0.5


def test_external_dynamic_prior_closes_loop_with_executed_control():
    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    decoder = PriorParameterization(
        {
            "kind": "local_subgoal",
            "subgoal_decoder": "dynamic_first_order",
            "subgoal_command_delay": 0.04,
        },
        action_spec,
        (0.1, 0.2),
    )
    prior = ExternalActionPrior(decoder, GoalWarmStartPrior())
    prior.set_parameters(np.asarray((-1.0, 0.0)))
    first = prior.propose(_observation(), PointGoal(1.0, 0.0), 6, action_spec)
    assert first.metadata["initial_v"] == 0.1
    assert first.metadata["previous_v_cmd"] == 0.0
    prior.observe_safety_decision(SimpleNamespace(
        executed_control=ControlCommand(np.asarray((0.2, 0.3)))
    ))
    second = prior.propose(_observation(), PointGoal(1.0, 0.0), 6, action_spec)
    assert second.metadata["previous_v_cmd"] == 0.2
    assert second.metadata["previous_omega_cmd"] == 0.3


def test_ood_gate_falls_back_to_conventional_prior():
    class DummyAgent:
        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            return np.ones(4, dtype=np.float32), {"alpha": 0.2}

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder(
        {
            "lidar_sectors": 1,
            "include_previous_action": False,
            "include_safety_state": False,
        },
        action_spec,
    )
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    fallback = GoalWarmStartPrior()
    prior = TorchSACPrior(
        DummyAgent(),
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "ood",
            "ood_soft_threshold": 0.01,
            "ood_hard_threshold": 0.02,
        },
        fallback_prior=fallback,
    )
    observation = _observation()
    reference = PointGoal(1.0, 0.0)
    result = prior.propose(observation, reference, 5, action_spec)
    expected = fallback.propose(observation, reference, 5, action_spec)
    np.testing.assert_allclose(result.mean, expected.mean)
    assert result.metadata["gate_alpha"] == 0.0
    assert result.metadata["ood_score"] > 0.02


def test_exploration_gate_activates_and_latches_on_high_critic_disagreement():
    class DummyAgent:
        disagreement = 0.02

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            return np.ones(4, dtype=np.float32), {"alpha": 0.1}

        def critic_disagreement(self, observation, action):
            del observation, action
            return self.disagreement

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder(
        {
            "lidar_sectors": 1,
            "include_previous_action": False,
            "include_safety_state": False,
        },
        action_spec,
    )
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    agent = DummyAgent()
    prior = TorchSACPrior(
        agent,
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "exploration",
            "exploration_signal": "critic_disagreement",
            "exploration_soft_threshold": 0.05,
            "exploration_hard_threshold": 0.20,
            "exploration_latch": True,
        },
    )
    observation = _observation()
    reference = PointGoal(1.0, 0.0)
    low = prior.propose(observation, reference, 5, action_spec)
    assert low.metadata["gate_alpha"] == 0.0
    assert low.metadata["exploration_activation"] == 0.0

    agent.disagreement = 0.30
    high = prior.propose(observation, reference, 5, action_spec)
    assert high.metadata["gate_alpha"] == 1.0
    assert high.metadata["exploration_latch_alpha"] == 1.0

    agent.disagreement = 0.0
    latched = prior.propose(observation, reference, 5, action_spec)
    assert latched.metadata["exploration_activation"] == 0.0
    assert latched.metadata["gate_alpha"] == 1.0
    prior.reset()
    reset = prior.propose(observation, reference, 5, action_spec)
    assert reset.metadata["gate_alpha"] == 0.0

    late_observation = RobotObservation(
        1.0,
        observation.pose,
        observation.twist,
        scan=observation.scan,
    )
    agent.disagreement = 0.30
    late = prior.propose(late_observation, reference, 5, action_spec)
    assert late.metadata["exploration_trigger_open"] is False
    assert late.metadata["gate_alpha"] == 0.0


def test_near_goal_gate_transitions_smoothly_to_traditional_mppi():
    config = GateConfig(
        near_goal_fallback_enabled=True,
        near_goal_full_fallback_distance=0.5,
        near_goal_full_rl_distance=1.5,
    )
    assert _distance_gate_alpha(0.5, config) == 0.0
    assert _distance_gate_alpha(1.5, config) == 1.0
    middle = _distance_gate_alpha(1.0, config)
    assert abs(middle - 0.5) < 1e-12
    assert 0.0 < _distance_gate_alpha(0.75, config) < middle
    assert middle < _distance_gate_alpha(1.25, config) < 1.0


def test_torch_prior_applies_correction_advantage_before_decoding():
    class DummyCorrectionAgent:
        is_correction_policy = True

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            return np.ones(4, dtype=np.float32), {
                "policy_mode": "frozen_bc_correction",
                "applied_correction_abs_mean": 0.2,
            }

        def filter_correction_by_advantage(
            self,
            observation,
            candidate_action,
            gate_mode,
            critic_source,
            threshold,
            uncertainty_multiplier,
        ):
            del observation, candidate_action
            assert gate_mode == "hard"
            assert critic_source == "target"
            assert threshold == 0.0
            assert uncertainty_multiplier == 3.0
            return np.zeros(4, dtype=np.float32), {
                "correction_advantage_gate_mode": gate_mode,
                "correction_advantage_critic_source": critic_source,
                "correction_advantage_threshold": threshold,
                "correction_advantage_uncertainty_multiplier": (
                    uncertainty_multiplier
                ),
                "correction_advantage_gate_alpha": 0.0,
                "target_conservative_advantage": -0.25,
                "applied_correction_abs_mean": 0.0,
            }

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder(
        {
            "lidar_sectors": 1,
            "include_previous_action": False,
            "include_safety_state": False,
        },
        action_spec,
    )
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    prior = TorchSACPrior(
        DummyCorrectionAgent(),
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "none",
            "correction_advantage_gate_mode": "hard",
            "correction_advantage_critic_source": "target",
            "correction_advantage_threshold": 0.0,
            "correction_advantage_uncertainty_multiplier": 3.0,
        },
    )
    result = prior.propose(
        _observation(), PointGoal(1.0, 0.0), 5, action_spec
    )
    assert result.metadata["correction_advantage_gate_alpha"] == 0.0
    assert result.metadata["target_conservative_advantage"] == -0.25
    assert result.metadata["applied_correction_abs_mean"] == 0.0


@pytest.mark.parametrize(
    "mapping,match",
    [
        ({"correction_advantage_gate_mode": "soft"}, "mode"),
        ({"correction_advantage_critic_source": "ensemble"}, "source"),
        ({"correction_advantage_threshold": float("nan")}, "threshold"),
        (
            {"correction_advantage_uncertainty_multiplier": 0.5},
            "multiplier",
        ),
    ],
)
def test_correction_advantage_gate_config_fails_closed(mapping, match):
    config = GateConfig.from_mapping(mapping)
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_hypothetical_target_encoding_does_not_mutate_online_history():
    action_spec = body_velocity_action((0.0, 0.5), 1.0)
    encoder = ObservationEncoder(
        ObservationEncoderConfig(lidar_sectors=4, history_frames=2),
        action_spec,
    )
    reference = PointGoal(2.0, 0.0)
    current = _observation()
    encoder.encode(current, reference, previous_action=np.zeros(2))
    history_before = [item.copy() for item in encoder._history]
    hypothetical = RobotObservation(
        current.timestamp + 0.5,
        Pose2D(0.4, 0.1, 0.2),
        Twist2D(0.2, 0.1),
        scan=current.scan,
    )
    target = reference.target_at(
        hypothetical.timestamp, hypothetical.pose.as_array()
    )
    encoded = encoder.encode_to_target(
        hypothetical,
        target,
        previous_action=np.asarray((0.2, 0.1)),
        update_history=False,
    )

    assert encoded.shape == (encoder.dimension,)
    assert len(encoder._history) == len(history_before)
    for actual, expected in zip(encoder._history, history_before):
        np.testing.assert_array_equal(actual, expected)
