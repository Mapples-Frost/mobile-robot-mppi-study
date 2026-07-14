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
