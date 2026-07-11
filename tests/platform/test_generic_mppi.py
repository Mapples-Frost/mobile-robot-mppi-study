import numpy as np

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import ActionSpec, body_velocity_action, dynamic_unicycle_state, unicycle_state
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, LegacyUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.policies.priors import RLPolicyPrior


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
