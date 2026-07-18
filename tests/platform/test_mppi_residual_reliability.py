import numpy as np

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import body_velocity_action, dynamic_unicycle_state
from mobile_robot_mppi.core.types import (
    ControlCommand, Pose2D, RobotObservation, SafetyDecision, Twist2D,
)
from mobile_robot_mppi.learning.models import InnovationGatedResidualDynamics
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, ResidualPrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


class _VelocityResidual:
    state_dim = 5
    control_dim = 2
    model = None

    def derivative(self, state, control, time=None):
        del control, time
        result = np.zeros_like(np.asarray(state, dtype=np.float64))
        result[..., 3] = 1.0
        return result


def _observation(time, v):
    return RobotObservation(
        time, Pose2D(0.0, 0.0, 0.0), Twist2D(float(v), 0.0)
    )


def _controller():
    nominal = DynamicUnicyclePrediction()
    gate = InnovationGatedResidualDynamics(
        _VelocityResidual(), state_indices=(3,), state_scales=(1.0,),
        minimum_samples=1, confidence_z=0.0, off_threshold=0.0,
        on_threshold=0.1, rise_rate=1.0, fall_rate=1.0,
    )
    return MppiController(
        ResidualPrediction(nominal, gate), dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.4), 1.0),
        MppiConfig(
            horizon=3, num_samples=4, dt=0.1, integrator="euler",
            noise_sigma=(0.05, 0.1), seed=5,
        ),
    )


def _zero_decision(proposed):
    return SafetyDecision(
        proposed_control=proposed,
        executed_control=ControlCommand(np.zeros(2)),
        overridden=True,
        reason="test",
    )


def test_reliability_uses_only_completed_transition_and_preview_is_side_effect_free():
    controller = _controller()
    first = controller.plan(_observation(0.0, 0.0), PointGoal(1.0, 0.0))
    assert first.diagnostics["residual_reliability_alpha"] == 0.0
    controller.observe_safety_decision(_zero_decision(first.proposed_control))

    next_observation = _observation(0.1, 0.1)
    preview = controller.preview_plan(next_observation, PointGoal(1.0, 0.0))
    assert preview.diagnostics["residual_reliability_alpha"] == 0.0
    assert controller.dynamics.residual.samples == 0

    second = controller.plan(next_observation, PointGoal(1.0, 0.0))
    assert second.diagnostics["residual_reliability_alpha"] == 1.0
    assert second.diagnostics["residual_reliability_samples"] == 1
    assert second.diagnostics["residual_reliability_residual_error"] == 0.0
    assert second.diagnostics["residual_reliability_nominal_error"] > 0.0


def test_controller_reset_clears_reliability_history():
    controller = _controller()
    controller.dynamics.residual.observe_prediction_errors(
        np.asarray((0.0, 0.0, 0.0, 1.0, 0.0)), np.zeros(5)
    )
    assert controller.dynamics.residual.alpha == 1.0
    controller.reset(seed=17)
    assert controller.dynamics.residual.alpha == 0.0
    assert controller.dynamics.residual.samples == 0
    assert controller._reliability_previous_state is None
    assert controller._reliability_pending_control is None
