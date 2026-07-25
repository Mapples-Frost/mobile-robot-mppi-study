import numpy as np

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    dynamic_unicycle_state,
)
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.learning.models import (
    CausalStallGatedResidualDynamics,
)
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    ResidualPrediction,
)
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


class _ZeroResidual:
    state_dim = 5
    control_dim = 2
    model = None

    def derivative(self, state, control, time=None):
        del control, time
        return np.zeros_like(np.asarray(state, dtype=np.float64))


def _observation(timestamp):
    return RobotObservation(
        timestamp,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
    )


def test_stall_context_uses_only_completed_plan_and_preview_is_side_effect_free():
    residual = CausalStallGatedResidualDynamics(
        _ZeroResidual(),
        position_indices=(0, 1),
        speed_index=3,
        consecutive_steps=1,
    )
    controller = MppiController(
        ResidualPrediction(DynamicUnicyclePrediction(), residual),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.4), 1.0),
        MppiConfig(
            horizon=3,
            num_samples=4,
            dt=0.1,
            integrator="euler",
            noise_sigma=(0.05, 0.1),
            seed=5,
        ),
    )
    goal = PointGoal(1.0, 0.0)

    first = controller.plan(_observation(0.0), goal)
    assert first.diagnostics["residual_stall_guard_latched"] is False
    assert residual.context_steps == 1
    assert controller._previous_probabilistic_risk == 0.0

    preview = controller.preview_plan(_observation(0.1), goal)
    assert preview.diagnostics["residual_stall_guard_latched"] is False
    assert residual.context_steps == 1

    second = controller.plan(_observation(0.1), goal)
    assert second.diagnostics["residual_stall_guard_latched"] is True
    assert residual.context_steps == 2
    assert residual.authority == 0.0

    controller.reset(seed=17)
    assert residual.context_steps == 0
    assert residual.authority == 1.0
    assert controller._previous_probabilistic_risk == 1.0
