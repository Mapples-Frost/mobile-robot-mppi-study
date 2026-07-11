import numpy as np
import pytest

from mobile_robot_mppi.core.references import PointGoal, PoseGoal, TimeTrajectoryReference
from mobile_robot_mppi.core.spaces import ActionSpec, dynamic_unicycle_state


def test_dynamic_goal_is_not_a_global_constant():
    state = np.asarray((0.0, 0.0, 0.2, 0.0, 0.0))
    first = PointGoal(3.0, 3.0).target_at(0.0, state)
    second = PointGoal(-1.0, 2.0).target_at(0.0, state)
    assert (first.pose.x, first.pose.y) == (3.0, 3.0)
    assert (second.pose.x, second.pose.y) == (-1.0, 2.0)


def test_periodic_state_error_wraps_heading():
    spec = dynamic_unicycle_state()
    predicted = np.asarray((0.0, 0.0, np.pi - 0.01, 0.0, 0.0))
    target = np.asarray((0.0, 0.0, -np.pi + 0.01, 0.0, 0.0))
    assert spec.error(predicted, target)[2] == pytest.approx(-0.02)


def test_action_space_can_have_more_than_two_dimensions():
    spec = ActionSpec(
        ("v_cmd", "omega_cmd", "aux"),
        np.asarray((0.0, -1.0, -2.0)),
        np.asarray((0.5, 1.0, 2.0)),
    )
    assert spec.dimension == 3
    assert np.allclose(spec.clip(np.asarray((1.0, -2.0, 4.0))), (0.5, -1.0, 2.0))


def test_time_reference_interpolates_angle_on_circle():
    reference = TimeTrajectoryReference((0.0, 1.0), ((0.0, 0.0, 3.0), (1.0, 1.0, -3.0)))
    target = reference.target_at(0.5, np.zeros(3))
    assert target.pose.x == pytest.approx(0.5)
    assert abs(target.pose.theta) > 3.0
