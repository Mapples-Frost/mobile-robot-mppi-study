import numpy as np
import pytest

from mobile_robot_mppi.core.references import (
    PointGoal,
    PoseGoal,
    PolylineReference,
    TimeTrajectoryReference,
    WaypointReference,
)
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


def test_waypoint_reference_uses_looser_intermediate_tolerance_only():
    reference = WaypointReference(
        ((0.4, 0.0), (0.8, 0.0)), tolerance=0.1, waypoint_tolerance=0.3
    )
    intermediate = reference.target_at(0.0, np.asarray((0.0, 0.0, 0.0)))
    assert intermediate.reference_id == "waypoint_0"
    assert intermediate.position_tolerance == pytest.approx(0.3)

    final = reference.target_at(0.1, np.asarray((0.15, 0.0, 0.0)))
    assert final.reference_id == "waypoint_1"
    assert final.position_tolerance == pytest.approx(0.1)


def test_waypoint_reference_exposes_terminal_approach_phase():
    reference = WaypointReference(
        ((0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0)),
        tolerance=0.05,
        waypoint_tolerance=0.11,
        terminal_approach_count=2,
    )
    assert reference.target_at(0.0, np.asarray((0.0, 0.0, 0.0))).phase == "tracking"
    assert reference.target_at(0.1, np.asarray((0.2, 0.0, 0.0))).phase == "tracking"
    approach = reference.target_at(0.2, np.asarray((0.4, 0.0, 0.0)))
    assert approach.phase == "terminal_approach"
    terminal = reference.target_at(0.3, np.asarray((0.6, 0.0, 0.0)))
    assert terminal.phase == "terminal"
    assert terminal.is_terminal


def test_polyline_reference_progress_is_monotonic_and_uses_final_tolerance_only():
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        tolerance=0.1,
        lookahead_distance=0.3,
        terminal_approach_distance=0.5,
    )
    first = reference.target_at(0.0, np.asarray((0.2, 0.1, 0.0)))
    first_progress = reference.progress
    assert first.position_tolerance == 0.0
    assert first.phase == "tracking"

    reference.target_at(0.1, np.asarray((0.8, 0.0, 0.0)))
    advanced_progress = reference.progress
    reference.target_at(0.2, np.asarray((0.1, 0.0, 0.0)))
    assert reference.progress >= advanced_progress > first_progress

    final = reference.target_at(0.3, np.asarray((1.0, 0.8, 0.0)))
    assert final.is_terminal
    assert final.phase == "terminal"
    assert final.position_tolerance == pytest.approx(0.1)


def test_polyline_projection_is_signed_and_preview_is_side_effect_free():
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        lookahead_distance=0.3,
    )
    reference.target_at(0.0, np.asarray((0.4, 0.0, 0.0)))
    live_progress = reference.progress

    left = reference.project(
        np.asarray((0.6, 0.2)), minimum_progress=live_progress
    )
    assert left.signed_cross_track_error > 0.0
    assert left.cross_track_error == pytest.approx(0.2)
    assert left.remaining == pytest.approx(reference.total_length - 0.6)
    assert left.curvature > 0.0

    target = reference.preview_target_at(
        1.0, np.asarray((0.9, 0.1, 0.0)), progress_floor=live_progress
    )
    assert target.reference_id.startswith("polyline_")
    assert reference.progress == pytest.approx(live_progress)


def test_polyline_projection_rejects_invalid_inputs():
    reference = PolylineReference(((0.0, 0.0), (1.0, 0.0)))
    with pytest.raises(ValueError, match="finite x/y"):
        reference.project(np.asarray((np.nan, 0.0)))
    with pytest.raises(ValueError, match="outside"):
        reference.project(np.asarray((0.0, 0.0)), minimum_progress=2.0)


def test_polyline_preview_poses_are_vectorized_clipped_and_side_effect_free():
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        lookahead_distance=0.3,
    )
    reference.target_at(0.0, np.asarray((0.4, 0.0, 0.0)))
    progress_before = reference.progress

    poses = reference.preview_poses(
        np.asarray((0.0, 0.4, 2.0)), progress_floor=progress_before
    )

    assert poses.shape == (3, 3)
    np.testing.assert_allclose(poses[0, :2], (0.4, 0.0))
    np.testing.assert_allclose(poses[1, :2], (0.8, 0.0))
    np.testing.assert_allclose(poses[2, :2], (1.0, 1.0))
    assert poses[2, 2] == pytest.approx(np.pi / 2.0)
    assert reference.progress == pytest.approx(progress_before)
