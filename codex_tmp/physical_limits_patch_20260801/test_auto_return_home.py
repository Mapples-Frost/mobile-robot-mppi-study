import pytest

import real_robot_operator


def test_outbound_goal_stops_when_auto_return_is_disabled():
    phase, reason = real_robot_operator._mission_transition(
        "outbound", 0.29, 0.0, False
    )
    assert phase == "complete"
    assert reason == "goal_reached"


def test_outbound_goal_switches_to_recorded_home():
    phase, reason = real_robot_operator._mission_transition(
        "outbound", 0.29, 0.0, True
    )
    assert phase == "return_home"
    assert reason == "outbound_goal_reached"


def test_return_requires_both_home_position_and_heading():
    assert real_robot_operator._mission_transition(
        "return_home", 0.19, 0.11, True
    ) == ("return_home", None)
    assert real_robot_operator._mission_transition(
        "return_home", 0.21, 0.05, True
    ) == ("return_home", None)
    assert real_robot_operator._mission_transition(
        "return_home", 0.19, 0.09, True
    ) == ("complete", "home_pose_reached")


@pytest.mark.parametrize(
    ("error", "expected_sign"),
    ((0.50, 1.0), (-0.50, -1.0)),
)
def test_home_heading_alignment_is_slow_and_directional(
    error, expected_sign
):
    command = real_robot_operator._home_heading_alignment_command(
        error, 0.50
    )
    assert abs(command) <= 0.20
    assert command * expected_sign > 0.0


def test_home_heading_alignment_stops_inside_tolerance():
    assert real_robot_operator._home_heading_alignment_command(
        0.09, 0.50
    ) == 0.0


def test_consecutive_intervention_refreshes_lock_without_flipping_sign():
    sign, last_cycle = real_robot_operator._refresh_reactive_turn_memory(
        -1.0,
        19,
        22,
        {"turn_sign": 1.0},
    )
    assert sign == -1.0
    assert last_cycle == 22


def test_return_boundary_guard_rotates_inward_before_crossing_margin():
    linear, angular, diagnostics = (
        real_robot_operator._apply_return_boundary_guard(
            -0.26,
            2.13,
            -2.98,
            0.35,
            -0.35,
            0.0,
            0.0,
            -0.75,
            5.75,
            -1.50,
            5.75,
            0.50,
        )
    )
    assert linear == 0.0
    assert angular > 0.0
    assert diagnostics["nearest_clearance_m"] == pytest.approx(0.24)


def test_return_boundary_guard_leaves_interior_command_unchanged():
    linear, angular, diagnostics = (
        real_robot_operator._apply_return_boundary_guard(
            2.0,
            2.0,
            0.0,
            0.30,
            -0.10,
            0.0,
            0.0,
            -0.75,
            5.75,
            -1.50,
            5.75,
            0.50,
        )
    )
    assert (linear, angular, diagnostics) == (0.30, -0.10, None)
