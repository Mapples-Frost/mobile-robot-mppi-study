import numpy as np

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.core.types import ControlCommand, PlanResult
from mobile_robot_mppi.real_robot.encounter_control import (
    EncounterControlAuthority,
    EncounterControlConfig,
    EncounterReferenceAuthority,
)


def _plan(control=(0.1, -0.3), **diagnostics):
    values = {
        "probabilistic_obstacle_hard_violation": False,
        "probabilistic_obstacle_maximum_step_probability": 0.04,
        "probabilistic_obstacle_probability_mass": 0.6,
    }
    values.update(diagnostics)
    return PlanResult(
        proposed_control=ControlCommand(control, timestamp=2.0),
        control_sequence=np.tile(np.asarray(control, dtype=np.float64), (12, 1)),
        predicted_trajectory=np.zeros((13, 3), dtype=np.float64),
        diagnostics=values,
    )


def _intent(**values):
    intent = {
        "encounter_phase": "straight_crossing",
        "encounter_track_index": 2,
        "encounter_strategy": "behind_pass",
        "encounter_locked_steering_side": 1,
        "encounter_entry_goal_origin": (0.0, 0.0),
        "encounter_temporary_waypoint": (2.5, 0.8),
        "encounter_distance_m": 1.8,
    }
    intent.update(values)
    return intent


def _base_reference():
    return PolylineReference(
        ((0.0, 0.0), (5.0, 0.0)),
        lookahead_distance=1.4,
        terminal_approach_distance=1.15,
        corridor_half_width=0.9,
        footprint_radius=0.25,
    )


def test_disabled_authorities_are_bit_exact_noops():
    config = EncounterControlConfig(enabled=False)
    plan = _plan()
    base = _base_reference()

    assert EncounterReferenceAuthority(config).select(
        base, _intent(), (0.0, 0.0, 0.0), (5.0, 0.0)
    ) is base
    assert EncounterControlAuthority(config).apply(
        plan, _intent(), (0.0, 0.0, 0.0), {"emergency_stop": False}
    ) is plan


def test_crossing_reference_bends_to_locked_side_and_keeps_real_goal():
    authority = EncounterReferenceAuthority(
        EncounterControlConfig(enabled=True)
    )

    selected = authority.select(
        _base_reference(), _intent(), (0.2, 0.0, 0.0), (5.0, 0.0)
    )

    assert selected is not None
    assert np.allclose(selected.points[0], (0.0, 0.0))
    assert selected.points[1, 1] > 0.0
    assert np.allclose(selected.points[-1], (5.0, 0.0))


def test_active_mode_corrects_wrong_turn_and_supplies_safe_speed_floor():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )

    result = authority.apply(
        _plan(control=(0.05, -0.4)),
        _intent(),
        pose=(0.0, 0.0, 0.0),
        guard_result={"emergency_stop": False, "min_rear_range": 2.0},
    )

    assert result.proposed_control.v >= 0.4
    assert result.proposed_control.omega > 0.0
    assert result.diagnostics["encounter_control_applied"] is True
    assert result.diagnostics["encounter_control_authoritative"] is True
    assert np.all(result.control_sequence[:3, 1] > 0.0)


def test_front_pass_uses_high_progress_speed_but_remains_bounded():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )

    result = authority.apply(
        _plan(control=(0.2, 0.1)),
        _intent(encounter_strategy="front_pass"),
        pose=(0.0, 0.0, 0.0),
        guard_result={"emergency_stop": False},
    )

    assert result.proposed_control.v == 0.48
    assert result.proposed_control.v <= 0.5


def test_frontal_close_approach_reverses_only_with_measured_rear_clearance():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    intent = _intent(
        encounter_phase="frontal_approach",
        encounter_strategy="left_bypass",
        encounter_distance_m=0.75,
    )

    clear = authority.apply(
        _plan(control=(0.1, -0.2)), intent, (0.0, 0.0, 0.0),
        {"emergency_stop": False, "min_rear_range": 1.2},
    )
    blocked = authority.apply(
        _plan(control=(0.1, -0.2)), intent, (0.0, 0.0, 0.0),
        {"emergency_stop": False, "min_rear_range": 0.5},
    )

    assert clear.proposed_control.v == -0.18
    assert clear.diagnostics["encounter_control_reverse_applied"] is True
    assert blocked.proposed_control.v > 0.0
    assert blocked.diagnostics["encounter_control_reverse_available"] is False


def test_hard_stop_is_not_overridden_by_mode_speed_or_turn_authority():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    plan = _plan(control=(0.1, -0.2))

    result = authority.apply(
        plan, _intent(), (0.0, 0.0, 0.0),
        {"emergency_stop": True, "min_rear_range": 2.0},
    )

    assert np.array_equal(result.proposed_control.values, np.zeros(2))
    assert result.diagnostics["encounter_control_hard_safety_pending"] is True
    assert result.diagnostics["encounter_control_motion_owner"] == "hard_stop"


def test_open_corridor_temporal_alarm_does_not_cancel_encounter_passage():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    result = authority.apply(
        _plan(control=(0.1, -0.2)),
        _intent(),
        (0.0, 0.0, 0.0),
        {
            "emergency_stop": True,
            "reason": "temporal_collision_risk",
            "min_front_range": 1.0,
            "valid_near_body_count": 0,
        },
    )

    assert result.proposed_control.v == 0.4
    assert result.proposed_control.omega > 0.0
    assert result.diagnostics[
        "encounter_control_temporal_risk_deferred"
    ] is True
    assert result.diagnostics["encounter_control_motion_owner"] == "encounter"


def test_side_near_body_alarm_uses_clear_locked_side_instead_of_deadlock():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    result = authority.apply(
        _plan(control=(-0.3, 0.6)),
        _intent(
            encounter_phase="frontal_approach",
            encounter_strategy="left_bypass",
            encounter_locked_steering_side=1,
            encounter_temporary_waypoint=(2.0, 0.9),
            encounter_distance_m=0.65,
        ),
        (0.0, 0.0, 0.0),
        {
            "emergency_stop": True,
            "reason": "near_body_hard_stop",
            "min_front_range": 3.2,
            "min_left_side_range": 2.0,
            "min_right_side_range": 0.30,
            "valid_near_body_count": 1,
            "min_rear_range": 2.0,
        },
    )

    assert result.proposed_control.v == 0.32
    assert result.proposed_control.omega > 0.0
    assert result.diagnostics[
        "encounter_control_side_hard_stop_deferred"
    ] is True
    assert result.diagnostics["encounter_control_reverse_applied"] is False


def test_front_hard_stop_allows_only_locked_side_turn_when_side_is_clear():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    result = authority.apply(
        _plan(control=(-0.3, -0.6)),
        _intent(
            encounter_locked_steering_side=-1,
            encounter_temporary_waypoint=(2.0, -0.9),
            encounter_distance_m=0.65,
        ),
        (0.0, 0.0, 0.0),
        {
            "emergency_stop": True,
            "reason": "hard_stop",
            "min_front_range": 0.40,
            "min_left_side_range": 0.35,
            "min_right_side_range": 2.40,
            "valid_near_body_count": 0,
        },
    )

    assert result.proposed_control.v == 0.0
    assert result.proposed_control.omega < 0.0
    assert result.diagnostics[
        "encounter_control_side_hard_stop_turn_only"
    ] is True


def test_front_hard_stop_creeps_into_wide_locked_side_bypass():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    result = authority.apply(
        _plan(control=(-0.3, -0.6)),
        _intent(
            encounter_locked_steering_side=-1,
            encounter_temporary_waypoint=(2.0, -0.9),
            encounter_distance_m=0.65,
        ),
        (0.0, 0.0, 0.0),
        {
            "emergency_stop": True,
            "reason": "hard_stop",
            "min_front_range": 0.46,
            "min_left_side_range": 0.35,
            "min_right_side_range": 2.0,
            "valid_near_body_count": 0,
        },
    )

    assert result.proposed_control.v == 0.12
    assert result.proposed_control.omega < 0.0
    assert result.diagnostics[
        "encounter_control_side_hard_stop_turn_creep"
    ] is True


def test_mode_rejects_unrequested_planner_reverse_even_when_risk_is_high():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )

    result = authority.apply(
        _plan(
            control=(-0.3, -0.6),
            probabilistic_obstacle_hard_violation=True,
            probabilistic_obstacle_maximum_step_probability=0.8,
            probabilistic_obstacle_probability_mass=8.0,
        ),
        _intent(
            encounter_phase="frontal_approach",
            encounter_strategy="right_bypass",
            encounter_locked_steering_side=-1,
            encounter_temporary_waypoint=(2.0, -0.9),
            encounter_distance_m=1.0,
        ),
        pose=(0.0, 0.0, 0.0),
        guard_result={"emergency_stop": False, "min_rear_range": 0.5},
    )

    assert result.proposed_control.v == 0.32
    assert result.proposed_control.omega < 0.0
    assert result.diagnostics[
        "encounter_control_planner_reverse_rejected"
    ] is True
    assert result.diagnostics["encounter_control_reverse_applied"] is False
    assert result.diagnostics["encounter_control_motion_owner"] == "encounter"


def test_rejoin_reference_returns_to_frozen_goal_line():
    reference = EncounterReferenceAuthority(
        EncounterControlConfig(enabled=True)
    ).select(
        _base_reference(),
        _intent(
            encounter_phase="rejoin",
            encounter_temporary_waypoint=(2.3, 0.0),
        ),
        pose=(1.2, 0.7, 0.5),
        goal=(5.0, 0.0),
    )

    assert np.allclose(reference.points[:, 1], 0.0)


def test_rejoin_yaw_uses_goal_line_heading_not_tangent_lookahead():
    authority = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    )
    result = authority.apply(
        _plan(control=(0.36, 0.0)),
        _intent(
            encounter_phase="rejoin",
            encounter_strategy="right_bypass",
            encounter_locked_steering_side=-1,
            encounter_temporary_waypoint=(3.4, 0.12),
            encounter_rejoin_heading_error_rad=0.50,
        ),
        pose=(2.6, 0.55, -0.50),
        guard_result={"emergency_stop": False},
    )

    assert result.diagnostics["encounter_control_heading_error_rad"] == (
        0.50
    )
    assert result.proposed_control.omega > 0.0
    assert result.proposed_control.v == 0.15


def test_planning_context_explicitly_revokes_competing_authorities():
    context = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    ).planning_context(_intent())

    assert context["encounter_control_authoritative"] is True
    assert context["encounter_control_inhibit_rear_pass"] is True
    assert context["encounter_control_inhibit_forward_passage"] is True
    assert context["encounter_control_inhibit_dynamic_escape"] is True
    assert context["encounter_control_stop_only_hard_safety"] is True
    assert context["encounter_control_hard_safety_retained"] is True
    assert context[
        "encounter_control_hard_stop_escape_motion_allowed"
    ] is False
