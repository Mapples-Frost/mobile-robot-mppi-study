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

    assert np.array_equal(
        result.proposed_control.values, plan.proposed_control.values
    )
    assert result.diagnostics["encounter_control_hard_safety_pending"] is True


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


def test_planning_context_explicitly_revokes_competing_authorities():
    context = EncounterControlAuthority(
        EncounterControlConfig(enabled=True)
    ).planning_context(_intent())

    assert context["encounter_control_authoritative"] is True
    assert context["encounter_control_inhibit_rear_pass"] is True
    assert context["encounter_control_inhibit_forward_passage"] is True
    assert context["encounter_control_inhibit_dynamic_escape"] is True
    assert context["encounter_control_hard_safety_retained"] is True
