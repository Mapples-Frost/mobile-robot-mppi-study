import pytest

from reverse_safety_contract import (
    apply_front_arc_creep,
    apply_front_clearance_arc_governor,
    apply_near_body_speed_governor,
    apply_reactive_arc_hold,
    apply_turn_direction_lock,
    apply_provisional_collision_intervention,
    apply_directional_clearance,
    angular_slew_step_budget,
    bound_signed_speed,
    couple_differential_drive_command,
    limit_angular_command_step,
    motion_session_expired,
    paired_rear_clearance,
    paired_rear_is_fresh,
    protected_reverse_escape,
    stabilize_goal_heading,
    verified_near_body_turn_escape,
)


def test_reverse_is_preserved_and_hard_capped():
    assert bound_signed_speed(-0.01, 0.12, 0.10, True) == -0.01
    assert bound_signed_speed(-0.20, 0.12, 0.10, True) == -0.10
    assert bound_signed_speed(0.20, 0.12, 0.10, True) == 0.12


def test_reverse_can_still_be_disabled_explicitly():
    assert bound_signed_speed(-0.02, 0.12, 0.03, False) == 0.0


def test_rear_clearance_stops_and_slows_reverse_only():
    stopped, stopped_reason = apply_directional_clearance(
        -0.03, 2.0, 0.40, 0.45, 0.80, 0.45, 0.80
    )
    slowed, slowed_reason = apply_directional_clearance(
        -0.03, 2.0, 0.625, 0.45, 0.80, 0.45, 0.80
    )
    clear, clear_reason = apply_directional_clearance(
        -0.03, 2.0, 1.0, 0.45, 0.80, 0.45, 0.80
    )

    assert stopped == 0.0
    assert stopped_reason == "rear_hard_stop"
    assert slowed == pytest.approx(-0.015)
    assert slowed_reason == "rear_slow"
    assert clear == -0.03
    assert clear_reason == "command_accepted"


def test_front_and_rear_clearance_are_direction_specific():
    reverse, reverse_reason = apply_directional_clearance(
        -0.03, 0.20, 1.0, 0.45, 0.80, 0.45, 0.80
    )
    forward, forward_reason = apply_directional_clearance(
        0.12, 1.0, 0.20, 0.45, 0.80, 0.45, 0.80
    )

    assert reverse == -0.03
    assert reverse_reason == "command_accepted"
    assert forward == 0.12
    assert forward_reason == "command_accepted"


def test_rear_pair_uses_more_conservative_sensor():
    assert paired_rear_clearance(0.72, 0.55) == pytest.approx(0.55)
    assert paired_rear_clearance(None, 0.55) is None


def test_rear_pair_requires_both_fresh_valid_sensors():
    assert paired_rear_is_fresh(0.8, 0.8, 0.02, 0.03, 0.4)
    assert not paired_rear_is_fresh(0.8, None, 0.02, 0.03, 0.4)
    assert not paired_rear_is_fresh(0.8, 0.8, 0.02, 0.41, 0.4)


def test_wheel_envelope_preserves_curvature_by_uniform_scaling():
    linear, angular, applied = couple_differential_drive_command(
        0.58, 0.50, 0.33, 0.60
    )
    assert applied
    assert linear / angular == pytest.approx(0.58 / 0.50)
    assert max(
        abs(linear + angular * 0.165),
        abs(linear - angular * 0.165),
    ) == pytest.approx(0.60)


def test_near_body_escape_requires_dynamic_threat_and_clear_fresh_rear():
    reverse, angular, applied = protected_reverse_escape(
        -0.06, -0.25, 0.10, 0.80, 0.80, True, True
    )
    assert applied
    assert reverse == pytest.approx(-0.06)
    assert angular == pytest.approx(-0.25)

    assert protected_reverse_escape(
        -0.06, -0.25, 0.10, 0.80, 0.80, False, True
    ) == (0.0, 0.0, False)
    assert protected_reverse_escape(
        -0.06, -0.25, 0.10, 0.70, 0.80, True, True
    ) == (0.0, 0.0, False)


def test_provisional_collision_candidate_slows_and_steers_away():
    candidate = {
        "closest_approach_time_s": 1.2,
        "closest_approach_distance_m": 0.2,
        "distance_m": 1.1,
        "lateral_m": 0.3,
    }
    linear, angular, selected = apply_provisional_collision_intervention(
        0.50, 0.02, (candidate,), 0.35
    )
    assert linear == pytest.approx(0.14)
    assert angular == pytest.approx(-0.28)
    assert selected["forward_cap_mps"] == pytest.approx(0.14)
    assert selected["turn_sign"] == -1.0


def test_synchronized_threat_uses_low_speed_arc_not_rotation_only():
    candidate = {
        "closest_approach_time_s": 0.7,
        "closest_approach_distance_m": 0.62,
        "collision_clearance_m": 0.02,
        "distance_m": 1.2,
        "lateral_m": -0.2,
    }
    linear, angular, selected = apply_provisional_collision_intervention(
        0.40, 0.05, (candidate,), 0.50
    )
    assert linear == pytest.approx(0.10)
    assert angular == pytest.approx(0.32)
    assert selected["forward_cap_mps"] == pytest.approx(0.10)


def test_turn_direction_hold_applies_only_inside_lateral_deadband():
    candidate = {
        "closest_approach_time_s": 1.2,
        "closest_approach_distance_m": 0.70,
        "collision_clearance_m": 0.10,
        "distance_m": 1.4,
        "lateral_m": -0.1,
    }
    _, angular, selected = apply_provisional_collision_intervention(
        0.30, 0.0, (candidate,), 0.50, preferred_turn_sign=-1.0
    )
    assert angular == pytest.approx(-0.28)
    assert selected["turn_sign"] == -1.0


def test_verified_turn_lock_overrides_tracker_lateral_jump():
    candidate = {
        "closest_approach_time_s": 1.2,
        "closest_approach_distance_m": 0.40,
        "collision_clearance_m": -0.10,
        "distance_m": 1.4,
        "lateral_m": -0.4,
    }
    _, angular, selected = apply_provisional_collision_intervention(
        0.30, 0.0, (candidate,), 0.50, preferred_turn_sign=-1.0
    )
    assert angular == pytest.approx(-0.28)
    assert selected["turn_sign"] == -1.0


def test_goal_direction_selects_side_for_near_center_threat():
    candidate = {
        "closest_approach_time_s": 1.2,
        "closest_approach_distance_m": 0.40,
        "collision_clearance_m": -0.10,
        "distance_m": 1.4,
        "lateral_m": 0.1,
    }
    _, angular, selected = apply_provisional_collision_intervention(
        0.30, 0.0, (candidate,), 0.50, goal_turn_sign=1.0
    )
    assert angular == pytest.approx(0.28)
    assert selected["turn_sign"] == 1.0


def test_far_horizon_candidate_cannot_take_over_control():
    candidate = {
        "closest_approach_time_s": 3.6,
        "closest_approach_distance_m": 0.20,
        "collision_clearance_m": -0.80,
        "distance_m": 2.4,
        "lateral_m": 0.1,
    }
    assert apply_provisional_collision_intervention(
        0.30, 0.20, (candidate,), 0.50
    ) == (0.30, 0.20, None)


def test_confirmed_dynamic_collision_can_trigger_at_three_seconds():
    candidate = {
        "closest_approach_time_s": 2.8,
        "closest_approach_distance_m": 0.40,
        "collision_clearance_m": -0.20,
        "distance_m": 1.5,
        "lateral_m": 0.2,
    }
    linear, angular, selected = apply_provisional_collision_intervention(
        0.40, 0.10, (candidate,), 0.50, goal_turn_sign=1.0
    )
    assert selected is not None
    assert linear == pytest.approx(0.18)
    assert angular == pytest.approx(0.18)


def test_mapless_confirmed_dynamic_can_trigger_at_full_forecast_horizon():
    candidate = {
        "closest_approach_time_s": 3.6,
        "closest_approach_distance_m": 0.74,
        "collision_clearance_m": -0.31,
        "distance_m": 2.55,
        "lateral_m": -0.23,
        "mapless_dynamic_confirmed": True,
    }
    linear, angular, selected = apply_provisional_collision_intervention(
        0.31, -0.36, (candidate,), 0.50, goal_turn_sign=-1.0
    )
    assert selected is not None
    assert linear == pytest.approx(0.18)
    assert angular == pytest.approx(-0.18)


def test_vetted_wide_vehicle_starts_with_early_shallow_arc():
    candidate = {
        "closest_approach_time_s": 2.8,
        "closest_approach_distance_m": 0.70,
        "distance_m": 2.2,
        "lateral_m": 0.4,
        "early_wide_vehicle": True,
    }
    linear, angular, selected = apply_provisional_collision_intervention(
        0.40, 0.40, (candidate,), 0.50
    )
    assert linear == pytest.approx(0.20)
    assert angular == pytest.approx(-0.16)
    assert selected["turn_magnitude_radps"] == pytest.approx(0.16)


def test_goal_heading_stabilizer_corrects_opposite_nominal_turn_only():
    linear, angular, applied = stabilize_goal_heading(
        0.40, -0.20, 0.60, 0.50
    )
    assert applied
    assert linear == pytest.approx(0.20)
    assert angular == pytest.approx(0.50)
    assert stabilize_goal_heading(
        0.30, 0.20, 0.60, 0.50
    ) == (0.30, 0.20, False)


def test_only_immediate_center_distance_collapses_arc_to_stop():
    candidate = {
        "closest_approach_time_s": 0.2,
        "closest_approach_distance_m": 0.30,
        "collision_clearance_m": -0.30,
        "distance_m": 0.40,
        "lateral_m": 0.1,
    }
    linear, _, selected = apply_provisional_collision_intervention(
        0.30, 0.0, (candidate,), 0.50
    )
    assert linear == 0.0
    assert selected["forward_cap_mps"] == 0.0


def test_reactive_arc_hold_preserves_direction_and_caps_forward_speed():
    linear, angular, applied = apply_reactive_arc_hold(
        0.45, 0.05, -1.0, 0.50
    )
    assert applied
    assert linear == pytest.approx(0.18)
    assert angular == pytest.approx(-0.30)


def test_reactive_arc_hold_never_creates_forward_motion_from_stop_or_reverse():
    assert apply_reactive_arc_hold(
        0.0, 0.0, 1.0, 0.50
    ) == (0.0, 0.0, False)
    assert apply_reactive_arc_hold(
        -0.05, 0.10, -1.0, 0.50
    ) == (-0.05, 0.10, False)


def test_direction_lock_preserves_turn_while_stopped():
    assert apply_turn_direction_lock(
        0.0, 0.45, -1.0, 0.50
    ) == (0.0, -0.45, True)


def test_angular_command_step_prevents_instant_sign_reversal():
    value, applied = limit_angular_command_step(0.50, -0.50)
    assert applied
    assert value == pytest.approx(-0.15)
    assert limit_angular_command_step(
        0.20, 0.10
    ) == (0.20, False)


def test_angular_slew_budget_uses_actual_interval_and_caps_delays():
    assert angular_slew_step_budget(0.10, 1.0, 0.35) == pytest.approx(0.10)
    assert angular_slew_step_budget(1.20, 1.0, 0.35) == pytest.approx(0.35)
    assert angular_slew_step_budget(-0.10, 1.0, 0.35) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        angular_slew_step_budget(0.10, 0.0, 0.35)


def test_elapsed_slew_budget_prevents_a_full_sign_flip_after_a_short_cycle():
    step = angular_slew_step_budget(0.10, 1.0, 0.35)
    value, applied = limit_angular_command_step(-0.50, 0.50, step)
    assert applied
    assert value == pytest.approx(0.40)


def test_front_clearance_governor_tightens_existing_static_avoidance_arc():
    linear, angular, details = apply_front_clearance_arc_governor(
        0.35, -0.50, 1.46, 0.50
    )
    assert details is not None
    assert linear == pytest.approx(0.1825)
    assert angular == pytest.approx(-0.50)
    assert details["turn_sign"] == -1.0


def test_front_clearance_governor_does_not_oversteer_a_gentle_turn():
    linear, angular, details = apply_front_clearance_arc_governor(
        0.25, 0.22, 1.20, 0.50
    )
    assert details is not None
    assert linear < 0.25
    assert angular == pytest.approx(0.30)


def test_front_clearance_governor_turns_away_from_obstacle_bearing():
    linear, angular, details = apply_front_clearance_arc_governor(
        0.25, 0.50, 1.10, 0.50, front_obstacle_angle_rad=0.20
    )
    assert linear < 0.25
    assert angular == pytest.approx(-0.35)
    assert details["turn_away_applied"]
    _, angular, _ = apply_front_clearance_arc_governor(
        0.25, -0.50, 1.10, 0.50, front_obstacle_angle_rad=-0.20
    )
    assert angular == pytest.approx(0.35)
    linear, angular, details = apply_front_clearance_arc_governor(
        0.0, 0.50, 0.78, 0.50, front_obstacle_angle_rad=0.20
    )
    assert linear == 0.0
    assert angular == pytest.approx(-0.35)
    assert details["turn_away_applied"]


def test_front_clearance_governor_never_invents_a_turn_or_motion():
    assert apply_front_clearance_arc_governor(
        0.35, 0.10, 1.00, 0.50
    ) == (0.35, 0.10, None)
    assert apply_front_clearance_arc_governor(
        0.0, -0.50, 1.00, 0.50
    ) == (0.0, -0.50, None)
    assert apply_front_clearance_arc_governor(
        0.35, -0.50, None, 0.50
    ) == (0.35, -0.50, None)


def test_near_body_governor_slows_before_side_obstacle_enters_front_sector():
    linear, details = apply_near_body_speed_governor(0.35, 0.72)
    assert details is not None
    assert linear == pytest.approx(0.09529411764705883)


def test_verified_near_body_escape_requires_open_side_and_matching_turn():
    angular, details = verified_near_body_turn_escape(
        0.9, -0.30, 0.80, 0.20, 0.50
    )
    assert details is not None
    assert angular == pytest.approx(0.18)
    assert verified_near_body_turn_escape(
        -0.9, -0.30, 0.80, 0.20, 0.50
    ) == (0.0, None)
    assert verified_near_body_turn_escape(
        0.9, -0.30, 0.40, 0.80, 0.50
    ) == (0.0, None)


def test_front_hard_stop_is_replaced_by_vanishing_arc_creep():
    value, reason = apply_front_arc_creep(0.20, 0.68, 0.808, 0.35)
    assert reason == "front_arc_creep"
    assert 0.0 < value < 0.08
    assert apply_front_arc_creep(
        0.20, 1.0, 0.808, 0.35
    ) == (0.20, None)


def test_motion_session_timeout_starts_only_after_first_command():
    assert not motion_session_expired(None, 500.0, 120.0)
    assert not motion_session_expired(400.0, 500.0, 120.0)
    assert motion_session_expired(300.0, 500.0, 120.0)
