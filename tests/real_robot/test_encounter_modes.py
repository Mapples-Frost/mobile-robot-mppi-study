import math

import pytest

from mobile_robot_mppi.real_robot.encounter_modes import (
    EncounterModeConfig,
    EncounterModeManager,
)


def _tracker(position, velocity, *, index=2, change=False, nis=1.0):
    speed = math.hypot(*velocity)
    return {
        "mapless_dynamic_track_indices": (index,),
        "forecast_track_indices": (index,),
        "tracks": ({
            "track_index": index,
            "associated": True,
            "forecast_valid": True,
            "motion_confirmed": True,
            "mapless_classification": "dynamic",
            "measurement_x": position[0],
            "measurement_y": position[1],
            "measurement_velocity_x_mps": velocity[0],
            "measurement_velocity_y_mps": velocity[1],
            "measurement_speed_mps": speed,
            "innovation_nis": nis,
            "change_triggered": change,
            "selected_support_beams": 5,
        },),
    }


def _update(manager, timestamp, position, velocity, **kwargs):
    return manager.update(
        timestamp_s=timestamp,
        pose=kwargs.pop("pose", (0.0, 0.0, 0.0)),
        goal=(5.0, 0.0),
        robot_speed_mps=kwargs.pop("robot_speed_mps", 0.4),
        tracker_diagnostics=_tracker(position, velocity, **kwargs),
    )


def test_left_to_right_crossing_confirms_and_locks_opposite_rear_pass_side():
    manager = EncounterModeManager()
    result = None
    for step, y in enumerate((1.0, 0.95, 0.90)):
        result = _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))

    assert result["encounter_candidate_mode"] == "straight_crossing"
    assert result["encounter_confirmed_mode"] == "straight_crossing"
    assert result["encounter_phase"] == "straight_crossing"
    assert result["encounter_strategy"] == "behind_pass"
    # Positive is the robot's left, opposite a left-to-right human velocity.
    assert result["encounter_locked_steering_side"] == 1
    assert result["encounter_rear_pass_inhibited_shadow"] is True
    assert result["encounter_control_enabled"] is False


def test_oblique_crossing_and_frontal_approach_use_goal_relative_angle():
    crossing = EncounterModeManager()
    frontal = EncounterModeManager()

    oblique = _update(crossing, 0.0, (2.0, 1.0), (-0.3, -0.4))
    approach = _update(frontal, 0.0, (2.0, 0.1), (-0.5, 0.0))

    assert oblique["encounter_candidate_mode"] == "oblique_crossing"
    assert 35.0 < oblique["encounter_approach_angle_deg"] < 40.0
    assert approach["encounter_candidate_mode"] == "frontal_approach"
    assert approach["encounter_approach_angle_deg"] == pytest.approx(90.0)


def test_stationary_and_receding_are_explicit_non_encounter_classes():
    stationary = _update(
        EncounterModeManager(), 0.0, (2.0, 0.2), (0.01, 0.0)
    )
    receding = _update(
        EncounterModeManager(), 0.0, (2.0, 0.2), (0.4, 0.0)
    )

    assert stationary["encounter_candidate_mode"] == "stationary"
    assert receding["encounter_candidate_mode"] == "receding"
    assert stationary["encounter_phase"] == "idle"
    assert receding["encounter_phase"] == "idle"


def test_crossing_to_frontal_change_uses_two_cycle_fast_path():
    manager = EncounterModeManager()
    for step, y in enumerate((1.0, 0.95, 0.90)):
        _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))

    first = _update(
        manager, 0.3, (1.96, 0.87), (-0.5, 0.0), change=True, nis=8.0
    )
    second = _update(manager, 0.4, (1.91, 0.87), (-0.5, 0.0))

    assert first["encounter_change_detected"] is True
    assert first["encounter_candidate_fast_path"] is True
    assert first["encounter_confirmed_mode"] == "straight_crossing"
    assert second["encounter_confirmed_mode"] == "frontal_approach"
    assert second["encounter_phase"] == "frontal_approach"
    assert second["encounter_strategy"] in ("left_bypass", "right_bypass")


def test_track_position_jump_is_not_a_human_behaviour_change():
    manager = EncounterModeManager()
    _update(manager, 0.0, (2.0, 1.0), (0.0, -0.5))
    jumped = _update(
        manager, 0.1, (2.0, -3.5), (-0.5, 0.0), change=True, nis=20.0
    )

    assert jumped["encounter_track_continuous"] is False
    assert jumped["encounter_change_detected"] is False
    assert jumped["encounter_change_reason"] == "position_jump"
    assert jumped["encounter_candidate_fast_path"] is False


def test_crossing_frozen_goal_line_enters_rejoin_without_returning_to_idle():
    manager = EncounterModeManager()
    for step, y in enumerate((0.40, 0.30, 0.20)):
        result = _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))
    assert result["encounter_phase"] == "straight_crossing"

    crossed = _update(manager, 0.3, (2.0, 0.08), (0.0, -0.5))

    assert crossed["encounter_line_crossed"] is True
    assert crossed["encounter_phase"] == "rejoin"
    assert crossed["encounter_rear_pass_inhibited_shadow"] is True
    assert crossed["encounter_forward_passage_inhibited_shadow"] is True


def test_rejoin_requires_consecutive_goal_alignment_and_risk_clearance():
    manager = EncounterModeManager(EncounterModeConfig(rejoin_clear_cycles=3))
    for step, y in enumerate((0.40, 0.30, 0.20, 0.08)):
        _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))

    result = None
    for step in range(3):
        result = manager.update(
            timestamp_s=0.4 + 0.1 * step,
            pose=(0.2, 0.0, 0.0),
            goal=(5.0, 0.0),
            robot_speed_mps=0.4,
            tracker_diagnostics={"tracks": ()},
        )

    assert result["encounter_phase"] == "idle"
    assert result["encounter_strategy"] == "none"
    assert result["encounter_rear_pass_inhibited_shadow"] is False


def test_same_track_reversal_is_detected_but_new_track_identity_is_not():
    same = EncounterModeManager()
    other = EncounterModeManager()
    for step, y in enumerate((0.7, 0.65, 0.6)):
        _update(same, 0.1 * step, (2.0, y), (0.0, -0.5))
        _update(other, 0.1 * step, (2.0, y), (0.0, -0.5))

    reversed_same = _update(same, 0.3, (2.0, 0.6), (0.0, 0.5))
    changed_identity = _update(
        other, 0.3, (2.0, 0.6), (0.0, 0.5), index=4, change=True
    )

    assert reversed_same["encounter_change_detected"] is True
    assert reversed_same["encounter_track_continuous"] is True
    assert changed_identity["encounter_change_detected"] is False
    assert changed_identity["encounter_track_switched"] is True
