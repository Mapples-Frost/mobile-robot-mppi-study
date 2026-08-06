import math

import numpy as np
import pytest

from mobile_robot_mppi.obstacles.collision_risk import (
    GaussianMixtureObstacleForecast,
)
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


def _multi_tracker(tracks):
    indices = tuple(track[0] for track in tracks)
    return {
        "mapless_dynamic_track_indices": indices,
        "forecast_track_indices": indices,
        "tracks": tuple({
            "track_index": index,
            "associated": True,
            "forecast_valid": True,
            "motion_confirmed": True,
            "mapless_classification": "dynamic",
            "measurement_x": position[0],
            "measurement_y": position[1],
            "measurement_velocity_x_mps": velocity[0],
            "measurement_velocity_y_mps": velocity[1],
            "innovation_nis": 1.0,
            "change_triggered": False,
            "selected_support_beams": support,
        } for index, position, velocity, support in tracks),
    }


def _update(manager, timestamp, position, velocity, **kwargs):
    forecasts = kwargs.pop("forecasts", ())
    return manager.update(
        timestamp_s=timestamp,
        pose=kwargs.pop("pose", (0.0, 0.0, 0.0)),
        goal=(5.0, 0.0),
        robot_speed_mps=kwargs.pop("robot_speed_mps", 0.4),
        tracker_diagnostics=_tracker(position, velocity, **kwargs),
        forecasts=forecasts,
    )


def _forecast(points, *, dt=0.2, std=0.05):
    means = np.asarray(points, dtype=np.float64)[:, None, :]
    horizon = means.shape[0]
    covariance = np.eye(2, dtype=np.float64) * std ** 2
    return GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=dt,
        component_means=means,
        component_covariances=np.tile(covariance, (horizon, 1, 1, 1)),
        component_weights=np.ones((horizon, 1), dtype=np.float64),
        radius_m=0.25,
        source="encounter_test",
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
    assert result["encounter_temporary_waypoint"][1] > 0.0
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


def test_probability_forecast_owns_mode_geometry_when_velocity_fit_lags():
    manager = EncounterModeManager()
    crossing_forecast = _forecast((
        (2.0, 0.90),
        (2.0, 0.80),
        (2.0, 0.70),
        (2.0, 0.60),
    ))

    result = _update(
        manager,
        0.0,
        (2.0, 1.0),
        (-0.5, 0.0),  # stale fit still claims frontal approach
        forecasts=(crossing_forecast,),
    )

    assert result["encounter_candidate_mode"] == "straight_crossing"
    assert result["encounter_velocity_source"] == "gaussian_mixture_forecast"
    assert result["encounter_forecast_available"] is True
    assert result["encounter_human_lateral_velocity_mps"] == pytest.approx(-0.5)


def test_probability_forecast_supplies_goal_line_intersection_and_cpa():
    manager = EncounterModeManager()
    crossing_forecast = _forecast((
        (1.6, 0.75),
        (1.6, 0.50),
        (1.6, 0.25),
        (1.6, 0.00),
        (1.6, -0.25),
    ), dt=0.4, std=0.08)

    result = _update(
        manager,
        0.0,
        (1.6, 1.0),
        (0.0, -0.2),
        forecasts=(crossing_forecast,),
    )

    assert result["encounter_forecast_goal_line_crossing_used"] is True
    assert result[
        "encounter_predicted_goal_line_crossing_time_s"
    ] == pytest.approx(1.6)
    assert result[
        "encounter_predicted_goal_line_crossing_position_m"
    ] == pytest.approx(1.6)
    assert result["encounter_forecast_cpa_used"] is True
    assert result["encounter_forecast_d_cpa_conservative_m"] <= (
        result["encounter_forecast_d_cpa_mean_m"]
    )


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


def test_frontal_bypass_chooses_clearer_static_side_and_locks_it():
    manager = EncounterModeManager(EncounterModeConfig(shadow_only=False))
    result = None
    for step, x in enumerate((2.0, 1.95, 1.90)):
        result = manager.update(
            timestamp_s=0.1 * step,
            pose=(0.0, 0.0, 0.0),
            goal=(5.0, 0.0),
            robot_speed_mps=0.4,
            tracker_diagnostics=_tracker((x, 0.05), (-0.5, 0.0)),
            local_obstacles=((1.0, 0.55, 0.2), (1.7, 0.80, 0.2)),
        )

    assert result["encounter_phase"] == "frontal_approach"
    assert result["encounter_strategy"] == "right_bypass"
    assert result["encounter_locked_steering_side"] == -1
    assert result["encounter_left_bypass_clearance_m"] < (
        result["encounter_right_bypass_clearance_m"]
    )
    assert result["encounter_rear_pass_inhibited"] is True
    assert result["encounter_dynamic_escape_inhibited"] is True


def test_crossing_front_pass_is_selected_only_with_time_and_space_margin():
    manager = EncounterModeManager()
    result = None
    for step, y in enumerate((1.20, 1.175, 1.15)):
        result = _update(
            manager,
            0.1 * step,
            (0.8, y),
            (0.0, -0.25),
            robot_speed_mps=0.4,
        )

    assert result["encounter_phase"] == "straight_crossing"
    assert result["encounter_front_space_clear"] is True
    assert result["encounter_front_pass_feasible"] is True
    assert result["encounter_strategy"] == "front_pass"
    assert result["encounter_locked_steering_side"] == -1
    assert result["encounter_temporary_waypoint"][1] < 0.0


def test_person_stopping_in_active_crossing_does_not_cancel_bypass():
    manager = EncounterModeManager()
    for step, y in enumerate((0.8, 0.75, 0.70)):
        _update(manager, 0.1 * step, (1.2, y), (0.0, -0.5))

    result = None
    for step in range(3, 6):
        result = _update(manager, 0.1 * step, (1.2, 0.68), (0.0, 0.0))

    assert result["encounter_confirmed_mode"] == "stationary"
    assert result["encounter_phase"] == "straight_crossing"
    assert result["encounter_strategy"] == "behind_pass"


def test_crossing_to_frontal_change_uses_high_confidence_fast_path():
    manager = EncounterModeManager()
    for step, y in enumerate((1.0, 0.95, 0.90)):
        _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))

    first = _update(
        manager, 0.3, (1.96, 0.87), (-0.5, 0.0), change=True, nis=8.0
    )
    second = _update(manager, 0.4, (1.91, 0.87), (-0.5, 0.0))

    assert first["encounter_change_detected"] is True
    assert first["encounter_candidate_fast_path"] is True
    assert first["encounter_active_reclassification"] is True
    assert first["encounter_confirmed_mode"] == "frontal_approach"
    assert first["encounter_phase"] == "frontal_approach"
    assert first["encounter_strategy"] in ("left_bypass", "right_bypass")
    assert second["encounter_phase"] == "frontal_approach"


def test_frontal_to_crossing_change_uses_current_candidate_immediately():
    manager = EncounterModeManager()
    for step, x in enumerate((2.0, 1.95, 1.90)):
        _update(manager, 0.1 * step, (x, 0.1), (-0.5, 0.0))

    result = _update(
        manager,
        0.3,
        (1.85, 0.05),
        (0.0, -0.5),
        change=True,
        nis=8.0,
    )

    assert result["encounter_candidate_mode"] == "straight_crossing"
    assert result["encounter_active_reclassification"] is True
    assert result["encounter_confirmed_mode"] == "straight_crossing"
    assert result["encounter_phase"] == "straight_crossing"
    assert result["encounter_strategy"] == "behind_pass"


def test_stable_frontal_reclassification_replaces_crossing_without_change_flag():
    manager = EncounterModeManager(EncounterModeConfig(
        change_score_threshold=1.1
    ))
    for step, y in enumerate((1.0, 0.95, 0.90)):
        _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))

    result = None
    for step, x in enumerate((1.95, 1.90, 1.85), start=3):
        result = _update(manager, 0.1 * step, (x, 0.87), (-0.5, 0.0))

    assert result["encounter_candidate_fast_path"] is False
    assert result["encounter_confirmed_mode"] == "frontal_approach"
    assert result["encounter_frontal_admissible"] is True
    assert result["encounter_phase"] == "frontal_approach"


def test_forecast_detected_crossing_to_frontal_mutation_relocks_strategy():
    manager = EncounterModeManager()
    for step, y in enumerate((1.0, 0.95, 0.90)):
        crossing = _forecast(tuple(
            (2.0, y - 0.1 * horizon) for horizon in range(1, 5)
        ))
        _update(
            manager,
            0.1 * step,
            (2.0, y),
            (0.0, -0.5),
            forecasts=(crossing,),
        )

    def frontal_forecast(x, y):
        return _forecast(tuple(
            (x - 0.1 * horizon, y) for horizon in range(1, 5)
        ))

    first = _update(
        manager,
        0.3,
        (1.96, 0.87),
        (0.0, -0.5),  # stale raw fit; forecast has already turned toward car
        change=True,
        nis=8.0,
        forecasts=(frontal_forecast(1.96, 0.87),),
    )
    second = _update(
        manager,
        0.4,
        (1.91, 0.87),
        (0.0, -0.5),
        forecasts=(frontal_forecast(1.91, 0.87),),
    )

    assert first["encounter_candidate_mode"] == "frontal_approach"
    assert first["encounter_change_detected"] is True
    assert "forecast_direction" in first["encounter_change_reason"]
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

    first_evidence = _update(
        manager, 0.3, (2.0, 0.08), (0.0, -0.5)
    )
    crossed = _update(manager, 0.4, (2.0, -0.02), (0.0, -0.5))

    assert first_evidence["encounter_completion_evidence"] is True
    assert first_evidence["encounter_line_crossed"] is False
    assert crossed["encounter_line_crossed"] is True
    assert crossed["encounter_phase"] == "rejoin"
    assert crossed["encounter_temporary_waypoint"][1] == pytest.approx(0.0)
    assert crossed["encounter_rear_pass_inhibited_shadow"] is True
    assert crossed["encounter_forward_passage_inhibited_shadow"] is True


def test_rejoin_requires_consecutive_goal_alignment_and_risk_clearance():
    manager = EncounterModeManager(EncounterModeConfig(rejoin_clear_cycles=3))
    for step, y in enumerate((0.40, 0.30, 0.20, 0.08, -0.02)):
        _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))

    result = None
    for step in range(3):
        result = manager.update(
            timestamp_s=0.5 + 0.1 * step,
            pose=(0.2, 0.0, 0.0),
            goal=(5.0, 0.0),
            robot_speed_mps=0.4,
            tracker_diagnostics={"tracks": ()},
        )

    assert result["encounter_phase"] == "idle"
    assert result["encounter_strategy"] == "none"
    assert result["encounter_rear_pass_inhibited_shadow"] is False


def test_rejoin_is_not_restarted_by_same_confirmed_crossing_track():
    manager = EncounterModeManager()
    for step, y in enumerate((0.40, 0.30, 0.20, 0.08, -0.02)):
        result = _update(
            manager,
            0.1 * step,
            (2.0, y),
            (0.0, -0.5),
            pose=(0.0, 0.0, 0.0),
        )

    assert result["encounter_phase"] == "rejoin"
    continued = _update(
        manager,
        0.5,
        (2.0, -0.12),
        (0.0, -0.5),
        pose=(2.0, 0.40, 0.0),
    )

    assert continued["encounter_phase"] == "rejoin"
    assert continued["encounter_phase_before"] == "rejoin"
    assert continued["encounter_temporary_waypoint"][1] == pytest.approx(0.0)


def test_rejoin_does_not_release_near_goal_with_lateral_error():
    manager = EncounterModeManager()
    for step, y in enumerate((0.40, 0.30, 0.20, 0.08, -0.02)):
        result = _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))
    assert result["encounter_phase"] == "rejoin"

    continued = None
    for step in range(4):
        continued = manager.update(
            timestamp_s=0.6 + 0.1 * step,
            pose=(5.0, 0.30, 0.0),
            goal=(5.0, 0.0),
            robot_speed_mps=0.2,
            tracker_diagnostics={"tracks": ()},
        )

    assert continued["encounter_phase"] == "rejoin"
    assert continued["encounter_strategy"] != "none"


def test_stationary_robot_cannot_complete_off_center_frontal_bypass():
    manager = EncounterModeManager()
    result = None
    for step, x in enumerate((2.0, 1.95, 1.90, 1.85, 1.80)):
        result = _update(
            manager, 0.1 * step, (x, -0.80), (-0.5, 0.0)
        )

    assert result["encounter_phase"] == "frontal_approach"
    assert result["encounter_line_crossed"] is False
    assert result["encounter_completion_evidence"] is False


def test_active_encounter_rejects_spatially_unrelated_track_slot():
    manager = EncounterModeManager()
    for step, x in enumerate((2.0, 1.95, 1.90)):
        entered = _update(
            manager, 0.1 * step, (x, -0.10), (-0.5, 0.0)
        )
    assert entered["encounter_phase"] == "frontal_approach"

    unrelated = _update(
        manager,
        0.3,
        (-0.4, -0.7),
        (0.1, 0.0),
        index=4,
    )

    assert unrelated["encounter_track_index"] is None
    assert unrelated["encounter_phase"] == "frontal_approach"
    assert unrelated["encounter_line_crossed"] is False
    assert unrelated["encounter_lost_track_cycles"] == 1


def test_active_mode_holds_through_short_track_dropout():
    manager = EncounterModeManager()
    for step, y in enumerate((0.8, 0.75, 0.70)):
        _update(manager, 0.1 * step, (1.2, y), (0.0, -0.5))

    result = None
    for step in range(4):
        result = manager.update(
            timestamp_s=0.3 + 0.1 * step,
            pose=(0.0, 0.0, 0.0),
            goal=(5.0, 0.0),
            robot_speed_mps=0.4,
            tracker_diagnostics={"tracks": ()},
        )

    assert result["encounter_phase"] == "straight_crossing"
    assert result["encounter_track_hold_active"] is True
    assert result["encounter_lost_track_cycles"] == 4


def test_same_person_reversal_survives_a_leg_track_slot_change():
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
    assert changed_identity["encounter_change_detected"] is True
    assert changed_identity["encounter_track_continuous"] is True
    assert changed_identity["encounter_track_switched"] is False
    assert changed_identity["encounter_person_member_track_indices"] == (4,)


def test_frontal_admission_rejects_a_far_lateral_track():
    manager = EncounterModeManager()
    result = None
    for step, x in enumerate((0.17, 0.14, 0.11)):
        result = _update(
            manager, 0.1 * step, (x, -3.73), (-0.5, 0.0)
        )

    assert result["encounter_candidate_mode"] == "frontal_approach"
    assert result["encounter_frontal_admissible"] is False
    assert abs(result["encounter_human_lateral_position_m"]) > (
        result["encounter_frontal_corridor_half_width_m"]
    )
    assert result["encounter_phase"] == "idle"


def test_frontal_admission_rejects_a_track_behind_the_chassis():
    manager = EncounterModeManager()
    result = None
    for step, x in enumerate((-0.30, -0.25, -0.20)):
        result = _update(manager, 0.1 * step, (x, 0.0), (0.5, 0.0))

    assert result["encounter_frontal_admissible"] is False
    assert result["encounter_phase"] == "idle"


def test_person_temporal_closing_recovers_frontal_mode_from_stale_velocity():
    manager = EncounterModeManager()
    result = None
    for step, x in enumerate((0.95, 0.85, 0.75, 0.65, 0.55)):
        result = _update(
            manager,
            0.1 * step,
            (x, 0.0),
            (0.0, -0.3),  # stale tracker fit still claims a crossing
        )

    assert result["encounter_person_temporal_velocity_valid"] is True
    assert result[
        "encounter_person_temporal_longitudinal_velocity_mps"
    ] < -0.5
    assert result["encounter_candidate_mode"] == "frontal_approach"
    assert result["encounter_frontal_admissible"] is True
    assert result["encounter_phase"] == "frontal_approach"


def test_crossing_admission_requires_a_future_goal_line_intersection():
    manager = EncounterModeManager()
    result = None
    for step, y in enumerate((0.70, 0.75, 0.80)):
        result = _update(manager, 0.1 * step, (5.4, y), (0.0, 0.5))

    assert result["encounter_candidate_mode"] == "straight_crossing"
    assert result["encounter_crossing_admissible"] is False
    assert result["encounter_phase"] == "idle"


def test_leg_tracks_fuse_and_keep_one_person_id_across_slot_changes():
    manager = EncounterModeManager()

    first = manager.update(
        timestamp_s=0.0,
        pose=(0.0, 0.0, 0.0),
        goal=(5.0, 0.0),
        robot_speed_mps=0.4,
        tracker_diagnostics=_multi_tracker((
            (2, (2.0, 0.35), (0.0, -0.4), 5),
            (3, (2.0, 0.75), (0.0, -0.4), 4),
        )),
    )
    second = manager.update(
        timestamp_s=0.1,
        pose=(0.0, 0.0, 0.0),
        goal=(5.0, 0.0),
        robot_speed_mps=0.4,
        tracker_diagnostics=_multi_tracker((
            (4, (2.0, 0.31), (0.0, -0.4), 5),
            (5, (2.0, 0.71), (0.0, -0.4), 4),
        )),
    )

    assert first["encounter_person_fused"] is True
    assert first["encounter_person_member_track_indices"] == (2, 3)
    assert second["encounter_person_fused"] is True
    assert second["encounter_person_member_track_indices"] == (4, 5)
    assert second["encounter_person_id"] == first["encounter_person_id"]
    assert second["encounter_track_continuous"] is True
    assert second["encounter_person_association_residual_m"] < 0.1


def test_person_fusion_does_not_chain_merge_a_crowd():
    manager = EncounterModeManager()
    result = manager.update(
        timestamp_s=0.0,
        pose=(0.0, 0.0, 0.0),
        goal=(5.0, 0.0),
        robot_speed_mps=0.4,
        tracker_diagnostics=_multi_tracker((
            (2, (2.0, 0.00), (0.0, -0.4), 5),
            (3, (2.0, 0.40), (0.0, -0.4), 5),
            (4, (2.0, 0.80), (0.0, -0.4), 5),
        )),
    )

    assert result["encounter_person_count"] == 2
    assert len(result["encounter_person_member_track_indices"]) == 2


def test_rejoin_waypoint_rolls_forward_and_clamps_to_the_goal():
    manager = EncounterModeManager()
    for step, y in enumerate((0.40, 0.30, 0.20, 0.08, -0.02)):
        result = _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))
    assert result["encounter_phase"] == "rejoin"
    initial_x = result["encounter_temporary_waypoint"][0]

    advanced = manager.update(
        timestamp_s=0.5,
        pose=(1.20, 0.50, 0.0),
        goal=(5.0, 0.0),
        robot_speed_mps=0.4,
        tracker_diagnostics={"tracks": ()},
    )
    near_goal = manager.update(
        timestamp_s=0.6,
        pose=(4.50, 0.50, 0.0),
        goal=(5.0, 0.0),
        robot_speed_mps=0.4,
        tracker_diagnostics={"tracks": ()},
    )

    assert advanced["encounter_temporary_waypoint"][0] > initial_x
    assert advanced["encounter_temporary_waypoint"][0] > 1.20
    assert near_goal["encounter_temporary_waypoint"] == pytest.approx((5.0, 0.0))


def test_confirmed_new_frontal_encounter_interrupts_rejoin():
    manager = EncounterModeManager()
    for step, y in enumerate((0.40, 0.30, 0.20, 0.08, -0.02)):
        result = _update(manager, 0.1 * step, (2.0, y), (0.0, -0.5))
    assert result["encounter_phase"] == "rejoin"

    for step, x in enumerate((0.90, 0.82, 0.74), start=5):
        result = _update(
            manager,
            0.1 * step,
            (x, 0.05),
            (-0.5, 0.0),
            index=7,
        )

    assert result["encounter_frontal_admissible"] is True
    assert result["encounter_confirmed_mode"] == "frontal_approach"
    assert result["encounter_phase"] == "frontal_approach"
