import json

import numpy as np

from deploy.raspberry_pi5_scout.run_remote_cuda_full import (
    _goal_stop_requested,
    _planner_diagnostic_trace,
    _real_robot_diagnostic_payload,
    _safety_diagnostic_trace,
    _select_runtime_controller,
)
from deploy.raspberry_pi5_scout.run_silent_full import _install_mapless_tracker
from deploy.raspberry_pi5_scout.run_silent_full import _json_value


def test_planner_diagnostic_trace_keeps_causal_fields_and_removes_duplicates():
    diagnostics = {
        "probabilistic_obstacle_traversal_candidate_selected": True,
        "probabilistic_obstacle_stop_maximum_probability": 0.42,
        "paper_guided_elite_count": 7,
        "supervised_influence_survived_guard": False,
        "residual_policy_authority_mean": 0.25,
        "optimizer_selected_cost_gap": 1.5,
        "physical_tracker_motion_fallback_applied": True,
        "physical_tracker_motion_refresh_requested": False,
        "physical_goal_bearing_error_rad": -0.42,
        "physical_goal_distance_m": 3.1,
        "dynamic_obstacle_tracker_trace": {"large": "separate"},
        "probabilistic_obstacle_forecast_trace": [{"large": "separate"}],
        "unrelated_internal_payload": np.ones((8, 8)),
    }

    trace = _planner_diagnostic_trace(diagnostics)

    assert trace["probabilistic_obstacle_traversal_candidate_selected"] is True
    assert trace["probabilistic_obstacle_stop_maximum_probability"] == 0.42
    assert trace["paper_guided_elite_count"] == 7
    assert trace["supervised_influence_survived_guard"] is False
    assert trace["residual_policy_authority_mean"] == 0.25
    assert trace["optimizer_selected_cost_gap"] == 1.5
    assert trace["physical_tracker_motion_fallback_applied"] is True
    assert trace["physical_tracker_motion_refresh_requested"] is False
    assert trace["physical_goal_bearing_error_rad"] == -0.42
    assert trace["physical_goal_distance_m"] == 3.1
    assert "dynamic_obstacle_tracker_trace" not in trace
    assert "probabilistic_obstacle_forecast_trace" not in trace
    assert "unrelated_internal_payload" not in trace


def test_real_robot_diagnostic_payload_preserves_track_evidence_and_forecast():
    tracker = {
        "tracks": [{
            "track_index": 2,
            "measurement_x": 1.25,
            "measurement_y": -0.2,
            "measurement_speed_mps": 0.31,
            "selected_support_beams": 4,
            "mapless_classification": "dynamic",
            "mapless_motion_evidence": {
                "fitted_speed_mps": 0.29,
                "direction_coherence": 0.88,
            },
        }],
        "mapless_dynamic_track_indices": (2,),
    }
    forecast = [{
        "track_index": 2,
        "component_means": np.asarray([[[1.3, -0.2]]]),
        "component_covariances": np.eye(2)[None, None, :, :],
        "component_weights": np.asarray([[1.0]]),
    }]
    planner = {
        "probabilistic_obstacle_forecast_trace": forecast,
        "probabilistic_obstacle_traversal_candidate_selected": False,
    }
    safety = {
        "dynamic_obstacle_near_body_match": True,
        "executed_v": -0.1,
    }

    payload = _real_robot_diagnostic_payload(tracker, planner, safety)
    encoded = json.dumps(_json_value(payload), sort_keys=True)

    assert payload["schema_version"] == "pc_pi_full_proposed_diagnostics_v2"
    assert payload["tracker"]["tracks"][0]["mapless_classification"] == "dynamic"
    assert payload["tracker"]["tracks"][0]["mapless_motion_evidence"][
        "direction_coherence"
    ] == 0.88
    assert payload["forecasts"][0]["track_index"] == 2
    assert payload["planner"][
        "probabilistic_obstacle_traversal_candidate_selected"
    ] is False
    assert payload["safety"]["executed_v"] == -0.1
    assert "component_covariances" in encoded


def test_safety_trace_summarizes_front_points_without_logging_the_cloud():
    trace = _safety_diagnostic_trace({
        "reason": "front_obstacle_slow",
        "front_points": (
            {"range": 1.2, "x": 1.0, "y": 0.1},
            {"range": 0.7, "x": 0.6, "y": -0.1},
        ),
        "raw_points_base": (
            {"range": 4.0, "x": 4.0, "y": 0.0},
        ),
    })

    assert trace["reason"] == "front_obstacle_slow"
    assert trace["front_point_count"] == 2
    assert trace["front_point_minimum_range_m"] == 0.7
    assert trace["raw_points_base_count"] == 1
    assert trace["raw_points_base_minimum_range_m"] == 4.0
    assert "front_points" not in trace
    assert "raw_points_base" not in trace


def test_goal_stop_is_opt_in_and_uses_point_distance():
    requested, distance = _goal_stop_requested(
        2.86, 2.80, 3.0, 3.0, 0.25, enabled=True
    )
    disabled, _ = _goal_stop_requested(
        2.86, 2.80, 3.0, 3.0, 0.25, enabled=False
    )

    assert requested
    assert distance < 0.25
    assert not disabled


def test_nominal_runtime_fallback_is_explicit_and_validated():
    nominal = object()
    full = type("Full", (), {"nominal_controller": nominal})()

    assert _select_runtime_controller(full, False) is full
    assert _select_runtime_controller(full, True) is nominal
    with np.testing.assert_raises_regex(TypeError, "matched nominal"):
        _select_runtime_controller(object(), True)


def test_human_leg_tracker_profile_is_explicit_and_default_is_unchanged(
    monkeypatch,
):
    import deploy.raspberry_pi5_scout.run_silent_full as runner

    captured = []
    bootstrap_captured = []
    base = object()
    bootstrap = object()
    perception = type("Perception", (), {"dynamic_obstacle_tracker": base})()
    monkeypatch.setattr(
        runner.MotionBootstrapMultiObstacleTracker,
        "from_existing",
        staticmethod(
            lambda value, **kwargs: bootstrap_captured.append((value, kwargs))
            or bootstrap
        ),
    )
    monkeypatch.setattr(
        runner.MaplessStaticDynamicFilter,
        "from_existing",
        staticmethod(lambda value, **kwargs: captured.append((value, kwargs)) or object()),
    )

    _install_mapless_tracker(perception, human_leg_mode=False)
    assert bootstrap_captured[-1] == (base, {})
    assert captured[-1] == (bootstrap, {})

    perception.dynamic_obstacle_tracker = base
    _install_mapless_tracker(perception, human_leg_mode=True)
    assert bootstrap_captured[-1] == (
        base,
        {
            "required_motion_intervals": 2,
            "minimum_total_displacement_m": 0.06,
            "temporal_flow_threat_preemption_enabled": True,
            "temporal_flow_threat_maximum_ttc_s": 3.0,
            "temporal_flow_threat_minimum_support_beams": 3,
            "temporal_flow_threat_angle_tolerance_deg": 25.0,
            "temporal_flow_threat_range_tolerance_m": 0.75,
            "temporal_flow_threat_hold_cycles": 12,
        },
    )
    _, profile = captured[-1]
    assert profile["allow_temporal_flow_provisional"] is True
    assert profile["allow_compact_dynamic"] is True
    assert profile["dynamic_displacement_m"] == 0.14
    assert profile["minimum_direction_coherence"] == 0.60
    assert profile["static_track_eviction_cycles"] == 3
    assert profile["allow_recent_vehicle_fragment_dynamic"] is True
    assert profile["dynamic_hold_cycles"] == 12
    assert profile["dynamic_memory_ttl_s"] == 1.50
    assert profile[
        "dynamic_classification_temporal_corroboration_enabled"
    ] is True
    assert profile[
        "dynamic_classification_temporal_corroboration_maximum_ttc_s"
    ] == 6.0
    assert profile[
        "dynamic_classification_temporal_corroboration_hold_cycles"
    ] == 12
