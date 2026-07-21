import numpy as np
import pytest

from experiments.rl.analyze_l246_tracking_root_causes import (
    classify_root_causes,
    detect_projection_events,
    estimate_step_budget,
    first_true_index,
    rolling_window_gain,
    rolling_window_mean,
)


def test_step_budget_estimates_are_distinct_and_ceiled():
    result = estimate_step_budget(
        path_length_m=10.0,
        control_dt_s=0.1,
        maximum_speed_mps=0.5,
        mean_forward_applied_speed_mps=0.25,
        observed_steps=100,
        final_completion_ratio=0.2,
    )
    assert result == {
        "theoretical_min_steps": 200,
        "applied_velocity_estimated_steps": 400,
        "empirical_progress_estimated_steps": 500,
    }


def test_step_budget_rejects_nonpositive_physics():
    with pytest.raises(ValueError):
        estimate_step_budget(0.0, 0.1, 0.5, 0.2, 100, 0.5)


def test_rolling_metrics_use_frozen_window_endpoint_semantics():
    values = np.asarray([0.0, 0.01, 0.02, 0.02, 0.02])
    gain = rolling_window_gain(values, window=3)
    mean = rolling_window_mean(np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]), window=3)
    assert np.isnan(gain[:2]).all()
    assert gain[2] == pytest.approx(0.02)
    assert gain[4] == pytest.approx(0.0)
    assert mean[2] == pytest.approx(2.0 / 3.0)
    assert first_true_index(gain <= 0.005) == 4


def test_projection_events_apply_preregistered_strict_thresholds():
    progress = np.asarray([0.10, 0.101, 0.098, 0.129, 0.127])
    events = detect_projection_events(progress)
    assert events["regression_indices"].tolist() == [2, 4]
    assert events["jump_indices"].tolist() == [3]


def test_root_cause_classification_separates_primary_and_deployment_flags():
    budget = {
        "scene": "infinity",
        "theoretical_min_steps": 750,
        "final_completion_ratio": 0.1,
    }
    event = {
        "termination_reason": "boundary_violation",
        "stall_class": "repeated_turning",
        "first_dense_safety_step": -1,
        "proposal_fallback_fraction_mean": 0.9,
        "planner_ms_p99": 400.0,
        "planner_deadline_ms": 100.0,
        "planner_deadline_miss_rate": 1.0,
        "success": False,
        "center_crossing_changes": 0,
        "progress_regressions": 0,
        "progress_jumps": 0,
    }
    result = classify_root_causes(budget, event)
    assert result["primary_root_causes"] == (
        "step_budget|candidate_boundary_cost_or_termination"
    )
    assert result["infinity_projection_status"] == "center_not_reached"
    assert result["planner_deadline_failed"] is True
