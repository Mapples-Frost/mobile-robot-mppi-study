import pytest

from experiments.rl.evaluate_direct_actor_paths import summarize


def test_direct_actor_path_summary_aggregates_episode_units():
    rows = [
        {
            "success": True,
            "collision": False,
            "return": 2.0,
            "cross_track_rmse": 0.1,
            "cross_track_max": 0.2,
            "heading_rmse": 0.3,
            "path_completion_ratio": 1.0,
            "raw_path_completion_ratio": 1.0,
            "minimum_clearance": 0.5,
            "unbounded_clearance": False,
            "control_jerk": 0.4,
        },
        {
            "success": False,
            "collision": True,
            "return": -2.0,
            "cross_track_rmse": 0.3,
            "cross_track_max": 0.4,
            "heading_rmse": 0.5,
            "path_completion_ratio": 0.5,
            "raw_path_completion_ratio": 1.0,
            "minimum_clearance": -0.1,
            "unbounded_clearance": False,
            "control_jerk": 0.8,
        },
    ]
    result = summarize(rows)
    assert result["episodes"] == 2
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["collision_rate"] == pytest.approx(0.5)
    assert result["mean_cross_track_rmse"] == pytest.approx(0.2)
    assert result["mean_path_completion_ratio"] == pytest.approx(0.75)
    assert result["mean_raw_path_completion_ratio"] == pytest.approx(1.0)
    assert result["mean_minimum_clearance"] == pytest.approx(0.2)


def test_direct_actor_path_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        summarize([])
