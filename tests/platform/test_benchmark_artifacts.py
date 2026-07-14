import json

import pytest

from mobile_robot_mppi.evaluation.benchmark import (
    aggregate_episode_summaries,
    write_benchmark_artifacts,
)


def test_multi_seed_aggregation_reports_rates_and_statistics(tmp_path):
    rows = [
        {"seed": 1, "success": True, "collision": False, "final_goal_distance": 0.1,
         "termination_reason": "goal_reached"},
        {"seed": 2, "success": False, "collision": True, "final_goal_distance": 1.1,
         "termination_reason": "collision"},
    ]
    aggregate = aggregate_episode_summaries(rows)
    assert aggregate["num_runs"] == 2
    assert aggregate["success_rate"] == pytest.approx(0.5)
    assert aggregate["collision_rate"] == pytest.approx(0.5)
    assert aggregate["final_goal_distance"]["mean"] == pytest.approx(0.6)
    assert aggregate["termination_counts"] == {"collision": 1, "goal_reached": 1}
    write_benchmark_artifacts(tmp_path, rows, {"kind": "smoke"})
    assert (tmp_path / "runs.csv").exists()
    payload = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert payload["metadata"]["kind"] == "smoke"


def test_empty_benchmark_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_episode_summaries([])
