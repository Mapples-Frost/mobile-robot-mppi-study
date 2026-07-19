import numpy as np

from experiments.rl.run_full_proposed_factorial import (
    ARMS,
    factorial_schedule,
    metrics_for_profile,
    path_tracking_metrics,
)
from experiments.rl.analyze_full_proposed_factorial import (
    validate_complete_factorial,
)


def test_factorial_schedule_is_complete_reproducible_and_blocked():
    domains = [{"name": "nominal"}, {"name": "shift"}]
    scenes = [{"name": "clean"}]
    first = factorial_schedule((48, 49), domains, scenes, 20260736)
    second = factorial_schedule((48, 49), domains, scenes, 20260736)

    assert first == second
    assert len(first) == 2 * 2 * 1 * 4
    blocks = {}
    for job in first:
        blocks.setdefault(job["block"], []).append(job)
    assert len(blocks) == 4
    for jobs in blocks.values():
        assert {job["arm"] for job in jobs} == set(ARMS)
        assert sorted(
            job["run_order_within_block"] for job in jobs
        ) == [0, 1, 2, 3]


def test_factorial_analysis_rejects_missing_arm_and_accepts_complete_blocks():
    rows = []
    for seed in (48, 49):
        for arm in ARMS:
            rows.append({
                "seed": seed,
                "scene": "clean",
                "physics_domain": "nominal",
                "factorial_arm": arm,
                "block": "clean::nominal::seed%d" % seed,
            })
    result = validate_complete_factorial(rows)
    assert result["episode_count"] == 8
    assert result["block_count"] == 2
    assert result["seeds"] == [48, 49]

    missing = rows[:-1]
    try:
        validate_complete_factorial(missing)
    except ValueError as exc:
        assert "incomplete factorial blocks" in str(exc)
    else:
        raise AssertionError("missing factorial arm was accepted")


def test_path_tracking_profile_and_saved_trajectory_metrics(tmp_path):
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(
        "x,y,theta\n"
        "0.0,0.0,0.0\n"
        "0.5,0.1,0.0\n"
        "1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    result = path_tracking_metrics(
        trajectory, [[0.0, 0.0], [1.0, 0.0]]
    )
    assert result["path_completion_ratio"] == 1.0
    assert result["tangent_heading_rmse"] == 0.0
    assert np.isclose(
        result["path_cross_track_rmse_recomputed"],
        0.1 / np.sqrt(3.0),
    )
    assert not metrics_for_profile("path_tracking")["cross_track_rmse"]
    assert metrics_for_profile("path_tracking")["path_completion_ratio"]
