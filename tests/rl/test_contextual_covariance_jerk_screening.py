from collections import Counter

from experiments.rl.run_contextual_covariance_jerk_screening import (
    _add_screen_gate,
    _audit_rows,
    _blocked_schedule,
    _select_candidate,
)


def test_blocked_schedule_balances_every_arm_and_run_position():
    spec = {
        "schedule_seed": 9,
        "scenes": ["a", "b"],
        "physics_domains": [{"name": "d1"}, {"name": "d2"}],
        "seeds": [1, 2],
        "arms": [{"name": name} for name in ("a", "b", "c", "d")],
    }
    schedule = _blocked_schedule(spec)
    assert len(schedule) == 32
    for block in range(8):
        rows = [row for row in schedule if row["block_index"] == block]
        assert {row["arm"]["name"] for row in rows} == {"a", "b", "c", "d"}
    counts = Counter(
        (row["arm"]["name"], row["run_position"]) for row in schedule
    )
    assert set(counts.values()) == {2}


def _contrast(name, issued=-0.01, applied=-0.01, elapsed=0.1):
    return {
        "treatment": name,
        "success_delta_mean": 0.0,
        "collision_delta_mean": 0.0,
        "cross_track_rmse_delta_ci95": [-0.001, 0.001],
        "elapsed_s_delta_mean": 0.0,
        "elapsed_s_delta_ci95": [-0.2, elapsed],
        "control_jerk_delta_mean": issued,
        "control_jerk_delta_ci95": [issued - 0.002, issued + 0.002],
        "applied_control_jerk_delta_mean": applied,
        "applied_control_jerk_delta_ci95": [applied - 0.002, applied + 0.002],
    }


def test_screen_gate_requires_both_jerk_metrics_and_selects_applied_minimum():
    good = _add_screen_gate(_contrast("good", -0.010, -0.008), 0.002, 0.5)
    better = _add_screen_gate(_contrast("better", -0.009, -0.012), 0.002, 0.5)
    failed = _add_screen_gate(_contrast("failed", -0.010, 0.001), 0.002, 0.5)
    assert good["candidate_gate_passed"]
    assert better["candidate_gate_passed"]
    assert not failed["candidate_gate_passed"]
    assert _select_candidate({
        "good": good, "better": better, "failed": failed
    }) == "better"


def test_audit_rejects_duplicate_or_incomplete_blocks():
    spec = {
        "scenes": ["scene.yaml"],
        "physics_domains": [{"name": "domain"}],
        "seeds": [1],
        "arms": [{"name": "a"}, {"name": "b"}],
    }
    rows = []
    for position, condition in enumerate(("a", "b")):
        row = {
            "scene": "scene", "physics_domain": "domain", "seed": 1,
            "condition": condition, "run_position": position,
            "success": True, "collision": False,
        }
        row.update({metric: 0.1 for metric in (
            "cross_track_rmse", "elapsed_s", "control_jerk",
            "applied_control_jerk", "planner_compute_ms_mean",
        )})
        rows.append(row)
    assert _audit_rows(rows, spec)["passed"]
    assert not _audit_rows(rows + [dict(rows[0])], spec)["passed"]
