import pytest

from experiments.dynamic_uncertainty.analyze_rl_hss_stage4 import (
    summarize_pairs,
)


def _pair(seed, *, collision, distance_delta, compute_delta):
    return {
        "dynamics": "nominal",
        "model_block": -1,
        "episode_seed": seed,
        "rl_off_success": True,
        "rl_on_success": False,
        "success_delta": -1,
        "rl_off_collision": False,
        "rl_on_collision": collision,
        "collision_delta": int(collision),
        "final_goal_distance_delta_m": distance_delta,
        "completion_delta": -distance_delta / 10.0,
        "minimum_clearance_delta_m": -0.1 if collision else 0.0,
        "planner_p95_delta_ms": compute_delta,
    }


def test_pair_summary_keeps_episode_units_and_raw_values():
    records = [
        _pair(
            730100006,
            collision=False,
            distance_delta=0.4,
            compute_delta=50.0,
        ),
        _pair(
            730100008,
            collision=True,
            distance_delta=0.8,
            compute_delta=70.0,
        ),
    ]

    summary = summarize_pairs(records)

    assert summary["pair_count"] == 2
    assert summary["obstacle_seed_count"] == 2
    assert summary["rl_off_success_count"] == 2
    assert summary["rl_on_success_count"] == 0
    assert summary["rl_on_collision_count"] == 1
    assert summary["success_delta"]["raw"] == [-1.0, -1.0]
    assert summary["final_goal_distance_delta_m"]["raw"] == [0.4, 0.8]
    assert summary["final_goal_distance_delta_m"]["median"] == pytest.approx(
        0.6
    )
    assert summary["planner_p95_delta_ms"]["median"] == pytest.approx(60.0)


def test_empty_pair_summary_is_explicit():
    assert summarize_pairs([]) == {"pair_count": 0}
