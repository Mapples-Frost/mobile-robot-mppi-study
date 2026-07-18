import pytest

from mobile_robot_mppi.evaluation.paired_checkpoint import (
    gate2_closed_loop_decision,
    paired_checkpoint_effects,
)


def row(seed, scene, goal, jerk, success=True, collision=False):
    return {
        "method": "simple_combination",
        "scene": scene,
        "physics_domain": "seen",
        "seed": seed,
        "success": success,
        "collision": collision,
        "final_goal_distance": goal,
        "control_jerk": jerk,
        "minimum_clearance": 0.3,
        "stuck_steps": 1,
        "spin_steps": 0,
        "planner_compute_ms_mean": 5.0,
    }


def test_paired_effects_resample_seed_clusters_and_use_favorable_sign():
    control = [
        row(1, "a", 1.0, 0.4),
        row(1, "b", 2.0, 0.6),
        row(2, "a", 1.0, 0.4),
        row(2, "b", 2.0, 0.6),
    ]
    aligned = [
        row(1, "a", 0.8, 0.3),
        row(1, "b", 1.8, 0.5),
        row(2, "a", 0.8, 0.3),
        row(2, "b", 1.8, 0.5),
    ]
    result = paired_checkpoint_effects(
        control, aligned, bootstrap_samples=100, seed=7
    )

    assert result["paired_cells"] == 4
    assert result["independent_clusters"] == 2
    assert result["metrics"]["final_goal_distance"][
        "favorable_effect"
    ] == pytest.approx(0.2)
    assert result["metrics"]["control_jerk"][
        "favorable_effect"
    ] == pytest.approx(0.1)
    assert gate2_closed_loop_decision(result)[
        "closed_loop_development_passed"
    ]


def test_paired_effects_reject_mismatched_cells():
    control = [row(1, "a", 1.0, 0.4), row(2, "a", 1.0, 0.4)]
    aligned = [row(1, "a", 0.9, 0.3), row(2, "b", 0.9, 0.3)]

    with pytest.raises(ValueError, match="paired cells differ"):
        paired_checkpoint_effects(control, aligned)


def test_gate_rejects_primary_tradeoff():
    control = [row(1, "a", 1.0, 0.4), row(2, "a", 1.0, 0.4)]
    aligned = [row(1, "a", 0.8, 0.5), row(2, "a", 0.8, 0.5)]
    result = paired_checkpoint_effects(
        control, aligned, bootstrap_samples=0
    )

    decision = gate2_closed_loop_decision(result)
    assert decision["positive_primary_outcome"]
    assert not decision["control_jerk_not_worse"]
    assert not decision["closed_loop_development_passed"]
