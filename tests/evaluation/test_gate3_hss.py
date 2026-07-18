from mobile_robot_mppi.evaluation.gate3_hss import (
    gate3_development_decision,
    paired_hss_schedule,
)


def _metric(control, adaptive, favorable):
    return {
        "control_mean": control,
        "aligned_mean": adaptive,
        "favorable_effect": favorable,
    }


def test_hss_schedule_pairs_both_arms_per_block():
    schedule = paired_hss_schedule(
        (32, 33),
        ({"name": "domain"},),
        ({"name": "scene"},),
        7,
    )
    assert len(schedule) == 4
    for seed in (32, 33):
        assert {
            row["arm"] for row in schedule if row["seed"] == seed
        } == {"fixed", "adaptive"}


def test_gate3_development_allows_frozen_engineering_margins():
    result = {
        "metrics": {
            "success": _metric(0.5, 0.5, 0.0),
            "collision": _metric(0.0, 0.0, 0.0),
            "final_goal_distance": _metric(1.0, 0.9, 0.1),
            "control_jerk": _metric(0.2, 0.201, -0.001),
            "stuck_steps": _metric(20.0, 18.0, 2.0),
            "planner_compute_ms_mean": _metric(10.0, 10.5, -0.5),
            "paper_total_rollouts_mean": _metric(200.0, 200.0, 0.0),
        }
    }
    rows = [{
        "reliability_low_fraction": 0.4,
        "reliability_medium_fraction": 0.4,
        "reliability_high_fraction": 0.2,
    }]

    decision = gate3_development_decision(result, rows)

    assert decision["closed_loop_development_passed"]
    assert decision["low_and_nonlow_authority_exercised"]

