from experiments.rl.analyze_gate3_hss import _confirmation_gate


def _metric(control, adaptive, effect, interval):
    return {
        "control_mean": control,
        "aligned_mean": adaptive,
        "favorable_effect": effect,
        "ci95": interval,
    }


def test_confirmation_gate_requires_favorable_primary_interval():
    comparison = {
        "metrics": {
            "success": _metric(0.0, 0.0, 0.0, [0.0, 0.0]),
            "collision": _metric(0.0, 0.0, 0.0, [0.0, 0.0]),
            "final_goal_distance": _metric(
                2.0, 1.8, 0.2, [0.05, 0.35]
            ),
            "stuck_steps": _metric(5.0, 4.0, 1.0, [-1.0, 3.0]),
            "control_jerk": _metric(
                0.10, 0.1005, -0.0005, [-0.002, 0.001]
            ),
        }
    }

    result = _confirmation_gate(comparison)

    assert result["favorable_primary_bootstrap_interval"]
    assert result["favorable_primary_intervals"][
        "final_goal_distance"
    ]
    assert result["jerk_within_one_percent_margin"]

