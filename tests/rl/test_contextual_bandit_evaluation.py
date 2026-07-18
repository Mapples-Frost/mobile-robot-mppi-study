from experiments.rl.run_contextual_covariance_bandit_evaluation import _paired_gate


def _row(condition, seed, rmse, time_s, jerk=0.2):
    return {
        "scene": "heldout",
        "physics_domain": "domain",
        "condition": condition,
        "seed": seed,
        "cross_track_rmse": rmse,
        "time_to_goal_s": time_s,
        "control_jerk": jerk,
        "planner_compute_ms_mean": 4.0,
        "success": True,
        "collision": False,
    }


def test_paired_gate_requires_precision_time_and_safety():
    rows = []
    for seed in range(5):
        rows.append(_row("learned_contextual_bandit", seed, 0.04, 8.0, 0.1))
        rows.append(_row("strongest_global_fixed", seed, 0.04, 10.0, 0.2))
    result = _paired_gate(rows, margin=0.002, bootstrap_seed=17)
    assert result["precision_gate_passed"]
    assert result["time_gate_passed"]
    assert result["safety_gate_passed"]
    assert result["primary_gate_passed"]
    assert result["time_to_goal_s_delta_mean"] == -2.0


def test_paired_gate_rejects_a_faster_but_imprecise_policy():
    rows = []
    for seed in range(5):
        rows.append(_row("learned_contextual_bandit", seed, 0.045, 8.0))
        rows.append(_row("strongest_global_fixed", seed, 0.04, 10.0))
    result = _paired_gate(rows, margin=0.002, bootstrap_seed=18)
    assert not result["precision_gate_passed"]
    assert result["time_gate_passed"]
    assert not result["primary_gate_passed"]
