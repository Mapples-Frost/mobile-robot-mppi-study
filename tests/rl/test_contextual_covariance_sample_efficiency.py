from experiments.rl.run_contextual_covariance_sample_efficiency import (
    _add_gate,
    _paired_comparison,
)


def _row(scene, seed, samples, condition, rmse, elapsed, compute):
    return {
        "scene": scene,
        "physics_domain": "domain",
        "seed": seed,
        "num_samples": samples,
        "condition": condition,
        "cross_track_rmse": rmse,
        "elapsed_s": elapsed,
        "control_jerk": 0.2,
        "planner_compute_ms_mean": compute,
        "success": True,
        "collision": False,
    }


def test_half_budget_efficiency_gate_passes_noninferior_tracking():
    rows = []
    for scene in ("a", "b", "c", "d"):
        for seed in range(5):
            rows.append(_row(
                scene, seed, 50, "learned_contextual_bandit", 0.010, 9.0, 4.0
            ))
            rows.append(_row(
                scene, seed, 100, "strongest_global_fixed", 0.011, 10.0, 7.0
            ))
    result = _paired_comparison(rows, 50, 100, 7)
    gated = _add_gate(result, 0.002)
    assert gated["precision_gate_passed"]
    assert gated["elapsed_gate_passed"]
    assert gated["compute_gate_passed"]
    assert gated["primary_gate_passed"]
