from experiments.rl.run_contextual_covariance_jerk_confirmation import (
    _complete_package_gate,
    _half_budget_gate,
    _smoothing_gate,
)


def _contrast():
    return {
        "success_delta_mean": 0.0,
        "collision_delta_mean": 0.0,
        "cross_track_rmse_delta_ci95": [-0.001, 0.001],
        "elapsed_s_delta_ci95": [-2.0, -0.2],
        "planner_compute_ms_mean_delta_ci95": [-15.0, -5.0],
        "control_jerk_delta_ci95": [-0.02, -0.005],
        "applied_control_jerk_delta_ci95": [-0.01, -0.002],
    }


def test_all_l97_gates_require_their_frozen_clauses():
    assert _half_budget_gate(_contrast(), 0.002)["gate_passed"]
    assert _smoothing_gate(_contrast(), 0.002, 0.5)["gate_passed"]
    assert _complete_package_gate(_contrast(), 0.002)["gate_passed"]
    failed = _contrast()
    failed["applied_control_jerk_delta_ci95"] = [-0.01, 0.001]
    assert not _smoothing_gate(failed, 0.002, 0.5)["gate_passed"]
    assert not _complete_package_gate(failed, 0.002)["gate_passed"]
