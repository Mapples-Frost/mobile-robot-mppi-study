from experiments.rl.run_l278_recovery_retention_diagnosis import _gate


def test_l278_gate_passes_only_with_retention_and_scene_coverage():
    seeds = [
        {
            "seed": seed,
            "relative_test_rmse_increase_6k": 0.1,
            "median_test_return_loss_6k": 0.2,
            "test_reentry_loss_6k": 0.0,
        }
        for seed in (1, 2, 3)
    ]
    rollouts = []
    for stage in ("initial", "step6000"):
        for scene in ("a", "b", "c", "d", "e", "f"):
            rollouts.append({
                "stage": stage,
                "split": "test",
                "scene": scene,
                "return_change_vs_initial": 0.0 if stage == "initial" else 1.0,
                "collision": False,
                "boundary_violation": False,
            })
    gate = {
        "maximum_median_relative_test_rmse_increase": 0.20,
        "maximum_median_test_return_loss": 0.50,
        "maximum_median_test_reentry_loss": 0.10,
        "minimum_scenes_with_nonnegative_test_return_change": 4,
        "maximum_collision_boundary_failure_increase": 0,
    }
    checks, _ = _gate(seeds, rollouts, gate)
    assert all(checks.values())
    seeds[0]["relative_test_rmse_increase_6k"] = 1.0
    seeds[1]["relative_test_rmse_increase_6k"] = 1.0
    checks, _ = _gate(seeds, rollouts, gate)
    assert checks["test_rmse_retained"] is False

