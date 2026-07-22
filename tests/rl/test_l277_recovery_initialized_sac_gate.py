import pytest

from experiments.rl.run_l277_recovery_initialized_sac_probe import (
    _gate,
    _paired_metrics,
)


def _validation_rows(final_completion=0.5, final_cte=0.4, final_goal=2.0):
    rows = []
    for step, completion, cte, goal in (
        (0, 0.4, 0.6, 2.2),
        (3000, 0.42, 0.55, 2.1),
        (6000, final_completion, final_cte, final_goal),
    ):
        for scene in ("a", "b", "c"):
            rows.append({
                "global_step": str(step),
                "scene": scene,
                "success": "False",
                "collision": "False",
                "path_completion_ratio": str(completion),
                "cross_track_rmse": str(cte),
                "goal_distance": str(goal),
                "return": "1.0",
            })
    return rows


def test_l277_paired_metrics_require_complete_fixed_grid():
    metrics = _paired_metrics(_validation_rows())
    assert metrics["completion_change"] == pytest.approx(0.1)
    assert metrics["cte_improvement"] == pytest.approx(0.2)
    assert metrics["goal_distance_improvement"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="grid"):
        _paired_metrics(_validation_rows()[:-1])


def test_l277_gate_requires_engineering_and_directional_coverage():
    paired = _paired_metrics(_validation_rows())
    runs = [
        {"metrics": paired, "engineering": {"complete": True}}
        for _ in range(3)
    ]
    gate = {
        "maximum_per_seed_mean_completion_regression": 0.02,
        "minimum_median_completion_change": 0.0,
        "minimum_median_cte_improvement": 0.05,
        "minimum_median_goal_distance_improvement": 0.10,
        "minimum_seeds_with_cte_or_goal_improvement": 2,
        "minimum_scenes_with_cte_improvement": 2,
        "maximum_collision_increase_per_seed": 0,
        "maximum_success_loss_per_seed": 0,
    }
    checks, _ = _gate(runs, {"a": 0.1, "b": 0.1, "c": -0.1}, gate)
    assert all(checks.values())
    checks, _ = _gate(runs, {"a": 0.1, "b": -0.1, "c": -0.1}, gate)
    assert checks["scene_cte_coverage"] is False

