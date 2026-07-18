import copy

from experiments.rl.summarize_scene_complexity_gate_ablation import (
    _development_gate,
    _pair_rows,
)


DESIGN = {
    "activation_epsilon": 0.05,
    "maximum_simple_mean_gate_alpha": 0.15,
    "maximum_simple_active_fraction": 0.25,
    "minimum_complex_mean_gate_alpha": 0.10,
    "minimum_complex_active_fraction": 0.15,
    "maximum_complex_active_fraction": 0.85,
    "minimum_alpha_reduction_vs_always": 0.50,
    "maximum_simple_success_losses_vs_traditional": 0,
    "maximum_collision_regressions_vs_traditional": 0,
    "minimum_complex_net_success_gain_vs_traditional": 1,
    "minimum_complex_mean_distance_improvement_m": 0.01,
}


def _episodes():
    rows = []
    for role, scene in (("simple", "clean"), ("complex", "trap")):
        for seed in (1, 2):
            for condition in (
                "traditional_mppi",
                "frozen_bc_prior",
                "lcb_always",
                "complexity_lcb",
            ):
                success = role == "simple"
                distance = 0.20 if success else 0.80
                if role == "complex" and condition == "complexity_lcb":
                    success = True
                    distance = 0.20
                rows.append({
                    "training_seed": 11,
                    "scene": scene,
                    "scene_role": role,
                    "episode_seed": seed,
                    "condition": condition,
                    "success": success,
                    "collision": False,
                    "final_goal_distance": distance,
                })
    return rows


def _steps():
    rows = []
    for role, scene in (("simple", "clean"), ("complex", "trap")):
        for seed in (1, 2):
            for condition in ("lcb_always", "complexity_lcb"):
                for step in range(10):
                    alpha = 1.0 if condition == "lcb_always" else (
                        0.0
                        if role == "simple" or step >= 8
                        else 0.50
                    )
                    rows.append({
                        "training_seed": 11,
                        "scene": scene,
                        "scene_role": role,
                        "episode_seed": seed,
                        "condition": condition,
                        "step": step,
                        "rl_gate_alpha": alpha,
                        "rl_scene_complexity_score": alpha,
                    })
    return rows


def test_pair_rows_uses_traditional_and_bc_concurrent_references():
    paired = _pair_rows(_episodes())
    candidate = [
        row for row in paired
        if row["scene"] == "trap"
        and row["condition"] == "complexity_lcb"
    ]
    assert len(candidate) == 2
    assert all(row["traditional_success_gained"] for row in candidate)
    assert all(
        abs(row["traditional_distance_improvement"] - 0.6) < 1e-12
        for row in candidate
    )


def test_preregistered_development_gate_passes_only_complete_pattern():
    result = _development_gate(_episodes(), _steps(), DESIGN)
    assert result["passed"] is True
    assert result["simple_gate"]["mean_alpha"] == 0.0
    assert abs(result["complex_gate"]["mean_alpha"] - 0.4) < 1e-12
    assert abs(result["alpha_reduction_vs_always"] - 0.8) < 1e-12
    assert result["complex_net_success_gain_vs_traditional"] == 2

    failed_steps = copy.deepcopy(_steps())
    for row in failed_steps:
        if row["scene_role"] == "simple" and row["condition"] == "complexity_lcb":
            row["rl_gate_alpha"] = 1.0
    failed = _development_gate(_episodes(), failed_steps, DESIGN)
    assert failed["passed"] is False
    assert failed["checks"]["simple_mean_alpha"] is False
    assert "keep_test_sealed" in failed["decision"]
