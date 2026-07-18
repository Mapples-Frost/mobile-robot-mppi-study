import copy

from experiments.rl.summarize_scene_complexity_sample_efficiency import (
    _clean_exact_fallback,
    _development_gate,
    _pair_rows,
)


DESIGN = {
    "low_budget_sample_counts": [50, 100],
    "comparison_low_sample_count": 100,
    "comparison_high_sample_count": 400,
    "maximum_collision_regressions": 0,
    "minimum_low_budget_blocking_net_success_gain": 2,
    "maximum_low_vs_high_success_rate_shortfall": 0.05,
    "minimum_blocking_scenes_matching_high_budget_traditional": 1,
    "maximum_low_vs_high_planner_time_ratio": 0.60,
}


def _episodes():
    rows = []
    for role, scene in (("control", "clean"), ("blocking", "trap")):
        for seed in (1, 2):
            for num_samples in (50, 100, 200, 400):
                for condition in ("traditional_mppi", "complexity_lcb"):
                    gated = condition == "complexity_lcb"
                    success = role == "control" or gated
                    rows.append({
                        "training_seed": 11,
                        "scene": scene,
                        "scene_role": role,
                        "episode_seed": seed,
                        "num_samples": num_samples,
                        "condition": condition,
                        "success": success,
                        "collision": False,
                        "final_goal_distance": 0.2 if success else 0.8,
                        "planner_compute_ms_mean": num_samples * 0.02 + (0.5 if gated else 0.0),
                    })
    return rows


def _steps():
    rows = []
    for seed in (1, 2):
        for num_samples in (50, 100, 200, 400):
            for condition in ("traditional_mppi", "complexity_lcb"):
                for step in range(3):
                    rows.append({
                        "training_seed": 11,
                        "scene": "clean",
                        "scene_role": "control",
                        "episode_seed": seed,
                        "num_samples": num_samples,
                        "condition": condition,
                        "step": step,
                        "executed_v": 0.1,
                        "executed_omega": 0.0,
                        "goal_distance": 1.0 - 0.1 * step,
                        "collision": False,
                        "safety_override": False,
                        "rl_gate_alpha": 0.0,
                    })
    return rows


def test_pair_rows_is_paired_within_sample_count():
    paired = _pair_rows(_episodes())
    assert len(paired) == 16
    blocking_low = [
        row for row in paired
        if row["scene"] == "trap" and int(row["num_samples"]) == 50
    ]
    assert len(blocking_low) == 2
    assert all(row["traditional_success_gained"] for row in blocking_low)


def test_clean_fallback_requires_stepwise_identity_and_zero_alpha():
    result = _clean_exact_fallback(_steps())
    assert result["exact"] is True
    changed = copy.deepcopy(_steps())
    changed[-1]["executed_omega"] = 0.1
    assert _clean_exact_fallback(changed)["exact"] is False


def test_development_gate_passes_complete_safe_sample_efficiency_pattern():
    episodes = _episodes()
    paired = _pair_rows(episodes)
    audit = {
        "complete_factorial": True,
        "duplicate_episode_keys": 0,
        "sealed_l25_seeds_used": [],
        "sealed_l26_seeds_used": [],
    }
    result = _development_gate(episodes, paired, _steps(), DESIGN, audit)
    assert result["passed"] is True
    assert result["low_budget_blocking_net_success_gain"] == 4
    assert result["blocking_scenes_matching_high_budget_traditional"] == 1

    failed = copy.deepcopy(episodes)
    failed[-1]["collision"] = True
    failed_result = _development_gate(
        failed, _pair_rows(failed), _steps(), DESIGN, audit
    )
    assert failed_result["passed"] is False
    assert failed_result["checks"]["no_collision_regression"] is False
