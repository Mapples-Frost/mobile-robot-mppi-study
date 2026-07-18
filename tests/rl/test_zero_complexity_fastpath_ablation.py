import copy

from experiments.rl.analyze_zero_complexity_fastpath_results import (
    _analysis,
    _pairs,
)
from experiments.rl.summarize_zero_complexity_fastpath_ablation import (
    _behavior_equivalence,
    _timing,
)


def _episodes():
    rows = []
    for training_seed in (11, 12):
        for scene_role, scene in (("control", "clean"), ("blocking", "trap")):
            for episode_seed in (1, 2):
                for condition in ("standard_complexity", "zero_complexity_fastpath"):
                    standard = condition == "standard_complexity"
                    planner_ms = 10.0 if standard else (8.0 if scene_role == "control" else 9.0)
                    rows.append({
                        "training_seed": training_seed,
                        "scene": scene,
                        "scene_role": scene_role,
                        "episode_seed": episode_seed,
                        "condition": condition,
                        "success": True,
                        "collision": False,
                        "planner_compute_ms_mean": planner_ms,
                    })
    return rows


def _steps():
    rows = []
    for episode in _episodes():
        for step in (0, 1):
            fast = episode["condition"] == "zero_complexity_fastpath"
            if episode["scene_role"] == "control":
                skipped = float(fast)
            else:
                skipped = float(fast and step == 0)
            rows.append({
                "training_seed": episode["training_seed"],
                "scene": episode["scene"],
                "scene_role": episode["scene_role"],
                "episode_seed": episode["episode_seed"],
                "condition": episode["condition"],
                "step": step,
                "executed_v": 0.2,
                "executed_omega": -0.1,
                "goal_distance": 1.0 - 0.1 * step,
                "rl_gate_alpha": 0.0,
                "collision": False,
                "safety_override": False,
                "rl_learned_inference_skipped": skipped,
            })
    return rows


def test_fastpath_summary_requires_exact_paired_behavior():
    result = _behavior_equivalence(_episodes(), _steps())
    assert result["steps_compared"] == 16
    assert result["step_behavior_mismatches"] == 0
    assert result["episode_success_mismatches"] == 0
    assert result["episode_collision_mismatches"] == 0
    assert all(value == 0.0 for value in result["max_abs_differences"].values())

    changed = copy.deepcopy(_steps())
    changed[-1]["executed_omega"] = 0.0
    assert _behavior_equivalence(_episodes(), changed)["step_behavior_mismatches"] == 1


def test_fastpath_timing_separates_control_and_blocking_skip_rates():
    result = _timing(_episodes(), _steps())
    assert abs(result["control"]["time_reduction_fraction"] - 0.2) < 1e-12
    assert abs(result["blocking"]["time_reduction_fraction"] - 0.1) < 1e-12
    assert result["control"]["zero_complexity_fastpath"]["inference_skip_fraction"] == 1.0
    assert result["blocking"]["zero_complexity_fastpath"]["inference_skip_fraction"] == 0.5


def test_vectorized_nested_timing_analysis_preserves_constant_effect():
    import numpy as np

    pairs = _pairs(_episodes())
    result = _analysis(pairs, np.random.RandomState(7), replicates=500)
    assert abs(result["control"]["time_reduction_fraction"] - 0.2) < 1e-12
    assert abs(result["blocking"]["time_reduction_fraction"] - 0.1) < 1e-12
    assert np.allclose(result["control"]["reduction_ci95"], (0.2, 0.2))
    assert np.allclose(result["blocking"]["reduction_ci95"], (0.1, 0.1))
