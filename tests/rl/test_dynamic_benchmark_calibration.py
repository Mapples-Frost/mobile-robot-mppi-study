from pathlib import Path

from experiments.rl.run_dynamic_benchmark_calibration import (
    _calibration_design,
    _condition_config,
)
from experiments.rl.summarize_dynamic_benchmark_calibration import (
    _candidate_rows,
    _select_candidates,
)
from experiments.rl.evaluate_rl_sampling_prior import (
    _apply_physics_domain,
    _apply_scene_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "rl" / "dynamic_benchmark_calibration_l36.yaml"


def test_l36_calibration_never_loads_rl_memory_or_residual_model():
    config, design = _calibration_design(CONFIG)
    base = _apply_scene_config(config, ROOT / design["base_scene"])
    base = _apply_physics_domain(
        base, ROOT / design["physics_domain_config"], design["physics_domain"]
    )
    condition = _condition_config(
        base, design, design["candidates"][0], design["episode_seeds"][0]
    )
    assert condition["planner"]["sampling_prior"] == "goal_warm_start"
    assert condition["planner"]["prediction_mode"] == "nominal"
    assert "checkpoint" not in condition["planner"]
    assert condition["rl"]["enabled"] is False
    assert condition["memory"]["enable"] is False
    assert condition["perception"]["temporal_scan_guard"]["safety_enabled"] is True


def test_l36_selection_is_deterministic_and_requires_mixed_outcomes():
    episodes = []
    for candidate, successes, collisions in (
        ("a", 6, 2), ("b", 4, 4), ("c", 2, 6), ("d", 8, 0)
    ):
        outcomes = (["success"] * successes) + (["collision"] * collisions)
        for index, outcome in enumerate(outcomes):
            episodes.append({
                "candidate": candidate,
                "success": outcome == "success",
                "collision": outcome == "collision",
                "final_goal_distance": float(index),
            })
    rows = _candidate_rows(episodes, ["a", "b", "c", "d"])
    assert [row["candidate"] for row in rows if row["eligible"]] == ["a", "b", "c"]
    selected = _select_candidates(rows, [0.25, 0.50, 0.75])
    assert [row["candidate"] for row in selected] == ["a", "b", "c"]

