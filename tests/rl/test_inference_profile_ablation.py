import copy

import numpy as np

from experiments.rl.summarize_inference_profile_ablation import (
    _behavior_equivalence,
    _bool,
    _nested_reduction_interval,
)


CONDITIONS = (
    "reference_full_diagnostics",
    "selected_critic_only",
    "selected_critic_base_reuse",
)


def _episodes():
    rows = []
    for training_seed in (11, 12):
        for scene_role, scene in (("control", "clean"), ("blocking", "trap")):
            for episode_seed in (1, 2):
                for condition in CONDITIONS:
                    rows.append({
                        "training_seed": training_seed,
                        "scene": scene,
                        "scene_role": scene_role,
                        "episode_seed": episode_seed,
                        "condition": condition,
                        "success": True,
                        "collision": False,
                    })
    return rows


def _steps():
    rows = []
    for episode in _episodes():
        for step in (0, 1):
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
                "rl_gate_alpha": 0.5,
                "rl_correction_advantage_gate_alpha": 1.0,
                "rl_selected_consensus_lcb": 0.2,
                "rl_target_q1_base": 1.0,
                "rl_target_q2_base": 2.0,
                "rl_target_q1_candidate": 1.2,
                "rl_target_q2_candidate": 2.1,
                "collision": False,
                "safety_override": False,
            })
    return rows


def test_csv_boolean_parser_accepts_numeric_and_text_encodings():
    assert _bool("1.0") is True
    assert _bool("0.0") is False
    assert _bool("true") is True
    assert _bool(False) is False


def test_l28_behavior_equivalence_is_stepwise_and_gate_sensitive():
    result = _behavior_equivalence(_episodes(), _steps())
    assert all(
        values["step_behavior_mismatches"] == 0
        for values in result.values()
    )
    changed = copy.deepcopy(_steps())
    row = next(
        value for value in changed
        if value["condition"] == "selected_critic_base_reuse"
    )
    row["rl_selected_consensus_lcb"] = 0.3
    changed_result = _behavior_equivalence(_episodes(), changed)
    assert changed_result["selected_critic_base_reuse"][
        "selected_gate_mismatches"
    ] == 1


def test_l28_nested_reduction_bootstrap_respects_constant_paired_effect():
    rows = []
    for training_seed in (11, 12):
        for episode_seed in (1, 2):
            for scene in ("a", "b"):
                rows.append({
                    "training_seed": training_seed,
                    "episode_seed": episode_seed,
                    "scene": scene,
                    "reference": 10.0,
                    "candidate": 8.0,
                })
    interval = _nested_reduction_interval(
        rows, "reference", "candidate"
    )
    assert np.allclose(interval, (0.2, 0.2))
