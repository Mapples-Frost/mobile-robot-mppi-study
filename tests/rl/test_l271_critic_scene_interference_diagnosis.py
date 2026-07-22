import numpy as np
import torch

from experiments.rl.run_l271_critic_scene_interference_diagnosis import (
    _cosine_summary,
    _gate_for_seed,
    _selected_indices,
)


def test_selected_indices_are_balanced_unique_and_deterministic():
    groups = np.repeat(np.arange(6), 20)
    first = _selected_indices(groups, scene_count=6, rows_per_scene=8, seed=7)
    second = _selected_indices(groups, scene_count=6, rows_per_scene=8, seed=7)
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 48
    assert np.bincount(groups[first], minlength=6).tolist() == [8] * 6


def test_cosine_summary_detects_exact_cancellation():
    vectors = [
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]),
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, -1.0]),
        torch.tensor([1.0, 1.0]), torch.tensor([-1.0, -1.0]),
    ]
    pairs, summary = _cosine_summary(vectors)
    assert len(pairs) == 15
    assert summary["cancellation_ratio"] == 1.0
    assert summary["negative_pair_fraction"] > 0.0


def test_gate_requires_both_null_extremes_and_effect_sizes():
    real = {
        "input_action_columns": {
            "median_off_diagonal_cosine": -0.20,
            "gradient_norm_ratio": 2.0,
        },
        "all_parameters": {
            "cancellation_ratio": 0.70,
            "gradient_norm_ratio": 2.0,
        },
    }
    null = [{
        "input_action_columns": {"median_off_diagonal_cosine": 0.10},
        "all_parameters": {"cancellation_ratio": 0.30},
    } for _ in range(24)]
    result = _gate_for_seed(real, null, {
        "null_lower_quantile": 0.05,
        "null_upper_quantile": 0.95,
        "minimum_action_input_median_cosine_effect": 0.05,
        "minimum_all_parameter_cancellation_effect": 0.05,
        "maximum_gradient_norm_ratio": 20.0,
    })
    assert result["seed_block_pass"] is True
    assert all(result["checks"].values())
