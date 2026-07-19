import copy

import pytest

from experiments.rl.analyze_sealed_reliability_confirmation import analyze


def _rows(full=0.90, simple=0.95):
    rows = []
    for scene in ("path_a", "path_b"):
        for domain in ("nominal_seen", "delay_seen", "combined_unseen"):
            for seed in (1, 2):
                for arm, rmse in (
                    ("ordinary_fixed", 1.0),
                    ("value_fixed", simple),
                    ("ordinary_adaptive", 0.96),
                    ("full_proposed", full),
                ):
                    rows.append({
                        "scene": scene,
                        "physics_domain": domain,
                        "seed": seed,
                        "factorial_arm": arm,
                        "success": "True",
                        "collision": "False",
                        "cross_track_rmse": rmse,
                        "control_jerk": 1.0,
                        "rollout_budget_per_decision": 100,
                        "paper_iterations": 2,
                    })
    return rows


def _analyze(rows):
    return analyze(
        rows,
        (1, 2),
        ("path_a", "path_b"),
        ("nominal_seen", "delay_seen", "combined_unseen"),
    )


def test_sealed_confirmation_gate_passes_complete_positive_design():
    result = _analyze(_rows())
    assert result["episode_count"] == 48
    assert result["gate_passed"] is True
    assert result["comparisons"]["stable_cells"] == 6
    effect = result["paired_seed_effects"]["full_vs_ordinary"]
    assert effect["clusters"] == 2
    assert effect["mean_difference"] == pytest.approx(-0.1)
    assert effect["bootstrap_ci95"][1] < 0.0


def test_sealed_confirmation_rejects_simple_combination_regression():
    result = _analyze(_rows(full=0.97, simple=0.95))
    assert result["checks"]["pooled_full_beats_ordinary"] is True
    assert (
        result["checks"]["pooled_full_beats_simple_combination"]
        is False
    )
    assert result["gate_passed"] is False


def test_sealed_confirmation_rejects_incomplete_or_duplicate_blocks():
    rows = _rows()
    with pytest.raises(ValueError, match="expected 48 episodes"):
        _analyze(rows[:-1])
    duplicated = copy.deepcopy(rows)
    duplicated[-1] = copy.deepcopy(duplicated[0])
    with pytest.raises(ValueError, match="duplicate"):
        _analyze(duplicated)
