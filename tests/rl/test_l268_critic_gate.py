from experiments.rl.evaluate_l268_critic_gate import _gate_from_rows, _number


THRESHOLDS = {
    "maximum_absolute_mean_q": 500.0,
    "maximum_mean_quantile_spread": 100.0,
    "minimum_aggregate_in_support_spearman": 0.05,
    "minimum_paired_median_spearman_improvement": 0.20,
    "minimum_improving_seed_blocks": 2,
    "minimum_recovery_forward_accuracy": 0.60,
    "minimum_pair_accuracy_improvement": 0.15,
    "maximum_other_set_pair_accuracy_decrease": 0.05,
    "minimum_top3_improvement": 0.10,
    "minimum_scene_spearman_change": -0.10,
}


def _rows(treatment_spearman=0.20, treatment_pair=0.70):
    rows = []
    for seed in (1, 2, 3):
        for arm in ("control", "recovery_balanced"):
            treatment = arm == "recovery_balanced"
            rows.append({
                "seed": seed,
                "arm": arm,
                "in_support_spearman": treatment_spearman if treatment else -0.10,
                "in_support_top3": 0.90 if treatment else 0.70,
                "in_support_pair_accuracy": treatment_pair if treatment else 0.40,
                "heldout_pair_accuracy": 0.70 if treatment else 0.60,
                "finite": True,
                "actor_alpha_unchanged": True,
                "maximum_absolute_q": 10.0,
                "mean_quantile_spread": 2.0,
            })
    return rows


def test_l268_gate_passes_only_joint_frozen_checks():
    checks, _ = _gate_from_rows(
        _rows(), THRESHOLDS, {"scene_a": 0.0, "scene_b": -0.05}
    )
    assert all(checks.values())


def test_l268_gate_fails_without_critic_ranking_repair():
    checks, _ = _gate_from_rows(
        _rows(treatment_spearman=-0.05, treatment_pair=0.45),
        THRESHOLDS,
        {"scene_a": -0.20},
    )
    assert not checks["aggregate_in_support_spearman_positive"]
    assert not checks["recovery_forward_pair_accuracy"]
    assert not checks["no_scene_spearman_collapse"]


def test_l268_gate_treats_nan_as_missing_not_a_numeric_scene_change():
    assert _number("nan") is None
    assert _number("inf") is None
    assert _number("-inf") is None
