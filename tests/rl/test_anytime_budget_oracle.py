import numpy as np

from experiments.rl.run_anytime_budget_oracle import (
    DIAGNOSTIC_FEATURES,
    _fit_ridge,
    _oracle_actions,
    _policy_summary,
    _predict_ridge,
    analyze,
)


def _row(seed, gain, feature_value=0.0):
    row = {
        "scene": "route",
        "physics_domain": "plant",
        "seed": seed,
        "relative_true_cost_gain": gain,
        "true_cost50": 100.0,
        "true_cost100": 100.0 * (1.0 - gain),
        "collision100": 0,
    }
    row.update({name: float(feature_value) for name in DIAGNOSTIC_FEATURES})
    return row


def test_oracle_requires_frozen_practical_margin_and_safety():
    rows = [_row(1, 0.009), _row(2, 0.011), _row(3, 0.20)]
    rows[-1]["collision100"] = 1
    np.testing.assert_array_equal(
        _oracle_actions(rows, 0.01), np.asarray((False, True, False))
    )


def test_ridge_uses_only_declared_features_and_predicts_direction():
    rows = [_row(index, gain, value) for index, (gain, value) in enumerate((
        (-0.02, -2.0), (-0.01, -1.0), (0.01, 1.0), (0.02, 2.0)
    ))]
    model = _fit_ridge(rows, penalty=0.01)
    prediction = _predict_ridge(model, rows)
    assert prediction[0] < prediction[-1]
    assert model["coefficients"].shape == (len(DIAGNOSTIC_FEATURES) + 1,)


def test_policy_summary_averages_anchors_inside_episode():
    rows = [_row(1, 0.10), _row(1, -0.10), _row(2, 0.20), _row(2, 0.00)]
    summary = _policy_summary(
        rows, np.asarray((True, False, True, False)), 50, 100, 7
    )
    assert summary["anchors"] == 4
    assert summary["episodes"] == 2
    assert summary["mean_budget"] == 75.0
    assert summary["true_cost_delta_vs_k50_mean"] == -7.5


def test_budget_calibrated_ridge_freezes_discovery_quantile():
    rows = []
    for split, seed_offset in (("discovery", 0), ("evaluation", 10)):
        for index, gain in enumerate((-0.02, -0.01, 0.01, 0.03)):
            row = _row(seed_offset + index, gain, gain)
            row.update({
                "split": split,
                "candidate_prefix_sha256": "same",
                "candidate_full_prefix_sha256": "same",
                "collision50": 0,
            })
            rows.append(row)
    spec = {
        "design_id": "test",
        "minimum_relative_true_cost_gain": 0.01,
        "ridge_penalty": 0.01,
        "ridge_target_add_fraction": 0.50,
        "base_budget": 50,
        "maximum_budget": 100,
        "bootstrap_seed": 3,
        "primary_gate": {
            "minimum_oracle_add_fraction": 0.0,
            "maximum_oracle_add_fraction": 1.0,
            "minimum_oracle_relative_gain": 0.0,
            "maximum_learned_mean_budget": 100.0,
            "minimum_learned_oracle_gain_fraction": 0.0,
        },
    }
    result = analyze(rows, spec)
    assert result["ridge_model"]["target_add_fraction"] == 0.5
    assert result["ridge_model"]["discovery_add_fraction"] == 0.5
