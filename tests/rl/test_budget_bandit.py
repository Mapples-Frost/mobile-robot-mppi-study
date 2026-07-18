import numpy as np

from mobile_robot_mppi.rl.budget_bandit import PrimalDualBudgetBandit
from experiments.rl.evaluate_anytime_budget_bandit import (
    _matched_budget_randomization_test,
    _paired_policy_contrast,
    evaluate,
)


def _bandit(**overrides):
    values = {
        "ridge": 1.0,
        "exploration_alpha": 0.0,
        "epsilon": 0.0,
        "target_add_fraction": 0.5,
        "dual_learning_rate": 0.1,
        "maximum_dual_price": 1.0,
        "seed": 3,
    }
    values.update(overrides)
    return PrimalDualBudgetBandit(
        ("x",), np.asarray((0.0,)), np.asarray((1.0,)), **values
    )


def test_positive_observed_advantage_learns_to_add_for_matching_context():
    bandit = _bandit(target_add_fraction=0.9, dual_learning_rate=0.001)
    features = {"x": 1.0}
    for _ in range(40):
        bandit.update(features, True, 0.2)
    decision = bandit.decide(features)
    assert decision.add_samples
    assert decision.predicted_advantage > decision.dual_price


def test_dual_price_increases_above_budget_and_decreases_below_budget():
    bandit = _bandit()
    for _ in range(5):
        bandit.update({"x": 0.0}, True, 0.1)
    high = bandit.dual_price
    assert high > 0.0
    for _ in range(5):
        bandit.update({"x": 0.0}, False)
    assert bandit.dual_price < high


def test_checkpoint_roundtrip_preserves_deterministic_decision():
    bandit = _bandit()
    bandit.update({"x": 2.0}, True, 0.3)
    restored = PrimalDualBudgetBandit.from_state_dict(bandit.state_dict())
    first = bandit.decide({"x": 2.0})
    second = restored.decide({"x": 2.0})
    assert first.add_samples == second.add_samples
    assert first.predicted_advantage == second.predicted_advantage
    assert first.dual_price == second.dual_price


def test_policy_contrast_uses_episode_means_not_anchor_pseudoreplicates():
    rows = []
    for seed in (1, 2):
        for gain in (0.1, 0.0):
            rows.append({
                "scene": "route", "physics_domain": "plant", "seed": seed,
                "true_cost50": 100.0,
                "true_cost100": 100.0 * (1.0 - gain),
            })
    result = _paired_policy_contrast(
        rows,
        np.asarray((True, False, True, False)),
        np.zeros(4, dtype=bool),
        7,
    )
    assert result["paired_episodes"] == 2
    assert result["true_cost_delta_mean"] == -5.0


def test_deployment_dual_is_calibrated_only_from_discovery_predictions():
    rows = []
    for split, seeds in (("discovery", (1, 2)), ("evaluation", (3, 4))):
        for seed in seeds:
            for anchor, feature in enumerate((-1.0, 1.0)):
                row = {
                    "split": split,
                    "scene": "route",
                    "physics_domain": "plant",
                    "seed": seed,
                    "anchor_index": anchor,
                    "true_cost50": 10.0,
                    "true_cost100": 9.0 if feature > 0.0 else 10.5,
                    "relative_true_cost_gain": 0.1 if feature > 0.0 else -0.05,
                    "collision100": 0,
                    "x": feature,
                }
                rows.append(row)
    config = {
        "ridge": 1.0,
        "exploration_alpha": 1.0,
        "epsilon": 0.0,
        "target_add_fraction": 0.5,
        "dual_learning_rate": 0.01,
        "maximum_dual_price": 1.0,
        "training_seed": 3,
        "bootstrap_seed": 4,
        "random_seed": 5,
        "base_budget": 50,
        "maximum_budget": 100,
        "deployment_target_add_fraction": 0.5,
        "minimum_relative_true_cost_gain": 0.01,
        "primary_gate": {
            "minimum_add_fraction": 0.0,
            "maximum_add_fraction": 1.0,
            "maximum_mean_budget": 100.0,
            "minimum_oracle_gain_fraction": 0.0,
        },
    }
    import experiments.rl.evaluate_anytime_budget_bandit as module
    original = module.DIAGNOSTIC_FEATURES
    module.DIAGNOSTIC_FEATURES = ("x",)
    try:
        result = evaluate(rows, config)
    finally:
        module.DIAGNOSTIC_FEATURES = original
    assert result["deployment_target_add_fraction"] == 0.5
    assert result["discovery_deployment_add_fraction"] == 0.5
    assert result["training_final_dual_price"] != result["deployment_dual_price"]


def test_randomization_preserves_stratum_budget_and_detects_context_signal():
    rows = []
    actions = []
    for seed in range(8):
        for anchor in range(4):
            beneficial = anchor == 0
            rows.append({
                "scene": "route",
                "physics_domain": "plant",
                "seed": seed,
                "true_cost50": 10.0,
                "true_cost100": 5.0 if beneficial else 11.0,
            })
            actions.append(beneficial)
    result = _matched_budget_randomization_test(
        rows, np.asarray(actions), 1000, 19
    )
    assert result["observed"] == -1.25
    assert result["null_mean"] > result["observed"]
    assert result["one_sided_p_value"] < 0.01
