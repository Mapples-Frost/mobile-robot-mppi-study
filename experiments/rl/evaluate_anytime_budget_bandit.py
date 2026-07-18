#!/usr/bin/env python3
"""Train/evaluate the constrained contextual budget bandit on branch records."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.run_anytime_budget_oracle import (
    DIAGNOSTIC_FEATURES,
    _oracle_actions,
    _policy_episode_differences,
    _policy_summary,
    _read_csv,
)
from experiments.rl.run_covariance_context_oracle import _hierarchical_ci
from mobile_robot_mppi.rl.budget_bandit import PrimalDualBudgetBandit


def _episode_order(rows, seed):
    episodes = defaultdict(list)
    for row in rows:
        key = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        episodes[key].append(row)
    keys = sorted(episodes)
    np.random.RandomState(int(seed)).shuffle(keys)
    output = []
    for key in keys:
        output.extend(sorted(episodes[key], key=lambda row: int(row["anchor_index"])))
    return output


def train_bandit(rows, config):
    bandit = PrimalDualBudgetBandit.from_rows(
        rows,
        DIAGNOSTIC_FEATURES,
        ridge=float(config["ridge"]),
        exploration_alpha=float(config["exploration_alpha"]),
        epsilon=float(config["epsilon"]),
        target_add_fraction=float(config["target_add_fraction"]),
        dual_learning_rate=float(config["dual_learning_rate"]),
        maximum_dual_price=float(config["maximum_dual_price"]),
        seed=int(config["training_seed"]),
    )
    trace = []
    for row in _episode_order(rows, config["training_seed"]):
        decision = bandit.decide(row, explore=True)
        advantage = float(row["relative_true_cost_gain"])
        if int(row["collision100"]):
            advantage = -1.0
        bandit.update(
            row, decision,
            advantage if decision.add_samples else None,
        )
        trace.append({
            "add": decision.add_samples,
            "exploratory": decision.exploratory,
            "predicted_advantage": decision.predicted_advantage,
            "width": decision.confidence_width,
            "dual_price_before_update": decision.dual_price,
            "observed_advantage": advantage if decision.add_samples else None,
        })
    return bandit, trace


def _paired_policy_contrast(rows, first, second, bootstrap_seed):
    episodes = defaultdict(list)
    for row, first_add, second_add in zip(rows, first, second):
        first_cost = float(
            row["true_cost100"] if first_add else row["true_cost50"]
        )
        second_cost = float(
            row["true_cost100"] if second_add else row["true_cost50"]
        )
        key = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        episodes[key].append(first_cost - second_cost)
    grouped = defaultdict(list)
    for (scene, domain, _seed), values in episodes.items():
        grouped[scene + "__" + domain].append(float(np.mean(values)))
    values = [value for group in grouped.values() for value in group]
    return {
        "paired_episodes": len(episodes),
        "true_cost_delta_mean": float(np.mean(values)),
        "true_cost_delta_ci95": _hierarchical_ci(
            dict(grouped), int(bootstrap_seed)
        ),
    }


def _matched_budget_randomization_test(rows, actions, repeats, seed):
    """Test whether learned context allocation beats stratified random ADDs.

    Each random policy has exactly the learned policy's ADD count inside every
    scene/physics stratum.  The statistic remains the equally weighted mean of
    episode-level cost deltas, so eight anchors from one episode never become
    eight independent experimental units.
    """
    actions = np.asarray(actions, dtype=bool)
    repeats = int(repeats)
    if actions.shape != (len(rows),) or repeats <= 0:
        raise ValueError("randomization inputs are invalid")
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(str(row["scene"]), str(row["physics_domain"]))].append(index)

    def statistic(policy):
        values = _policy_episode_differences(rows, policy)
        return float(np.mean(list(values.values())))

    observed = statistic(actions)
    rng = np.random.RandomState(int(seed))
    null = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        random_actions = np.zeros(len(rows), dtype=bool)
        for indices in strata.values():
            indices = np.asarray(indices, dtype=np.int64)
            count = int(np.sum(actions[indices]))
            if count:
                random_actions[rng.choice(indices, count, replace=False)] = True
        null[repeat] = statistic(random_actions)
    # Lower true cost is better. The +1 correction keeps the Monte-Carlo test
    # conservative and prevents an impossible zero p-value.
    p_value = float((1 + np.sum(null <= observed)) / (repeats + 1))
    return {
        "repeats": repeats,
        "stratification": "scene_x_physics_domain_exact_add_count",
        "statistic": "equal_weight_episode_mean_true_cost_delta_vs_k50",
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_ci95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "one_sided_p_value": p_value,
    }


def evaluate(rows, config):
    discovery = [row for row in rows if row["split"] == "discovery"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    bandit, trace = train_bandit(discovery, config)
    training_final_dual_price = float(bandit.dual_price)
    deployment_target = config.get("deployment_target_add_fraction")
    if deployment_target is not None:
        deployment_target = float(deployment_target)
        if not 0.0 < deployment_target < 1.0:
            raise ValueError("deployment_target_add_fraction must be in (0, 1)")
        discovery_predictions = np.asarray([
            bandit.predict(row)[0] for row in discovery
        ], dtype=np.float64)
        # The online dual variable serves exploration and reward acquisition.
        # Deployment uses a discovery-only quantile so that a transient final
        # dual update cannot silently violate the declared compute envelope on
        # a new episode distribution.
        bandit.dual_price = float(np.clip(
            np.quantile(discovery_predictions, 1.0 - deployment_target),
            0.0,
            bandit.maximum_dual_price,
        ))
    discovery_deployment_actions = np.asarray([
        bandit.decide(row, explore=False).add_samples for row in discovery
    ])
    actions = np.asarray([
        bandit.decide(row, explore=False).add_samples for row in evaluation
    ])
    base_budget = int(config.get("base_budget", 50))
    maximum_budget = int(config.get("maximum_budget", 100))
    bootstrap_seed = int(config["bootstrap_seed"])
    oracle = _oracle_actions(
        evaluation, float(config.get("minimum_relative_true_cost_gain", 0.01))
    )
    rng = np.random.RandomState(int(config.get("random_seed", bootstrap_seed ^ 19)))
    random_actions = rng.uniform(size=len(evaluation)) < float(np.mean(actions))
    policies = {
        "fixed_k50": np.zeros(len(evaluation), dtype=bool),
        "fixed_k100": np.ones(len(evaluation), dtype=bool),
        "matched_random": random_actions,
        "bandit": actions,
        "oracle": oracle,
    }
    summaries = {
        name: _policy_summary(
            evaluation, values, base_budget, maximum_budget,
            bootstrap_seed + offset,
        )
        for offset, (name, values) in enumerate(policies.items())
    }
    oracle_gain = summaries["oracle"]["relative_gain_vs_k50"]
    bandit_gain = summaries["bandit"]["relative_gain_vs_k50"]
    retention = bandit_gain / max(oracle_gain, 1e-12)
    bandit_vs_random = _paired_policy_contrast(
        evaluation, actions, random_actions, bootstrap_seed + 100
    )
    randomization = _matched_budget_randomization_test(
        evaluation,
        actions,
        int(config.get("randomization_repeats", 5000)),
        int(config.get("randomization_seed", bootstrap_seed + 101)),
    )
    gate_config = dict(config.get("primary_gate", {}))
    gate = {
        "selective": bool(
            float(gate_config.get("minimum_add_fraction", 0.10))
            <= summaries["bandit"]["add_fraction"]
            <= float(gate_config.get("maximum_add_fraction", 0.75))
        ),
        "superior_to_k50": bool(
            summaries["bandit"]["true_cost_delta_vs_k50_ci95"][1] < 0.0
        ),
        "compute_feasible": bool(
            summaries["bandit"]["mean_budget"]
            <= float(gate_config.get("maximum_mean_budget", 80.0))
        ),
        "oracle_gain_fraction": float(retention),
        "utility_retention": bool(
            retention >= float(gate_config.get("minimum_oracle_gain_fraction", 0.30))
        ),
    }
    gate["passed"] = bool(all(
        value for key, value in gate.items() if key != "oracle_gain_fraction"
    ))
    return {
        "schema_version": 1,
        "algorithm": "primal_dual_linear_contextual_bandit",
        "training_records": len(discovery),
        "evaluation_records": len(evaluation),
        "training_add_fraction": float(np.mean([row["add"] for row in trace])),
        "training_exploratory_fraction": float(np.mean([
            bool(row["exploratory"]) for row in trace
        ])),
        "training_final_dual_price": training_final_dual_price,
        "deployment_dual_price": float(bandit.dual_price),
        "deployment_target_add_fraction": deployment_target,
        "discovery_deployment_add_fraction": float(np.mean(
            discovery_deployment_actions
        )),
        "evaluation_policies": summaries,
        "bandit_vs_matched_random": bandit_vs_random,
        "matched_budget_randomization_test": randomization,
        "primary_gate": gate,
        "bandit": bandit.state_dict(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = evaluate(_read_csv(args.records), config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
