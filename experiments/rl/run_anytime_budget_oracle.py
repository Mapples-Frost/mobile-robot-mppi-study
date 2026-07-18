#!/usr/bin/env python3
"""Nested K50/K100 MuJoCo counterfactual budget diagnostic for L100."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from experiments.rl.run_control_sequence_ranking_diagnostic import (
    _array_sha256,
    _state_vector,
    _true_trajectories,
)
from experiments.rl.run_covariance_context_oracle import _hierarchical_ci
from experiments.rl.run_covariance_sampling_oracle_diagnostic import (
    _apply_terminal_candidate_constraints,
    _finalize_weighted_sequence,
    _terminal_context,
)
from mobile_robot_mppi.core.config import config_hash, deep_merge, git_sha, load_yaml
from mobile_robot_mppi.rl.contextual_bandit import (
    REFERENCE_GEOMETRY_FEATURE_NAMES,
    polyline_geometry_features,
    polyline_window,
)
from mobile_robot_mppi.runtime.factories import make_components


DIAGNOSTIC_FEATURES = (
    "ess_fraction",
    "weight_entropy_fraction",
    "maximum_weight",
    "split_control_disagreement",
    "weighted_action_dispersion",
    "normalized_cost_spread",
    "sample_saturation_fraction",
    "weighted_perturbation_norm",
    "residual_reliability_alpha",
    "residual_reliability_last_relative_improvement",
) + tuple("local_" + name for name in REFERENCE_GEOMETRY_FEATURE_NAMES[1:])


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stable_seed(*values):
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _nested_candidates(controller, prior, count, rng, terminal_context):
    count = int(count)
    center = np.asarray(prior.mean, dtype=np.float64)
    covariance = controller._sampling_covariance(prior)
    noise = rng.multivariate_normal(
        np.zeros(controller.action_spec.dimension),
        covariance,
        size=(count, controller.config.horizon),
    )
    candidates = np.clip(
        center[None, :, :] + noise,
        controller.action_spec.lower[None, None, :],
        controller.action_spec.upper[None, None, :],
    )
    candidates[0] = center
    return _apply_terminal_candidate_constraints(
        controller, candidates, terminal_context
    )


def _weighted_sequence(
    controller, state, target, candidates, prior, preceding_action, context
):
    evaluation = controller.evaluate_control_sequences(
        state, candidates, target, ()
    )
    weighting = controller.importance_weights(
        prior.mean, candidates, evaluation.costs, prior.covariance
    )
    perturbations = candidates - prior.mean[None, :, :]
    sequence = prior.mean + np.sum(
        weighting.weights[:, None, None] * perturbations, axis=0
    )
    sequence = _finalize_weighted_sequence(
        controller, sequence, preceding_action, context
    )
    return sequence, evaluation.costs, weighting.weights


def _first_batch_diagnostics(
    controller, state, target, candidates, prior, preceding_action, context
):
    sequence, costs, weights = _weighted_sequence(
        controller, state, target, candidates, prior, preceding_action, context
    )
    count = len(candidates)
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-300))))
    scale = np.maximum(
        controller.action_spec.upper - controller.action_spec.lower, 1e-9
    )
    first_actions = candidates[:, 0, :]
    mean_action = np.sum(weights[:, None] * first_actions, axis=0)
    centered = (first_actions - mean_action[None, :]) / scale[None, :]
    dispersion = float(np.sum(weights[:, None] * centered ** 2))

    split_actions = []
    for indices in (np.arange(0, count, 2), np.arange(1, count, 2)):
        split_sequence, _costs, _weights = _weighted_sequence(
            controller,
            state,
            target,
            candidates[indices],
            prior,
            preceding_action,
            context,
        )
        split_actions.append(split_sequence[0])
    split_disagreement = float(
        np.linalg.norm((split_actions[0] - split_actions[1]) / scale)
    )
    cost_q10, cost_q50, cost_q90 = np.quantile(costs, (0.10, 0.50, 0.90))
    residual = getattr(controller.dynamics, "residual", None)
    reliability = getattr(residual, "diagnostics", None)
    reliability_values = reliability() if callable(reliability) else {}
    result = {
        "ess_fraction": float((1.0 / np.sum(weights ** 2)) / count),
        "weight_entropy_fraction": float(entropy / np.log(count)),
        "maximum_weight": float(np.max(weights)),
        "split_control_disagreement": split_disagreement,
        "weighted_action_dispersion": dispersion,
        "normalized_cost_spread": float(
            (cost_q90 - cost_q10) / (abs(cost_q50) + 1e-9)
        ),
        "sample_saturation_fraction": float(np.mean(
            (candidates <= controller.action_spec.lower[None, None, :])
            | (candidates >= controller.action_spec.upper[None, None, :])
        )),
        "weighted_perturbation_norm": float(
            np.linalg.norm(sequence - prior.mean)
        ),
        "residual_reliability_alpha": float(
            reliability_values.get("residual_reliability_alpha", 1.0)
        ),
        "residual_reliability_last_relative_improvement": float(
            reliability_values.get(
                "residual_reliability_last_relative_improvement", 0.0
            )
        ),
    }
    return sequence, costs, weights, result


def _local_features(reference, progress, window_m):
    values = polyline_geometry_features(
        polyline_window(reference.points, progress, window_m)
    )
    return {
        "local_" + name: float(value)
        for name, value in zip(REFERENCE_GEOMETRY_FEATURE_NAMES[1:], values[1:])
    }


def _collect_episode(spec, item):
    config = _apply_scene_config(
        load_yaml(ROOT / spec["base_config"]), ROOT / item["scene_path"]
    )
    config = deep_merge(config, {"plant": item["plant_override"]})
    config = deep_merge(config, spec.get("shared_override", {}))
    config["experiment"]["seed"] = int(item["seed"])
    config["planner"].update({
        "prediction_mode": "icode_residual",
        "checkpoint": str(ROOT / spec["icode_checkpoint"]),
        "residual_torch_num_threads": 1,
        "residual_torchscript": True,
        "sampling_prior": "contextual_bandit_covariance",
        "num_samples": int(spec["maximum_budget"]),
    })
    config.setdefault("rl", {}).update({
        "enabled": True,
        "policy_id": "linucb_contextual_covariance_l89",
        "checkpoint": str(ROOT / spec["bandit_checkpoint"]),
    })
    config.setdefault("memory", {})["enable"] = False
    if config.get("scene", {}).get("obstacles"):
        raise ValueError("L100 counterfactual branches require obstacle-free scenes")

    components = make_components(config, ROOT)
    plant = components["plant"]
    controller = components["controller"]
    reference = components["reference"]
    state_spec = components["state_spec"]
    dt = float(config["experiment"]["control_dt"])
    seed = int(item["seed"])
    initial = np.asarray(config["experiment"]["initial_state"], dtype=np.float64)
    truth = plant.reset(seed, initial)
    observation = components["sensors"].reset(truth, seed)
    reset_perception = getattr(components["perception"], "reset", None)
    if callable(reset_perception):
        reset_perception()
    reset_reference = getattr(reference, "reset", None)
    if callable(reset_reference):
        reset_reference()
    controller.reset(seed=seed)
    total_length = float(reference.total_length)
    anchors = list(float(value) for value in spec["anchor_progress_fractions"])
    next_anchor = 0
    rows = []

    try:
        for step_index in range(int(config["experiment"]["max_steps"])):
            perceived = components["perception"].process(observation)
            state = _state_vector(truth, state_spec)
            target = reference.target_at(observation.timestamp, state)
            progress_fraction = float(reference.progress / total_length)
            while next_anchor < len(anchors) and progress_fraction >= anchors[next_anchor]:
                anchor_fraction = anchors[next_anchor]
                preceding = controller.previous_action.copy()
                prior = controller._prior(perceived.observation, reference)
                context = _terminal_context(controller, state, target)
                snapshot = plant.snapshot()
                rng = np.random.RandomState(_stable_seed(
                    spec["design_id"], item["split"], item["scene_path"],
                    item["physics_domain"], seed, anchor_fraction,
                ))
                candidates = _nested_candidates(
                    controller, prior, spec["maximum_budget"], rng, context
                )
                base_count = int(spec["base_budget"])
                sequence50, costs50, weights50, diagnostics = (
                    _first_batch_diagnostics(
                        controller,
                        state,
                        target,
                        candidates[:base_count],
                        prior,
                        preceding,
                        context,
                    )
                )
                sequence100, costs100, weights100 = _weighted_sequence(
                    controller,
                    state,
                    target,
                    candidates,
                    prior,
                    preceding,
                    context,
                )
                true_paths, collisions = _true_trajectories(
                    plant,
                    snapshot,
                    np.stack((sequence50, sequence100)),
                    state_spec,
                    dt,
                )
                controller.previous_action = preceding.copy()
                true_costs = controller.cost_trajectories(
                    true_paths,
                    np.stack((sequence50, sequence100)),
                    target,
                    (),
                )
                plant.restore(snapshot)
                controller.previous_action = preceding.copy()
                relative_gain = float(
                    (true_costs[0] - true_costs[1])
                    / (abs(true_costs[0]) + 1e-9)
                )
                row = {
                    "split": item["split"],
                    "scene": str(config["scene"]["name"]),
                    "physics_domain": item["physics_domain"],
                    "seed": seed,
                    "anchor_index": next_anchor,
                    "anchor_fraction": anchor_fraction,
                    "actual_progress_fraction": progress_fraction,
                    "step": step_index,
                    "snapshot_time": float(snapshot.time),
                    "target_phase": str(target.phase),
                    "base_budget": base_count,
                    "maximum_budget": int(spec["maximum_budget"]),
                    "candidate_prefix_sha256": _array_sha256(
                        candidates[:base_count]
                    ),
                    "candidate_full_prefix_sha256": _array_sha256(
                        candidates[:base_count].copy()
                    ),
                    "candidate_full_sha256": _array_sha256(candidates),
                    "prior_covariance": json.dumps(
                        np.asarray(prior.covariance).tolist()
                    ),
                    "prior_metadata": json.dumps(prior.metadata, sort_keys=True),
                    "predicted_cost50_min": float(np.min(costs50)),
                    "predicted_cost100_min": float(np.min(costs100)),
                    "ess100_fraction": float(
                        (1.0 / np.sum(weights100 ** 2)) / len(weights100)
                    ),
                    "action_difference_norm": float(np.linalg.norm(
                        (sequence100[0] - sequence50[0])
                        / np.maximum(
                            controller.action_spec.upper
                            - controller.action_spec.lower,
                            1e-9,
                        )
                    )),
                    "true_cost50": float(true_costs[0]),
                    "true_cost100": float(true_costs[1]),
                    "true_cost_gain": float(true_costs[0] - true_costs[1]),
                    "relative_true_cost_gain": relative_gain,
                    "collision50": int(collisions[0]),
                    "collision100": int(collisions[1]),
                    **diagnostics,
                    **_local_features(
                        reference, reference.progress,
                        float(spec["local_route_window_m"]),
                    ),
                }
                rows.append(row)
                next_anchor += 1

            plan = controller.plan(perceived.observation, reference)
            decision = components["safety"].arbitrate(
                plan.proposed_control, perceived.guard
            )
            controller.observe_safety_decision(decision)
            transition = plant.step(decision.executed_control, dt)
            truth = transition.ground_truth
            if truth.collision:
                raise RuntimeError("L100 live reference episode collided")
            observation = components["sensors"].observe(truth)
            if next_anchor == len(anchors):
                break
    finally:
        plant.close()
    if next_anchor != len(anchors):
        raise RuntimeError(
            "episode reached %d/%d anchor fractions" % (next_anchor, len(anchors))
        )
    return rows, config


def _feature_matrix(rows):
    values = np.asarray(
        [[float(row[name]) for name in DIAGNOSTIC_FEATURES] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("budget features contain NaN or Inf")
    return values


def _fit_ridge(rows, penalty):
    x = _feature_matrix(rows)
    y = np.asarray(
        [float(row["relative_true_cost_gain"]) for row in rows],
        dtype=np.float64,
    )
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    regularizer = np.eye(design.shape[1]) * float(penalty)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ y
    )
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _predict_ridge(model, rows):
    x = (_feature_matrix(rows) - model["mean"]) / model["scale"]
    return np.column_stack((np.ones(len(x)), x)) @ model["coefficients"]


def _fit_stump(rows, margin):
    best = None
    for feature in DIAGNOSTIC_FEATURES:
        values = np.asarray([float(row[feature]) for row in rows])
        thresholds = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        for direction in ("low", "high"):
            for threshold in thresholds:
                add = values <= threshold if direction == "low" else values >= threshold
                selected = np.where(
                    add,
                    [float(row["true_cost100"]) for row in rows],
                    [float(row["true_cost50"]) for row in rows],
                )
                objective = float(np.mean(selected))
                candidate = (objective, float(np.mean(add)), feature, direction, float(threshold))
                if best is None or candidate < best:
                    best = candidate
    return {
        "feature": best[2], "direction": best[3], "threshold": best[4],
        "discovery_add_fraction": best[1], "margin": float(margin),
    }


def _stump_actions(model, rows):
    values = np.asarray([float(row[model["feature"]]) for row in rows])
    return values <= model["threshold"] if model["direction"] == "low" else values >= model["threshold"]


def _oracle_actions(rows, margin):
    return np.asarray([
        float(row["relative_true_cost_gain"]) >= float(margin)
        and not int(row["collision100"])
        for row in rows
    ], dtype=bool)


def _policy_episode_differences(rows, actions):
    episodes = defaultdict(list)
    for row, add in zip(rows, actions):
        key = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        chosen = float(row["true_cost100"] if add else row["true_cost50"])
        episodes[key].append(chosen - float(row["true_cost50"]))
    return {key: float(np.mean(values)) for key, values in episodes.items()}


def _policy_summary(rows, actions, base_budget, maximum_budget, bootstrap_seed):
    actions = np.asarray(actions, dtype=bool)
    differences = _policy_episode_differences(rows, actions)
    grouped = defaultdict(list)
    for (scene, domain, _seed), value in differences.items():
        grouped[scene + "__" + domain].append(value)
    chosen = np.asarray([
        float(row["true_cost100"] if add else row["true_cost50"])
        for row, add in zip(rows, actions)
    ])
    base = np.asarray([float(row["true_cost50"]) for row in rows])
    return {
        "anchors": len(rows),
        "episodes": len(differences),
        "add_fraction": float(np.mean(actions)),
        "mean_budget": float(
            base_budget + (maximum_budget - base_budget) * np.mean(actions)
        ),
        "true_cost_mean": float(np.mean(chosen)),
        "true_cost_delta_vs_k50_mean": float(np.mean(list(differences.values()))),
        "true_cost_delta_vs_k50_ci95": _hierarchical_ci(
            dict(grouped), int(bootstrap_seed)
        ),
        "relative_gain_vs_k50": float(
            np.mean(base - chosen) / (np.mean(np.abs(base)) + 1e-9)
        ),
    }


def analyze(rows, spec):
    discovery = [row for row in rows if row["split"] == "discovery"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    margin = float(spec["minimum_relative_true_cost_gain"])
    ridge = _fit_ridge(discovery, spec["ridge_penalty"])
    stump = _fit_stump(discovery, margin)
    ridge_discovery_prediction = _predict_ridge(ridge, discovery)
    ridge_evaluation_prediction = _predict_ridge(ridge, evaluation)
    target_add_fraction = spec.get("ridge_target_add_fraction")
    if target_add_fraction is None:
        ridge_threshold = margin
    else:
        target_add_fraction = float(target_add_fraction)
        if not 0.0 < target_add_fraction < 1.0:
            raise ValueError("ridge_target_add_fraction must be in (0, 1)")
        ridge_threshold = float(np.quantile(
            ridge_discovery_prediction, 1.0 - target_add_fraction
        ))
    ridge_actions = ridge_evaluation_prediction >= ridge_threshold
    stump_actions = _stump_actions(stump, evaluation)
    oracle_actions = _oracle_actions(evaluation, margin)
    rng = np.random.RandomState(int(spec["bootstrap_seed"]) ^ 0xA11CE)
    random_actions = rng.uniform(size=len(evaluation)) < float(np.mean(ridge_actions))
    policies = {
        "fixed_k50": np.zeros(len(evaluation), dtype=bool),
        "fixed_k100": np.ones(len(evaluation), dtype=bool),
        "matched_random": random_actions,
        "stump": stump_actions,
        "ridge": ridge_actions,
        "oracle": oracle_actions,
    }
    policy_summaries = {
        name: _policy_summary(
            evaluation, actions, int(spec["base_budget"]),
            int(spec["maximum_budget"]), int(spec["bootstrap_seed"]) + offset,
        )
        for offset, (name, actions) in enumerate(policies.items())
    }
    oracle_gain = policy_summaries["oracle"]["relative_gain_vs_k50"]
    ridge_gain = policy_summaries["ridge"]["relative_gain_vs_k50"]
    retention = ridge_gain / max(oracle_gain, 1e-12)
    gate_spec = spec["primary_gate"]
    gate = {
        "oracle_selectivity": bool(
            float(gate_spec["minimum_oracle_add_fraction"])
            <= policy_summaries["oracle"]["add_fraction"]
            <= float(gate_spec["maximum_oracle_add_fraction"])
        ),
        "oracle_practical_gain": bool(
            oracle_gain >= float(gate_spec["minimum_oracle_relative_gain"])
        ),
        "ridge_superiority": bool(
            policy_summaries["ridge"]["true_cost_delta_vs_k50_ci95"][1] < 0.0
        ),
        "ridge_efficiency": bool(
            policy_summaries["ridge"]["mean_budget"]
            <= float(gate_spec["maximum_learned_mean_budget"])
        ),
        "ridge_oracle_gain_fraction": float(retention),
        "ridge_utility_retention": bool(
            retention >= float(gate_spec["minimum_learned_oracle_gain_fraction"])
        ),
    }
    gate["passed"] = bool(all(
        value for key, value in gate.items()
        if key != "ridge_oracle_gain_fraction"
    ))
    hashes_match = all(
        row["candidate_prefix_sha256"] == row["candidate_full_prefix_sha256"]
        for row in rows
    )
    audit = {
        "records": len(rows),
        "discovery_records": len(discovery),
        "evaluation_records": len(evaluation),
        "episodes": len(set(
            (row["split"], row["scene"], row["physics_domain"], int(row["seed"]))
            for row in rows
        )),
        "nested_prefix_hashes_match": hashes_match,
        "counterfactual_collisions": int(sum(
            int(row["collision50"]) + int(row["collision100"]) for row in rows
        )),
        "finite_features": bool(np.isfinite(_feature_matrix(rows)).all()),
    }
    audit["passed"] = bool(
        hashes_match and audit["counterfactual_collisions"] == 0
        and audit["finite_features"]
    )
    gate["integrity"] = audit["passed"]
    gate["passed"] = bool(gate["passed"] and audit["passed"])
    return {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "audit": audit,
        "feature_names": list(DIAGNOSTIC_FEATURES),
        "ridge_model": {
            "mean": ridge["mean"].tolist(),
            "scale": ridge["scale"].tolist(),
            "coefficients": ridge["coefficients"].tolist(),
            "decision_threshold": float(ridge_threshold),
            "target_add_fraction": target_add_fraction,
            "discovery_add_fraction": float(np.mean(
                ridge_discovery_prediction >= ridge_threshold
            )),
        },
        "stump_model": stump,
        "evaluation_policies": policy_summaries,
        "primary_gate": gate,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=True)

    schedule = []
    for split, seeds in (
        ("discovery", spec["discovery_seeds"]),
        ("evaluation", spec["evaluation_seeds"]),
    ):
        for scene_path in spec["scenes"]:
            for domain in spec["physics_domains"]:
                for seed in seeds:
                    schedule.append({
                        "split": split,
                        "scene_path": scene_path,
                        "physics_domain": domain["name"],
                        "plant_override": domain.get("plant_override", {}),
                        "seed": int(seed),
                    })
    np.random.RandomState(int(spec["schedule_seed"])).shuffle(schedule)
    with (output / "schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2, sort_keys=True)

    records_path = output / "anchors.csv"
    rows = _read_csv(records_path) if records_path.exists() else []
    complete = Counter(
        (row["split"], row["scene"], row["physics_domain"], int(row["seed"]))
        for row in rows
    )
    attempted = 0
    resolved_hashes = []
    for item in schedule:
        scene_name = str(load_yaml(ROOT / item["scene_path"])["scene"]["name"])
        key = (item["split"], scene_name, item["physics_domain"], item["seed"])
        if complete[key] == len(spec["anchor_progress_fractions"]):
            continue
        if args.max_episodes is not None and attempted >= int(args.max_episodes):
            break
        attempted += 1
        episode_rows, resolved = _collect_episode(spec, item)
        rows.extend(episode_rows)
        resolved_hashes.append(config_hash(resolved))
        _write_csv(records_path, rows)

    expected_episodes = len(schedule)
    expected_records = expected_episodes * len(spec["anchor_progress_fractions"])
    payload = {
        "design_id": spec["design_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(spec),
        "completed_records": len(rows),
        "expected_records": expected_records,
        "complete": len(rows) == expected_records,
        "resolved_config_hashes_this_invocation": resolved_hashes,
    }
    if payload["complete"]:
        payload["analysis"] = analyze(rows, spec)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
