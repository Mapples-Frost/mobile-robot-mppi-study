#!/usr/bin/env python3
"""Train an offline LinUCB covariance option policy from closed-loop returns."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.run_covariance_context_oracle import _hierarchical_ci
from mobile_robot_mppi.rl.contextual_bandit import (
    LinUCBCovarianceBandit,
    REFERENCE_GEOMETRY_FEATURE_NAMES,
    polyline_geometry_features,
)


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scene_features(paths):
    result = {}
    for relative in paths:
        path = ROOT / relative
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        name = str(config["experiment"]["name"])
        result[name] = polyline_geometry_features(config["task"]["points"])
    return result


def _reward(row, config):
    time_s = float(row["time_to_goal_s"])
    rmse = float(row["cross_track_rmse"])
    cost = time_s + float(config["cross_track_weight_s_per_m"]) * rmse
    if str(row["success"]).lower() != "true":
        cost += float(config["failure_penalty_s"])
    if str(row["collision"]).lower() == "true":
        cost += float(config["collision_penalty_s"])
    return -cost / float(config["scale_s"])


def _filtered(rows, split, scenes, actions):
    scenes = set(scenes)
    actions = set(actions)
    return [
        row for row in rows
        if row["split"] == split
        and row["scene"] in scenes
        and row["candidate"] in actions
    ]


def _fit(rows, features, actions, ridge, alpha, reward_config):
    bandit = LinUCBCovarianceBandit(
        actions,
        len(REFERENCE_GEOMETRY_FEATURE_NAMES),
        ridge=ridge,
        exploration_alpha=alpha,
    )
    for row in rows:
        bandit.update(
            features[row["scene"]],
            row["candidate"],
            _reward(row, reward_config),
        )
    return bandit


def _mean_regret(bandit, rows, features, reward_config):
    grouped = {}
    for row in rows:
        key = (row["scene"], row["physics_domain"], int(row["seed"]))
        grouped.setdefault(key, {})[row["candidate"]] = _reward(row, reward_config)
    regrets = []
    accuracy = []
    for key, outcomes in sorted(grouped.items()):
        selected = bandit.decide(features[key[0]], explore=False).action
        if selected not in outcomes or len(outcomes) < 2:
            raise ValueError("validation context is missing a covariance action")
        best = max(outcomes, key=outcomes.get)
        regrets.append(outcomes[best] - outcomes[selected])
        accuracy.append(float(selected == best))
    return float(np.mean(regrets)), float(np.mean(accuracy)), len(grouped)


def _heldout_gate(bandit, rows, features, comparator, margin, bootstrap_seed):
    indexed = {
        (row["scene"], row["physics_domain"], int(row["seed"]), row["candidate"]): row
        for row in rows
    }
    differences = {"time": [], "rmse": [], "jerk": []}
    hierarchical = {"time": {}, "rmse": {}, "jerk": {}}
    safety = []
    decisions = {}
    keys = sorted({key[:3] for key in indexed})
    for scene, domain, seed in keys:
        selected = bandit.decide(features[scene], explore=False).action
        decisions[scene] = selected
        left = indexed[(scene, domain, seed, selected)]
        right = indexed[(scene, domain, seed, comparator)]
        context = scene + "__" + domain
        for name, column in (
            ("time", "time_to_goal_s"),
            ("rmse", "cross_track_rmse"),
            ("jerk", "control_jerk"),
        ):
            delta = float(left[column]) - float(right[column])
            differences[name].append(delta)
            hierarchical[name].setdefault(context, []).append(delta)
        safety.append((
            float(str(left["success"]).lower() == "true")
            - float(str(right["success"]).lower() == "true"),
            float(str(left["collision"]).lower() == "true")
            - float(str(right["collision"]).lower() == "true"),
        ))
    output = {"paired_episodes": len(keys), "decisions": decisions}
    for offset, name in enumerate(("time", "rmse", "jerk")):
        output[name + "_delta_mean"] = float(np.mean(differences[name]))
        output[name + "_delta_ci95"] = _hierarchical_ci(
            hierarchical[name], bootstrap_seed + offset
        )
    output["success_delta_mean"] = float(np.mean([value[0] for value in safety]))
    output["collision_delta_mean"] = float(np.mean([value[1] for value in safety]))
    output["precision_gate_passed"] = bool(output["rmse_delta_ci95"][1] <= margin)
    output["time_gate_passed"] = bool(output["time_delta_ci95"][1] < 0.0)
    output["safety_gate_passed"] = bool(
        output["success_delta_mean"] >= 0.0
        and output["collision_delta_mean"] <= 0.0
    )
    output["primary_gate_passed"] = bool(
        output["precision_gate_passed"]
        and output["time_gate_passed"]
        and output["safety_gate_passed"]
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = _scene_features(config["scene_configs"])
    actions = tuple(config["action_scales"])
    train_rows = []
    validation_rows = []
    for source in config["data_sources"]:
        rows = _read_rows(ROOT / source["path"])
        train_rows.extend(_filtered(
            rows, source["train_split"], source["train_scenes"], actions
        ))
        validation_rows.extend(_filtered(
            rows,
            source["validation_split"],
            source["validation_scenes"],
            actions,
        ))
    if not train_rows or not validation_rows:
        raise ValueError("bandit train/validation data cannot be empty")

    candidates = []
    for ridge in config["bandit"]["ridge_candidates"]:
        model = _fit(
            train_rows,
            features,
            actions,
            float(ridge),
            float(config["bandit"]["exploration_alpha"]),
            config["reward"],
        )
        regret, accuracy, contexts = _mean_regret(
            model, validation_rows, features, config["reward"]
        )
        candidates.append({
            "ridge": float(ridge),
            "validation_mean_regret": regret,
            "validation_action_accuracy": accuracy,
            "validation_contexts": contexts,
        })
    selected_hyperparameter = min(
        candidates,
        key=lambda value: (
            value["validation_mean_regret"],
            -value["validation_action_accuracy"],
            value["ridge"],
        ),
    )
    bandit = _fit(
        train_rows,
        features,
        actions,
        selected_hyperparameter["ridge"],
        float(config["bandit"]["exploration_alpha"]),
        config["reward"],
    )

    predictions = []
    for scene, vector in sorted(features.items()):
        decision = bandit.decide(vector, explore=False)
        predictions.append({
            "scene": scene,
            "action": decision.action,
            "features": vector.tolist(),
            "predicted_rewards": dict(decision.predicted_rewards),
            "confidence_widths": dict(decision.confidence_widths),
        })

    test = config["heldout_policy_test"]
    heldout_rows = _filtered(
        _read_rows(ROOT / test["source"]),
        test["split"],
        test["scenes"],
        actions,
    )
    heldout = _heldout_gate(
        bandit,
        heldout_rows,
        features,
        str(test["comparator"]),
        float(test["cross_track_noninferiority_margin_m"]),
        int(config["seed"]),
    )
    checkpoint = {
        "schema_version": 1,
        "model_class": "LinUCBCovarianceBandit",
        "git_sha": _git_sha(),
        "config": config,
        "feature_names": list(REFERENCE_GEOMETRY_FEATURE_NAMES),
        "action_scales": config["action_scales"],
        "bandit": bandit.state_dict(),
        "selected_hyperparameter": selected_hyperparameter,
        "train_rows": len(train_rows),
    }
    summary = {
        "design_id": config["design_id"],
        "git_sha": checkpoint["git_sha"],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "ridge_selection": candidates,
        "selected_hyperparameter": selected_hyperparameter,
        "predictions": predictions,
        "heldout_policy_test": heldout,
        "checkpoint": str(output_dir / "checkpoint.json"),
    }
    with (output_dir / "checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2, sort_keys=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
