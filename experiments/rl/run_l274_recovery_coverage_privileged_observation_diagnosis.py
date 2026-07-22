#!/usr/bin/env python3
"""Run the frozen L274 recovery-coverage x privileged-observation diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.rl.run_l273_critic_capacity_action_representation_diagnosis import (
    StateActionSampler,
    _batch,
    _json_dump,
    _metrics,
    _read_csv,
    _sac_config,
    _save_checkpoint,
    _sha256,
    _standardize_by_state,
    _verify,
    _write_csv,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.sac import QNetwork, quantile_huber_loss


PROTOCOL = "L274"
DEFAULT_CONFIG = ROOT / "configs/rl/l274_recovery_coverage_privileged_observation_diagnosis.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l274_recovery_coverage_privileged_observation_diagnosis"


def _load_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload["l274"]
    if section["protocol"] != PROTOCOL or section["actor_training_authorized"] is not False:
        raise ValueError("L274 must remain diagnostic-only")
    references = json.dumps(
        {key: value for key, value in section.items() if key != "forbidden_tokens"},
        sort_keys=True,
    ).lower()
    entered = [
        token for token in section["forbidden_tokens"] if token.lower() in references
    ]
    if entered:
        raise ValueError("forbidden artifact entered L274 config: %s" % entered)
    return payload, section


def _privileged_vector(state):
    path = state["path_state"]
    progress = float(path["progress"])
    remaining = float(path["remaining"])
    total = max(progress + remaining, 1e-9)
    values = np.asarray([
        float(path["signed_cross_track_error"]),
        float(path["cross_track_error"]),
        float(path["heading_error"]),
        float(path["curvature"]),
        progress / total,
        remaining / total,
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("L274 privileged path vector is non-finite")
    return values


def _rows_from_l263(section):
    source = section["source"]
    paths = {
        "states": ROOT / source["l263_states"],
        "returns": ROOT / source["l263_returns"],
        "actions": ROOT / source["l263_actions"],
    }
    for name, path in paths.items():
        _verify(path, source["l263_%s_sha256" % name])
    states = json.loads(paths["states"].read_text(encoding="utf-8"))
    state_by_id = {row["state_id"]: row for row in states}
    actions = {
        (row["state_id"], row["action_id"]): row for row in _read_csv(paths["actions"])
    }
    rows = []
    for row in _read_csv(paths["returns"]):
        state = state_by_id[row["state_id"]]
        if state["scene_role"] != source["train_role"]:
            continue
        if row["continuation"] != source["continuation"]:
            continue
        if int(row["horizon"]) != int(source["horizon"]):
            continue
        action = actions[(row["state_id"], row["action_id"])]
        rows.append({
            "state_id": row["state_id"], "action_id": row["action_id"],
            "scene": state["scene"],
            "observation": np.asarray(state["normalized_observation"], dtype=np.float32),
            "privileged": _privileged_vector(state),
            "action": np.asarray(
                [float(action["normalized_v"]), float(action["normalized_omega"])],
                dtype=np.float32,
            ),
            "return": float(row["discounted_return"]),
            "origin": "l263",
        })
    counts = defaultdict(int)
    for row in rows:
        counts[row["state_id"]] += 1
    if len(counts) != int(source["expected_states"]):
        raise ValueError("L274 L263 source state count mismatch")
    if set(counts.values()) != {int(source["expected_actions_per_state"])}:
        raise ValueError("L274 L263 action count mismatch")
    return _standardize_by_state(rows, section["design"]["target_std_floor"])


def _assign_folds(states, folds):
    grouped = defaultdict(list)
    for row in states:
        grouped[row["scene"]].append(row)
    assignment = {}
    for scene, scene_rows in sorted(grouped.items()):
        ordered = sorted(
            scene_rows,
            key=lambda row: (
                str(row["severity"]), str(row["side"]), int(row["chain_id"]),
                row["state_id"],
            ),
        )
        for index, row in enumerate(ordered):
            assignment[row["state_id"]] = index % int(folds)
    return assignment


def _rows_from_recovery(section):
    recovery = section["recovery"]
    state_path, return_path = ROOT / recovery["states"], ROOT / recovery["returns"]
    _verify(state_path, recovery["states_sha256"])
    _verify(return_path, recovery["returns_sha256"])
    states = json.loads(state_path.read_text(encoding="utf-8"))
    state_by_id = {row["state_id"]: row for row in states}
    assignment = _assign_folds(states, section["cross_fit"]["folds"])
    rows = []
    for row in _read_csv(return_path):
        state = state_by_id[row["state_id"]]
        rows.append({
            "state_id": row["state_id"], "action_id": row["action_id"],
            "scene": row["scene"],
            "observation": np.asarray(state["normalized_observation"], dtype=np.float32),
            "privileged": _privileged_vector(state),
            "action": np.asarray(
                [float(row["normalized_v"]), float(row["normalized_omega"])],
                dtype=np.float32,
            ),
            "return": float(row["discounted_return"]),
            "origin": "l268_recovery",
            "fold": int(assignment[row["state_id"]]),
        })
    counts = defaultdict(int)
    scenes = defaultdict(set)
    for row in rows:
        counts[row["state_id"]] += 1
        scenes[row["scene"]].add(row["state_id"])
    if len(counts) != int(recovery["expected_states"]):
        raise ValueError("L274 recovery state count mismatch")
    if len(scenes) != int(recovery["expected_scenes"]):
        raise ValueError("L274 recovery scene count mismatch")
    if set(map(len, scenes.values())) != {int(recovery["expected_states_per_scene"])}:
        raise ValueError("L274 recovery scene balance mismatch")
    if set(counts.values()) != {int(recovery["expected_actions_per_state"])}:
        raise ValueError("L274 recovery action count mismatch")
    state_fold_counts = defaultdict(set)
    for row in rows:
        state_fold_counts[row["fold"]].add(row["state_id"])
    if set(map(len, state_fold_counts.values())) != {
        int(section["cross_fit"]["heldout_states_per_fold"])
    }:
        raise ValueError("L274 fold size mismatch")
    rows = _standardize_by_state(rows, section["design"]["target_std_floor"])
    for row in rows:
        row["fold"] = int(assignment[row["state_id"]])
    return rows, assignment


def _privileged_stats(rows, floor):
    unique = {}
    for row in rows:
        unique[row["state_id"]] = row["privileged"]
    values = np.stack([unique[key] for key in sorted(unique)]).astype(np.float64)
    return values.mean(axis=0), np.maximum(values.std(axis=0), float(floor))


def _feature_rows(rows, privileged, stats=None):
    output = []
    for row in rows:
        observation = row["observation"]
        if privileged:
            mean, std = stats
            extra = ((row["privileged"] - mean) / std).astype(np.float32)
            observation = np.concatenate((observation, extra)).astype(np.float32)
        output.append({**row, "observation": observation})
    return output


def _train_model(train_rows, eval_rows, arm, seed, fold, section, output, progress):
    design = section["design"]
    privileged = bool(arm["privileged_features"])
    stats = _privileged_stats(train_rows, section["features"]["std_floor"]) if privileged else None
    train_features = _feature_rows(train_rows, privileged, stats)
    eval_features = _feature_rows(eval_rows, privileged, stats)
    device = torch.device(design["device"])
    torch.manual_seed(int(seed) + (0 if fold is None else 1000 * int(fold)))
    torch.cuda.manual_seed_all(int(seed) + (0 if fold is None else 1000 * int(fold)))
    config = _sac_config(design, design["hidden_sizes"])
    observation_dim = int(section["features"]["base_observation_dim"]) + (
        int(section["features"]["privileged_dim"]) if privileged else 0
    )
    model = QNetwork(observation_dim, 2, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(design["learning_rate"]))
    sampler = StateActionSampler(train_features, int(seed) + (fold or 0) * 1000)
    model.train()
    for update in range(1, int(design["updates"]) + 1):
        observations, actions, targets = _batch(
            sampler.sample(int(design["batch_size"])), "raw", device
        )
        values = model(observations, actions)
        loss = quantile_huber_loss(
            values, targets, float(design["critic_quantile_huber_kappa"])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(design["gradient_clip_norm"])
        )
        optimizer.step()
        scalar_loss = float(loss.detach().cpu())
        if not np.isfinite(scalar_loss):
            raise FloatingPointError("L274 loss is non-finite")
        if update in design["checkpoint_updates"]:
            fold_name = "shared" if fold is None else "fold_%d" % fold
            path = output / ("seed_%d" % seed) / arm["name"] / fold_name / "checkpoints" / (
                "update_%06d.pt" % update
            )
            digest = _save_checkpoint(
                path, model, optimizer, sampler,
                {"protocol": PROTOCOL, "seed": seed, "arm": arm["name"],
                 "fold": fold_name, "updates": update,
                 "privileged_probe_only": privileged},
            )
            progress.append({
                "seed": seed, "arm": arm["name"], "fold": fold_name,
                "completed_updates": update, "checkpoint_sha256": digest,
                "finite": True,
            })
            _write_csv(output / "progress.csv", progress)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(eval_features), 512):
            chunk = eval_features[start:start + 512]
            observations, actions, _ = _batch(chunk, "raw", device)
            q = model(observations, actions).mean(dim=-1).cpu().numpy()
            predictions.extend({**row, "prediction": float(value)} for row, value in zip(chunk, q))
    del model, optimizer
    torch.cuda.empty_cache()
    return predictions


def _paired_check(results, treatment, section):
    gate = section["gate"]
    by = {(row["seed"], row["arm"]): row for row in results}
    seeds = sorted({row["seed"] for row in results})
    pair_gains, spearman_gains = [], []
    consistent = 0
    for seed in seeds:
        baseline, candidate = by[(seed, "source_69d")], by[(seed, treatment)]
        pair = candidate["recovery_forward_accuracy"] - baseline["recovery_forward_accuracy"]
        spear = candidate["three_action_spearman"] - baseline["three_action_spearman"]
        pair_gains.append(pair)
        spearman_gains.append(spear)
        consistent += int(pair > 0.0 and spear > 0.0)
    scenes = sorted(by[(seeds[0], "source_69d")]["scene_pair_accuracy"])
    changes = {}
    for scene in scenes:
        changes[scene] = float(np.mean([
            by[(seed, treatment)]["scene_pair_accuracy"][scene]
            - by[(seed, "source_69d")]["scene_pair_accuracy"][scene]
            for seed in seeds
        ]))
    aggregate_accuracy = float(np.mean([
        by[(seed, treatment)]["recovery_forward_accuracy"] for seed in seeds
    ]))
    passed = bool(
        aggregate_accuracy >= float(gate["minimum_aggregate_pair_accuracy"])
        and float(np.median(pair_gains)) >= float(gate["minimum_paired_median_pair_gain"])
        and float(np.median(spearman_gains)) >= float(gate["minimum_paired_median_spearman_gain"])
        and consistent >= int(gate["minimum_consistent_seed_blocks"])
        and sum(value > 0.0 for value in changes.values()) >= int(gate["minimum_scenes_improved"])
        and min(changes.values()) >= -float(gate["maximum_scene_pair_decrease"])
    )
    return {
        "arm": treatment, "gate_pass": passed,
        "aggregate_pair_accuracy": aggregate_accuracy,
        "paired_median_pair_gain": float(np.median(pair_gains)),
        "paired_median_spearman_gain": float(np.median(spearman_gains)),
        "consistent_seed_blocks": consistent,
        "scenes_improved": sum(value > 0.0 for value in changes.values()),
        "minimum_scene_pair_change": min(changes.values()),
        "scene_pair_changes": changes,
    }


def _decision(results, section):
    checks = {
        arm: _paired_check(results, arm, section)
        for arm in (
            "source_privileged75d", "recovery_augmented_69d",
            "recovery_augmented_privileged75d",
        )
    }
    privileged = checks["source_privileged75d"]["gate_pass"]
    coverage = checks["recovery_augmented_69d"]["gate_pass"]
    combined = checks["recovery_augmented_privileged75d"]["gate_pass"]
    if combined and (privileged and coverage or not privileged and not coverage):
        status, selected = (
            "combined_coverage_observation_limited",
            "recovery_augmented_privileged75d",
        )
    elif privileged:
        status, selected = "privileged_observation_limited", "source_privileged75d"
    elif coverage:
        status, selected = "recovery_state_coverage_limited", "recovery_augmented_69d"
    elif combined:
        status, selected = (
            "combined_coverage_observation_limited",
            "recovery_augmented_privileged75d",
        )
    else:
        status, selected = "temporal_or_target_dynamics_unresolved", None
    return status, selected, list(checks.values())


def run(config_path: Path, output: Path):
    payload, section = _load_config(config_path)
    summary_path = ROOT / section["l273"]["summary"]
    _verify(summary_path, section["l273"]["summary_sha256"])
    prior = json.loads(summary_path.read_text(encoding="utf-8"))
    if prior["decision"] != section["l273"]["required_decision"]:
        raise ValueError("L274 requires the frozen L273 decision")
    if section["design"]["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L274 requires CUDA")
    source_rows = _rows_from_l263(section)
    recovery_rows, assignment = _rows_from_recovery(section)
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "fold_assignment.json", assignment)
    arms = {row["name"]: row for row in section["design"]["arms"]}
    progress, results, predictions_out, states_out = [], [], [], []
    for seed in section["design"]["paired_seeds"]:
        order = np.random.RandomState(
            int(section["design"]["arm_order_seed"]) + int(seed)
        ).permutation(len(arms))
        arm_names = [list(arms)[index] for index in order]
        for arm_name in arm_names:
            arm = arms[arm_name]
            predictions = []
            if arm["recovery_training"]:
                for fold in range(int(section["cross_fit"]["folds"])):
                    recovery_train = [row for row in recovery_rows if row["fold"] != fold]
                    recovery_eval = [row for row in recovery_rows if row["fold"] == fold]
                    predictions.extend(_train_model(
                        source_rows + recovery_train, recovery_eval, arm, seed, fold,
                        section, output, progress,
                    ))
            else:
                predictions = _train_model(
                    source_rows, recovery_rows, arm, seed, None, section, output, progress,
                )
            if len({row["state_id"] for row in predictions}) != 36:
                raise RuntimeError("L274 cross-fitted prediction coverage is incomplete")
            state_rows, metrics = _metrics(
                predictions, tuple(section["recovery"]["primary_pair"])
            )
            results.append({
                "seed": seed, "arm": arm_name, "finite": True,
                "recovery_forward_accuracy": metrics["recovery_forward_accuracy"],
                "three_action_spearman": metrics["mean_state_spearman"],
                "top1_agreement": metrics["top1_agreement"],
                "target_mse": metrics["target_mse"],
                "scene_pair_accuracy": metrics["scene_pair_accuracy"],
            })
            predictions_out.extend({
                "seed": seed, "arm": arm_name, "state_id": row["state_id"],
                "action_id": row["action_id"], "scene": row["scene"],
                "fold": row["fold"], "return": row["return"],
                "standardized_target": row["target"], "prediction": row["prediction"],
            } for row in predictions)
            states_out.extend({"seed": seed, "arm": arm_name, **row} for row in state_rows)
            print(json.dumps({"completed_seed": seed, "completed_arm": arm_name}))
    expected = len(section["design"]["paired_seeds"]) * len(arms)
    if len(results) != expected or not all(row["finite"] for row in results):
        raise RuntimeError("L274 result matrix is incomplete")
    status, selected, comparisons = _decision(results, section)
    flat_results = [{
        **{key: value for key, value in row.items() if key != "scene_pair_accuracy"},
        "scene_pair_accuracy": json.dumps(row["scene_pair_accuracy"], sort_keys=True),
    } for row in results]
    flat_comparisons = [{
        **{key: value for key, value in row.items() if key != "scene_pair_changes"},
        "scene_pair_changes": json.dumps(row["scene_pair_changes"], sort_keys=True),
    } for row in comparisons]
    _write_csv(output / "results.csv", flat_results)
    _write_csv(output / "comparisons.csv", flat_comparisons)
    _write_csv(output / "predictions.csv", predictions_out)
    _write_csv(output / "metrics_by_state.csv", states_out)
    summary = {
        "protocol": PROTOCOL, "status": "complete", "decision": status,
        "selected_diagnostic_arm": selected, "actor_training_authorized": False,
        "git_sha": git_sha(ROOT), "config_sha256": _sha256(config_path),
        "paired_seed_count": len(section["design"]["paired_seeds"]),
        "arm_count": len(arms), "crossfit_folds": section["cross_fit"]["folds"],
        "source_states": len({row["state_id"] for row in source_rows}),
        "recovery_states": len({row["state_id"] for row in recovery_rows}),
        "comparisons": comparisons, "all_finite": True,
    }
    _json_dump(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
