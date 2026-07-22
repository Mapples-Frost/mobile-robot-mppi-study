#!/usr/bin/env python3
"""Frozen evaluation-only L280 Actor-gradient alignment diagnosis."""

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
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import _load_agent
from mobile_robot_mppi.rl.recovery_retention import load_split, sha256_file


DEFAULT_CONFIG = ROOT / "configs/rl/l280_anchor_gradient_alignment_diagnosis.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l280_anchor_gradient_alignment_diagnosis"


def _flatten_partition(agent, partition):
    values = []
    for name, parameter in agent.actor.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if partition == "all":
            values.append(gradient.reshape(-1))
        elif partition == "trunk" and not name.startswith("network.4"):
            values.append(gradient.reshape(-1))
        elif name == "network.4.weight":
            if partition == "v_mean":
                values.append(gradient[0].reshape(-1))
            elif partition == "omega_mean":
                values.append(gradient[1].reshape(-1))
            elif partition == "log_std":
                values.append(gradient[agent.action_dim:].reshape(-1))
        elif name == "network.4.bias":
            if partition == "v_mean":
                values.append(gradient[0:1])
            elif partition == "omega_mean":
                values.append(gradient[1:2])
            elif partition == "log_std":
                values.append(gradient[agent.action_dim:])
    if not values:
        raise RuntimeError("L280 Actor partition is empty: %s" % partition)
    return torch.cat(values).detach().cpu().numpy().astype(np.float64)


def _gradients(agent, loss, partitions):
    agent.actor.zero_grad(set_to_none=True)
    loss.backward()
    result = {name: _flatten_partition(agent, name) for name in partitions}
    agent.actor.zero_grad(set_to_none=True)
    return result


def _sac_gradients(agent, observations, partitions, torch_seed):
    torch.manual_seed(int(torch_seed))
    if agent.device.type == "cuda":
        torch.cuda.manual_seed_all(int(torch_seed))
    tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=agent.device
    )
    action, log_probability, _, _, _ = agent._sample_policy(tensor)
    q1 = agent.critic1(tensor, action)
    q2 = agent.critic2(tensor, action)
    q = agent._critic_risk_value(torch.minimum(q1, q2))
    loss = (agent.alpha.detach() * log_probability - q).mean()
    return _gradients(agent, loss, partitions), float(loss.detach().cpu())


def _anchor_gradients(agent, observations, actions, config, partitions):
    mean_loss, log_std_loss, _ = agent._behavior_cloning_losses(
        observations, actions, float(config["anchor_target_log_std"])
    )
    loss = (
        float(config["anchor_mean_weight"]) * mean_loss
        + float(config["anchor_log_std_weight"]) * log_std_loss
    )
    return _gradients(agent, loss, partitions), float(loss.detach().cpu())


def _comparison(first, second):
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    denominator = first_norm * second_norm
    cosine = 0.0 if denominator <= 0.0 else float(np.dot(first, second) / denominator)
    return {
        "cosine": cosine,
        "first_norm": first_norm,
        "second_norm": second_norm,
        "second_over_first_norm": (
            1.0e30 if first_norm <= 0.0 and second_norm > 0.0
            else (0.0 if first_norm <= 0.0 else second_norm / first_norm)
        ),
    }


def _decision(rows, config):
    gate = config["gate"]
    stage_rows = [row for row in rows if row["stage"] == "step6000"]
    components = ("trunk", "v_mean", "omega_mean")
    online_seed_counts = {}
    internal_seed_counts = {}
    dominance = {}
    for component in components:
        online_seed_counts[component] = 0
        internal_seed_counts[component] = 0
        ratios = []
        for seed in sorted({row["seed"] for row in stage_rows}):
            selected = [row for row in stage_rows if row["seed"] == seed]
            online = sum(
                row["sac_vs_combined_%s_cosine" % component]
                < float(gate["maximum_conflict_cosine"])
                and row["sac_vs_combined_%s_second_over_first_norm" % component]
                >= float(gate["minimum_conflict_norm_ratio"])
                for row in selected
            )
            internal = sum(
                row["recovery_vs_source_%s_cosine" % component]
                < float(gate["maximum_conflict_cosine"])
                for row in selected
            )
            online_seed_counts[component] += int(
                online >= int(gate["minimum_conflicted_scenes"])
            )
            internal_seed_counts[component] += int(
                internal >= int(gate["minimum_conflicted_scenes"])
            )
            ratios.extend(
                row["sac_vs_combined_%s_second_over_first_norm" % component]
                for row in selected
            )
        dominance[component] = float(np.median(ratios))
    online = [
        name for name, count in online_seed_counts.items()
        if count >= int(gate["minimum_conflicted_seeds"])
    ]
    internal = [
        name for name, count in internal_seed_counts.items()
        if count >= int(gate["minimum_conflicted_seeds"])
    ]
    dominant = [
        name for name, ratio in dominance.items()
        if ratio >= float(gate["minimum_dominance_norm_ratio"])
    ]
    if online and internal:
        decision = "online_and_internal_anchor_conflict"
    elif online:
        decision = "online_vs_anchor_conflict"
    elif internal:
        decision = "internal_anchor_conflict"
    elif dominant:
        decision = "anchor_dominance_without_directional_conflict"
    else:
        decision = "gradient_conflict_not_supported"
    return decision, {
        "online_conflicted_seed_counts": online_seed_counts,
        "internal_conflicted_seed_counts": internal_seed_counts,
        "median_combined_anchor_over_sac_norm": dominance,
        "online_conflicted_components": online,
        "internal_conflicted_components": internal,
        "dominant_components": dominant,
    }


def run(config_path, output, device="cuda"):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    summary_path = ROOT / config["l279_summary"]["path"]
    if sha256_file(summary_path) != config["l279_summary"]["sha256"]:
        raise ValueError("L280 L279 summary SHA256 mismatch")
    l279 = json.loads(summary_path.read_text(encoding="utf-8"))
    if l279["decision"] != "retention_anchor_gate_fail":
        raise ValueError("L280 requires the frozen negative L279 Gate")
    dataset = ROOT / config["anchor_dataset"]
    if sha256_file(dataset / "manifest.json") != config["anchor_manifest_sha256"]:
        raise ValueError("L280 anchor manifest SHA256 mismatch")
    references = "\n".join(
        [config["anchor_dataset"]]
        + [stage["path"] for row in config["checkpoints"] for stage in (row["step3000"], row["step6000"])]
    ).lower()
    if any(token.lower() in references for token in config["forbidden_tokens"]):
        raise ValueError("forbidden artifact entered L280")
    anchor = load_split(dataset, "train")
    recovery_indices = np.flatnonzero(anchor["source_kinds"] == 0)
    source_indices = np.flatnonzero(anchor["source_kinds"] == 1)
    batch_size = int(config["batch_size"])
    if len(recovery_indices) != len(source_indices) or len(recovery_indices) != 9216:
        raise ValueError("L280 anchor balance contract drifted")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    progress = []
    partitions = tuple(config["partitions"])
    for run_index, run_spec in enumerate(config["checkpoints"]):
        seed = int(run_spec["seed"])
        for stage_index, stage in enumerate(("step3000", "step6000")):
            checkpoint = ROOT / run_spec[stage]["path"]
            if sha256_file(checkpoint) != run_spec[stage]["sha256"]:
                raise ValueError("L280 checkpoint SHA256 mismatch")
            payload, agent, normalizer = _load_agent(checkpoint, device)
            replay = payload["replay_buffer"]
            size = int(replay["size"])
            observations = np.asarray(replay["observations"][:size], dtype=np.float32)
            groups = np.asarray(replay["groups"][:size], dtype=np.int64)
            if set(np.unique(groups)) != set(range(6)):
                raise ValueError("L280 replay does not cover all six scenes")
            for scene in range(6):
                rng = np.random.RandomState(
                    int(config["diagnostic_seed"]) + run_index * 10000
                    + stage_index * 1000 + scene
                )
                scene_pool = np.flatnonzero(groups == scene)
                scene_indices = rng.choice(scene_pool, size=batch_size, replace=True)
                recovery_sample = rng.choice(recovery_indices, size=batch_size, replace=True)
                source_sample = rng.choice(source_indices, size=batch_size, replace=True)
                half = batch_size // 2
                combined_sample = np.concatenate((
                    rng.choice(recovery_indices, size=half, replace=True),
                    rng.choice(source_indices, size=batch_size - half, replace=True),
                ))
                normalized_scene = normalizer.normalize(observations[scene_indices])
                sac_grad, sac_loss = _sac_gradients(
                    agent, normalized_scene, partitions,
                    int(config["diagnostic_seed"]) + run_index * 10000
                    + stage_index * 1000 + scene,
                )
                gradients = {"sac": sac_grad}
                losses = {"sac": sac_loss}
                for name, indices in (
                    ("recovery", recovery_sample),
                    ("source", source_sample),
                    ("combined", combined_sample),
                ):
                    grad, loss = _anchor_gradients(
                        agent,
                        normalizer.normalize(anchor["observations"][indices]),
                        anchor["teacher_actions"][indices],
                        config, partitions,
                    )
                    gradients[name] = grad
                    losses[name] = loss
                row = {
                    "seed": seed, "stage": stage, "scene_group": scene,
                    "sac_loss": losses["sac"],
                    "recovery_loss": losses["recovery"],
                    "source_loss": losses["source"],
                    "combined_loss": losses["combined"],
                }
                for partition in partitions:
                    for first, second in (
                        ("sac", "recovery"), ("sac", "source"),
                        ("sac", "combined"), ("recovery", "source"),
                    ):
                        metrics = _comparison(
                            gradients[first][partition], gradients[second][partition]
                        )
                        for key, value in metrics.items():
                            row["%s_vs_%s_%s_%s" % (
                                first, second, partition, key
                            )] = value
                numeric = np.asarray([
                    value for value in row.values()
                    if isinstance(value, (float, np.floating))
                ])
                if not np.isfinite(numeric).all():
                    raise FloatingPointError("L280 produced non-finite diagnostics")
                rows.append(row)
            progress.append({
                "seed": seed, "stage": stage, "completed_blocks": len(progress) + 1,
                "expected_blocks": 6, "device": device,
            })
            _write_csv(output / "progress.csv", progress)
            print(json.dumps(progress[-1], sort_keys=True), flush=True)
    decision, metrics = _decision(rows, config)
    summary = {
        "protocol": "L280", "status": "complete", "decision": decision,
        "metrics": metrics, "row_count": len(rows), "all_finite": True,
        "actor_training_authorized": False,
        "final_map_evaluation_authorized": False,
    }
    _write_csv(output / "gradient_rows.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(Path(args.config).resolve(), Path(args.output_dir).resolve(), args.device)


if __name__ == "__main__":
    main()
