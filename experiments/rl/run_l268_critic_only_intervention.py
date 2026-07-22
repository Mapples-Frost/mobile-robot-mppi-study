#!/usr/bin/env python3
"""Run the preregistered paired L268 Critic-only replay intervention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l264_l266_critic_failure_diagnosis import (
    _replay_arrays,
    _support_metrics,
    _transition_diagnostics,
)
from experiments.rl.run_l263_counterfactual_actor_diagnosis import _spearman
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.replay import ReplayBuffer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


DEFAULT_CONFIG = ROOT / "configs/rl/l268_recovery_balanced_intervention.yaml"
DEFAULT_DATASET = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/"
    "recovery_dataset"
)
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/"
    "critic_only"
)
DEFAULT_HELDOUT = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/"
    "heldout_recovery_diagnostic"
)
AUDIT_CSV = ROOT / (
    "results/research_platform/rl/l264_l266_critic_failure_diagnosis/"
    "l266_transition_audit.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _tree_hash(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array")
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda value: str(value)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _frozen_hashes(agent):
    return {
        "actor": _tree_hash(agent.actor.state_dict()),
        "actor_optimizer": _tree_hash(agent.actor_optimizer.state_dict()),
        "log_alpha": _tree_hash(agent.log_alpha),
        "alpha_optimizer": _tree_hash(agent.alpha_optimizer.state_dict()),
    }


def _load_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(payload["l268"])
    checkpoint = ROOT / config["source_checkpoint"]
    if _sha256(checkpoint) != config["source_checkpoint_sha256"]:
        raise ValueError("L268 source checkpoint SHA256 mismatch")
    manifest_path = DEFAULT_DATASET / "manifest.json"
    return payload, config, checkpoint, manifest_path


def _student_fields_ok(payload):
    return set(payload.files) == {
        "observations", "actions", "rewards", "constraint_costs",
        "next_observations", "dones", "groups", "chain_ids", "steps",
    }


def _recovery_pool(dataset: Path, manifest):
    fields = defaultdict(list)
    provenance = defaultdict(list)
    chains = [row for row in manifest["chains"] if row["split"] == "train"]
    for record in chains:
        shard = dataset / record["student_npz"]
        with np.load(shard, allow_pickle=False) as payload:
            if not _student_fields_ok(payload):
                raise ValueError("recovery shard violates student field whitelist")
            count = int(payload["actions"].shape[0])
            if payload["observations"].shape != (count, 69):
                raise ValueError("recovery shard is not 69D")
            if not all(np.isfinite(payload[name]).all() for name in payload.files):
                raise FloatingPointError("recovery shard contains NaN or Inf")
            for name in (
                "observations", "actions", "rewards", "constraint_costs",
                "next_observations", "dones", "groups", "chain_ids",
            ):
                fields[(int(record["scene_index"]), name)].append(payload[name].copy())
            provenance[int(record["scene_index"])].append({
                "chain_id": int(record["chain_id"]),
                "attempt_id": record["attempt_id"],
                "split": record["split"],
                "transition_count": count,
                "sha256": _sha256(shard),
            })
    result = {}
    for group in range(6):
        result[group] = {
            name: np.concatenate(fields[(group, name)], axis=0)
            for name in (
                "observations", "actions", "rewards", "constraint_costs",
                "next_observations", "dones", "groups", "chain_ids",
            )
        }
    return result, provenance


def _original_nonrecovery_indices(replay):
    audit = _read_csv(AUDIT_CSV)
    stage_by_id = {int(row["transition_id"]): row["stage"] for row in audit}
    indices = defaultdict(list)
    for index, transition_id in enumerate(replay["transition_ids"]):
        if stage_by_id[int(transition_id)] != "recovery":
            indices[int(replay["groups"][index])].append(index)
    return {group: np.asarray(values, dtype=np.int64) for group, values in indices.items()}


def _sample_chain_balanced(pool, count, rng):
    chain_ids = np.unique(pool["chain_ids"])
    if chain_ids.size <= 0:
        raise ValueError("recovery pool contains no chain IDs")
    base, remainder = divmod(int(count), int(chain_ids.size))
    order = rng.permutation(chain_ids.size)
    quotas = np.full(chain_ids.size, base, dtype=np.int64)
    quotas[order[:remainder]] += 1
    selected = []
    multiplicities = {}
    for chain_id, quota in zip(chain_ids, quotas):
        candidates = np.flatnonzero(pool["chain_ids"] == chain_id)
        picked = rng.choice(candidates, size=int(quota), replace=quota > candidates.size)
        selected.append(picked)
        multiplicities[str(int(chain_id))] = {
            "source_count": int(candidates.size),
            "sample_count": int(quota),
            "unique_sample_count": int(np.unique(picked).size),
        }
    indices = np.concatenate(selected).astype(np.int64)
    return indices[rng.permutation(indices.size)], multiplicities


def _build_treatment(replay, recovery, config, seed):
    rng = np.random.RandomState(int(seed))
    nonrecovery = _original_nonrecovery_indices(replay)
    chunks = defaultdict(list)
    provenance = {"seed": int(seed), "scenes": {}}
    recovery_count = int(config["replay"]["recovery_transitions_per_scene"])
    original_count = int(config["replay"]["original_transitions_per_scene"])
    source_names = (
        "observations", "actions", "rewards", "constraint_costs",
        "next_observations", "dones", "groups",
    )
    for group in range(6):
        recovery_indices, chain_multiplicity = _sample_chain_balanced(
            recovery[group], recovery_count, rng,
        )
        candidates = nonrecovery[group]
        original_indices = rng.choice(
            candidates, size=original_count,
            replace=original_count > candidates.size,
        )
        for name in source_names:
            chunks[name].append(np.concatenate((
                recovery[group][name][recovery_indices],
                replay[name][original_indices],
            ), axis=0))
        provenance["scenes"][str(group)] = {
            "recovery_source_count": int(recovery[group]["actions"].shape[0]),
            "recovery_sample_count": recovery_count,
            "original_nonrecovery_source_count": int(candidates.size),
            "original_sample_count": original_count,
            "original_unique_sample_count": int(np.unique(original_indices).size),
            "chain_multiplicity": chain_multiplicity,
        }
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    permutation = rng.permutation(arrays["actions"].shape[0])
    arrays = {name: value[permutation] for name, value in arrays.items()}
    arrays["outcomes"] = np.full((permutation.size,), -1, dtype=np.int8)
    arrays["transition_ids"] = np.arange(permutation.size, dtype=np.int64)
    if arrays["actions"].shape != (6000, 2):
        raise RuntimeError("L268 treatment replay must contain exactly 6000 rows")
    return arrays, provenance


def _buffer_from_arrays(arrays, seed):
    count = int(arrays["actions"].shape[0])
    buffer = ReplayBuffer(count, 69, 2, seed=seed)
    for name in (
        "observations", "actions", "rewards", "constraint_costs",
        "next_observations", "dones", "groups", "outcomes", "transition_ids",
    ):
        getattr(buffer, name)[:count] = arrays[name]
    buffer.size = count
    buffer.position = 0
    buffer.next_transition_id = count
    return buffer


def _load_agent(payload, device, seed):
    state = payload["agent"]
    agent = SACAgent(
        int(state["observation_dim"]), int(state["action_dim"]),
        SACConfig.from_mapping(state["config"]), device=device, seed=seed,
    )
    agent.load_state_dict(state, load_optimizers=True)
    agent.train()
    normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
    return agent, normalizer


def _normalized_batch(buffer, normalizer, batch_size):
    batch = buffer.sample(batch_size, strategy="scene_balanced")
    batch["observations"] = normalizer.normalize(batch["observations"])
    batch["next_observations"] = normalizer.normalize(batch["next_observations"])
    return batch


def _evaluate_l263(agent, config, output):
    l263 = ROOT / config["diagnostic"]["l263_dir"]
    states = json.loads((l263 / "state_manifest.json").read_text(encoding="utf-8"))
    state_by_id = {row["state_id"]: row for row in states}
    support_rows = _read_csv(ROOT / config["diagnostic"]["l265_support_csv"])
    grouped = defaultdict(list)
    for row in support_rows:
        grouped[row["state_id"]].append(row)
    updated = []
    agent.eval()
    with torch.no_grad():
        for state_id, rows in sorted(grouped.items()):
            observation = np.asarray(
                state_by_id[state_id]["normalized_observation"], dtype=np.float32,
            )
            observations = np.repeat(observation[None, :], len(rows), axis=0)
            actions = np.asarray([
                (float(row["normalized_v"]), float(row["normalized_omega"]))
                for row in rows
            ], dtype=np.float32)
            ot = torch.as_tensor(observations, device=agent.device)
            at = torch.as_tensor(actions, device=agent.device)
            q1 = agent.target_critic1(ot, at).mean(dim=-1)
            q2 = agent.target_critic2(ot, at).mean(dim=-1)
            minimum = torch.minimum(q1, q2).cpu().numpy()
            for row, value in zip(rows, minimum):
                item = dict(row)
                item["target_minimum_q"] = float(value)
                updated.append(item)
    per_state, pairs, summary = _support_metrics(updated, l263)
    _write_csv(output / "l263_action_ranking.csv", updated)
    _write_csv(output / "l263_metrics_by_state.csv", per_state)
    _write_csv(output / "l263_recovery_forward_pairs.csv", pairs)
    _write_csv(output / "l263_support_summary.csv", summary)
    return per_state, pairs, summary


def _evaluate_heldout(agent, heldout: Path, output: Path):
    summary_path = heldout / "summary.json"
    states_path = heldout / "state_manifest.json"
    returns_path = heldout / "action_returns_h40.csv"
    frozen = json.loads(summary_path.read_text(encoding="utf-8"))
    if frozen.get("protocol") != "L268" or frozen.get("status") != (
        "heldout_recovery_diagnostic_complete"
    ):
        raise ValueError("L268 held-out recovery diagnostic is not complete")
    if frozen.get("formal_critic_results_read_before_freeze") is not False:
        raise ValueError("held-out returns were not frozen before Critic results")
    if int(frozen.get("state_count", 0)) != 36:
        raise ValueError("L268 held-out diagnostic must contain 36 states")

    states = json.loads(states_path.read_text(encoding="utf-8"))
    state_by_id = {row["state_id"]: row for row in states}
    return_rows = _read_csv(returns_path)
    grouped = defaultdict(list)
    for row in return_rows:
        grouped[row["state_id"]].append(row)
    if set(grouped) != set(state_by_id):
        raise ValueError("held-out state/action coverage mismatch")

    ranking_rows = []
    pair_rows = []
    spearman_values = []
    scene_accuracy = defaultdict(list)
    agent.eval()
    with torch.no_grad():
        for state_id, rows in sorted(grouped.items()):
            if {row["action_id"] for row in rows} != {
                "recorded_recovery", "source_actor", "fast_forward"
            }:
                raise ValueError("held-out state action set drifted")
            observation = np.asarray(
                state_by_id[state_id]["normalized_observation"], dtype=np.float32,
            )
            observations = np.repeat(observation[None, :], len(rows), axis=0)
            actions = np.asarray([
                (float(row["normalized_v"]), float(row["normalized_omega"]))
                for row in rows
            ], dtype=np.float32)
            ot = torch.as_tensor(observations, device=agent.device)
            at = torch.as_tensor(actions, device=agent.device)
            q1 = agent.target_critic1(ot, at).mean(dim=-1)
            q2 = agent.target_critic2(ot, at).mean(dim=-1)
            minimum = torch.minimum(q1, q2).cpu().numpy()
            for row, value in zip(rows, minimum):
                item = dict(row)
                item["target_minimum_q"] = float(value)
                ranking_rows.append(item)
            returns = np.asarray([
                float(row["discounted_return"]) for row in rows
            ], dtype=np.float64)
            correlation = _spearman(minimum, returns)
            if correlation is not None:
                spearman_values.append(correlation)
            by_action = {
                row["action_id"]: (float(row["discounted_return"]), float(value))
                for row, value in zip(rows, minimum)
            }
            return_delta = (
                by_action["recorded_recovery"][0] - by_action["fast_forward"][0]
            )
            q_delta = (
                by_action["recorded_recovery"][1] - by_action["fast_forward"][1]
            )
            correct = bool(
                (return_delta > 0.0 and q_delta > 0.0)
                or (return_delta < 0.0 and q_delta < 0.0)
                or (abs(return_delta) <= 1e-12 and abs(q_delta) <= 1e-12)
            )
            state = state_by_id[state_id]
            scene_accuracy[state["scene"]].append(float(correct))
            pair_rows.append({
                "state_id": state_id,
                "scene": state["scene"],
                "split": state["split"],
                "severity": state["severity"],
                "side": state["side"],
                "true_return_delta_recovery_minus_forward": return_delta,
                "critic_q_delta_recovery_minus_forward": q_delta,
                "ranking_correct": correct,
                "true_recovery_preferred": return_delta > 0.0,
                "critic_recovery_preferred": q_delta > 0.0,
                "three_action_spearman": correlation,
            })

    summary = {
        "state_count": len(grouped),
        "mean_three_action_spearman": (
            None if not spearman_values else float(np.mean(spearman_values))
        ),
        "recovery_forward_pair_accuracy": float(np.mean([
            float(row["ranking_correct"]) for row in pair_rows
        ])),
        "true_recovery_preferred_fraction": float(np.mean([
            float(row["true_recovery_preferred"]) for row in pair_rows
        ])),
        "critic_recovery_preferred_fraction": float(np.mean([
            float(row["critic_recovery_preferred"]) for row in pair_rows
        ])),
        "scene_pair_accuracy": {
            scene: float(np.mean(values))
            for scene, values in sorted(scene_accuracy.items())
        },
        "heldout_summary_sha256": _sha256(summary_path),
        "heldout_states_sha256": _sha256(states_path),
        "heldout_returns_sha256": _sha256(returns_path),
    }
    _write_csv(output / "heldout_action_ranking.csv", ranking_rows)
    _write_csv(output / "heldout_recovery_forward_pairs.csv", pair_rows)
    _json_dump(output / "heldout_summary.json", summary)
    return pair_rows, summary


def _save_checkpoint(path, source_payload, agent, normalizer, buffer, state):
    payload = dict(source_payload)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["git_sha"] = git_sha(ROOT)
    payload["agent"] = agent.state_dict()
    payload["normalizer"] = normalizer.state_dict()
    payload["training_state"] = dict(state)
    payload["replay_buffer"] = buffer.state_dict(include_data=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return _sha256(path)


def run(
    config_path: Path, dataset: Path, heldout: Path, output: Path, device: str,
):
    _, config, checkpoint, manifest_path = _load_config(config_path)
    if dataset.resolve() != DEFAULT_DATASET.resolve():
        manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "L268" or manifest.get("status") != "complete":
        raise ValueError("L268 Critic intervention requires a complete formal dataset")
    if manifest["accepted_chain_count"] != 108:
        raise ValueError("L268 formal dataset must contain exactly 108 chains")
    heldout_frozen = json.loads(
        (heldout / "summary.json").read_text(encoding="utf-8")
    )
    if heldout_frozen.get("status") != "heldout_recovery_diagnostic_complete":
        raise ValueError("formal Critic training requires frozen held-out H40 returns")
    source_payload = load_sac_checkpoint(checkpoint, map_location=device)
    replay = _replay_arrays(source_payload)
    recovery, chain_provenance = _recovery_pool(dataset, manifest)
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "recovery_chain_provenance.json", chain_provenance)
    seeds = [int(value) for value in config["critic_only"]["seeds"]]
    order_rng = np.random.RandomState(int(config["critic_only"]["arm_order_seed"]))
    arm_orders = {
        str(seed): [str(value) for value in order_rng.permutation(
            np.asarray(("control", "recovery_balanced"), dtype=object)
        )]
        for seed in seeds
    }
    _json_dump(output / "arm_orders.json", arm_orders)
    results = []
    for seed in seeds:
        treatment, treatment_provenance = _build_treatment(
            replay, recovery, config, seed,
        )
        _json_dump(
            output / ("seed_%d_treatment_replay_provenance.json" % seed),
            treatment_provenance,
        )
        for arm in arm_orders[str(seed)]:
            arrays = replay if arm == "control" else treatment
            buffer = _buffer_from_arrays(arrays, seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            agent, normalizer = _load_agent(source_payload, device, seed)
            frozen_before = _frozen_hashes(agent)
            run_dir = output / ("seed_%d" % seed) / arm
            update_rows = []
            for update in range(1, int(config["critic_only"]["updates"]) + 1):
                metrics = agent.update(
                    _normalized_batch(
                        buffer, normalizer, config["critic_only"]["batch_size"]
                    ),
                    update_actor=False,
                )
                if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
                    raise FloatingPointError("L268 Critic update produced NaN or Inf")
                update_rows.append({
                    "seed": seed, "arm": arm, "critic_update": update, **metrics,
                })
                if update % int(config["critic_only"]["checkpoint_interval"]) == 0:
                    frozen_now = _frozen_hashes(agent)
                    if frozen_now != frozen_before:
                        raise RuntimeError("Actor or alpha mutated during Critic-only training")
                    checkpoint_path = run_dir / "checkpoints" / (
                        "critic_update_%06d.pt" % update
                    )
                    digest = _save_checkpoint(
                        checkpoint_path, source_payload, agent, normalizer, buffer,
                        {
                            "protocol": "L268",
                            "arm": arm,
                            "critic_seed": seed,
                            "critic_updates": update,
                            "source_checkpoint": str(checkpoint),
                            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
                            "config_sha256": _sha256(config_path),
                            "dataset_manifest_sha256": _sha256(manifest_path),
                            "actor_alpha_hashes": frozen_now,
                            "actor_update_applied": False,
                            "replay_size": buffer.size,
                        },
                    )
                    _json_dump(run_dir / ("checkpoint_%06d.json" % update), {
                        "path": str(checkpoint_path), "sha256": digest,
                        "actor_alpha_hashes": frozen_now,
                    })
            _write_csv(run_dir / "updates.csv", update_rows)
            frozen_after = _frozen_hashes(agent)
            if frozen_after != frozen_before:
                raise RuntimeError("Actor or alpha mutation detected after Critic phase")
            per_state, pairs, support = _evaluate_l263(agent, config, run_dir)
            heldout_pairs, heldout_summary = _evaluate_heldout(
                agent, heldout, run_dir,
            )
            diagnostics = _transition_diagnostics(agent, normalizer, arrays)
            stability = {
                "finite": bool(all(np.isfinite(value).all() for value in diagnostics.values())),
                "mean_online_q": float(np.mean(diagnostics["online_q_mean"])),
                "mean_quantile_spread": float(np.mean(diagnostics["quantile_spread"])),
                "maximum_absolute_q": float(np.max(np.abs(diagnostics["online_q_mean"]))),
                "actor_alpha_unchanged": frozen_after == frozen_before,
            }
            _json_dump(run_dir / "stability.json", stability)
            in_support = next(row for row in support if row["support"] == "in_support")
            results.append({
                "seed": seed,
                "arm": arm,
                "in_support_spearman": in_support["mean_spearman"],
                "in_support_top3": in_support["mean_top3_agreement"],
                "in_support_pair_accuracy": in_support["recovery_forward_pair_accuracy"],
                "heldout_pair_accuracy": heldout_summary[
                    "recovery_forward_pair_accuracy"
                ],
                "heldout_mean_three_action_spearman": heldout_summary[
                    "mean_three_action_spearman"
                ],
                **stability,
            })
            _write_csv(output / "progress.csv", results)
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _write_csv(output / "results.csv", results)
    _json_dump(output / "integrity.json", {
        "status": "critic_only_complete_pending_gate_aggregation",
        "protocol": "L268",
        "result_rows": len(results),
        "paired_seed_count": len(seeds),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "config_sha256": _sha256(config_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "heldout_summary_sha256": _sha256(heldout / "summary.json"),
        "heldout_states_sha256": _sha256(heldout / "state_manifest.json"),
        "heldout_returns_sha256": _sha256(heldout / "action_returns_h40.csv"),
        "all_actor_alpha_unchanged": all(row["actor_alpha_unchanged"] for row in results),
    })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--heldout-diagnostic", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.config.resolve(), args.dataset.resolve(),
        args.heldout_diagnostic.resolve(), args.output_dir.resolve(), args.device,
    )


if __name__ == "__main__":
    main()
