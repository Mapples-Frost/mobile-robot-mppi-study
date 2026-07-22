#!/usr/bin/env python3
"""Run the preregistered read-only L271 Critic scene-interference audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.rl.run_l268_critic_only_intervention import (
    _build_treatment,
    _frozen_hashes,
    _load_agent,
    _recovery_pool,
    _replay_arrays,
    _tree_hash,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.sac import quantile_huber_loss


PROTOCOL = "L271"
DEFAULT_CONFIG = ROOT / "configs/rl/l271_critic_scene_interference_diagnosis.yaml"
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l271_critic_scene_interference_diagnosis"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str):
    actual = _sha256(path)
    if actual != str(expected).lower():
        raise ValueError("SHA256 mismatch for %s: %s" % (path, actual))
    return actual


def _write_csv(path: Path, rows):
    rows = list(rows)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload["l271"]
    if section["protocol"] != PROTOCOL or section["conditional_training"] is not False:
        raise ValueError("L271 must remain a read-only diagnostic")
    references = json.dumps({
        key: value for key, value in section.items() if key != "forbidden_tokens"
    }, sort_keys=True).lower()
    entered = [
        token for token in section["forbidden_tokens"] if token.lower() in references
    ]
    if entered:
        raise ValueError("forbidden artifact entered L271 config: %s" % entered)
    return payload, section


def _array_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)


def _verify_embedded_replay(embedded, expected, scene_count):
    names = (
        "observations", "actions", "rewards", "constraint_costs",
        "next_observations", "dones", "groups", "outcomes", "transition_ids",
    )
    for name in names:
        if not _array_equal(embedded[name], expected[name]):
            raise ValueError("embedded L268 treatment replay drifted: %s" % name)
    groups, counts = np.unique(embedded["groups"], return_counts=True)
    if groups.tolist() != list(range(int(scene_count))) or not np.all(counts == 1000):
        raise ValueError("L268 treatment replay is not 1000 rows per scene")
    if not all(np.isfinite(embedded[name]).all() for name in names):
        raise FloatingPointError("L268 treatment replay contains NaN or Inf")


def _selected_indices(groups, scene_count, rows_per_scene, seed):
    rng = np.random.RandomState(int(seed))
    chunks = []
    for scene in range(int(scene_count)):
        candidates = np.flatnonzero(np.asarray(groups).reshape(-1) == scene)
        if candidates.size < int(rows_per_scene):
            raise ValueError("insufficient rows for L271 scene block")
        chunks.append(np.sort(rng.choice(
            candidates, size=int(rows_per_scene), replace=False,
        )))
    return np.concatenate(chunks).astype(np.int64)


def _fixed_tensors(agent, normalizer, arrays, indices, actor_noise_seed):
    observation = normalizer.normalize(arrays["observations"][indices])
    next_observation = normalizer.normalize(arrays["next_observations"][indices])
    tensors = {
        "observations": torch.as_tensor(
            observation, dtype=torch.float32, device=agent.device,
        ),
        "actions": torch.as_tensor(
            arrays["actions"][indices], dtype=torch.float32, device=agent.device,
        ),
        "rewards": torch.as_tensor(
            arrays["rewards"][indices], dtype=torch.float32, device=agent.device,
        ),
        "next_observations": torch.as_tensor(
            next_observation, dtype=torch.float32, device=agent.device,
        ),
        "dones": torch.as_tensor(
            arrays["dones"][indices], dtype=torch.float32, device=agent.device,
        ),
    }
    torch.manual_seed(int(actor_noise_seed))
    if agent.device.type == "cuda":
        torch.cuda.manual_seed_all(int(actor_noise_seed))
    with torch.no_grad():
        next_action, next_log_probability, _, _, _ = agent._sample_policy(
            tensors["next_observations"]
        )
        next_distribution = torch.minimum(
            agent.target_critic1(tensors["next_observations"], next_action),
            agent.target_critic2(tensors["next_observations"], next_action),
        )
        tensors["targets"] = tensors["rewards"] + (
            (1.0 - tensors["dones"]) * float(agent.config.gamma)
            * (next_distribution - agent.alpha.detach() * next_log_probability)
        )
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise FloatingPointError("L271 fixed tensors contain NaN or Inf")
    return tensors


def _linear_layers(critic):
    layers = [module for module in critic.network if isinstance(module, nn.Linear)]
    if len(layers) < 2:
        raise ValueError("L271 requires a multi-layer Critic")
    return layers[0], layers[-1]


def _gradient_views(agent, tensors, selected):
    observation = tensors["observations"][selected]
    action = tensors["actions"][selected]
    target = tensors["targets"][selected]
    q1 = agent.critic1(observation, action)
    q2 = agent.critic2(observation, action)
    if agent.is_quantile_critic:
        loss1 = quantile_huber_loss(
            q1, target, agent.config.critic_quantile_huber_kappa,
        )
        loss2 = quantile_huber_loss(
            q2, target, agent.config.critic_quantile_huber_kappa,
        )
    else:
        loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
    loss = 0.5 * (loss1 + loss2)
    parameters = list(agent.critic1.parameters()) + list(agent.critic2.parameters())
    gradients = torch.autograd.grad(loss, parameters, create_graph=False)
    by_parameter = {id(parameter): gradient.detach() for parameter, gradient in zip(
        parameters, gradients
    )}
    first1, last1 = _linear_layers(agent.critic1)
    first2, last2 = _linear_layers(agent.critic2)

    def flatten(values):
        return torch.cat([value.reshape(-1) for value in values])

    observation_dim = int(agent.observation_dim)
    views = {
        "all_parameters": flatten(gradients),
        "input_observation_columns": flatten((
            by_parameter[id(first1.weight)][:, :observation_dim],
            by_parameter[id(first2.weight)][:, :observation_dim],
        )),
        "input_action_columns": flatten((
            by_parameter[id(first1.weight)][:, observation_dim:],
            by_parameter[id(first2.weight)][:, observation_dim:],
        )),
        "output_layer": flatten((
            by_parameter[id(last1.weight)], by_parameter[id(last1.bias)],
            by_parameter[id(last2.weight)], by_parameter[id(last2.bias)],
        )),
    }
    if not bool(torch.isfinite(loss)) or not all(
        bool(torch.isfinite(value).all()) for value in views.values()
    ):
        raise FloatingPointError("L271 gradient contains NaN or Inf")
    return views, float(loss.detach().cpu())


def _cosine_summary(vectors):
    matrix = torch.stack([value.to(dtype=torch.float64) for value in vectors])
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if bool(torch.any(norms <= 0.0)):
        raise FloatingPointError("L271 encountered a zero gradient")
    normalized = matrix / norms[:, None]
    cosine = normalized @ normalized.T
    pairs = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            pairs.append((left, right, float(cosine[left, right].cpu())))
    values = np.asarray([row[2] for row in pairs], dtype=np.float64)
    cancellation = 1.0 - float(
        torch.linalg.vector_norm(matrix.sum(dim=0)).cpu()
        / torch.sum(norms).cpu()
    )
    summary = {
        "median_off_diagonal_cosine": float(np.median(values)),
        "mean_off_diagonal_cosine": float(np.mean(values)),
        "negative_pair_fraction": float(np.mean(values < 0.0)),
        "cancellation_ratio": cancellation,
        "minimum_gradient_norm": float(torch.min(norms).cpu()),
        "maximum_gradient_norm": float(torch.max(norms).cpu()),
        "gradient_norm_ratio": float((torch.max(norms) / torch.min(norms)).cpu()),
    }
    if not np.isfinite(np.asarray(list(summary.values()), dtype=np.float64)).all():
        raise FloatingPointError("L271 cosine summary contains NaN or Inf")
    return pairs, summary


def _partition(agent, tensors, labels, layer_names):
    gradients = {name: [] for name in layer_names}
    losses = []
    for label in range(6):
        selected = torch.as_tensor(
            np.flatnonzero(np.asarray(labels) == label),
            dtype=torch.long, device=agent.device,
        )
        views, loss = _gradient_views(agent, tensors, selected)
        losses.append(loss)
        for name in layer_names:
            gradients[name].append(views[name])
    summaries, pairs = {}, {}
    for name in layer_names:
        pairs[name], summaries[name] = _cosine_summary(gradients[name])
    return summaries, pairs, losses


def _gate_for_seed(real, null_rows, gate):
    action_real = real["input_action_columns"]["median_off_diagonal_cosine"]
    cancellation_real = real["all_parameters"]["cancellation_ratio"]
    action_null = np.asarray([
        row["input_action_columns"]["median_off_diagonal_cosine"]
        for row in null_rows
    ], dtype=np.float64)
    cancellation_null = np.asarray([
        row["all_parameters"]["cancellation_ratio"] for row in null_rows
    ], dtype=np.float64)
    action_q05 = float(np.quantile(action_null, float(gate["null_lower_quantile"])))
    cancellation_q95 = float(np.quantile(
        cancellation_null, float(gate["null_upper_quantile"])
    ))
    action_effect = float(np.median(action_null) - action_real)
    cancellation_effect = float(cancellation_real - np.median(cancellation_null))
    norm_ratio = float(real["all_parameters"]["gradient_norm_ratio"])
    checks = {
        "action_cosine_below_null_q05": action_real <= action_q05,
        "cancellation_above_null_q95": cancellation_real >= cancellation_q95,
        "action_cosine_effect": action_effect >= float(
            gate["minimum_action_input_median_cosine_effect"]
        ),
        "cancellation_effect": cancellation_effect >= float(
            gate["minimum_all_parameter_cancellation_effect"]
        ),
        "gradient_norm_ratio": norm_ratio <= float(gate["maximum_gradient_norm_ratio"]),
    }
    return {
        "real_action_input_median_cosine": action_real,
        "null_action_input_q05": action_q05,
        "action_input_median_cosine_effect": action_effect,
        "real_all_parameter_cancellation": cancellation_real,
        "null_all_parameter_cancellation_q95": cancellation_q95,
        "all_parameter_cancellation_effect": cancellation_effect,
        "all_parameter_gradient_norm_ratio": norm_ratio,
        "checks": checks,
        "seed_block_pass": all(checks.values()),
    }


def run(config_path: Path, output: Path):
    _, section = _load_config(config_path)
    l268 = section["l268"]
    source_path = ROOT / section["source_checkpoint"]
    config268_path = ROOT / l268["config"]
    dataset = ROOT / l268["dataset"]
    manifest_path = dataset / "manifest.json"
    critic_dir = ROOT / l268["critic_dir"]
    _verify(source_path, section["source_checkpoint_sha256"])
    _verify(config268_path, l268["config_sha256"])
    _verify(manifest_path, l268["dataset_manifest_sha256"])
    _verify(critic_dir / "integrity.json", l268["critic_integrity_sha256"])
    _verify(critic_dir / "results.csv", l268["critic_results_sha256"])
    l270_path = ROOT / section["l270"]["summary"]
    _verify(l270_path, section["l270"]["summary_sha256"])
    l270 = json.loads(l270_path.read_text(encoding="utf-8"))
    if l270["decision"] != section["l270"]["required_decision"]:
        raise ValueError("L271 requires the frozen unresolved L270 decision")

    config268 = yaml.safe_load(config268_path.read_text(encoding="utf-8"))["l268"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_payload = load_sac_checkpoint(source_path, map_location="cpu")
    source_replay = _replay_arrays(source_payload)
    recovery, _ = _recovery_pool(dataset, manifest)
    audit = section["audit"]
    layer_names = list(audit["layers"])
    metric_rows, pair_rows, gate_rows = [], [], []
    all_unchanged = True

    for checkpoint_spec in l268["treatment_checkpoints"]:
        seed = int(checkpoint_spec["seed"])
        checkpoint_path = ROOT / checkpoint_spec["path"]
        _verify(checkpoint_path, checkpoint_spec["sha256"])
        expected_replay, _ = _build_treatment(
            source_replay, recovery, config268, seed,
        )
        payload = load_sac_checkpoint(checkpoint_path, map_location=audit["device"])
        embedded_replay = _replay_arrays(payload)
        _verify_embedded_replay(
            embedded_replay, expected_replay, audit["scene_count"],
        )
        agent, normalizer = _load_agent(payload, audit["device"], seed)
        agent.eval()
        before = _tree_hash(agent.state_dict())
        frozen_actor_before = _frozen_hashes(agent)
        indices = _selected_indices(
            embedded_replay["groups"], audit["scene_count"],
            audit["rows_per_scene"], int(audit["sample_seed"]) + seed,
        )
        tensors = _fixed_tensors(
            agent, normalizer, embedded_replay, indices,
            int(audit["actor_noise_seed"]) + seed,
        )
        real_labels = np.repeat(
            np.arange(int(audit["scene_count"]), dtype=np.int64),
            int(audit["rows_per_scene"]),
        )
        real, real_pairs, real_losses = _partition(
            agent, tensors, real_labels, layer_names,
        )
        null_results = []
        rng = np.random.RandomState(int(audit["null_seed"]) + seed)
        for partition in range(int(audit["null_partitions"])):
            shuffled = rng.permutation(real_labels)
            null, null_pairs, null_losses = _partition(
                agent, tensors, shuffled, layer_names,
            )
            null_results.append(null)
            for layer in layer_names:
                metric_rows.append({
                    "seed": seed, "partition_type": "null",
                    "partition": partition, "layer": layer,
                    "mean_critic_loss": float(np.mean(null_losses)),
                    **null[layer],
                })
                for left, right, cosine in null_pairs[layer]:
                    pair_rows.append({
                        "seed": seed, "partition_type": "null",
                        "partition": partition, "layer": layer,
                        "left_group": left, "right_group": right,
                        "cosine": cosine,
                    })
        for layer in layer_names:
            metric_rows.append({
                "seed": seed, "partition_type": "real", "partition": -1,
                "layer": layer, "mean_critic_loss": float(np.mean(real_losses)),
                **real[layer],
            })
            for left, right, cosine in real_pairs[layer]:
                pair_rows.append({
                    "seed": seed, "partition_type": "real", "partition": -1,
                    "layer": layer, "left_group": left, "right_group": right,
                    "cosine": cosine,
                })
        gate_row = _gate_for_seed(real, null_results, section["gate"])
        gate_rows.append({"seed": seed, **gate_row})
        after = _tree_hash(agent.state_dict())
        unchanged = before == after and frozen_actor_before == _frozen_hashes(agent)
        all_unchanged = all_unchanged and unchanged
        if not unchanged:
            raise RuntimeError("L271 mutated a frozen network or optimizer")
        print(json.dumps({"completed_seed": seed, "frozen_state_unchanged": True}))
        del agent, tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passed_blocks = sum(row["seed_block_pass"] for row in gate_rows)
    finite = all(np.isfinite(float(row[key])) for row in metric_rows for key in (
        "mean_critic_loss", "median_off_diagonal_cosine",
        "mean_off_diagonal_cosine", "negative_pair_fraction",
        "cancellation_ratio", "minimum_gradient_norm",
        "maximum_gradient_norm", "gradient_norm_ratio",
    ))
    gate_pass = bool(
        finite and all_unchanged and passed_blocks >= int(
            section["gate"]["minimum_seed_blocks_both_extreme"]
        )
    )
    decision = section["next_if_pass"] if gate_pass else section["next_if_fail"]
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "gradient_partition_metrics.csv", metric_rows)
    _write_csv(output / "gradient_cosine_pairs.csv", pair_rows)
    _write_csv(output / "gate_by_seed.csv", [{
        **{key: value for key, value in row.items() if key != "checks"},
        "checks": json.dumps(row["checks"], sort_keys=True),
    } for row in gate_rows])
    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "critic_scene_interference_gate_pass": gate_pass,
        "decision": decision,
        "conditional_training_authorized": False,
        "passed_seed_blocks": int(passed_blocks),
        "total_seed_blocks": len(gate_rows),
        "finite": finite,
        "frozen_state_unchanged": all_unchanged,
        "device": str(audit["device"]),
        "selected_rows_per_seed": int(audit["scene_count"]) * int(
            audit["rows_per_scene"]
        ),
        "null_partitions_per_seed": int(audit["null_partitions"]),
        "gate_by_seed": gate_rows,
        "git_sha": git_sha(ROOT),
        "config_sha256": _sha256(config_path),
        "l270_decision": l270["decision"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
