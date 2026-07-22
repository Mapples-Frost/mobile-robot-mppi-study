#!/usr/bin/env python3
"""Run the preregistered L272 equal-compute Critic architecture probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import _spearman
from experiments.rl.run_l268_critic_only_intervention import (
    _buffer_from_arrays,
    _build_treatment,
    _frozen_hashes,
    _load_agent,
    _normalized_batch,
    _recovery_pool,
    _replay_arrays,
    _save_checkpoint,
)
from experiments.rl.run_l271_critic_scene_interference_diagnosis import (
    _verify_embedded_replay,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.sac import QNetwork, quantile_huber_loss


PROTOCOL = "L272"
DEFAULT_CONFIG = ROOT / "configs/rl/l272_critic_architecture_causal_probe.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l272_critic_architecture_causal_probe"


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


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload["l272"]
    if section["protocol"] != PROTOCOL or section["actor_training_authorized"] is not False:
        raise ValueError("L272 must remain Critic-only")
    references = json.dumps({
        key: value for key, value in section.items() if key != "forbidden_tokens"
    }, sort_keys=True).lower()
    entered = [
        token for token in section["forbidden_tokens"] if token.lower() in references
    ]
    if entered:
        raise ValueError("forbidden artifact entered L272 config: %s" % entered)
    return payload, section


def _filter_scene(arrays, scene):
    mask = np.asarray(arrays["groups"]).reshape(-1) == int(scene)
    output = {
        name: np.asarray(value)[mask].copy() for name, value in arrays.items()
    }
    if output["actions"].shape != (1000, 2):
        raise ValueError("L272 per-scene replay must contain exactly 1000 rows")
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("L272 per-scene replay contains NaN or Inf")
    return output


def _copy_conditioned_critic(source, observation_dim, context_dim, action_dim, config, device):
    target = QNetwork(observation_dim + context_dim, action_dim, config).to(device)
    source_state = source.state_dict()
    target_state = target.state_dict()
    first_weight = next(
        key for key, value in source_state.items()
        if key.endswith("weight") and value.ndim == 2
        and value.shape[1] == observation_dim + action_dim
    )
    for key, source_value in source_state.items():
        if key == first_weight:
            destination = torch.zeros_like(target_state[key])
            destination[:, :observation_dim] = source_value[:, :observation_dim]
            destination[:, observation_dim + context_dim:] = source_value[:, observation_dim:]
            target_state[key] = destination
        else:
            if target_state[key].shape != source_value.shape:
                raise ValueError("conditioned Critic transplant shape mismatch: %s" % key)
            target_state[key] = source_value.detach().clone()
    target.load_state_dict(target_state)
    weight = target.state_dict()[first_weight]
    if not bool(torch.all(weight[:, observation_dim:observation_dim + context_dim] == 0.0)):
        raise RuntimeError("scene-context input columns were not zero initialized")
    return target


def _conditioned_bundle(source_agent, context_dim):
    arguments = (
        source_agent.observation_dim, int(context_dim), source_agent.action_dim,
        source_agent.config, source_agent.device,
    )
    critics = {
        "critic1": _copy_conditioned_critic(source_agent.critic1, *arguments),
        "critic2": _copy_conditioned_critic(source_agent.critic2, *arguments),
        "target_critic1": _copy_conditioned_critic(source_agent.target_critic1, *arguments),
        "target_critic2": _copy_conditioned_critic(source_agent.target_critic2, *arguments),
    }
    critics["optimizer1"] = torch.optim.Adam(
        critics["critic1"].parameters(), lr=source_agent.config.critic_lr,
    )
    critics["optimizer2"] = torch.optim.Adam(
        critics["critic2"].parameters(), lr=source_agent.config.critic_lr,
    )
    return critics


def _augment(observation, groups, context_dim):
    context = F.one_hot(
        groups.reshape(-1).to(dtype=torch.long), num_classes=int(context_dim)
    ).to(dtype=observation.dtype)
    return torch.cat((observation, context), dim=-1)


def _conditioned_update(bundle, actor_agent, normalizer, batch, context_dim):
    device = actor_agent.device
    observation = torch.as_tensor(
        normalizer.normalize(batch["observations"]), dtype=torch.float32, device=device,
    )
    next_observation = torch.as_tensor(
        normalizer.normalize(batch["next_observations"]), dtype=torch.float32, device=device,
    )
    action = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    reward = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    done = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
    groups = torch.as_tensor(batch["groups"], dtype=torch.long, device=device)
    augmented = _augment(observation, groups, context_dim)
    next_augmented = _augment(next_observation, groups, context_dim)
    with torch.no_grad():
        next_action, next_log_probability, _, _, _ = actor_agent._sample_policy(
            next_observation
        )
        next_distribution = torch.minimum(
            bundle["target_critic1"](next_augmented, next_action),
            bundle["target_critic2"](next_augmented, next_action),
        )
        target = reward + (1.0 - done) * actor_agent.config.gamma * (
            next_distribution - actor_agent.alpha.detach() * next_log_probability
        )
    q1 = bundle["critic1"](augmented, action)
    q2 = bundle["critic2"](augmented, action)
    if actor_agent.is_quantile_critic:
        loss1 = quantile_huber_loss(
            q1, target, actor_agent.config.critic_quantile_huber_kappa,
        )
        loss2 = quantile_huber_loss(
            q2, target, actor_agent.config.critic_quantile_huber_kappa,
        )
    else:
        loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
    for critic, optimizer, loss in (
        (bundle["critic1"], bundle["optimizer1"], loss1),
        (bundle["critic2"], bundle["optimizer2"], loss2),
    ):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            critic.parameters(), actor_agent.config.gradient_clip_norm,
        )
        optimizer.step()
    with torch.no_grad():
        for target_critic, critic in (
            (bundle["target_critic1"], bundle["critic1"]),
            (bundle["target_critic2"], bundle["critic2"]),
        ):
            for target_parameter, parameter in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_parameter.mul_(1.0 - actor_agent.config.tau).add_(
                    parameter, alpha=actor_agent.config.tau,
                )
    values = {
        "critic1_loss": float(loss1.detach().cpu()),
        "critic2_loss": float(loss2.detach().cpu()),
        "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
        "target_q_mean": float(target.mean().detach().cpu()),
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
        raise FloatingPointError("L272 conditioned Critic update is non-finite")
    return values


def _save_conditioned(path, bundle, actor_agent, buffer, state):
    payload = {
        "protocol": PROTOCOL,
        "state": dict(state),
        "critic1": bundle["critic1"].state_dict(),
        "critic2": bundle["critic2"].state_dict(),
        "target_critic1": bundle["target_critic1"].state_dict(),
        "target_critic2": bundle["target_critic2"].state_dict(),
        "optimizer1": bundle["optimizer1"].state_dict(),
        "optimizer2": bundle["optimizer2"].state_dict(),
        "actor_alpha_hashes": _frozen_hashes(actor_agent),
        "replay_buffer": buffer.state_dict(include_data=True),
        "git_sha": git_sha(ROOT),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return _sha256(path)


def _heldout(section):
    directory = ROOT / section["l268"]["heldout"]
    paths = {
        "summary": directory / "summary.json",
        "states": directory / "state_manifest.json",
        "returns": directory / "action_returns_h40.csv",
    }
    for key, path in paths.items():
        _verify(path, section["l268"]["heldout_%s_sha256" % key])
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    states = json.loads(paths["states"].read_text(encoding="utf-8"))
    returns = _read_csv(paths["returns"])
    if summary["state_count"] != 36 or len(states) != 36 or len(returns) != 108:
        raise ValueError("L272 held-out diagnostic coverage drifted")
    grouped = defaultdict(list)
    for row in returns:
        grouped[row["state_id"]].append(row)
    if set(grouped) != {row["state_id"] for row in states}:
        raise ValueError("L272 held-out state/action keys drifted")
    return states, grouped


def _ranking_rows(states, grouped_returns, evaluate):
    output = []
    for state in sorted(states, key=lambda row: row["state_id"]):
        rows = sorted(grouped_returns[state["state_id"]], key=lambda row: row["action_id"])
        if {row["action_id"] for row in rows} != {
            "recorded_recovery", "source_actor", "fast_forward",
        }:
            raise ValueError("L272 held-out action set drifted")
        observation = np.repeat(np.asarray(
            state["normalized_observation"], dtype=np.float32,
        )[None, :], 3, axis=0)
        actions = np.asarray([
            (float(row["normalized_v"]), float(row["normalized_omega"]))
            for row in rows
        ], dtype=np.float32)
        q1, q2 = evaluate(state, observation, actions)
        q1, q2 = np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64)
        if q1.shape != q2.shape or q1.shape[0] != 3:
            raise ValueError("L272 evaluator returned an invalid Critic distribution")
        means = np.minimum(q1.mean(axis=-1), q2.mean(axis=-1))
        spreads = 0.5 * (
            np.ptp(q1, axis=-1) + np.ptp(q2, axis=-1)
        ) if q1.shape[-1] > 1 else np.zeros(3, dtype=np.float64)
        if not np.isfinite(np.concatenate((means, spreads))).all():
            raise FloatingPointError("L272 held-out Q values are non-finite")
        for row, mean, spread in zip(rows, means, spreads):
            output.append({
                "state_id": state["state_id"], "scene": state["scene"],
                "scene_index": int(state["scene_index"]), "split": state["split"],
                "action_id": row["action_id"],
                "discounted_return": float(row["discounted_return"]),
                "target_minimum_q": float(mean), "quantile_spread": float(spread),
            })
    return output


def _summarize_ranking(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    state_rows, scene_values = [], defaultdict(lambda: defaultdict(list))
    for state_id, actions in sorted(grouped.items()):
        actions = sorted(actions, key=lambda row: row["action_id"])
        returns = np.asarray([row["discounted_return"] for row in actions])
        values = np.asarray([row["target_minimum_q"] for row in actions])
        by_action = {row["action_id"]: row for row in actions}
        return_delta = (
            by_action["recorded_recovery"]["discounted_return"]
            - by_action["fast_forward"]["discounted_return"]
        )
        q_delta = (
            by_action["recorded_recovery"]["target_minimum_q"]
            - by_action["fast_forward"]["target_minimum_q"]
        )
        pair_correct = bool(
            (return_delta > 0.0 and q_delta > 0.0)
            or (return_delta < 0.0 and q_delta < 0.0)
            or (abs(return_delta) <= 1e-12 and abs(q_delta) <= 1e-12)
        )
        true_best = sorted(
            actions, key=lambda row: (-row["discounted_return"], row["action_id"])
        )[0]["action_id"]
        critic_best = sorted(
            actions, key=lambda row: (-row["target_minimum_q"], row["action_id"])
        )[0]["action_id"]
        spearman = _spearman(values, returns)
        scene = actions[0]["scene"]
        item = {
            "state_id": state_id, "scene": scene,
            "pair_correct": pair_correct,
            "three_action_spearman": spearman,
            "top1_correct": true_best == critic_best,
            "true_best_action": true_best, "critic_best_action": critic_best,
        }
        state_rows.append(item)
        scene_values[scene]["pair"].append(float(pair_correct))
        scene_values[scene]["spearman"].append(float(spearman))
        scene_values[scene]["top1"].append(float(true_best == critic_best))
    summary = {
        "state_count": len(state_rows),
        "recovery_forward_pair_accuracy": float(np.mean([
            row["pair_correct"] for row in state_rows
        ])),
        "mean_three_action_spearman": float(np.mean([
            row["three_action_spearman"] for row in state_rows
        ])),
        "top1_action_agreement": float(np.mean([
            row["top1_correct"] for row in state_rows
        ])),
        "maximum_absolute_mean_q": float(max(
            abs(row["target_minimum_q"]) for row in rows
        )),
        "mean_quantile_spread": float(np.mean([
            row["quantile_spread"] for row in rows
        ])),
        "scene_pair_accuracy": {
            scene: float(np.mean(values["pair"]))
            for scene, values in sorted(scene_values.items())
        },
    }
    if summary["state_count"] != 36:
        raise ValueError("L272 aggregate ranking must contain 36 states")
    numeric = [
        value for key, value in summary.items()
        if key not in ("scene_pair_accuracy",)
    ] + list(summary["scene_pair_accuracy"].values())
    if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
        raise FloatingPointError("L272 ranking summary contains NaN or Inf")
    return state_rows, summary


def _agent_evaluator(agent):
    def evaluate(state, observations, actions):
        del state
        with torch.no_grad():
            observation = torch.as_tensor(
                observations, dtype=torch.float32, device=agent.device,
            )
            action = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
            q1 = agent.target_critic1(observation, action)
            q2 = agent.target_critic2(observation, action)
        return q1.cpu().numpy(), q2.cpu().numpy()
    return evaluate


def _conditioned_evaluator(bundle, device, context_dim):
    def evaluate(state, observations, actions):
        with torch.no_grad():
            observation = torch.as_tensor(
                observations, dtype=torch.float32, device=device,
            )
            groups = torch.full(
                (len(observations),), int(state["scene_index"]),
                dtype=torch.long, device=device,
            )
            augmented = _augment(observation, groups, context_dim)
            action = torch.as_tensor(actions, dtype=torch.float32, device=device)
            q1 = bundle["target_critic1"](augmented, action)
            q2 = bundle["target_critic2"](augmented, action)
        return q1.cpu().numpy(), q2.cpu().numpy()
    return evaluate


def _gate(results, gate, selection_order):
    by_seed_arm = {(int(row["seed"]), row["arm"]): row for row in results}
    seeds = sorted({int(row["seed"]) for row in results})
    scenes = sorted(next(iter(by_seed_arm.values()))["scene_pair_accuracy"])
    arms = [arm for arm in selection_order]
    gate_rows = []
    for arm in arms:
        treatment = [by_seed_arm[(seed, arm)] for seed in seeds]
        control = [by_seed_arm[(seed, "shared_69d")] for seed in seeds]
        pair_changes = [
            left["recovery_forward_pair_accuracy"] - right["recovery_forward_pair_accuracy"]
            for left, right in zip(treatment, control)
        ]
        spearman_changes = [
            left["mean_three_action_spearman"] - right["mean_three_action_spearman"]
            for left, right in zip(treatment, control)
        ]
        top1_changes = [
            left["top1_action_agreement"] - right["top1_action_agreement"]
            for left, right in zip(treatment, control)
        ]
        scene_changes = {
            scene: float(np.mean([
                by_seed_arm[(seed, arm)]["scene_pair_accuracy"][scene]
                - by_seed_arm[(seed, "shared_69d")]["scene_pair_accuracy"][scene]
                for seed in seeds
            ])) for scene in scenes
        }
        aggregate_pair = float(np.mean([
            row["recovery_forward_pair_accuracy"] for row in treatment
        ]))
        aggregate_spearman = float(np.mean([
            row["mean_three_action_spearman"] for row in treatment
        ]))
        aggregate_top1 = float(np.mean([
            row["top1_action_agreement"] for row in treatment
        ]))
        jointly_improving = sum(
            pair > 0.0 and spearman > 0.0
            for pair, spearman in zip(pair_changes, spearman_changes)
        )
        stable = all(
            bool(row["finite"]) and bool(row["actor_alpha_unchanged"])
            and row["maximum_absolute_mean_q"] <= float(gate["maximum_absolute_mean_q"])
            and row["mean_quantile_spread"] <= float(gate["maximum_mean_quantile_spread"])
            for row in treatment
        )
        checks = {
            "aggregate_pair_accuracy": aggregate_pair >= float(
                gate["minimum_aggregate_pair_accuracy"]
            ),
            "paired_median_pair_improvement": float(np.median(pair_changes)) >= float(
                gate["minimum_paired_median_pair_accuracy_improvement"]
            ),
            "aggregate_spearman": aggregate_spearman >= float(
                gate["minimum_aggregate_three_action_spearman"]
            ),
            "paired_median_spearman_improvement": float(np.median(spearman_changes)) >= float(
                gate["minimum_paired_median_spearman_improvement"]
            ),
            "aggregate_top1": aggregate_top1 >= float(
                gate["minimum_aggregate_top1_agreement"]
            ),
            "paired_median_top1_improvement": float(np.median(top1_changes)) >= float(
                gate["minimum_paired_median_top1_improvement"]
            ),
            "jointly_improving_seed_blocks": jointly_improving >= int(
                gate["minimum_jointly_improving_seed_blocks"]
            ),
            "scene_coverage": sum(value > 0.0 for value in scene_changes.values()) >= int(
                gate["minimum_scenes_with_pair_accuracy_improvement"]
            ),
            "no_scene_collapse": min(scene_changes.values()) >= -float(
                gate["maximum_scene_pair_accuracy_decrease"]
            ),
            "finite_stable_actor_frozen": stable,
        }
        gate_rows.append({
            "arm": arm, "aggregate_pair_accuracy": aggregate_pair,
            "paired_pair_changes": pair_changes,
            "paired_median_pair_improvement": float(np.median(pair_changes)),
            "aggregate_three_action_spearman": aggregate_spearman,
            "paired_spearman_changes": spearman_changes,
            "paired_median_spearman_improvement": float(np.median(spearman_changes)),
            "aggregate_top1_agreement": aggregate_top1,
            "paired_top1_changes": top1_changes,
            "paired_median_top1_improvement": float(np.median(top1_changes)),
            "jointly_improving_seed_blocks": jointly_improving,
            "scene_pair_accuracy_changes": scene_changes,
            "scenes_improved": sum(value > 0.0 for value in scene_changes.values()),
            "minimum_scene_pair_accuracy_change": min(scene_changes.values()),
            "checks": checks, "gate_pass": all(checks.values()),
        })
    passing = {row["arm"] for row in gate_rows if row["gate_pass"]}
    selected = next((arm for arm in selection_order if arm in passing), None)
    return gate_rows, selected


def run(config_path: Path, output: Path):
    _, section = _load_config(config_path)
    source_path = ROOT / section["source_checkpoint"]
    _verify(source_path, section["source_checkpoint_sha256"])
    l271_path = ROOT / section["l271"]["summary"]
    _verify(l271_path, section["l271"]["summary_sha256"])
    l271 = json.loads(l271_path.read_text(encoding="utf-8"))
    if bool(l271["critic_scene_interference_gate_pass"]) is not bool(
        section["l271"]["required_gate_pass"]
    ):
        raise ValueError("L272 requires the frozen positive L271 Gate")
    l268 = section["l268"]
    config268_path = ROOT / l268["config"]
    dataset = ROOT / l268["dataset"]
    manifest_path = dataset / "manifest.json"
    _verify(config268_path, l268["config_sha256"])
    _verify(manifest_path, l268["dataset_manifest_sha256"])
    config268 = yaml.safe_load(config268_path.read_text(encoding="utf-8"))["l268"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_states, heldout_returns = _heldout(section)
    device = section["design"]["device"]
    source_payload = load_sac_checkpoint(source_path, map_location=device)
    source_replay = _replay_arrays(source_payload)
    recovery, _ = _recovery_pool(dataset, manifest)
    output.mkdir(parents=True, exist_ok=False)

    shared_specs = {
        int(row["seed"]): row for row in l268["shared_checkpoints"]
    }
    seeds = [int(value) for value in section["design"]["paired_seeds"]]
    arm_rng = np.random.RandomState(int(section["design"]["arm_order_seed"]))
    arm_orders = {
        str(seed): [str(value) for value in arm_rng.permutation(
            np.asarray(section["design"]["arm_order"], dtype=object)
        )] for seed in seeds
    }
    _json_dump(output / "arm_orders.json", arm_orders)
    progress, results, all_rankings, all_states = [], [], [], []

    for seed in seeds:
        expected_replay, _ = _build_treatment(source_replay, recovery, config268, seed)
        shared_path = ROOT / shared_specs[seed]["path"]
        _verify(shared_path, shared_specs[seed]["sha256"])
        shared_payload = load_sac_checkpoint(shared_path, map_location=device)
        shared_replay = _replay_arrays(shared_payload)
        _verify_embedded_replay(
            shared_replay, expected_replay, section["design"]["scene_count"],
        )
        shared_agent, _ = _load_agent(shared_payload, device, seed)
        shared_before = _frozen_hashes(shared_agent)
        ranking = _ranking_rows(
            heldout_states, heldout_returns, _agent_evaluator(shared_agent),
        )
        state_rows, summary = _summarize_ranking(ranking)
        summary.update({
            "seed": seed, "arm": "shared_69d", "finite": True,
            "actor_alpha_unchanged": shared_before == _frozen_hashes(shared_agent),
        })
        results.append(summary)
        all_rankings.extend({"seed": seed, "arm": "shared_69d", **row} for row in ranking)
        all_states.extend({"seed": seed, "arm": "shared_69d", **row} for row in state_rows)
        del shared_agent
        torch.cuda.empty_cache()

        for arm in arm_orders[str(seed)]:
            if arm == "per_scene_equal_compute":
                scene_rng = np.random.RandomState(
                    int(section["design"]["scene_order_seed"]) + seed
                )
                scene_order = [int(value) for value in scene_rng.permutation(
                    int(section["design"]["scene_count"])
                )]
                arm_rankings = []
                actor_unchanged = True
                for scene in scene_order:
                    arrays = _filter_scene(expected_replay, scene)
                    buffer = _buffer_from_arrays(arrays, seed * 10 + scene)
                    agent, normalizer = _load_agent(source_payload, device, seed)
                    frozen = _frozen_hashes(agent)
                    updates = int(section["design"]["per_scene"]["updates_per_critic"])
                    interval = int(section["design"]["per_scene"]["checkpoint_interval"])
                    for update in range(1, updates + 1):
                        metrics = agent.update(
                            _normalized_batch(
                                buffer, normalizer, section["design"]["batch_size"]
                            ), update_actor=False,
                        )
                        if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
                            raise FloatingPointError("L272 per-scene update is non-finite")
                        if update % interval == 0:
                            if frozen != _frozen_hashes(agent):
                                raise RuntimeError("L272 per-scene Actor or alpha mutated")
                            path = output / ("seed_%d" % seed) / arm / (
                                "scene_%d" % scene
                            ) / "checkpoints" / ("critic_update_%06d.pt" % update)
                            digest = _save_checkpoint(
                                path, source_payload, agent, normalizer, buffer,
                                {
                                    "protocol": PROTOCOL, "arm": arm, "seed": seed,
                                    "scene": scene, "critic_updates": update,
                                    "source_checkpoint_sha256": section[
                                        "source_checkpoint_sha256"
                                    ], "actor_update_applied": False,
                                },
                            )
                            progress.append({
                                "seed": seed, "arm": arm, "scene": scene,
                                "completed_updates": update, "checkpoint_sha256": digest,
                                "actor_alpha_unchanged": True,
                            })
                            _write_csv(output / "progress.csv", progress)
                    scene_states = [
                        row for row in heldout_states if int(row["scene_index"]) == scene
                    ]
                    arm_rankings.extend(_ranking_rows(
                        scene_states, heldout_returns, _agent_evaluator(agent),
                    ))
                    actor_unchanged = actor_unchanged and frozen == _frozen_hashes(agent)
                    del agent, buffer
                    torch.cuda.empty_cache()
                state_rows, summary = _summarize_ranking(arm_rankings)
                summary.update({
                    "seed": seed, "arm": arm, "finite": True,
                    "actor_alpha_unchanged": actor_unchanged,
                })
                results.append(summary)
                all_rankings.extend({"seed": seed, "arm": arm, **row} for row in arm_rankings)
                all_states.extend({"seed": seed, "arm": arm, **row} for row in state_rows)

            elif arm == "scene_conditioned_equal_compute":
                buffer = _buffer_from_arrays(expected_replay, seed)
                actor_agent, normalizer = _load_agent(source_payload, device, seed)
                frozen = _frozen_hashes(actor_agent)
                context_dim = int(section["design"]["scene_conditioned"]["context_dim"])
                bundle = _conditioned_bundle(actor_agent, context_dim)
                updates = int(section["design"]["scene_conditioned"]["updates"])
                interval = int(section["design"]["scene_conditioned"]["checkpoint_interval"])
                for update in range(1, updates + 1):
                    batch = buffer.sample(
                        int(section["design"]["batch_size"]),
                        strategy=section["design"]["scene_conditioned"]["sampling_strategy"],
                    )
                    metrics = _conditioned_update(
                        bundle, actor_agent, normalizer, batch, context_dim,
                    )
                    if update % interval == 0:
                        if frozen != _frozen_hashes(actor_agent):
                            raise RuntimeError("L272 conditioned Actor or alpha mutated")
                        path = output / ("seed_%d" % seed) / arm / "checkpoints" / (
                            "critic_update_%06d.pt" % update
                        )
                        digest = _save_conditioned(
                            path, bundle, actor_agent, buffer,
                            {
                                "seed": seed, "arm": arm, "critic_updates": update,
                                "source_checkpoint_sha256": section[
                                    "source_checkpoint_sha256"
                                ], "actor_update_applied": False,
                                "context_dim": context_dim,
                            },
                        )
                        progress.append({
                            "seed": seed, "arm": arm, "scene": "all",
                            "completed_updates": update, "checkpoint_sha256": digest,
                            "actor_alpha_unchanged": True,
                        })
                        _write_csv(output / "progress.csv", progress)
                ranking = _ranking_rows(
                    heldout_states, heldout_returns,
                    _conditioned_evaluator(bundle, actor_agent.device, context_dim),
                )
                state_rows, summary = _summarize_ranking(ranking)
                summary.update({
                    "seed": seed, "arm": arm, "finite": True,
                    "actor_alpha_unchanged": frozen == _frozen_hashes(actor_agent),
                })
                results.append(summary)
                all_rankings.extend({"seed": seed, "arm": arm, **row} for row in ranking)
                all_states.extend({"seed": seed, "arm": arm, **row} for row in state_rows)
                del actor_agent, bundle, buffer
                torch.cuda.empty_cache()
            else:
                raise ValueError("unknown L272 arm: %s" % arm)
            print(json.dumps({"completed_seed": seed, "completed_arm": arm}))

    expected_keys = {
        (seed, arm) for seed in seeds for arm in (
            "shared_69d", "per_scene_equal_compute", "scene_conditioned_equal_compute",
        )
    }
    if {(row["seed"], row["arm"]) for row in results} != expected_keys:
        raise RuntimeError("L272 arm/seed coverage is incomplete")
    gate_rows, selected = _gate(
        results, section["gate"], section["selection_order"],
    )
    passing = {row["arm"] for row in gate_rows if row["gate_pass"]}
    if "scene_conditioned_equal_compute" in passing:
        decision = section["next_if_conditioned_pass"]
    elif "per_scene_equal_compute" in passing:
        decision = section["next_if_only_per_scene_pass"]
    else:
        decision = section["next_if_fail"]
    _write_csv(output / "action_ranking.csv", all_rankings)
    _write_csv(output / "metrics_by_state.csv", all_states)
    _write_csv(output / "results.csv", [{
        **{key: value for key, value in row.items() if key != "scene_pair_accuracy"},
        "scene_pair_accuracy": json.dumps(row["scene_pair_accuracy"], sort_keys=True),
    } for row in results])
    _write_csv(output / "gate_by_arm.csv", [{
        **{key: value for key, value in row.items()
           if key not in ("paired_pair_changes", "paired_spearman_changes",
                          "paired_top1_changes", "scene_pair_accuracy_changes", "checks")},
        "paired_pair_changes": json.dumps(row["paired_pair_changes"]),
        "paired_spearman_changes": json.dumps(row["paired_spearman_changes"]),
        "paired_top1_changes": json.dumps(row["paired_top1_changes"]),
        "scene_pair_accuracy_changes": json.dumps(
            row["scene_pair_accuracy_changes"], sort_keys=True,
        ),
        "checks": json.dumps(row["checks"], sort_keys=True),
    } for row in gate_rows])
    summary = {
        "protocol": PROTOCOL, "status": "complete",
        "selected_arm": selected, "decision": decision,
        "actor_training_authorized": False,
        "git_sha": git_sha(ROOT), "config_sha256": _sha256(config_path),
        "paired_seed_count": len(seeds), "arm_count": 3,
        "heldout_state_count": len(heldout_states),
        "all_actor_alpha_unchanged": all(
            row["actor_alpha_unchanged"] for row in results
        ),
        "all_finite": all(row["finite"] for row in results),
        "gate_by_arm": gate_rows,
    }
    _json_dump(output / "summary.json", summary)
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
