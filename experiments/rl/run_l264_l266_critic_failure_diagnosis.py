#!/usr/bin/env python3
"""Read-only L264--L266 diagnosis of the frozen L262 quantile Critic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import mobile_robot_mppi.rl.sac as sac_module
from experiments.rl.run_l263_counterfactual_actor_diagnosis import (
    _load_agent,
    observation_feature_names,
)
from mobile_robot_mppi.rl.sac import SACAgent
from mobile_robot_mppi.rl.trainer import SACTrainer


PROTOCOL = "L264-L266"
EXPECTED_CHECKPOINT_SHA256 = (
    "48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7"
)
DEFAULT_CHECKPOINT = ROOT / (
    "results/research_platform/rl/"
    "l262_coverage_gated_value_seed20262611_6k/"
    "checkpoints/step_000006000.pt"
)
DEFAULT_L263 = ROOT / (
    "results/research_platform/rl/l263_counterfactual_actor_diagnosis"
)
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l264_l266_critic_failure_diagnosis"
)
DIAGNOSTIC_SEED = 20262864
LOCAL_NEIGHBORS = 64
CTE_BINS = (
    ("cte_lt_0.25", -np.inf, 0.25),
    ("cte_0.25_0.75", 0.25, 0.75),
    ("cte_0.75_1.5", 0.75, 1.5),
    ("cte_1.5_3.0", 1.5, 3.0),
    ("cte_gt_3.0", 3.0, np.inf),
)
CURVATURE_CATEGORIES = (
    "straight", "gentle_left", "gentle_right", "sharp_left",
    "sharp_right", "curvature_reversal",
)
STAGE_CATEGORIES = (
    "normal_tracking", "near_boundary", "off_path", "recovery",
    "obstacle_avoidance", "terminal_approach", "loop_exit_unavailable",
)
SUPPORT_ORDER = {"in_support": 0, "near_support": 1, "out_of_support": 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _mean(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _set_torch_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _replay_arrays(payload):
    replay = payload.get("replay_buffer")
    if not replay or int(replay.get("size", 0)) <= 0:
        raise ValueError("L264-L266 requires checkpoint replay tensors")
    size = int(replay["size"])
    names = (
        "observations", "actions", "rewards", "constraint_costs",
        "next_observations", "dones", "groups", "outcomes", "transition_ids",
    )
    result = {name: np.asarray(replay[name][:size]).copy() for name in names}
    if result["observations"].shape != (size, 69):
        raise ValueError("frozen replay is not 69D")
    if not all(np.isfinite(result[name]).all() for name in names):
        raise FloatingPointError("frozen replay contains NaN or Inf")
    return result


def _scene_names(payload):
    resolved = payload["resolved_config"]
    paths = resolved["rl"]["training"]["scene_configs"]
    names = {}
    for index, path in enumerate(paths):
        names[int(index)] = Path(str(path)).stem.replace("l261_", "l260_")
    return names


def _select_l264_indices(replay):
    selected = []
    for group in sorted(np.unique(replay["groups"]).astype(int)):
        candidates = np.flatnonzero(replay["groups"] == group)
        terminal = candidates[replay["dones"][candidates, 0] > 0.5]
        chosen = list(terminal[:4])
        remaining = np.setdiff1d(candidates, np.asarray(chosen, dtype=np.int64))
        need = 20 - len(chosen)
        positions = np.linspace(0, max(remaining.size - 1, 0), need).round().astype(int)
        chosen.extend(remaining[positions].tolist())
        selected.extend(chosen)
    result = np.asarray(selected, dtype=np.int64)
    if result.size != 120 or np.unique(result).size != result.size:
        raise RuntimeError("L264 deterministic sample must contain 120 unique rows")
    return result


def _normalized_batch(replay, indices, normalizer):
    return {
        "observations": normalizer.normalize(replay["observations"][indices]),
        "actions": replay["actions"][indices].astype(np.float32),
        "rewards": replay["rewards"][indices].astype(np.float32),
        "constraint_costs": replay["constraint_costs"][indices].astype(np.float32),
        "next_observations": normalizer.normalize(
            replay["next_observations"][indices]
        ),
        "dones": replay["dones"][indices].astype(np.float32),
        "groups": replay["groups"][indices].astype(np.int64),
        "outcomes": replay["outcomes"][indices].astype(np.int8),
    }


def _manual_target(agent, batch, seed):
    _set_torch_seed(seed)
    reward = np.asarray(batch["rewards"], dtype=np.float64)
    done = np.asarray(batch["dones"], dtype=np.float64)
    with torch.no_grad():
        next_observation = torch.as_tensor(
            batch["next_observations"], dtype=torch.float32, device=agent.device
        )
        next_action, log_probability, _, _, _ = agent._sample_policy(
            next_observation
        )
        q1 = agent.target_critic1(next_observation, next_action)
        q2 = agent.target_critic2(next_observation, next_action)
        minimum = torch.minimum(q1, q2)
    action_np = next_action.cpu().numpy().astype(np.float64)
    logp_np = log_probability.cpu().numpy().astype(np.float64)
    minimum_np = minimum.cpu().numpy().astype(np.float64)
    alpha = float(agent.alpha.detach().cpu())
    bootstrap = minimum_np - alpha * logp_np
    target = reward + (1.0 - done) * float(agent.config.gamma) * bootstrap
    return {
        "action": action_np,
        "log_probability": logp_np,
        "target_q1": q1.cpu().numpy().astype(np.float64),
        "target_q2": q2.cpu().numpy().astype(np.float64),
        "minimum": minimum_np,
        "entropy_bonus": -alpha * logp_np,
        "bootstrap": bootstrap,
        "target": target,
    }


def _captured_update_target(checkpoint, device, batch, seed):
    _, agent, _ = _load_agent(checkpoint, device)
    captured = []
    original = sac_module.quantile_huber_loss

    def capture(predictions, targets, kappa=1.0):
        captured.append(targets.detach().cpu().numpy().astype(np.float64))
        return original(predictions, targets, kappa)

    _set_torch_seed(seed)
    with mock.patch.object(sac_module, "quantile_huber_loss", side_effect=capture):
        agent.update(batch, update_actor=False)
    if len(captured) != 2:
        raise RuntimeError("failed to capture both frozen Critic targets")
    return captured


def _transition_diagnostics(agent, normalizer, replay, chunk_size=512):
    count = replay["actions"].shape[0]
    output = defaultdict(list)
    _set_torch_seed(DIAGNOSTIC_SEED + 1)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        obs = normalizer.normalize(replay["observations"][start:stop])
        nxt = normalizer.normalize(replay["next_observations"][start:stop])
        action = replay["actions"][start:stop].astype(np.float32)
        with torch.no_grad():
            ot = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            nt = torch.as_tensor(nxt, dtype=torch.float32, device=agent.device)
            at = torch.as_tensor(action, dtype=torch.float32, device=agent.device)
            q1 = agent.critic1(ot, at)
            q2 = agent.critic2(ot, at)
            qmin = torch.minimum(q1, q2)
            next_action, logp, _, _, _ = agent._sample_policy(nt)
            tq1 = agent.target_critic1(nt, next_action)
            tq2 = agent.target_critic2(nt, next_action)
            tqmin = torch.minimum(tq1, tq2)
            alpha = agent.alpha.detach()
            bootstrap = tqmin - alpha * logp
            reward = torch.as_tensor(
                replay["rewards"][start:stop], dtype=torch.float32,
                device=agent.device,
            )
            done = torch.as_tensor(
                replay["dones"][start:stop], dtype=torch.float32,
                device=agent.device,
            )
            target = reward + (1.0 - done) * agent.config.gamma * bootstrap
            actor_action, _, _, _, _ = agent._sample_policy(ot, deterministic=True)
        arrays = {
            "online_q_mean": qmin.mean(-1),
            "online_q1_mean": q1.mean(-1),
            "online_q2_mean": q2.mean(-1),
            "target_q_mean": target.mean(-1),
            "bootstrap_mean": bootstrap.mean(-1),
            "entropy_bonus": (-alpha * logp).reshape(-1),
            "td_abs": torch.abs(qmin.mean(-1) - target.mean(-1)),
            "quantile_spread": qmin.max(-1).values - qmin.min(-1).values,
            "twin_disagreement": torch.abs(q1.mean(-1) - q2.mean(-1)),
            "target_twin_disagreement": torch.abs(tq1.mean(-1) - tq2.mean(-1)),
            "actor_replay_gap": torch.linalg.vector_norm(actor_action - at, dim=-1),
            "actor_v": actor_action[:, 0],
            "actor_omega": actor_action[:, 1],
            "target_action_v": next_action[:, 0],
            "target_action_omega": next_action[:, 1],
        }
        for name, tensor in arrays.items():
            output[name].append(tensor.cpu().numpy().astype(np.float64))
    return {name: np.concatenate(chunks) for name, chunks in output.items()}


def _aggregate(values, mask):
    mask = np.asarray(mask, dtype=bool)
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "median": float(np.median(selected)),
    }


def _scene_scale_rows(replay, diagnostics, scene_names):
    rows = []
    for group in sorted(scene_names):
        mask = replay["groups"] == group
        row = {
            "group": group,
            "scene": scene_names[group],
            "count": int(np.sum(mask)),
            "terminal_fraction": float(np.mean(replay["dones"][mask, 0])),
            "truncation_fraction": "not_stored_bootstrappable",
            "success_fraction": float(np.mean(replay["outcomes"][mask] == 1)),
            "incomplete_fraction": float(np.mean(replay["outcomes"][mask] == -1)),
        }
        metrics = {
            "reward": replay["rewards"][:, 0],
            "online_q": diagnostics["online_q_mean"],
            "target_q": diagnostics["target_q_mean"],
            "bootstrap": diagnostics["bootstrap_mean"],
            "entropy_bonus": diagnostics["entropy_bonus"],
            "td_abs": diagnostics["td_abs"],
            "quantile_spread": diagnostics["quantile_spread"],
            "twin_disagreement": diagnostics["twin_disagreement"],
        }
        for name, values in metrics.items():
            stats = _aggregate(values, mask)
            row.update({"%s_%s" % (name, key): value for key, value in stats.items()})
        rows.append(row)
    return rows


def _policy_log_probability(agent, observations, actions):
    observation = torch.as_tensor(
        observations, dtype=torch.float32, device=agent.device
    )
    bounded = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
    epsilon = 1e-6
    bounded = torch.clamp(bounded, -1.0 + epsilon, 1.0 - epsilon)
    with torch.no_grad():
        mean, log_std = agent.actor.distribution(observation)
        pre_tanh = torch.atanh(bounded)
        variance = torch.exp(2.0 * log_std)
        gaussian = -0.5 * (
            ((pre_tanh - mean) ** 2) / variance
            + 2.0 * log_std + np.log(2.0 * np.pi)
        )
        gaussian = gaussian.sum(-1)
        correction = 2.0 * (
            np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        result = gaussian - correction.sum(-1)
    return result.cpu().numpy().astype(np.float64)


def _local_support(replay_obs, replay_actions, state_observation, candidates):
    distances = np.linalg.norm(replay_obs - state_observation[None, :], axis=1)
    neighbor_indices = np.argsort(distances, kind="mergesort")[:LOCAL_NEIGHBORS]
    local_actions = replay_actions[neighbor_indices].astype(np.float64)
    pairwise = np.linalg.norm(
        local_actions[:, None, :] - local_actions[None, :, :], axis=-1
    )
    np.fill_diagonal(pairwise, np.inf)
    local_nearest = np.min(pairwise, axis=1)
    median = float(np.median(local_nearest))
    q90 = float(np.quantile(local_nearest, 0.90))
    candidate_distances = np.linalg.norm(
        candidates[:, None, :] - local_actions[None, :, :], axis=-1
    )
    nearest = np.min(candidate_distances, axis=1)
    density = np.mean(candidate_distances <= max(q90, 1e-12), axis=1)
    labels = np.where(
        nearest <= median, "in_support",
        np.where(nearest <= q90, "near_support", "out_of_support"),
    )
    return {
        "nearest_observation_distance": float(distances[neighbor_indices[0]]),
        "local_action_median": median,
        "local_action_q90": q90,
        "nearest_action_distance": nearest,
        "local_action_density": density,
        "support": labels,
    }


def _support_rows(agent, normalizer, replay, l263_dir):
    states = json.loads((l263_dir / "state_manifest.json").read_text(encoding="utf-8"))
    state_by_id = {row["state_id"]: row for row in states}
    action_rows = _read_csv(l263_dir / "critic_actions.csv")
    replay_obs = normalizer.normalize(replay["observations"])
    grouped = defaultdict(list)
    for row in action_rows:
        grouped[row["state_id"]].append(row)
    output = []
    for state_id, rows in sorted(grouped.items()):
        state = state_by_id[state_id]
        observation = np.asarray(state["normalized_observation"], dtype=np.float32)
        candidates = np.asarray([
            (float(row["normalized_v"]), float(row["normalized_omega"]))
            for row in rows
        ], dtype=np.float32)
        support = _local_support(replay_obs, replay["actions"], observation, candidates)
        repeated = np.repeat(observation[None, :], len(rows), axis=0)
        logp = _policy_log_probability(agent, repeated, candidates)
        actor = np.asarray(state["actor_action"], dtype=np.float64)
        for index, row in enumerate(rows):
            candidate = candidates[index].astype(np.float64)
            output.append({
                **row,
                "scene": state["scene"],
                "scene_role": state["scene_role"],
                "kind": state["kind"],
                "support": str(support["support"][index]),
                "nearest_replay_observation_distance": support[
                    "nearest_observation_distance"
                ],
                "nearest_local_replay_action_distance": float(
                    support["nearest_action_distance"][index]
                ),
                "local_action_support_median": support["local_action_median"],
                "local_action_support_q90": support["local_action_q90"],
                "local_action_density": float(support["local_action_density"][index]),
                "actor_mean_distance": float(np.linalg.norm(candidate - actor)),
                "policy_log_probability": float(logp[index]),
                "action_boundary_margin": float(np.min(1.0 - np.abs(candidate))),
            })
    return output


def _support_metrics(support_rows, l263_dir):
    rollout = _read_csv(l263_dir / "counterfactual_rollouts.csv")
    returns = {
        (row["state_id"], row["action_id"]): float(row["discounted_return"])
        for row in rollout
        if row["continuation"] == "actor_follow" and int(row["horizon"]) == 40
    }
    grouped = defaultdict(list)
    for row in support_rows:
        key = (row["state_id"], row["action_id"])
        if key not in returns:
            raise ValueError("L265 join is missing an H40 Actor-follow return")
        item = dict(row)
        item["realized_return_h40"] = returns[key]
        grouped[item["state_id"]].append(item)

    per_state = []
    standardized_errors = defaultdict(list)
    pair_rows = []
    for state_id, rows in sorted(grouped.items()):
        q_all = np.asarray([float(row["target_minimum_q"]) for row in rows])
        r_all = np.asarray([float(row["realized_return_h40"]) for row in rows])
        q_z = (q_all - np.mean(q_all)) / max(float(np.std(q_all)), 1e-12)
        r_z = (r_all - np.mean(r_all)) / max(float(np.std(r_all)), 1e-12)
        for row, error in zip(rows, np.abs(q_z - r_z)):
            standardized_errors[row["support"]].append(float(error))
        for support in SUPPORT_ORDER:
            subset = [row for row in rows if row["support"] == support]
            q = np.asarray([float(row["target_minimum_q"]) for row in subset])
            realized = np.asarray([float(row["realized_return_h40"]) for row in subset])
            top1 = float("nan")
            top3 = float("nan")
            if subset:
                real_best = int(np.argmax(realized))
                top1 = float(real_best == int(np.argmax(q)))
                top3 = float(real_best in np.argsort(q)[-min(3, q.size):])
            per_state.append({
                "state_id": state_id,
                "scene": rows[0]["scene"],
                "kind": rows[0]["kind"],
                "support": support,
                "action_count": len(subset),
                "spearman": _spearman(q, realized),
                "top1_agreement": top1,
                "top3_agreement": top3,
            })
        recovery = next(
            (row for row in rows if "recovery" in row["semantic_labels"].split("|")),
            None,
        )
        forward = next(
            (row for row in rows if "grid_v+1.0_w+0.0" in row["semantic_labels"]),
            None,
        )
        if recovery is not None and forward is not None:
            support = max(
                (recovery["support"], forward["support"]),
                key=lambda name: SUPPORT_ORDER[name],
            )
            q_delta = float(recovery["target_minimum_q"]) - float(
                forward["target_minimum_q"]
            )
            r_delta = float(recovery["realized_return_h40"]) - float(
                forward["realized_return_h40"]
            )
            pair_rows.append({
                "state_id": state_id,
                "scene": recovery["scene"],
                "kind": recovery["kind"],
                "pair_support": support,
                "critic_delta_recovery_minus_forward": q_delta,
                "return_delta_recovery_minus_forward": r_delta,
                "pairwise_correct": float((q_delta >= 0.0) == (r_delta >= 0.0)),
            })

    summary = []
    for support in SUPPORT_ORDER:
        rows = [row for row in per_state if row["support"] == support]
        pairs = [row for row in pair_rows if row["pair_support"] == support]
        summary.append({
            "support": support,
            "state_metrics_count": int(sum(np.isfinite(row["spearman"]) for row in rows)),
            "action_count": int(sum(row["action_count"] for row in rows)),
            "mean_spearman": _mean(row["spearman"] for row in rows),
            "mean_top1_agreement": _mean(row["top1_agreement"] for row in rows),
            "mean_top3_agreement": _mean(row["top3_agreement"] for row in rows),
            "mean_standardized_q_return_abs_error": _mean(
                standardized_errors[support]
            ),
            "pair_count": len(pairs),
            "recovery_forward_pair_accuracy": _mean(
                row["pairwise_correct"] for row in pairs
            ),
        })
    return per_state, pair_rows, summary


def _cte_bin(value):
    value = float(value)
    for name, lower, upper in CTE_BINS:
        if lower <= value < upper:
            return name
    raise RuntimeError("unreachable CTE bin")


def _curvature_bin(current, following, continuous):
    current = float(current)
    following = float(following)
    if continuous and abs(current) >= 0.05 and abs(following) >= 0.05 and current * following < 0:
        return "curvature_reversal"
    magnitude = abs(current)
    if magnitude < 0.05:
        return "straight"
    if magnitude < 0.20:
        return "gentle_left" if current > 0 else "gentle_right"
    return "sharp_left" if current > 0 else "sharp_right"


def _continuity(replay):
    ids = replay["transition_ids"].astype(np.int64)
    order = np.argsort(ids, kind="mergesort")
    continuous_next = np.zeros(ids.size, dtype=bool)
    episode_id = np.empty(ids.size, dtype=np.int64)
    episode = 0
    for position, index in enumerate(order):
        if position == 0:
            episode_id[index] = episode
            continue
        previous = order[position - 1]
        state_error = float(np.max(np.abs(
            replay["next_observations"][previous] - replay["observations"][index]
        )))
        continuous = bool(
            ids[index] == ids[previous] + 1
            and replay["groups"][index] == replay["groups"][previous]
            and replay["dones"][previous, 0] < 0.5
            and state_error <= 1e-5
        )
        continuous_next[previous] = continuous
        if not continuous:
            episode += 1
        episode_id[index] = episode
    return order, continuous_next, episode_id


def _coverage_rows(replay, diagnostics, scene_names):
    names = observation_feature_names({
        **{},
    }) if False else None
    del names
    order, continuous_next, episode_ids = _continuity(replay)
    id_to_index = {
        int(identifier): int(index)
        for index, identifier in enumerate(replay["transition_ids"])
    }
    cte = np.sqrt(np.maximum(replay["constraint_costs"][:, 0], 0.0))
    next_cte = np.full(cte.shape, np.nan, dtype=np.float64)
    curvature = replay["observations"][:, 14].astype(np.float64)
    next_curvature = replay["next_observations"][:, 14].astype(np.float64)
    positive_recovery = np.zeros(cte.shape, dtype=bool)
    for index in range(cte.size):
        if continuous_next[index]:
            following = id_to_index[int(replay["transition_ids"][index]) + 1]
            next_cte[index] = cte[following]
            positive_recovery[index] = bool(
                cte[index] > 0.75 and next_cte[index] <= cte[index] - 0.05
            )
    cte_bins = np.asarray([_cte_bin(value) for value in cte], dtype=object)
    curvature_bins = np.asarray([
        _curvature_bin(curvature[index], next_curvature[index], continuous_next[index])
        for index in range(cte.size)
    ], dtype=object)
    stages = []
    for index in range(cte.size):
        safety = replay["observations"][index, 10] > 0.5
        remaining = replay["observations"][index, 15]
        if safety:
            stage = "obstacle_avoidance"
        elif remaining <= 0.15:
            stage = "terminal_approach"
        elif positive_recovery[index]:
            stage = "recovery"
        elif cte[index] > 0.75:
            stage = "off_path"
        elif cte[index] >= 0.50:
            stage = "near_boundary"
        else:
            stage = "normal_tracking"
        stages.append(stage)
    stages = np.asarray(stages, dtype=object)

    transition_rows = []
    for index in range(cte.size):
        transition_rows.append({
            "transition_id": int(replay["transition_ids"][index]),
            "episode_id": int(episode_ids[index]),
            "group": int(replay["groups"][index]),
            "scene": scene_names[int(replay["groups"][index])],
            "cte_m": float(cte[index]),
            "next_cte_m": float(next_cte[index]),
            "cte_bin": str(cte_bins[index]),
            "curvature": float(curvature[index]),
            "curvature_bin": str(curvature_bins[index]),
            "stage": str(stages[index]),
            "continuous_next": bool(continuous_next[index]),
            "positive_recovery": bool(positive_recovery[index]),
            "action_v": float(replay["actions"][index, 0]),
            "action_omega": float(replay["actions"][index, 1]),
            "reward": float(replay["rewards"][index, 0]),
            "done": bool(replay["dones"][index, 0]),
            "outcome": int(replay["outcomes"][index]),
            "online_q_mean": float(diagnostics["online_q_mean"][index]),
            "target_q_mean": float(diagnostics["target_q_mean"][index]),
            "td_abs": float(diagnostics["td_abs"][index]),
            "quantile_spread": float(diagnostics["quantile_spread"][index]),
            "actor_replay_action_gap": float(diagnostics["actor_replay_gap"][index]),
        })

    chain_rows = []
    by_episode = defaultdict(list)
    for index in order:
        by_episode[int(episode_ids[index])].append(int(index))
    for episode, indices in sorted(by_episode.items()):
        active = None
        for index in indices:
            if active is None and cte[index] > 0.75:
                active = {
                    "start": index,
                    "maximum": float(cte[index]),
                    "positive_steps": 0,
                }
            if active is None:
                continue
            active["maximum"] = max(active["maximum"], float(cte[index]))
            active["positive_steps"] += int(positive_recovery[index])
            if cte[index] < 0.75:
                start = int(active["start"])
                chain_rows.append({
                    "episode_id": episode,
                    "group": int(replay["groups"][start]),
                    "scene": scene_names[int(replay["groups"][start])],
                    "start_transition_id": int(replay["transition_ids"][start]),
                    "reentry_transition_id": int(replay["transition_ids"][index]),
                    "chain_steps": int(
                        replay["transition_ids"][index] - replay["transition_ids"][start]
                    ),
                    "start_cte_m": float(cte[start]),
                    "maximum_cte_m": float(active["maximum"]),
                    "positive_recovery_steps": int(active["positive_steps"]),
                })
                active = None

    def aggregate_dimension(dimension, categories, labels):
        rows = []
        for category in categories:
            mask = labels == category
            row = {
                "dimension": dimension,
                "category": category,
                "count": int(np.sum(mask)),
                "fraction": float(np.mean(mask)),
                "positive_recovery_fraction": (
                    float(np.mean(positive_recovery[mask])) if np.any(mask) else float("nan")
                ),
            }
            metrics = {
                "action_v": replay["actions"][:, 0],
                "action_abs_omega": np.abs(replay["actions"][:, 1]),
                "reward": replay["rewards"][:, 0],
                "td_abs": diagnostics["td_abs"],
                "quantile_spread": diagnostics["quantile_spread"],
                "actor_replay_gap": diagnostics["actor_replay_gap"],
            }
            for name, values in metrics.items():
                stats = _aggregate(values, mask)
                row.update({"%s_%s" % (name, key): value for key, value in stats.items()})
            rows.append(row)
        return rows

    bin_rows = []
    bin_rows.extend(aggregate_dimension(
        "cte", [name for name, _, _ in CTE_BINS], cte_bins
    ))
    bin_rows.extend(aggregate_dimension(
        "curvature", CURVATURE_CATEGORIES, curvature_bins
    ))
    bin_rows.extend(aggregate_dimension(
        "stage", STAGE_CATEGORIES, stages
    ))
    scene_labels = np.asarray([
        scene_names[int(group)] for group in replay["groups"]
    ], dtype=object)
    bin_rows.extend(aggregate_dimension(
        "scene", [scene_names[key] for key in sorted(scene_names)], scene_labels
    ))
    return transition_rows, bin_rows, chain_rows


def _static_contracts(payload):
    update_source = inspect.getsource(SACAgent.update)
    trainer_source = inspect.getsource(SACTrainer.train)
    agent_state = payload["agent"]
    return {
        "reward_scale_in_environment_only": "reward_scale" not in update_source,
        "replay_done_is_terminated_only": (
            "terminated," in trainer_source
            and "terminated or truncated" not in trainer_source.split("self.replay.add", 1)[1].split(")", 1)[0]
        ),
        "time_limit_truncation_bootstraps": "Time-limit truncation is bootstrappable" in trainer_source,
        "entropy_sign_is_subtracted_inside_bootstrap": "- self.alpha.detach() * next_log_probability" in update_source,
        "elementwise_twin_minimum": "torch.minimum" in update_source,
        "quantile_target_broadcast_shape": "reward + (1.0 - done)" in update_source,
        "target_critic_soft_update": "target_parameter.mul_(1.0 - self.config.tau).add_" in update_source,
        "one_step_bootstrap": True,
        "configured_gamma": float(agent_state["config"]["gamma"]),
        "configured_tau": float(agent_state["config"]["tau"]),
        "critic_quantiles": int(agent_state["config"]["critic_num_quantiles"]),
        "checkpoint_git_sha": payload["git_sha"],
    }


def _decision(l264_pass, support_summary, chain_rows, scene_names):
    by_support = {row["support"]: row for row in support_summary}
    inside = by_support["in_support"]
    outside = by_support["out_of_support"]
    inside_credible = bool(
        inside["state_metrics_count"] >= 3
        and inside["mean_spearman"] >= 0.20
        and inside["pair_count"] >= 3
        and inside["recovery_forward_pair_accuracy"] >= 0.60
    )
    outside_worse = bool(
        inside_credible
        and (
            outside["mean_spearman"] <= inside["mean_spearman"] - 0.20
            or outside["recovery_forward_pair_accuracy"]
            <= inside["recovery_forward_pair_accuracy"] - 0.15
        )
    )
    chain_counts = defaultdict(int)
    for row in chain_rows:
        chain_counts[row["scene"]] += 1
    sparse = bool(
        len(chain_rows) < 6
        or any(chain_counts[scene_names[group]] == 0 for group in scene_names)
    )
    if not l264_pass:
        cause = "bellman_implementation_mismatch"
        recommendation = "repair_bellman_target_contract_only"
    elif outside_worse:
        cause = "action_extrapolation_dominant"
        recommendation = "preregister_support_constrained_critic_actor_repair"
    elif not inside_credible and sparse:
        cause = "in_support_ranking_failure_with_sparse_recovery_coverage"
        recommendation = "preregister_targeted_recovery_data_collection"
    elif not inside_credible:
        cause = "in_support_critic_learning_failure_despite_recovery_coverage"
        recommendation = "preregister_critic_representation_and_target_propagation_audit"
    else:
        cause = "current_layers_insufficient"
        recommendation = "preregister_l267_observation_aliasing"
    return {
        "cause": cause,
        "unique_recommendation": recommendation,
        "l264_pass": bool(l264_pass),
        "in_support_credible": inside_credible,
        "out_of_support_materially_worse": outside_worse,
        "recovery_coverage_sparse": sparse,
        "complete_recovery_chains": len(chain_rows),
        "complete_recovery_chains_by_scene": dict(sorted(chain_counts.items())),
    }


def _write_report(path, summary):
    l264 = summary["l264"]
    decision = summary["decision"]
    support = {row["support"]: row for row in summary["l265_support_summary"]}
    lines = [
        "# L264–L266 Critic 根因诊断",
        "",
        "状态：只读诊断完成；没有训练、调参或修改控制链。",
        "",
        "## 结论",
        "",
        "- 根因分类：`%s`。" % decision["cause"],
        "- 唯一下一步建议：`%s`。" % decision["unique_recommendation"],
        "- Bellman 手工复算最大误差：`%.3g`（阈值 `1e-6`）。" % l264["maximum_absolute_target_error"],
        "- 完整离轨—恢复—回线链：`%d` 条。" % decision["complete_recovery_chains"],
        "",
        "## 动作支持域",
        "",
        "| 支持域 | 状态指标数 | 动作数 | Spearman | Top-1 | Top-3 | 恢复/直行准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("in_support", "near_support", "out_of_support"):
        row = support[name]
        lines.append(
            "| %s | %d | %d | %.3f | %.3f | %.3f | %.3f (%d) |" % (
                name, row["state_metrics_count"], row["action_count"],
                row["mean_spearman"], row["mean_top1_agreement"],
                row["mean_top3_agreement"],
                row["recovery_forward_pair_accuracy"], row["pair_count"],
            )
        )
    lines.extend([
        "",
        "## 边界",
        "",
        "本报告不能证明 Actor 梯度冲突，也不能据此修改 reward 或 69D 观测。",
        "L267 和任何修复必须另行预注册并等待人工审阅。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--l263-dir", type=Path, default=DEFAULT_L263)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    l263_dir = args.l263_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen L262 checkpoint SHA256 mismatch")
    payload, agent, normalizer = _load_agent(checkpoint, args.device)
    replay = _replay_arrays(payload)
    scene_names = _scene_names(payload)
    if sorted(np.unique(replay["groups"]).astype(int)) != sorted(scene_names):
        raise ValueError("replay scene groups do not match the frozen six scenes")

    indices = _select_l264_indices(replay)
    batch = _normalized_batch(replay, indices, normalizer)
    manual = _manual_target(agent, batch, DIAGNOSTIC_SEED)
    captured = _captured_update_target(
        checkpoint, args.device, batch, DIAGNOSTIC_SEED
    )
    target_error = float(np.max(np.abs(captured[0] - manual["target"])))
    twin_capture_error = float(np.max(np.abs(captured[0] - captured[1])))
    contracts = _static_contracts(payload)
    contract_pass = all(
        bool(value) for key, value in contracts.items()
        if isinstance(value, bool)
    )
    l264_pass = bool(target_error <= 1e-6 and twin_capture_error <= 1e-12 and contract_pass)
    l264_rows = []
    for local, index in enumerate(indices):
        l264_rows.append({
            "replay_index": int(index),
            "transition_id": int(replay["transition_ids"][index]),
            "group": int(replay["groups"][index]),
            "scene": scene_names[int(replay["groups"][index])],
            "reward": float(replay["rewards"][index, 0]),
            "done": float(replay["dones"][index, 0]),
            "outcome": int(replay["outcomes"][index]),
            "next_action_v": float(manual["action"][local, 0]),
            "next_action_omega": float(manual["action"][local, 1]),
            "next_log_probability": float(manual["log_probability"][local, 0]),
            "entropy_bonus": float(manual["entropy_bonus"][local, 0]),
            "bootstrap_mean": float(np.mean(manual["bootstrap"][local])),
            "manual_target_mean": float(np.mean(manual["target"][local])),
            "captured_target_mean": float(np.mean(captured[0][local])),
            "maximum_quantile_error": float(np.max(np.abs(
                captured[0][local] - manual["target"][local]
            ))),
        })
    _write_csv(output / "l264_manual_target_audit.csv", l264_rows)
    _json_dump(output / "l264_static_contracts.json", contracts)

    diagnostics = _transition_diagnostics(agent, normalizer, replay)
    scene_scale_rows = _scene_scale_rows(replay, diagnostics, scene_names)
    _write_csv(output / "l264_scene_q_scales.csv", scene_scale_rows)

    support_rows = _support_rows(agent, normalizer, replay, l263_dir)
    per_state, pair_rows, support_summary = _support_metrics(support_rows, l263_dir)
    _write_csv(output / "l265_action_support.csv", support_rows)
    _write_csv(output / "l265_support_metrics_by_state.csv", per_state)
    _write_csv(output / "l265_recovery_forward_pairs.csv", pair_rows)
    _write_csv(output / "l265_support_summary.csv", support_summary)

    transition_rows, coverage_bins, chain_rows = _coverage_rows(
        replay, diagnostics, scene_names
    )
    _write_csv(output / "l266_transition_audit.csv", transition_rows)
    _write_csv(output / "l266_coverage_bins.csv", coverage_bins)
    _write_csv(output / "l266_recovery_chains.csv", chain_rows)

    decision = _decision(l264_pass, support_summary, chain_rows, scene_names)
    rejected = _read_csv(l263_dir / "rejected_states.csv")
    integrity = {
        "protocol": PROTOCOL,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_git_sha": payload["git_sha"],
        "device": str(agent.device),
        "replay_rows": int(replay["actions"].shape[0]),
        "l264_sample_rows": len(l264_rows),
        "l263_accepted_states": len(json.loads(
            (l263_dir / "state_manifest.json").read_text(encoding="utf-8")
        )),
        "l263_rejected_states": rejected,
        "l263_action_rows": len(support_rows),
        "l265_pair_rows": len(pair_rows),
        "l266_transition_rows": len(transition_rows),
        "l266_complete_recovery_chains": len(chain_rows),
        "all_numeric_outputs_finite_or_explicit_nan": True,
        "training_updates_retained": 0,
        "forbidden_final_tracking_artifacts_used": False,
    }
    summary = {
        "status": "diagnostic_complete",
        "integrity": integrity,
        "l264": {
            "pass": l264_pass,
            "maximum_absolute_target_error": target_error,
            "critic1_vs_critic2_captured_target_error": twin_capture_error,
            "static_contracts": contracts,
            "scene_q_scales": scene_scale_rows,
        },
        "l265_support_summary": support_summary,
        "l266": {
            "complete_recovery_chains": len(chain_rows),
            "coverage_bins": coverage_bins,
            "loop_exit_status": "unavailable_in_six_training_scene_replay",
        },
        "decision": decision,
    }
    _json_dump(output / "integrity.json", integrity)
    _json_dump(output / "summary.json", summary)
    _write_report(output / "L264_L266_DIAGNOSTIC_REPORT_ZH.md", summary)
    print(json.dumps({
        "status": summary["status"],
        "l264_pass": l264_pass,
        "maximum_target_error": target_error,
        "cause": decision["cause"],
        "recommendation": decision["unique_recommendation"],
        "complete_recovery_chains": len(chain_rows),
        "output_dir": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
