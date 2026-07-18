#!/usr/bin/env python3
"""Run paired BC/correction critic-advantage diagnostics on fixed seeds."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, git_sha, load_yaml
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.environment import MppiPriorEnv
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


CONDITIONS = {
    "bc": ("none", "online"),
    "correction_none": ("none", "online"),
    "correction_online_hard": ("hard", "online"),
    "correction_target_hard": ("hard", "target"),
}


def _seeds(text):
    values = [int(value) for value in str(text).split(",") if value.strip()]
    if not values:
        raise ValueError("at least one diagnostic seed is required")
    if len(values) != len(set(values)):
        raise ValueError("diagnostic seeds must be unique")
    if any(value < 0 or value > 2 ** 32 - 1 for value in values):
        raise ValueError("diagnostic seeds must be in [0, 2**32 - 1]")
    return values


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _evaluation_config(config_path, scene_path):
    base = load_yaml(_resolved_path(config_path))
    scene = load_yaml(_resolved_path(scene_path))
    overlay = {
        "planner": copy.deepcopy(base["planner"]),
        "rl": copy.deepcopy(base.get("rl", {})),
    }
    if "memory" in base:
        overlay["memory"] = copy.deepcopy(base["memory"])
    resolved = deep_merge(scene, overlay)
    resolved.setdefault("rl", {}).setdefault(
        "intrinsic_exploration", {}
    )["enabled"] = False
    resolved["rl"].setdefault("training", {})[
        "initial_state_noise"
    ] = [0.0, 0.0, 0.0, 0.0, 0.0]
    return resolved


def _load_agent(path, device):
    payload = load_sac_checkpoint(path, map_location=device)
    state = payload["agent"]
    agent = SACAgent(
        state["observation_dim"],
        state["action_dim"],
        SACConfig.from_mapping(state["config"]),
        device=device,
    )
    agent.load_state_dict(state, load_optimizers=False)
    agent.eval()
    normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
    if normalizer.dimension != agent.observation_dim:
        raise ValueError("checkpoint normalizer and agent dimensions differ")
    if not agent.is_correction_policy:
        raise ValueError("L18 requires a frozen_bc_correction checkpoint")
    return agent, normalizer, payload


def _aggregate_step_diagnostics(step_rows):
    if not step_rows:
        raise RuntimeError("diagnostic episode produced no steps")
    online = np.asarray([
        row["online_conservative_advantage"] for row in step_rows
    ], dtype=np.float64)
    target = np.asarray([
        row["target_conservative_advantage"] for row in step_rows
    ], dtype=np.float64)
    gate = np.asarray([
        row["correction_advantage_gate_alpha"] for row in step_rows
    ], dtype=np.float64)
    raw = np.asarray([
        row["raw_applied_correction_abs_mean"] for row in step_rows
    ], dtype=np.float64)
    applied = np.asarray([
        row["applied_correction_abs_mean"] for row in step_rows
    ], dtype=np.float64)
    selected_lcb = np.asarray([
        row.get("selected_consensus_lcb", 0.0) for row in step_rows
    ], dtype=np.float64)
    return {
        "online_conservative_advantage_mean": float(online.mean()),
        "online_conservative_advantage_min": float(online.min()),
        "online_conservative_advantage_max": float(online.max()),
        "online_positive_advantage_fraction": float(np.mean(online >= 0.0)),
        "target_conservative_advantage_mean": float(target.mean()),
        "target_conservative_advantage_min": float(target.min()),
        "target_conservative_advantage_max": float(target.max()),
        "target_positive_advantage_fraction": float(np.mean(target >= 0.0)),
        "advantage_gate_accept_fraction": float(gate.mean()),
        "raw_applied_correction_abs_mean": float(raw.mean()),
        "applied_correction_abs_mean": float(applied.mean()),
        "applied_correction_abs_max": float(max(
            row["applied_correction_abs_max"] for row in step_rows
        )),
        "selected_consensus_lcb_mean": float(selected_lcb.mean()),
        "selected_consensus_lcb_min": float(selected_lcb.min()),
        "selected_consensus_lcb_max": float(selected_lcb.max()),
    }


def _run_episode(
    resolved_config,
    agent,
    normalizer,
    seed,
    condition,
    gate_mode,
    critic_source,
    threshold,
    uncertainty_multiplier=1.0,
):
    environment = MppiPriorEnv(resolved_config, ROOT, seed=seed)
    observation, reset_info = environment.reset(seed=seed)
    total_reward = 0.0
    step_rows = []
    last_info = {}
    try:
        while True:
            normalized = normalizer.normalize(observation)
            candidate, policy = agent.select_action(
                normalized, deterministic=True
            )
            action, advantage = agent.filter_correction_by_advantage(
                normalized,
                candidate,
                gate_mode=gate_mode,
                critic_source=critic_source,
                threshold=threshold,
                uncertainty_multiplier=uncertainty_multiplier,
            )
            diagnostics = dict(policy)
            diagnostics.update(advantage)
            observation, reward, terminated, truncated, last_info = (
                environment.step(action)
            )
            total_reward += float(reward)
            row = {
                "condition": condition,
                "seed": int(seed),
                "step": len(step_rows),
                "reward": float(reward),
                "goal_distance": float(last_info.get(
                    "goal_distance", float("inf")
                )),
                "success": bool(last_info.get("success", False)),
                "collision": bool(last_info.get("collision", False)),
            }
            row.update(diagnostics)
            step_rows.append(row)
            if terminated or truncated:
                break
    finally:
        environment.close()
    episode = {
        "condition": condition,
        "seed": int(seed),
        "scene": str(last_info.get(
            "scene", reset_info.get("scene", "unknown")
        )),
        "steps": len(step_rows),
        "return": float(total_reward),
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "goal_distance": float(last_info.get("goal_distance", float("inf"))),
        "gate_mode": gate_mode,
        "critic_source": critic_source,
        "advantage_threshold": float(threshold),
        "advantage_uncertainty_multiplier": float(
            uncertainty_multiplier
        ),
    }
    episode.update(_aggregate_step_diagnostics(step_rows))
    return episode, step_rows


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(left, right, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return None
    if rank:
        left = _rank(left)
        right = _rank(right)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _paired_rows(episodes):
    baseline = {
        int(row["seed"]): row for row in episodes if row["condition"] == "bc"
    }
    rows = []
    for candidate in episodes:
        if candidate["condition"] == "bc":
            continue
        reference = baseline.get(int(candidate["seed"]))
        if reference is None:
            raise ValueError("candidate episode has no paired BC seed")
        row = dict(candidate)
        row.update({
            "bc_return": float(reference["return"]),
            "bc_success": bool(reference["success"]),
            "bc_collision": bool(reference["collision"]),
            "bc_goal_distance": float(reference["goal_distance"]),
            "return_delta": float(candidate["return"] - reference["return"]),
            "goal_distance_improvement": float(
                reference["goal_distance"] - candidate["goal_distance"]
            ),
            "success_delta": int(candidate["success"])
            - int(reference["success"]),
            "collision_delta": int(candidate["collision"])
            - int(reference["collision"]),
            "bc_success_lost": bool(
                reference["success"] and not candidate["success"]
            ),
            "success_gained": bool(
                not reference["success"] and candidate["success"]
            ),
            "collision_regression": bool(
                not reference["collision"] and candidate["collision"]
            ),
        })
        rows.append(row)
    return rows


def _condition_summary(rows):
    result = {}
    for condition in sorted(set(row["condition"] for row in rows)):
        group = [row for row in rows if row["condition"] == condition]
        online = [row["online_conservative_advantage_mean"] for row in group]
        target = [row["target_conservative_advantage_mean"] for row in group]
        return_delta = [row["return_delta"] for row in group]
        distance_delta = [row["goal_distance_improvement"] for row in group]
        result[condition] = {
            "episodes": len(group),
            "successes": int(sum(row["success"] for row in group)),
            "collisions": int(sum(row["collision"] for row in group)),
            "bc_success_losses": int(sum(
                row["bc_success_lost"] for row in group
            )),
            "success_gains": int(sum(row["success_gained"] for row in group)),
            "collision_regressions": int(sum(
                row["collision_regression"] for row in group
            )),
            "mean_return_delta": float(np.mean(return_delta)),
            "mean_goal_distance_improvement": float(np.mean(distance_delta)),
            "mean_online_conservative_advantage": float(np.mean(online)),
            "mean_target_conservative_advantage": float(np.mean(target)),
            "mean_gate_accept_fraction": float(np.mean([
                row["advantage_gate_accept_fraction"] for row in group
            ])),
            "online_return_pearson": _correlation(online, return_delta),
            "online_return_spearman": _correlation(
                online, return_delta, rank=True
            ),
            "online_distance_pearson": _correlation(online, distance_delta),
            "target_return_pearson": _correlation(target, return_delta),
            "target_return_spearman": _correlation(
                target, return_delta, rank=True
            ),
            "target_distance_pearson": _correlation(target, distance_delta),
        }
    return result


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty diagnostic table")
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        default="bc,correction_none,correction_online_hard,correction_target_hard",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold):
        raise ValueError("--threshold must be finite")
    selected_conditions = [
        value.strip() for value in args.conditions.split(",") if value.strip()
    ]
    unknown = [value for value in selected_conditions if value not in CONDITIONS]
    if unknown:
        raise ValueError("unknown diagnostic conditions: %s" % unknown)
    if "bc" not in selected_conditions:
        raise ValueError("paired diagnostic conditions must include bc")

    seeds = _seeds(args.seeds)
    resolved = _evaluation_config(args.config, args.scene_config)
    baseline_path = _resolved_path(args.baseline_checkpoint)
    candidate_path = _resolved_path(args.candidate_checkpoint)
    baseline_agent, baseline_normalizer, baseline_payload = _load_agent(
        baseline_path, args.device
    )
    candidate_agent, candidate_normalizer, candidate_payload = _load_agent(
        candidate_path, args.device
    )
    if (
        baseline_agent.observation_dim != candidate_agent.observation_dim
        or baseline_agent.action_dim != candidate_agent.action_dim
    ):
        raise ValueError("baseline and candidate checkpoint dimensions differ")
    if (
        baseline_agent.frozen_base_actor_sha256()
        != candidate_agent.frozen_base_actor_sha256()
    ):
        raise ValueError("baseline and candidate do not share the same frozen BC")

    episodes = []
    steps = []
    for condition in selected_conditions:
        gate_mode, critic_source = CONDITIONS[condition]
        agent = baseline_agent if condition == "bc" else candidate_agent
        normalizer = (
            baseline_normalizer if condition == "bc" else candidate_normalizer
        )
        for seed in seeds:
            episode, episode_steps = _run_episode(
                resolved,
                agent,
                normalizer,
                seed,
                condition,
                gate_mode,
                critic_source,
                args.threshold,
            )
            episode["training_seed"] = int(args.training_seed)
            episode["checkpoint"] = str(
                baseline_path if condition == "bc" else candidate_path
            )
            for row in episode_steps:
                row["training_seed"] = int(args.training_seed)
                row["checkpoint"] = episode["checkpoint"]
            episodes.append(episode)
            steps.extend(episode_steps)
            print(json.dumps({
                "condition": condition,
                "seed": seed,
                "success": episode["success"],
                "return": episode["return"],
                "goal_distance": episode["goal_distance"],
                "online_advantage": episode[
                    "online_conservative_advantage_mean"
                ],
                "target_advantage": episode[
                    "target_conservative_advantage_mean"
                ],
                "accept_fraction": episode[
                    "advantage_gate_accept_fraction"
                ],
            }, sort_keys=True))

    paired = _paired_rows(episodes)
    summary = {
        "training_seed": int(args.training_seed),
        "diagnostic_seeds": seeds,
        "scene": str(resolved.get("scene", {}).get("name", "unknown")),
        "threshold": float(args.threshold),
        "baseline_checkpoint": str(baseline_path),
        "candidate_checkpoint": str(candidate_path),
        "baseline_git_sha": baseline_payload.get("git_sha"),
        "candidate_git_sha": candidate_payload.get("git_sha"),
        "run_git_sha": git_sha(ROOT),
        "frozen_base_actor_sha256": (
            candidate_agent.frozen_base_actor_sha256()
        ),
        "conditions": _condition_summary(paired),
    }
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "paired_episodes.csv", paired)
    _write_csv(output / "steps.csv", steps)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output / "config_snapshot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
