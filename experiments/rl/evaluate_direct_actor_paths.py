#!/usr/bin/env python3
"""Evaluate a frozen direct-control SAC Actor on explicit path/domain blocks."""

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
from mobile_robot_mppi.rl.environment import DirectControlEnv
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


def _csv_values(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _environment_configs(base, scene_paths, domain_path, domain_names):
    domains = [{"name": "embedded", "plant_override": {}, "role": "embedded"}]
    if domain_path is not None:
        specification = load_yaml(domain_path)
        declared = list(specification.get("physics_domains", {}).get("domains", ()))
        lookup = {str(item["name"]): item for item in declared}
        missing = [name for name in domain_names if name not in lookup]
        if missing:
            raise ValueError("unknown physics domains: %s" % missing)
        domains = [lookup[name] for name in domain_names]
    result = []
    for scene_path in scene_paths:
        scene = load_yaml(scene_path)
        controlled = {
            "planner": copy.deepcopy(base["planner"]),
            "rl": copy.deepcopy(base["rl"]),
        }
        scene = deep_merge(scene, controlled)
        for domain in domains:
            item = deep_merge(
                scene, {"plant": copy.deepcopy(domain.get("plant_override", {}))}
            )
            scene_name = str(item.get("scene", {}).get("name", "scene"))
            domain_name = str(domain["name"])
            item.setdefault("scene", {})["name"] = "%s__%s" % (
                scene_name, domain_name
            )
            item.setdefault("experiment", {})["physics_domain"] = domain_name
            item["experiment"]["physics_domain_role"] = str(
                domain.get("role", "unknown")
            )
            result.append(item)
    return result


def _load_actor(checkpoint, device):
    payload = load_sac_checkpoint(checkpoint, map_location=device)
    state = payload["agent"]
    agent = SACAgent(
        int(state["observation_dim"]),
        int(state["action_dim"]),
        SACConfig.from_mapping(state["config"]),
        device=device,
        seed=0,
    )
    agent.load_state_dict(state, load_optimizers=False)
    agent.eval()
    return payload, agent, RunningNormalizer.from_state_dict(
        payload["normalizer"]
    )


def _episode(config, checkpoint, agent, normalizer, seed, output_dir):
    environment = DirectControlEnv(config, ROOT, seed=seed)
    observation, _ = environment.reset(seed=seed)
    if environment.observation_dim != agent.observation_dim:
        environment.close()
        raise ValueError(
            "checkpoint/environment observation dimensions differ: %d != %d"
            % (agent.observation_dim, environment.observation_dim)
        )
    if environment.policy_action_dim != agent.action_dim:
        environment.close()
        raise ValueError("checkpoint/environment action dimensions differ")
    rows = []
    total_return = 0.0
    try:
        while True:
            normalized = normalizer.normalize(observation)
            action, _ = agent.select_action(normalized, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(
                action
            )
            total_return += float(reward)
            rows.append({
                "step": len(rows),
                "time": float(environment.truth.timestamp),
                "x": float(environment.truth.pose.x),
                "y": float(environment.truth.pose.y),
                "theta": float(environment.truth.pose.theta),
                "v": float(environment.truth.twist.v),
                "omega": float(environment.truth.twist.omega),
                "normalized_action_0": float(action[0]),
                "normalized_action_1": float(action[1]),
                "executed_v": float(info["executed_control"][0]),
                "executed_omega": float(info["executed_control"][1]),
                "cross_track_error": float(info["cross_track_error"]),
                "path_progress": float(info["path_progress"]),
                "path_remaining": float(info["path_remaining"]),
                "path_heading_error": float(info["path_heading_error"]),
                "minimum_clearance": float(info["minimum_clearance"]),
                "reward": float(reward),
                "success": bool(info["success"]),
                "collision": bool(info["collision"]),
            })
            if terminated or truncated:
                break
    finally:
        environment.close()
    if not rows:
        raise RuntimeError("direct Actor episode produced no steps")
    executed = np.asarray([
        (row["executed_v"], row["executed_omega"]) for row in rows
    ])
    control_dt = float(config["experiment"]["control_dt"])
    differences = np.diff(executed, axis=0) / control_dt
    jerk = (
        0.0
        if differences.size == 0
        else float(np.mean(np.linalg.norm(differences, axis=1)))
    )
    total_length = float(
        environment.components["reference"].total_length
    )
    final = rows[-1]
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "scene": str(config["scene"]["name"]),
        "physics_domain": str(config["experiment"]["physics_domain"]),
        "seed": int(seed),
        "steps": len(rows),
        "success": bool(final["success"]),
        "collision": bool(final["collision"]),
        "return": float(total_return),
        "cross_track_rmse": float(np.sqrt(np.mean([
            row["cross_track_error"] ** 2 for row in rows
        ]))),
        "cross_track_max": float(max(
            row["cross_track_error"] for row in rows
        )),
        "heading_rmse": float(np.sqrt(np.mean([
            row["path_heading_error"] ** 2 for row in rows
        ]))),
        "path_completion_ratio": float(np.clip(
            final["path_progress"] / max(total_length, 1e-12), 0.0, 1.0
        )),
        "minimum_clearance": float(min(
            row["minimum_clearance"] for row in rows
        )),
        "control_jerk": jerk,
    }
    if not all(
        np.isfinite(value)
        for value in summary.values()
        if isinstance(value, float)
    ):
        raise FloatingPointError("direct Actor evaluation produced NaN or Inf")
    _write_csv(output_dir / "trajectory.csv", rows)
    return summary


def summarize(rows):
    if not rows:
        raise ValueError("at least one direct Actor episode is required")
    numeric = (
        "return",
        "cross_track_rmse",
        "cross_track_max",
        "heading_rmse",
        "path_completion_ratio",
        "minimum_clearance",
        "control_jerk",
    )
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "collision_rate": float(np.mean([float(row["collision"]) for row in rows])),
        **{
            "mean_%s" % name: float(np.mean([row[name] for row in rows]))
            for name in numeric
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-configs", nargs="+", required=True)
    parser.add_argument("--physics-domain-config")
    parser.add_argument("--physics-domains", default="")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    base = load_yaml(args.config)
    domains = _csv_values(args.physics_domains)
    if bool(args.physics_domain_config) != bool(domains):
        raise ValueError(
            "physics domain config and non-empty domains are required together"
        )
    configs = _environment_configs(
        base,
        [Path(value).resolve() for value in args.scene_configs],
        None if args.physics_domain_config is None else Path(
            args.physics_domain_config
        ).resolve(),
        domains,
    )
    payload, agent, normalizer = _load_actor(args.checkpoint, args.device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = _csv_values(args.seeds, int)
    snapshot = {
        "schema_version": 1,
        "git_sha": git_sha(ROOT),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_git_sha": payload.get("git_sha"),
        "config": str(Path(args.config).resolve()),
        "scene_configs": [
            str(Path(value).resolve()) for value in args.scene_configs
        ],
        "physics_domain_config": (
            None
            if args.physics_domain_config is None
            else str(Path(args.physics_domain_config).resolve())
        ),
        "physics_domains": domains,
        "seeds": seeds,
        "resolved_environments": configs,
    }
    with (output / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows = []
    for config in configs:
        for seed in seeds:
            run_name = "%s/seed_%04d" % (config["scene"]["name"], seed)
            rows.append(_episode(
                config,
                args.checkpoint,
                agent,
                normalizer,
                seed,
                output / "runs" / run_name,
            ))
    _write_csv(output / "episodes.csv", rows)
    result = summarize(rows)
    result["checkpoint"] = str(Path(args.checkpoint).resolve())
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
