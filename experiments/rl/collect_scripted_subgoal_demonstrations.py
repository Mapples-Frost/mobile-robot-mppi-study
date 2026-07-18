#!/usr/bin/env python3
"""Collect leakage-safe behavior-cloning demonstrations in MuJoCo.

The privileged polyline is a data-collection teacher only.  It never enters
the MPPI obstacle input and never enters a student shard.  MPPI continues to
see obstacles exclusively through LaserScan/local_obstacle_layer, with the
unchanged safety arbiter between proposed and executed control.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.demonstrations import (
    DEMONSTRATION_SCHEMA,
    DEMONSTRATION_SCHEMA_VERSION,
    DEMONSTRATION_SPLITS,
    split_episode_seeds,
    write_demonstration_manifest,
    write_demonstration_shard,
)
from mobile_robot_mppi.rl.environment import MppiPriorEnv
from mobile_robot_mppi.rl.scripted_subgoal import (
    ScriptedPolylineSubgoal,
    ScriptedSubgoalConfig,
)

from experiments.rl.run_scripted_subgoal_upper_bound import (
    _offline_route,
    _resolved_scene,
)


DEFAULT_SCENES = ("configs/research/mujoco_u_trap_long_board.yaml",)


def _csv_values(text, cast=int):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def _slug(value):
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "scene"


def _write_csv(path, rows, fieldnames):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_episode(environment, policy, episode_id, seed, split):
    """Collect one episode from exact environment observations and teacher actions."""

    observation, reset_info = environment.reset(seed=int(seed))
    observation = np.asarray(observation, dtype=np.float32).copy()
    policy.reset()
    observations = []
    teacher_actions = []
    episode_ids = []
    steps = []
    audit_rows = []
    last_info = {}
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        # The teacher is privileged because its internal polyline was computed
        # offline.  Its pose is the same perceived pose available to the stack,
        # while simulator truth below is written only to the audit sidecar.
        perceived_pose = environment.perceived.observation.pose
        truth_before = environment.truth.pose
        action, teacher_diagnostic = policy.action(perceived_pose)
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape != (2,) or not np.isfinite(action).all():
            raise FloatingPointError("teacher must emit a finite normalized action [2]")
        if np.any(action < -1.000001) or np.any(action > 1.000001):
            raise FloatingPointError("teacher action left normalized [-1, 1] range")

        observations.append(observation.copy())
        teacher_actions.append(action.copy())
        episode_ids.append(int(episode_id))
        steps.append(int(step))
        next_observation, reward, terminated, truncated, last_info = environment.step(
            action
        )
        next_observation = np.asarray(next_observation, dtype=np.float32).copy()
        if next_observation.shape != observation.shape or not np.isfinite(
            next_observation
        ).all():
            raise FloatingPointError("environment returned an invalid encoded observation")
        truth_after = environment.truth
        proposed = np.asarray(last_info["proposed_control"], dtype=np.float64)
        executed = np.asarray(last_info["executed_control"], dtype=np.float64)
        audit_rows.append({
            "episode_id": int(episode_id),
            "split": str(split),
            "seed": int(seed),
            "step": int(step),
            "time_after_step": float(truth_after.timestamp),
            "truth_x_before": float(truth_before.x),
            "truth_y_before": float(truth_before.y),
            "truth_theta_before": float(truth_before.theta),
            "perceived_x_before": float(perceived_pose.x),
            "perceived_y_before": float(perceived_pose.y),
            "perceived_theta_before": float(perceived_pose.theta),
            "truth_x_after": float(truth_after.pose.x),
            "truth_y_after": float(truth_after.pose.y),
            "truth_theta_after": float(truth_after.pose.theta),
            "teacher_distance_action": float(action[0]),
            "teacher_bearing_action": float(action[1]),
            "route_progress": float(teacher_diagnostic["route_progress"]),
            "target_progress": float(teacher_diagnostic["target_progress"]),
            "route_length": float(teacher_diagnostic["route_length"]),
            "cross_track_error": float(teacher_diagnostic["cross_track_error"]),
            "target_x": float(teacher_diagnostic["target_x"]),
            "target_y": float(teacher_diagnostic["target_y"]),
            "target_distance": float(teacher_diagnostic["target_distance"]),
            "target_bearing": float(teacher_diagnostic["target_bearing"]),
            "proposed_v": float(proposed[0]),
            "proposed_omega": float(proposed[1]),
            "executed_v": float(executed[0]),
            "executed_omega": float(executed[1]),
            "reward": float(reward),
            "goal_distance": float(last_info["goal_distance"]),
            "minimum_clearance": float(last_info["minimum_clearance"]),
            "safety_override": bool(last_info["safety_override"]),
            "collision": bool(last_info["collision"]),
            "success": bool(last_info["success"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        observation = next_observation
        step += 1

    arrays = {
        "observation": np.asarray(observations, dtype=np.float32),
        "teacher_action": np.asarray(teacher_actions, dtype=np.float32),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
    }
    summary = {
        "episode_id": int(episode_id),
        "scene": str(environment.config["scene"]["name"]),
        "split": str(split),
        "seed": int(seed),
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "steps": int(step),
        "final_goal_distance": float(last_info.get("goal_distance", float("inf"))),
        "minimum_clearance": float(min(
            row["minimum_clearance"] for row in audit_rows
        )),
        "safety_interventions": int(sum(
            int(row["safety_override"]) for row in audit_rows
        )),
        "pose_source": str(environment.config["sensors"].get(
            "pose_source", "wheel_odometry"
        )),
        "twist_source": str(environment.config["sensors"].get(
            "twist_source", "wheel_odometry"
        )),
        "reset_goal_distance": float(reset_info["goal_distance"]),
        "included_in_training_shard": bool(last_info.get("success", False)),
    }
    return arrays, audit_rows, summary


def _empty_arrays(observation_dim):
    return {
        "observation": np.empty((0, int(observation_dim)), dtype=np.float32),
        "teacher_action": np.empty((0, 2), dtype=np.float32),
        "episode_id": np.empty((0,), dtype=np.int64),
        "step": np.empty((0,), dtype=np.int64),
    }


def _concatenate_episodes(episodes, observation_dim):
    if not episodes:
        return _empty_arrays(observation_dim)
    return {
        key: np.concatenate([episode[key] for episode in episodes], axis=0)
        for key in ("observation", "teacher_action", "episode_id", "step")
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl-config",
        default="configs/rl/sac_mppi_utrap_exploration_l11_no_intrinsic.yaml",
    )
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        default="41,42,43,44,45,46,47,48,49,50,51,52,53,54,55",
    )
    parser.add_argument("--lookahead", type=float, default=0.70)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--route-margin", type=float, default=0.20)
    parser.add_argument("--endpoint-route-margin", type=float, default=0.10)
    parser.add_argument("--route-resolution", type=float, default=0.04)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260714)
    parser.add_argument(
        "--allow-non-ground-truth-localization",
        action="store_true",
        help="explicit robustness dataset; default controlled condition requires ground truth",
    )
    parser.add_argument(
        "--allow-empty-splits",
        action="store_true",
        help="write an empty split if every assigned teacher episode failed",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    seeds = _csv_values(args.seeds, int)
    if not np.isfinite(args.lookahead) or args.lookahead <= 0.0:
        raise ValueError("teacher lookahead must be positive and finite")
    if args.num_samples <= 0 or args.max_steps <= 0:
        raise ValueError("num_samples and max_steps must be positive")
    split_plan = split_episode_seeds(
        seeds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    seed_to_split = {
        int(seed): name
        for name, values in split_plan.items()
        for seed in values
    }
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is not empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit" / "routes").mkdir(parents=True, exist_ok=True)
    (output / "audit" / "resolved_configs").mkdir(parents=True, exist_ok=True)
    (output / "audit" / "trajectories").mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir(parents=True, exist_ok=True)

    rl_config = load_yaml(args.rl_config)
    successful = {name: [] for name in DEMONSTRATION_SPLITS}
    episode_summaries = []
    observation_dim = None
    observation_config = None
    prior_config = None
    sensor_conditions = {}
    seen_scene_names = set()
    episode_id = 0

    for scene_path in args.configs:
        template = _resolved_scene(
            rl_config, scene_path, seeds[0], args.num_samples, args.max_steps
        )
        scene_name = str(template["scene"]["name"])
        if scene_name in seen_scene_names:
            raise ValueError("duplicate scene name in demonstration collection: %s" % scene_name)
        seen_scene_names.add(scene_name)
        scene_slug = _slug(scene_name)
        route, route_audit = _offline_route(
            template,
            args.route_margin,
            args.route_resolution,
            endpoint_margin=args.endpoint_route_margin,
        )
        with (output / "audit" / "routes" / (scene_slug + ".json")).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {"route": route.tolist(), "feasibility_audit": route_audit},
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

        for current_seed in seeds:
            split = seed_to_split[int(current_seed)]
            config = _resolved_scene(
                rl_config,
                scene_path,
                current_seed,
                args.num_samples,
                args.max_steps,
            )
            environment = MppiPriorEnv(config, ROOT, seed=current_seed)
            if environment.encoder.config.include_absolute_pose:
                environment.close()
                raise ValueError(
                    "behavior-cloning demonstrations forbid absolute pose in student observations"
                )
            if environment.policy_action_dim != 2:
                environment.close()
                raise ValueError(
                    "scripted behavior cloning requires a two-dimensional local-subgoal prior"
                )
            pose_source = str(environment.config["sensors"].get(
                "pose_source", "wheel_odometry"
            ))
            twist_source = str(environment.config["sensors"].get(
                "twist_source", "wheel_odometry"
            ))
            if (
                not args.allow_non_ground_truth_localization
                and (pose_source != "ground_truth" or twist_source != "ground_truth")
            ):
                environment.close()
                raise ValueError(
                    "controlled BC collection requires ground_truth pose/twist; "
                    "use --allow-non-ground-truth-localization for an explicit robustness dataset"
                )
            current_observation_config = environment.encoder.config.to_dict()
            current_prior_config = environment.parameterization.config.to_dict()
            if observation_dim is None:
                observation_dim = int(environment.observation_dim)
                observation_config = current_observation_config
                prior_config = current_prior_config
            elif (
                observation_dim != int(environment.observation_dim)
                or observation_config != current_observation_config
                or prior_config != current_prior_config
            ):
                environment.close()
                raise ValueError(
                    "all demonstration scenes must share one observation/prior contract"
                )
            sensor_conditions[scene_name] = {
                "pose_source": pose_source,
                "twist_source": twist_source,
            }
            policy = ScriptedPolylineSubgoal(
                route,
                environment.config["rl"]["prior"],
                ScriptedSubgoalConfig(lookahead_distance=float(args.lookahead)),
            )
            try:
                arrays, audit_rows, summary = _collect_episode(
                    environment,
                    policy,
                    episode_id=episode_id,
                    seed=current_seed,
                    split=split,
                )
                # Resolved configs contain simulator scene/plant truth and are
                # therefore retained exclusively in the audit sidecar.
                config_path = (
                    output / "audit" / "resolved_configs"
                    / ("episode_%06d.json" % episode_id)
                )
                with config_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        environment.config,
                        handle,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
            finally:
                environment.close()
            audit_path = (
                output / "audit" / "trajectories"
                / ("episode_%06d.csv" % episode_id)
            )
            _write_csv(audit_path, audit_rows, list(audit_rows[0]))
            episode_summaries.append(summary)
            if summary["success"]:
                successful[split].append({
                    **arrays,
                    "seed": int(current_seed),
                    "scene": scene_name,
                })
            print(
                "episode=%d scene=%s split=%s seed=%d success=%s distance=%.3f collision=%s"
                % (
                    episode_id,
                    scene_name,
                    split,
                    current_seed,
                    summary["success"],
                    summary["final_goal_distance"],
                    summary["collision"],
                )
            )
            episode_id += 1

    if observation_dim is None:
        raise RuntimeError("no demonstration episode was collected")
    split_descriptors = {}
    for split in DEMONSTRATION_SPLITS:
        arrays = _concatenate_episodes(successful[split], observation_dim)
        if not args.allow_empty_splits and split_plan[split] and not successful[split]:
            raise RuntimeError(
                "all teacher episodes assigned to %s failed; refusing empty research split"
                % split
            )
        shard_path = output / "splits" / (split + ".npz")
        shard = write_demonstration_shard(
            shard_path, arrays, allow_empty=True
        )
        split_descriptors[split] = {
            "file": str(shard_path.relative_to(output).as_posix()),
            "sha256": shard["sha256"],
            "bytes": int(shard["bytes"]),
            "samples": int(shard["sample_count"]),
            "episodes": int(shard["episode_count"]),
            "seeds": sorted(set(
                int(episode["seed"]) for episode in successful[split]
            )),
        }

    summary_fields = list(episode_summaries[0])
    _write_csv(
        output / "audit" / "episodes.csv", episode_summaries, summary_fields
    )
    success_count = int(sum(int(row["success"]) for row in episode_summaries))
    failure_count = len(episode_summaries) - success_count
    manifest = {
        "schema": DEMONSTRATION_SCHEMA,
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "config": {
            "rl_config": str(Path(args.rl_config)),
            "scene_configs": [str(Path(value)) for value in args.configs],
            "lookahead": float(args.lookahead),
            "num_samples": int(args.num_samples),
            "max_steps": int(args.max_steps),
            "route_margin": float(args.route_margin),
            "endpoint_route_margin": float(args.endpoint_route_margin),
            "route_resolution": float(args.route_resolution),
            "split_seed": int(args.split_seed),
            "validation_fraction": float(args.validation_fraction),
            "test_fraction": float(args.test_fraction),
            "sensor_conditions": sensor_conditions,
        },
        "observation_dim": int(observation_dim),
        "action_dim": 2,
        "observation_encoder": observation_config,
        "prior_parameterization": prior_config,
        "teacher": {
            "class": "ScriptedPolylineSubgoal",
            "action_space": "normalized_local_subgoal_distance_bearing",
            "student_observation_source": "MppiPriorEnv.reset_and_step",
            "privileged_route_training_only": True,
        },
        "split_plan": {
            name: [int(value) for value in split_plan[name]]
            for name in DEMONSTRATION_SPLITS
        },
        "splits": split_descriptors,
        "counts": {
            "requested_episodes": len(episode_summaries),
            "successful_episodes": success_count,
            "failed_episodes": failure_count,
            "training_samples": int(sum(
                value["samples"] for value in split_descriptors.values()
            )),
            "failed_episodes_in_training_shards": 0,
        },
        "audit": {
            "directory": "audit",
            "included_in_training_shards": False,
            "contains_privileged_route": True,
            "contains_simulator_truth": True,
            "contains_teacher_waypoints": True,
            "episodes_csv": "audit/episodes.csv",
        },
    }
    write_demonstration_manifest(output, manifest)
    print(json.dumps({
        "dataset": str(output),
        "counts": manifest["counts"],
        "splits": manifest["splits"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
