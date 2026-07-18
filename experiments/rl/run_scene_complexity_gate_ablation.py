#!/usr/bin/env python3
"""Run one training seed of the preregistered L25 paired scene ablation."""

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.core.spaces import action_spec_from_config
from mobile_robot_mppi.planning.mppi import MppiConfig
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.rl.prior import TorchSACPrior
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config


CONDITIONS = (
    "traditional_mppi",
    "frozen_bc_prior",
    "lcb_always",
    "complexity_lcb",
)

STEP_FIELDS = (
    "time",
    "goal_distance",
    "clearance",
    "collision",
    "safety_override",
    "executed_v",
    "executed_omega",
    "rl_gate_alpha",
    "rl_correction_advantage_gate_alpha",
    "rl_scene_complexity_score",
    "rl_scene_complexity_front_proximity",
    "rl_scene_complexity_constriction",
    "rl_scene_complexity_density",
    "rl_scene_complexity_front_clearance_m",
    "rl_scene_complexity_left_clearance_m",
    "rl_scene_complexity_right_clearance_m",
    "rl_scene_complexity_near_obstacle_fraction",
    "rl_scene_complexity_scan_valid",
    "rl_learned_inference_skipped",
)


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty L25 table")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scene_entries(base):
    design = base["rl"]["scene_complexity_ablation"]
    result = []
    for role, key in (("simple", "simple_scenes"), ("complex", "complex_scenes")):
        for value in design[key]:
            path = _resolved_path(value)
            scene = load_yaml(path)
            result.append({
                "role": role,
                "path": path,
                "name": str(scene["scene"]["name"]),
            })
    names = [row["name"] for row in result]
    if len(names) != len(set(names)):
        raise ValueError("L25 scene names must be unique")
    return result


def _condition_config(base, scene_path, condition, checkpoint, seed):
    resolved = _apply_scene_config(base, scene_path)
    resolved = copy.deepcopy(resolved)
    resolved["experiment"]["seed"] = int(seed)
    resolved["experiment"]["name"] = "l25_%s_seed_%d" % (
        condition,
        int(seed),
    )
    sensor_overrides = dict(resolved.get("rl", {}).get("sensor_overrides", {}))
    if sensor_overrides:
        resolved.setdefault("sensors", {}).update(sensor_overrides)
    resolved.setdefault("memory", {})["enable"] = False
    resolved["planner"]["prediction_mode"] = "nominal"
    gate = resolved.setdefault("rl", {}).setdefault("gate", {})
    if condition == "traditional_mppi":
        resolved["planner"]["sampling_prior"] = "goal_warm_start"
        resolved["rl"]["enabled"] = False
        resolved["rl"]["checkpoint"] = None
        return resolved

    resolved["planner"]["sampling_prior"] = "rl"
    resolved["rl"]["enabled"] = True
    resolved["rl"]["checkpoint"] = str(Path(checkpoint).resolve())
    if condition == "frozen_bc_prior":
        gate["mode"] = "none"
        gate["correction_advantage_gate_mode"] = "none"
        gate["correction_advantage_uncertainty_multiplier"] = 1.0
    elif condition == "lcb_always":
        gate["mode"] = "none"
        gate["correction_advantage_gate_mode"] = "lcb"
        gate["correction_advantage_critic_source"] = "target"
        gate["correction_advantage_threshold"] = 0.0
        gate["correction_advantage_uncertainty_multiplier"] = 2.0
    elif condition == "complexity_lcb":
        gate["mode"] = "complexity"
        gate["correction_advantage_gate_mode"] = "lcb"
        gate["correction_advantage_critic_source"] = "target"
        gate["correction_advantage_threshold"] = 0.0
        gate["correction_advantage_uncertainty_multiplier"] = 2.0
    else:
        raise ValueError("unknown L25 condition %s" % condition)
    return resolved


def _make_prior(config, checkpoint):
    action_spec = action_spec_from_config(config["action_space"])
    planner_config = dict(config["planner"])
    planner_config.setdefault("dt", config["experiment"]["control_dt"])
    planner_config.setdefault("seed", config["experiment"].get("seed", 0))
    mppi = MppiConfig.from_mapping(planner_config, action_spec.dimension)
    fallback = GoalWarmStartPrior(
        planner_config.get("prior_v_gain", 0.8),
        planner_config.get("prior_yaw_gain", 1.2),
        planner_config.get("prior_translation_heading_gate_rad"),
        planner_config.get(
            "prior_translation_heading_gate_terminal_only", False
        ),
    )
    return TorchSACPrior.from_checkpoint(
        checkpoint,
        action_spec,
        mppi.noise_sigma,
        device="cpu",
        gate_config=config["rl"].get("gate", {}),
        fallback_prior=fallback,
        policy_id=str(config["rl"].get("policy_id", "l25")),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--conditions")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    base = load_yaml(_resolved_path(args.config))
    design = base["rl"]["scene_complexity_ablation"]
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in design["development_episode_seeds"]]
    )
    conditions = (
        [value.strip() for value in args.conditions.split(",") if value.strip()]
        if args.conditions
        else [str(value) for value in design["conditions"]]
    )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("L25 seeds must be nonempty and unique")
    if tuple(conditions) != tuple(CONDITIONS):
        if set(conditions) != set(CONDITIONS):
            raise ValueError("L25 requires all four preregistered conditions")
    sealed = {int(value) for value in design["sealed_test_episode_seeds"]}
    if sealed.intersection(seeds) and not args.allow_sealed_test:
        raise ValueError("sealed L25 seeds require --allow-sealed-test")

    baseline = _resolved_path(args.baseline_checkpoint)
    candidate = _resolved_path(args.candidate_checkpoint)
    scenes = _scene_entries(base)
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    schedule = [
        {
            "scene": scene,
            "episode_seed": int(seed),
            "condition": condition,
        }
        for scene in scenes
        for seed in seeds
        for condition in conditions
    ]
    schedule_rng = random.Random(
        int(design["schedule_seed"]) + int(args.training_seed)
    )
    schedule_rng.shuffle(schedule)
    schedule_rows = []
    for index, item in enumerate(schedule):
        schedule_rows.append({
            "run_order": index,
            "training_seed": int(args.training_seed),
            "scene": item["scene"]["name"],
            "scene_role": item["scene"]["role"],
            "episode_seed": item["episode_seed"],
            "condition": item["condition"],
        })
    _write_csv(output / "condition_schedule.csv", schedule_rows)

    representative = _condition_config(
        base, scenes[0]["path"], "frozen_bc_prior", baseline, seeds[0]
    )
    priors = {
        "frozen_bc_prior": _make_prior(representative, baseline),
    }
    for condition in ("lcb_always", "complexity_lcb"):
        condition_config = _condition_config(
            base, scenes[0]["path"], condition, candidate, seeds[0]
        )
        priors[condition] = _make_prior(condition_config, candidate)

    episodes = []
    steps = []
    for index, item in enumerate(schedule):
        scene = item["scene"]
        condition = item["condition"]
        seed = int(item["episode_seed"])
        checkpoint = baseline if condition == "frozen_bc_prior" else candidate
        config = _condition_config(
            base, scene["path"], condition, checkpoint, seed
        )
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output
            / "runs"
            / scene["name"]
            / condition
            / ("seed_%d" % seed)
        )
        metrics_path = run_dir / "metrics.json"
        trajectory_path = run_dir / "trajectory.csv"
        if metrics_path.exists() and trajectory_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                episode = dict(json.load(handle))
            episode.pop("provenance", None)
            episode.pop("metadata", None)
        else:
            result = ExperimentRunner(
                config,
                ROOT,
                output_dir=run_dir,
                headless=True,
                rl_policy=priors.get(condition),
            ).run()
            episode = dict(result.summary)
        episode.update({
            "run_order": index,
            "training_seed": int(args.training_seed),
            "scene": scene["name"],
            "scene_role": scene["role"],
            "episode_seed": seed,
            "condition": condition,
            "checkpoint": (
                "none"
                if condition == "traditional_mppi"
                else str(checkpoint)
            ),
        })
        episodes.append(episode)
        with trajectory_path.open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            for step_index, raw in enumerate(csv.DictReader(handle)):
                row = {
                    "training_seed": int(args.training_seed),
                    "scene": scene["name"],
                    "scene_role": scene["role"],
                    "episode_seed": seed,
                    "condition": condition,
                    "step": step_index,
                }
                row.update({name: raw[name] for name in STEP_FIELDS})
                steps.append(row)

    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "gate_steps.csv", steps)
    metadata = {
        "training_seed": int(args.training_seed),
        "episode_seeds": seeds,
        "sealed_test_seeds_used": sorted(sealed.intersection(seeds)),
        "conditions": list(CONDITIONS),
        "scenes": [
            {"name": row["name"], "role": row["role"], "path": str(row["path"])}
            for row in scenes
        ],
        "schedule_seed": int(design["schedule_seed"]) + int(args.training_seed),
        "baseline_checkpoint": str(baseline),
        "baseline_checkpoint_sha256": _sha256(baseline),
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": _sha256(candidate),
        "run_git_sha": git_sha(ROOT),
        "episodes": len(episodes),
        "steps": len(steps),
        "interpretation_guard": (
            "episode rows are nested in independently trained checkpoints; "
            "development seeds cannot be reported as independent actor trainings"
        ),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
