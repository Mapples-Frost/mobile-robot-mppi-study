#!/usr/bin/env python3
"""Run one checkpoint block of the preregistered L26 K-factorial."""

import argparse
import csv
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
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from experiments.rl.run_scene_complexity_gate_ablation import (
    STEP_FIELDS,
    _condition_config,
    _make_prior,
    _resolved_path,
    _sha256,
    _write_csv,
)


CONDITIONS = ("traditional_mppi", "complexity_lcb")


def _scene_entries(base):
    design = base["rl"]["sample_efficiency_ablation"]
    result = []
    for role, key in (("control", "control_scenes"), ("blocking", "blocking_scenes")):
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
        raise ValueError("L26 scene names must be unique")
    return result


def _resolved_condition(base, scene_path, condition, checkpoint, seed, num_samples):
    config = _condition_config(base, scene_path, condition, checkpoint, seed)
    config["planner"]["num_samples"] = int(num_samples)
    config["experiment"]["name"] = "l26_%s_k%d_seed_%d" % (
        condition,
        int(num_samples),
        int(seed),
    )
    return config


def _parse_int_list(text, default):
    if text is None:
        return [int(value) for value in default]
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--sample-counts")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-sealed-test", action="store_true")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    config_path = _resolved_path(args.config)
    base = load_yaml(config_path)
    design = base["rl"]["sample_efficiency_ablation"]
    seeds = _parse_int_list(args.seeds, design["development_episode_seeds"])
    sample_counts = _parse_int_list(args.sample_counts, design["sample_counts"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("L26 seeds must be nonempty and unique")
    if not sample_counts or len(sample_counts) != len(set(sample_counts)):
        raise ValueError("L26 sample counts must be nonempty and unique")
    if any(value <= 0 for value in sample_counts):
        raise ValueError("L26 sample counts must be positive")

    l26_sealed = {int(value) for value in design["sealed_test_episode_seeds"]}
    l25_forbidden = {int(value) for value in design["forbidden_l25_seeds"]}
    protected = l26_sealed | l25_forbidden
    if protected.intersection(seeds) and not args.allow_sealed_test:
        raise ValueError("sealed L25/L26 seeds require --allow-sealed-test")

    checkpoint = _resolved_path(args.candidate_checkpoint)
    scenes = _scene_entries(base)
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    schedule = [
        {
            "scene": scene,
            "episode_seed": int(seed),
            "condition": condition,
            "num_samples": int(num_samples),
        }
        for scene in scenes
        for seed in seeds
        for num_samples in sample_counts
        for condition in CONDITIONS
    ]
    schedule_seed = int(design["schedule_seed"]) + int(args.training_seed)
    random.Random(schedule_seed).shuffle(schedule)
    schedule_rows = []
    for index, item in enumerate(schedule):
        schedule_rows.append({
            "run_order": index,
            "training_seed": int(args.training_seed),
            "scene": item["scene"]["name"],
            "scene_role": item["scene"]["role"],
            "episode_seed": item["episode_seed"],
            "num_samples": item["num_samples"],
            "condition": item["condition"],
        })
    _write_csv(output / "condition_schedule.csv", schedule_rows)

    representative = _resolved_condition(
        base,
        scenes[0]["path"],
        "complexity_lcb",
        checkpoint,
        seeds[0],
        sample_counts[0],
    )
    prior = _make_prior(representative, checkpoint)

    episodes = []
    steps = []
    for index, item in enumerate(schedule):
        scene = item["scene"]
        condition = item["condition"]
        seed = int(item["episode_seed"])
        num_samples = int(item["num_samples"])
        config = _resolved_condition(
            base, scene["path"], condition, checkpoint, seed, num_samples
        )
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output
            / "runs"
            / scene["name"]
            / ("k_%04d" % num_samples)
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
                rl_policy=prior if condition == "complexity_lcb" else None,
            ).run()
            episode = dict(result.summary)
        episode.update({
            "run_order": index,
            "training_seed": int(args.training_seed),
            "scene": scene["name"],
            "scene_role": scene["role"],
            "episode_seed": seed,
            "num_samples": num_samples,
            "condition": condition,
            "checkpoint": (
                "none" if condition == "traditional_mppi" else str(checkpoint)
            ),
        })
        episodes.append(episode)
        with trajectory_path.open("r", newline="", encoding="utf-8") as handle:
            for step_index, raw in enumerate(csv.DictReader(handle)):
                row = {
                    "training_seed": int(args.training_seed),
                    "scene": scene["name"],
                    "scene_role": scene["role"],
                    "episode_seed": seed,
                    "num_samples": num_samples,
                    "condition": condition,
                    "step": step_index,
                }
                row.update({name: raw[name] for name in STEP_FIELDS})
                steps.append(row)

    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "gate_steps.csv", steps)
    metadata = {
        "study": "L26_sample_efficiency",
        "training_seed": int(args.training_seed),
        "episode_seeds": seeds,
        "sample_counts": sample_counts,
        "conditions": list(CONDITIONS),
        "scenes": [
            {"name": row["name"], "role": row["role"], "path": str(row["path"])}
            for row in scenes
        ],
        "schedule_seed": schedule_seed,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": _sha256(checkpoint),
        "sealed_l25_seeds_used": sorted(l25_forbidden.intersection(seeds)),
        "sealed_l26_seeds_used": sorted(l26_sealed.intersection(seeds)),
        "run_git_sha": git_sha(ROOT),
        "episodes": len(episodes),
        "steps": len(steps),
        "interpretation_guard": (
            "checkpoint is the independent policy-training replicate; episode and K "
            "measurements are paired/nested within checkpoint"
        ),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
