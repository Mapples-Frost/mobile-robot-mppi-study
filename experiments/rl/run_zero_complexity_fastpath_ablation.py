#!/usr/bin/env python3
"""Run one checkpoint block of the preregistered L27 fast-path ablation."""

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
from experiments.rl.run_scene_complexity_sample_efficiency import _scene_entries


CONDITIONS = ("standard_complexity", "zero_complexity_fastpath")


def _resolved_condition(base, scene_path, condition, checkpoint, seed, num_samples):
    config = _condition_config(
        base, scene_path, "complexity_lcb", checkpoint, seed
    )
    config["planner"]["num_samples"] = int(num_samples)
    config["rl"]["gate"]["skip_zero_complexity_inference"] = bool(
        condition == "zero_complexity_fastpath"
    )
    config["experiment"]["name"] = "l27_%s_seed_%d" % (
        condition, int(seed)
    )
    return config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-sealed-test", action="store_true")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(1)
    config_path = _resolved_path(args.config)
    base = load_yaml(config_path)
    design = base["rl"]["zero_complexity_fastpath_ablation"]
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in design["development_episode_seeds"]]
    )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("L27 seeds must be nonempty and unique")
    protected = {
        int(value)
        for key in (
            "forbidden_l25_seeds",
            "sealed_test_episode_seeds",
        )
        for value in (
            base["rl"]["sample_efficiency_ablation"].get(key, [])
            if key == "forbidden_l25_seeds"
            else design.get(key, [])
        )
    }
    protected.update(int(value) for value in base["rl"]["sample_efficiency_ablation"]["sealed_test_episode_seeds"])
    if protected.intersection(seeds) and not args.allow_sealed_test:
        raise ValueError("sealed L25/L26/L27 seeds require --allow-sealed-test")

    checkpoint = _resolved_path(args.candidate_checkpoint)
    scenes = _scene_entries(base)
    num_samples = int(design["num_samples"])
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    schedule = [
        {"scene": scene, "episode_seed": seed, "condition": condition}
        for scene in scenes
        for seed in seeds
        for condition in CONDITIONS
    ]
    schedule_seed = int(design["schedule_seed"]) + int(args.training_seed)
    random.Random(schedule_seed).shuffle(schedule)
    _write_csv(output / "condition_schedule.csv", [
        {
            "run_order": index,
            "training_seed": int(args.training_seed),
            "scene": item["scene"]["name"],
            "scene_role": item["scene"]["role"],
            "episode_seed": int(item["episode_seed"]),
            "num_samples": num_samples,
            "condition": item["condition"],
        }
        for index, item in enumerate(schedule)
    ])

    priors = {}
    for condition in CONDITIONS:
        representative = _resolved_condition(
            base, scenes[0]["path"], condition, checkpoint, seeds[0], num_samples
        )
        priors[condition] = _make_prior(representative, checkpoint)

    episodes = []
    steps = []
    for index, item in enumerate(schedule):
        scene = item["scene"]
        seed = int(item["episode_seed"])
        condition = item["condition"]
        config = _resolved_condition(
            base, scene["path"], condition, checkpoint, seed, num_samples
        )
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = output / "runs" / scene["name"] / condition / ("seed_%d" % seed)
        metrics_path = run_dir / "metrics.json"
        trajectory_path = run_dir / "trajectory.csv"
        if metrics_path.exists() and trajectory_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                episode = dict(json.load(handle))
            episode.pop("provenance", None)
            episode.pop("metadata", None)
        else:
            result = ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True,
                rl_policy=priors[condition],
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
            "checkpoint": str(checkpoint),
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
        "study": "L27_zero_complexity_fastpath",
        "training_seed": int(args.training_seed),
        "episode_seeds": seeds,
        "num_samples": num_samples,
        "conditions": list(CONDITIONS),
        "schedule_seed": schedule_seed,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": _sha256(checkpoint),
        "protected_seeds_used": sorted(protected.intersection(seeds)),
        "run_git_sha": git_sha(ROOT),
        "episodes": len(episodes),
        "steps": len(steps),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
