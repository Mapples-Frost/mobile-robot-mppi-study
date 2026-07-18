#!/usr/bin/env python3
"""Calibrate dynamic-scene identifiability using only traditional MPPI."""

import argparse
import copy
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

from experiments.rl.evaluate_rl_sampling_prior import (
    _apply_physics_domain,
    _apply_scene_config,
)
from mobile_robot_mppi.core.config import deep_merge, git_sha, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty L36 table")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _calibration_design(config_path):
    config = load_yaml(_resolved_path(config_path))
    design = dict(config["rl"]["benchmark_calibration"])
    seeds = [int(value) for value in design["episode_seeds"]]
    forbidden = {int(value) for value in design.get("forbidden_episode_seeds", ())}
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("L36 episode seeds must be nonempty and unique")
    if forbidden.intersection(seeds):
        raise ValueError("L36 seeds overlap protected development/confirmation seeds")
    candidates = list(design["candidates"])
    names = [str(item["name"]) for item in candidates]
    if len(candidates) < 3 or len(names) != len(set(names)):
        raise ValueError("L36 candidates must contain at least three unique names")
    design["episode_seeds"] = seeds
    design["candidates"] = candidates
    return config, design


def _condition_config(base, design, candidate, episode_seed):
    config = copy.deepcopy(base)
    config["scene"]["name"] = "l36_%s" % candidate["name"]
    config["scene"]["obstacles"] = [copy.deepcopy(candidate["obstacle"])]
    config["experiment"]["name"] = "l36_%s_seed_%d" % (
        candidate["name"], int(episode_seed)
    )
    config["experiment"]["seed"] = int(episode_seed)
    config.setdefault("memory", {})["enable"] = False
    config["planner"]["sampling_prior"] = "goal_warm_start"
    config["planner"]["prediction_mode"] = "nominal"
    config["planner"].pop("checkpoint", None)
    config.setdefault("rl", {})["enabled"] = False
    config["rl"]["checkpoint"] = None
    sensor_overrides = dict(config["rl"].get("sensor_overrides", {}))
    if sensor_overrides:
        config.setdefault("sensors", {}).update(sensor_overrides)
    perception_overrides = dict(
        config["rl"].get("training", {}).get("perception_overrides", {})
    )
    if perception_overrides:
        config["perception"] = deep_merge(
            config.get("perception", {}), perception_overrides
        )
    config["benchmark_calibration"] = {
        "design_id": str(design["design_id"]),
        "candidate": str(candidate["name"]),
        "policy": "traditional_goal_warm_start",
        "prediction_mode": "nominal",
        "physics_domain": str(design["physics_domain"]),
    }
    return config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = _resolved_path(args.config)
    config, design = _calibration_design(config_path)
    base = _apply_scene_config(config, _resolved_path(design["base_scene"]))
    base = _apply_physics_domain(
        base,
        _resolved_path(design["physics_domain_config"]),
        design["physics_domain"],
    )
    output = _resolved_path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("L36 output exists; pass --resume to audit and continue")
    output.mkdir(parents=True, exist_ok=True)

    schedule = [
        {"candidate": candidate, "episode_seed": seed}
        for candidate in design["candidates"]
        for seed in design["episode_seeds"]
    ]
    random.Random(int(design["schedule_seed"])).shuffle(schedule)
    schedule_rows = [{
        "run_order": index,
        "candidate": str(item["candidate"]["name"]),
        "episode_seed": int(item["episode_seed"]),
    } for index, item in enumerate(schedule)]
    _write_csv(output / "condition_schedule.csv", schedule_rows)
    metadata = {
        "schema_version": 1,
        "design_id": str(design["design_id"]),
        "source_config": str(config_path),
        "run_git_sha": git_sha(ROOT),
        "schedule_seed": int(design["schedule_seed"]),
        "episode_seeds": list(design["episode_seeds"]),
        "candidates": [copy.deepcopy(item) for item in design["candidates"]],
        "policy": "traditional_goal_warm_start",
        "prediction_mode": "nominal",
        "rl_enabled": False,
        "memory_enabled": False,
        "physics_domain": str(design["physics_domain"]),
        "expected_episodes": len(schedule),
        "interpretation_guard": (
            "L36 calibrates benchmark identifiability only; it contains no RL or ICODE comparison"
        ),
    }
    (output / "frozen_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episodes = []
    for index, item in enumerate(schedule):
        candidate = item["candidate"]
        seed = int(item["episode_seed"])
        run_dir = output / "runs" / candidate["name"] / ("seed_%d" % seed)
        metrics_path = run_dir / "metrics.json"
        trajectory_path = run_dir / "trajectory.csv"
        if metrics_path.exists() and trajectory_path.exists():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            result.pop("metadata", None)
            result.pop("provenance", None)
        else:
            condition = _condition_config(base, design, candidate, seed)
            run = ExperimentRunner(
                condition, ROOT, output_dir=run_dir, headless=True
            ).run()
            result = dict(run.summary)
        result.update(schedule_rows[index])
        result["policy"] = "traditional_goal_warm_start"
        result["prediction_mode"] = "nominal"
        result["rl_enabled"] = False
        result["memory_enabled"] = False
        episodes.append(result)
    _write_csv(output / "calibration_episodes.csv", episodes)
    metadata["observed_episodes"] = len(episodes)
    metadata["completed"] = bool(len(episodes) == len(schedule))
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

