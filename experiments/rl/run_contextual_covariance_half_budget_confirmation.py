#!/usr/bin/env python3
"""Single-process confirmation of learned-K50 versus fixed-K100."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from experiments.rl.run_contextual_covariance_sample_efficiency import (
    _add_gate,
    _paired_comparison,
)
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_schedule(spec):
    pairs = []
    for scene_path in spec["scenes"]:
        for domain in spec["physics_domains"]:
            for seed in spec["seeds"]:
                pairs.append((scene_path, domain, int(seed)))
    rng = np.random.RandomState(int(spec["schedule_seed"]))
    rng.shuffle(pairs)
    schedule = []
    arms = list(spec["arms"])
    for pair_index, (scene_path, domain, seed) in enumerate(pairs):
        order = arms if pair_index % 2 == 0 else list(reversed(arms))
        for arm in order:
            schedule.append({
                "pair_index": pair_index,
                "scene_path": scene_path,
                "physics_domain": domain,
                "seed": seed,
                "arm": arm,
            })
    return schedule


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    base = load_yaml(ROOT / spec["base_config"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = _paired_schedule(spec)
    with (output / "schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2, sort_keys=True)
    rows = []
    for schedule_index, item in enumerate(schedule):
        if args.max_runs is not None and schedule_index >= int(args.max_runs):
            break
        arm = item["arm"]
        domain = item["physics_domain"]
        samples = int(arm["num_samples"])
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        config["experiment"]["seed"] = int(item["seed"])
        config["experiment"]["name"] = "l95_%s_%s_%s_%d" % (
            config["scene"]["name"], domain["name"], arm["name"], item["seed"]
        )
        config["planner"]["num_samples"] = samples
        if arm["kind"] == "contextual_bandit":
            config["planner"]["sampling_prior"] = "contextual_bandit_covariance"
            config.setdefault("rl", {})["enabled"] = True
            config["rl"]["policy_id"] = "linucb_contextual_covariance_l89"
            config["rl"]["checkpoint"] = spec["checkpoint"]
        elif arm["kind"] == "fixed":
            config["planner"]["sampling_prior"] = "fixed_covariance"
            config["planner"]["fixed_covariance_scale"] = list(arm["scale"])
            config.setdefault("rl", {})["enabled"] = False
            config["rl"].pop("checkpoint", None)
        else:
            raise ValueError("unknown confirmation arm kind")
        run_dir = (
            output / "runs" / domain["name"] / config["scene"]["name"]
            / arm["name"] / ("seed_%d" % item["seed"])
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        else:
            summary = ExperimentRunner(
                copy.deepcopy(config), ROOT, run_dir, headless=True
            ).run().summary
        row = dict(summary)
        row.update({
            "pair_index": int(item["pair_index"]),
            "scene": str(config["scene"]["name"]),
            "physics_domain": str(domain["name"]),
            "num_samples": samples,
            "condition": str(arm["name"]),
            "seed": int(item["seed"]),
            "elapsed_s": float(summary["steps"])
            * float(config["experiment"]["control_dt"]),
        })
        rows.append(row)
        _write_csv(output / "episodes.csv", rows)
    if len(rows) != len(schedule):
        print(json.dumps({"complete": False, "completed_runs": len(rows)}, indent=2))
        return 0
    comparison = _paired_comparison(
        rows, 50, 100, int(spec["bootstrap_seed"])
    )
    comparison = _add_gate(
        comparison, float(spec["cross_track_noninferiority_margin_m"])
    )
    summary = {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "completed_runs": len(rows),
        "checkpoint": spec["checkpoint"],
        "evaluation": comparison,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
