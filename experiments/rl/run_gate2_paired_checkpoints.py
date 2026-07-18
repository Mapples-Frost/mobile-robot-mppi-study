#!/usr/bin/env python3
"""Paired MuJoCo comparison of ordinary and value-aligned ICODE."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.run_gate1_simple_combination import (
    load_physics_domains,
    load_scenes,
    method_config,
    parse_ints,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.paired_checkpoint import (
    gate2_closed_loop_decision,
    paired_checkpoint_schedule,
    paired_checkpoint_effects,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs/research/mujoco_strong_mppi_baseline.yaml"
        ),
    )
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--control-icode-checkpoint", required=True)
    parser.add_argument("--aligned-icode-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="21,22,23")
    parser.add_argument("--total-rollouts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--schedule-seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--physics-domain-config", required=True)
    parser.add_argument("--scene-configs", required=True)
    parser.add_argument("--physics-domains", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    base = load_yaml(args.config)
    seeds = parse_ints(args.seeds)
    domain_names = tuple(
        item.strip()
        for item in args.physics_domains.split(",")
        if item.strip()
    )
    domains = load_physics_domains(
        args.physics_domain_config, domain_names, ()
    )
    scene_paths = tuple(
        item.strip()
        for item in args.scene_configs.split(",")
        if item.strip()
    )
    scenes = load_scenes(base, scene_paths)
    schedule = paired_checkpoint_schedule(
        seeds, domains, scenes, args.schedule_seed
    )
    checkpoints = {
        "control": str(Path(args.control_icode_checkpoint).resolve()),
        "value_aligned": str(
            Path(args.aligned_icode_checkpoint).resolve()
        ),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    scene_by_name = {item["name"]: item for item in scenes}
    domain_by_name = {item["name"]: item for item in domains}
    for job_index, job in enumerate(schedule):
        variant = job["variant"]
        config = method_config(
            scene_by_name[job["scene"]]["config"],
            "simple_combination",
            True,
            True,
            args.actor_checkpoint,
            checkpoints[variant],
            args.total_rollouts,
            args.iterations,
            job["seed"],
            domain_by_name[job["physics_domain"]],
        )
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output / "runs" / variant / config["experiment"]["name"]
        )
        experiment = ExperimentRunner(
            config, ROOT, run_dir, headless=True
        ).run()
        row = dict(experiment.summary)
        row.update({
            "method": "simple_combination",
            "checkpoint_variant": variant,
            "scene": job["scene"],
            "scene_source": scene_by_name[job["scene"]]["source"],
            "physics_domain": job["physics_domain"],
            "physics_domain_role": str(
                domain_by_name[job["physics_domain"]].get(
                    "role", "unknown"
                )
            ),
            "seed": int(job["seed"]),
            "block": job["block"],
            "run_order_within_block": int(
                job["run_order_within_block"]
            ),
            "global_run_order": int(job_index),
            "actor_checkpoint": str(
                Path(args.actor_checkpoint).resolve()
            ),
            "icode_checkpoint": checkpoints[variant],
            "rollout_budget_per_decision": int(args.total_rollouts),
            "paper_iterations": int(args.iterations),
        })
        rows.append(row)
        write_csv(output / "progress.csv", rows)

    control_rows = [
        row for row in rows if row["checkpoint_variant"] == "control"
    ]
    aligned_rows = [
        row
        for row in rows
        if row["checkpoint_variant"] == "value_aligned"
    ]
    write_csv(output / "control_episodes.csv", control_rows)
    write_csv(output / "aligned_episodes.csv", aligned_rows)
    comparison = paired_checkpoint_effects(
        control_rows,
        aligned_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.schedule_seed,
    )
    comparison["gate"] = gate2_closed_loop_decision(comparison)
    provenance = {
        "git_sha": git_sha(),
        "actor_checkpoint": str(Path(args.actor_checkpoint).resolve()),
        "actor_checkpoint_sha256": file_sha256(args.actor_checkpoint),
        "control_icode_checkpoint": checkpoints["control"],
        "control_icode_checkpoint_sha256": file_sha256(
            args.control_icode_checkpoint
        ),
        "aligned_icode_checkpoint": checkpoints["value_aligned"],
        "aligned_icode_checkpoint_sha256": file_sha256(
            args.aligned_icode_checkpoint
        ),
        "schedule_seed": int(args.schedule_seed),
        "seeds": list(seeds),
        "scenes": [
            {"name": item["name"], "source": item["source"]}
            for item in scenes
        ],
        "physics_domains": domains,
        "total_rollouts": int(args.total_rollouts),
        "iterations": int(args.iterations),
    }
    (output / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output),
        "paired_cells": comparison["paired_cells"],
        "gate": comparison["gate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
