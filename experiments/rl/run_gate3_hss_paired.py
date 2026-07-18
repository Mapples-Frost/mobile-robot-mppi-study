#!/usr/bin/env python3
"""Equal-budget fixed versus reliability-adaptive HSS MuJoCo experiment."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


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
from mobile_robot_mppi.evaluation.gate3_hss import (
    gate3_development_decision,
    paired_hss_schedule,
)
from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_effects,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


COMPARISON_METRICS = {
    "success": True,
    "collision": False,
    "final_goal_distance": False,
    "control_jerk": False,
    "stuck_steps": False,
    "planner_compute_ms_mean": False,
    "paper_total_rollouts_mean": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_paths(value):
    result = [
        str(Path(item.strip()).resolve())
        for item in str(value).split(",") if item.strip()
    ]
    if len(result) < 2:
        raise ValueError("Gate 3 requires at least two ICODE checkpoints")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs/research/mujoco_strong_mppi_baseline.yaml"
        ),
    )
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--icode-checkpoints", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="32,33,34")
    parser.add_argument("--total-rollouts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--schedule-seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--physics-domain-config", required=True)
    parser.add_argument("--scene-configs", required=True)
    parser.add_argument("--physics-domains", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    base = load_yaml(args.config)
    seeds = parse_ints(args.seeds)
    domains = load_physics_domains(
        args.physics_domain_config,
        tuple(
            item.strip()
            for item in args.physics_domains.split(",")
            if item.strip()
        ),
        (),
    )
    scenes = load_scenes(
        base,
        tuple(
            item.strip()
            for item in args.scene_configs.split(",")
            if item.strip()
        ),
    )
    schedule = paired_hss_schedule(
        seeds, domains, scenes, args.schedule_seed
    )
    checkpoints = _parse_paths(args.icode_checkpoints)
    calibration_summary_path = Path(
        args.calibration_summary
    ).resolve()
    calibration = json.loads(
        calibration_summary_path.read_text(encoding="utf-8")
    )
    if not calibration.get("selected_candidate"):
        raise ValueError("calibration summary lacks a selected candidate")
    reliability_config = dict(
        calibration["runtime_reliability_config"]
    )
    calibration_config_path = Path(args.calibration_config).resolve()
    with calibration_config_path.open("r", encoding="utf-8") as handle:
        calibration_config = yaml.safe_load(handle)
    ensemble_config = calibration_config["ensemble"]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_by_name = {item["name"]: item for item in scenes}
    domain_by_name = {item["name"]: item for item in domains}
    rows = []
    for job_index, job in enumerate(schedule):
        arm = job["arm"]
        config = method_config(
            scene_by_name[job["scene"]]["config"],
            "gate3_hss",
            True,
            True,
            args.actor_checkpoint,
            checkpoints[0],
            args.total_rollouts,
            args.iterations,
            job["seed"],
            domain_by_name[job["physics_domain"]],
        )
        planner = config["planner"]
        planner["checkpoints"] = checkpoints
        planner.pop("checkpoint", None)
        planner["residual_ensemble"] = {
            "disagreement_scales": list(
                ensemble_config["disagreement_scales"]
            ),
            "innovation_scales": list(
                ensemble_config["innovation_scales"]
            ),
            "innovation_decay": float(
                ensemble_config["innovation_decay"]
            ),
            "support_soft_z": float(
                ensemble_config["support_soft_z"]
            ),
            "support_hard_z": float(
                ensemble_config["support_hard_z"]
            ),
        }
        if arm == "adaptive":
            planner["paper_rl_driven"]["reliability"] = dict(
                reliability_config
            )
        else:
            fixed = dict(reliability_config)
            fixed["enabled"] = False
            planner["paper_rl_driven"]["reliability"] = fixed
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output / "runs" / arm / config["experiment"]["name"]
        )
        experiment = ExperimentRunner(
            config, ROOT, run_dir, headless=True
        ).run()
        row = dict(experiment.summary)
        row.update({
            "method": "gate3_hss",
            "hss_arm": arm,
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
            "icode_checkpoints": "|".join(checkpoints),
            "rollout_budget_per_decision": int(args.total_rollouts),
            "paper_iterations": int(args.iterations),
        })
        rows.append(row)
        _write_csv(output / "progress.csv", rows)

    fixed_rows = [row for row in rows if row["hss_arm"] == "fixed"]
    adaptive_rows = [
        row for row in rows if row["hss_arm"] == "adaptive"
    ]
    _write_csv(output / "fixed_episodes.csv", fixed_rows)
    _write_csv(output / "adaptive_episodes.csv", adaptive_rows)
    comparison = paired_checkpoint_effects(
        fixed_rows,
        adaptive_rows,
        method="gate3_hss",
        metrics=COMPARISON_METRICS,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.schedule_seed,
    )
    comparison["gate"] = gate3_development_decision(
        comparison, adaptive_rows
    )
    provenance = {
        "git_sha": _git_sha(),
        "actor_checkpoint": str(Path(args.actor_checkpoint).resolve()),
        "actor_checkpoint_sha256": _sha256(args.actor_checkpoint),
        "icode_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in checkpoints
        ],
        "calibration_summary": str(calibration_summary_path),
        "calibration_summary_sha256": _sha256(
            calibration_summary_path
        ),
        "calibration_config": str(calibration_config_path),
        "calibration_config_sha256": _sha256(
            calibration_config_path
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

