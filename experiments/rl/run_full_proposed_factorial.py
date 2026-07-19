#!/usr/bin/env python3
"""Randomized equal-budget Full Proposed 2x2 MuJoCo factorial."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.run_gate1_simple_combination import (
    load_physics_domains,
    load_scenes,
    method_config,
    parse_ints,
)
from experiments.rl.summarize_icode_path_tracking import project_polyline
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.factorial import (
    blocked_factorial_contrasts,
)
from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_effects,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


ARMS = (
    "ordinary_fixed",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)
FACTORIAL_METHOD = {
    "ordinary_fixed": "traditional_mppi",
    "value_fixed": "icode_mppi",
    "ordinary_adaptive": "rl_driven_mppi",
    "full_proposed": "simple_combination",
}
POINT_GOAL_METRICS = {
    "success": True,
    "collision": False,
    "final_goal_distance": False,
    "control_jerk": False,
    "stuck_steps": False,
    "planner_compute_ms_mean": False,
    "paper_total_rollouts_mean": False,
}
PATH_TRACKING_METRICS = {
    "success": True,
    "collision": False,
    "cross_track_rmse": False,
    "path_completion_ratio": True,
    "tangent_heading_rmse": False,
    "cross_track_max": False,
    "control_jerk": False,
    "planner_compute_ms_mean": False,
    "paper_total_rollouts_mean": False,
}
# Backward-compatible public name used by earlier point-goal tooling.
COMPARISON_METRICS = POINT_GOAL_METRICS


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
        for item in str(value).split(",")
        if item.strip()
    ]
    if len(result) < 2:
        raise ValueError("factorial requires at least two checkpoints")
    return result


def metrics_for_profile(profile):
    if profile == "point_goal":
        return POINT_GOAL_METRICS
    if profile == "path_tracking":
        return PATH_TRACKING_METRICS
    raise ValueError("unknown metric profile: %s" % profile)


def path_tracking_metrics(trajectory_path, points):
    """Compute auditable path metrics from one saved trajectory."""

    with Path(trajectory_path).open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("path trajectory is empty")
    xy = np.asarray(
        [[float(row["x"]), float(row["y"])] for row in rows],
        dtype=np.float64,
    )
    theta = np.asarray(
        [float(row["theta"]) for row in rows], dtype=np.float64
    )
    cross_track, heading_error, completion = project_polyline(
        points, xy, theta
    )
    result = {
        "path_completion_ratio": float(completion[-1]),
        "tangent_heading_rmse": float(
            np.sqrt(np.mean(np.square(heading_error)))
        ),
        "path_cross_track_rmse_recomputed": float(
            np.sqrt(np.mean(np.square(cross_track)))
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite path-tracking metric")
    return result


def _load_reliability(summary_path, config_path):
    summary_path = Path(summary_path).resolve()
    config_path = Path(config_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("selected_candidate"):
        raise ValueError(
            "reliability calibration lacks a selected candidate"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {
        "runtime": dict(summary["runtime_reliability_config"]),
        "ensemble": dict(config["ensemble"]),
        "summary_path": str(summary_path),
        "config_path": str(config_path),
    }


def factorial_schedule(seeds, domains, scenes, schedule_seed):
    """Return four randomized arms in every seed-scene-domain block."""

    rng = np.random.RandomState(int(schedule_seed))
    jobs = []
    for seed in seeds:
        for scene in scenes:
            for domain in domains:
                block = "%s::%s::seed%d" % (
                    scene["name"], domain["name"], int(seed)
                )
                order = list(ARMS)
                rng.shuffle(order)
                for within, arm in enumerate(order):
                    jobs.append({
                        "seed": int(seed),
                        "scene": scene["name"],
                        "physics_domain": domain["name"],
                        "block": block,
                        "arm": arm,
                        "run_order_within_block": int(within),
                    })
    for index, job in enumerate(jobs):
        job["global_run_order"] = int(index)
    return jobs


def _paired(
    rows,
    control_arm,
    aligned_arm,
    label,
    samples,
    seed,
    metrics=None,
):
    control = [
        dict(row, method=label)
        for row in rows
        if row["factorial_arm"] == control_arm
    ]
    aligned = [
        dict(row, method=label)
        for row in rows
        if row["factorial_arm"] == aligned_arm
    ]
    return paired_checkpoint_effects(
        control,
        aligned,
        method=label,
        metrics=metrics or COMPARISON_METRICS,
        bootstrap_samples=int(samples),
        seed=int(seed),
    )


def _factorial(rows, samples, seed, metrics=None):
    result = {}
    for metric, higher_is_better in (
        metrics or COMPARISON_METRICS
    ).items():
        result[metric] = blocked_factorial_contrasts(
            rows,
            metric,
            higher_is_better=higher_is_better,
            bootstrap_samples=int(samples),
            seed=int(seed),
            cluster_key="seed",
        )
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
    parser.add_argument("--ordinary-checkpoints", required=True)
    parser.add_argument("--value-checkpoints", required=True)
    parser.add_argument("--ordinary-calibration-summary", required=True)
    parser.add_argument("--ordinary-calibration-config", required=True)
    parser.add_argument("--value-calibration-summary", required=True)
    parser.add_argument("--value-calibration-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--total-rollouts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--schedule-seed", type=int, default=20260736)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--physics-domain-config", required=True)
    parser.add_argument("--scene-configs", required=True)
    parser.add_argument("--physics-domains", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--metric-profile",
        choices=("point_goal", "path_tracking"),
        default="point_goal",
    )
    parser.add_argument(
        "--terminal-guidance-radius", type=float, default=0.0
    )
    parser.add_argument(
        "--terminal-guided-fraction-floor", type=float, default=0.0
    )
    args = parser.parse_args(argv)
    comparison_metrics = metrics_for_profile(args.metric_profile)

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
    if args.metric_profile == "path_tracking":
        invalid = [
            item["name"]
            for item in scenes
            if item["config"].get("task", {}).get("type") != "polyline"
        ]
        if invalid:
            raise ValueError(
                "path_tracking profile requires polyline scenes: %s"
                % ", ".join(invalid)
            )
    schedule = factorial_schedule(
        seeds, domains, scenes, args.schedule_seed
    )
    ordinary_checkpoints = _parse_paths(args.ordinary_checkpoints)
    value_checkpoints = _parse_paths(args.value_checkpoints)
    ordinary_reliability = _load_reliability(
        args.ordinary_calibration_summary,
        args.ordinary_calibration_config,
    )
    value_reliability = _load_reliability(
        args.value_calibration_summary,
        args.value_calibration_config,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_by_name = {item["name"]: item for item in scenes}
    domain_by_name = {item["name"]: item for item in domains}
    rows = []
    for job in schedule:
        arm = job["arm"]
        value_aligned = arm in ("value_fixed", "full_proposed")
        adaptive = arm in ("ordinary_adaptive", "full_proposed")
        checkpoints = (
            value_checkpoints if value_aligned else ordinary_checkpoints
        )
        calibration = (
            value_reliability if value_aligned else ordinary_reliability
        )
        config = method_config(
            scene_by_name[job["scene"]]["config"],
            arm,
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
        ensemble = calibration["ensemble"]
        planner["residual_ensemble"] = {
            "disagreement_scales": list(
                ensemble["disagreement_scales"]
            ),
            "innovation_scales": list(ensemble["innovation_scales"]),
            "innovation_decay": float(ensemble["innovation_decay"]),
            "support_soft_z": float(ensemble["support_soft_z"]),
            "support_hard_z": float(ensemble["support_hard_z"]),
        }
        reliability = dict(calibration["runtime"])
        reliability["enabled"] = bool(adaptive)
        planner["paper_rl_driven"]["reliability"] = reliability
        planner["paper_rl_driven"]["terminal_guidance_radius"] = (
            float(args.terminal_guidance_radius) if adaptive else 0.0
        )
        planner["paper_rl_driven"][
            "terminal_guided_fraction_floor"
        ] = (
            float(args.terminal_guided_fraction_floor)
            if adaptive
            else 0.0
        )
        planner["paper_rl_driven"].pop(
            "conservative_terminal", None
        )
        if args.max_steps > 0:
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output / "runs" / arm / config["experiment"]["name"]
        )
        experiment = ExperimentRunner(
            config, ROOT, run_dir, headless=True
        ).run()
        row = dict(experiment.summary)
        if args.metric_profile == "path_tracking":
            tracking = path_tracking_metrics(
                run_dir / "trajectory.csv",
                config["task"]["points"],
            )
            if not math.isclose(
                float(row["cross_track_rmse"]),
                tracking["path_cross_track_rmse_recomputed"],
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "saved trajectory and episode cross-track metrics disagree"
                )
            row.update(tracking)
        row.update({
            "method": FACTORIAL_METHOD[arm],
            "factorial_arm": arm,
            "value_alignment": int(value_aligned),
            "adaptive_hss": int(adaptive),
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
            "global_run_order": int(job["global_run_order"]),
            "actor_checkpoint": str(
                Path(args.actor_checkpoint).resolve()
            ),
            "icode_checkpoints": "|".join(checkpoints),
            "reliability_calibration_summary": calibration[
                "summary_path"
            ],
            "rollout_budget_per_decision": int(args.total_rollouts),
            "paper_iterations": int(args.iterations),
            "metric_profile": str(args.metric_profile),
            "terminal_guidance_radius": (
                float(args.terminal_guidance_radius) if adaptive else 0.0
            ),
            "terminal_guided_fraction_floor": (
                float(args.terminal_guided_fraction_floor)
                if adaptive
                else 0.0
            ),
        })
        rows.append(row)
        _write_csv(output / "progress.csv", rows)

    for arm in ARMS:
        _write_csv(
            output / ("%s_episodes.csv" % arm),
            [row for row in rows if row["factorial_arm"] == arm],
        )
    unique_seeds = sorted({int(row["seed"]) for row in rows})
    if len(unique_seeds) < 2:
        (output / "shard_complete.json").write_text(
            json.dumps({
                "status": "complete_factorial_shard",
                "seeds": unique_seeds,
                "episodes": len(rows),
                "analysis_deferred": (
                    "seed-cluster inference requires at least two seeds"
                ),
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0
    paired = {
        "full_vs_simple": _paired(
            rows,
            "ordinary_fixed",
            "full_proposed",
            "full_vs_simple",
            args.bootstrap_samples,
            args.schedule_seed,
            comparison_metrics,
        ),
        "value_at_fixed": _paired(
            rows,
            "ordinary_fixed",
            "value_fixed",
            "value_at_fixed",
            args.bootstrap_samples,
            args.schedule_seed + 1,
            comparison_metrics,
        ),
        "hss_at_ordinary": _paired(
            rows,
            "ordinary_fixed",
            "ordinary_adaptive",
            "hss_at_ordinary",
            args.bootstrap_samples,
            args.schedule_seed + 2,
            comparison_metrics,
        ),
        "hss_at_value": _paired(
            rows,
            "value_fixed",
            "full_proposed",
            "hss_at_value",
            args.bootstrap_samples,
            args.schedule_seed + 3,
            comparison_metrics,
        ),
        "value_at_adaptive": _paired(
            rows,
            "ordinary_adaptive",
            "full_proposed",
            "value_at_adaptive",
            args.bootstrap_samples,
            args.schedule_seed + 4,
            comparison_metrics,
        ),
    }
    factorial = _factorial(
        rows,
        args.bootstrap_samples,
        args.schedule_seed,
        comparison_metrics,
    )
    provenance = {
        "git_sha": _git_sha(),
        "actor_checkpoint": {
            "path": str(Path(args.actor_checkpoint).resolve()),
            "sha256": _sha256(args.actor_checkpoint),
        },
        "ordinary_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in ordinary_checkpoints
        ],
        "value_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in value_checkpoints
        ],
        "ordinary_calibration": ordinary_reliability,
        "value_calibration": value_reliability,
        "schedule_seed": int(args.schedule_seed),
        "seeds": list(seeds),
        "scenes": [
            {"name": item["name"], "source": item["source"]}
            for item in scenes
        ],
        "physics_domains": domains,
        "total_rollouts": int(args.total_rollouts),
        "iterations": int(args.iterations),
        "max_steps": int(args.max_steps),
        "metric_profile": str(args.metric_profile),
    }
    for name, value in (
        ("paired_comparisons.json", paired),
        ("factorial_contrasts.json", factorial),
        ("provenance.json", provenance),
        ("schedule.json", schedule),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "output_dir": str(output),
        "episodes": len(rows),
        "blocks": len(rows) // len(ARMS),
        "full_vs_simple": paired["full_vs_simple"]["metrics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
