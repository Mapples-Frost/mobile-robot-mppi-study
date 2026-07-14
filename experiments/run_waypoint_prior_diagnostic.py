#!/usr/bin/env python3
"""Causal diagnostic: direct goal versus offline waypoint reference.

Offline geometry is used only to create a task reference. Runtime obstacle input
still comes exclusively from LaserScan/local_obstacle_layer, and scan_guard
retains final control authority. This is not a deployable global planner.
"""

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import audit_static_scene
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_CONFIGS = (
    "configs/research/mujoco_lab_complex.yaml",
    "configs/research/mujoco_narrow_corridor.yaml",
    "configs/research/mujoco_u_trap_long_board.yaml",
)


def sample_waypoints(path, goal, spacing=0.45):
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("offline path must contain at least two points")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    distances = np.arange(float(spacing), cumulative[-1], float(spacing))
    waypoints = []
    for distance in distances:
        index = min(int(np.searchsorted(cumulative, distance)), len(points) - 1)
        before = max(0, index - 1)
        length = max(cumulative[index] - cumulative[before], 1e-12)
        ratio = (distance - cumulative[before]) / length
        point = points[before] + ratio * (points[index] - points[before])
        waypoints.append([float(point[0]), float(point[1])])
    final = [float(goal[0]), float(goal[1])]
    if not waypoints or np.linalg.norm(np.asarray(waypoints[-1]) - final) > 1e-9:
        waypoints.append(final)
    return waypoints


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["method"], row["reference_mode"]), []).append(row)
    summary = []
    for (scene, method, mode), values in sorted(groups.items()):
        summary.append({
            "scene": scene,
            "method": method,
            "reference_mode": mode,
            "runs": len(values),
            "success_rate": float(np.mean([row["success"] for row in values])),
            "collision_rate": float(np.mean([row["collision"] for row in values])),
            "final_goal_distance_mean": float(np.mean([row["final_goal_distance"] for row in values])),
            "safety_interventions_mean": float(np.mean([row["safety_interventions"] for row in values])),
            "stuck_steps_mean": float(np.mean([row["stuck_steps"] for row in values])),
            "planner_compute_ms_mean": float(np.mean([row["planner_compute_ms_mean"] for row in values])),
        })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--spacing", type=float, default=0.45)
    parser.add_argument("--route-margin", type=float, default=0.15)
    parser.add_argument(
        "--goal", nargs=2, type=float, metavar=("X", "Y"),
        help=(
            "diagnostic terminal goal; when omitted, retain each scene config goal. "
            "This override never changes the stored scene or planner obstacle input"
        ),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("direct_goal", "offline_waypoints", "offline_polyline"),
        default=("direct_goal", "offline_waypoints"),
    )
    parser.add_argument("--icode-checkpoint")
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    methods = [("nominal", "nominal", None)]
    if args.icode_checkpoint:
        methods.append(("icode", "icode_residual", args.icode_checkpoint))
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is not empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    routes = {}
    rows = []
    for config_path in args.configs:
        base = load_yaml(config_path)
        scene = str(base["scene"]["name"])
        goal = list(args.goal) if args.goal is not None else base["task"]["position"]
        if args.goal is not None:
            base["task"] = dict(base["task"])
            base["task"]["type"] = "point_goal"
            base["task"]["position"] = [float(goal[0]), float(goal[1])]
        audit = audit_static_scene(
            base["scene"], base["experiment"]["initial_state"][:2], goal,
            base["plant"]["robot"]["collision_radius"], margin=args.route_margin,
            resolution=0.04, include_path=True,
        )
        if not audit["path_exists"]:
            raise ValueError("no offline diagnostic route for scene %s" % scene)
        waypoints = sample_waypoints(audit["path"], goal, args.spacing)
        routes[scene] = {"waypoints": waypoints, "geometry_audit": audit}
        for mode in args.modes:
            for method, prediction_mode, checkpoint in methods:
                for seed in seeds:
                    config = copy.deepcopy(base)
                    config["experiment"]["seed"] = seed
                    config["experiment"]["max_steps"] = int(args.max_steps)
                    config["planner"]["seed"] = seed
                    config["planner"]["num_samples"] = int(args.num_samples)
                    config["planner"]["prediction_mode"] = prediction_mode
                    if checkpoint:
                        config["planner"]["checkpoint"] = checkpoint
                    if mode == "offline_waypoints":
                        config["task"] = {
                            "type": "waypoints",
                            "waypoints": waypoints,
                            "position_tolerance": float(base["task"].get("position_tolerance", 0.3)),
                            "waypoint_tolerance": min(float(args.spacing), 0.40),
                            "terminal_approach_count": min(3, len(waypoints)),
                        }
                    elif mode == "offline_polyline":
                        config["task"] = {
                            "type": "polyline",
                            "points": waypoints,
                            "position_tolerance": float(base["task"].get("position_tolerance", 0.3)),
                            "lookahead_distance": float(args.spacing),
                            "terminal_approach_distance": 2.0 * float(args.spacing),
                        }
                    run_dir = output / "runs" / scene / mode / method / ("seed_%04d" % seed)
                    result = ExperimentRunner(config, ROOT, run_dir, headless=True).run()
                    row = dict(result.summary)
                    row.update({"scene": scene, "reference_mode": mode, "method": method, "seed": seed})
                    rows.append(row)
                    print("scene=%s mode=%s method=%s seed=%d success=%s distance=%.3f" % (
                        scene, mode, method, seed, row["success"], row["final_goal_distance"]
                    ))
    summary = aggregate(rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_sha": git_sha(ROOT),
                "diagnostic_only": True,
                "seeds": seeds,
                "num_samples": int(args.num_samples),
                "goal_override": None if args.goal is None else list(args.goal),
            },
            "routes": routes, "summary": summary, "episodes": rows,
        }, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
