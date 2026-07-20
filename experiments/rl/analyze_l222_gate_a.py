#!/usr/bin/env python3
"""Audit the preregistered L222 hard-map Gate A MuJoCo episodes.

This development-only diagnostic validates the frozen execution contract and
then reconstructs monotonic route progress from the executed trajectories. It
does not expose simulator obstacle truth to any online controller.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.references import PolylineReference


SCENES = {
    "l222_serpentine_safe": "mujoco_l222_serpentine_safe_polyline.yaml",
    "l222_nested_u_safe": "mujoco_l222_nested_u_safe_polyline.yaml",
    "l222_cylinder_spiral_safe": "mujoco_l222_cylinder_spiral_safe_polyline.yaml",
}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s in %s" % (name, row))
    return value


def _episode_files(run_dir):
    metrics = list((run_dir / "runs").glob("*/*/metrics.json"))
    if len(metrics) != 1:
        raise ValueError("expected exactly one episode in %s" % run_dir)
    episode = metrics[0].parent
    required = ("metrics.json", "trajectory.csv", "config_resolved.yaml", "provenance.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise ValueError("missing episode artifacts %s in %s" % (missing, episode))
    return episode


def _audit_episode(run_dir, expected_git_sha):
    episode = _episode_files(run_dir)
    metrics = json.loads((episode / "metrics.json").read_text(encoding="utf-8"))
    metadata = metrics["metadata"]
    provenance = metrics["provenance"]
    config = load_yaml(episode / "config_resolved.yaml")
    scene = str(config["scene"]["name"])
    if scene not in SCENES:
        raise ValueError("unexpected Gate A scene: %s" % scene)
    expected_config = load_yaml(ROOT / "configs" / "research" / SCENES[scene])
    if provenance["git_sha"] != expected_git_sha:
        raise ValueError("unexpected Git SHA in %s" % episode)
    contract = {
        "plant_backend": metadata.get("plant_backend"),
        "mujoco_version": str(metadata.get("mujoco_version")),
        "prediction_mode": metadata.get("prediction_mode"),
        "seed": int(metadata.get("seed")),
        "rl_enabled": bool(metadata.get("rl_enabled")),
        "num_samples": int(config["planner"]["num_samples"]),
        "iterations": int(config["experiment"]["paper_iterations"]),
        "path_preview_enabled": bool(config["planner"]["path_preview_enabled"]),
        "path_preview_speed_mps": float(config["planner"]["path_preview_speed_mps"]),
        "path_preview_heading_weight": float(
            config["planner"]["path_preview_heading_weight"]
        ),
    }
    expected_contract = {
        "plant_backend": "mujoco_diff_drive",
        "mujoco_version": "3.2.3",
        "prediction_mode": "icode_residual",
        "seed": 91001,
        "rl_enabled": False,
        "num_samples": 30,
        "iterations": 1,
        "path_preview_enabled": True,
        "path_preview_speed_mps": 0.45,
        "path_preview_heading_weight": 0.25,
    }
    if contract != expected_contract:
        raise ValueError("Gate A contract mismatch: %s" % contract)
    if config["task"]["points"] != expected_config["task"]["points"]:
        raise ValueError("resolved reference differs from frozen scene: %s" % scene)

    trajectory = _read_csv(episode / "trajectory.csv")
    if not trajectory:
        raise ValueError("empty trajectory in %s" % episode)
    reference = PolylineReference(
        config["task"]["points"],
        tolerance=float(config["task"]["position_tolerance"]),
        lookahead_distance=float(config["task"]["lookahead_distance"]),
        terminal_approach_distance=float(config["task"]["terminal_approach_distance"]),
    )
    progress = 0.0
    progress_values = []
    cross_track_values = []
    for row in trajectory:
        projection = reference.project(
            np.asarray((_finite(row, "x"), _finite(row, "y"))),
            minimum_progress=progress,
        )
        progress = projection.progress
        progress_values.append(progress)
        cross_track_values.append(projection.cross_track_error)
    tail = trajectory[-min(200, len(trajectory)):]
    tail_progress = progress_values[-len(tail):]
    reasons = Counter(row["safety_reason"] for row in tail)
    last = trajectory[-1]
    output = {
        "scene": scene,
        "success": bool(metrics["success"]),
        "collision": bool(metrics["collision"]),
        "steps": int(metrics["steps"]),
        "termination_reason": str(metrics["termination_reason"]),
        "final_goal_distance_m": float(metrics["final_goal_distance"]),
        "minimum_clearance_m": float(metrics["minimum_clearance"]),
        "trajectory_length_m": float(metrics["trajectory_length"]),
        "reference_length_m": reference.total_length,
        "final_route_progress_m": progress_values[-1],
        "final_route_progress_ratio": progress_values[-1] / reference.total_length,
        "last_200_route_progress_m": tail_progress[-1] - tail_progress[0],
        "cross_track_rmse_recomputed_m": float(
            np.sqrt(np.mean(np.square(cross_track_values)))
        ),
        "final_x": _finite(last, "x"),
        "final_y": _finite(last, "y"),
        "final_theta": _finite(last, "theta"),
        "final_target_x": _finite(last, "target_x"),
        "final_target_y": _finite(last, "target_y"),
        "last_200_abs_proposed_v_mean": float(np.mean([
            abs(_finite(row, "proposed_v")) for row in tail
        ])),
        "last_200_abs_applied_v_mean": float(np.mean([
            abs(_finite(row, "applied_v")) for row in tail
        ])),
        "last_200_abs_applied_omega_mean": float(np.mean([
            abs(_finite(row, "applied_omega")) for row in tail
        ])),
        "last_200_front_slow_fraction": reasons["front_obstacle_slow"] / len(tail),
        "last_200_front_soft_block_fraction": reasons["front_soft_block"] / len(tail),
        "last_200_near_body_hard_stop_fraction": (
            reasons["near_body_hard_stop"] / len(tail)
        ),
        "metrics_cross_track_rmse_m": float(metrics["cross_track_rmse"]),
        "safety_interventions": int(metrics["safety_interventions"]),
        "episode_dir": str(episode),
        "git_sha": provenance["git_sha"],
        "config_hash": provenance["config_hash"],
    }
    windows = []
    for start in range(0, len(trajectory), 200):
        stop = min(start + 200, len(trajectory))
        block = trajectory[start:stop]
        block_reasons = Counter(row["safety_reason"] for row in block)
        progress_delta = progress_values[stop - 1] - progress_values[start]
        elapsed = max(1, stop - start) * float(config["experiment"]["control_dt"])
        windows.append({
            "scene": scene,
            "start_step": start,
            "stop_step": stop,
            "start_progress_m": progress_values[start],
            "stop_progress_m": progress_values[stop - 1],
            "progress_delta_m": progress_delta,
            "progress_rate_mps": progress_delta / elapsed,
            "proposed_v_mean": float(np.mean([
                _finite(row, "proposed_v") for row in block
            ])),
            "applied_v_mean": float(np.mean([
                _finite(row, "applied_v") for row in block
            ])),
            "abs_applied_omega_mean": float(np.mean([
                abs(_finite(row, "applied_omega")) for row in block
            ])),
            "front_slow_fraction": block_reasons["front_obstacle_slow"] / len(block),
            "front_soft_block_fraction": block_reasons["front_soft_block"] / len(block),
            "near_body_hard_stop_fraction": (
                block_reasons["near_body_hard_stop"] / len(block)
            ),
        })
    return output, contract, windows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_dirs = sorted(path for path in ROOT.glob(args.run_glob) if path.is_dir())
    if len(run_dirs) != 3:
        raise ValueError("expected three Gate A run directories, got %d" % len(run_dirs))
    rows = []
    contracts = []
    windows = []
    for run_dir in run_dirs:
        row, contract, episode_windows = _audit_episode(
            run_dir, args.expected_git_sha
        )
        rows.append(row)
        contracts.append(contract)
        windows.extend(episode_windows)
    if len({json.dumps(item, sort_keys=True) for item in contracts}) != 1:
        raise ValueError("Gate A execution contract differs across scenes")
    if {row["scene"] for row in rows} != set(SCENES):
        raise ValueError("Gate A scene matrix is incomplete")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "gate_a_episode_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "gate_a_progress_windows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    passed = (
        sum(bool(row["success"]) for row in rows) >= 2
        and not any(bool(row["collision"]) for row in rows)
    )
    report = {
        "status": "gate_a_passed" if passed else "gate_a_failed",
        "formal_claim_allowed": False,
        "development_seed": 91001,
        "successes": sum(bool(row["success"]) for row in rows),
        "collisions": sum(bool(row["collision"]) for row in rows),
        "required_successes": 2,
        "execution_contract": contracts[0],
        "episodes": rows,
    }
    (output / "gate_a_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
