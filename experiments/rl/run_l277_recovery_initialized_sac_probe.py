#!/usr/bin/env python3
"""Run and evaluate the frozen L277 paired small-budget SAC probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rl/l277_recovery_initialized_sac_gate.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l277_recovery_initialized_sac_probe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_checkpoint(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for value in payload["agent"].values():
        if isinstance(value, dict):
            for tensor in value.values():
                if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
                    return False
        elif torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            return False
    return True


def _step_metrics(rows, step):
    selected = [row for row in rows if int(row["global_step"]) == int(step)]
    if len(selected) != 3 or len({row["scene"] for row in selected}) != 3:
        raise ValueError("L277 validation step does not contain exactly three scenes")
    return {
        "successes": sum(str(row["success"]).lower() == "true" for row in selected),
        "collisions": sum(str(row["collision"]).lower() == "true" for row in selected),
        "mean_completion": float(np.mean([float(row["path_completion_ratio"]) for row in selected])),
        "mean_cte": float(np.mean([float(row["cross_track_rmse"]) for row in selected])),
        "mean_goal_distance": float(np.mean([float(row["goal_distance"]) for row in selected])),
        "mean_return": float(np.mean([float(row["return"]) for row in selected])),
    }


def _paired_metrics(rows):
    expected_steps = {0, 3000, 6000}
    actual_steps = {int(row["global_step"]) for row in rows}
    if actual_steps != expected_steps or len(rows) != 9:
        raise ValueError("L277 validation grid is incomplete")
    initial = _step_metrics(rows, 0)
    diagnostic = _step_metrics(rows, 3000)
    final = _step_metrics(rows, 6000)
    return {
        "initial": initial,
        "diagnostic_3000": diagnostic,
        "final_6000": final,
        "completion_change": final["mean_completion"] - initial["mean_completion"],
        "cte_improvement": initial["mean_cte"] - final["mean_cte"],
        "goal_distance_improvement": (
            initial["mean_goal_distance"] - final["mean_goal_distance"]
        ),
        "return_change": final["mean_return"] - initial["mean_return"],
        "collision_increase": final["collisions"] - initial["collisions"],
        "success_loss": initial["successes"] - final["successes"],
    }


def _validate_run(run, training):
    output = ROOT / run["output_dir"]
    rows = _read_csv(output / "validation_episodes.csv")
    metrics = _paired_metrics(rows)
    checkpoints = {
        step: output / "checkpoints" / ("step_%09d.pt" % step)
        for step in training["expected_checkpoints"]
    }
    if not all(path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("L277 required checkpoint is missing")
    final_payload = torch.load(checkpoints[6000], map_location="cpu", weights_only=False)
    state = final_payload["training_state"]
    replay = final_payload.get("replay_buffer")
    replay_has_data = bool(replay is not None and "observations" in replay)
    initialization = state.get("actor_initialization") or {}
    expected_source = str((ROOT / run["initialization"]).resolve())
    engineering = {
        "global_step_exact": int(state.get("global_step", -1)) == 6000,
        "replay_included": replay_has_data,
        "finite_3k": _finite_checkpoint(checkpoints[3000]),
        "finite_6k": _finite_checkpoint(checkpoints[6000]),
        "full_agent_initialization": initialization.get("mode") == (
            "full_agent_parameters_and_normalizer_only"
        ),
        "initialization_path_exact": initialization.get("checkpoint") == expected_source,
        "optimizer_state_not_imported": initialization.get("optimizer_state_imported") is False,
        "replay_not_imported": initialization.get("replay_imported") is False,
        "training_counters_reset": initialization.get("training_counters_reset") is True,
    }
    return rows, metrics, engineering


def _gate(run_results, scene_improvements, gate):
    completion = [row["metrics"]["completion_change"] for row in run_results]
    cte = [row["metrics"]["cte_improvement"] for row in run_results]
    goal = [row["metrics"]["goal_distance_improvement"] for row in run_results]
    improved_seeds = sum(c > 0.0 or g > 0.0 for c, g in zip(cte, goal))
    checks = {
        "engineering_complete": all(
            all(row["engineering"].values()) for row in run_results
        ),
        "collision_nonincrease_each_seed": all(
            row["metrics"]["collision_increase"]
            <= int(gate["maximum_collision_increase_per_seed"])
            for row in run_results
        ),
        "success_nonloss_each_seed": all(
            row["metrics"]["success_loss"]
            <= int(gate["maximum_success_loss_per_seed"])
            for row in run_results
        ),
        "completion_regression_bound_each_seed": all(
            value >= -float(gate["maximum_per_seed_mean_completion_regression"])
            for value in completion
        ),
        "median_completion_nonnegative": float(np.median(completion))
        >= float(gate["minimum_median_completion_change"]),
        "cte_or_goal_effect": (
            float(np.median(cte)) >= float(gate["minimum_median_cte_improvement"])
            or float(np.median(goal))
            >= float(gate["minimum_median_goal_distance_improvement"])
        ),
        "seed_direction_coverage": improved_seeds
        >= int(gate["minimum_seeds_with_cte_or_goal_improvement"]),
        "scene_cte_coverage": sum(value > 0.0 for value in scene_improvements.values())
        >= int(gate["minimum_scenes_with_cte_improvement"]),
    }
    metrics = {
        "seed_completion_changes": completion,
        "median_completion_change": float(np.median(completion)),
        "seed_cte_improvements": cte,
        "median_cte_improvement": float(np.median(cte)),
        "seed_goal_distance_improvements": goal,
        "median_goal_distance_improvement": float(np.median(goal)),
        "seeds_with_cte_or_goal_improvement": int(improved_seeds),
        "scene_cte_improvements": scene_improvements,
        "scenes_with_cte_improvement": int(
            sum(value > 0.0 for value in scene_improvements.values())
        ),
    }
    return checks, metrics


def _evaluate(config, output):
    run_results = []
    scene_values = defaultdict(list)
    all_validation_rows = []
    for run in config["runs"]:
        rows, metrics, engineering = _validate_run(run, config["training"])
        run_results.append({
            "seed": int(run["seed"]),
            "metrics": metrics,
            "engineering": engineering,
        })
        for row in rows:
            all_validation_rows.append({"seed": int(run["seed"]), **row})
        by_scene_step = {
            (row["scene"], int(row["global_step"])): float(row["cross_track_rmse"])
            for row in rows
        }
        for scene in sorted({row["scene"] for row in rows}):
            scene_values[scene].append(
                by_scene_step[(scene, 0)] - by_scene_step[(scene, 6000)]
            )
    scene_improvements = {
        scene: float(np.mean(values)) for scene, values in sorted(scene_values.items())
    }
    checks, metrics = _gate(run_results, scene_improvements, config["gate"])
    summary = {
        "protocol": "L277",
        "status": "complete",
        "gate_pass": bool(all(checks.values())),
        "decision": (
            "small_budget_sac_gate_pass" if all(checks.values())
            else "small_budget_sac_gate_fail"
        ),
        "checks": checks,
        "metrics": metrics,
        "runs": run_results,
        "final_map_evaluation_authorized": False,
        "larger_validation_preregistration_authorized": bool(all(checks.values())),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_validation_rows[0]))
        writer.writeheader()
        writer.writerows(all_validation_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def run(config_path: Path, output: Path, device: str, evaluate_only=False):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary_path = ROOT / config["l276_summary"]["path"]
    if _sha256(summary_path) != config["l276_summary"]["sha256"]:
        raise ValueError("L277 frozen L276 summary SHA256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("gate_pass"):
        raise ValueError("L277 requires the positive frozen L276 Gate")
    for run_spec in config["runs"]:
        initialization = ROOT / run_spec["initialization"]
        if _sha256(initialization) != run_spec["initialization_sha256"]:
            raise ValueError("L277 initialization checkpoint SHA256 mismatch")
        if evaluate_only:
            continue
        run_output = ROOT / run_spec["output_dir"]
        if run_output.exists():
            raise FileExistsError("L277 formal run output already exists: %s" % run_output)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "experiments/rl/train_rl_sampling_prior.py"),
            "--config",
            str(ROOT / run_spec["config"]),
            "--initialize-agent-from",
            str(initialization),
            "--device",
            device,
            "--output-dir",
            str(run_output),
        ]
        print(json.dumps({
            "stage": "training_start",
            "seed": int(run_spec["seed"]),
            "device": device,
        }, sort_keys=True), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        print(json.dumps({
            "stage": "training_complete",
            "seed": int(run_spec["seed"]),
            "device": device,
        }, sort_keys=True), flush=True)
    return _evaluate(config, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run(Path(args.config).resolve(), Path(args.output_dir).resolve(), args.device, args.evaluate_only)


if __name__ == "__main__":
    main()

