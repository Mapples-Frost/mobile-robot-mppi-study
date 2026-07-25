"""Compare optimized Stage 3 episodes with the qualified Stage 2 matrix."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _controls(rows, prefix):
    return np.column_stack(
        (_number(rows, prefix + "_v"), _number(rows, prefix + "_omega"))
    )


def _trajectory_metrics(baseline_rows, optimized_rows):
    count = min(len(baseline_rows), len(optimized_rows))
    baseline_position = np.column_stack(
        (_number(baseline_rows[:count], "x"), _number(baseline_rows[:count], "y"))
    )
    optimized_position = np.column_stack(
        (
            _number(optimized_rows[:count], "x"),
            _number(optimized_rows[:count], "y"),
        )
    )
    position = np.linalg.norm(
        optimized_position - baseline_position, axis=1
    )
    baseline_control = _controls(baseline_rows[:count], "executed")
    optimized_control = _controls(optimized_rows[:count], "executed")
    control = np.linalg.norm(optimized_control - baseline_control, axis=1)
    return {
        "common_steps": int(count),
        "position_difference_m": {
            "mean": float(np.mean(position)),
            "p95": float(np.quantile(position, 0.95)),
            "maximum": float(np.max(position)),
        },
        "executed_control_difference": {
            "mean_l2": float(np.mean(control)),
            "p95_l2": float(np.quantile(control, 0.95)),
            "maximum_l2": float(np.max(control)),
            "fraction_above_0_01": float(np.mean(control > 0.01)),
        },
    }


def _key(row):
    return (
        str(row["condition"]),
        int(row["model_block"]),
        int(row["seed"]),
    )


def _run_dir(artifact_root, row):
    value = Path(str(row["run_dir"]))
    return value if value.is_absolute() else ROOT / value


def run(baseline_root, optimized_root):
    baseline_root = Path(baseline_root).resolve()
    optimized_root = Path(optimized_root).resolve()
    baseline_rows = {
        _key(row): row
        for row in _read_csv(baseline_root / "episode_summary.csv")
    }
    optimized_rows = {
        _key(row): row
        for row in _read_csv(optimized_root / "episode_summary.csv")
    }
    if set(baseline_rows) != set(optimized_rows):
        raise ValueError("baseline and optimized episode cells differ")
    records = []
    for key in sorted(baseline_rows):
        baseline = baseline_rows[key]
        optimized = optimized_rows[key]
        baseline_trajectory = _read_csv(
            _run_dir(baseline_root, baseline) / "trajectory.csv"
        )
        optimized_trajectory = _read_csv(
            _run_dir(optimized_root, optimized) / "trajectory.csv"
        )
        baseline_p95 = float(baseline["planner_p95_compute_ms"])
        optimized_p95 = float(optimized["planner_p95_compute_ms"])
        record = {
            "condition": key[0],
            "model_block": key[1],
            "episode_seed": key[2],
            "baseline_success": bool(int(baseline["success"])),
            "optimized_success": bool(int(optimized["success"])),
            "baseline_collision": bool(int(baseline["collision"])),
            "optimized_collision": bool(int(optimized["collision"])),
            "baseline_steps": int(baseline["steps"]),
            "optimized_steps": int(optimized["steps"]),
            "step_delta": int(optimized["steps"]) - int(baseline["steps"]),
            "baseline_planner_p95_ms": baseline_p95,
            "optimized_planner_p95_ms": optimized_p95,
            "planner_p95_speedup_fraction": float(
                1.0 - optimized_p95 / baseline_p95
            ),
            **_trajectory_metrics(
                baseline_trajectory, optimized_trajectory
            ),
        }
        if key[0] == "icode_residual":
            baseline_acceptance = np.mean(
                _number(
                    baseline_trajectory,
                    "residual_safety_shield_accepted",
                )
                > 0.5
            )
            optimized_acceptance = np.mean(
                _number(
                    optimized_trajectory,
                    "residual_safety_shield_accepted",
                )
                > 0.5
            )
            record.update(
                {
                    "baseline_shield_acceptance_fraction": float(
                        baseline_acceptance
                    ),
                    "optimized_shield_acceptance_fraction": float(
                        optimized_acceptance
                    ),
                    "shield_acceptance_delta": float(
                        optimized_acceptance - baseline_acceptance
                    ),
                }
            )
        records.append(record)
    residual = [
        row for row in records if row["condition"] == "icode_residual"
    ]
    result = {
        "schema_version": 1,
        "stage": "residual_runtime_stage3_equivalence_audit",
        "baseline": str(baseline_root.relative_to(ROOT)),
        "optimized": str(optimized_root.relative_to(ROOT)),
        "episode_count": len(records),
        "all_optimized_success": all(
            row["optimized_success"] for row in records
        ),
        "all_optimized_collision_free": all(
            not row["optimized_collision"] for row in records
        ),
        "all_outcomes_match": all(
            row["baseline_success"] == row["optimized_success"]
            and row["baseline_collision"] == row["optimized_collision"]
            for row in records
        ),
        "residual_planner_p95_ms": {
            "maximum": float(
                max(row["optimized_planner_p95_ms"] for row in residual)
            ),
            "median": float(
                np.median(
                    [row["optimized_planner_p95_ms"] for row in residual]
                )
            ),
            "median_speedup_fraction": float(
                np.median(
                    [row["planner_p95_speedup_fraction"] for row in residual]
                )
            ),
        },
        "records": records,
        "sealed_seeds_opened": False,
        "rl_enabled": False,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.baseline, args.optimized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
