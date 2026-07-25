#!/usr/bin/env python3
"""Audit realized control contribution in the Stage 2 paired matrix.

The nominal and residual episodes share an obstacle-process seed, but once
their states diverge their controls are not exact same-state counterfactuals.
Accordingly, this script labels control differences as *paired realized*
differences and does not claim that every difference is caused solely by the
current step's residual acceptance decision.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _controls(rows, prefix):
    return np.column_stack(
        (_number(rows, prefix + "_v"), _number(rows, prefix + "_omega"))
    )


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
    }


def _difference_metrics(nominal, residual):
    count = min(len(nominal), len(residual))
    delta = np.linalg.norm(residual[:count] - nominal[:count], axis=1)
    return {
        "common_steps": int(count),
        "mean_l2": float(np.mean(delta)),
        "p95_l2": float(np.quantile(delta, 0.95)),
        "maximum_l2": float(np.max(delta)),
        "fraction_above_1e_3": float(np.mean(delta > 1.0e-3)),
        "fraction_above_0_01": float(np.mean(delta > 0.01)),
        "fraction_above_0_05": float(np.mean(delta > 0.05)),
        "_delta": delta,
    }


def _finite_float(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _pair_record(root, nominal_row, residual_row):
    nominal_rows = _read_csv(root / nominal_row["run_dir"] / "trajectory.csv")
    residual_rows = _read_csv(root / residual_row["run_dir"] / "trajectory.csv")
    proposed = _difference_metrics(
        _controls(nominal_rows, "proposed"),
        _controls(residual_rows, "proposed"),
    )
    executed = _difference_metrics(
        _controls(nominal_rows, "executed"),
        _controls(residual_rows, "executed"),
    )
    applied = _difference_metrics(
        _controls(nominal_rows, "applied"),
        _controls(residual_rows, "applied"),
    )
    common = int(executed["common_steps"])
    accepted = (
        _number(residual_rows[:common], "residual_safety_shield_accepted")
        > 0.5
    )
    effective = accepted & (executed["_delta"] > 0.01)
    nominal_position = np.column_stack(
        (_number(nominal_rows[:common], "x"), _number(nominal_rows[:common], "y"))
    )
    residual_position = np.column_stack(
        (
            _number(residual_rows[:common], "x"),
            _number(residual_rows[:common], "y"),
        )
    )
    position_delta = np.linalg.norm(
        residual_position - nominal_position, axis=1
    )
    nominal_clearance = _number(nominal_rows, "clearance")
    residual_clearance = _number(residual_rows, "clearance")
    probability = _number(
        residual_rows,
        "probabilistic_obstacle_maximum_step_probability",
    )
    shield_deviation = _number(
        residual_rows, "residual_safety_shield_maximum_position_deviation_m"
    )
    baseline_probability = _number(
        residual_rows,
        "residual_safety_shield_nominal_baseline_maximum_probability",
    )
    candidate_nominal_probability = _number(
        residual_rows,
        "residual_safety_shield_candidate_nominal_maximum_probability",
    )
    candidate_residual_probability = _number(
        residual_rows,
        "residual_safety_shield_candidate_residual_maximum_probability",
    )
    progress_noninferior = (
        _number(
            residual_rows,
            "residual_safety_shield_nominal_progress_noninferior",
        )
        > 0.5
    )
    fallback = ~(
        _number(residual_rows, "residual_safety_shield_accepted") > 0.5
    )
    rejection_causes = {
        "nominal_view_risk_increase": (
            candidate_nominal_probability > baseline_probability + 1.0e-12
        ),
        "model_position_tube_violation": shield_deviation > 0.10 + 1.0e-12,
        "nominal_progress_regression": ~progress_noninferior,
        "candidate_probability_above_hard_threshold": (
            np.maximum(
                candidate_nominal_probability, candidate_residual_probability
            )
            > 0.20
        ),
    }
    explained = np.zeros(len(residual_rows), dtype=bool)
    for cause in rejection_causes.values():
        explained |= cause
    for metrics in (proposed, executed, applied):
        metrics.pop("_delta")
    return {
        "episode_seed": int(residual_row["seed"]),
        "model_block": int(residual_row["model_block"]),
        "checkpoint_seed": int(residual_row["checkpoint_seed"]),
        "nominal_steps": int(nominal_row["steps"]),
        "residual_steps": int(residual_row["steps"]),
        "step_delta": int(residual_row["steps"]) - int(nominal_row["steps"]),
        "nominal_success": bool(int(nominal_row["success"])),
        "residual_success": bool(int(residual_row["success"])),
        "nominal_collision": bool(int(nominal_row["collision"])),
        "residual_collision": bool(int(residual_row["collision"])),
        "shield_acceptance_fraction": float(np.mean(accepted)),
        "paired_realized_effective_acceptance_fraction": float(
            np.mean(effective)
        ),
        "accepted_steps_with_executed_delta_fraction": (
            float(np.mean(effective[accepted])) if np.any(accepted) else 0.0
        ),
        "paired_proposed_control_difference": proposed,
        "paired_executed_control_difference": executed,
        "paired_applied_control_difference": applied,
        "paired_position_difference": {
            "mean_m": float(np.mean(position_delta)),
            "p95_m": float(np.quantile(position_delta, 0.95)),
            "maximum_m": float(np.max(position_delta)),
            "fraction_above_0_10_m": float(np.mean(position_delta > 0.10)),
        },
        "nominal_clearance_m": {
            **_quantiles(nominal_clearance),
            "fraction_below_0_50": float(
                np.mean(nominal_clearance < 0.50)
            ),
            "fraction_below_0_75": float(
                np.mean(nominal_clearance < 0.75)
            ),
        },
        "residual_clearance_m": {
            **_quantiles(residual_clearance),
            "fraction_below_0_50": float(
                np.mean(residual_clearance < 0.50)
            ),
            "fraction_below_0_75": float(
                np.mean(residual_clearance < 0.75)
            ),
        },
        "residual_risk": {
            "maximum_step_probability": _finite_float(np.max(probability)),
            "p95_maximum_step_probability": _finite_float(
                np.quantile(probability, 0.95)
            ),
            "fraction_above_hard_threshold_0_20": float(
                np.mean(probability > 0.20)
            ),
        },
        "shield_model_position_deviation_m": {
            "mean": float(np.mean(shield_deviation)),
            "p95": float(np.quantile(shield_deviation, 0.95)),
            "maximum": float(np.max(shield_deviation)),
        },
        "fallback_reason_overlap": {
            name: (
                float(np.mean(cause[fallback])) if np.any(fallback) else 0.0
            )
            for name, cause in rejection_causes.items()
        },
        "fallback_unexplained_fraction": (
            float(np.mean((~explained)[fallback]))
            if np.any(fallback)
            else 0.0
        ),
    }


def _block_summary(records):
    summaries = []
    for block in sorted({record["model_block"] for record in records}):
        selected = [
            record for record in records if record["model_block"] == block
        ]
        summaries.append(
            {
                "model_block": block,
                "checkpoint_seed": selected[0]["checkpoint_seed"],
                "episode_count": len(selected),
                "collision_count": sum(
                    int(record["residual_collision"]) for record in selected
                ),
                "success_count": sum(
                    int(record["residual_success"]) for record in selected
                ),
                "median_step_delta": float(
                    np.median([record["step_delta"] for record in selected])
                ),
                "median_shield_acceptance_fraction": float(
                    np.median(
                        [
                            record["shield_acceptance_fraction"]
                            for record in selected
                        ]
                    )
                ),
                "median_effective_acceptance_fraction": float(
                    np.median(
                        [
                            record[
                                "paired_realized_effective_acceptance_fraction"
                            ]
                            for record in selected
                        ]
                    )
                ),
                "median_executed_control_difference_fraction_above_0_01": float(
                    np.median(
                        [
                            record["paired_executed_control_difference"][
                                "fraction_above_0_01"
                            ]
                            for record in selected
                        ]
                    )
                ),
                "maximum_paired_position_difference_m": float(
                    max(
                        record["paired_position_difference"]["maximum_m"]
                        for record in selected
                    )
                ),
            }
        )
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.artifact_root).resolve()
    episode_rows = _read_csv(root / "episode_summary.csv")
    nominal = {
        int(row["seed"]): row
        for row in episode_rows
        if row["condition"] == "nominal"
    }
    residual = [
        row for row in episode_rows if row["condition"] == "icode_residual"
    ]
    records = [
        _pair_record(root.parent.parent, nominal[int(row["seed"])], row)
        for row in residual
    ]
    records.sort(key=lambda row: (row["model_block"], row["episode_seed"]))
    result = {
        "schema_version": 1,
        "interpretation": (
            "paired realized differences after closed-loop state divergence; "
            "not exact same-state per-step counterfactual effects"
        ),
        "pair_count": len(records),
        "all_residual_success": all(
            record["residual_success"] for record in records
        ),
        "all_residual_collision_free": all(
            not record["residual_collision"] for record in records
        ),
        "records": records,
        "block_summary": _block_summary(records),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
