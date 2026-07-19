#!/usr/bin/env python3
"""Merge independently executed factorial shards and analyze seed clusters."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.run_full_proposed_factorial import (  # noqa: E402
    ARMS,
    _factorial,
    _paired,
    _write_csv,
    metrics_for_profile,
)


def _coerce(value):
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer() and "." not in str(value):
        return int(number)
    return number


def load_shards(paths):
    rows = []
    for directory in paths:
        progress = Path(directory).resolve() / "progress.csv"
        with progress.open("r", newline="", encoding="utf-8") as handle:
            rows.extend({
                key: _coerce(value)
                for key, value in row.items()
            } for row in csv.DictReader(handle))
    return rows


def validate_complete_factorial(rows, required_metrics=()):
    blocks = {}
    identities = set()
    for row in rows:
        identity = (
            int(row["seed"]),
            str(row["scene"]),
            str(row["physics_domain"]),
            str(row["factorial_arm"]),
        )
        if identity in identities:
            raise ValueError("duplicate factorial episode: %r" % (identity,))
        identities.add(identity)
        for metric in required_metrics:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "missing or nonnumeric metric %s for %r"
                    % (metric, identity)
                )
            if not np.isfinite(value):
                raise ValueError(
                    "nonfinite metric %s for %r" % (metric, identity)
                )
        blocks.setdefault(str(row["block"]), set()).add(
            str(row["factorial_arm"])
        )
    incomplete = {
        block: sorted(set(ARMS) - arms)
        for block, arms in blocks.items()
        if arms != set(ARMS)
    }
    if incomplete:
        raise ValueError("incomplete factorial blocks: %r" % incomplete)
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 2:
        raise ValueError("analysis requires at least two independent seeds")
    return {
        "episode_count": len(rows),
        "block_count": len(blocks),
        "seeds": seeds,
    }


def analyze(rows, bootstrap_samples, seed, metrics=None):
    paired = {
        "full_vs_simple": _paired(
            rows,
            "ordinary_fixed",
            "full_proposed",
            "full_vs_simple",
            bootstrap_samples,
            seed,
            metrics,
        ),
        "value_at_fixed": _paired(
            rows,
            "ordinary_fixed",
            "value_fixed",
            "value_at_fixed",
            bootstrap_samples,
            seed + 1,
            metrics,
        ),
        "hss_at_ordinary": _paired(
            rows,
            "ordinary_fixed",
            "ordinary_adaptive",
            "hss_at_ordinary",
            bootstrap_samples,
            seed + 2,
            metrics,
        ),
        "hss_at_value": _paired(
            rows,
            "value_fixed",
            "full_proposed",
            "hss_at_value",
            bootstrap_samples,
            seed + 3,
            metrics,
        ),
        "value_at_adaptive": _paired(
            rows,
            "ordinary_adaptive",
            "full_proposed",
            "value_at_adaptive",
            bootstrap_samples,
            seed + 4,
            metrics,
        ),
    }
    return paired, _factorial(
        rows, bootstrap_samples, seed, metrics
    )


def _arm_summary(rows, metric_profile):
    result = {}
    for arm in ARMS:
        selected = [
            row for row in rows if row["factorial_arm"] == arm
        ]
        summary = {
            "episodes": len(selected),
            "successes": int(sum(bool(row["success"]) for row in selected)),
            "collisions": int(sum(bool(row["collision"]) for row in selected)),
            "final_goal_distance_mean": float(sum(
                float(row["final_goal_distance"]) for row in selected
            ) / len(selected)),
            "control_jerk_mean": float(sum(
                float(row["control_jerk"]) for row in selected
            ) / len(selected)),
            "planner_compute_ms_mean": float(sum(
                float(row["planner_compute_ms_mean"]) for row in selected
            ) / len(selected)),
        }
        if metric_profile == "path_tracking":
            summary.update({
                "cross_track_rmse_mean": float(np.mean([
                    float(row["cross_track_rmse"]) for row in selected
                ])),
                "path_completion_ratio_mean": float(np.mean([
                    float(row["path_completion_ratio"])
                    for row in selected
                ])),
                "tangent_heading_rmse_mean": float(np.mean([
                    float(row["tangent_heading_rmse"])
                    for row in selected
                ])),
                "cross_track_max_mean": float(np.mean([
                    float(row["cross_track_max"]) for row in selected
                ])),
            })
        result[arm] = summary
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260736)
    parser.add_argument(
        "--metric-profile",
        choices=("point_goal", "path_tracking"),
        default="point_goal",
    )
    args = parser.parse_args(argv)

    inputs = [
        Path(item.strip()).resolve()
        for item in args.input_dirs.split(",")
        if item.strip()
    ]
    rows = load_shards(inputs)
    comparison_metrics = metrics_for_profile(args.metric_profile)
    validation = validate_complete_factorial(
        rows, tuple(comparison_metrics)
    )
    paired, factorial = analyze(
        rows, args.bootstrap_samples, args.seed, comparison_metrics
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "combined_progress.csv", rows)
    (output / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "factorial_contrasts.json").write_text(
        json.dumps(factorial, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        **validation,
        "arms": _arm_summary(rows, args.metric_profile),
        "metric_profile": str(args.metric_profile),
        "bootstrap_samples": int(args.bootstrap_samples),
        "analysis_seed": int(args.seed),
        "input_dirs": [str(path) for path in inputs],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip(),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
