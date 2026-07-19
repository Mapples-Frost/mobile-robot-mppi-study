#!/usr/bin/env python3
"""Select one L185 checkpoint using the preregistered lexicographic rule."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_STEPS = (10000, 20000, 30000, 40000, 50000, 60000)


def _boolean(value):
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError("invalid boolean value: %s" % value)


def _finite(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("%s contains NaN or Inf" % name)
    return value


def aggregate_validation_rows(rows, expected_steps=DEFAULT_STEPS):
    """Aggregate complete validation blocks and return the selected step."""

    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        step = int(float(row["global_step"]))
        key = (step, str(row["scene"]), int(float(row["seed"])))
        if key in seen:
            raise ValueError("duplicate validation scene/seed at step %d" % step)
        seen.add(key)
        grouped[step].append(row)
    expected = tuple(int(value) for value in expected_steps)
    missing = [step for step in expected if step not in grouped]
    if missing:
        raise ValueError("missing validation checkpoints: %s" % missing)
    unexpected = sorted(set(grouped) - set(expected))
    if unexpected:
        raise ValueError("unexpected validation checkpoints: %s" % unexpected)
    block_size = len(grouped[expected[0]])
    if block_size <= 0 or any(len(grouped[step]) != block_size for step in expected):
        raise ValueError("validation checkpoint blocks are incomplete")

    aggregates = []
    for step in expected:
        block = grouped[step]
        collisions = sum(int(_boolean(row["collision"])) for row in block)
        successes = sum(int(_boolean(row["success"])) for row in block)
        cross_track = float(np.mean([
            _finite(row, "cross_track_rmse") for row in block
        ]))
        completion = float(np.mean([
            _finite(row, "path_completion_ratio") for row in block
        ]))
        mean_return = float(np.mean([
            _finite(row, "return") for row in block
        ]))
        aggregates.append({
            "global_step": step,
            "episodes": block_size,
            "collisions": collisions,
            "successes": successes,
            "success_rate": successes / float(block_size),
            "collision_rate": collisions / float(block_size),
            "mean_cross_track_rmse": cross_track,
            "mean_path_completion_ratio": completion,
            "mean_return": mean_return,
        })
    selected = min(
        aggregates,
        key=lambda row: (
            row["collisions"],
            -row["successes"],
            row["mean_cross_track_rmse"],
            -row["mean_path_completion_ratio"],
            -row["mean_return"],
            row["global_step"],
        ),
    )
    return {
        "schema_version": 1,
        "selection_rule": [
            "minimum_collisions",
            "maximum_successes",
            "minimum_mean_cross_track_rmse",
            "maximum_mean_path_completion_ratio",
            "maximum_mean_return",
            "earliest_checkpoint",
        ],
        "expected_steps": list(expected),
        "episodes_per_checkpoint": block_size,
        "aggregates": aggregates,
        "selected_global_step": int(selected["global_step"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--expected-steps",
        default=",".join(str(value) for value in DEFAULT_STEPS),
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    source = run_dir / "validation_episodes.csv"
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = tuple(
        int(value.strip())
        for value in str(args.expected_steps).split(",")
        if value.strip()
    )
    result = aggregate_validation_rows(rows, expected_steps=expected)
    step = result["selected_global_step"]
    checkpoint = run_dir / "checkpoints" / ("step_%09d.pt" % step)
    if not checkpoint.is_file():
        raise FileNotFoundError("selected checkpoint is missing: %s" % checkpoint)
    result.update({
        "run_dir": str(run_dir),
        "validation_csv": str(source),
        "selected_checkpoint": str(checkpoint),
    })
    output = Path(
        args.output or run_dir / "path_checkpoint_selection.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
