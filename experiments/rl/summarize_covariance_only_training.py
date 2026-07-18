#!/usr/bin/env python3
"""Summarize independent covariance-only SAC training blocks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _boolean(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _summarize_directory(path):
    csv_path = path / "validation_episodes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        groups.setdefault(int(row["global_step"]), []).append(row)
    checkpoints = []
    for step, values in sorted(groups.items()):
        item = {
            "global_step": step,
            "episodes": len(values),
            "mean_return": float(np.mean([float(v["return"]) for v in values])),
            "mean_cross_track_rmse": float(np.mean([
                float(v["cross_track_rmse"]) for v in values
            ])),
            "success_rate": float(np.mean([_boolean(v["success"]) for v in values])),
            "collision_rate": float(np.mean([_boolean(v["collision"]) for v in values])),
            "mean_covariance_scale_v": float(np.mean([
                float(v["covariance_scale_0_mean"]) for v in values
            ])),
            "mean_covariance_scale_omega": float(np.mean([
                float(v["covariance_scale_1_mean"]) for v in values
            ])),
            "mean_covariance_scale_v_std": float(np.mean([
                float(v["covariance_scale_0_std"]) for v in values
            ])),
            "mean_covariance_scale_omega_std": float(np.mean([
                float(v["covariance_scale_1_std"]) for v in values
            ])),
        }
        checkpoints.append(item)
    initial = next(item for item in checkpoints if item["global_step"] == 0)
    eligible = [
        item for item in checkpoints
        if item["success_rate"] >= initial["success_rate"]
        and item["collision_rate"] <= initial["collision_rate"]
    ]
    best = max(eligible, key=lambda item: item["mean_return"])
    return {
        "directory": str(path.resolve()),
        "initial": initial,
        "best": best,
        "cross_track_reduction": (
            initial["mean_cross_track_rmse"] - best["mean_cross_track_rmse"]
        ),
        "cross_track_reduction_fraction": (
            (initial["mean_cross_track_rmse"] - best["mean_cross_track_rmse"])
            / initial["mean_cross_track_rmse"]
        ),
        "checkpoints": checkpoints,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    blocks = [_summarize_directory(Path(value)) for value in args.directories]
    reductions = [block["cross_track_reduction_fraction"] for block in blocks]
    summary = {
        "schema_version": 1,
        "blocks": blocks,
        "block_count": len(blocks),
        "mean_cross_track_reduction_fraction": float(np.mean(reductions)),
        "min_cross_track_reduction_fraction": float(np.min(reductions)),
        "blocks_with_positive_reduction": int(sum(value > 0.0 for value in reductions)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        key: value for key, value in summary.items() if key != "blocks"
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
