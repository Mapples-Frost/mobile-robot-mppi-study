#!/usr/bin/env python3
"""Paired, seed-clustered summary for path-conditioned Actor development."""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_effects,
)


METRICS = {
    "success": True,
    "collision": False,
    "return": True,
    "cross_track_rmse": False,
    "cross_track_max": False,
    "heading_rmse": False,
    "path_completion_ratio": True,
    "control_jerk": False,
}


def read_episode_rows(directory, method):
    path = Path(directory).resolve() / "episodes.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row, method=str(method)) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("episode table is empty: %s" % path)
    return rows


def parse_candidate(value):
    try:
        label, directory = str(value).split("=", 1)
    except ValueError as exc:
        raise ValueError(
            "candidate must use LABEL=DIRECTORY syntax"
        ) from exc
    label = label.strip()
    if not label:
        raise ValueError("candidate label must not be empty")
    return label, directory.strip()


def summarize(
    baseline_dir,
    candidates,
    bootstrap_samples=10000,
    seed=20260719,
):
    baseline = read_episode_rows(baseline_dir, "actor")
    result = {
        "baseline_dir": str(Path(baseline_dir).resolve()),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "metrics": dict(METRICS),
        "candidates": {},
    }
    for offset, (label, directory) in enumerate(candidates):
        if label in result["candidates"]:
            raise ValueError("duplicate candidate label: %s" % label)
        candidate = read_episode_rows(directory, "actor")
        result["candidates"][label] = {
            "directory": str(Path(directory).resolve()),
            "paired_effects": paired_checkpoint_effects(
                baseline,
                candidate,
                method="actor",
                metrics=METRICS,
                bootstrap_samples=int(bootstrap_samples),
                seed=int(seed) + offset,
            ),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="repeatable LABEL=DIRECTORY candidate",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)
    candidates = [parse_candidate(value) for value in args.candidate]
    result = summarize(
        args.baseline_dir,
        candidates,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    concise = {}
    for label, item in result["candidates"].items():
        metrics = item["paired_effects"]["metrics"]
        concise[label] = {
            name: {
                "candidate_mean": values["aligned_mean"],
                "favorable_effect": values["favorable_effect"],
                "ci95": values["ci95"],
            }
            for name, values in metrics.items()
            if name in (
                "success",
                "cross_track_rmse",
                "path_completion_ratio",
                "control_jerk",
            )
        }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
