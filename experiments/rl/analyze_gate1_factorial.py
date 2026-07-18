#!/usr/bin/env python3
"""Recompute Gate 1 contrasts without rerunning closed-loop episodes."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.evaluation.factorial import (
    blocked_factorial_contrasts,
)


METRICS = (
    ("success", True),
    ("final_goal_distance", False),
    ("control_jerk", False),
    ("minimum_clearance", True),
    ("planner_compute_ms_mean", False),
)


def analyze(rows, bootstrap_samples=5000, seed=20260718):
    return {
        metric: blocked_factorial_contrasts(
            rows,
            metric,
            higher_is_better=higher_is_better,
            bootstrap_samples=bootstrap_samples,
            seed=int(seed) + offset,
            cluster_key="seed",
        )
        for offset, (metric, higher_is_better) in enumerate(METRICS)
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args(argv)

    with Path(args.episodes_csv).open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    result = analyze(rows, args.bootstrap_samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
