#!/usr/bin/env python3
"""Apply the frozen L187 terminal-convergence development decisions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = {
    "ordinary_fixed",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
}


def _truth(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError("expected boolean value, received %r" % value)
    return normalized == "true"


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("factorial table is empty: %s" % path)
    return rows


def _validate(rows):
    keys = set()
    for row in rows:
        arm = str(row["factorial_arm"])
        if arm not in ARMS:
            raise ValueError("unknown factorial arm: %s" % arm)
        key = (
            arm,
            str(row["physics_domain"]),
            int(row["seed"]),
        )
        if key in keys:
            raise ValueError("duplicate factorial cell: %r" % (key,))
        keys.add(key)
    return keys


def diagnostic_decision(control_rows, candidate_rows):
    control_keys = _validate(control_rows)
    candidate_keys = _validate(candidate_rows)
    if control_keys != candidate_keys:
        raise ValueError("T0 and T1 factorial cells differ")

    def aggregate(rows):
        return {
            "episodes": len(rows),
            "successes": int(sum(_truth(row["success"]) for row in rows)),
            "collisions": int(sum(_truth(row["collision"]) for row in rows)),
            "mean_cross_track_rmse": float(np.mean([
                float(row["cross_track_rmse"]) for row in rows
            ])),
            "mean_control_jerk": float(np.mean([
                float(row["control_jerk"]) for row in rows
            ])),
        }

    control = aggregate(control_rows)
    candidate = aggregate(candidate_rows)
    cross_track_ratio = (
        candidate["mean_cross_track_rmse"]
        / max(control["mean_cross_track_rmse"], 1e-12)
    )
    jerk_ratio = (
        candidate["mean_control_jerk"]
        / max(control["mean_control_jerk"], 1e-12)
    )
    criteria = {
        "increased_successes": (
            candidate["successes"] > control["successes"]
        ),
        "zero_candidate_collisions": candidate["collisions"] == 0,
        "cross_track_regression_within_10pct": cross_track_ratio <= 1.10,
        "jerk_regression_within_10pct": jerk_ratio <= 1.10,
    }
    return {
        "t0": control,
        "t1": candidate,
        "cross_track_ratio_t1_over_t0": cross_track_ratio,
        "jerk_ratio_t1_over_t0": jerk_ratio,
        "criteria": criteria,
        "candidate_selected": bool(all(criteria.values())),
    }


def confirmation_decision(rows):
    _validate(rows)
    full = {
        (str(row["physics_domain"]), int(row["seed"])): row
        for row in rows
        if row["factorial_arm"] == "full_proposed"
    }
    ordinary = {
        (str(row["physics_domain"]), int(row["seed"])): row
        for row in rows
        if row["factorial_arm"] == "ordinary_fixed"
    }
    if not full or set(full) != set(ordinary):
        raise ValueError(
            "confirmation requires paired Full Proposed and ordinary fixed"
        )
    full_rows = list(full.values())
    ordinary_rows = [ordinary[key] for key in full]
    full_cross_track = float(np.mean([
        float(row["cross_track_rmse"]) for row in full_rows
    ]))
    ordinary_cross_track = float(np.mean([
        float(row["cross_track_rmse"]) for row in ordinary_rows
    ]))
    full_jerk = float(np.mean([
        float(row["control_jerk"]) for row in full_rows
    ]))
    ordinary_jerk = float(np.mean([
        float(row["control_jerk"]) for row in ordinary_rows
    ]))
    criteria = {
        "full_success_every_episode": all(
            _truth(row["success"]) for row in full_rows
        ),
        "full_zero_collisions": not any(
            _truth(row["collision"]) for row in full_rows
        ),
        "full_cross_track_within_5pct": (
            full_cross_track <= 1.05 * ordinary_cross_track
        ),
        "full_jerk_within_10pct": (
            full_jerk <= 1.10 * ordinary_jerk
        ),
    }
    return {
        "paired_blocks": len(full),
        "full_successes": int(sum(
            _truth(row["success"]) for row in full_rows
        )),
        "full_collisions": int(sum(
            _truth(row["collision"]) for row in full_rows
        )),
        "full_cross_track_rmse_mean": full_cross_track,
        "ordinary_cross_track_rmse_mean": ordinary_cross_track,
        "full_control_jerk_mean": full_jerk,
        "ordinary_control_jerk_mean": ordinary_jerk,
        "criteria": criteria,
        "integration_gate_passed": bool(all(criteria.values())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--confirmation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = {
        "diagnostic": diagnostic_decision(
            _read(args.t0), _read(args.t1)
        ),
        "confirmation": (
            None
            if not args.confirmation
            else confirmation_decision(_read(args.confirmation))
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
