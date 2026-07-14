#!/usr/bin/env python3
"""Aggregate held-out RL metrics across independent training seeds."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


NUMERIC_METRICS = (
    "final_goal_distance",
    "trajectory_length",
    "minimum_clearance",
    "control_jerk",
    "safety_interventions",
    "planner_compute_ms_mean",
)


def _parse_run(value):
    parts = str(value).split("=", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError("--run must use METHOD=TRAINING_SEED=RESULT_DIR")
    method, training_seed, result_dir = parts
    return method, int(training_seed), Path(result_dir).resolve()


def _read_run(method, training_seed, result_dir):
    path = result_dir / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError("missing held-out episodes: %s" % path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("held-out episode table is empty: %s" % path)
    evaluation_seeds = tuple(sorted(int(row["seed"]) for row in rows))
    record = {
        "method": str(method),
        "training_seed": int(training_seed),
        "result_dir": str(result_dir),
        "evaluation_episodes": len(rows),
        "evaluation_seeds": ",".join(str(seed) for seed in evaluation_seeds),
        "success_rate": float(np.mean([
            float(row["success"] == "True") for row in rows
        ])),
        "collision_rate": float(np.mean([
            float(row["collision"] == "True") for row in rows
        ])),
    }
    for metric in NUMERIC_METRICS:
        record[metric + "_mean"] = float(np.mean([
            float(row[metric]) for row in rows
        ]))
    return record, evaluation_seeds


def summarize(run_specs):
    records = []
    evaluation_seed_sets = defaultdict(set)
    for method, training_seed, result_dir in run_specs:
        record, evaluation_seeds = _read_run(
            method, training_seed, result_dir
        )
        records.append(record)
        evaluation_seed_sets[method].add(evaluation_seeds)
    for method, seed_sets in evaluation_seed_sets.items():
        if len(seed_sets) != 1:
            raise ValueError(
                "method %s was evaluated on inconsistent held-out seeds" % method
            )

    aggregates = {}
    for method in sorted({record["method"] for record in records}):
        selected = [record for record in records if record["method"] == method]
        if len({record["training_seed"] for record in selected}) != len(selected):
            raise ValueError("duplicate training seed for method %s" % method)
        total_episodes = int(sum(record["evaluation_episodes"] for record in selected))
        aggregate = {
            "training_seeds": len(selected),
            "training_seed_values": sorted(
                int(record["training_seed"]) for record in selected
            ),
            "evaluation_episodes_total": total_episodes,
            "evaluation_seeds": selected[0]["evaluation_seeds"],
        }
        for metric in (
            "success_rate",
            "collision_rate",
        ) + tuple(name + "_mean" for name in NUMERIC_METRICS):
            values = np.asarray(
                [float(record[metric]) for record in selected], dtype=np.float64
            )
            aggregate[metric + "_training_seed_mean"] = float(values.mean())
            aggregate[metric + "_training_seed_std"] = float(
                values.std(ddof=1) if values.size > 1 else 0.0
            )
        aggregate["success_rate_pooled"] = float(np.average(
            [record["success_rate"] for record in selected],
            weights=[record["evaluation_episodes"] for record in selected],
        ))
        aggregate["collision_rate_pooled"] = float(np.average(
            [record["collision_rate"] for record in selected],
            weights=[record["evaluation_episodes"] for record in selected],
        ))
        aggregates[method] = aggregate
    return records, aggregates


def _write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="METHOD=TRAINING_SEED=RESULT_DIR; repeat for every trained model",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_specs = [_parse_run(value) for value in args.run]
    records, aggregates = summarize(run_specs)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_training_seed.csv", records)
    with (output / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2, sort_keys=True)
    print(json.dumps(aggregates, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
