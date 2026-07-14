"""Reproducible aggregation helpers for multi-seed controller benchmarks."""

import csv
import json
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def aggregate_episode_summaries(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("at least one episode summary is required")
    aggregate: Dict[str, Any] = {"num_runs": len(rows)}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        values = [row.get(key) for row in rows]
        if all(isinstance(value, bool) for value in values):
            aggregate[key + "_rate"] = float(np.mean(values))
            continue
        finite_numeric = [
            float(value)
            for value in values
            if isinstance(value, Real)
            and not isinstance(value, bool)
            and np.isfinite(float(value))
        ]
        if len(finite_numeric) == len(values):
            data = np.asarray(finite_numeric, dtype=np.float64)
            aggregate[key] = {
                "mean": float(data.mean()),
                "std": float(data.std(ddof=0)),
                "min": float(data.min()),
                "max": float(data.max()),
            }
    reasons = Counter(str(row.get("termination_reason", "unknown")) for row in rows)
    aggregate["termination_counts"] = dict(sorted(reasons.items()))
    return aggregate


def write_benchmark_artifacts(output_dir, rows, metadata=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty benchmark")
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "aggregate": aggregate_episode_summaries(rows),
        "runs": list(rows),
        "metadata": dict(metadata or {}),
    }
    with (output / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    return payload
