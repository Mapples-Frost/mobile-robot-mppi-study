#!/usr/bin/env python3
"""Read-only analysis for a completed seven-arm final benchmark."""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.run_final_paper_benchmark import (
    ARMS,
    _analyse,
    metrics_for_profile,
)
from experiments.rl.run_gate1_simple_combination import parse_ints


# These outcomes were explicitly frozen for the point-goal confirmation in
# docs/rl/217_complex_navigation_sealed_preregistration_2026-07-20.md.  Failed
# episodes have no time_to_goal_s, so ``steps`` is the non-selective bounded
# completion-time endpoint (all failures remain at the frozen step limit).
PREDECLARED_POINT_GOAL_METRICS = {
    "steps": False,
    "trajectory_length": False,
    "minimum_clearance": True,
    "spin_steps": False,
    "mean_abs_omega": False,
    "planner_compute_ms_p95": False,
    "planner_compute_ms_max": False,
    "safety_interventions": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_completed_rows(rows, expected_seeds):
    """Reject incomplete, duplicated, qualification or foreign-seed rows."""

    rows = list(rows)
    expected_seeds = tuple(int(value) for value in expected_seeds)
    if not rows:
        raise ValueError("formal progress table is empty")
    observed_seeds = sorted({int(row["seed"]) for row in rows})
    if observed_seeds != sorted(expected_seeds):
        raise ValueError(
            "formal seeds differ: expected=%s observed=%s"
            % (sorted(expected_seeds), observed_seeds)
        )
    if any(int(row.get("qualification", 0)) != 0 for row in rows):
        raise ValueError("qualification rows are forbidden in formal analysis")
    grouped = defaultdict(list)
    keys = set()
    for row in rows:
        key = (
            str(row["scene"]),
            str(row["physics_domain"]),
            int(row["seed"]),
            str(row["benchmark_arm"]),
        )
        if key in keys:
            raise ValueError("duplicated formal cell: %s" % (key,))
        keys.add(key)
        grouped[str(row["block"])].append(row)
    bad = {}
    for block, block_rows in grouped.items():
        observed = {str(row["benchmark_arm"]) for row in block_rows}
        if len(block_rows) != len(ARMS) or observed != set(ARMS):
            bad[block] = {
                "rows": len(block_rows),
                "missing": sorted(set(ARMS) - observed),
                "unexpected": sorted(observed - set(ARMS)),
            }
    if bad:
        raise ValueError("incomplete formal blocks: %s" % bad)
    return {
        "episodes": len(rows),
        "blocks": len(grouped),
        "independent_seeds": len(observed_seeds),
        "seeds": observed_seeds,
        "arms": list(ARMS),
        "qualification_rows": 0,
    }


def _numeric(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().lower() in (
        "true", "false"
    ):
        value = value.strip().lower() == "true"
    result = float(value)
    return result if np.isfinite(result) else None


def aggregate(rows, fields):
    """Aggregate descriptives without treating episodes as independent."""

    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark_arm"], "all")].append(row)
        groups[(row["benchmark_arm"], row["physics_domain"])].append(row)
    result = []
    for (arm, domain), values in sorted(groups.items()):
        item = {
            "benchmark_arm": arm,
            "physics_domain": domain,
            "episodes": len(values),
            "independent_seeds": len({row["seed"] for row in values}),
        }
        for field in fields:
            data = [
                value
                for value in (_numeric(row, field) for row in values)
                if value is not None
            ]
            item[field + "_mean"] = (
                float(np.mean(data)) if data else ""
            )
            item[field + "_std_episode_descriptive"] = (
                float(np.std(data, ddof=1)) if len(data) >= 2 else ""
            )
        result.append(item)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    args = parser.parse_args(argv)

    result_dir = Path(args.result_dir).resolve()
    progress_path = result_dir / "progress.csv"
    provenance_path = result_dir / "provenance.json"
    rows = _read_csv(progress_path)
    seeds = parse_ints(args.seeds)
    audit = validate_completed_rows(rows, seeds)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "formal_preregistered_benchmark":
        raise ValueError("provenance is not a formal benchmark")
    if sorted(provenance.get("seeds", ())) != sorted(seeds):
        raise ValueError("provenance seeds do not match analysis seeds")
    profiles = {str(row["metric_profile"]) for row in rows}
    if len(profiles) != 1:
        raise ValueError("metric profile changes inside formal results")
    profile = next(iter(profiles))
    metrics = dict(metrics_for_profile(profile))
    if profile == "point_goal":
        metrics.update(PREDECLARED_POINT_GOAL_METRICS)
    paired, factorial = _analyse(
        rows,
        int(args.bootstrap_samples),
        int(args.bootstrap_seed),
        metrics,
    )
    (result_dir / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (result_dir / "factorial_contrasts.json").write_text(
        json.dumps(factorial, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = list(metrics) + [
        "minimum_clearance",
        "spin_steps",
        "planner_compute_ms_p95",
    ]
    _write_csv(
        result_dir / "descriptive_summary.csv",
        aggregate(rows, fields),
    )
    audit.update({
        "status": "complete_and_analysed",
        "independent_unit": "seed",
        "repeated_strata": ["scene", "physics_domain"],
        "progress_path": str(progress_path),
        "progress_sha256": _sha256(progress_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metric_profile": profile,
    })
    (result_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
