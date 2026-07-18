#!/usr/bin/env python3
"""Merge disjoint paired-checkpoint shards and recompute clustered effects."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.evaluation.paired_checkpoint import (
    gate2_closed_loop_decision,
    paired_checkpoint_effects,
    validate_paired_run_provenance,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        required=True,
        help="comma-separated paired-run directories with disjoint seeds",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args(argv)
    run_dirs = [
        Path(value.strip()).resolve()
        for value in args.run_dirs.split(",")
        if value.strip()
    ]
    manifests = [load_json(path / "provenance.json") for path in run_dirs]
    validation = validate_paired_run_provenance(manifests)
    control_rows = []
    aligned_rows = []
    sources = []
    for path in run_dirs:
        control_path = path / "control_episodes.csv"
        aligned_path = path / "aligned_episodes.csv"
        control_rows.extend(load_csv(control_path))
        aligned_rows.extend(load_csv(aligned_path))
        sources.append({
            "run_dir": str(path),
            "provenance_sha256": sha256(path / "provenance.json"),
            "control_episodes_sha256": sha256(control_path),
            "aligned_episodes_sha256": sha256(aligned_path),
        })
    comparison = paired_checkpoint_effects(
        control_rows,
        aligned_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    comparison["gate"] = gate2_closed_loop_decision(comparison)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "control_episodes.csv", control_rows)
    write_csv(output / "aligned_episodes.csv", aligned_rows)
    (output / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "merge_provenance.json").write_text(
        json.dumps({
            "schema_version": 1,
            "validation": validation,
            "sources": sources,
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.seed),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output),
        "paired_cells": comparison["paired_cells"],
        "independent_clusters": comparison["independent_clusters"],
        "gate": comparison["gate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
