#!/usr/bin/env python3
"""Audit and merge preregistered final-benchmark shards.

The script is deliberately strict: it refuses qualification rows, incomplete
randomized blocks, duplicate cells, missing run artifacts, configuration-hash
mismatches, or inconsistent Git/manifest/checkpoint provenance.  Raw runs stay
in their immutable shard directories; the merged directory contains a compact
index and analysis-ready tables that reference those sources.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.rl.run_final_paper_benchmark import ARMS
from mobile_robot_mppi.core.config import config_hash


REQUIRED_RUN_FILES = (
    "config_resolved.yaml",
    "trajectory.csv",
    "metrics.json",
    "provenance.json",
)
REQUIRED_NUMERIC_FIELDS = (
    "steps",
    "success",
    "collision",
    "final_goal_distance",
    "trajectory_length",
    "minimum_clearance",
    "control_jerk",
    "stuck_steps",
    "planner_compute_ms_mean",
    "planner_compute_ms_max",
    "paper_total_rollouts_mean",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_key(row, arm_key="benchmark_arm"):
    return (
        str(row["scene"]),
        str(row["physics_domain"]),
        int(row["seed"]),
        str(row[arm_key]),
    )


def numeric(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return float(value.strip().lower() == "true")
    return float(value)


def verify_hash_reference(item, label):
    path = Path(item["path"])
    if not path.is_file():
        raise ValueError("missing %s: %s" % (label, path))
    observed = sha256(path)
    expected = str(item["sha256"])
    if observed != expected:
        raise ValueError(
            "%s hash mismatch: expected=%s observed=%s path=%s"
            % (label, expected, observed, path)
        )
    return {"path": str(path), "sha256": observed}


def checkpoint_references(provenance):
    references = []
    for key in (
        "actor_checkpoint",
        "ordinary_checkpoints",
        "ordinary_reliability_gate",
        "value_checkpoints",
        "value_reliability_gate",
    ):
        value = provenance.get(key)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        references.extend((key, item) for item in values)
    return references


def validate_run_artifacts(shard, row, expected_git_sha):
    arm = str(row["benchmark_arm"])
    run_name = "%s__%s__%s__seed%d" % (
        arm,
        row["scene"],
        row["physics_domain"],
        int(row["seed"]),
    )
    run_dir = shard / "runs" / arm / run_name
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValueError("missing run artifacts in %s: %s" % (run_dir, missing))

    provenance = read_json(run_dir / "provenance.json")
    if provenance.get("git_sha") != expected_git_sha:
        raise ValueError("per-run Git SHA mismatch: %s" % run_dir)
    resolved = yaml.safe_load(
        (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    observed_config_hash = config_hash(resolved)
    if provenance.get("config_hash") != observed_config_hash:
        raise ValueError("per-run config hash mismatch: %s" % run_dir)

    trajectory = read_csv(run_dir / "trajectory.csv")
    if not trajectory:
        raise ValueError("empty trajectory: %s" % run_dir)
    metrics = read_json(run_dir / "metrics.json")
    for field in REQUIRED_NUMERIC_FIELDS:
        if field not in row or row[field] in (None, ""):
            raise ValueError("missing required progress field %s: %s" % (field, run_dir))
        value = numeric(row[field])
        if not math.isfinite(value):
            raise ValueError("non-finite progress field %s: %s" % (field, run_dir))
        if field in metrics and not math.isclose(
            value, numeric(metrics[field]), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("metrics/progress mismatch for %s: %s" % (field, run_dir))
    if numeric(row["success"]) == 1.0:
        value = numeric(row["time_to_goal_s"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("invalid successful time-to-goal: %s" % run_dir)
    return {
        "run_dir": str(run_dir.resolve()),
        "config_hash": observed_config_hash,
        "trajectory_rows": len(trajectory),
    }


def audit_and_merge(shards, output):
    shards = [Path(path).resolve() for path in shards]
    output = Path(output).resolve()
    if len(shards) < 2:
        raise ValueError("at least two shard directories are required")

    rows = []
    schedule = []
    provenance_items = []
    source_shards = []
    shard_indices = set()
    selected_seed_sets = []
    verified_hashes = {}
    run_index = []

    for shard in shards:
        provenance_path = shard / "provenance.json"
        progress_path = shard / "progress.csv"
        schedule_path = shard / "schedule.json"
        if not all(path.is_file() for path in (provenance_path, progress_path, schedule_path)):
            raise ValueError("shard lacks top-level provenance/progress/schedule: %s" % shard)
        provenance = read_json(provenance_path)
        shard_rows = read_csv(progress_path)
        shard_schedule = read_json(schedule_path)
        if provenance.get("status") != "formal_preregistered_benchmark":
            raise ValueError("non-formal shard: %s" % shard)
        shard_meta = provenance.get("formal_shard", {})
        shard_index = int(shard_meta.get("index", -1))
        shard_count = int(shard_meta.get("count", -1))
        if shard_count != len(shards) or shard_index in shard_indices:
            raise ValueError("invalid or duplicated shard metadata: %s" % shard)
        shard_indices.add(shard_index)
        shard_seeds = sorted({int(row["seed"]) for row in shard_rows})
        if shard_seeds != sorted(int(seed) for seed in provenance.get("seeds", [])):
            raise ValueError("shard rows/provenance seed mismatch: %s" % shard)
        selected_seed_sets.append(set(shard_seeds))
        if any(int(row.get("qualification", 0)) != 0 for row in shard_rows):
            raise ValueError("qualification row in formal shard: %s" % shard)

        progress_keys = {row_key(row) for row in shard_rows}
        schedule_keys = {row_key(item, arm_key="arm") for item in shard_schedule}
        if len(progress_keys) != len(shard_rows) or len(schedule_keys) != len(shard_schedule):
            raise ValueError("duplicate progress or schedule identity in %s" % shard)
        if progress_keys != schedule_keys:
            raise ValueError("progress/schedule identities differ in %s" % shard)

        for label, reference in checkpoint_references(provenance):
            cache_key = (str(reference["path"]), str(reference["sha256"]))
            if cache_key not in verified_hashes:
                verified_hashes[cache_key] = verify_hash_reference(reference, label)
        for row in shard_rows:
            run_index.append(validate_run_artifacts(shard, row, provenance["git_sha"]))

        source_shards.append({
            "index": shard_index,
            "path": str(shard),
            "seeds": shard_seeds,
            "episodes": len(shard_rows),
            "progress_sha256": sha256(progress_path),
            "schedule_sha256": sha256(schedule_path),
            "provenance_sha256": sha256(provenance_path),
        })
        provenance_items.append(provenance)
        rows.extend(shard_rows)
        schedule.extend(shard_schedule)

    if shard_indices != set(range(len(shards))):
        raise ValueError("shard indices are not contiguous from zero")
    for index, seeds in enumerate(selected_seed_sets):
        others = set().union(*(item for j, item in enumerate(selected_seed_sets) if j != index))
        if seeds & others:
            raise ValueError("seed assigned to multiple shards: %s" % sorted(seeds & others))

    reference = provenance_items[0]
    invariant_fields = (
        "arms",
        "git_sha",
        "manifest_sha256",
        "metric_profile",
        "physics_domains",
        "scenes",
        "schedule_seed",
        "sealed_seeds",
        "status",
    )
    for provenance in provenance_items[1:]:
        for field in invariant_fields:
            if provenance.get(field) != reference.get(field):
                raise ValueError("cross-shard provenance mismatch: %s" % field)

    sealed_seeds = sorted(int(seed) for seed in reference["sealed_seeds"])
    observed_seeds = sorted(set().union(*selected_seed_sets))
    if observed_seeds != sealed_seeds:
        raise ValueError("shard union differs from sealed seed manifest")
    arms = tuple(reference["arms"])
    scenes = tuple(item["name"] for item in reference["scenes"])
    domains = tuple(item["name"] for item in reference["physics_domains"])
    expected = {
        (scene, domain, seed, arm)
        for scene in scenes
        for domain in domains
        for seed in sealed_seeds
        for arm in arms
    }
    observed = {row_key(row) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            "formal matrix mismatch: expected=%d observed_rows=%d unique=%d missing=%d extra=%d"
            % (len(expected), len(rows), len(observed), len(expected - observed), len(observed - expected))
        )
    if set(arms) != set(ARMS):
        raise ValueError("formal arm set differs from the frozen seven-arm design")

    blocks = defaultdict(list)
    for row in rows:
        blocks[str(row["block"])].append(row)
    for block, block_rows in blocks.items():
        if len(block_rows) != len(arms) or {row["benchmark_arm"] for row in block_rows} != set(arms):
            raise ValueError("incomplete randomized complete block: %s" % block)
    orders = [int(item["global_run_order"]) for item in schedule]
    if len(orders) != len(set(orders)) or sorted(orders) != list(range(len(schedule))):
        raise ValueError("global schedule order is not a unique 0..N-1 permutation")

    manifest = Path(reference["manifest"])
    if not manifest.is_file() or sha256(manifest) != reference["manifest_sha256"]:
        raise ValueError("frozen manifest missing or hash mismatch")

    output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row["global_run_order"]))
    schedule.sort(key=lambda item: int(item["global_run_order"]))
    write_csv(output / "progress.csv", rows)
    (output / "schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    merged_provenance = dict(reference)
    merged_provenance.update({
        "seeds": sealed_seeds,
        "formal_shard": {"count": len(shards), "index": "merged"},
        "source_shards": sorted(source_shards, key=lambda item: item["index"]),
        "raw_runs_are_referenced_not_copied": True,
    })
    (output / "provenance.json").write_text(
        json.dumps(merged_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_index.sort(key=lambda item: item["run_dir"])
    (output / "run_artifact_index.json").write_text(
        json.dumps(run_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    field_missing = {
        field: sum(row.get(field) in (None, "") for row in rows)
        for field in rows[0]
    }
    audit = {
        "status": "passed",
        "episodes": len(rows),
        "unique_cells": len(observed),
        "blocks": len(blocks),
        "independent_seeds": len(sealed_seeds),
        "arms": list(arms),
        "scenes": list(scenes),
        "physics_domains": list(domains),
        "qualification_rows": 0,
        "duplicate_cells": 0,
        "verified_run_artifacts": len(run_index),
        "verified_files_per_run": list(REQUIRED_RUN_FILES),
        "verified_external_hash_references": len(verified_hashes),
        "git_sha": reference["git_sha"],
        "manifest_sha256": reference["manifest_sha256"],
        "progress_sha256": sha256(output / "progress.csv"),
        "schedule_sha256": sha256(output / "schedule.json"),
        "field_missing_counts": field_missing,
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
        "source_shards": sorted(source_shards, key=lambda item: item["index"]),
    }
    (output / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    audit = audit_and_merge(args.shards, args.output_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
