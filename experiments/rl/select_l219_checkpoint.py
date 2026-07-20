#!/usr/bin/env python3
"""Select one L219 checkpoint using validation-only control metrics."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _boolean(value):
    return str(value).strip().lower() == "true"


def _mean(values):
    values = list(values)
    return sum(values) / float(len(values))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = []

    for seed in (20262191, 20262192, 20262193):
        run = root / ("l219_expanded_actor_seed%d_30k_v2" % seed)
        summary = json.loads(
            (run / "training_summary.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (run / "run_metadata.json").read_text(encoding="utf-8")
        )
        if (
            int(summary["global_step"]) != 30000
            or int(summary["update_records"]) != 29001
            or int(metadata["training"]["seed"]) != seed
            or metadata["git_sha"] != "afb121835652e9918c83dede6593aa4f207a73c0"
        ):
            raise ValueError("L219 training provenance is invalid: %s" % run)
        grouped = defaultdict(list)
        for row in _rows(run / "validation_episodes.csv"):
            grouped[int(row["global_step"])].append(row)
        if set(grouped) != {0, 5000, 10000, 15000, 20000, 25000, 30000}:
            raise ValueError("validation checkpoint grid is incomplete")
        for step, rows in grouped.items():
            if len(rows) != 30:
                raise ValueError("each checkpoint must have 30 validation cells")
            checkpoint = (
                run / "checkpoints" / (
                    "initial.pt" if step == 0 else "step_%09d.pt" % step
                )
            )
            if not checkpoint.is_file():
                raise ValueError("candidate checkpoint is missing: %s" % checkpoint)
            candidates.append({
                "training_seed": seed,
                "global_step": step,
                "episodes": len(rows),
                "successes": sum(_boolean(row["success"]) for row in rows),
                "collisions": sum(_boolean(row["collision"]) for row in rows),
                "mean_path_completion": _mean(
                    float(row["path_completion_ratio"]) for row in rows
                ),
                "mean_cross_track_rmse": _mean(
                    float(row["cross_track_rmse"]) for row in rows
                ),
                "mean_goal_distance": _mean(
                    float(row["goal_distance"]) for row in rows
                ),
                "mean_return": _mean(float(row["return"]) for row in rows),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            })

    # Validation-only, lexicographic control rule. Safety is a hard first
    # criterion; success and completion precede scalar reward because extreme
    # accumulated penalties made reward magnitudes unsuitable for selecting a
    # path-navigation prior in the first L219 run.
    selected = min(candidates, key=lambda row: (
        row["collisions"],
        -row["successes"],
        -row["mean_path_completion"],
        row["mean_cross_track_rmse"],
        row["mean_goal_distance"],
        -row["mean_return"],
        row["training_seed"],
        row["global_step"],
    ))
    fields = list(candidates[0]) + ["selected"]
    with (output / "checkpoint_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                **candidate,
                "selected": candidate["checkpoint"] == selected["checkpoint"],
            })
    payload = {
        "status": "development_validation_selection",
        "formal_claim_allowed": False,
        "independent_training_seeds": [20262191, 20262192, 20262193],
        "candidate_count": len(candidates),
        "validation_cells_per_candidate": 30,
        "selection_rule": [
            "minimize collisions",
            "maximize successes",
            "maximize mean path completion",
            "minimize mean cross-track RMSE",
            "minimize mean goal distance",
            "maximize mean return",
            "deterministic seed/step tie break",
        ],
        "selected": selected,
    }
    (output / "checkpoint_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
