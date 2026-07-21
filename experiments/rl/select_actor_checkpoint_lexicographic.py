#!/usr/bin/env python3
"""Select an actor checkpoint from frozen validation episodes.

The selector is intentionally independent of training-time scalar scores.  It
implements the safety-first ordering preregistered for L222/L241 and writes a
machine-readable ranking with hashes for later provenance audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


METRICS = (
    "cross_track_rmse",
    "path_completion_ratio",
    "goal_distance",
    "return",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("invalid boolean value: {!r}".format(value))


def _checkpoint_path(run_dir: Path, global_step: int) -> Path:
    name = "initial.pt" if global_step == 0 else "step_{:09d}.pt".format(global_step)
    path = run_dir / "checkpoints" / name
    if not path.is_file():
        raise FileNotFoundError("validation checkpoint is missing: {}".format(path))
    return path


def _load_rows(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / "validation_episodes.csv"
    if not path.is_file():
        raise FileNotFoundError("validation log is missing: {}".format(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("validation log is empty: {}".format(path))
    required = {"global_step", "scene", "seed", "collision", "success", *METRICS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("{} is missing columns {}".format(path, sorted(missing)))
    return rows


def aggregate_run(run_dir: Path) -> List[Dict[str, object]]:
    rows = _load_rows(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "training_summary.json"
    if not metadata_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("run metadata or training summary is missing in {}".format(run_dir))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_seed = int(metadata["training"]["seed"])
    grouped: MutableMapping[int, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["global_step"])].append(row)

    expected_count = None
    aggregates: List[Dict[str, object]] = []
    for global_step, group in sorted(grouped.items()):
        checkpoint_candidate = (
            run_dir / "checkpoints" / (
                "initial.pt" if global_step == 0 else f"step_{global_step:09d}.pt"
            )
        )
        # Validation is emitted every 10k while L257 checkpoints are frozen at
        # 0/30k intervals. Keep the audit explicit and rank only reproducible
        # candidates with an actual checkpoint artifact.
        if not checkpoint_candidate.is_file():
            continue
        keys = [(row["scene"], int(row["seed"])) for row in group]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate validation episode key in {} at step {}".format(run_dir, global_step))
        if expected_count is None:
            expected_count = len(group)
        elif len(group) != expected_count:
            raise ValueError("incomplete validation block in {} at step {}".format(run_dir, global_step))
        values: Dict[str, List[float]] = {}
        for metric in METRICS:
            values[metric] = [float(row[metric]) for row in group]
            if not all(math.isfinite(value) for value in values[metric]):
                raise ValueError("non-finite {} in {} at step {}".format(metric, run_dir, global_step))
        checkpoint = _checkpoint_path(run_dir, global_step)
        aggregates.append(
            {
                "train_seed": train_seed,
                "global_step": global_step,
                "validation_episodes": len(group),
                "collisions": sum(_as_bool(row["collision"]) for row in group),
                "successes": sum(_as_bool(row["success"]) for row in group),
                "mean_path_completion_ratio": sum(values["path_completion_ratio"]) / len(group),
                "mean_cross_track_rmse": sum(values["cross_track_rmse"]) / len(group),
                "mean_goal_distance": sum(values["goal_distance"]) / len(group),
                "mean_return": sum(values["return"]) / len(group),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "validation_csv_sha256": _sha256(run_dir / "validation_episodes.csv"),
                "run_metadata_sha256": _sha256(metadata_path),
                "git_sha": metadata.get("git_sha", ""),
                "training_global_step": int(summary["global_step"]),
                "update_records": int(summary["update_records"]),
            }
        )
    return aggregates


def ranking_key(row: Mapping[str, object]) -> tuple:
    """Return the frozen L222/L241 safety-first lexicographic key."""

    return (
        int(row["collisions"]),
        -int(row["successes"]),
        -float(row["mean_path_completion_ratio"]),
        float(row["mean_cross_track_rmse"]),
        float(row["mean_goal_distance"]),
        -float(row["mean_return"]),
        int(row["train_seed"]),
        int(row["global_step"]),
    )


def select(run_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        candidates.extend(aggregate_run(run_dir.resolve()))
    if not candidates:
        raise ValueError("no candidates were provided")
    ranked = sorted(candidates, key=ranking_key)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["selected"] = rank == 1
    return ranked


def write_outputs(ranked: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ranked[0].keys())
    with (output_dir / "candidate_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)
    payload = {
        "selection_contract": [
            "min_collisions",
            "max_successes",
            "max_mean_path_completion_ratio",
            "min_mean_cross_track_rmse",
            "min_mean_goal_distance",
            "max_mean_return",
            "min_train_seed_tie_break",
            "min_global_step_tie_break",
        ],
        "candidate_count": len(ranked),
        "selected": dict(ranked[0]),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    ranked = select(args.run_dir)
    write_outputs(ranked, args.output_dir)
    print(json.dumps(ranked[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
