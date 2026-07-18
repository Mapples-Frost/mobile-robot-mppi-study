#!/usr/bin/env python3
"""Merge and fail-closed audit L21 counterfactual branch datasets."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.run_correction_advantage_diagnostic import _write_csv
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.risk_dataset import (
    assess_risk_dataset_sufficiency,
    burst_accounting_valid,
)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_numeric(rows):
    for row in rows:
        for value in row.values():
            if value is None or str(value).strip() == "":
                return False
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_dirs = [Path(value).resolve() for value in args.run_dir]
    if len(run_dirs) < 2:
        raise ValueError("risk dataset audit requires multiple training seeds")

    metadata_rows = []
    sample_rows = []
    reference_rows = []
    arrays = []
    training_seeds = []
    expected = None
    per_run = {}
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "metadata.json").read_text(
            encoding="utf-8"
        ))
        training_seed = int(metadata["training_seed"])
        if training_seed in training_seeds:
            raise ValueError("risk dataset training seeds must be unique")
        training_seeds.append(training_seed)
        contract = {
            name: metadata[name]
            for name in (
                "train_episode_seeds",
                "validation_episode_seeds",
                "sealed_test_episode_seeds",
                "branch_steps",
                "branch_horizon",
                "candidate_beta",
                "replay_atol",
                "label_config",
                "feature_schema",
                "scene",
            )
        }
        contract.update({
            "intervention_type": metadata.get(
                "intervention_type", "single_step_correction"
            ),
            "intervention_steps": int(metadata.get("intervention_steps", 1)),
        })
        if expected is None:
            expected = contract
        elif contract != expected:
            raise ValueError("risk dataset run contracts differ")
        rows = _read_csv(run_dir / "samples.csv")
        refs = _read_csv(run_dir / "reference_episodes.csv")
        data = np.load(run_dir / "samples.npz")
        required_arrays = (
            "features",
            "labels",
            "training_seed",
            "episode_seed",
            "branch_step",
            "split",
            "return_delta",
            "goal_distance_improvement",
        )
        if tuple(data.files) != required_arrays:
            raise ValueError("risk dataset NPZ schema differs")
        if any(data[name].shape[0] != len(rows) for name in required_arrays):
            raise ValueError("risk dataset NPZ and CSV row counts differ")
        if data["features"].shape[1] != int(
            metadata["feature_schema"]["feature_dim"]
        ):
            raise ValueError("risk dataset feature dimension differs")
        run_max_replay = max(
            float(row["max_replay_observation_error"]) for row in refs
        )
        per_run[str(training_seed)] = {
            "samples": len(rows),
            "train_samples": sum(row["split"] == "train" for row in rows),
            "validation_samples": sum(
                row["split"] == "validation" for row in rows
            ),
            "label_counts": metadata["label_counts"],
            "considered_branch_points": sum(
                int(row["considered_branch_points"]) for row in refs
            ),
            "accepted_branch_points": sum(
                int(row["accepted_branch_points"]) for row in refs
            ),
            "max_replay_observation_error": run_max_replay,
        }
        metadata_rows.append(metadata)
        sample_rows.extend(rows)
        reference_rows.extend(refs)
        arrays.append({name: data[name].copy() for name in required_arrays})

    keys = [
        (
            int(row["training_seed"]),
            int(row["episode_seed"]),
            int(row["branch_step"]),
        )
        for row in sample_rows
    ]
    duplicate_keys = len(keys) - len(set(keys))
    finite_numeric = _finite_numeric(sample_rows + reference_rows)
    arrays_finite = all(
        np.isfinite(values).all()
        for data in arrays
        for name, values in data.items()
        if np.issubdtype(values.dtype, np.number)
    )
    max_replay_error = max(
        row["max_replay_observation_error"] for row in per_run.values()
    )
    burst_contract_checked = all(
        "candidate_intervention_steps_requested" in row
        for row in sample_rows
    )
    burst_contract_passed = bool(
        all(burst_accounting_valid(row) for row in sample_rows)
        if burst_contract_checked
        else int(expected["intervention_steps"]) == 1
    )
    train_rows = [row for row in sample_rows if row["split"] == "train"]
    validation_rows = [
        row for row in sample_rows if row["split"] == "validation"
    ]
    label_counts = {
        split: dict(Counter(row["label"] for row in rows))
        for split, rows in (
            ("train", train_rows),
            ("validation", validation_rows),
        )
    }
    quality_passed = bool(
        duplicate_keys == 0
        and finite_numeric
        and arrays_finite
        and max_replay_error <= float(expected["replay_atol"])
        and burst_contract_passed
    )
    summary = {
        "quality_passed": quality_passed,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "label_counts": label_counts,
        "train_training_seeds": sorted(set(
            int(row["training_seed"]) for row in train_rows
        )),
        "validation_training_seeds": sorted(set(
            int(row["training_seed"]) for row in validation_rows
        )),
    }
    decision = assess_risk_dataset_sufficiency(summary)
    result = {
        "run_git_sha": git_sha(ROOT),
        "training_seeds": training_seeds,
        "contract": expected,
        "quality_passed": quality_passed,
        "duplicate_sample_keys": duplicate_keys,
        "finite_csv_numeric": finite_numeric,
        "finite_npz_arrays": arrays_finite,
        "max_replay_observation_error": max_replay_error,
        "burst_contract_checked": burst_contract_checked,
        "burst_contract_passed": burst_contract_passed,
        "samples": len(sample_rows),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "label_counts": label_counts,
        "per_run": per_run,
        "continuous_targets": {},
        "sufficiency_decision": decision,
        "sealed_test_seeds_opened": False,
    }
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        result["continuous_targets"][split] = {}
        for name in ("return_delta", "goal_distance_improvement"):
            values = np.asarray([float(row[name]) for row in rows])
            result["continuous_targets"][split][name] = {
                "min": float(values.min()),
                "q10": float(np.quantile(values, 0.1)),
                "median": float(np.median(values)),
                "q90": float(np.quantile(values, 0.9)),
                "max": float(values.max()),
            }
        if burst_contract_checked:
            accepted_steps = np.asarray([
                int(row["candidate_intervention_accepted_steps"])
                for row in rows
            ], dtype=np.float64)
            accept_fraction = np.asarray([
                float(row["candidate_intervention_accept_fraction"])
                for row in rows
            ], dtype=np.float64)
            result["continuous_targets"][split]["burst_execution"] = {
                "accepted_steps_min": int(accepted_steps.min()),
                "accepted_steps_median": float(np.median(accepted_steps)),
                "accepted_steps_max": int(accepted_steps.max()),
                "accept_fraction_mean": float(np.mean(accept_fraction)),
            }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "samples_all_training_seeds.csv", sample_rows)
    _write_csv(output / "reference_episodes_all_training_seeds.csv", reference_rows)
    np.savez_compressed(
        output / "samples_all_training_seeds.npz",
        **{
            name: np.concatenate([data[name] for data in arrays], axis=0)
            for name in arrays[0]
        }
    )
    with (output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
