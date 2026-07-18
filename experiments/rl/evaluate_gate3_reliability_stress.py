#!/usr/bin/env python3
"""Evaluate frozen L193 reliability on the independent graded L194 set."""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha

from experiments.rl.calibrate_gate3_reliability import (
    _load_models,
    _load_split,
    _rank_correlation,
    _resolve,
    apply_candidate,
    episode_level_summary,
    evaluate_episode_gate,
    extract_window_signals,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write empty stress table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path):
    config_path = _resolve(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    calibration_config_path = _resolve(config["calibration_config"])
    with calibration_config_path.open("r", encoding="utf-8") as handle:
        calibration_config = yaml.safe_load(handle)
    calibration_summary_path = _resolve(config["calibration_summary"])
    calibration_summary = json.loads(
        calibration_summary_path.read_text(encoding="utf-8")
    )
    candidate = calibration_summary["selected_candidate"]
    models = _load_models(calibration_config)
    dataset, dataset_path = _load_split(
        _resolve(config["dataset_dir"]), str(config["split"])
    )
    signals = extract_window_signals(
        dataset, models, calibration_config, str(config["split"])
    )
    evaluated = apply_candidate(
        signals, candidate, calibration_config
    )
    episodes = episode_level_summary(evaluated)
    grid_gate = dict(calibration_config["calibration_grid"])
    grid_gate.update({
        "minimum_episodes_per_occupied_bin": int(
            config["gate"]["minimum_episodes_per_occupied_bin"]
        ),
        "minimum_occupied_bins": int(
            config["gate"]["minimum_occupied_bins"]
        ),
    })
    discrete = evaluate_episode_gate(episodes, grid_gate)
    authority = [row["authority"] for row in episodes]
    error = [row["rollout_error"] for row in episodes]
    rank = _rank_correlation(authority, error)
    low = [row for row in episodes if row["level"] == "low"]
    nonlow = [row for row in episodes if row["level"] != "low"]
    by_domain = defaultdict(list)
    for row in episodes:
        by_domain[row["physics_domain"]].append(row)
    domain_summary = {}
    for name, rows in sorted(by_domain.items()):
        domain_summary[name] = {
            "episodes": len(rows),
            "mean_authority": float(np.mean([
                row["authority"] for row in rows
            ])),
            "mean_rollout_error": float(np.mean([
                row["rollout_error"] for row in rows
            ])),
        }
    nominal = domain_summary["nominal_seen"]
    combined = domain_summary["combined_unseen"]
    gate_config = config["gate"]
    conditions = {
        "rank_association": rank <= float(
            gate_config[
                "maximum_authority_error_rank_correlation"
            ]
        ),
        "discrete_bin_support": bool(discrete["passed"]),
        "low_episode_support": len(low) >= int(
            gate_config["minimum_low_confidence_episodes"]
        ),
        "nonlow_episode_support": len(nonlow) >= int(
            gate_config["minimum_nonlow_confidence_episodes"]
        ),
        "low_error_above_nonlow": (
            bool(low)
            and bool(nonlow)
            and float(np.mean([
                row["rollout_error"] for row in low
            ]))
            > float(np.mean([
                row["rollout_error"] for row in nonlow
            ]))
        ),
        "combined_authority_below_nominal": (
            combined["mean_authority"] < nominal["mean_authority"]
        ),
        "combined_error_above_nominal": (
            combined["mean_rollout_error"]
            > nominal["mean_rollout_error"]
        ),
    }
    gate_passed = bool(all(conditions.values()))
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "gate": "Gate 3A2 graded independent reliability stress test",
        "gate_passed": gate_passed,
        "conditions": conditions,
        "git_sha": git_sha(ROOT),
        "config_path": str(config_path),
        "config": config,
        "calibration_config": str(calibration_config_path),
        "calibration_config_sha256": _sha256(calibration_config_path),
        "calibration_summary": str(calibration_summary_path),
        "calibration_summary_sha256": _sha256(
            calibration_summary_path
        ),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "episode_count": len(episodes),
        "authority_error_rank_correlation": rank,
        "low_mean_rollout_error": (
            float(np.mean([row["rollout_error"] for row in low]))
            if low else None
        ),
        "nonlow_mean_rollout_error": (
            float(np.mean([
                row["rollout_error"] for row in nonlow
            ]))
            if nonlow else None
        ),
        "discrete_evaluation": discrete,
        "domain_summary": domain_summary,
        "independent_unit": "episode",
    }
    (output_dir / "stress_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "stress_windows.csv", evaluated)
    _write_csv(output_dir / "stress_episodes.csv", episodes)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = evaluate(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

