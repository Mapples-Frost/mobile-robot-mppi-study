#!/usr/bin/env python3
"""Summarize the preregistered L57 high-dynamic ICODE offline gate."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.learning.models import load_platform_checkpoint


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _reduction(learned, nominal):
    return 1.0 - float(learned) / max(float(nominal), 1e-12)


def summarize(config, run_dirs):
    gate_config = config["offline_gate"]
    horizons = [int(value) for value in gate_config["required_horizons"]]
    expected_seeds = sorted(int(value) for value in gate_config["required_training_seeds"])
    expected_hashes = {
        str(name): str(value).lower()
        for name, value in gate_config["expected_dataset_sha256"].items()
    }
    blocks = []
    artifact_errors = []
    for run_dir in (Path(value).resolve() for value in run_dirs):
        checkpoint = run_dir / "best.pt"
        test_path = run_dir / "evaluation_test.json"
        unseen_path = run_dir / "evaluation_unseen.json"
        if not checkpoint.exists() or not test_path.exists() or not unseen_path.exists():
            artifact_errors.append(str(run_dir))
            continue
        model, payload = load_platform_checkpoint(checkpoint, "cpu")
        test = json.loads(test_path.read_text(encoding="utf-8"))
        unseen = json.loads(unseen_path.read_text(encoding="utf-8"))
        seed = int(payload["training_config"].get("seed", -1))
        provenance = payload.get("dataset_provenance", {})
        observed_hashes = {
            name: str(record.get("sha256", "")).lower()
            for name, record in provenance.get("splits", {}).items()
        }
        if observed_hashes != expected_hashes:
            artifact_errors.append("dataset hash mismatch: %s" % run_dir)
        if model.model_type != "icode_residual":
            artifact_errors.append("model type mismatch: %s" % run_dir)
        if [float(value) for value in model.residual_output_mask] != [0.0, 0.0, 0.0, 1.0, 1.0]:
            artifact_errors.append("residual mask mismatch: %s" % run_dir)
        for split_name, metrics in (("test", test), ("unseen", unseen)):
            if metrics.get("window_selection") != "all":
                artifact_errors.append("incomplete window evaluation: %s %s" % (
                    run_dir, split_name
                ))
            if [int(value) for value in horizons if "rollout_rmse_h%d" % value not in metrics]:
                artifact_errors.append("missing horizon: %s %s" % (run_dir, split_name))
        block = {
            "run_dir": str(run_dir),
            "training_seed": seed,
            "best_validation_multistep_rmse": float(
                payload["best_validation_multistep_rmse"]
            ),
            "parameter_count": int(payload["parameter_count"]),
        }
        for split_name, metrics in (("test", test), ("unseen", unseen)):
            block["%s_active_derivative_reduction" % split_name] = _reduction(
                metrics["active_residual_derivative_rmse"],
                metrics["active_nominal_residual_derivative_rmse"],
            )
            for horizon in horizons:
                block["%s_h%d_rollout_reduction" % (split_name, horizon)] = _reduction(
                    metrics["rollout_rmse_h%d" % horizon],
                    metrics["nominal_rollout_rmse_h%d" % horizon],
                )
                block["%s_h%d_position_reduction" % (split_name, horizon)] = _reduction(
                    metrics["endpoint_h%d" % horizon]["position_rmse"],
                    metrics["nominal_endpoint_h%d" % horizon]["position_rmse"],
                )
                block["%s_h%d_windows" % (split_name, horizon)] = int(
                    metrics["rollout_windows_h%d" % horizon]
                )
        blocks.append(block)

    observed_seeds = sorted(row["training_seed"] for row in blocks)
    h36 = 36
    aggregates = {}
    for split in ("test", "unseen"):
        for metric in ("active_derivative_reduction", "h36_rollout_reduction", "h36_position_reduction"):
            field = "%s_%s" % (split, metric)
            values = [float(row[field]) for row in blocks]
            aggregates["mean_%s" % field] = float(np.mean(values)) if values else float("nan")
            aggregates["positive_blocks_%s" % field] = int(sum(value > 0.0 for value in values))

    minimum_blocks = int(gate_config["minimum_positive_model_blocks"])
    derivative_floor = float(gate_config["minimum_active_derivative_reduction"])
    checks = {
        "artifact_integrity": bool(
            not artifact_errors and len(blocks) == len(expected_seeds)
            and observed_seeds == expected_seeds
        ),
        "all_test_active_derivative_positive": bool(
            blocks and min(row["test_active_derivative_reduction"] for row in blocks)
            > derivative_floor
        ),
        "all_unseen_active_derivative_positive": bool(
            blocks and min(row["unseen_active_derivative_reduction"] for row in blocks)
            > derivative_floor
        ),
        "test_h36_positive_blocks": aggregates.get(
            "positive_blocks_test_h36_rollout_reduction", 0
        ) >= minimum_blocks,
        "unseen_h36_positive_blocks": aggregates.get(
            "positive_blocks_unseen_h36_rollout_reduction", 0
        ) >= minimum_blocks,
        "mean_test_h36_rollout_reduction": aggregates.get(
            "mean_test_h36_rollout_reduction", float("nan")
        ) >= float(gate_config["minimum_mean_test_h36_rollout_reduction"]),
        "mean_unseen_h36_rollout_reduction": aggregates.get(
            "mean_unseen_h36_rollout_reduction", float("nan")
        ) >= float(gate_config["minimum_mean_unseen_h36_rollout_reduction"]),
        "mean_test_h36_position_reduction": aggregates.get(
            "mean_test_h36_position_reduction", float("nan")
        ) >= float(gate_config["minimum_mean_test_h36_position_reduction"]),
        "mean_unseen_h36_position_reduction": aggregates.get(
            "mean_unseen_h36_position_reduction", float("nan")
        ) >= float(gate_config["minimum_mean_unseen_h36_position_reduction"]),
    }
    finite = all(
        math.isfinite(float(value))
        for row in blocks for key, value in row.items()
        if key not in ("run_dir",)
    ) and all(math.isfinite(float(value)) for value in aggregates.values())
    checks["finite_metrics"] = finite
    return {
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "aggregates": aggregates,
        "model_blocks": blocks,
        "artifact_errors": artifact_errors,
        "thresholds": dict(gate_config),
        "interpretation_guard": (
            "Passing is an offline prediction gate, not closed-loop MPPI evidence."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    result = summarize(config, [_resolved(value) for value in args.run_dirs])
    output = _resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
