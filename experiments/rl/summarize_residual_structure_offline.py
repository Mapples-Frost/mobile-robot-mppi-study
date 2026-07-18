#!/usr/bin/env python3
"""Audit the preregistered L60 parameter-matched MLP--ICODE ablation."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.learning.models import load_platform_checkpoint


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _reduction(learned, nominal):
    return 1.0 - float(learned) / max(float(nominal), 1e-12)


def _relative_advantage(lower_error, higher_error):
    """Positive when the first argument has lower error than the second."""

    return (float(higher_error) - float(lower_error)) / max(
        float(higher_error), 1e-12
    )


def _load_run(run_dir, expected_type, expected_seed, expected_hashes, horizons):
    run_dir = _resolved(run_dir)
    checkpoint = run_dir / "best.pt"
    model, payload = load_platform_checkpoint(checkpoint, "cpu")
    if model.model_type != expected_type:
        raise ValueError("model type mismatch at %s" % run_dir)
    if int(payload["training_config"].get("seed", -1)) != int(expected_seed):
        raise ValueError("training seed mismatch at %s" % run_dir)
    observed_hashes = {
        name: str(record.get("sha256", "")).lower()
        for name, record in payload.get("dataset_provenance", {}).get(
            "splits", {}
        ).items()
    }
    if observed_hashes != expected_hashes:
        raise ValueError("dataset hash mismatch at %s" % run_dir)
    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "parameter_count": int(model.parameter_count()),
        "best_validation_multistep_rmse": float(
            payload["best_validation_multistep_rmse"]
        ),
    }
    for split in ("test", "unseen"):
        path = run_dir / ("evaluation_%s.json" % split)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("model_type") != expected_type:
            raise ValueError("evaluation model type mismatch at %s" % path)
        if metrics.get("window_selection") != "all":
            raise ValueError("evaluation does not cover all windows at %s" % path)
        result["%s_active_derivative_reduction" % split] = _reduction(
            metrics["active_residual_derivative_rmse"],
            metrics["active_nominal_residual_derivative_rmse"],
        )
        for horizon in horizons:
            key = "h%d" % horizon
            result["%s_%s_rollout_rmse" % (split, key)] = float(
                metrics["rollout_rmse_h%d" % horizon]
            )
            result["%s_%s_nominal_rollout_rmse" % (split, key)] = float(
                metrics["nominal_rollout_rmse_h%d" % horizon]
            )
            result["%s_%s_rollout_reduction" % (split, key)] = _reduction(
                metrics["rollout_rmse_h%d" % horizon],
                metrics["nominal_rollout_rmse_h%d" % horizon],
            )
            result["%s_%s_position_rmse" % (split, key)] = float(
                metrics["endpoint_h%d" % horizon]["position_rmse"]
            )
            result["%s_%s_position_reduction" % (split, key)] = _reduction(
                metrics["endpoint_h%d" % horizon]["position_rmse"],
                metrics["nominal_endpoint_h%d" % horizon]["position_rmse"],
            )
            result["%s_%s_windows" % (split, key)] = int(
                metrics["rollout_windows_h%d" % horizon]
            )
    return result


def summarize(config):
    design = config["structure_ablation"]
    horizons = [int(value) for value in design["required_horizons"]]
    expected_hashes = {
        str(name): str(value).lower()
        for name, value in config["offline_gate"][
            "expected_dataset_sha256"
        ].items()
    }
    blocks = []
    artifact_errors = []
    for raw in design["model_blocks"]:
        seed = int(raw["training_seed"])
        try:
            icode = _load_run(
                raw["icode_run_dir"], "icode_residual", seed,
                expected_hashes, horizons,
            )
            mlp = _load_run(
                raw["mlp_run_dir"], "mlp_residual", seed,
                expected_hashes, horizons,
            )
            parameter_difference = abs(
                mlp["parameter_count"] - icode["parameter_count"]
            ) / float(icode["parameter_count"])
            block = {
                "training_seed": seed,
                "icode": icode,
                "mlp": mlp,
                "parameter_count_relative_difference": parameter_difference,
            }
            for split in ("test", "unseen"):
                for horizon in horizons:
                    key = "%s_h%d" % (split, horizon)
                    block["%s_icode_improvement_over_mlp_rollout" % key] = (
                        _relative_advantage(
                            icode["%s_rollout_rmse" % key],
                            mlp["%s_rollout_rmse" % key],
                        )
                    )
                    block["%s_icode_improvement_over_mlp_position" % key] = (
                        _relative_advantage(
                            icode["%s_position_rmse" % key],
                            mlp["%s_position_rmse" % key],
                        )
                    )
            blocks.append(block)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            artifact_errors.append("seed %d: %s" % (seed, error))

    h36 = 36
    aggregates = {}
    for model_name in ("icode", "mlp"):
        for split in ("test", "unseen"):
            for metric in (
                "active_derivative_reduction",
                "h36_rollout_reduction",
                "h36_position_reduction",
            ):
                values = [
                    float(row[model_name]["%s_%s" % (split, metric)])
                    for row in blocks
                ]
                key = "%s_%s_%s" % (model_name, split, metric)
                aggregates["mean_%s" % key] = (
                    float(np.mean(values)) if values else float("nan")
                )
                aggregates["positive_blocks_%s" % key] = int(
                    sum(value > 0.0 for value in values)
                )
    for split in ("test", "unseen"):
        field = "%s_h36_icode_improvement_over_mlp_rollout" % split
        values = [float(row[field]) for row in blocks]
        aggregates["mean_%s" % field] = (
            float(np.mean(values)) if values else float("nan")
        )
        aggregates["positive_blocks_%s" % field] = int(
            sum(value > 0.0 for value in values)
        )

    expected_seeds = sorted(
        int(value) for value in design["required_training_seeds"]
    )
    integrity = bool(
        not artifact_errors
        and sorted(row["training_seed"] for row in blocks) == expected_seeds
        and all(
            row["parameter_count_relative_difference"]
            <= float(design["maximum_parameter_count_relative_difference"])
            for row in blocks
        )
    )
    eligibility = design["mlp_eligibility"]
    minimum_blocks = int(eligibility["minimum_positive_model_blocks"])
    mlp_checks = {
        "artifact_and_capacity_integrity": integrity,
        "all_test_active_derivative_positive": bool(
            blocks and min(
                row["mlp"]["test_active_derivative_reduction"] for row in blocks
            ) > float(eligibility["minimum_active_derivative_reduction"])
        ),
        "all_unseen_active_derivative_positive": bool(
            blocks and min(
                row["mlp"]["unseen_active_derivative_reduction"] for row in blocks
            ) > float(eligibility["minimum_active_derivative_reduction"])
        ),
        "test_h36_positive_blocks": aggregates.get(
            "positive_blocks_mlp_test_h36_rollout_reduction", 0
        ) >= minimum_blocks,
        "unseen_h36_positive_blocks": aggregates.get(
            "positive_blocks_mlp_unseen_h36_rollout_reduction", 0
        ) >= minimum_blocks,
        "mean_test_h36_rollout_reduction": aggregates.get(
            "mean_mlp_test_h36_rollout_reduction", float("nan")
        ) >= float(eligibility["minimum_mean_test_h36_rollout_reduction"]),
        "mean_unseen_h36_rollout_reduction": aggregates.get(
            "mean_mlp_unseen_h36_rollout_reduction", float("nan")
        ) >= float(eligibility["minimum_mean_unseen_h36_rollout_reduction"]),
        "mean_test_h36_position_reduction": aggregates.get(
            "mean_mlp_test_h36_position_reduction", float("nan")
        ) >= float(eligibility["minimum_mean_test_h36_position_reduction"]),
        "mean_unseen_h36_position_reduction": aggregates.get(
            "mean_mlp_unseen_h36_position_reduction", float("nan")
        ) >= float(eligibility["minimum_mean_unseen_h36_position_reduction"]),
    }
    hypothesis = design["structural_hypothesis"]
    structural_checks = {
        "positive_test_h36_blocks": aggregates.get(
            "positive_blocks_test_h36_icode_improvement_over_mlp_rollout", 0
        ) >= int(hypothesis["minimum_positive_test_h36_model_blocks"]),
        "positive_unseen_h36_blocks": aggregates.get(
            "positive_blocks_unseen_h36_icode_improvement_over_mlp_rollout", 0
        ) >= int(hypothesis["minimum_positive_unseen_h36_model_blocks"]),
        "mean_test_h36_advantage": aggregates.get(
            "mean_test_h36_icode_improvement_over_mlp_rollout", float("nan")
        ) >= float(
            hypothesis["minimum_mean_test_h36_icode_improvement_over_mlp"]
        ),
        "mean_unseen_h36_advantage": aggregates.get(
            "mean_unseen_h36_icode_improvement_over_mlp_rollout", float("nan")
        ) >= float(
            hypothesis["minimum_mean_unseen_h36_icode_improvement_over_mlp"]
        ),
    }
    finite = all(
        math.isfinite(float(value)) for value in aggregates.values()
    ) and all(
        math.isfinite(float(row["parameter_count_relative_difference"]))
        for row in blocks
    )
    mlp_checks["finite_metrics"] = finite
    structural_checks["finite_metrics"] = finite
    return {
        "artifact_integrity": integrity,
        "mlp_closed_loop_eligible": bool(all(mlp_checks.values())),
        "structural_hypothesis_supported_offline": bool(
            integrity and all(structural_checks.values())
        ),
        "mlp_eligibility_checks": mlp_checks,
        "structural_hypothesis_checks": structural_checks,
        "aggregates": aggregates,
        "model_blocks": blocks,
        "artifact_errors": artifact_errors,
        "thresholds": dict(design),
        "interpretation_guard": (
            "Offline structure evidence grants closed-loop eligibility only; "
            "it is not an MPPI control result."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(load_yaml(_resolved(args.config)))
    output = _resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["mlp_closed_loop_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

