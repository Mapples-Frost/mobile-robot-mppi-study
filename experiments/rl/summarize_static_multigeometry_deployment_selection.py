#!/usr/bin/env python3
"""Select training checkpoints using fresh deployment-aligned episodes."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml  # noqa: E402


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(row, name, default=float("nan")):
    value = row.get(name)
    return default if value in (None, "") else float(value)


def _row_key(row):
    return str(row["scene"]), int(row["episode_seed"])


def _candidate_effect(base_rows, candidate_rows, thresholds, block):
    baseline = {_row_key(row): row for row in base_rows}
    candidate = {_row_key(row): row for row in candidate_rows}
    if set(baseline) != set(candidate) or len(base_rows) != len(baseline):
        raise ValueError("candidate/base paired episode matrix mismatch")
    pairs = [(baseline[key], candidate[key]) for key in sorted(baseline)]
    success_losses = sum(
        _bool(first["success"]) and not _bool(second["success"])
        for first, second in pairs
    )
    success_gains = sum(
        not _bool(first["success"]) and _bool(second["success"])
        for first, second in pairs
    )
    collision_regressions = sum(
        not _bool(first["collision"]) and _bool(second["collision"])
        for first, second in pairs
    )
    distance_improvements = [
        _float(first, "final_goal_distance")
        - _float(second, "final_goal_distance")
        for first, second in pairs
    ]
    return_available = all(
        first.get("return") not in (None, "")
        and second.get("return") not in (None, "")
        for first, second in pairs
    )
    return_improvements = (
        [
            _float(second, "return") - _float(first, "return")
            for first, second in pairs
        ]
        if return_available else []
    )
    gate_alpha = sum(
        _float(second, "rl_correction_advantage_gate_alpha_mean", 0.0)
        for _, second in pairs
    ) / len(pairs)
    mean_distance = sum(distance_improvements) / len(distance_improvements)
    mean_return = (
        sum(return_improvements) / len(return_improvements)
        if return_improvements else None
    )
    noninferior = (
        success_losses <= int(thresholds["maximum_success_losses"])
        and collision_regressions <= int(
            thresholds["maximum_collision_regressions"]
        )
        and mean_distance >= -float(
            thresholds["maximum_mean_goal_distance_increase_m"]
        )
    )
    meaningful = (
        success_gains >= int(thresholds["minimum_success_gains"])
        or mean_distance >= float(
            thresholds["minimum_mean_goal_distance_improvement_m"]
        )
    )
    nondegenerate = gate_alpha >= float(
        thresholds["minimum_selected_correction_gate_alpha_mean"]
    )
    step = int(candidate_rows[0]["candidate_step"])
    return {
        "model_block": int(block),
        "candidate_id": str(candidate_rows[0]["candidate_id"]),
        "candidate_step": step,
        "checkpoint": str(candidate_rows[0]["rl_checkpoint"]),
        "pairs": len(pairs),
        "success_losses": int(success_losses),
        "success_gains": int(success_gains),
        "net_success_gain": int(success_gains - success_losses),
        "collision_regressions": int(collision_regressions),
        "mean_final_distance_improvement_m": float(mean_distance),
        "mean_return_improvement": (
            None if mean_return is None else float(mean_return)
        ),
        "return_tiebreak_available": bool(return_available),
        "mean_correction_gate_alpha": float(gate_alpha),
        "noninferior": bool(noninferior),
        "meaningful_improvement": bool(meaningful),
        "correction_non_degenerate": bool(nondegenerate),
        "qualified": bool(noninferior and meaningful and nondegenerate),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    study_label = str(design.get("study_label", "L71"))
    thresholds = design["deployment_checkpoint_selection"]
    expected_blocks = len(design["model_blocks"])
    expected_scenes = {
        str(load_yaml(_resolved(row["path"]))["scene"]["name"])
        for row in design["scenes"]
    }
    expected_seeds = {int(value) for value in design["development_episode_seeds"]}
    expected_candidates = {"base"} | {
        "step_%09d" % int(value) for value in thresholds["candidate_steps"]
    }
    confirmation_mode = bool(design.get("confirmation_mode", False))
    configured_sealed = {
        int(value)
        for value in design.get("sealed_confirmation_episode_seeds", ())
    }
    errors = []
    effects = []
    selections = []
    maximum_base_gate_alpha = 0.0
    for block in range(expected_blocks):
        block_dir = Path(input_dir) / ("block_%d" % block)
        try:
            rows = _read_csv(block_dir / "episodes.csv")
            metadata = json.loads(
                (block_dir / "metadata.json").read_text(encoding="utf-8")
            )
            if int(metadata["model_block"]) != block:
                errors.append("block %d metadata index mismatch" % block)
            observed_sealed = {
                int(value)
                for value in metadata.get("sealed_confirmation_seeds_used", ())
            }
            if confirmation_mode:
                if observed_sealed != expected_seeds:
                    errors.append(
                        "block %d confirmation sealed-seed set mismatch" % block
                    )
                if expected_seeds != configured_sealed:
                    errors.append(
                        "block %d confirmation config does not bind the "
                        "expected seeds as sealed" % block
                    )
            elif observed_sealed:
                errors.append("block %d used sealed seeds" % block)
            if metadata.get("protected_previous_seeds_used"):
                errors.append("block %d reused protected seeds" % block)
            observed = {
                (str(row["scene"]), int(row["episode_seed"]), str(row["candidate_id"]))
                for row in rows
            }
            expected = {
                (scene, seed, candidate)
                for scene in expected_scenes
                for seed in expected_seeds
                for candidate in expected_candidates
            }
            if observed != expected or len(rows) != len(observed):
                errors.append("block %d episode matrix mismatch" % block)
                continue
            finite_fields = (
                "final_goal_distance", "planner_compute_ms_mean",
                "rl_correction_advantage_gate_alpha_mean",
            )
            if any(
                not math.isfinite(_float(row, field))
                for row in rows for field in finite_fields
            ):
                errors.append("block %d contains non-finite metrics" % block)
                continue
            base_rows = [row for row in rows if row["candidate_id"] == "base"]
            maximum_base_gate_alpha = max(
                maximum_base_gate_alpha,
                max(_float(row, "rl_correction_advantage_gate_alpha_mean", 0.0)
                    for row in base_rows),
            )
            block_effects = []
            for candidate_id in sorted(expected_candidates - {"base"}):
                selected = [
                    row for row in rows if row["candidate_id"] == candidate_id
                ]
                effect = _candidate_effect(base_rows, selected, thresholds, block)
                block_effects.append(effect)
                effects.append(effect)
            qualified = [row for row in block_effects if row["qualified"]]
            if qualified:
                selected = max(qualified, key=lambda row: (
                    int(row["net_success_gain"]),
                    float(row["mean_final_distance_improvement_m"]),
                    (
                        float(row["mean_return_improvement"])
                        if row["mean_return_improvement"] is not None
                        else float("-inf")
                    ),
                    -int(row["candidate_step"]),
                ))
                selection = dict(selected)
                selection["selected_nonzero"] = True
            else:
                selection = {
                    "model_block": block,
                    "candidate_id": "base",
                    "candidate_step": 0,
                    "checkpoint": str(base_rows[0]["rl_checkpoint"]),
                    "selected_nonzero": False,
                    "selection_reason": "no_deployment_candidate_qualified",
                }
            selections.append(selection)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append("block %d: %s" % (block, error))
    nonzero = sum(bool(row.get("selected_nonzero", False)) for row in selections)
    checks = {
        "artifact_integrity": not errors and len(selections) == expected_blocks,
        "frozen_base_exact": maximum_base_gate_alpha == 0.0,
        "minimum_nonzero_selected_blocks": nonzero >= int(
            thresholds["minimum_nonzero_selected_blocks"]
        ),
    }
    return {
        "study_label": study_label,
        "confirmation_mode": confirmation_mode,
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "artifact_errors": errors,
        "maximum_frozen_base_gate_alpha": maximum_base_gate_alpha,
        "nonzero_selected_blocks": nonzero,
        "candidate_effects": effects,
        "selections": selections,
        "thresholds": dict(thresholds),
        "interpretation_guard": str(design.get(
            "interpretation_guard",
            (
                "%s is a fresh deployment-aligned model-selection set; "
                "it is development evidence, not final-test evidence."
                % study_label
            ),
        )),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    result = summarize(config, input_dir)
    prefix = str(
        config["rl"]["cross_layer_factorial"].get(
            "artifact_prefix", "l71"
        )
    )
    (input_dir / ("%s_deployment_selection_summary.json" % prefix)).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        input_dir / ("%s_candidate_effects.csv" % prefix),
        result["candidate_effects"],
    )
    _write_csv(
        input_dir / ("%s_selected_models.csv" % prefix),
        result["selections"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
