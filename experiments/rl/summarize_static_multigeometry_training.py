#!/usr/bin/env python3
"""Audit preregistered multi-geometry SAC training replicas."""

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


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _mean_finite(rows, field):
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in (None, "")
        and math.isfinite(float(row[field]))
    ]
    return None if not values else sum(values) / len(values)


def _replay_batch_diagnostics(path, tail_rows=5000):
    if not Path(path).exists():
        return None
    rows = _read_csv(path)
    if not rows:
        return None
    selected = rows[-min(len(rows), int(tail_rows)):]
    scene_fields = sorted(
        name for name in selected[0]
        if name.startswith("batch_scene_fraction_")
    )
    prefix = "batch_scene_fraction_"
    scene_means = {
        name[len(prefix):]: _mean_finite(selected, name)
        for name in scene_fields
    }
    uniform = 1.0 / len(scene_means) if scene_means else None
    return {
        "tail_update_rows": len(selected),
        "scene_fraction_means": scene_means,
        "maximum_scene_mean_deviation_from_uniform": (
            max(abs(value - uniform) for value in scene_means.values())
            if scene_means and all(value is not None for value in scene_means.values())
            else None
        ),
        "success_fraction_mean": _mean_finite(
            selected, "batch_success_fraction"
        ),
        "labeled_fraction_mean": _mean_finite(
            selected, "batch_labeled_fraction"
        ),
    }


def summarize(
    config_paths,
    run_dirs,
    minimum_nonzero_blocks=2,
    study_label="L70",
    interpretation_guard=None,
):
    if len(config_paths) != len(run_dirs) or not config_paths:
        raise ValueError(
            "multi-geometry audit requires one nonempty config/run pair per block"
        )
    blocks = []
    errors = []
    for index, (config_path, run_dir) in enumerate(zip(config_paths, run_dirs)):
        config_path = _resolved(config_path)
        run_dir = _resolved(run_dir)
        try:
            config = load_yaml(config_path)
            expected_steps = int(config["rl"]["training"]["total_steps"])
            expected_replay_sampling = str(
                config["rl"]["training"]["replay_sampling"]
            )
            summary = _read_json(run_dir / "training_summary.json")
            rows = _read_csv(run_dir / "validation_episodes.csv")
            selected_step = int(
                summary["checkpoint_selection"]["best_global_step"]
            )
            expected_validation_steps = set(range(
                0,
                expected_steps + 1,
                int(config["rl"]["training"]["evaluation_interval"]),
            ))
            observed_validation_steps = {
                int(row["global_step"]) for row in rows
            }
            scene_names = {
                str(row["scene"]) for row in rows
                if int(row["global_step"]) == 0
            }
            evaluation_episodes = int(
                config["rl"]["training"]["evaluation_episodes"]
            )
            expected_rows = (
                len(expected_validation_steps)
                * len(scene_names)
                * evaluation_episodes
            )
            block_errors = []
            if int(summary["global_step"]) != expected_steps:
                block_errors.append("training did not reach total_steps")
            if int(summary.get("interrupted_episodes", -1)) != 0:
                block_errors.append("interrupted episodes are nonzero")
            if observed_validation_steps != expected_validation_steps:
                block_errors.append("validation checkpoint schedule mismatch")
            if len(rows) != expected_rows:
                block_errors.append("validation row count mismatch")
            if str(summary.get("replay_sampling")) != expected_replay_sampling:
                block_errors.append("replay strategy does not match config")
            replay_counts = dict(summary.get("replay_group_counts", {}))
            if len(replay_counts) != len(scene_names) or any(
                int(value) <= 0 for value in replay_counts.values()
            ):
                block_errors.append("not every scene populated replay")
            selected_rows = [
                row for row in rows
                if int(row["global_step"]) == selected_step
            ]
            reference_rows = [
                row for row in rows
                if int(row["global_step"]) == 0
            ]
            if len(selected_rows) != len(scene_names) * evaluation_episodes:
                block_errors.append("selected checkpoint validation rows missing")
            selection_row = selected_rows[0] if selected_rows else {}

            def _optional_int(name):
                value = selection_row.get(name)
                return None if value in (None, "") else int(value)

            def _optional_float(name):
                value = selection_row.get(name)
                return None if value in (None, "") else float(value)

            block = {
                "block": index,
                "training_seed": int(config["rl"]["training"]["seed"]),
                "run_dir": str(run_dir),
                "expected_steps": expected_steps,
                "completed_steps": int(summary["global_step"]),
                "selected_global_step": selected_step,
                "selected_nonzero": selected_step > 0,
                "reference_successes": sum(
                    row["success"].strip().lower() == "true"
                    for row in reference_rows
                ),
                "reference_collisions": sum(
                    row["collision"].strip().lower() == "true"
                    for row in reference_rows
                ),
                "reference_mean_goal_distance_m": (
                    sum(float(row["goal_distance"]) for row in reference_rows)
                    / len(reference_rows)
                    if reference_rows else float("nan")
                ),
                "selected_successes": sum(
                    row["success"].strip().lower() == "true"
                    for row in selected_rows
                ),
                "selected_collisions": sum(
                    row["collision"].strip().lower() == "true"
                    for row in selected_rows
                ),
                "selected_mean_goal_distance_m": (
                    sum(float(row["goal_distance"]) for row in selected_rows)
                    / len(selected_rows)
                    if selected_rows else float("nan")
                ),
                "paired_success_gains": _optional_int(
                    "selection_success_gains"
                ),
                "paired_success_losses": _optional_int(
                    "selection_success_losses"
                ),
                "paired_collision_regressions": _optional_int(
                    "selection_collision_regressions"
                ),
                "paired_mean_goal_distance_improvement_m": _optional_float(
                    "selection_mean_goal_distance_improvement"
                ),
                "replay_sampling": str(summary.get("replay_sampling")),
                "replay_group_counts": replay_counts,
                "replay_outcome_counts": dict(
                    summary.get("replay_outcome_counts", {})
                ),
                "replay_batch_diagnostics": _replay_batch_diagnostics(
                    run_dir / "updates.csv"
                ),
                "artifact_integrity": not block_errors,
                "artifact_errors": block_errors,
            }
            blocks.append(block)
            errors.extend("block %d: %s" % (index, value) for value in block_errors)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append("block %d: %s" % (index, error))
    nonzero = sum(bool(row["selected_nonzero"]) for row in blocks)
    checks = {
        "artifact_integrity": not errors and len(blocks) == len(run_dirs),
        "all_training_runs_completed": bool(blocks) and all(
            row["completed_steps"] == row["expected_steps"] for row in blocks
        ),
        "configured_replay_populated_all_scenes": bool(blocks) and all(
            len(row["replay_group_counts"]) >= 2
            and all(int(value) > 0 for value in row["replay_group_counts"].values())
            for row in blocks
        ),
        "minimum_nonzero_selected_blocks": nonzero >= int(minimum_nonzero_blocks),
    }
    return {
        "study_label": str(study_label),
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "artifact_errors": errors,
        "minimum_nonzero_selected_blocks": int(minimum_nonzero_blocks),
        "nonzero_selected_blocks": nonzero,
        "blocks": blocks,
        "interpretation_guard": interpretation_guard or (
            "%s uses the preregistered raw-policy selection gate; the gate is "
            "reported as written and is not retrospectively relaxed."
            % str(study_label)
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-nonzero-blocks", type=int, default=2)
    parser.add_argument("--study-label", default="L70")
    parser.add_argument("--artifact-prefix", default="l70_training_audit")
    parser.add_argument("--interpretation-guard")
    args = parser.parse_args(argv)
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = summarize(
        args.config,
        args.run_dir,
        minimum_nonzero_blocks=args.minimum_nonzero_blocks,
        study_label=args.study_label,
        interpretation_guard=args.interpretation_guard,
    )
    (output / ("%s.json" % args.artifact_prefix)).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / ("%s_blocks.csv" % args.artifact_prefix),
        result["blocks"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
