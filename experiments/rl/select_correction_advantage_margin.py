#!/usr/bin/env python3
"""Audit multi-training-seed margin grids and freeze one global decision."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.run_correction_advantage_diagnostic import (
    _condition_summary,
    _write_csv,
)
from experiments.rl.summarize_correction_advantage_diagnostic import (
    _run_quality,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.calibration import (
    AdvantageMarginSelectionConfig,
    select_advantage_margin,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_dirs = [Path(value).resolve() for value in args.run_dir]
    if len(run_dirs) < 2:
        raise ValueError("margin selection requires multiple training seeds")

    config = AdvantageMarginSelectionConfig()
    config.validate()
    expected_margins = [
        float(value) for value in config.candidate_margins
    ]
    expected_conditions = ["bc"]
    condition_margin = {}
    all_paired = []
    run_summaries = []
    quality = {}
    training_seeds = []
    expected_episode_seeds = None
    for run_dir in run_dirs:
        with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        margins = [float(value) for value in summary["margins"]]
        if margins != expected_margins:
            raise ValueError("run margin grid differs from preregistration")
        mapping = {
            str(name): float(value)
            for name, value in summary["condition_margin"].items()
        }
        if sorted(mapping.values()) != expected_margins:
            raise ValueError("run condition-to-margin mapping is incomplete")
        if not condition_margin:
            condition_margin = mapping
            expected_conditions.extend(sorted(mapping))
        elif mapping != condition_margin:
            raise ValueError("runs use different condition-to-margin mappings")
        training_seed = int(summary["training_seed"])
        if training_seed in training_seeds:
            raise ValueError("training seeds must be unique")
        training_seeds.append(training_seed)
        episode_seeds = [int(value) for value in summary["calibration_seeds"]]
        if expected_episode_seeds is None:
            expected_episode_seeds = episode_seeds
        elif episode_seeds != expected_episode_seeds:
            raise ValueError("runs use different calibration seeds")
        run_quality, paired = _run_quality(
            run_dir, expected_conditions, episode_seeds
        )
        quality[str(training_seed)] = run_quality
        all_paired.extend(paired)
        run_summaries.append(summary)

    pooled_by_condition = _condition_summary(all_paired)
    candidates = {}
    for condition, margin in condition_margin.items():
        candidates[margin] = {
            "pooled": pooled_by_condition[condition],
            "per_training_seed": {
                str(seed): summary["conditions"][condition]
                for seed, summary in zip(training_seeds, run_summaries)
            },
        }
    selection = select_advantage_margin(candidates, config)
    quality_passed = all(
        row["episode_duplicate_keys"] == 0
        and row["paired_duplicate_keys"] == 0
        and row["step_duplicate_keys"] == 0
        and row["finite_numeric"]
        and row["missing_critical_step_rows"] == 0
        and row["conditions_match"]
        and row["seeds_match"]
        and row["bc_raw_correction_max"] == 0.0
        for row in quality.values()
    )
    if not quality_passed:
        selection["selected_mode"] = "bc_fallback"
        selection["selected_margin"] = None
        selection["selection_reason"] = "data_quality_gate_failed"
    result = {
        "phase": "calibration",
        "run_git_sha": git_sha(ROOT),
        "training_seeds": training_seeds,
        "calibration_seeds": expected_episode_seeds,
        "quality_passed": quality_passed,
        "quality": quality,
        "condition_margin": condition_margin,
        "pooled_by_condition": pooled_by_condition,
        "per_training_seed": {
            condition: {
                str(seed): summary["conditions"][condition]
                for seed, summary in zip(training_seeds, run_summaries)
            }
            for condition in sorted(condition_margin)
        },
        "selection": selection,
        "selection_seeds_opened": False,
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "paired_calibration_rows.csv", all_paired)
    with (output / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
