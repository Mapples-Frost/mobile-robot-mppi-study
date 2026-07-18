#!/usr/bin/env python3
"""Extend the paired path-tracking audit with L58 high-dynamic gates."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (
    CONDITIONS, _base_key, _hierarchical_bootstrap, _read_csv, summarize,
)
from mobile_robot_mppi.core.config import load_yaml


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def summarize_l58(config, input_dir):
    base = summarize(config, input_dir)
    design = config["rl"]["cross_layer_factorial"]
    effects = _read_csv(Path(input_dir) / "path_tracking_paired_effects.csv")
    unseen_scenes = set()
    for item in design["scenes"]:
        if "unseen" in str(item["role"]).lower():
            scene_config = load_yaml(_resolved(item["path"]))
            unseen_scenes.add(str(scene_config["scene"]["name"]))
    if not unseen_scenes:
        raise ValueError("L58 requires at least one scene role containing 'unseen'")
    unseen = [row for row in effects if str(row["scene"]) in unseen_scenes]
    per_block_unseen = []
    for block in range(len(design["model_blocks"])):
        rows = [row for row in unseen if int(row["model_block"]) == block]
        per_block_unseen.append({
            "model_block": block,
            "pairs": len(rows),
            "mean_cross_track_improvement_m": float(np.mean([
                float(row["cross_track_improvement_m"]) for row in rows
            ])),
            "mean_relative_cross_track_reduction": float(np.mean([
                float(row["relative_cross_track_reduction"]) for row in rows
            ])),
        })
    unseen_bootstrap = _hierarchical_bootstrap(
        unseen, "cross_track_improvement_m",
        int(design["bootstrap_seed"]) + 1,
        int(design["bootstrap_replicates"]),
    )
    per_scene = []
    for scene in sorted({str(row["scene"]) for row in effects}):
        rows = [row for row in effects if str(row["scene"]) == scene]
        per_scene.append({
            "scene": scene,
            "pairs": len(rows),
            "mean_cross_track_improvement_m": float(np.mean([
                float(row["cross_track_improvement_m"]) for row in rows
            ])),
            "mean_relative_cross_track_reduction": float(np.mean([
                float(row["relative_cross_track_reduction"]) for row in rows
            ])),
        })

    episode_rows = []
    for block in range(len(design["model_blocks"])):
        episode_rows.extend(_read_csv(
            Path(input_dir) / ("block_%d" % block) / "episodes.csv"
        ))
    lookup = {
        _base_key(row) + (str(row["condition"]),): row for row in episode_rows
    }
    jerk_changes = []
    for key, nominal in lookup.items():
        if key[-1] != CONDITIONS[0]:
            continue
        icode = lookup[key[:-1] + (CONDITIONS[1],)]
        nominal_jerk = float(nominal["applied_control_jerk"])
        jerk_changes.append(
            (float(icode["applied_control_jerk"]) - nominal_jerk)
            / max(nominal_jerk, 1e-12)
        )
    mean_jerk_increase = float(np.mean(jerk_changes))

    gate_cfg = design["eligibility_gate"]
    unseen_relative = float(np.mean([
        float(row["relative_cross_track_reduction"]) for row in unseen
    ]))
    additional = {
        "positive_unseen_model_blocks": int(sum(
            row["mean_cross_track_improvement_m"] > 0.0
            for row in per_block_unseen
        )),
        "unseen_relative_cross_track_rmse_reduction": unseen_relative,
        "unseen_cross_track_improvement_ci95_lower_m": unseen_bootstrap["ci95_lower"],
        "positive_scenes": int(sum(
            row["mean_cross_track_improvement_m"] > 0.0 for row in per_scene
        )),
        "relative_applied_jerk_increase": mean_jerk_increase,
    }
    additional_passed = bool(
        additional["positive_unseen_model_blocks"]
        >= int(gate_cfg["minimum_positive_unseen_model_blocks"])
        and additional["unseen_relative_cross_track_rmse_reduction"]
        >= float(gate_cfg["minimum_unseen_relative_cross_track_rmse_reduction"])
        and additional["unseen_cross_track_improvement_ci95_lower_m"]
        > float(gate_cfg["minimum_unseen_cross_track_improvement_ci95_lower_m"])
        and additional["positive_scenes"] >= int(gate_cfg["minimum_positive_scenes"])
        and additional["relative_applied_jerk_increase"]
        <= float(gate_cfg["maximum_relative_applied_jerk_increase"])
    )
    base["eligibility_gate"].update(additional)
    base["eligibility_gate"]["base_path_gate_passed"] = bool(
        base["eligibility_gate"]["passed"]
    )
    base["eligibility_gate"]["additional_high_dynamic_gate_passed"] = additional_passed
    base["eligibility_gate"]["passed"] = bool(
        base["eligibility_gate"]["passed"] and additional_passed
    )
    base["per_unseen_model_block"] = per_block_unseen
    base["per_scene"] = per_scene
    base["hierarchical_bootstrap"]["unseen_cross_track_improvement_m"] = unseen_bootstrap
    base["interpretation_guard"] = (
        "L59 is an independent sealed-seed confirmation on the frozen fixed-plant benchmark."
        if bool(design.get("confirmation_mode", False)) else
        "L58 is a new-seed closed-loop development gate, not sealed confirmation."
    )
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    result = summarize_l58(config, input_dir)
    (input_dir / "high_dynamic_closed_loop_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["eligibility_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
