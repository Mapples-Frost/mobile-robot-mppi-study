#!/usr/bin/env python3
"""Audit the preregistered L55 path-efficiency confirmation."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_cross_layer_gate import summarize
from mobile_robot_mppi.core.config import load_yaml


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def confirm(config, input_dir):
    effects, base_summary = summarize(config, input_dir)
    design = config["rl"]["cross_layer_factorial"]
    cfg = design["efficiency_confirmation_gate"]
    path_interval = base_summary["hierarchical_bootstrap"][
        "traditional_icode_reliability_gate"
    ]["path_length_reduction_m"]
    cross_track_interval = base_summary["hierarchical_bootstrap"][
        "traditional_icode_reliability_gate"
    ]["cross_track_improvement_m"]
    per_scene_values = defaultdict(list)
    for row in effects:
        per_scene_values[str(row["scene"])].append(
            float(row["path_length_reduction_m"])
        )
    per_scene = [
        {
            "scene": scene,
            "path_length_reduction_m": float(np.mean(values)),
            "pairs": len(values),
        }
        for scene, values in sorted(per_scene_values.items())
    ]
    base_gate = base_summary["cross_layer_gate"]
    aggregate = base_summary["aggregate"]["traditional_icode_reliability_gate"]
    per_block = base_summary["per_model_block"]
    gate = {
        "artifact_integrity": bool(base_gate["artifact_integrity"]),
        "path_length_reduction_m": float(path_interval["estimate"]),
        "path_length_ci95_lower_m": float(path_interval["ci95_lower"]),
        "path_length_ci95_upper_m": float(path_interval["ci95_upper"]),
        "positive_path_length_model_blocks": int(sum(
            row["path_length_reduction_m"] > 0.0 for row in per_block
        )),
        "positive_path_length_scenes": int(sum(
            row["path_length_reduction_m"] > 0.0 for row in per_scene
        )),
        "cross_track_improvement_ci95_lower_m": float(
            cross_track_interval["ci95_lower"]
        ),
        "minimum_long_delay_mean_alpha": float(
            base_gate["minimum_long_delay_mean_alpha"]
        ),
        "maximum_matched_delay_mean_alpha": float(
            base_gate["maximum_matched_delay_mean_alpha"]
        ),
        "net_success_gain": int(aggregate["net_success_gain"]),
        "net_collision_increase": int(aggregate["net_collision_increase"]),
        "completion_ratio_difference": float(
            aggregate["completion_ratio_difference"]
        ),
        "candidate_mean_planner_compute_ms": float(
            base_gate["candidate_mean_planner_compute_ms"]
        ),
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["path_length_ci95_lower_m"]
        > float(cfg["minimum_path_length_reduction_ci95_lower_m"])
        and gate["positive_path_length_model_blocks"]
        >= int(cfg["minimum_positive_path_length_model_blocks"])
        and gate["positive_path_length_scenes"]
        >= int(cfg["minimum_positive_path_length_scenes"])
        and gate["cross_track_improvement_ci95_lower_m"]
        >= float(cfg["minimum_cross_track_improvement_ci95_lower_m"])
        and gate["minimum_long_delay_mean_alpha"]
        >= float(cfg["minimum_long_delay_mean_alpha"])
        and gate["maximum_matched_delay_mean_alpha"]
        <= float(cfg["maximum_matched_delay_mean_alpha"])
        and gate["net_success_gain"] >= int(cfg["minimum_net_success_gain"])
        and gate["net_collision_increase"] <= int(cfg["maximum_net_collision_increase"])
        and gate["completion_ratio_difference"]
        >= float(cfg["minimum_completion_ratio_difference"])
        and gate["candidate_mean_planner_compute_ms"]
        <= float(cfg["maximum_candidate_mean_planner_compute_ms"])
    )
    return effects, {
        "design_id": str(design["design_id"]),
        "audit": base_summary["audit"],
        "aggregate": base_summary["aggregate"],
        "hierarchical_bootstrap": base_summary["hierarchical_bootstrap"],
        "per_model_block": per_block,
        "per_model_block_activation": base_summary["per_model_block_activation"],
        "per_scene": per_scene,
        "efficiency_confirmation_gate": gate,
        "interpretation_guard": (
            "L55 independently confirms only the preregistered path-efficiency "
            "claim; jerk remains a reported secondary outcome."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    effects, summary = confirm(config, input_dir)
    # The base summarizer already produces the paired-effect CSV in its CLI;
    # retain a self-contained JSON for this confirmation gate.
    (input_dir / "efficiency_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["efficiency_confirmation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

