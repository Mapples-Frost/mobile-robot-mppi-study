#!/usr/bin/env python3
"""Exploratory paired efficiency analysis after the L47 primary gate failed."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (
    _base_key, _hierarchical_bootstrap, _read_csv,
)
from mobile_robot_mppi.core.config import load_yaml


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def analyze(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    rows = []
    for block in range(len(design["model_blocks"])):
        rows.extend(_read_csv(Path(input_dir) / ("block_%d" % block) / "episodes.csv"))
    lookup = {_base_key(row) + (str(row["condition"]),): row for row in rows}
    effects = []
    for base in sorted({_base_key(row) for row in rows}):
        nominal = lookup[base + ("traditional_nominal",)]
        icode = lookup[base + ("traditional_icode",)]
        effects.append({
            "model_block": base[0], "scene": base[1],
            "physics_domain": base[2], "episode_seed": base[3],
            "step_reduction": float(nominal["steps"]) - float(icode["steps"]),
            "path_length_reduction_m": float(nominal["trajectory_length"]) - float(icode["trajectory_length"]),
            "control_jerk_reduction": float(nominal["control_jerk"]) - float(icode["control_jerk"]),
            "applied_jerk_reduction": float(nominal["applied_control_jerk"]) - float(icode["applied_control_jerk"]),
            "final_distance_improvement_m": float(nominal["final_goal_distance"]) - float(icode["final_goal_distance"]),
        })
    fields = (
        "step_reduction", "path_length_reduction_m", "control_jerk_reduction",
        "applied_jerk_reduction", "final_distance_improvement_m",
    )
    intervals = {
        field: _hierarchical_bootstrap(
            effects, field, int(design["bootstrap_seed"]) + 100 + index,
            int(design["bootstrap_replicates"]),
        )
        for index, field in enumerate(fields)
    }
    per_block = []
    for block in range(len(design["model_blocks"])):
        selected = [row for row in effects if row["model_block"] == block]
        record = {"model_block": block, "pairs": len(selected)}
        record.update({field: float(np.mean([row[field] for row in selected])) for field in fields})
        per_block.append(record)
    return {
        "analysis_type": "exploratory_secondary_after_failed_primary",
        "pairs": len(effects),
        "hierarchical_bootstrap": intervals,
        "per_model_block": per_block,
        "multiplicity_guard": "These endpoints generate L48 hypotheses only; they do not replace the failed L47 primary endpoint.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    result = analyze(config, input_dir)
    (input_dir / "secondary_efficiency_exploratory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

