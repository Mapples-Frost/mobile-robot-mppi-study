#!/usr/bin/env python3
"""Audit and analyze the L41 residual-only closed-loop eligibility gate."""

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

from experiments.rl.summarize_icode_delay_identifiability import _bootstrap
from mobile_robot_mppi.core.config import load_yaml


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _base(row):
    return int(row["model_block"]), str(row["scene"]), str(row["physics_domain"]), int(row["episode_seed"])


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    rows = []
    protected = set()
    for block in range(len(design["model_blocks"])):
        block_dir = Path(input_dir) / ("block_%d" % block)
        rows.extend(_read_csv(block_dir / "episodes.csv"))
        metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
        protected.update(int(value) for value in metadata["previous_protected_seeds_used"])
    scenes = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"]))
        for scene in scenes for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in ("traditional_nominal", "traditional_icode")
    }
    observed = {_base(row) + (str(row["condition"]),) for row in rows}
    lookup = {_base(row) + (str(row["condition"]),): row for row in rows}
    effects = []
    for base in sorted({_base(row) for row in rows}):
        nominal = lookup[base + ("traditional_nominal",)]
        icode = lookup[base + ("traditional_icode",)]
        effects.append({
            "model_block": base[0], "scene": base[1],
            "physics_domain": base[2], "episode_seed": base[3],
            "success_difference": int(_bool(icode["success"])) - int(_bool(nominal["success"])),
            "collision_difference": int(_bool(icode["collision"])) - int(_bool(nominal["collision"])),
            "goal_distance_improvement_m": float(nominal["final_goal_distance"]) - float(icode["final_goal_distance"]),
        })
    per_block = []
    for block in range(len(design["model_blocks"])):
        selected = [row for row in effects if int(row["model_block"]) == block]
        per_block.append({
            "model_block": block,
            "net_success_gain": int(sum(row["success_difference"] for row in selected)),
            "net_collision_increase": int(sum(row["collision_difference"] for row in selected)),
            "mean_goal_distance_improvement_m": float(np.mean([row["goal_distance_improvement_m"] for row in selected])),
        })
    pooled = {
        "net_success_gain": int(sum(row["success_difference"] for row in effects)),
        "net_collision_increase": int(sum(row["collision_difference"] for row in effects)),
        "mean_goal_distance_improvement_m": float(np.mean([row["goal_distance_improvement_m"] for row in effects])),
    }
    icode_rows = [row for row in rows if row["condition"] == "traditional_icode"]
    icode_compute = float(np.mean([float(row["planner_compute_ms_mean"]) for row in icode_rows]))
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in rows} & sealed)
    nonfinite = any(
        not np.isfinite(float(row[name]))
        for row in rows for name in ("final_goal_distance", "planner_compute_ms_mean")
    )
    gate_cfg = design["eligibility_gate"]
    gate = {
        "artifact_integrity": bool(observed == expected and not protected and not used_sealed and not nonfinite),
        "positive_success_model_blocks": int(sum(row["net_success_gain"] > 0 for row in per_block)),
        "collision_noninferior_model_blocks": int(sum(row["net_collision_increase"] <= 0 for row in per_block)),
        "pooled_net_success_gain": pooled["net_success_gain"],
        "pooled_net_collision_increase": pooled["net_collision_increase"],
        "mean_goal_distance_improvement_m": pooled["mean_goal_distance_improvement_m"],
        "icode_mean_planner_compute_ms": icode_compute,
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["positive_success_model_blocks"] >= int(gate_cfg["minimum_positive_success_model_blocks"])
        and gate["collision_noninferior_model_blocks"] >= int(gate_cfg["minimum_collision_noninferior_model_blocks"])
        and gate["pooled_net_success_gain"] >= int(gate_cfg["minimum_pooled_net_success_gain"])
        and gate["pooled_net_collision_increase"] <= int(gate_cfg["maximum_pooled_net_collision_increase"])
        and gate["mean_goal_distance_improvement_m"] >= float(gate_cfg["minimum_mean_goal_distance_improvement_m"])
        and gate["icode_mean_planner_compute_ms"] <= float(gate_cfg["maximum_icode_mean_planner_compute_ms"])
    )
    bootstrap_seed = int(design["bootstrap_seed"])
    replicates = int(design["bootstrap_replicates"])
    bootstrap = {
        field: _bootstrap(effects, field, bootstrap_seed + index, replicates)
        for index, field in enumerate(("success_difference", "collision_difference", "goal_distance_improvement_m"))
    }
    return {
        "design_id": str(design.get("design_id")),
        "audit": {
            "expected_episodes": len(expected), "observed_episodes": len(rows),
            "missing_keys": len(expected - observed), "unexpected_keys": len(observed - expected),
            "protected_seeds_used": sorted(protected), "sealed_seeds_used": used_sealed,
            "nonfinite": bool(nonfinite),
        },
        "per_model_block": per_block,
        "pooled_effect": pooled,
        "hierarchical_bootstrap": bootstrap,
        "eligibility_gate": gate,
        "interpretation_guard": "L41 is residual-only development eligibility, not independent confirmation or RL evidence.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    summary = summarize(config, input_dir)
    (input_dir / "eligibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
