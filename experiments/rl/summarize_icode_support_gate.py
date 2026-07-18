#!/usr/bin/env python3
"""Audit L42 nominal, ungated ICODE and support-gated ICODE."""

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


CONDITIONS = (
    "traditional_nominal", "traditional_icode", "traditional_icode_support_gate"
)


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _base(row):
    return int(row["model_block"]), str(row["scene"]), str(row["physics_domain"]), int(row["episode_seed"])


def _comparison(first, second, label, base):
    return {
        "comparison": label,
        "model_block": base[0], "scene": base[1],
        "physics_domain": base[2], "episode_seed": base[3],
        "success_difference": int(_bool(second["success"])) - int(_bool(first["success"])),
        "collision_difference": int(_bool(second["collision"])) - int(_bool(first["collision"])),
        "goal_distance_improvement_m": float(first["final_goal_distance"]) - float(second["final_goal_distance"]),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    rows = []
    protected = set()
    for block in range(len(design["model_blocks"])):
        root = Path(input_dir) / ("block_%d" % block)
        rows.extend(_read(root / "episodes.csv"))
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        protected.update(int(value) for value in metadata["previous_protected_seeds_used"])
    scenes = [str(load_yaml(_resolved_path(item["path"]))["scene"]["name"]) for item in design["scenes"]]
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"])) for scene in scenes
        for domain in design["physics_domains"] for seed in design["development_episode_seeds"]
        for condition in CONDITIONS
    }
    observed = {_base(row) + (str(row["condition"]),) for row in rows}
    lookup = {_base(row) + (str(row["condition"]),): row for row in rows}
    effects = []
    for base in sorted({_base(row) for row in rows}):
        nominal = lookup[base + (CONDITIONS[0],)]
        ungated = lookup[base + (CONDITIONS[1],)]
        gated = lookup[base + (CONDITIONS[2],)]
        effects.extend((
            _comparison(nominal, gated, "gated_vs_nominal", base),
            _comparison(ungated, gated, "gated_vs_ungated", base),
        ))

    def aggregate(selected):
        return {
            "net_success_gain": int(sum(row["success_difference"] for row in selected)),
            "net_collision_increase": int(sum(row["collision_difference"] for row in selected)),
            "mean_goal_distance_improvement_m": float(np.mean([row["goal_distance_improvement_m"] for row in selected])),
        }

    primary = [row for row in effects if row["comparison"] == "gated_vs_nominal"]
    secondary = [row for row in effects if row["comparison"] == "gated_vs_ungated"]
    per_block = []
    for block in range(len(design["model_blocks"])):
        values = aggregate([row for row in primary if row["model_block"] == block])
        values["model_block"] = block
        per_block.append(values)
    pooled = aggregate(primary)
    gated_vs_ungated = aggregate(secondary)
    gated_rows = [row for row in rows if row["condition"] == CONDITIONS[2]]
    reduced = float(np.mean([float(row["residual_support_reduced_fraction_mean"]) for row in gated_rows]))
    disabled = float(np.mean([float(row["residual_support_disabled_fraction_mean"]) for row in gated_rows]))
    compute = float(np.mean([float(row["planner_compute_ms_mean"]) for row in gated_rows]))
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in rows} & sealed)
    nonfinite = any(not np.isfinite(float(row[name])) for row in rows for name in ("final_goal_distance", "planner_compute_ms_mean"))
    cfg = design["eligibility_gate"]
    gate = {
        "artifact_integrity": bool(observed == expected and not protected and not used_sealed and not nonfinite),
        "positive_success_model_blocks": int(sum(row["net_success_gain"] > 0 for row in per_block)),
        "collision_noninferior_model_blocks": int(sum(row["net_collision_increase"] <= 0 for row in per_block)),
        "pooled_net_success_gain": pooled["net_success_gain"],
        "pooled_net_collision_increase": pooled["net_collision_increase"],
        "mean_goal_distance_improvement_m": pooled["mean_goal_distance_improvement_m"],
        "gated_vs_ungated_collision_increase": gated_vs_ungated["net_collision_increase"],
        "gated_vs_ungated_goal_distance_improvement_m": gated_vs_ungated["mean_goal_distance_improvement_m"],
        "support_reduced_fraction": reduced,
        "support_disabled_fraction": disabled,
        "gated_mean_planner_compute_ms": compute,
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["positive_success_model_blocks"] >= int(cfg["minimum_positive_success_model_blocks"])
        and gate["collision_noninferior_model_blocks"] >= int(cfg["minimum_collision_noninferior_model_blocks"])
        and gate["pooled_net_success_gain"] >= int(cfg["minimum_pooled_net_success_gain"])
        and gate["pooled_net_collision_increase"] <= int(cfg["maximum_pooled_net_collision_increase"])
        and gate["mean_goal_distance_improvement_m"] >= float(cfg["minimum_mean_goal_distance_improvement_m"])
        and gate["gated_vs_ungated_collision_increase"] <= int(cfg["maximum_gated_vs_ungated_collision_increase"])
        and gate["gated_vs_ungated_goal_distance_improvement_m"] >= float(cfg["minimum_gated_vs_ungated_goal_distance_improvement_m"])
        and gate["support_reduced_fraction"] >= float(cfg["minimum_support_reduced_fraction"])
        and gate["support_disabled_fraction"] <= float(cfg["maximum_support_disabled_fraction"])
        and gate["gated_mean_planner_compute_ms"] <= float(cfg["maximum_icode_mean_planner_compute_ms"])
    )
    bootstrap = {
        comparison: {
            field: _bootstrap(selected, field, int(design["bootstrap_seed"]) + index, int(design["bootstrap_replicates"]))
            for index, field in enumerate(("success_difference", "collision_difference", "goal_distance_improvement_m"))
        }
        for comparison, selected in (("gated_vs_nominal", primary), ("gated_vs_ungated", secondary))
    }
    return {
        "design_id": str(design["design_id"]),
        "audit": {"expected_episodes": len(expected), "observed_episodes": len(rows), "missing_keys": len(expected-observed), "unexpected_keys": len(observed-expected), "protected_seeds_used": sorted(protected), "sealed_seeds_used": used_sealed, "nonfinite": nonfinite},
        "per_model_block": per_block,
        "gated_vs_nominal": pooled,
        "gated_vs_ungated": gated_vs_ungated,
        "hierarchical_bootstrap": bootstrap,
        "eligibility_gate": gate,
        "interpretation_guard": "L42 is a support-gate development experiment; confirmation seeds remain sealed.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    summary = summarize(config, input_dir)
    (input_dir / "support_gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
