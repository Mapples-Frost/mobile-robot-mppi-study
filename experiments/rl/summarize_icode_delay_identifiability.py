#!/usr/bin/env python3
"""Audit and summarize the preregistered L38 delay diagnostic."""

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

from mobile_robot_mppi.core.config import load_yaml


CONDITIONS = ("traditional_nominal", "traditional_icode")


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _base_key(row):
    return (
        int(row["model_block"]), str(row["scene"]),
        str(row["physics_domain"]), int(row["episode_seed"]),
    )


def _expected_keys(config):
    design = config["rl"]["cross_layer_factorial"]
    scenes = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    return {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"]))
        for scene in scenes
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in CONDITIONS
    }


def _paired_effects(rows):
    lookup = {
        _base_key(row) + (str(row["condition"]),): row for row in rows
    }
    effects = []
    for base in sorted({_base_key(row) for row in rows}):
        nominal = lookup[base + ("traditional_nominal",)]
        icode = lookup[base + ("traditional_icode",)]
        effects.append({
            "model_block": base[0],
            "scene": base[1],
            "physics_domain": base[2],
            "episode_seed": base[3],
            "success_difference": int(_bool(icode["success"])) - int(_bool(nominal["success"])),
            "collision_difference": int(_bool(icode["collision"])) - int(_bool(nominal["collision"])),
            "goal_distance_improvement_m": float(nominal["final_goal_distance"]) - float(icode["final_goal_distance"]),
        })
    return effects


def _bootstrap(rows, field, seed, replicates):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["model_block"])].append(float(row[field]))
    blocks = np.asarray(sorted(grouped), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        means = []
        for block in sampled_blocks:
            values = np.asarray(grouped[int(block)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples[index] = float(np.mean(means))
    estimate = float(np.mean([np.mean(grouped[int(block)]) for block in blocks]))
    return {
        "estimate": estimate,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    rows = []
    protected = set()
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    for block_index in range(len(design["model_blocks"])):
        block_dir = Path(input_dir) / ("block_%d" % block_index)
        rows.extend(_read_csv(block_dir / "episodes.csv"))
        metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
        protected.update(int(value) for value in metadata["previous_protected_seeds_used"])
        sealed.update(int(value) for value in metadata["sealed_confirmation_seeds_used"])
    observed = {
        _base_key(row) + (str(row["condition"]),) for row in rows
    }
    expected = _expected_keys(config)
    nonfinite = 0
    for row in rows:
        for field in ("final_goal_distance", "planner_compute_ms_mean"):
            if not np.isfinite(float(row[field])):
                nonfinite += 1
    effects = _paired_effects(rows) if observed == expected else []
    grouped = defaultdict(list)
    for row in effects:
        grouped[str(row["physics_domain"])].append(row)
    matched = grouped["combined_matched_delay"]
    long_delay = grouped["combined_long_delay"]

    def aggregate(selected):
        return {
            "pairs": len(selected),
            "net_success_gain": int(sum(row["success_difference"] for row in selected)),
            "net_collision_increase": int(sum(row["collision_difference"] for row in selected)),
            "mean_goal_distance_improvement_m": float(np.mean([
                row["goal_distance_improvement_m"] for row in selected
            ])),
        }

    matched_summary = aggregate(matched)
    long_summary = aggregate(long_delay)
    success_interaction = float(np.mean([row["success_difference"] for row in long_delay])) - float(np.mean([row["success_difference"] for row in matched]))
    distance_interaction = float(np.mean([row["goal_distance_improvement_m"] for row in long_delay])) - float(np.mean([row["goal_distance_improvement_m"] for row in matched]))
    gate_cfg = design["diagnostic_gate"]
    integrity = bool(
        observed == expected and not nonfinite and not protected
        and not any(int(row["episode_seed"]) in set(design["sealed_confirmation_episode_seeds"]) for row in rows)
    )
    matched_eligible = bool(
        matched_summary["net_success_gain"] >= int(gate_cfg["minimum_matched_delay_net_success_gain"])
        and matched_summary["net_collision_increase"] <= int(gate_cfg["maximum_matched_delay_net_collision_increase"])
        and matched_summary["mean_goal_distance_improvement_m"] >= float(gate_cfg["minimum_matched_delay_goal_distance_improvement_m"])
    )
    material_delay_loss = bool(
        success_interaction <= float(gate_cfg["minimum_long_minus_matched_success_interaction"])
        or distance_interaction <= float(gate_cfg["maximum_long_minus_matched_goal_distance_interaction_m"])
    )
    bootstrap_seed = int(design["bootstrap_seed"])
    replicates = int(design["bootstrap_replicates"])
    bootstrap = {}
    for domain, selected in grouped.items():
        bootstrap[domain] = {
            field: _bootstrap(selected, field, bootstrap_seed + index, replicates)
            for index, field in enumerate((
                "success_difference", "collision_difference", "goal_distance_improvement_m"
            ))
        }
    return {
        "design_id": "l38_icode_delay_identifiability_v1",
        "audit": {
            "expected_episodes": len(expected),
            "observed_episodes": len(rows),
            "missing_keys": len(expected - observed),
            "unexpected_keys": len(observed - expected),
            "nonfinite_values": nonfinite,
            "protected_seeds_used": sorted(protected),
            "sealed_seeds_used": sorted({int(row["episode_seed"]) for row in rows} & set(design["sealed_confirmation_episode_seeds"])),
        },
        "matched_delay": matched_summary,
        "long_delay": long_summary,
        "delay_interaction": {
            "success_difference": success_interaction,
            "goal_distance_improvement_m": distance_interaction,
        },
        "hierarchical_bootstrap": bootstrap,
        "diagnostic_gate": {
            "artifact_integrity": integrity,
            "matched_delay_eligible": matched_eligible,
            "material_delay_degradation": material_delay_loss,
            "passed": bool(integrity and matched_eligible and material_delay_loss),
        },
        "interpretation_guard": "L38 diagnoses delay moderation on development seeds; it is not an efficacy confirmation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    summary = summarize(config, input_dir)
    path = input_dir / "delay_diagnostic_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
