#!/usr/bin/env python3
"""Summarize L47 delay-aligned structured ICODE path tracking."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (
    _base_key, _bool, _hierarchical_bootstrap, _path_specs,
    _read_csv, _tracking_metrics, _write_csv,
)
from mobile_robot_mppi.core.config import load_yaml


NOMINAL = "traditional_nominal"
PRIMARY = "traditional_icode"
GATED = "traditional_icode_support_gate"


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _effect(base, label, reference_tracking, candidate_tracking, reference_episode, candidate_episode):
    reference_rmse = float(reference_tracking["cross_track_rmse_m"])
    candidate_rmse = float(candidate_tracking["cross_track_rmse_m"])
    return {
        "comparison": label, "model_block": base[0], "scene": base[1],
        "physics_domain": base[2], "episode_seed": base[3],
        "cross_track_improvement_m": reference_rmse - candidate_rmse,
        "relative_cross_track_reduction": (reference_rmse - candidate_rmse) / max(reference_rmse, 1e-12),
        "heading_rmse_improvement_rad": float(reference_tracking["tangent_heading_rmse_rad"]) - float(candidate_tracking["tangent_heading_rmse_rad"]),
        "completion_ratio_difference": float(candidate_tracking["completion_ratio"]) - float(reference_tracking["completion_ratio"]),
        "success_difference": int(_bool(candidate_episode["success"])) - int(_bool(reference_episode["success"])),
        "collision_difference": int(_bool(candidate_episode["collision"])) - int(_bool(reference_episode["collision"])),
    }


def _aggregate(rows):
    return {
        "pairs": len(rows),
        "mean_cross_track_improvement_m": float(np.mean([row["cross_track_improvement_m"] for row in rows])),
        "mean_relative_cross_track_reduction": float(np.mean([row["relative_cross_track_reduction"] for row in rows])),
        "mean_heading_rmse_improvement_rad": float(np.mean([row["heading_rmse_improvement_rad"] for row in rows])),
        "mean_completion_ratio_difference": float(np.mean([row["completion_ratio_difference"] for row in rows])),
        "net_success_gain": int(sum(row["success_difference"] for row in rows)),
        "net_collision_increase": int(sum(row["collision_difference"] for row in rows)),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    episode_rows, step_rows, protected, metadata_sealed = [], [], set(), set()
    for block in range(len(design["model_blocks"])):
        block_dir = Path(input_dir) / ("block_%d" % block)
        episode_rows.extend(_read_csv(block_dir / "episodes.csv"))
        step_rows.extend(_read_csv(block_dir / "factorial_steps.csv"))
        metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
        protected.update(int(value) for value in metadata["previous_protected_seeds_used"])
        metadata_sealed.update(int(value) for value in metadata["sealed_confirmation_seeds_used"])
    path_specs = _path_specs(design)
    tracking_rows = _tracking_metrics(step_rows, path_specs)
    tracking = {_base_key(row) + (str(row["condition"]),): row for row in tracking_rows}
    episodes = {_base_key(row) + (str(row["condition"]),): row for row in episode_rows}
    conditions = (NOMINAL, PRIMARY, GATED)
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"])) for scene in path_specs
        for domain in design["physics_domains"] for seed in design["development_episode_seeds"]
        for condition in conditions
    }
    effects = []
    if set(episodes) == expected and set(tracking) == expected:
        for base in sorted(key[:-1] for key in expected if key[-1] == NOMINAL):
            effects.extend((
                _effect(base, "primary_vs_nominal", tracking[base + (NOMINAL,)], tracking[base + (PRIMARY,)], episodes[base + (NOMINAL,)], episodes[base + (PRIMARY,)]),
                _effect(base, "gated_vs_nominal", tracking[base + (NOMINAL,)], tracking[base + (GATED,)], episodes[base + (NOMINAL,)], episodes[base + (GATED,)]),
                _effect(base, "gated_vs_primary", tracking[base + (PRIMARY,)], tracking[base + (GATED,)], episodes[base + (PRIMARY,)], episodes[base + (GATED,)]),
            ))
    comparisons = {
        label: [row for row in effects if row["comparison"] == label]
        for label in ("primary_vs_nominal", "gated_vs_nominal", "gated_vs_primary")
    }
    aggregates = {label: _aggregate(rows) for label, rows in comparisons.items()}
    primary_rows = comparisons["primary_vs_nominal"]
    per_block = []
    for block in range(len(design["model_blocks"])):
        values = _aggregate([row for row in primary_rows if row["model_block"] == block])
        values["model_block"] = block
        per_block.append(values)
    bootstrap = {
        label: _hierarchical_bootstrap(rows, "cross_track_improvement_m", int(design["bootstrap_seed"]) + index, int(design["bootstrap_replicates"]))
        for index, (label, rows) in enumerate(comparisons.items())
    }
    primary = aggregates["primary_vs_nominal"]
    primary_episodes = [row for row in episode_rows if row["condition"] == PRIMARY]
    compute = float(np.mean([float(row["planner_compute_ms_mean"]) for row in primary_episodes]))
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in episode_rows} & sealed)
    nonfinite = any(not math.isfinite(float(row[field])) for row in tracking_rows for field in ("cross_track_rmse_m", "tangent_heading_rmse_rad", "completion_ratio"))
    integrity = bool(set(episodes) == expected and set(tracking) == expected and not protected and not metadata_sealed and not used_sealed and not nonfinite)
    gate_cfg = design["eligibility_gate"]
    gate = {
        "artifact_integrity": integrity,
        "positive_tracking_model_blocks": int(sum(row["mean_cross_track_improvement_m"] > 0.0 for row in per_block)),
        "relative_cross_track_rmse_reduction": primary["mean_relative_cross_track_reduction"],
        "cross_track_improvement_ci95_lower_m": bootstrap["primary_vs_nominal"]["ci95_lower"],
        "net_success_gain": primary["net_success_gain"],
        "net_collision_increase": primary["net_collision_increase"],
        "completion_ratio_difference": primary["mean_completion_ratio_difference"],
        "candidate_mean_planner_compute_ms": compute,
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["positive_tracking_model_blocks"] >= int(gate_cfg["minimum_positive_tracking_model_blocks"])
        and gate["relative_cross_track_rmse_reduction"] >= float(gate_cfg["minimum_relative_cross_track_rmse_reduction"])
        and gate["cross_track_improvement_ci95_lower_m"] > float(gate_cfg["minimum_cross_track_improvement_ci95_lower_m"])
        and gate["net_success_gain"] >= int(gate_cfg["minimum_net_success_gain"])
        and gate["net_collision_increase"] <= int(gate_cfg["maximum_net_collision_increase"])
        and gate["completion_ratio_difference"] >= float(gate_cfg["minimum_completion_ratio_difference"])
        and gate["candidate_mean_planner_compute_ms"] <= float(gate_cfg["maximum_candidate_mean_planner_compute_ms"])
    )
    summary = {
        "design_id": str(design["design_id"]),
        "audit": {"expected_episodes": len(expected), "observed_episodes": len(episode_rows), "missing_keys": len(expected-set(episodes)), "unexpected_keys": len(set(episodes)-expected), "protected_seeds_used": sorted(protected), "sealed_seeds_used": used_sealed, "metadata_sealed_seeds_used": sorted(metadata_sealed), "nonfinite": nonfinite},
        "comparisons": aggregates, "per_model_block_primary": per_block,
        "hierarchical_bootstrap": bootstrap, "eligibility_gate": gate,
        "interpretation_guard": "L47 is development evidence; sealed confirmation remains closed unless the primary gate passes.",
    }
    _write_csv(Path(input_dir) / "path_tracking_episode_metrics.csv", tracking_rows)
    _write_csv(Path(input_dir) / "path_tracking_paired_effects.csv", effects)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    summary = summarize(config, input_dir)
    (input_dir / "delay_aligned_path_tracking_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

