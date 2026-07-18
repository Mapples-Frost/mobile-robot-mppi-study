#!/usr/bin/env python3
"""Audit the preregistered L48 efficiency and smoothness confirmation."""

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
ICODE = "traditional_icode"


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _aggregate(rows):
    fields = (
        "path_length_reduction_m", "control_jerk_reduction",
        "applied_jerk_reduction", "step_reduction",
        "cross_track_improvement_m", "relative_cross_track_reduction",
        "completion_ratio_difference",
    )
    result = {field: float(np.mean([row[field] for row in rows])) for field in fields}
    result.update({
        "pairs": len(rows),
        "net_success_gain": int(sum(row["success_difference"] for row in rows)),
        "net_collision_increase": int(sum(row["collision_difference"] for row in rows)),
    })
    return result


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
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"])) for scene in path_specs
        for domain in design["physics_domains"] for seed in design["development_episode_seeds"]
        for condition in (NOMINAL, ICODE)
    }
    effects = []
    if set(episodes) == expected and set(tracking) == expected:
        for base in sorted(key[:-1] for key in expected if key[-1] == NOMINAL):
            nominal, icode = episodes[base + (NOMINAL,)], episodes[base + (ICODE,)]
            nominal_tracking, icode_tracking = tracking[base + (NOMINAL,)], tracking[base + (ICODE,)]
            nominal_rmse = float(nominal_tracking["cross_track_rmse_m"])
            icode_rmse = float(icode_tracking["cross_track_rmse_m"])
            effects.append({
                "model_block": base[0], "scene": base[1], "physics_domain": base[2], "episode_seed": base[3],
                "path_length_reduction_m": float(nominal["trajectory_length"]) - float(icode["trajectory_length"]),
                "control_jerk_reduction": float(nominal["control_jerk"]) - float(icode["control_jerk"]),
                "applied_jerk_reduction": float(nominal["applied_control_jerk"]) - float(icode["applied_control_jerk"]),
                "step_reduction": float(nominal["steps"]) - float(icode["steps"]),
                "cross_track_improvement_m": nominal_rmse - icode_rmse,
                "relative_cross_track_reduction": (nominal_rmse - icode_rmse) / max(nominal_rmse, 1e-12),
                "completion_ratio_difference": float(icode_tracking["completion_ratio"]) - float(nominal_tracking["completion_ratio"]),
                "success_difference": int(_bool(icode["success"])) - int(_bool(nominal["success"])),
                "collision_difference": int(_bool(icode["collision"])) - int(_bool(nominal["collision"])),
            })
    aggregate = _aggregate(effects)
    interval_fields = (
        "path_length_reduction_m", "control_jerk_reduction",
        "applied_jerk_reduction", "cross_track_improvement_m",
    )
    bootstrap = {
        field: _hierarchical_bootstrap(
            effects, field, int(design["bootstrap_seed"]) + index,
            int(design["bootstrap_replicates"]),
        )
        for index, field in enumerate(interval_fields)
    }
    per_block = []
    for block in range(len(design["model_blocks"])):
        values = _aggregate([row for row in effects if row["model_block"] == block])
        values["model_block"] = block
        per_block.append(values)
    candidate_rows = [row for row in episode_rows if row["condition"] == ICODE]
    compute = float(np.mean([float(row["planner_compute_ms_mean"]) for row in candidate_rows]))
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in episode_rows} & sealed)
    nonfinite = any(not math.isfinite(float(row[field])) for row in tracking_rows for field in ("cross_track_rmse_m", "completion_ratio"))
    integrity = bool(set(episodes) == expected and set(tracking) == expected and not protected and not metadata_sealed and not used_sealed and not nonfinite)
    cfg = design["confirmation_gate"]
    gate = {
        "artifact_integrity": integrity,
        "positive_path_length_model_blocks": int(sum(row["path_length_reduction_m"] > 0.0 for row in per_block)),
        "positive_control_jerk_model_blocks": int(sum(row["control_jerk_reduction"] > 0.0 for row in per_block)),
        "mean_path_length_reduction_m": aggregate["path_length_reduction_m"],
        "path_length_reduction_ci95_lower_m": bootstrap["path_length_reduction_m"]["ci95_lower"],
        "control_jerk_reduction_ci95_lower": bootstrap["control_jerk_reduction"]["ci95_lower"],
        "applied_jerk_reduction_ci95_lower": bootstrap["applied_jerk_reduction"]["ci95_lower"],
        "relative_cross_track_rmse_increase": -aggregate["relative_cross_track_reduction"],
        "net_success_gain": aggregate["net_success_gain"],
        "net_collision_increase": aggregate["net_collision_increase"],
        "completion_ratio_difference": aggregate["completion_ratio_difference"],
        "candidate_mean_planner_compute_ms": compute,
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["positive_path_length_model_blocks"] >= int(cfg["minimum_positive_path_length_model_blocks"])
        and gate["positive_control_jerk_model_blocks"] >= int(cfg["minimum_positive_control_jerk_model_blocks"])
        and gate["mean_path_length_reduction_m"] >= float(cfg["minimum_mean_path_length_reduction_m"])
        and gate["path_length_reduction_ci95_lower_m"] > float(cfg["minimum_path_length_reduction_ci95_lower_m"])
        and gate["control_jerk_reduction_ci95_lower"] > float(cfg["minimum_control_jerk_reduction_ci95_lower"])
        and gate["applied_jerk_reduction_ci95_lower"] > float(cfg["minimum_applied_jerk_reduction_ci95_lower"])
        and gate["relative_cross_track_rmse_increase"] <= float(cfg["maximum_relative_cross_track_rmse_increase"])
        and gate["net_success_gain"] >= int(cfg["minimum_net_success_gain"])
        and gate["net_collision_increase"] <= int(cfg["maximum_net_collision_increase"])
        and gate["completion_ratio_difference"] >= float(cfg["minimum_completion_ratio_difference"])
        and gate["candidate_mean_planner_compute_ms"] <= float(cfg["maximum_candidate_mean_planner_compute_ms"])
    )
    summary = {
        "design_id": str(design["design_id"]),
        "audit": {"expected_episodes": len(expected), "observed_episodes": len(episode_rows), "missing_keys": len(expected-set(episodes)), "unexpected_keys": len(set(episodes)-expected), "protected_seeds_used": sorted(protected), "sealed_seeds_used": used_sealed, "metadata_sealed_seeds_used": sorted(metadata_sealed), "nonfinite": nonfinite},
        "aggregate": aggregate, "per_model_block": per_block,
        "hierarchical_bootstrap": bootstrap, "confirmation_gate": gate,
        "interpretation_guard": (
            "%s is an independent confirmation; all preregistered gate clauses "
            "must pass before the joint gate is reported as passed."
            % str(design["design_id"])
        ),
    }
    _write_csv(Path(input_dir) / "efficiency_paired_effects.csv", effects)
    _write_csv(Path(input_dir) / "path_tracking_episode_metrics.csv", tracking_rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    summary = summarize(config, input_dir)
    (input_dir / "efficiency_confirmation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
