#!/usr/bin/env python3
"""Audit the preregistered L51 oracle-domain residual-gate upper bound."""

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
ORACLE = "traditional_icode_oracle_domain_gate"
CONDITIONS = (NOMINAL, ICODE, ORACLE)
IDENTITY_FIELDS = (
    "time", "x", "y", "theta", "v", "omega", "goal_distance",
    "collision", "executed_v", "executed_omega", "applied_v",
    "applied_omega", "safety_override", "safety_reason",
)


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _aggregate(rows):
    fields = (
        "path_length_reduction_m", "control_jerk_reduction",
        "applied_jerk_reduction", "cross_track_improvement_m",
        "relative_cross_track_reduction", "completion_ratio_difference",
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
        for condition in CONDITIONS
    }

    step_lookup = {}
    duplicate_steps = 0
    for row in step_rows:
        key = _base_key(row) + (str(row["condition"]), int(row["step"]))
        if key in step_lookup:
            duplicate_steps += 1
        step_lookup[key] = row
    identity_mismatches = []
    active = set(str(value) for value in design["oracle_residual_active_domains"])
    for base in sorted(key[:-1] for key in expected if key[-1] == ORACLE):
        comparator = ICODE if base[2] in active else NOMINAL
        oracle_rows = sorted(
            (key, row) for key, row in step_lookup.items()
            if key[:4] == base and key[4] == ORACLE
        )
        comparator_rows = sorted(
            (key, row) for key, row in step_lookup.items()
            if key[:4] == base and key[4] == comparator
        )
        if len(oracle_rows) != len(comparator_rows):
            identity_mismatches.append({"base": base, "field": "step_count"})
            continue
        for (_, oracle_row), (_, comparator_row) in zip(oracle_rows, comparator_rows):
            for field in IDENTITY_FIELDS:
                if str(oracle_row[field]) != str(comparator_row[field]):
                    identity_mismatches.append({
                        "base": base, "step": int(oracle_row["step"]), "field": field,
                    })
                    break

    effects = []
    if set(episodes) == expected and set(tracking) == expected:
        for base in sorted(key[:-1] for key in expected if key[-1] == NOMINAL):
            nominal, oracle = episodes[base + (NOMINAL,)], episodes[base + (ORACLE,)]
            nominal_tracking = tracking[base + (NOMINAL,)]
            oracle_tracking = tracking[base + (ORACLE,)]
            nominal_rmse = float(nominal_tracking["cross_track_rmse_m"])
            oracle_rmse = float(oracle_tracking["cross_track_rmse_m"])
            effects.append({
                "model_block": base[0], "scene": base[1],
                "physics_domain": base[2], "episode_seed": base[3],
                "path_length_reduction_m": float(nominal["trajectory_length"]) - float(oracle["trajectory_length"]),
                "control_jerk_reduction": float(nominal["control_jerk"]) - float(oracle["control_jerk"]),
                "applied_jerk_reduction": float(nominal["applied_control_jerk"]) - float(oracle["applied_control_jerk"]),
                "cross_track_improvement_m": nominal_rmse - oracle_rmse,
                "relative_cross_track_reduction": (nominal_rmse - oracle_rmse) / max(nominal_rmse, 1e-12),
                "completion_ratio_difference": float(oracle_tracking["completion_ratio"]) - float(nominal_tracking["completion_ratio"]),
                "success_difference": int(_bool(oracle["success"])) - int(_bool(nominal["success"])),
                "collision_difference": int(_bool(oracle["collision"])) - int(_bool(nominal["collision"])),
            })
    aggregate = _aggregate(effects)
    intervals = {
        field: _hierarchical_bootstrap(
            effects, field, int(design["bootstrap_seed"]) + index,
            int(design["bootstrap_replicates"]),
        )
        for index, field in enumerate((
            "path_length_reduction_m", "control_jerk_reduction",
            "applied_jerk_reduction", "cross_track_improvement_m",
        ))
    }
    per_block = []
    for block in range(len(design["model_blocks"])):
        row = _aggregate([item for item in effects if item["model_block"] == block])
        row["model_block"] = block
        per_block.append(row)
    oracle_episodes = [row for row in episode_rows if row["condition"] == ORACLE]
    compute = float(np.mean([float(row["planner_compute_ms_mean"]) for row in oracle_episodes]))
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in episode_rows} & sealed)
    integrity = bool(
        set(episodes) == expected and set(tracking) == expected
        and duplicate_steps == 0 and not identity_mismatches
        and not protected and not metadata_sealed and not used_sealed
        and all(math.isfinite(float(value)) for row in effects for key, value in row.items()
                if key not in ("scene", "physics_domain"))
    )
    cfg = design["oracle_gate"]
    gate = {
        "artifact_integrity": integrity,
        "identity_mismatches": len(identity_mismatches),
        "issued_jerk_ci95_lower": intervals["control_jerk_reduction"]["ci95_lower"],
        "applied_jerk_ci95_lower": intervals["applied_jerk_reduction"]["ci95_lower"],
        "positive_applied_jerk_model_blocks": int(sum(
            row["applied_jerk_reduction"] > 0.0 for row in per_block
        )),
        "relative_cross_track_rmse_increase": -aggregate["relative_cross_track_reduction"],
        "net_success_gain": aggregate["net_success_gain"],
        "net_collision_increase": aggregate["net_collision_increase"],
        "completion_ratio_difference": aggregate["completion_ratio_difference"],
        "candidate_mean_planner_compute_ms": compute,
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["issued_jerk_ci95_lower"] > 0.0
        and gate["applied_jerk_ci95_lower"] > 0.0
        and gate["positive_applied_jerk_model_blocks"] >= int(cfg["minimum_positive_applied_jerk_model_blocks"])
        and gate["relative_cross_track_rmse_increase"] <= float(cfg["maximum_relative_cross_track_rmse_increase"])
        and gate["net_success_gain"] >= int(cfg["minimum_net_success_gain"])
        and gate["net_collision_increase"] <= int(cfg["maximum_net_collision_increase"])
        and gate["completion_ratio_difference"] >= float(cfg["minimum_completion_ratio_difference"])
        and compute <= float(cfg["maximum_candidate_mean_planner_compute_ms"])
    )
    return effects, {
        "design_id": str(design["design_id"]),
        "audit": {
            "expected_episodes": len(expected), "observed_episodes": len(episode_rows),
            "missing_keys": len(expected - set(episodes)),
            "unexpected_keys": len(set(episodes) - expected),
            "duplicate_steps": duplicate_steps,
            "identity_mismatches": identity_mismatches[:20],
            "protected_seeds_used": sorted(protected), "sealed_seeds_used": used_sealed,
            "metadata_sealed_seeds_used": sorted(metadata_sealed),
        },
        "aggregate": aggregate, "per_model_block": per_block,
        "hierarchical_bootstrap": intervals, "oracle_gate": gate,
        "interpretation_guard": "Oracle domain labels are simulator-only and this development result cannot be attributed to the causal online gate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    effects, summary = summarize(config, input_dir)
    _write_csv(input_dir / "oracle_paired_effects.csv", effects)
    (input_dir / "oracle_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["oracle_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
