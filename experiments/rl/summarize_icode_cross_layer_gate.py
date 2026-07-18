#!/usr/bin/env python3
"""Summarize the preregistered L54 causal cross-layer ICODE gate."""

import argparse
import json
import math
import sys
from collections import defaultdict
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
GATED = "traditional_icode_reliability_gate"
ORACLE = "traditional_icode_oracle_domain_gate"
CONDITIONS = (NOMINAL, ICODE, GATED, ORACLE)
IDENTITY_FIELDS = (
    "time", "x", "y", "theta", "v", "omega", "goal_distance",
    "collision", "executed_v", "executed_omega", "applied_v",
    "applied_omega", "safety_override", "safety_reason",
)


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _aggregate(rows):
    numeric = (
        "path_length_reduction_m", "control_jerk_reduction",
        "applied_jerk_reduction", "cross_track_improvement_m",
        "relative_cross_track_reduction", "completion_ratio_difference",
    )
    result = {name: float(np.mean([row[name] for row in rows])) for name in numeric}
    result.update({
        "pairs": len(rows),
        "net_success_gain": int(sum(row["success_difference"] for row in rows)),
        "net_collision_increase": int(sum(row["collision_difference"] for row in rows)),
    })
    return result


def _identity_mismatches(step_lookup, bases, condition, comparator_for_base):
    mismatches = []
    for base in sorted(bases):
        comparator = comparator_for_base(base)
        candidate_rows = sorted(
            (key, row) for key, row in step_lookup.items()
            if key[:4] == base and key[4] == condition
        )
        comparator_rows = sorted(
            (key, row) for key, row in step_lookup.items()
            if key[:4] == base and key[4] == comparator
        )
        if len(candidate_rows) != len(comparator_rows):
            mismatches.append({"base": base, "field": "step_count"})
            continue
        for (_, candidate_row), (_, comparator_row) in zip(
            candidate_rows, comparator_rows
        ):
            for field in IDENTITY_FIELDS:
                if str(candidate_row[field]) != str(comparator_row[field]):
                    mismatches.append({
                        "base": base, "step": int(candidate_row["step"]),
                        "field": field,
                    })
                    break
    return mismatches


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    episode_rows, step_rows, protected, metadata_sealed = [], [], set(), set()
    for block in range(len(design["model_blocks"])):
        block_dir = Path(input_dir) / ("block_%d" % block)
        episode_rows.extend(_read_csv(block_dir / "episodes.csv"))
        step_rows.extend(_read_csv(block_dir / "factorial_steps.csv"))
        metadata = json.loads(
            (block_dir / "metadata.json").read_text(encoding="utf-8")
        )
        protected.update(int(value) for value in metadata["previous_protected_seeds_used"])
        metadata_sealed.update(int(value) for value in metadata["sealed_confirmation_seeds_used"])

    path_specs = _path_specs(design)
    tracking_rows = _tracking_metrics(step_rows, path_specs)
    tracking = {_base_key(row) + (str(row["condition"]),): row for row in tracking_rows}
    episodes = {_base_key(row) + (str(row["condition"]),): row for row in episode_rows}
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"])) for scene in path_specs
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"] for condition in CONDITIONS
    }
    bases = {key[:-1] for key in expected if key[-1] == NOMINAL}

    step_lookup, duplicate_steps = {}, 0
    for row in step_rows:
        key = _base_key(row) + (str(row["condition"]), int(row["step"]))
        if key in step_lookup:
            duplicate_steps += 1
        step_lookup[key] = row
    active_domains = set(str(value) for value in design["oracle_residual_active_domains"])
    matched = str(design["reliability_calibration"]["matched_domain"])
    gated_identity = _identity_mismatches(
        step_lookup, [base for base in bases if base[2] == matched],
        GATED, lambda base: NOMINAL,
    )
    oracle_identity = _identity_mismatches(
        step_lookup, bases, ORACLE,
        lambda base: ICODE if base[2] in active_domains else NOMINAL,
    )

    effect_by_condition = {condition: [] for condition in (ICODE, GATED, ORACLE)}
    if set(episodes) == expected and set(tracking) == expected:
        for base in sorted(bases):
            nominal = episodes[base + (NOMINAL,)]
            nominal_tracking = tracking[base + (NOMINAL,)]
            nominal_rmse = float(nominal_tracking["cross_track_rmse_m"])
            for condition in effect_by_condition:
                candidate = episodes[base + (condition,)]
                candidate_tracking = tracking[base + (condition,)]
                candidate_rmse = float(candidate_tracking["cross_track_rmse_m"])
                effect_by_condition[condition].append({
                    "model_block": base[0], "scene": base[1],
                    "physics_domain": base[2], "episode_seed": base[3],
                    "path_length_reduction_m": float(nominal["trajectory_length"]) - float(candidate["trajectory_length"]),
                    "control_jerk_reduction": float(nominal["control_jerk"]) - float(candidate["control_jerk"]),
                    "applied_jerk_reduction": float(nominal["applied_control_jerk"]) - float(candidate["applied_control_jerk"]),
                    "cross_track_improvement_m": nominal_rmse - candidate_rmse,
                    "relative_cross_track_reduction": (nominal_rmse - candidate_rmse) / max(nominal_rmse, 1e-12),
                    "completion_ratio_difference": float(candidate_tracking["completion_ratio"]) - float(nominal_tracking["completion_ratio"]),
                    "success_difference": int(_bool(candidate["success"])) - int(_bool(nominal["success"])),
                    "collision_difference": int(_bool(candidate["collision"])) - int(_bool(nominal["collision"])),
                })

    aggregate = {
        condition: _aggregate(rows) for condition, rows in effect_by_condition.items()
    }
    intervals = {}
    bootstrap_seed = int(design["bootstrap_seed"])
    replicates = int(design["bootstrap_replicates"])
    for condition_index, (condition, rows) in enumerate(effect_by_condition.items()):
        intervals[condition] = {
            field: _hierarchical_bootstrap(
                rows, field, bootstrap_seed + condition_index * 10 + field_index,
                replicates,
            )
            for field_index, field in enumerate((
                "path_length_reduction_m", "control_jerk_reduction",
                "applied_jerk_reduction", "cross_track_improvement_m",
            ))
        }

    oracle_lookup = {
        (row["model_block"], row["scene"], row["physics_domain"], row["episode_seed"]): row
        for row in effect_by_condition[ORACLE]
    }
    retention = float(
        design["cross_layer_gate"]["minimum_applied_jerk_retention_vs_oracle"]
    )
    retention_rows = []
    for row in effect_by_condition[GATED]:
        key = (row["model_block"], row["scene"], row["physics_domain"], row["episode_seed"])
        item = dict(row)
        item["applied_jerk_retention_margin"] = (
            row["applied_jerk_reduction"]
            - retention * oracle_lookup[key]["applied_jerk_reduction"]
        )
        retention_rows.append(item)
    retention_interval = _hierarchical_bootstrap(
        retention_rows, "applied_jerk_retention_margin",
        bootstrap_seed + 100, replicates,
    )

    gated_steps = [row for row in step_rows if row["condition"] == GATED]
    gate_values = defaultdict(list)
    gate_active = defaultdict(list)
    cold_start_violations = 0
    minimum_samples = int(design["residual_reliability_gate"]["minimum_samples"])
    active_threshold = float(design["cross_layer_gate"]["active_alpha_threshold"])
    context_mismatches = []
    for row in gated_steps:
        block_domain = (int(row["model_block"]), str(row["physics_domain"]))
        alpha = float(row["residual_reliability_alpha"])
        gate_values[block_domain].append(alpha)
        gate_active[block_domain].append(alpha >= active_threshold)
        if int(row["residual_reliability_samples"]) < minimum_samples and alpha != 0.0:
            cold_start_violations += 1
        expected_context = 1.0 if str(row["physics_domain"]) in active_domains else 0.0
        if abs(float(row["residual_reliability_context_alpha"]) - expected_context) > 1e-12:
            context_mismatches.append({
                "model_block": int(row["model_block"]),
                "physics_domain": str(row["physics_domain"]),
                "episode_seed": int(row["episode_seed"]),
                "step": int(row["step"]),
            })
    per_block_activation = []
    for block in range(len(design["model_blocks"])):
        long_values = gate_values[(block, next(iter(active_domains)))]
        matched_values = gate_values[(block, matched)]
        per_block_activation.append({
            "model_block": block,
            "long_delay_mean_alpha": float(np.mean(long_values)),
            "matched_delay_mean_alpha": float(np.mean(matched_values)),
            "long_delay_active_fraction": float(np.mean(gate_active[(block, next(iter(active_domains)))])),
            "matched_delay_active_fraction": float(np.mean(gate_active[(block, matched)])),
        })

    per_block_effect = []
    for block in range(len(design["model_blocks"])):
        value = _aggregate([
            row for row in effect_by_condition[GATED] if row["model_block"] == block
        ])
        value["model_block"] = block
        per_block_effect.append(value)
    gated_episodes = [row for row in episode_rows if row["condition"] == GATED]
    compute = float(np.mean([
        float(row["planner_compute_ms_mean"]) for row in gated_episodes
    ]))
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in episode_rows} & sealed)
    confirmation_mode = bool(design.get("confirmation_mode", False))
    expected_confirmation_seeds = sorted(
        int(value) for value in design["development_episode_seeds"]
    )
    seed_integrity = bool(
        (
            used_sealed == expected_confirmation_seeds
            and sorted(metadata_sealed) == expected_confirmation_seeds
        )
        if confirmation_mode
        else (not used_sealed and not metadata_sealed)
    )
    all_effect_rows = [row for rows in effect_by_condition.values() for row in rows]
    integrity = bool(
        set(episodes) == expected and set(tracking) == expected
        and duplicate_steps == 0 and not gated_identity and not oracle_identity
        and not protected and seed_integrity
        and not context_mismatches and cold_start_violations == 0
        and all(math.isfinite(float(value)) for row in all_effect_rows
                for key, value in row.items() if key not in ("scene", "physics_domain"))
    )
    gated_aggregate = aggregate[GATED]
    cfg = design["cross_layer_gate"]
    gate = {
        "artifact_integrity": integrity,
        "issued_jerk_ci95_lower": intervals[GATED]["control_jerk_reduction"]["ci95_lower"],
        "applied_jerk_ci95_lower": intervals[GATED]["applied_jerk_reduction"]["ci95_lower"],
        "applied_jerk_retention_margin_ci95_lower": retention_interval["ci95_lower"],
        "positive_applied_jerk_model_blocks": int(sum(
            row["applied_jerk_reduction"] > 0.0 for row in per_block_effect
        )),
        "minimum_long_delay_mean_alpha": min(
            row["long_delay_mean_alpha"] for row in per_block_activation
        ),
        "maximum_matched_delay_mean_alpha": max(
            row["matched_delay_mean_alpha"] for row in per_block_activation
        ),
        "relative_cross_track_rmse_increase": -gated_aggregate["relative_cross_track_reduction"],
        "net_success_gain": gated_aggregate["net_success_gain"],
        "net_collision_increase": gated_aggregate["net_collision_increase"],
        "completion_ratio_difference": gated_aggregate["completion_ratio_difference"],
        "candidate_mean_planner_compute_ms": compute,
        "cold_start_violations": cold_start_violations,
        "context_mismatches": len(context_mismatches),
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["issued_jerk_ci95_lower"] > 0.0
        and gate["applied_jerk_ci95_lower"] > 0.0
        and gate["applied_jerk_retention_margin_ci95_lower"] > 0.0
        and gate["positive_applied_jerk_model_blocks"] >= int(cfg["minimum_positive_applied_jerk_model_blocks"])
        and gate["minimum_long_delay_mean_alpha"] >= float(cfg["minimum_long_delay_mean_alpha"])
        and gate["maximum_matched_delay_mean_alpha"] <= float(cfg["maximum_matched_delay_mean_alpha"])
        and gate["relative_cross_track_rmse_increase"] <= float(cfg["maximum_relative_cross_track_rmse_increase"])
        and gate["net_success_gain"] >= int(cfg["minimum_net_success_gain"])
        and gate["net_collision_increase"] <= int(cfg["maximum_net_collision_increase"])
        and gate["completion_ratio_difference"] >= float(cfg["minimum_completion_ratio_difference"])
        and compute <= float(cfg["maximum_candidate_mean_planner_compute_ms"])
    )
    summary = {
        "design_id": str(design["design_id"]),
        "audit": {
            "expected_episodes": len(expected), "observed_episodes": len(episode_rows),
            "missing_keys": len(expected - set(episodes)),
            "unexpected_keys": len(set(episodes) - expected),
            "duplicate_steps": duplicate_steps,
            "gated_identity_mismatches": gated_identity[:20],
            "oracle_identity_mismatches": oracle_identity[:20],
            "protected_seeds_used": sorted(protected),
            "sealed_seeds_used": used_sealed,
            "metadata_sealed_seeds_used": sorted(metadata_sealed),
            "confirmation_mode": confirmation_mode,
            "seed_integrity": seed_integrity,
        },
        "aggregate": aggregate,
        "hierarchical_bootstrap": intervals,
        "retention_interval": retention_interval,
        "per_model_block": per_block_effect,
        "per_model_block_activation": per_block_activation,
        "cross_layer_gate": gate,
        "interpretation_guard": (
            "L54 is a new-seed development result; sealed confirmation and "
            "measured real-robot latency are still required."
        ),
    }
    return effect_by_condition[GATED], summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    effects, summary = summarize(config, input_dir)
    _write_csv(input_dir / "cross_layer_gate_paired_effects.csv", effects)
    (input_dir / "cross_layer_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["cross_layer_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
