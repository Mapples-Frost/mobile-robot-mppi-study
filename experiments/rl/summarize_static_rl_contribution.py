#!/usr/bin/env python3
"""Audit whether SAC correction adds value beyond its frozen BC base."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (  # noqa: E402
    _bool,
    _hierarchical_bootstrap,
    _read_csv,
    _write_csv,
)
from mobile_robot_mppi.core.config import load_yaml  # noqa: E402


CONTRASTS = {
    "rl_vs_gated_bc": ("complexity_bc_icode", "gated_lcb_icode"),
    "rl_vs_traditional": ("traditional_icode", "gated_lcb_icode"),
    "gated_bc_vs_traditional": ("traditional_icode", "complexity_bc_icode"),
    "frozen_bc_vs_traditional": ("traditional_icode", "frozen_bc_icode"),
}


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _float(row, name):
    value = row.get(name)
    if value in (None, ""):
        return float("nan")
    return float(value)


def _key(row, include_condition=True):
    values = (
        int(row["model_block"]),
        str(row["scene"]),
        str(row["physics_domain"]),
        int(row["episode_seed"]),
    )
    return values + ((str(row["condition"]),) if include_condition else ())


def _condition_summary(rows):
    result = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "episodes": len(selected),
            "successes": int(sum(_bool(row["success"]) for row in selected)),
            "collisions": int(sum(_bool(row["collision"]) for row in selected)),
            "mean_final_goal_distance_m": float(np.mean([
                _float(row, "final_goal_distance") for row in selected
            ])),
            "mean_planner_compute_ms": float(np.mean([
                _float(row, "planner_compute_ms_mean") for row in selected
            ])),
        }
    return result


def _paired_effects(episodes):
    lookup = {_key(row): row for row in episodes}
    bases = sorted({_key(row, include_condition=False) for row in episodes})
    effects = []
    for contrast, (comparator, candidate) in CONTRASTS.items():
        for base in bases:
            first = lookup[base + (comparator,)]
            second = lookup[base + (candidate,)]
            effects.append({
                "contrast": contrast,
                "model_block": base[0],
                "scene": base[1],
                "physics_domain": base[2],
                "episode_seed": base[3],
                "success_difference": (
                    int(_bool(second["success"]))
                    - int(_bool(first["success"]))
                ),
                "collision_difference": (
                    int(_bool(second["collision"]))
                    - int(_bool(first["collision"]))
                ),
                "final_distance_improvement_m": (
                    _float(first, "final_goal_distance")
                    - _float(second, "final_goal_distance")
                ),
            })
    return effects


def _aggregate(rows, seed, replicates):
    def grouped(name):
        values = []
        for value in sorted({row[name] for row in rows}):
            selected = [row for row in rows if row[name] == value]
            values.append({
                name: value,
                "pairs": len(selected),
                "net_success_gain": int(sum(
                    int(row["success_difference"]) for row in selected
                )),
                "net_collision_increase": int(sum(
                    int(row["collision_difference"]) for row in selected
                )),
                "mean_final_distance_improvement_m": float(np.mean([
                    float(row["final_distance_improvement_m"])
                    for row in selected
                ])),
            })
        return values

    return {
        "pairs": len(rows),
        "net_success_gain": int(sum(
            int(row["success_difference"]) for row in rows
        )),
        "net_collision_increase": int(sum(
            int(row["collision_difference"]) for row in rows
        )),
        "mean_final_distance_improvement_m": float(np.mean([
            float(row["final_distance_improvement_m"]) for row in rows
        ])),
        "success_bootstrap": _hierarchical_bootstrap(
            rows, "success_difference", seed, replicates
        ),
        "distance_bootstrap": _hierarchical_bootstrap(
            rows, "final_distance_improvement_m", seed + 1, replicates
        ),
        "per_model_block": grouped("model_block"),
        "per_scene": grouped("scene"),
    }


def _clean_fallback(steps, clean_scene, candidates):
    fields = (
        "executed_v", "executed_omega", "goal_distance",
        "collision", "safety_override",
    )
    clean = [row for row in steps if row["scene"] == clean_scene]
    lookup = {
        _key(row) + (int(row["step"]),): row for row in clean
    }
    maximum = {field: 0.0 for field in fields}
    missing = []
    for row in clean:
        if row["condition"] != "traditional_icode":
            continue
        base = _key(row, include_condition=False)
        step = int(row["step"])
        for candidate in candidates:
            key = base + (candidate, step)
            if key not in lookup:
                missing.append(key)
                continue
            other = lookup[key]
            for field in fields:
                maximum[field] = max(
                    maximum[field],
                    abs(_float(row, field) - _float(other, field)),
                )
    return {
        "missing_paired_steps": len(missing),
        "max_absolute_difference": maximum,
        "overall_max_absolute_difference": max(maximum.values()),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    gate = design["static_rl_contribution_gate"]
    episodes = []
    steps = []
    artifact_errors = []
    sealed_used = set()
    protected_used = set()
    blocks = len(design["model_blocks"])
    for block in range(blocks):
        block_dir = Path(input_dir) / ("block_%d" % block)
        try:
            episodes.extend(_read_csv(block_dir / "episodes.csv"))
            steps.extend(_read_csv(block_dir / "factorial_steps.csv"))
            metadata = json.loads(
                (block_dir / "metadata.json").read_text(encoding="utf-8")
            )
            sealed_used.update(metadata["sealed_confirmation_seeds_used"])
            protected_used.update(metadata["previous_protected_seeds_used"])
            if int(metadata["model_block"]) != block:
                artifact_errors.append("metadata model block mismatch")
        except (FileNotFoundError, KeyError, ValueError) as error:
            artifact_errors.append("block %d: %s" % (block, error))

    scene_names = [
        str(load_yaml(_resolved(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    domain_names = [str(item["name"]) for item in design["physics_domains"]]
    conditions = [str(value) for value in design["conditions"]]
    expected = {
        (block, scene, domain, int(seed), condition)
        for block in range(blocks)
        for scene in scene_names
        for domain in domain_names
        for seed in design["development_episode_seeds"]
        for condition in conditions
    }
    observed = {_key(row) for row in episodes}
    if observed != expected:
        artifact_errors.append(
            "episode matrix mismatch: missing=%d extra=%d"
            % (len(expected - observed), len(observed - expected))
        )
    if len(episodes) != len(observed):
        artifact_errors.append("duplicate episode keys")
    if sealed_used and not bool(design.get("confirmation_mode", False)):
        artifact_errors.append("sealed confirmation seeds used in development")
    if protected_used:
        artifact_errors.append("protected prior seeds reused")

    finite_fields = (
        "final_goal_distance", "planner_compute_ms_mean",
        "rl_gate_alpha_mean", "rl_correction_advantage_gate_alpha_mean",
    )
    for row in episodes:
        for field in finite_fields:
            value = row.get(field)
            if value not in (None, "") and not np.isfinite(float(value)):
                artifact_errors.append("non-finite %s" % field)

    effects = _paired_effects(episodes) if not artifact_errors else []
    _write_csv(Path(input_dir) / "rl_contribution_paired_effects.csv", effects)
    blocking = set(gate["blocking_scenes"])
    replicates = int(design["bootstrap_replicates"])
    contrasts = {}
    for index, contrast in enumerate(CONTRASTS):
        selected = [
            row for row in effects
            if row["contrast"] == contrast and row["scene"] in blocking
        ]
        contrasts[contrast] = _aggregate(
            selected,
            int(design["bootstrap_seed"]) + 10 * index,
            replicates,
        )

    clean = _clean_fallback(
        steps,
        str(gate["clean_scene"]),
        ("complexity_bc_icode", "gated_lcb_icode"),
    )
    bc_steps = [
        row for row in steps if row["condition"] == "complexity_bc_icode"
    ]
    rl_blocking_steps = [
        row for row in steps
        if row["condition"] == "gated_lcb_icode" and row["scene"] in blocking
    ]
    maximum_base_alpha = max(
        [_float(row, "rl_correction_advantage_gate_alpha") for row in bc_steps]
        or [float("inf")]
    )
    correction_acceptance = float(np.mean([
        _float(row, "rl_correction_advantage_gate_alpha") > 0.5
        for row in rl_blocking_steps
    ])) if rl_blocking_steps else 0.0
    summary = _condition_summary(episodes)
    primary = contrasts.get("rl_vs_gated_bc", {})
    traditional = contrasts.get("rl_vs_traditional", {})
    checks = {
        "artifact_integrity": not artifact_errors,
        "clean_exact_fallback": (
            clean["missing_paired_steps"] == 0
            and clean["overall_max_absolute_difference"]
            <= float(gate["maximum_clean_step_absolute_difference"])
        ),
        "frozen_base_exact": maximum_base_alpha <= float(
            gate["maximum_frozen_base_gate_alpha"]
        ),
        "rl_correction_non_degenerate": correction_acceptance >= float(
            gate["minimum_rl_correction_acceptance_fraction"]
        ),
        "rl_vs_gated_bc_success": primary.get("net_success_gain", -10**9)
        >= int(gate["minimum_rl_vs_gated_bc_net_success_gain"]),
        "rl_vs_gated_bc_positive_blocks": sum(
            row["net_success_gain"] > 0
            for row in primary.get("per_model_block", [])
        ) >= int(gate["minimum_positive_rl_vs_gated_bc_model_blocks"]),
        "rl_vs_gated_bc_positive_scenes": sum(
            row["net_success_gain"] > 0
            for row in primary.get("per_scene", [])
        ) >= int(gate["minimum_positive_rl_vs_gated_bc_scenes"]),
        "rl_vs_gated_bc_collision": primary.get(
            "net_collision_increase", 10**9
        ) <= int(gate["maximum_rl_vs_gated_bc_net_collision_increase"]),
        "rl_vs_traditional_success": traditional.get(
            "net_success_gain", -10**9
        ) >= int(gate["minimum_rl_vs_traditional_net_success_gain"]),
        "rl_vs_traditional_collision": traditional.get(
            "net_collision_increase", 10**9
        ) <= int(gate["maximum_rl_vs_traditional_net_collision_increase"]),
        "compute": summary.get("gated_lcb_icode", {}).get(
            "mean_planner_compute_ms", float("inf")
        ) <= float(gate["maximum_mean_planner_compute_ms"]),
    }
    return {
        "artifact_integrity": not artifact_errors,
        "artifact_errors": artifact_errors,
        "static_rl_contribution_gate_passed": bool(all(checks.values())),
        "checks": checks,
        "contrasts": contrasts,
        "condition_summary": summary,
        "clean_fallback": clean,
        "maximum_frozen_base_gate_alpha": maximum_base_alpha,
        "rl_correction_acceptance_fraction": correction_acceptance,
        "thresholds": dict(gate),
        "interpretation_guard": (
            "L69 holds the qualified ICODE residual, physical plant, sensor "
            "source, MPPI budget and outer LaserScan complexity gate fixed. "
            "The primary contrast changes only the SAC correction relative "
            "to its exact embedded frozen-BC action. Memory is disabled."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    result = summarize(config, input_dir)
    (input_dir / "static_rl_contribution_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["static_rl_contribution_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

