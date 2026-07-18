#!/usr/bin/env python3
"""Audit and summarize the preregistered L32 development factorial."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.run_cross_layer_factorial import (
    _protected_previous_seeds,
    _resolved_path,
    _write_csv,
)
from experiments.rl.summarize_cross_layer_factorial import (
    _bool,
    _nested_bootstrap,
)


CONDITIONS = (
    "traditional_icode_no_temporal_safety",
    "traditional_icode_temporal_safety",
    "legacy_temporal_gated_lcb_icode",
    "robust_temporal_gated_lcb_icode",
    "robust_temporal_gated_lcb_icode_safety",
    "competence_gated_lcb_icode_safety",
)
COMPARISONS = (
    (
        "shared_safety_traditional",
        "traditional_icode_temporal_safety",
        "traditional_icode_no_temporal_safety",
    ),
    (
        "robust_estimator_without_safety",
        "robust_temporal_gated_lcb_icode",
        "legacy_temporal_gated_lcb_icode",
    ),
    (
        "shared_safety_with_robust_gate",
        "robust_temporal_gated_lcb_icode_safety",
        "robust_temporal_gated_lcb_icode",
    ),
    (
        "competence_separation",
        "competence_gated_lcb_icode_safety",
        "robust_temporal_gated_lcb_icode_safety",
    ),
)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row, include_condition=True):
    result = (
        int(row["model_block"]),
        str(row["scene"]),
        str(row["physics_domain"]),
        int(row["episode_seed"]),
    )
    return result + ((str(row["condition"]),) if include_condition else ())


def _expected_keys(config):
    design = config["rl"]["cross_layer_factorial"]
    scenes = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    return {
        (block, scene, str(domain["name"]), int(seed), str(condition))
        for block in range(len(design["model_blocks"]))
        for scene in scenes
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in design["conditions"]
    }


def _audit(config, rows, metadata):
    expected = _expected_keys(config)
    observed = [_key(row) for row in rows]
    observed_set = set(observed)
    design = config["rl"]["cross_layer_factorial"]
    protected = _protected_previous_seeds(
        design.get("protected_config_paths", ())
    )
    sealed = {
        int(value) for value in design["sealed_confirmation_episode_seeds"]
    }
    used = {int(row["episode_seed"]) for row in rows}
    invalid = 0
    for row in rows:
        try:
            values = (
                float(row["final_goal_distance"]),
                float(row["planner_compute_ms_mean"]),
                float(row.get("temporal_scan_closing_rate_mps_max", 0.0)),
            )
            _bool(row["success"])
            _bool(row["collision"])
            invalid += int(not all(math.isfinite(value) for value in values))
        except (KeyError, TypeError, ValueError):
            invalid += 1
    return {
        "expected_episode_keys": len(expected),
        "observed_episode_rows": len(rows),
        "observed_unique_episode_keys": len(observed_set),
        "missing_episode_keys": len(expected - observed_set),
        "unexpected_episode_keys": len(observed_set - expected),
        "duplicate_episode_keys": len(observed) - len(observed_set),
        "protected_seeds_used": sorted(used.intersection(protected)),
        "sealed_confirmation_seeds_used": sorted(used.intersection(sealed)),
        "invalid_metric_rows": invalid,
        "model_block_metadata_files": len(metadata),
    }


def _condition_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene_role"], row["physics_role"], row["condition"])].append(row)
    output = []
    for (scene, physics, condition), values in sorted(groups.items()):
        output.append({
            "scene_role": scene,
            "physics_role": physics,
            "condition": condition,
            "episodes": len(values),
            "successes": int(sum(_bool(row["success"]) for row in values)),
            "success_rate": float(np.mean([
                _bool(row["success"]) for row in values
            ])),
            "collisions": int(sum(_bool(row["collision"]) for row in values)),
            "collision_rate": float(np.mean([
                _bool(row["collision"]) for row in values
            ])),
            "final_goal_distance_mean": float(np.mean([
                float(row["final_goal_distance"]) for row in values
            ])),
            "minimum_clearance_mean": float(np.mean([
                float(row["minimum_clearance"]) for row in values
            ])),
            "temporal_safety_interventions_mean": float(np.mean([
                float(row.get("temporal_safety_interventions", 0.0))
                for row in values
            ])),
            "temporal_rate_max_mps": float(np.max([
                float(row.get("temporal_scan_closing_rate_mps_max", 0.0))
                for row in values
            ])),
            "rl_gate_alpha_mean": float(np.mean([
                float(row.get("rl_gate_alpha_mean", 0.0)) for row in values
            ])),
            "rl_hazard_activation_mean": float(np.mean([
                float(row.get("rl_hazard_activation_mean", 0.0))
                for row in values
            ])),
            "rl_competence_confidence_mean": float(np.mean([
                float(row.get("rl_competence_confidence_mean", 0.0))
                for row in values
            ])),
        })
    return output


def _paired_effects(rows):
    lookup = {_key(row): row for row in rows}
    output = []
    bases = sorted({_key(row, include_condition=False) for row in rows})
    for label, treatment, comparator in COMPARISONS:
        for base in bases:
            treated = lookup.get(base + (treatment,))
            control = lookup.get(base + (comparator,))
            if treated is None or control is None:
                continue
            output.append({
                "comparison": label,
                "treatment": treatment,
                "comparator": comparator,
                "model_block": base[0],
                "scene": base[1],
                "scene_role": treated["scene_role"],
                "physics_domain": base[2],
                "physics_role": treated["physics_role"],
                "episode_seed": base[3],
                "success_difference": (
                    int(_bool(treated["success"]))
                    - int(_bool(control["success"]))
                ),
                "collision_difference": (
                    int(_bool(treated["collision"]))
                    - int(_bool(control["collision"]))
                ),
                "final_distance_improvement_m": (
                    float(control["final_goal_distance"])
                    - float(treated["final_goal_distance"])
                ),
            })
    return output


def _paired_summary(effects):
    output = []
    for index, (label, treatment, comparator) in enumerate(COMPARISONS):
        values = [row for row in effects if row["comparison"] == label]
        bootstrap = [
            dict(row, value=row["final_distance_improvement_m"])
            for row in values
        ]
        distance_low, distance_high = _nested_bootstrap(
            bootstrap, seed=20260798 + index, replicates=20000
        )
        success_low, success_high = _nested_bootstrap(
            [dict(row, value=row["success_difference"]) for row in values],
            seed=20260808 + index,
            replicates=20000,
        )
        collision_low, collision_high = _nested_bootstrap(
            [dict(row, value=row["collision_difference"]) for row in values],
            seed=20260818 + index,
            replicates=20000,
        )
        output.append({
            "comparison": label,
            "treatment": treatment,
            "comparator": comparator,
            "pairs": len(values),
            "net_success_difference": int(sum(
                row["success_difference"] for row in values
            )),
            "net_collision_difference": int(sum(
                row["collision_difference"] for row in values
            )),
            "success_difference_mean": float(np.mean([
                row["success_difference"] for row in values
            ])) if values else None,
            "success_difference_ci95_low": success_low,
            "success_difference_ci95_high": success_high,
            "collision_difference_mean": float(np.mean([
                row["collision_difference"] for row in values
            ])) if values else None,
            "collision_difference_ci95_low": collision_low,
            "collision_difference_ci95_high": collision_high,
            "final_distance_improvement_mean_m": float(np.mean([
                row["final_distance_improvement_m"] for row in values
            ])) if values else None,
            "final_distance_improvement_ci95_low_m": distance_low,
            "final_distance_improvement_ci95_high_m": distance_high,
        })
    return output


def _development_gate(config, audit, rows, effects):
    design = config["rl"]["cross_layer_factorial"]
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    paired = {
        label: [row for row in effects if row["comparison"] == label]
        for label, _, _ in COMPARISONS
    }
    max_rate = max(
        float(row.get("temporal_scan_closing_rate_mps_max", 0.0))
        for row in rows
    ) if rows else float("inf")
    safety_conditions = (
        "traditional_icode_temporal_safety",
        "robust_temporal_gated_lcb_icode_safety",
        "competence_gated_lcb_icode_safety",
    )
    safety_collisions = {
        condition: int(sum(_bool(row["collision"]) for row in by_condition[condition]))
        for condition in safety_conditions
    }
    primary = by_condition["competence_gated_lcb_icode_safety"]
    primary_successes = int(sum(_bool(row["success"]) for row in primary))
    safety_reduction = -int(sum(
        row["collision_difference"]
        for row in paired["shared_safety_traditional"]
    ))
    competence_success = int(sum(
        row["success_difference"] for row in paired["competence_separation"]
    ))
    competence_collision = int(sum(
        row["collision_difference"] for row in paired["competence_separation"]
    ))
    required = {
        "complete_unique_matrix": (
            audit["missing_episode_keys"] == 0
            and audit["unexpected_episode_keys"] == 0
            and audit["duplicate_episode_keys"] == 0
            and audit["observed_episode_rows"] == audit["expected_episode_keys"]
        ),
        "protected_and_sealed_seeds_untouched": (
            not audit["protected_seeds_used"]
            and not audit["sealed_confirmation_seeds_used"]
        ),
        "finite_metrics_and_all_model_blocks": (
            audit["invalid_metric_rows"] == 0
            and audit["model_block_metadata_files"]
            == len(design["model_blocks"])
        ),
        "robust_rate_bound_respected": (
            max_rate <= float(design["maximum_temporal_rate_mps"]) + 1e-9
        ),
        "all_safety_conditions_collision_free": all(
            value <= int(design["maximum_safety_condition_collisions"])
            for value in safety_collisions.values()
        ),
    }
    efficacy = {
        "primary_success_at_least_90_percent": (
            primary_successes >= int(design["minimum_primary_successes"])
        ),
        "traditional_safety_collision_reduction_at_least_24": (
            safety_reduction >= int(design["minimum_safety_collision_reduction"])
        ),
        "competence_gate_success_noninferior": (
            competence_success
            >= int(design["minimum_competence_success_difference"])
        ),
        "competence_gate_collision_noninferior": (
            competence_collision
            <= int(design["maximum_competence_collision_difference"])
        ),
    }
    passed = all(required.values()) and all(efficacy.values())
    return {
        "decision": "development_pass" if passed else "development_fail",
        "passed": bool(passed),
        "required_checks": required,
        "efficacy_checks": efficacy,
        "estimands": {
            "maximum_observed_temporal_rate_mps": max_rate,
            "safety_condition_collisions": safety_collisions,
            "primary_episodes": len(primary),
            "primary_successes": primary_successes,
            "traditional_safety_net_collision_reduction": safety_reduction,
            "competence_net_success_difference": competence_success,
            "competence_net_collision_difference": competence_collision,
        },
        "interpretation": (
            "Development-only gate. A pass permits but does not automatically "
            "run sealed confirmation seeds."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config = load_yaml(_resolved_path(args.config))
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    steps = []
    metadata = []
    for value in args.run_dir:
        directory = _resolved_path(value)
        rows.extend(_read_csv(directory / "episodes.csv"))
        steps.extend(_read_csv(directory / "factorial_steps.csv"))
        metadata.append(json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        ))

    audit = _audit(config, rows, metadata)
    condition_summary = _condition_summary(rows)
    effects = _paired_effects(rows)
    paired_summary = _paired_summary(effects)
    gate = _development_gate(config, audit, rows, effects)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "factorial_steps.csv", steps)
    _write_csv(output / "condition_summary.csv", condition_summary)
    _write_csv(output / "paired_effects.csv", effects)
    _write_csv(output / "paired_summary.csv", paired_summary)
    for name, value in (
        ("audit.json", audit),
        ("development_gate.json", gate),
        ("source_metadata.json", metadata),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
