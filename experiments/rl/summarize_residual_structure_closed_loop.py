#!/usr/bin/env python3
"""Audit L61 nominal--MLP--ICODE parameter-matched closed-loop structure ablation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (
    _base_key,
    _bool,
    _hierarchical_bootstrap,
    _path_specs,
    _read_csv,
    _tracking_metrics,
    _write_csv,
)
from mobile_robot_mppi.core.config import load_yaml


CONDITIONS = ("traditional_nominal", "traditional_mlp", "traditional_icode")
CONTRASTS = {
    "mlp_vs_nominal": ("traditional_nominal", "traditional_mlp"),
    "icode_vs_nominal": ("traditional_nominal", "traditional_icode"),
    "icode_vs_mlp": ("traditional_mlp", "traditional_icode"),
}


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _paired_effect(reference_tracking, treatment_tracking, reference_episode, treatment_episode):
    reference_rmse = float(reference_tracking["cross_track_rmse_m"])
    treatment_rmse = float(treatment_tracking["cross_track_rmse_m"])
    return {
        "cross_track_improvement_m": reference_rmse - treatment_rmse,
        "relative_cross_track_reduction": (
            reference_rmse - treatment_rmse
        ) / max(reference_rmse, 1e-12),
        "heading_rmse_improvement_rad": (
            float(reference_tracking["tangent_heading_rmse_rad"])
            - float(treatment_tracking["tangent_heading_rmse_rad"])
        ),
        "completion_ratio_difference": (
            float(treatment_tracking["completion_ratio"])
            - float(reference_tracking["completion_ratio"])
        ),
        "success_difference": (
            int(_bool(treatment_episode["success"]))
            - int(_bool(reference_episode["success"]))
        ),
        "collision_difference": (
            int(_bool(treatment_episode["collision"]))
            - int(_bool(reference_episode["collision"]))
        ),
        "applied_jerk_improvement": (
            float(reference_episode["applied_control_jerk"])
            - float(treatment_episode["applied_control_jerk"])
        ),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    input_dir = Path(input_dir)
    episodes = []
    steps = []
    metadata = []
    for block in range(len(design["model_blocks"])):
        block_dir = input_dir / ("block_%d" % block)
        episodes.extend(_read_csv(block_dir / "episodes.csv"))
        steps.extend(_read_csv(block_dir / "factorial_steps.csv"))
        metadata.append(json.loads(
            (block_dir / "metadata.json").read_text(encoding="utf-8")
        ))
    path_specs = _path_specs(design)
    tracking = _tracking_metrics(steps, path_specs)
    tracking_lookup = {
        _base_key(row) + (str(row["condition"]),): row for row in tracking
    }
    episode_lookup = {
        _base_key(row) + (str(row["condition"]),): row for row in episodes
    }
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"]))
        for scene in path_specs
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in CONDITIONS
    }
    observed = set(episode_lookup)
    artifact_errors = []
    if observed != expected:
        artifact_errors.append("episode key mismatch")
    if set(tracking_lookup) != expected:
        artifact_errors.append("tracking key mismatch")
    if any(item["previous_protected_seeds_used"] for item in metadata):
        artifact_errors.append("protected episode seed used")
    confirmation_mode = bool(design.get("confirmation_mode", False))
    expected_seeds = sorted(
        int(value) for value in design["development_episode_seeds"]
    )
    metadata_sealed = sorted({
        int(value)
        for item in metadata
        for value in item["sealed_confirmation_seeds_used"]
    })
    if confirmation_mode:
        if metadata_sealed != expected_seeds:
            artifact_errors.append("confirmation seed-unlock mismatch")
    elif metadata_sealed:
        artifact_errors.append("sealed confirmation seed used")

    effects = []
    if not artifact_errors:
        bases = sorted(key[:-1] for key in expected if key[-1] == CONDITIONS[0])
        for base in bases:
            for name, (reference, treatment) in CONTRASTS.items():
                row = {
                    "model_block": base[0],
                    "scene": base[1],
                    "physics_domain": base[2],
                    "episode_seed": base[3],
                    "contrast": name,
                }
                row.update(_paired_effect(
                    tracking_lookup[base + (reference,)],
                    tracking_lookup[base + (treatment,)],
                    episode_lookup[base + (reference,)],
                    episode_lookup[base + (treatment,)],
                ))
                effects.append(row)
    if effects:
        _write_csv(input_dir / "structure_paired_effects.csv", effects)

    pooled = {}
    per_block = {}
    bootstraps = {}
    per_unseen_block = {}
    unseen_scenes = {
        str(load_yaml(_resolved(item["path"]))["scene"]["name"])
        for item in design["scenes"]
        if "unseen" in str(item["role"]).lower()
    }
    for name in CONTRASTS:
        rows = [row for row in effects if row["contrast"] == name]
        unseen = [row for row in rows if row["scene"] in unseen_scenes]
        pooled[name] = {
            "pairs": len(rows),
            "mean_cross_track_improvement_m": float(np.mean([
                row["cross_track_improvement_m"] for row in rows
            ])),
            "mean_relative_cross_track_reduction": float(np.mean([
                row["relative_cross_track_reduction"] for row in rows
            ])),
            "mean_heading_rmse_improvement_rad": float(np.mean([
                row["heading_rmse_improvement_rad"] for row in rows
            ])),
            "mean_completion_ratio_difference": float(np.mean([
                row["completion_ratio_difference"] for row in rows
            ])),
            "net_success_gain": int(sum(
                row["success_difference"] for row in rows
            )),
            "net_collision_increase": int(sum(
                row["collision_difference"] for row in rows
            )),
            "mean_applied_jerk_improvement": float(np.mean([
                row["applied_jerk_improvement"] for row in rows
            ])),
        }
        bootstraps[name] = _hierarchical_bootstrap(
            rows, "cross_track_improvement_m",
            int(design["bootstrap_seed"]),
            int(design["bootstrap_replicates"]),
        )
        bootstraps["%s_unseen" % name] = _hierarchical_bootstrap(
            unseen, "cross_track_improvement_m",
            int(design["bootstrap_seed"]) + 1,
            int(design["bootstrap_replicates"]),
        )
        per_block[name] = []
        per_unseen_block[name] = []
        for block in range(len(design["model_blocks"])):
            selected = [row for row in rows if row["model_block"] == block]
            selected_unseen = [
                row for row in unseen if row["model_block"] == block
            ]
            per_block[name].append({
                "model_block": block,
                "pairs": len(selected),
                "mean_cross_track_improvement_m": float(np.mean([
                    row["cross_track_improvement_m"] for row in selected
                ])),
            })
            per_unseen_block[name].append({
                "model_block": block,
                "pairs": len(selected_unseen),
                "mean_cross_track_improvement_m": float(np.mean([
                    row["cross_track_improvement_m"] for row in selected_unseen
                ])),
            })

    compute = {}
    for condition in CONDITIONS:
        rows = [row for row in episodes if row["condition"] == condition]
        compute[condition] = {
            "episodes": len(rows),
            "successes": int(sum(_bool(row["success"]) for row in rows)),
            "collisions": int(sum(_bool(row["collision"]) for row in rows)),
            "mean_planner_compute_ms": float(np.mean([
                float(row["planner_compute_ms_mean"]) for row in rows
            ])),
            "max_planner_compute_ms": float(np.max([
                float(row["planner_compute_ms_max"]) for row in rows
            ])),
        }

    gate = design["structure_gate"]
    learned_checks = {}
    for name in ("mlp_vs_nominal", "icode_vs_nominal"):
        learned_checks["%s_positive_blocks" % name] = int(sum(
            row["mean_cross_track_improvement_m"] > 0.0
            for row in per_block[name]
        )) >= int(gate["minimum_positive_learned_vs_nominal_blocks"])
        learned_checks["%s_ci95_lower" % name] = (
            bootstraps[name]["ci95_lower"]
            > float(gate["minimum_learned_vs_nominal_ci95_lower_m"])
        )
        learned_checks["%s_success" % name] = (
            pooled[name]["net_success_gain"]
            >= int(gate["minimum_net_success_gain_vs_nominal"])
        )
        learned_checks["%s_collision" % name] = (
            pooled[name]["net_collision_increase"]
            <= int(gate["maximum_net_collision_increase_vs_nominal"])
        )
        learned_checks["%s_completion" % name] = (
            pooled[name]["mean_completion_ratio_difference"]
            >= float(gate["minimum_completion_ratio_difference_vs_nominal"])
        )
    learned_checks["mlp_compute"] = (
        compute["traditional_mlp"]["mean_planner_compute_ms"]
        <= float(gate["maximum_mean_planner_compute_ms"])
    )
    learned_checks["icode_compute"] = (
        compute["traditional_icode"]["mean_planner_compute_ms"]
        <= float(gate["maximum_mean_planner_compute_ms"])
    )
    structure_checks = {
        "positive_icode_vs_mlp_blocks": int(sum(
            row["mean_cross_track_improvement_m"] > 0.0
            for row in per_block["icode_vs_mlp"]
        )) >= int(gate["minimum_positive_icode_vs_mlp_blocks"]),
        "icode_vs_mlp_ci95_lower": (
            bootstraps["icode_vs_mlp"]["ci95_lower"]
            > float(gate["minimum_icode_vs_mlp_ci95_lower_m"])
        ),
        "positive_unseen_icode_vs_mlp_blocks": int(sum(
            row["mean_cross_track_improvement_m"] > 0.0
            for row in per_unseen_block["icode_vs_mlp"]
        )) >= int(gate["minimum_positive_unseen_icode_vs_mlp_blocks"]),
        "unseen_icode_vs_mlp_ci95_lower": (
            bootstraps["icode_vs_mlp_unseen"]["ci95_lower"]
            > float(gate["minimum_unseen_icode_vs_mlp_ci95_lower_m"])
        ),
    }
    return {
        "artifact_integrity": not artifact_errors,
        "artifact_errors": artifact_errors,
        "learned_models_closed_loop_eligible": bool(
            not artifact_errors and all(learned_checks.values())
        ),
        "control_affine_structure_supported_closed_loop": bool(
            not artifact_errors and all(structure_checks.values())
        ),
        "learned_model_checks": learned_checks,
        "structure_checks": structure_checks,
        "pooled_contrasts": pooled,
        "per_model_block": per_block,
        "per_unseen_model_block": per_unseen_block,
        "hierarchical_bootstrap": bootstraps,
        "condition_summary": compute,
        "thresholds": dict(gate),
        "interpretation_guard": (
            "L62 is a sealed-seed residual-structure confirmation on one fixed "
            "clean-state MuJoCo plant; it is not cross-plant or RL evidence."
            if confirmation_mode else
            "L61 is a residual-structure development ablation on one fixed "
            "clean-state MuJoCo plant; it is not sealed confirmation or RL evidence."
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
    (input_dir / "residual_structure_closed_loop_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    passed = bool(
        result["learned_models_closed_loop_eligible"]
        and result["control_affine_structure_supported_closed_loop"]
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
