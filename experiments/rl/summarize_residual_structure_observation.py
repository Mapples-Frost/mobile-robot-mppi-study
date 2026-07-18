#!/usr/bin/env python3
"""Audit residual structures under frozen observation latency and odometry stress."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (  # noqa: E402
    _hierarchical_bootstrap,
    _read_csv,
)
from experiments.rl.summarize_residual_structure_closed_loop import (  # noqa: E402
    CONTRASTS,
    summarize as summarize_base,
)
from mobile_robot_mppi.core.config import load_yaml  # noqa: E402


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def _block_summary(rows):
    blocks = sorted({int(row["model_block"]) for row in rows})
    return [
        {
            "model_block": block,
            "pairs": len([row for row in rows if int(row["model_block"]) == block]),
            "mean_cross_track_improvement_m": _mean(
                [row for row in rows if int(row["model_block"]) == block],
                "cross_track_improvement_m",
            ),
        }
        for block in blocks
    ]


def _aggregate(rows):
    return {
        "pairs": len(rows),
        "mean_cross_track_improvement_m": _mean(rows, "cross_track_improvement_m"),
        "mean_relative_cross_track_reduction": _mean(
            rows, "relative_cross_track_reduction"
        ),
        "mean_completion_ratio_difference": _mean(
            rows, "completion_ratio_difference"
        ),
        "net_success_gain": int(sum(int(float(row["success_difference"])) for row in rows)),
        "net_collision_increase": int(sum(int(float(row["collision_difference"])) for row in rows)),
    }


def _calibration_audit(config, gate):
    calibration_config = load_yaml(_resolved(gate["calibration_config_path"]))
    calibration_summary = json.loads(
        _resolved(gate["calibration_summary_path"]).read_text(encoding="utf-8")
    )
    errors = []
    expected_failure = str(gate["expected_calibration_failure"])
    if calibration_summary.get("calibration_passed", True):
        errors.append("observation calibration unexpectedly marked passed")
    if calibration_summary.get("selection_errors") != [expected_failure]:
        errors.append("observation calibration failure differs from preregistration")
    design = config["rl"]["cross_layer_factorial"]
    calibration_domains = {
        str(item["name"]): item
        for item in calibration_config["rl"]["cross_layer_factorial"]["physics_domains"]
    }
    for item in design["physics_domains"]:
        source = calibration_domains.get(str(item["name"]))
        if source is None:
            errors.append("observation domain absent from calibration: %s" % item["name"])
            continue
        if item.get("plant_override", {}) != source.get("plant_override", {}):
            errors.append("plant override differs from calibration: %s" % item["name"])
        if item.get("sensor_override", {}) != source.get("sensor_override", {}):
            errors.append("sensor override differs from calibration: %s" % item["name"])
    if any(item.get("plant_override", {}) for item in design["physics_domains"]):
        errors.append("observation experiment changes the physical plant")
    return errors


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    gate = design["observation_structure_gate"]
    base = summarize_base(config, input_dir)
    effects = _read_csv(Path(input_dir) / "structure_paired_effects.csv")
    artifact_errors = list(base["artifact_errors"])
    artifact_errors.extend(_calibration_audit(config, gate))

    primary_domains = set(gate["primary_domains"])
    shifted_domains = set(gate["shifted_domains"])
    stress_domain = str(gate["stress_domain"])
    unseen_scenes = {
        str(load_yaml(_resolved(item["path"]))["scene"]["name"])
        for item in design["scenes"] if "unseen" in str(item["role"]).lower()
    }
    replicates = int(design["bootstrap_replicates"])
    result_sets = {}
    for contrast_index, contrast in enumerate(CONTRASTS):
        rows = [row for row in effects if row["contrast"] == contrast]
        subsets = {
            "primary": [row for row in rows if row["physics_domain"] in primary_domains],
            "shifted": [row for row in rows if row["physics_domain"] in shifted_domains],
            "unseen_primary": [
                row for row in rows
                if row["physics_domain"] in primary_domains and row["scene"] in unseen_scenes
            ],
            "stress": [row for row in rows if row["physics_domain"] == stress_domain],
        }
        entry = {}
        for subset_index, (name, selected) in enumerate(subsets.items()):
            entry[name] = {
                "aggregate": _aggregate(selected),
                "per_model_block": _block_summary(selected),
                "bootstrap": _hierarchical_bootstrap(
                    selected,
                    "cross_track_improvement_m",
                    int(design["bootstrap_seed"]) + 10 * contrast_index + subset_index,
                    replicates,
                ),
            }
        entry["per_domain"] = [
            {
                "physics_domain": domain,
                **_aggregate([row for row in rows if row["physics_domain"] == domain]),
            }
            for domain in sorted({row["physics_domain"] for row in rows})
        ]
        result_sets[contrast] = entry

    checks = {}
    for contrast in ("mlp_vs_nominal", "icode_vs_nominal"):
        primary = result_sets[contrast]["primary"]
        checks["%s_positive_primary_blocks" % contrast] = sum(
            row["mean_cross_track_improvement_m"] > 0.0
            for row in primary["per_model_block"]
        ) >= int(gate["minimum_positive_learned_vs_nominal_primary_blocks"])
        checks["%s_primary_ci95_lower" % contrast] = (
            primary["bootstrap"]["ci95_lower"]
            > float(gate["minimum_learned_vs_nominal_primary_ci95_lower_m"])
        )
        checks["%s_success" % contrast] = (
            primary["aggregate"]["net_success_gain"]
            >= int(gate["minimum_net_success_gain_vs_nominal"])
        )
        checks["%s_collision" % contrast] = (
            primary["aggregate"]["net_collision_increase"]
            <= int(gate["maximum_net_collision_increase_vs_nominal"])
        )
        checks["%s_completion" % contrast] = (
            primary["aggregate"]["mean_completion_ratio_difference"]
            >= float(gate["minimum_completion_ratio_difference_vs_nominal"])
        )

    direct = result_sets["icode_vs_mlp"]
    checks["icode_vs_mlp_positive_primary_blocks"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in direct["primary"]["per_model_block"]
    ) >= int(gate["minimum_positive_icode_vs_mlp_primary_blocks"])
    checks["icode_vs_mlp_positive_primary_domains"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in direct["per_domain"] if row["physics_domain"] in primary_domains
    ) >= int(gate["minimum_positive_icode_vs_mlp_primary_domains"])
    checks["icode_vs_mlp_primary_ci95_lower"] = (
        direct["primary"]["bootstrap"]["ci95_lower"]
        > float(gate["minimum_icode_vs_mlp_primary_ci95_lower_m"])
    )
    checks["icode_vs_mlp_positive_shifted_blocks"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in direct["shifted"]["per_model_block"]
    ) >= int(gate["minimum_positive_icode_vs_mlp_shifted_blocks"])
    checks["icode_vs_mlp_positive_shifted_domains"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in direct["per_domain"] if row["physics_domain"] in shifted_domains
    ) >= int(gate["minimum_positive_icode_vs_mlp_shifted_domains"])
    checks["icode_vs_mlp_shifted_ci95_lower"] = (
        direct["shifted"]["bootstrap"]["ci95_lower"]
        > float(gate["minimum_icode_vs_mlp_shifted_ci95_lower_m"])
    )
    checks["icode_vs_mlp_positive_unseen_primary_blocks"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in direct["unseen_primary"]["per_model_block"]
    ) >= int(gate["minimum_positive_unseen_primary_icode_vs_mlp_blocks"])
    checks["icode_vs_mlp_unseen_primary_ci95_lower"] = (
        direct["unseen_primary"]["bootstrap"]["ci95_lower"]
        > float(gate["minimum_unseen_primary_icode_vs_mlp_ci95_lower_m"])
    )
    checks["stress_collision_noninferiority"] = all(
        result_sets[contrast]["stress"]["aggregate"]["net_collision_increase"]
        <= int(gate["maximum_stress_collision_increase"])
        for contrast in ("mlp_vs_nominal", "icode_vs_nominal")
    )
    for condition in ("traditional_mlp", "traditional_icode"):
        checks["%s_compute" % condition] = (
            base["condition_summary"][condition]["mean_planner_compute_ms"]
            <= float(gate["maximum_mean_planner_compute_ms"])
        )

    return {
        "artifact_integrity": not artifact_errors,
        "artifact_errors": artifact_errors,
        "observation_structure_gate_passed": bool(
            not artifact_errors and all(checks.values())
        ),
        "checks": checks,
        "contrasts": result_sets,
        "condition_summary": base["condition_summary"],
        "base_summary": base,
        "thresholds": dict(gate),
        "interpretation_guard": (
            "Primary inference covers clean ground truth, bounded 100 ms "
            "observation latency, and bounded noise-plus-latency on one fixed "
            "MuJoCo plant. Raw wheel odometry is a non-gating stress analysis. "
            "RL and memory are disabled."
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
    (input_dir / "residual_structure_observation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["observation_structure_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

