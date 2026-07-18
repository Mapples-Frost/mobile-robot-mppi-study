#!/usr/bin/env python3
"""Audit frozen-domain nominal--MLP--ICODE cross-plant structure evidence."""

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


def _by_block(rows):
    blocks = sorted({int(row["model_block"]) for row in rows})
    return [
        {
            "model_block": block,
            "pairs": sum(int(row["model_block"]) == block for row in rows),
            "mean_cross_track_improvement_m": _mean(
                [row for row in rows if int(row["model_block"]) == block],
                "cross_track_improvement_m",
            ),
        }
        for block in blocks
    ]


def _calibration_audit(config, gate):
    calibration_config = load_yaml(_resolved(gate["calibration_config_path"]))
    calibration_summary = json.loads(
        _resolved(gate["calibration_summary_path"]).read_text(encoding="utf-8")
    )
    errors = []
    if not calibration_summary.get("calibration_passed", False):
        errors.append("referenced nominal calibration did not pass")
    design = config["rl"]["cross_layer_factorial"]
    selected_names = list(calibration_summary.get("selected_domain_names", ()))
    current_names = [str(item["name"]) for item in design["physics_domains"]]
    if current_names != selected_names:
        errors.append("cross-plant domains do not match frozen calibration selection")
    calibration_domains = {
        str(item["name"]): item.get("plant_override", {})
        for item in calibration_config["rl"]["cross_layer_factorial"]["physics_domains"]
    }
    for item in design["physics_domains"]:
        if item.get("plant_override", {}) != calibration_domains.get(str(item["name"])):
            errors.append("plant override differs from calibration: %s" % item["name"])
    return errors


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    gate = design["cross_plant_structure_gate"]
    base = summarize_base(config, input_dir)
    effects = _read_csv(Path(input_dir) / "structure_paired_effects.csv")
    artifact_errors = list(base["artifact_errors"])
    artifact_errors.extend(_calibration_audit(config, gate))

    anchor = str(gate["anchor_domain"])
    combined = str(gate["combined_domain"])
    shifted_domains = {
        str(item["name"]) for item in design["physics_domains"]
        if str(item["name"]) != anchor
    }
    unseen_scenes = {
        str(load_yaml(_resolved(item["path"]))["scene"]["name"])
        for item in design["scenes"] if "unseen" in str(item["role"]).lower()
    }

    domain_summary = {}
    shifted_bootstrap = {}
    unseen_shifted_bootstrap = {}
    combined_bootstrap = {}
    shifted_blocks = {}
    unseen_shifted_blocks = {}
    combined_blocks = {}
    for contrast in CONTRASTS:
        rows = [row for row in effects if row["contrast"] == contrast]
        shifted = [row for row in rows if row["physics_domain"] in shifted_domains]
        unseen_shifted = [row for row in shifted if row["scene"] in unseen_scenes]
        combined_rows = [row for row in rows if row["physics_domain"] == combined]
        domain_summary[contrast] = [
            {
                "physics_domain": domain,
                "pairs": len([row for row in rows if row["physics_domain"] == domain]),
                "mean_cross_track_improvement_m": _mean(
                    [row for row in rows if row["physics_domain"] == domain],
                    "cross_track_improvement_m",
                ),
                "mean_relative_cross_track_reduction": _mean(
                    [row for row in rows if row["physics_domain"] == domain],
                    "relative_cross_track_reduction",
                ),
            }
            for domain in sorted({row["physics_domain"] for row in rows})
        ]
        seed = int(design["bootstrap_seed"]) + 10 * list(CONTRASTS).index(contrast)
        replicates = int(design["bootstrap_replicates"])
        shifted_bootstrap[contrast] = _hierarchical_bootstrap(
            shifted, "cross_track_improvement_m", seed, replicates
        )
        unseen_shifted_bootstrap[contrast] = _hierarchical_bootstrap(
            unseen_shifted, "cross_track_improvement_m", seed + 1, replicates
        )
        combined_bootstrap[contrast] = _hierarchical_bootstrap(
            combined_rows, "cross_track_improvement_m", seed + 2, replicates
        )
        shifted_blocks[contrast] = _by_block(shifted)
        unseen_shifted_blocks[contrast] = _by_block(unseen_shifted)
        combined_blocks[contrast] = _by_block(combined_rows)

    checks = {}
    for contrast in ("mlp_vs_nominal", "icode_vs_nominal"):
        checks["%s_positive_shifted_blocks" % contrast] = sum(
            row["mean_cross_track_improvement_m"] > 0.0
            for row in shifted_blocks[contrast]
        ) >= int(gate["minimum_positive_learned_vs_nominal_shifted_blocks"])
        checks["%s_shifted_ci95_lower" % contrast] = (
            shifted_bootstrap[contrast]["ci95_lower"]
            > float(gate["minimum_learned_vs_nominal_shifted_ci95_lower_m"])
        )

    contrast = "icode_vs_mlp"
    checks["icode_vs_mlp_positive_shifted_blocks"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in shifted_blocks[contrast]
    ) >= int(gate["minimum_positive_icode_vs_mlp_shifted_blocks"])
    checks["icode_vs_mlp_positive_shifted_domains"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in domain_summary[contrast]
        if row["physics_domain"] in shifted_domains
    ) >= int(gate["minimum_positive_icode_vs_mlp_shifted_domains"])
    checks["icode_vs_mlp_shifted_ci95_lower"] = (
        shifted_bootstrap[contrast]["ci95_lower"]
        > float(gate["minimum_icode_vs_mlp_shifted_ci95_lower_m"])
    )
    checks["icode_vs_mlp_positive_unseen_shifted_blocks"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in unseen_shifted_blocks[contrast]
    ) >= int(gate["minimum_positive_unseen_icode_vs_mlp_shifted_blocks"])
    checks["icode_vs_mlp_unseen_shifted_ci95_lower"] = (
        unseen_shifted_bootstrap[contrast]["ci95_lower"]
        > float(gate["minimum_unseen_icode_vs_mlp_shifted_ci95_lower_m"])
    )
    checks["icode_vs_mlp_positive_combined_blocks"] = sum(
        row["mean_cross_track_improvement_m"] > 0.0
        for row in combined_blocks[contrast]
    ) >= int(gate["minimum_positive_combined_icode_vs_mlp_blocks"])
    checks["icode_vs_mlp_combined_ci95_lower"] = (
        combined_bootstrap[contrast]["ci95_lower"]
        > float(gate["minimum_combined_icode_vs_mlp_ci95_lower_m"])
    )

    return {
        "artifact_integrity": not artifact_errors,
        "artifact_errors": artifact_errors,
        "cross_plant_gate_passed": bool(
            not artifact_errors
            and base["learned_models_closed_loop_eligible"]
            and base["control_affine_structure_supported_closed_loop"]
            and all(checks.values())
        ),
        "base_closed_loop_summary": base,
        "cross_plant_checks": checks,
        "per_domain": domain_summary,
        "per_shifted_model_block": shifted_blocks,
        "per_unseen_shifted_model_block": unseen_shifted_blocks,
        "per_combined_model_block": combined_blocks,
        "shifted_hierarchical_bootstrap": shifted_bootstrap,
        "unseen_shifted_hierarchical_bootstrap": unseen_shifted_bootstrap,
        "combined_hierarchical_bootstrap": combined_bootstrap,
        "thresholds": dict(gate),
        "interpretation_guard": (
            "Models were trained only on the fixed L56 anchor plant. Physical "
            "domains were selected using nominal MPPI alone and then frozen. "
            + (
                "This is sealed-seed cross-plant confirmation, not RL or "
                "real-robot evidence."
                if bool(design.get("confirmation_mode", False)) else
                "This is development evidence, not sealed confirmation or RL evidence."
            )
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
    (input_dir / "residual_structure_cross_plant_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["cross_plant_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
