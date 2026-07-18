#!/usr/bin/env python3
"""Method-blind nominal-only calibration for the L63 cross-plant study."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import (
    _bool,
    _path_specs,
    _read_csv,
    _tracking_metrics,
)
from mobile_robot_mppi.core.config import load_yaml


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def summarize(
    config,
    input_dir,
    calibration_key="cross_plant_calibration",
    interpretation_guard=None,
):
    design = config["rl"]["cross_layer_factorial"]
    calibration = config[calibration_key]
    input_dir = Path(input_dir)
    episodes = _read_csv(input_dir / "block_0" / "episodes.csv")
    steps = _read_csv(input_dir / "block_0" / "factorial_steps.csv")
    metadata = json.loads(
        (input_dir / "block_0" / "metadata.json").read_text(encoding="utf-8")
    )
    tracking = _tracking_metrics(steps, _path_specs(design))
    tracking_lookup = {
        (
            str(row["scene"]), str(row["physics_domain"]),
            int(row["episode_seed"]), str(row["condition"]),
        ): row
        for row in tracking
    }
    episode_lookup = {
        (
            str(row["scene"]), str(row["physics_domain"]),
            int(row["episode_seed"]), str(row["condition"]),
        ): row
        for row in episodes
    }
    scene_names = {
        str(load_yaml(_resolved(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    }
    domain_names = {str(item["name"]) for item in design["physics_domains"]}
    seeds = {int(value) for value in design["development_episode_seeds"]}
    expected = {
        (scene, domain, seed, "traditional_nominal")
        for scene in scene_names for domain in domain_names for seed in seeds
    }
    artifact_errors = []
    if set(episode_lookup) != expected:
        artifact_errors.append("episode key mismatch")
    if set(tracking_lookup) != expected:
        artifact_errors.append("tracking key mismatch")
    if metadata["conditions"] != ["traditional_nominal"]:
        artifact_errors.append("calibration observed a learned method")
    if metadata["previous_protected_seeds_used"]:
        artifact_errors.append("protected seed used")
    if metadata["sealed_confirmation_seeds_used"]:
        artifact_errors.append("sealed seed used")

    grouped = defaultdict(list)
    for key, episode in episode_lookup.items():
        track = tracking_lookup[key]
        grouped[key[1]].append({
            "success": int(_bool(episode["success"])),
            "collision": int(_bool(episode["collision"])),
            "completion_ratio": float(track["completion_ratio"]),
            "cross_track_rmse_m": float(track["cross_track_rmse_m"]),
            "planner_compute_ms": float(episode["planner_compute_ms_mean"]),
        })
    domains = []
    for domain in sorted(grouped):
        rows = grouped[domain]
        domains.append({
            "domain": domain,
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "collision_rate": float(np.mean([row["collision"] for row in rows])),
            "mean_completion_ratio": float(np.mean([
                row["completion_ratio"] for row in rows
            ])),
            "mean_cross_track_rmse_m": float(np.mean([
                row["cross_track_rmse_m"] for row in rows
            ])),
            "mean_planner_compute_ms": float(np.mean([
                row["planner_compute_ms"] for row in rows
            ])),
        })
    lookup = {row["domain"]: row for row in domains}
    anchor_name = str(calibration["anchor_domain"])
    if anchor_name not in lookup:
        artifact_errors.append("anchor domain missing")
        anchor_rmse = float("nan")
    else:
        anchor_rmse = float(lookup[anchor_name]["mean_cross_track_rmse_m"])
    for row in domains:
        row["relative_cross_track_shift_from_anchor"] = abs(
            float(row["mean_cross_track_rmse_m"]) - anchor_rmse
        ) / max(anchor_rmse, 1e-12)
        row["eligible"] = bool(
            row["success_rate"] >= float(calibration["minimum_success_rate"])
            and row["collision_rate"] <= float(calibration["maximum_collision_rate"])
            and row["mean_completion_ratio"]
            >= float(calibration["minimum_mean_completion_ratio"])
        )

    selections = {}
    selection_errors = []
    minimum_shift = float(calibration["minimum_relative_cross_track_shift"])
    minimum_shift_by_group = {
        str(name): float(value)
        for name, value in calibration.get(
            "minimum_relative_cross_track_shift_by_group", {}
        ).items()
    }
    for group, candidates in calibration["groups"].items():
        eligible = [
            lookup[str(name)] for name in candidates
            if str(name) in lookup and lookup[str(name)]["eligible"]
        ]
        if not eligible:
            selection_errors.append("no eligible %s candidate" % group)
            continue
        selected = max(
            eligible,
            key=lambda row: (
                row["relative_cross_track_shift_from_anchor"], row["domain"]
            ),
        )
        selections[str(group)] = dict(selected)
        group_minimum_shift = minimum_shift_by_group.get(str(group), minimum_shift)
        if selected["relative_cross_track_shift_from_anchor"] < group_minimum_shift:
            selection_errors.append("%s shift below threshold" % group)

    return {
        "calibration_passed": bool(
            not artifact_errors and not selection_errors
            and lookup.get(anchor_name, {}).get("eligible", False)
        ),
        "artifact_integrity": not artifact_errors,
        "artifact_errors": artifact_errors,
        "selection_errors": selection_errors,
        "anchor_domain": anchor_name,
        "domain_summary": domains,
        "selected_domains": selections,
        "selected_domain_names": [anchor_name] + [
            selections[group]["domain"]
            for group in calibration["groups"]
            if group in selections
        ],
        "thresholds": dict(calibration),
        "interpretation_guard": interpretation_guard or (
            "Selection used traditional nominal MPPI only; no MLP or ICODE "
            "outcome was observed during physical-domain calibration."
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
    (input_dir / "cross_plant_calibration_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
