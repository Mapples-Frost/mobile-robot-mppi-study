#!/usr/bin/env python3
"""Extract pre-collision diagnostics from the L31 temporal-gate candidate."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY = "temporal_gated_lcb_icode"


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    if not rows:
        raise ValueError("no L31 collision episodes were found")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode_diagnostics(steps, window_s=1.0):
    groups = defaultdict(list)
    for row in steps:
        if row["condition"] != PRIMARY:
            continue
        key = (
            int(row["model_block"]), row["scene"], row["scene_role"],
            row["physics_domain"], row["physics_role"],
            int(row["episode_seed"]),
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        values.sort(key=lambda row: int(row["step"]))
        collision_rows = [row for row in values if float(row["collision"]) > 0.5]
        if not collision_rows:
            continue
        collision = collision_rows[0]
        collision_time = float(collision["time"])
        window = [
            row for row in values
            if collision_time - float(window_s) <= float(row["time"]) <= collision_time
        ]

        def numbers(field):
            return np.asarray([float(row[field]) for row in window], dtype=np.float64)

        gate = numbers("rl_gate_alpha")
        temporal = numbers("rl_temporal_closing_gate_alpha")
        closing = numbers("rl_temporal_closing_rate_mps")
        dynamic_distance = numbers("nearest_dynamic_obstacle_center_distance")
        clearance = numbers("clearance")
        output.append({
            "model_block": key[0],
            "scene": key[1],
            "scene_role": key[2],
            "physics_domain": key[3],
            "physics_role": key[4],
            "episode_seed": key[5],
            "collision_time_s": collision_time,
            "precollision_steps": len(window),
            "gate_alpha_at_collision": float(collision["rl_gate_alpha"]),
            "gate_alpha_mean_last_1s": float(np.mean(gate)),
            "temporal_alpha_max_last_1s": float(np.max(temporal)),
            "closing_rate_max_last_1s_mps": float(np.max(closing)),
            "dynamic_center_distance_min_last_1s_m": float(np.min(dynamic_distance)),
            "clearance_min_last_1s_m": float(np.min(clearance)),
            "safety_override_fraction_last_1s": float(np.mean([
                float(row["safety_override"]) for row in window
            ])),
            "executed_v_mean_last_1s_mps": float(np.mean([
                float(row["executed_v"]) for row in window
            ])),
            "executed_abs_omega_mean_last_1s_radps": float(np.mean([
                abs(float(row["executed_omega"])) for row in window
            ])),
        })
    return output


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene_role"], row["physics_role"])].append(row)
    output = []
    for (scene, physics), values in sorted(groups.items()):
        output.append({
            "scene_role": scene,
            "physics_role": physics,
            "collision_episodes": len(values),
            "collision_time_median_s": float(np.median([
                row["collision_time_s"] for row in values
            ])),
            "gate_alpha_at_collision_mean": float(np.mean([
                row["gate_alpha_at_collision"] for row in values
            ])),
            "temporal_alpha_max_last_1s_mean": float(np.mean([
                row["temporal_alpha_max_last_1s"] for row in values
            ])),
            "closing_rate_max_last_1s_mean_mps": float(np.mean([
                row["closing_rate_max_last_1s_mps"] for row in values
            ])),
            "dynamic_center_distance_min_last_1s_mean_m": float(np.mean([
                row["dynamic_center_distance_min_last_1s_m"] for row in values
            ])),
            "safety_override_fraction_last_1s_mean": float(np.mean([
                row["safety_override_fraction_last_1s"] for row in values
            ])),
            "executed_v_mean_last_1s_mps": float(np.mean([
                row["executed_v_mean_last_1s_mps"] for row in values
            ])),
            "executed_abs_omega_mean_last_1s_radps": float(np.mean([
                row["executed_abs_omega_mean_last_1s_radps"] for row in values
            ])),
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.summary_dir).resolve()
    episodes = _episode_diagnostics(_read(directory / "factorial_steps.csv"))
    summary = _summary(episodes)
    _write(directory / "failure_episode_diagnostics.csv", episodes)
    _write(directory / "failure_strata_diagnostics.csv", summary)
    payload = {
        "primary": PRIMARY,
        "collision_episodes": len(episodes),
        "strata": summary,
        "interpretation_guard": (
            "Post-outcome mechanism analysis only; these diagnostics may guide a "
            "new preregistered development experiment but cannot rescue L31."
        ),
    }
    (directory / "failure_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
