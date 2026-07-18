#!/usr/bin/env python3
"""Describe a locked BC-versus-SAC sealed confirmation without reselection."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml  # noqa: E402


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _finite(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError("%s is NaN or Inf" % field)
    return value


def _pair_rows(rows, target_id):
    keyed = {}
    for row in rows:
        key = (
            int(row["model_block"]),
            str(row["scene"]),
            int(row["episode_seed"]),
        )
        candidate = str(row["candidate_id"])
        if candidate not in ("base", target_id):
            raise ValueError("confirmation contains an unlocked candidate")
        if (key, candidate) in keyed:
            raise ValueError("confirmation contains a duplicate episode")
        keyed[(key, candidate)] = row
    keys = {key for key, _ in keyed}
    expected = {(key, candidate) for key in keys for candidate in ("base", target_id)}
    if set(keyed) != expected:
        raise ValueError("confirmation pairing matrix is incomplete")
    pairs = []
    for key in sorted(keys):
        base = keyed[(key, "base")]
        target = keyed[(key, target_id)]
        base_success = _bool(base["success"])
        target_success = _bool(target["success"])
        pairs.append({
            "model_block": key[0],
            "scene": key[1],
            "episode_seed": key[2],
            "base_success": base_success,
            "sac_success": target_success,
            "success_gain": int(not base_success and target_success),
            "success_loss": int(base_success and not target_success),
            "base_collision": _bool(base["collision"]),
            "sac_collision": _bool(target["collision"]),
            "collision_regression": int(
                not _bool(base["collision"]) and _bool(target["collision"])
            ),
            "base_final_goal_distance_m": _finite(
                base, "final_goal_distance"
            ),
            "sac_final_goal_distance_m": _finite(
                target, "final_goal_distance"
            ),
            "final_goal_distance_improvement_m": (
                _finite(base, "final_goal_distance")
                - _finite(target, "final_goal_distance")
            ),
            "minimum_clearance_change_m": (
                _finite(target, "minimum_clearance")
                - _finite(base, "minimum_clearance")
            ),
            "planner_compute_mean_change_ms": (
                _finite(target, "planner_compute_ms_mean")
                - _finite(base, "planner_compute_ms_mean")
            ),
            "control_jerk_change": (
                _finite(target, "control_jerk")
                - _finite(base, "control_jerk")
            ),
            "stuck_steps_change": (
                _finite(target, "stuck_steps")
                - _finite(base, "stuck_steps")
            ),
            "spin_steps_change": (
                _finite(target, "spin_steps")
                - _finite(base, "spin_steps")
            ),
            "safety_interventions_change": (
                _finite(target, "safety_interventions")
                - _finite(base, "safety_interventions")
            ),
            "sac_correction_gate_alpha_mean": _finite(
                target, "rl_correction_advantage_gate_alpha_mean"
            ),
        })
    return pairs


def _mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def _aggregate(rows, label):
    if not rows:
        raise ValueError("cannot aggregate an empty confirmation group")
    return {
        "group": str(label),
        "pairs": len(rows),
        "base_successes": sum(bool(row["base_success"]) for row in rows),
        "sac_successes": sum(bool(row["sac_success"]) for row in rows),
        "success_gains": sum(int(row["success_gain"]) for row in rows),
        "success_losses": sum(int(row["success_loss"]) for row in rows),
        "net_success_gain": sum(
            int(row["success_gain"]) - int(row["success_loss"])
            for row in rows
        ),
        "collision_regressions": sum(
            int(row["collision_regression"]) for row in rows
        ),
        "mean_final_goal_distance_improvement_m": _mean(
            rows, "final_goal_distance_improvement_m"
        ),
        "mean_minimum_clearance_change_m": _mean(
            rows, "minimum_clearance_change_m"
        ),
        "mean_planner_compute_change_ms": _mean(
            rows, "planner_compute_mean_change_ms"
        ),
        "mean_control_jerk_change": _mean(rows, "control_jerk_change"),
        "mean_stuck_steps_change": _mean(rows, "stuck_steps_change"),
        "mean_spin_steps_change": _mean(rows, "spin_steps_change"),
        "mean_safety_interventions_change": _mean(
            rows, "safety_interventions_change"
        ),
        "mean_sac_correction_gate_alpha": _mean(
            rows, "sac_correction_gate_alpha_mean"
        ),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    candidates = [
        int(value)
        for value in design["deployment_checkpoint_selection"]["candidate_steps"]
    ]
    if not bool(design.get("confirmation_mode", False)):
        raise ValueError("descriptive confirmation requires confirmation_mode")
    if len(candidates) != 1 or candidates[0] <= 0:
        raise ValueError("confirmation must lock exactly one nonzero checkpoint")
    target_id = "step_%09d" % candidates[0]
    rows = []
    for block in range(len(design["model_blocks"])):
        rows.extend(_read_csv(Path(input_dir) / ("block_%d" % block) / "episodes.csv"))
    pairs = _pair_rows(rows, target_id)
    expected_pairs = (
        len(design["model_blocks"])
        * len(design["scenes"])
        * len(design["development_episode_seeds"])
    )
    if len(pairs) != expected_pairs:
        raise ValueError("confirmation pair count does not match config")
    block_summaries = [
        _aggregate(
            [row for row in pairs if int(row["model_block"]) == block],
            "block_%d" % block,
        )
        for block in range(len(design["model_blocks"]))
    ]
    scene_names = sorted({str(row["scene"]) for row in pairs})
    scene_summaries = [
        _aggregate(
            [row for row in pairs if str(row["scene"]) == scene],
            scene,
        )
        for scene in scene_names
    ]
    discordant = [
        row for row in pairs
        if int(row["success_gain"]) or int(row["success_loss"])
    ]
    return {
        "study_label": str(design.get("study_label", "confirmation")),
        "confirmation_mode": True,
        "locked_candidate_step": candidates[0],
        "pair_count": len(pairs),
        "overall": _aggregate(pairs, "overall"),
        "by_model_block": block_summaries,
        "by_scene": scene_summaries,
        "discordant_success_pairs": discordant,
        "interpretation_guard": (
            "Episode-level summaries are descriptive. Episodes nested within "
            "the same training block are not independent model-training replicates."
        ),
    }, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    result, pairs = summarize(config, input_dir)
    prefix = str(
        config["rl"]["cross_layer_factorial"].get(
            "artifact_prefix", "confirmation"
        )
    )
    (input_dir / ("%s_confirmation_descriptive.json" % prefix)).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        input_dir / ("%s_confirmation_pairs.csv" % prefix),
        pairs,
    )
    _write_csv(
        input_dir / ("%s_confirmation_by_block.csv" % prefix),
        result["by_model_block"],
    )
    _write_csv(
        input_dir / ("%s_confirmation_by_scene.csv" % prefix),
        result["by_scene"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
