#!/usr/bin/env python3
"""Audit the preregistered L75 progress-competence activation experiment."""

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


def _float(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError("%s is NaN or Inf" % field)
    return value


def _key(row):
    return (
        int(row["model_block"]),
        str(row["scene"]),
        int(row["episode_seed"]),
    )


def _effect_rows(episodes, base_condition, candidate_condition):
    base = {
        _key(row): row for row in episodes
        if str(row["condition"]) == base_condition
    }
    candidate = {
        _key(row): row for row in episodes
        if str(row["condition"]) == candidate_condition
    }
    if set(base) != set(candidate):
        raise ValueError("base/candidate pairing matrix mismatch")
    rows = []
    for key in sorted(base):
        first = base[key]
        second = candidate[key]
        base_success = _bool(first["success"])
        candidate_success = _bool(second["success"])
        rows.append({
            "model_block": key[0],
            "scene": key[1],
            "episode_seed": key[2],
            "candidate_condition": candidate_condition,
            "base_success": base_success,
            "candidate_success": candidate_success,
            "success_gain": int(not base_success and candidate_success),
            "success_loss": int(base_success and not candidate_success),
            "base_collision": _bool(first["collision"]),
            "candidate_collision": _bool(second["collision"]),
            "collision_regression": int(
                not _bool(first["collision"])
                and _bool(second["collision"])
            ),
            "goal_distance_improvement_m": (
                _float(first, "final_goal_distance")
                - _float(second, "final_goal_distance")
            ),
            "minimum_clearance_change_m": (
                _float(second, "minimum_clearance")
                - _float(first, "minimum_clearance")
            ),
            "planner_compute_change_ms": (
                _float(second, "planner_compute_ms_mean")
                - _float(first, "planner_compute_ms_mean")
            ),
            "candidate_gate_alpha_mean": _float(
                second, "rl_gate_alpha_mean"
            ),
            "candidate_stagnation_activation_mean": _float(
                second, "rl_baseline_stagnation_activation_mean"
            ),
            "candidate_inference_skip_fraction": _float(
                second, "rl_learned_inference_skip_fraction"
            ),
        })
    return rows


def _mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def _aggregate(rows, group):
    if not rows:
        raise ValueError("cannot aggregate empty L75 effect group")
    return {
        "group": str(group),
        "pairs": len(rows),
        "success_gains": sum(int(row["success_gain"]) for row in rows),
        "success_losses": sum(int(row["success_loss"]) for row in rows),
        "net_success_gain": sum(
            int(row["success_gain"]) - int(row["success_loss"])
            for row in rows
        ),
        "collision_regressions": sum(
            int(row["collision_regression"]) for row in rows
        ),
        "mean_goal_distance_improvement_m": _mean(
            rows, "goal_distance_improvement_m"
        ),
        "mean_minimum_clearance_change_m": _mean(
            rows, "minimum_clearance_change_m"
        ),
        "mean_planner_compute_change_ms": _mean(
            rows, "planner_compute_change_ms"
        ),
        "mean_candidate_gate_alpha": _mean(
            rows, "candidate_gate_alpha_mean"
        ),
        "mean_stagnation_activation": _mean(
            rows, "candidate_stagnation_activation_mean"
        ),
        "mean_inference_skip_fraction": _mean(
            rows, "candidate_inference_skip_fraction"
        ),
    }


def _oracle_summary(episodes, conditions):
    keyed = {}
    for row in episodes:
        key = _key(row)
        keyed.setdefault(key, {})[str(row["condition"])] = row
    selected = []
    for key, methods in sorted(keyed.items()):
        if set(methods) != set(conditions):
            raise ValueError("oracle episode matrix is incomplete")

        def rank(row):
            return (
                _bool(row["success"]),
                not _bool(row["collision"]),
                -_float(row, "final_goal_distance"),
            )

        winner = max(methods, key=lambda name: rank(methods[name]))
        row = methods[winner]
        selected.append({
            "model_block": key[0],
            "scene": key[1],
            "episode_seed": key[2],
            "selected_condition": winner,
            "success": _bool(row["success"]),
            "collision": _bool(row["collision"]),
            "final_goal_distance": _float(row, "final_goal_distance"),
        })
    return {
        "pairs": len(selected),
        "successes": sum(bool(row["success"]) for row in selected),
        "collisions": sum(bool(row["collision"]) for row in selected),
        "mean_final_goal_distance": _mean(
            selected, "final_goal_distance"
        ),
        "selection_counts": {
            condition: sum(
                row["selected_condition"] == condition for row in selected
            )
            for condition in conditions
        },
    }, selected


def summarize(config, input_dirs):
    design = config["rl"]["cross_layer_factorial"]
    gate = design["progress_competence_gate"]
    conditions = [str(value) for value in design["conditions"]]
    base_condition = str(gate["base_condition"])
    always_condition = str(gate["always_condition"])
    progress_condition = str(gate["progress_condition"])
    if set(conditions) != {
        base_condition, always_condition, progress_condition
    }:
        raise ValueError("L75 condition contract mismatch")

    episodes = []
    metadata = []
    for raw in input_dirs:
        directory = _resolved(raw)
        episodes.extend(_read_csv(directory / "episodes.csv"))
        metadata.append(json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        ))

    expected_blocks = len(design["model_blocks"])
    expected_scenes = {
        str(load_yaml(_resolved(row["path"]))["scene"]["name"])
        for row in design["scenes"]
    }
    expected_seeds = {
        int(value) for value in design["development_episode_seeds"]
    }
    expected = {
        (block, scene, seed, condition)
        for block in range(expected_blocks)
        for scene in expected_scenes
        for seed in expected_seeds
        for condition in conditions
    }
    observed = {
        (_key(row)[0], _key(row)[1], _key(row)[2], str(row["condition"]))
        for row in episodes
    }
    duplicate_count = len(episodes) - len(observed)
    artifact_errors = []
    if observed != expected:
        artifact_errors.append("episode matrix mismatch")
    if duplicate_count:
        artifact_errors.append("duplicate episode keys")
    if len(metadata) != expected_blocks:
        artifact_errors.append("metadata block count mismatch")
    if any(item.get("sealed_confirmation_seeds_used") for item in metadata):
        artifact_errors.append("sealed seeds were used")
    if any(item.get("previous_protected_seeds_used") for item in metadata):
        artifact_errors.append("protected seeds were reused")
    finite_fields = (
        "final_goal_distance",
        "minimum_clearance",
        "planner_compute_ms_mean",
        "rl_gate_alpha_mean",
        "rl_correction_advantage_gate_alpha_mean",
        "rl_baseline_stagnation_activation_mean",
        "rl_learned_inference_skip_fraction",
    )
    try:
        for row in episodes:
            for field in finite_fields:
                _float(row, field)
    except (KeyError, TypeError, ValueError) as error:
        artifact_errors.append(str(error))

    always_effects = _effect_rows(
        episodes, base_condition, always_condition
    )
    progress_effects = _effect_rows(
        episodes, base_condition, progress_condition
    )
    effect_rows = always_effects + progress_effects
    summaries = []
    for condition, rows in (
        (always_condition, always_effects),
        (progress_condition, progress_effects),
    ):
        overall = _aggregate(rows, "%s:overall" % condition)
        overall["candidate_condition"] = condition
        summaries.append(overall)
        for block in range(expected_blocks):
            value = _aggregate(
                [
                    row for row in rows
                    if int(row["model_block"]) == block
                ],
                "%s:block_%d" % (condition, block),
            )
            value["candidate_condition"] = condition
            summaries.append(value)
        for scene in sorted(expected_scenes):
            value = _aggregate(
                [row for row in rows if str(row["scene"]) == scene],
                "%s:%s" % (condition, scene),
            )
            value["candidate_condition"] = condition
            summaries.append(value)

    progress_overall = _aggregate(progress_effects, "progress:overall")
    progress_blocks = [
        _aggregate(
            [
                row for row in progress_effects
                if int(row["model_block"]) == block
            ],
            "progress:block_%d" % block,
        )
        for block in range(expected_blocks)
    ]
    by_scene = {
        scene: _aggregate(
            [
                row for row in progress_effects
                if str(row["scene"]) == scene
            ],
            "progress:%s" % scene,
        )
        for scene in expected_scenes
    }
    corridor = by_scene["narrow_corridor"]
    u_trap = by_scene["u_trap_long_board"]
    always_alpha = _mean(always_effects, "candidate_gate_alpha_mean")
    progress_alpha = _mean(
        progress_effects, "candidate_gate_alpha_mean"
    )
    alpha_reduction = (
        1.0 - progress_alpha / always_alpha
        if always_alpha > 0.0 else 0.0
    )
    base_rows = [
        row for row in episodes
        if str(row["condition"]) == base_condition
    ]
    maximum_base_correction = max(
        _float(row, "rl_correction_advantage_gate_alpha_mean")
        for row in base_rows
    )
    checks = {
        "artifact_integrity": not artifact_errors,
        "minimum_nonnegative_net_success_blocks": (
            sum(
                int(row["net_success_gain"]) >= 0
                for row in progress_blocks
            )
            >= int(gate["minimum_nonnegative_net_success_blocks"])
        ),
        "minimum_total_success_gains": (
            int(progress_overall["success_gains"])
            >= int(gate["minimum_total_success_gains"])
        ),
        "maximum_total_success_losses": (
            int(progress_overall["success_losses"])
            <= int(gate["maximum_total_success_losses"])
        ),
        "minimum_total_net_success_gain": (
            int(progress_overall["net_success_gain"])
            >= int(gate["minimum_total_net_success_gain"])
        ),
        "maximum_u_trap_success_losses": (
            int(u_trap["success_losses"])
            <= int(gate["maximum_u_trap_success_losses"])
        ),
        "minimum_corridor_net_success_gain": (
            int(corridor["net_success_gain"])
            >= int(gate["minimum_corridor_net_success_gain"])
        ),
        "maximum_collision_regressions": (
            int(progress_overall["collision_regressions"])
            <= int(gate["maximum_collision_regressions"])
        ),
        "minimum_mean_goal_distance_improvement": (
            float(progress_overall["mean_goal_distance_improvement_m"])
            >= float(gate["minimum_mean_goal_distance_improvement_m"])
        ),
        "minimum_gate_alpha_reduction_vs_always": (
            alpha_reduction
            >= float(gate["minimum_gate_alpha_reduction_vs_always"])
        ),
        "minimum_corridor_stagnation_activation": (
            float(corridor["mean_stagnation_activation"])
            >= float(
                gate["minimum_corridor_stagnation_activation_mean"]
            )
        ),
        "frozen_base_correction_exact": (
            maximum_base_correction
            <= float(gate["maximum_frozen_base_correction_gate_alpha"])
        ),
    }
    oracle, oracle_rows = _oracle_summary(episodes, conditions)
    result = {
        "study_label": str(design.get("study_label", "L75")),
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "artifact_errors": artifact_errors,
        "expected_episodes": len(expected),
        "observed_episodes": len(episodes),
        "duplicate_episode_keys": duplicate_count,
        "progress_overall": progress_overall,
        "progress_by_block": progress_blocks,
        "progress_by_scene": by_scene,
        "always_overall": _aggregate(always_effects, "always:overall"),
        "gate_alpha_reduction_vs_always": float(alpha_reduction),
        "maximum_frozen_base_correction_gate_alpha": float(
            maximum_base_correction
        ),
        "episode_oracle": oracle,
        "thresholds": dict(gate),
        "interpretation_guard": str(design["interpretation_guard"]),
    }
    return result, episodes, effect_rows, summaries, oracle_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result, episodes, effects, summaries, oracle_rows = summarize(
        config, args.input_dir
    )
    (output / "l75_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "l75_episodes.csv", episodes)
    _write_csv(output / "l75_paired_effects.csv", effects)
    _write_csv(output / "l75_effect_summaries.csv", summaries)
    _write_csv(output / "l75_episode_oracle.csv", oracle_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
