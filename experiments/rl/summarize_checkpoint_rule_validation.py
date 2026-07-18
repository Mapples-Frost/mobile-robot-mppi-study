#!/usr/bin/env python3
"""Audit the preregistered L77 training-internal checkpoint rule."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml  # noqa: E402
from experiments.rl.summarize_correction_support_gate import (  # noqa: E402
    _aggregate,
    _effect_rows,
    _float,
    _key,
    _read_csv,
    _resolved,
    _write_csv,
)


def summarize(config, input_dirs):
    design = config["rl"]["cross_layer_factorial"]
    gate = design["checkpoint_rule_gate"]
    conditions = [str(value) for value in design["conditions"]]
    base_condition = str(gate["base_condition"])
    candidate_condition = str(gate["candidate_condition"])
    if conditions != [base_condition, candidate_condition]:
        raise ValueError("L77 condition contract mismatch")

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
    for block, values in enumerate(design["model_blocks"]):
        expected_checkpoint = str(_resolved(
            values["condition_checkpoints"][candidate_condition]
        ))
        rows = [
            row for row in episodes
            if int(row["model_block"]) == block
            and str(row["condition"]) == candidate_condition
        ]
        if any(str(row["rl_checkpoint"]) != expected_checkpoint for row in rows):
            artifact_errors.append(
                "block %d candidate checkpoint mismatch" % block
            )
    finite_fields = (
        "final_goal_distance",
        "minimum_clearance",
        "planner_compute_ms_mean",
        "rl_correction_advantage_gate_alpha_mean",
        "rl_correction_effective_gate_alpha_mean",
        "rl_applied_correction_abs_mean",
        "rl_ood_score_mean",
    )
    try:
        for row in episodes:
            for field in finite_fields:
                _float(row, field)
    except (KeyError, TypeError, ValueError) as error:
        artifact_errors.append(str(error))

    effects = _effect_rows(
        episodes, base_condition, candidate_condition
    )
    overall = _aggregate(effects, "overall")
    by_block = [
        _aggregate(
            [
                row for row in effects
                if int(row["model_block"]) == block
            ],
            "block_%d" % block,
        )
        for block in range(expected_blocks)
    ]
    by_scene = {
        scene: _aggregate(
            [row for row in effects if str(row["scene"]) == scene],
            scene,
        )
        for scene in expected_scenes
    }
    minimum_improvement = float(
        gate["minimum_block_success_gains_or_distance_improvement_m"]
    )
    block_eligibility = []
    for row in by_block:
        clauses = {
            "success_losses": (
                int(row["success_losses"])
                <= int(gate["maximum_block_success_losses"])
            ),
            "collision_regressions": (
                int(row["collision_regressions"])
                <= int(gate["maximum_block_collision_regressions"])
            ),
            "mean_goal_distance": (
                float(row["mean_goal_distance_improvement_m"])
                >= float(
                    gate["minimum_block_mean_goal_distance_improvement_m"]
                )
            ),
            "correction_active": (
                float(row["mean_advantage_gate_alpha"])
                >= float(gate["minimum_block_correction_gate_alpha"])
            ),
            "task_improvement": (
                int(row["success_gains"]) >= 1
                or float(row["mean_goal_distance_improvement_m"])
                >= minimum_improvement
            ),
        }
        block_eligibility.append({
            "group": row["group"],
            "eligible": bool(all(clauses.values())),
            "clauses": clauses,
        })

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
        "minimum_eligible_model_blocks": (
            sum(bool(row["eligible"]) for row in block_eligibility)
            >= int(gate["minimum_eligible_model_blocks"])
        ),
        "minimum_total_success_gains": (
            int(overall["success_gains"])
            >= int(gate["minimum_total_success_gains"])
        ),
        "maximum_total_success_losses": (
            int(overall["success_losses"])
            <= int(gate["maximum_total_success_losses"])
        ),
        "minimum_total_net_success_gain": (
            int(overall["net_success_gain"])
            >= int(gate["minimum_total_net_success_gain"])
        ),
        "maximum_total_collision_regressions": (
            int(overall["collision_regressions"])
            <= int(gate["maximum_total_collision_regressions"])
        ),
        "minimum_total_mean_goal_distance_improvement": (
            float(overall["mean_goal_distance_improvement_m"])
            >= float(
                gate["minimum_total_mean_goal_distance_improvement_m"]
            )
        ),
        "minimum_corridor_net_success_gain": (
            int(by_scene["narrow_corridor"]["net_success_gain"])
            >= int(gate["minimum_corridor_net_success_gain"])
        ),
        "maximum_clean_success_losses": (
            int(by_scene["clean_single_obstacle"]["success_losses"])
            <= int(gate["maximum_clean_success_losses"])
        ),
        "maximum_u_trap_success_losses": (
            int(by_scene["u_trap_long_board"]["success_losses"])
            <= int(gate["maximum_u_trap_success_losses"])
        ),
        "frozen_base_correction_exact": (
            maximum_base_correction
            <= float(gate["maximum_frozen_base_correction_gate_alpha"])
        ),
    }
    result = {
        "study_label": str(design.get("study_label", "L77")),
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "artifact_errors": artifact_errors,
        "expected_episodes": len(expected),
        "observed_episodes": len(episodes),
        "duplicate_episode_keys": duplicate_count,
        "overall": overall,
        "by_model_block": by_block,
        "model_block_eligibility": block_eligibility,
        "by_scene": by_scene,
        "selected_steps": [
            int(row["selected_step"]) for row in design["model_blocks"]
        ],
        "maximum_frozen_base_correction_gate_alpha": float(
            maximum_base_correction
        ),
        "thresholds": dict(gate),
        "interpretation_guard": str(design["interpretation_guard"]),
    }
    return result, episodes, effects


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result, episodes, effects = summarize(config, args.input_dir)
    (output / "l77_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "l77_episodes.csv", episodes)
    _write_csv(output / "l77_paired_effects.csv", effects)
    _write_csv(output / "l77_by_block.csv", result["by_model_block"])
    _write_csv(
        output / "l77_by_scene.csv", result["by_scene"].values()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
