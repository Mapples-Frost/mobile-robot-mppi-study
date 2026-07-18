#!/usr/bin/env python3
"""Merge, audit and gate the preregistered L28 inference ablation."""

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

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.run_inference_profile_ablation import CONDITIONS
from experiments.rl.run_scene_complexity_gate_ablation import (
    _resolved_path,
    _write_csv,
)
from experiments.rl.run_scene_complexity_sample_efficiency import (
    _scene_entries,
)


BOOTSTRAP_SEED = 20260782
BOOTSTRAP_REPLICATES = 20000
REFERENCE = "reference_full_diagnostics"
CANDIDATES = CONDITIONS[1:]


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    text = str(value).strip().lower()
    if text in ("true", "yes"):
        return True
    if text in ("false", "no", ""):
        return False
    try:
        return float(text) != 0.0
    except ValueError as exc:
        raise ValueError("invalid boolean value %r" % value) from exc


def _float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s" % name)
    return value


def _episode_key(row):
    return (
        int(row["training_seed"]), row["scene"],
        int(row["episode_seed"]), row["condition"],
    )


def _step_key(row):
    return _episode_key(row) + (int(row["step"]),)


def _behavior_equivalence(episodes, steps):
    episode_lookup = {_episode_key(row): row for row in episodes}
    episode_groups = sorted({key[:-1] for key in episode_lookup})
    step_lookup = {_step_key(row): row for row in steps}
    step_groups = sorted({key[:-2] for key in step_lookup})
    numeric_fields = (
        "executed_v", "executed_omega", "goal_distance",
        "rl_gate_alpha", "rl_correction_advantage_gate_alpha",
        "rl_selected_consensus_lcb", "rl_target_q1_base",
        "rl_target_q2_base", "rl_target_q1_candidate",
        "rl_target_q2_candidate",
    )
    output = {}
    for candidate in CANDIDATES:
        success_mismatches = 0
        collision_mismatches = 0
        for group in episode_groups:
            reference = episode_lookup[group + (REFERENCE,)]
            optimized = episode_lookup[group + (candidate,)]
            success_mismatches += int(
                _bool(reference["success"]) != _bool(optimized["success"])
            )
            collision_mismatches += int(
                _bool(reference["collision"]) != _bool(optimized["collision"])
            )

        mismatch_count = 0
        gate_mismatches = 0
        length_mismatches = 0
        compared = 0
        max_abs = {name: 0.0 for name in numeric_fields}
        for group in step_groups:
            reference_steps = sorted(
                key[-1] for key in step_lookup
                if key[:-2] == group and key[-2] == REFERENCE
            )
            candidate_steps = sorted(
                key[-1] for key in step_lookup
                if key[:-2] == group and key[-2] == candidate
            )
            if reference_steps != candidate_steps:
                length_mismatches += 1
                continue
            for step in reference_steps:
                reference = step_lookup[group + (REFERENCE, step)]
                optimized = step_lookup[group + (candidate, step)]
                compared += 1
                for name in numeric_fields:
                    delta = abs(
                        _float(reference, name) - _float(optimized, name)
                    )
                    max_abs[name] = max(max_abs[name], delta)
                    mismatch_count += int(delta != 0.0)
                    if name in (
                        "rl_correction_advantage_gate_alpha",
                        "rl_selected_consensus_lcb",
                    ):
                        gate_mismatches += int(delta != 0.0)
                for name in ("collision", "safety_override"):
                    mismatch_count += int(
                        _bool(reference[name]) != _bool(optimized[name])
                    )
        output[candidate] = {
            "episode_success_mismatches": success_mismatches,
            "episode_collision_mismatches": collision_mismatches,
            "step_behavior_mismatches": mismatch_count,
            "selected_gate_mismatches": gate_mismatches,
            "step_length_mismatch_groups": length_mismatches,
            "steps_compared": compared,
            "max_abs_differences": max_abs,
        }
    return output


def _episode_profiles(episodes, steps):
    episode_lookup = {_episode_key(row): row for row in episodes}
    grouped = defaultdict(list)
    for row in steps:
        grouped[_episode_key(row)].append(row)
    output = []
    component_fields = (
        "profile_prior_fallback_ms", "profile_prior_encode_normalize_ms",
        "profile_prior_gate_features_ms", "profile_prior_actor_ms",
        "profile_prior_advantage_ms", "profile_prior_decoder_ms",
        "profile_prior_outer_gate_ms", "profile_prior_total_ms",
        "profile_planner_prior_ms", "profile_mppi_sampling_ms",
        "profile_mppi_batch_rollout_ms", "profile_mppi_cost_ms",
        "profile_mppi_weighting_update_ms",
        "profile_mppi_final_rollout_ms", "profile_mppi_solve_total_ms",
    )
    for key, rows in sorted(grouped.items()):
        episode = episode_lookup[key]
        active = [
            row for row in rows
            if not _bool(row["rl_learned_inference_skipped"])
        ]
        record = {
            "training_seed": key[0],
            "scene": key[1],
            "episode_seed": key[2],
            "condition": key[3],
            "scene_role": episode["scene_role"],
            "steps": len(rows),
            "active_steps": len(active),
            "planner_compute_ms": _float(
                episode, "planner_compute_ms_mean"
            ),
            "unselected_diagnostics_fraction_active": (
                float(np.mean([
                    _float(
                        row,
                        "rl_unselected_critic_diagnostics_computed",
                    )
                    for row in active
                ])) if active else 0.0
            ),
        }
        for name in component_fields:
            record[name + "_all"] = float(np.mean([
                _float(row, name) for row in rows
            ]))
            record[name + "_active"] = (
                float(np.mean([_float(row, name) for row in active]))
                if active else 0.0
            )
        output.append(record)
    return output


def _pair_profiles(profiles):
    lookup = {
        (
            int(row["training_seed"]), row["scene"],
            int(row["episode_seed"]), row["condition"],
        ): row
        for row in profiles
    }
    groups = sorted({key[:-1] for key in lookup})
    output = []
    for training_seed, scene, episode_seed in groups:
        reference = lookup[
            (training_seed, scene, episode_seed, REFERENCE)
        ]
        for candidate in CANDIDATES:
            optimized = lookup[
                (training_seed, scene, episode_seed, candidate)
            ]
            output.append({
                "training_seed": training_seed,
                "scene": scene,
                "scene_role": reference["scene_role"],
                "episode_seed": episode_seed,
                "candidate": candidate,
                "reference_planner_ms": reference["planner_compute_ms"],
                "candidate_planner_ms": optimized["planner_compute_ms"],
                "reference_active_prior_ms": reference[
                    "profile_prior_total_ms_active"
                ],
                "candidate_active_prior_ms": optimized[
                    "profile_prior_total_ms_active"
                ],
            })
    return output


def _nested_reduction_interval(rows, reference_field, candidate_field):
    training_ids = sorted({int(row["training_seed"]) for row in rows})
    episode_ids = sorted({int(row["episode_seed"]) for row in rows})
    scenes = sorted({row["scene"] for row in rows})
    if len(training_ids) < 2:
        raise ValueError("L28 bootstrap requires multiple checkpoints")
    lookup = {
        (int(row["training_seed"]), int(row["episode_seed"]), row["scene"]): row
        for row in rows
    }
    expected = len(training_ids) * len(episode_ids) * len(scenes)
    if len(lookup) != expected:
        raise ValueError("L28 bootstrap requires a balanced design")
    shape = (len(training_ids), len(episode_ids), len(scenes))
    reference = np.empty(shape, dtype=np.float64)
    candidate = np.empty(shape, dtype=np.float64)
    for ti, training_seed in enumerate(training_ids):
        for ei, episode_seed in enumerate(episode_ids):
            for si, scene in enumerate(scenes):
                row = lookup[(training_seed, episode_seed, scene)]
                reference[ti, ei, si] = float(row[reference_field])
                candidate[ti, ei, si] = float(row[candidate_field])
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    selected_training = rng.randint(
        0, len(training_ids),
        size=(BOOTSTRAP_REPLICATES, len(training_ids)),
    )
    selected_episodes = rng.randint(
        0, len(episode_ids),
        size=(BOOTSTRAP_REPLICATES, len(training_ids), len(episode_ids)),
    )
    expanded_training = np.broadcast_to(
        selected_training[:, :, np.newaxis], selected_episodes.shape
    )
    reference_means = reference[
        expanded_training, selected_episodes
    ].mean(axis=(1, 2, 3))
    candidate_means = candidate[
        expanded_training, selected_episodes
    ].mean(axis=(1, 2, 3))
    reductions = 1.0 - candidate_means / reference_means
    return (
        float(np.quantile(reductions, 0.025)),
        float(np.quantile(reductions, 0.975)),
    )


def _timing(pairs):
    result = {}
    for candidate in CANDIDATES:
        rows = [
            row for row in pairs
            if row["candidate"] == candidate
            and row["scene_role"] == "blocking"
        ]
        active_rows = [
            row for row in rows
            if row["reference_active_prior_ms"] > 0.0
        ]
        if not active_rows:
            raise ValueError("L28 has no active-gate timing rows")
        reference_prior = float(np.mean([
            row["reference_active_prior_ms"] for row in active_rows
        ]))
        candidate_prior = float(np.mean([
            row["candidate_active_prior_ms"] for row in active_rows
        ]))
        reference_planner = float(np.mean([
            row["reference_planner_ms"] for row in rows
        ]))
        candidate_planner = float(np.mean([
            row["candidate_planner_ms"] for row in rows
        ]))
        result[candidate] = {
            "pairs": len(rows),
            "active_prior_pairs": len(active_rows),
            "reference_active_prior_ms": reference_prior,
            "candidate_active_prior_ms": candidate_prior,
            "active_prior_reduction_fraction": float(
                1.0 - candidate_prior / reference_prior
            ),
            "reference_planner_ms": reference_planner,
            "candidate_planner_ms": candidate_planner,
            "blocking_planner_reduction_fraction": float(
                1.0 - candidate_planner / reference_planner
            ),
            "blocking_planner_reduction_ci95": (
                _nested_reduction_interval(
                    rows,
                    "reference_planner_ms",
                    "candidate_planner_ms",
                )
            ),
        }
    return result


def _component_summary(profiles):
    fields = sorted(
        key for key in profiles[0]
        if key.endswith("_active") or key.endswith("_all")
    )
    output = []
    for role in ("control", "blocking"):
        for condition in CONDITIONS:
            rows = [
                row for row in profiles
                if row["scene_role"] == role
                and row["condition"] == condition
            ]
            record = {
                "scene_role": role,
                "condition": condition,
                "episodes": len(rows),
                "steps": int(sum(row["steps"] for row in rows)),
                "active_steps": int(sum(row["active_steps"] for row in rows)),
                "planner_compute_ms": float(np.mean([
                    row["planner_compute_ms"] for row in rows
                ])),
                "unselected_diagnostics_fraction_active": float(np.mean([
                    row["unselected_diagnostics_fraction_active"]
                    for row in rows
                ])),
            }
            record.update({
                name: float(np.mean([row[name] for row in rows]))
                for name in fields
            })
            output.append(record)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    base = load_yaml(_resolved_path(args.config))
    design = base["rl"]["inference_profile_ablation"]
    directories = [
        _resolved_path(value.strip())
        for value in args.input_dirs.split(",") if value.strip()
    ]
    episodes, steps, metadata = [], [], []
    for directory in directories:
        episodes.extend(_read(directory / "episodes.csv"))
        steps.extend(_read(directory / "profile_steps.csv"))
        metadata.append(json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        ))

    keys = [_episode_key(row) for row in episodes]
    training_seeds = sorted({int(row["training_seed"]) for row in episodes})
    expected = (
        len(training_seeds)
        * len(design["development_episode_seeds"])
        * len(_scene_entries(base))
        * len(CONDITIONS)
    )
    audit = {
        "expected_episodes": expected,
        "observed_episodes": len(episodes),
        "observed_steps": len(steps),
        "duplicate_episode_keys": len(keys) - len(set(keys)),
        "protected_seeds_used": sorted({
            value for item in metadata
            for value in item["protected_seeds_used"]
        }),
        "complete": (
            len(episodes) == expected and len(keys) == len(set(keys))
        ),
    }
    behavior = _behavior_equivalence(episodes, steps)
    profiles = _episode_profiles(episodes, steps)
    pairs = _pair_profiles(profiles)
    timing = _timing(pairs)
    components = _component_summary(profiles)
    selected_only = timing["selected_critic_only"]
    combined = timing["selected_critic_base_reuse"]
    checks = {
        "complete": audit["complete"],
        "protected_seeds_untouched": audit["protected_seeds_used"] == [],
        "selected_only_behavior_exact": (
            behavior["selected_critic_only"]["step_behavior_mismatches"]
            <= int(design["maximum_behavior_mismatch_steps"])
            and behavior["selected_critic_only"][
                "step_length_mismatch_groups"
            ] == 0
        ),
        "combined_behavior_exact": (
            behavior["selected_critic_base_reuse"][
                "step_behavior_mismatches"
            ] <= int(design["maximum_behavior_mismatch_steps"])
            and behavior["selected_critic_base_reuse"][
                "step_length_mismatch_groups"
            ] == 0
        ),
        "episode_success_exact": all(
            behavior[name]["episode_success_mismatches"]
            <= int(design["maximum_success_mismatches"])
            for name in CANDIDATES
        ),
        "episode_collision_exact": all(
            behavior[name]["episode_collision_mismatches"]
            <= int(design["maximum_collision_mismatches"])
            for name in CANDIDATES
        ),
        "selected_gate_exact": all(
            behavior[name]["selected_gate_mismatches"]
            <= int(design["maximum_selected_gate_mismatches"])
            for name in CANDIDATES
        ),
        "selected_only_prior_reduction": (
            selected_only["active_prior_reduction_fraction"]
            >= float(
                design[
                    "minimum_selected_only_prior_reduction_fraction"
                ]
            )
        ),
        "combined_prior_reduction": (
            combined["active_prior_reduction_fraction"]
            >= float(design["minimum_combined_prior_reduction_fraction"])
        ),
        "combined_blocking_planner_reduction": (
            combined["blocking_planner_reduction_fraction"]
            >= float(
                design[
                    "minimum_combined_blocking_planner_reduction_fraction"
                ]
            )
        ),
        "combined_reduction_ci_positive": (
            combined["blocking_planner_reduction_ci95"][0]
            > float(design["minimum_combined_reduction_ci95_low"])
        ),
    }
    passed = all(checks.values())
    gate = {
        "passed": passed,
        "decision": "development_pass" if passed else "development_fail",
        "checks": checks,
        "behavior": behavior,
        "timing": timing,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "profile_steps.csv", steps)
    _write_csv(output / "episode_profiles.csv", profiles)
    _write_csv(output / "paired_timing.csv", pairs)
    _write_csv(output / "component_summary.csv", components)
    for name, data in (("audit.json", audit), ("development_gate.json", gate)):
        (output / name).write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps({"audit": audit, "development_gate": gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
