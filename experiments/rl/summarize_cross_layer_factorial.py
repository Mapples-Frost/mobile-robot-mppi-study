#!/usr/bin/env python3
"""Audit, merge and evaluate the preregistered L29 development factorial."""

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
    CONDITION_SPECS,
    _protected_previous_seeds,
    _resolved_path,
    _write_csv,
)


POLICY_PAIRS = (
    ("traditional_nominal", "gated_lcb_nominal"),
    ("traditional_icode", "gated_lcb_icode"),
)
LCB_POLICY_PAIRS = (
    ("traditional_nominal", "lcb_nominal"),
    ("traditional_icode", "lcb_icode"),
)
ICODE_PAIRS = (
    ("traditional_nominal", "traditional_icode"),
    ("lcb_nominal", "lcb_icode"),
    ("gated_lcb_nominal", "gated_lcb_icode"),
)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    text = str(value).strip().lower()
    if text in ("true", "yes", "on"):
        return True
    if text in ("false", "no", "off", ""):
        return False
    try:
        return bool(float(text))
    except ValueError:
        raise ValueError("cannot parse boolean value %r" % value)


def _key(row, include_condition=True):
    values = (
        int(row["model_block"]),
        row["scene"],
        row["physics_domain"],
        int(row["episode_seed"]),
    )
    return values + ((row["condition"],) if include_condition else ())


def _expected_keys(config):
    design = config["rl"]["cross_layer_factorial"]
    scenes = [str(load_yaml(_resolved_path(item["path"]))["scene"]["name"]) for item in design["scenes"]]
    domains = [str(item["name"]) for item in design["physics_domains"]]
    conditions = [str(item) for item in design["conditions"]]
    seeds = [int(item) for item in design["development_episode_seeds"]]
    blocks = range(len(design["model_blocks"]))
    return {
        (block, scene, domain, seed, condition)
        for block in blocks
        for scene in scenes
        for domain in domains
        for seed in seeds
        for condition in conditions
    }


def _lookup(rows):
    output = {}
    duplicates = []
    for row in rows:
        key = _key(row)
        if key in output:
            duplicates.append(key)
        output[key] = row
    return output, duplicates


def _paired_effect_rows(rows):
    lookup, _ = _lookup(rows)
    effects = []
    bases = sorted({_key(row, include_condition=False) for row in rows})
    for block, scene, domain, seed in bases:
        prefix = (block, scene, domain, seed)
        sample = next(row for row in rows if _key(row, False) == prefix)
        for nominal, icode in ICODE_PAIRS:
            first = lookup.get(prefix + (nominal,))
            second = lookup.get(prefix + (icode,))
            if first is not None and second is not None:
                effects.append({
                    "effect": "icode_minus_nominal",
                    "policy": CONDITION_SPECS[nominal]["policy"],
                    "model_block": block,
                    "scene": scene,
                    "scene_role": sample["scene_role"],
                    "physics_domain": domain,
                    "physics_role": sample["physics_role"],
                    "episode_seed": seed,
                    "success_difference": int(_bool(second["success"])) - int(_bool(first["success"])),
                    "collision_difference": int(_bool(second["collision"])) - int(_bool(first["collision"])),
                    "final_distance_improvement_m": float(first["final_goal_distance"]) - float(second["final_goal_distance"]),
                })
        for effect_name, pairs in (
            ("lcb_minus_traditional", LCB_POLICY_PAIRS),
            ("gated_minus_traditional", POLICY_PAIRS),
        ):
            for traditional, learned in pairs:
                first = lookup.get(prefix + (traditional,))
                second = lookup.get(prefix + (learned,))
                if first is not None and second is not None:
                    effects.append({
                        "effect": effect_name,
                        "residual": CONDITION_SPECS[traditional]["residual"],
                        "model_block": block,
                        "scene": scene,
                        "scene_role": sample["scene_role"],
                        "physics_domain": domain,
                        "physics_role": sample["physics_role"],
                        "episode_seed": seed,
                        "success_difference": int(_bool(second["success"])) - int(_bool(first["success"])),
                        "collision_difference": int(_bool(second["collision"])) - int(_bool(first["collision"])),
                        "final_distance_improvement_m": float(first["final_goal_distance"]) - float(second["final_goal_distance"]),
                    })
    return effects


def _clean_fallback_audit(steps):
    clean = [row for row in steps if row["scene_role"] == "clean"]
    lookup = {}
    for row in clean:
        key = _key(row, include_condition=False) + (
            row["condition"], int(row["step"]),
        )
        lookup[key] = row
    numeric = (
        "goal_distance", "collision", "executed_v", "executed_omega",
        "safety_override", "rl_gate_alpha",
    )
    mismatches = 0
    missing = 0
    maximum = {name: 0.0 for name in numeric}
    comparisons = 0
    episode_keys = sorted({_key(row, include_condition=False) for row in clean})
    for episode_key in episode_keys:
        for traditional, gated in POLICY_PAIRS:
            traditional_steps = {
                int(row["step"]) for row in clean
                if _key(row, include_condition=False) == episode_key
                and row["condition"] == traditional
            }
            gated_steps = {
                int(row["step"]) for row in clean
                if _key(row, include_condition=False) == episode_key
                and row["condition"] == gated
            }
            for step in sorted(traditional_steps.union(gated_steps)):
                first = lookup.get(episode_key + (traditional, step))
                second = lookup.get(episode_key + (gated, step))
                if first is None or second is None:
                    missing += 1
                    continue
                comparisons += 1
                for name in numeric:
                    difference = abs(float(first[name]) - float(second[name]))
                    maximum[name] = max(maximum[name], difference)
                    mismatches += int(difference != 0.0)
                mismatches += int(first["safety_reason"] != second["safety_reason"])
    return {
        "comparisons": comparisons,
        "missing_step_pairs": missing,
        "mismatch_fields": mismatches,
        "maximum_absolute_difference": maximum,
        "exact": missing == 0 and mismatches == 0,
    }


def _nested_bootstrap(values, seed=20260792, replicates=20000):
    """Two-stage block/episode bootstrap for one mean paired effect."""

    grouped = defaultdict(lambda: defaultdict(list))
    for row in values:
        grouped[int(row["model_block"])][int(row["episode_seed"])].append(
            float(row["value"])
        )
    if not grouped:
        return [None, None]
    block_values = {
        block: {episode: float(np.mean(items)) for episode, items in episodes.items()}
        for block, episodes in grouped.items()
    }
    blocks = sorted(block_values)
    rng = np.random.RandomState(int(seed))
    output = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sampled = []
        for block in selected_blocks:
            episodes = sorted(block_values[int(block)])
            selected_episodes = rng.choice(
                episodes, size=len(episodes), replace=True
            )
            sampled.extend(
                block_values[int(block)][int(episode)]
                for episode in selected_episodes
            )
        output[index] = float(np.mean(sampled))
    return [
        float(np.percentile(output, 2.5)),
        float(np.percentile(output, 97.5)),
    ]


def _condition_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene_role"], row["physics_role"], row["condition"])].append(row)
    output = []
    for (scene_role, physics_role, condition), values in sorted(groups.items()):
        clearance = [
            float(item["minimum_clearance"])
            for item in values
            if str(item.get("minimum_clearance", "")).strip().lower()
            not in ("", "none", "nan")
        ]
        output.append({
            "scene_role": scene_role,
            "physics_role": physics_role,
            "condition": condition,
            "episodes": len(values),
            "model_blocks": len({int(item["model_block"]) for item in values}),
            "success_rate": float(np.mean([_bool(item["success"]) for item in values])),
            "collision_rate": float(np.mean([_bool(item["collision"]) for item in values])),
            "final_goal_distance_mean": float(np.mean([float(item["final_goal_distance"]) for item in values])),
            "minimum_clearance_mean": (
                float(np.mean(clearance)) if clearance else None
            ),
            "planner_compute_ms_mean": float(np.mean([float(item["planner_compute_ms_mean"]) for item in values])),
            "rl_gate_alpha_mean": float(np.mean([float(item["rl_gate_alpha_mean"]) for item in values])),
        })
    return output


def _combined_ranking(rows):
    blocking = [row for row in rows if row["scene_role"] in ("static_blocking", "dynamic_crossing")]
    groups = defaultdict(lambda: defaultdict(list))
    for row in blocking:
        groups[(row["scene_role"], row["physics_role"])][row["condition"]].append(row)
    output = []
    for (scene_role, physics_role), methods in sorted(groups.items()):
        success = {
            name: sum(_bool(item["success"]) for item in values)
            for name, values in methods.items()
        }
        distances = {
            name: float(np.mean([float(item["final_goal_distance"]) for item in values]))
            for name, values in methods.items()
        }
        best_success = max(success.values())
        combined = "gated_lcb_icode"
        output.append({
            "scene_role": scene_role,
            "physics_role": physics_role,
            "combined_successes": int(success[combined]),
            "best_successes": int(best_success),
            "combined_best_or_tied": bool(success[combined] == best_success),
            "combined_final_distance_mean": distances[combined],
            "best_success_methods": sorted(
                name for name, value in success.items() if value == best_success
            ),
        })
    return output


def _development_gate(config, audit, fallback, effects, rankings):
    design = config["rl"]["cross_layer_factorial"]
    unseen_icode = [
        row for row in effects
        if row["effect"] == "icode_minus_nominal" and row["physics_role"] == "unseen"
    ]
    static_gated = [
        row for row in effects
        if row["effect"] == "gated_minus_traditional" and row["scene_role"] == "static_blocking"
    ]
    dynamic_gated = [
        row for row in effects
        if row["effect"] == "gated_minus_traditional" and row["scene_role"] == "dynamic_crossing"
    ]
    learned_policy_effects = [
        row for row in effects
        if row["effect"] in ("lcb_minus_traditional", "gated_minus_traditional")
    ]
    collision_regressions = sum(
        int(row["collision_difference"] > 0) for row in learned_policy_effects
    )
    unseen_improvement = float(np.mean([
        row["final_distance_improvement_m"] for row in unseen_icode
    ]))
    static_success_gain = int(sum(row["success_difference"] for row in static_gated))
    dynamic_success_gain = int(sum(row["success_difference"] for row in dynamic_gated))
    combined_best = int(sum(row["combined_best_or_tied"] for row in rankings))
    required = {
        "complete": audit["missing_episode_keys"] <= int(design["maximum_missing_episode_keys"]) and audit["unexpected_episode_keys"] == 0,
        "no_duplicates": audit["duplicate_episode_keys"] <= int(design["maximum_duplicate_episode_keys"]),
        "protected_seeds_untouched": not audit["protected_seeds_used"],
        "clean_fallback_exact": fallback["mismatch_fields"] <= int(design["maximum_clean_fallback_step_mismatches"]) and fallback["missing_step_pairs"] == 0,
        "no_collision_regression": collision_regressions <= int(design["maximum_collision_regressions_vs_traditional"]),
    }
    efficacy = {
        "unseen_icode_distance": unseen_improvement >= float(design["minimum_unseen_icode_distance_improvement_m"]),
        "static_gated_success": static_success_gain >= int(design["minimum_static_gated_net_success_gain"]),
        "dynamic_gated_success": dynamic_success_gain >= int(design["minimum_dynamic_gated_net_success_gain"]),
        "combined_best_strata": combined_best >= int(design["minimum_combined_best_or_tied_blocking_strata"]),
    }
    passed = all(required.values()) and sum(efficacy.values()) >= 3
    unseen_bootstrap = _nested_bootstrap([
        {"model_block": row["model_block"], "episode_seed": row["episode_seed"], "value": row["final_distance_improvement_m"]}
        for row in unseen_icode
    ])
    return {
        "decision": "development_pass" if passed else "development_fail",
        "passed": bool(passed),
        "required_checks": required,
        "efficacy_checks": efficacy,
        "efficacy_checks_passed": int(sum(efficacy.values())),
        "estimands": {
            "unseen_icode_distance_improvement_m": unseen_improvement,
            "unseen_icode_distance_improvement_ci95": unseen_bootstrap,
            "static_gated_net_success_gain": static_success_gain,
            "dynamic_gated_net_success_gain": dynamic_success_gain,
            "collision_regressions_vs_traditional": collision_regressions,
            "combined_best_or_tied_blocking_strata": combined_best,
        },
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260792,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config = load_yaml(_resolved_path(args.config))
    episodes, steps, metadata = [], [], []
    for value in args.input_dir:
        directory = _resolved_path(value)
        episodes.extend(_read_csv(directory / "episodes.csv"))
        steps.extend(_read_csv(directory / "factorial_steps.csv"))
        metadata.append(json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
    lookup, duplicates = _lookup(episodes)
    expected = _expected_keys(config)
    observed = set(lookup)
    selected_seeds = {int(row["episode_seed"]) for row in episodes}
    design = config["rl"]["cross_layer_factorial"]
    protected = _protected_previous_seeds().union(
        int(value) for value in design["sealed_confirmation_episode_seeds"]
    )
    audit = {
        "expected_episodes": len(expected),
        "observed_episodes": len(episodes),
        "unique_episode_keys": len(observed),
        "missing_episode_keys": len(expected - observed),
        "unexpected_episode_keys": len(observed - expected),
        "duplicate_episode_keys": len(duplicates),
        "missing_key_examples": [list(value) for value in sorted(expected - observed)[:10]],
        "unexpected_key_examples": [list(value) for value in sorted(observed - expected)[:10]],
        "protected_seeds_used": sorted(protected.intersection(selected_seeds)),
        "model_blocks": sorted({int(row["model_block"]) for row in episodes}),
        "metadata_blocks": sorted(int(item["model_block"]) for item in metadata),
    }
    effects = _paired_effect_rows(episodes)
    fallback = _clean_fallback_audit(steps)
    rankings = _combined_ranking(episodes)
    summary = _condition_summary(episodes)
    gate = _development_gate(config, audit, fallback, effects, rankings)
    gate["clean_fallback"] = fallback
    gate["blocking_rankings"] = rankings

    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "factorial_steps.csv", steps)
    _write_csv(output / "condition_summary.csv", summary)
    _write_csv(output / "paired_effects.csv", effects)
    _write_csv(output / "blocking_rankings.csv", rankings)
    for name, value in (
        ("audit.json", audit),
        ("development_gate.json", gate),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    print(json.dumps({"audit": audit, "gate": gate}, indent=2, sort_keys=True))
    return 0 if audit["unexpected_episode_keys"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
