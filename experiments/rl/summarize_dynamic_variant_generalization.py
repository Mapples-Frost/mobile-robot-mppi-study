#!/usr/bin/env python3
"""Audit and summarize the preregistered L31 development factorial."""

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
    _protected_previous_seeds,
    _resolved_path,
    _write_csv,
)
from experiments.rl.summarize_cross_layer_factorial import (
    _bool,
    _nested_bootstrap,
)


PRIMARY = "temporal_gated_lcb_icode"
COMPARATORS = (
    "traditional_icode",
    "lcb_icode",
    "gated_lcb_icode",
    "temporal_gated_lcb_nominal",
)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row, condition=True):
    value = (
        int(row["model_block"]),
        str(row["scene"]),
        str(row["physics_domain"]),
        int(row["episode_seed"]),
    )
    return value + ((str(row["condition"]),) if condition else ())


def _expected_keys(config):
    design = config["rl"]["cross_layer_factorial"]
    scenes = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    return {
        (block, scene, str(domain["name"]), int(seed), str(condition))
        for block in range(len(design["model_blocks"]))
        for scene in scenes
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in design["conditions"]
    }


def _audit(config, rows, metadata):
    expected = _expected_keys(config)
    observed = [_key(row) for row in rows]
    observed_set = set(observed)
    design = config["rl"]["cross_layer_factorial"]
    protected = _protected_previous_seeds(
        design.get("protected_config_paths", ())
    )
    sealed = {int(value) for value in design["sealed_confirmation_episode_seeds"]}
    used = {int(row["episode_seed"]) for row in rows}
    invalid = 0
    for row in rows:
        try:
            values = (
                float(row["final_goal_distance"]),
                float(row["planner_compute_ms_mean"]),
                float(row.get("rl_gate_alpha_mean", 0.0)),
            )
            _bool(row["success"])
            _bool(row["collision"])
            invalid += int(not all(math.isfinite(value) for value in values))
        except (KeyError, TypeError, ValueError):
            invalid += 1
    wrong_checkpoint_blocks = 0
    for item in metadata:
        block = int(item["model_block"])
        expected_block = design["model_blocks"][block]
        wrong_checkpoint_blocks += int(
            int(item["rl_training_seed"]) != int(expected_block["rl_seed"])
            or int(item["icode_training_seed"]) != int(expected_block["icode_seed"])
        )
    return {
        "expected_episode_keys": len(expected),
        "observed_episode_rows": len(rows),
        "observed_unique_episode_keys": len(observed_set),
        "missing_episode_keys": len(expected - observed_set),
        "unexpected_episode_keys": len(observed_set - expected),
        "duplicate_episode_keys": len(observed) - len(observed_set),
        "protected_seeds_used": sorted(used.intersection(protected)),
        "sealed_confirmation_seeds_used": sorted(used.intersection(sealed)),
        "invalid_metric_rows": invalid,
        "wrong_checkpoint_blocks": wrong_checkpoint_blocks,
        "model_block_metadata_files": len(metadata),
    }


def _condition_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene_role"], row["physics_role"], row["condition"])].append(row)
    output = []
    for (scene, physics, condition), values in sorted(groups.items()):
        output.append({
            "scene_role": scene,
            "physics_role": physics,
            "condition": condition,
            "episodes": len(values),
            "model_blocks": len({int(item["model_block"]) for item in values}),
            "successes": int(sum(_bool(item["success"]) for item in values)),
            "success_rate": float(np.mean([_bool(item["success"]) for item in values])),
            "collisions": int(sum(_bool(item["collision"]) for item in values)),
            "collision_rate": float(np.mean([_bool(item["collision"]) for item in values])),
            "final_goal_distance_mean": float(np.mean([
                float(item["final_goal_distance"]) for item in values
            ])),
            "planner_compute_ms_mean": float(np.mean([
                float(item["planner_compute_ms_mean"]) for item in values
            ])),
            "rl_gate_alpha_mean": float(np.mean([
                float(item.get("rl_gate_alpha_mean", 0.0)) for item in values
            ])),
            "temporal_closing_alpha_mean": float(np.mean([
                float(item.get("rl_temporal_closing_gate_alpha_mean", 0.0))
                for item in values
            ])),
        })
    return output


def _paired_effects(rows):
    lookup = {_key(row): row for row in rows}
    effects = []
    for key, primary in lookup.items():
        if key[-1] != PRIMARY:
            continue
        for comparator_name in COMPARATORS:
            comparator = lookup.get(key[:-1] + (comparator_name,))
            if comparator is None:
                continue
            effects.append({
                "comparator": comparator_name,
                "model_block": key[0],
                "scene": key[1],
                "scene_role": primary["scene_role"],
                "physics_domain": key[2],
                "physics_role": primary["physics_role"],
                "episode_seed": key[3],
                "success_difference": (
                    int(_bool(primary["success"]))
                    - int(_bool(comparator["success"]))
                ),
                "collision_difference": (
                    int(_bool(primary["collision"]))
                    - int(_bool(comparator["collision"]))
                ),
                "final_distance_improvement_m": (
                    float(comparator["final_goal_distance"])
                    - float(primary["final_goal_distance"])
                ),
            })
    return effects


def _paired_summary(effects):
    output = []
    for comparator in COMPARATORS:
        values = [row for row in effects if row["comparator"] == comparator]
        bootstrap_values = [dict(row, value=row["final_distance_improvement_m"]) for row in values]
        ci = _nested_bootstrap(bootstrap_values, seed=20260795, replicates=20000)
        output.append({
            "primary": PRIMARY,
            "comparator": comparator,
            "pairs": len(values),
            "net_success_gain": int(sum(row["success_difference"] for row in values)),
            "net_collision_gain": int(sum(row["collision_difference"] for row in values)),
            "final_distance_improvement_mean_m": float(np.mean([
                row["final_distance_improvement_m"] for row in values
            ])) if values else None,
            "final_distance_improvement_ci95_low_m": ci[0],
            "final_distance_improvement_ci95_high_m": ci[1],
        })
    return output


def _strata_rankings(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["scene_role"], row["physics_role"])][row["condition"]].append(row)
    output = []
    for (scene, physics), methods in sorted(groups.items()):
        successes = {
            name: int(sum(_bool(item["success"]) for item in values))
            for name, values in methods.items()
        }
        collisions = {
            name: int(sum(_bool(item["collision"]) for item in values))
            for name, values in methods.items()
        }
        distances = {
            name: float(np.mean([float(item["final_goal_distance"]) for item in values]))
            for name, values in methods.items()
        }
        best = max(successes.values())
        output.append({
            "scene_role": scene,
            "physics_role": physics,
            "primary_successes": successes[PRIMARY],
            "primary_collisions": collisions[PRIMARY],
            "primary_final_goal_distance_mean": distances[PRIMARY],
            "best_successes": best,
            "primary_best_or_tied": successes[PRIMARY] == best,
            "best_success_methods": sorted(
                name for name, value in successes.items() if value == best
            ),
        })
    return output


def _development_gate(config, audit, rows, effects, rankings):
    design = config["rl"]["cross_layer_factorial"]
    primary = [row for row in rows if row["condition"] == PRIMARY]
    primary_successes = int(sum(_bool(row["success"]) for row in primary))
    primary_collisions = int(sum(_bool(row["collision"]) for row in primary))
    best_strata = int(sum(row["primary_best_or_tied"] for row in rankings))
    spatial = [row for row in effects if row["comparator"] == "gated_lcb_icode"]
    nominal = [
        row for row in effects
        if row["comparator"] == "temporal_gated_lcb_nominal"
    ]
    required = {
        "complete_unique_matrix": (
            audit["missing_episode_keys"] == 0
            and audit["unexpected_episode_keys"] == 0
            and audit["duplicate_episode_keys"] == 0
            and audit["observed_episode_rows"] == audit["expected_episode_keys"]
        ),
        "protected_and_sealed_seeds_untouched": (
            not audit["protected_seeds_used"]
            and not audit["sealed_confirmation_seeds_used"]
        ),
        "primary_collision_free": (
            len(primary) > 0
            and primary_collisions <= int(design["maximum_primary_collisions"])
        ),
        "finite_valid_metrics_and_checkpoints": (
            audit["invalid_metric_rows"] == 0
            and audit["wrong_checkpoint_blocks"] == 0
            and audit["model_block_metadata_files"]
            == len(design["model_blocks"])
        ),
    }
    efficacy = {
        "primary_success_at_least_90_percent": (
            primary_successes >= int(design["minimum_primary_successes"])
        ),
        "every_stratum_success_at_least_80_percent": (
            len(rankings) == 8
            and all(
                row["primary_successes"]
                >= int(design["minimum_primary_successes_per_stratum"])
                for row in rankings
            )
        ),
        "primary_best_or_tied_in_at_least_six_strata": (
            len(rankings) == 8
            and best_strata
            >= int(design["minimum_primary_best_or_tied_strata"])
        ),
        "net_success_gain_vs_spatial_at_least_24": (
            len(spatial) == len(primary)
            and sum(row["success_difference"] for row in spatial)
            >= int(design["minimum_primary_net_success_gain_vs_spatial"])
        ),
        "icode_noninferior_to_temporal_nominal": (
            len(nominal) == len(primary)
            and sum(row["success_difference"] for row in nominal)
            >= int(design["minimum_icode_net_success_gain_vs_temporal_nominal"])
            and sum(row["collision_difference"] for row in nominal)
            <= int(design["maximum_icode_net_collision_gain_vs_temporal_nominal"])
        ),
    }
    efficacy_passes = int(sum(bool(value) for value in efficacy.values()))
    passed = (
        all(required.values())
        and efficacy_passes >= int(design["minimum_efficacy_checks_required"])
    )
    return {
        "decision": "development_pass" if passed else "development_fail",
        "passed": bool(passed),
        "required_checks": required,
        "efficacy_checks": efficacy,
        "efficacy_checks_passed": efficacy_passes,
        "efficacy_checks_required": int(design["minimum_efficacy_checks_required"]),
        "estimands": {
            "primary_episodes": len(primary),
            "primary_successes": primary_successes,
            "primary_collisions": primary_collisions,
            "primary_best_or_tied_strata": best_strata,
            "net_success_gain_vs_spatial": int(sum(
                row["success_difference"] for row in spatial
            )),
            "net_success_gain_vs_temporal_nominal": int(sum(
                row["success_difference"] for row in nominal
            )),
            "net_collision_gain_vs_temporal_nominal": int(sum(
                row["collision_difference"] for row in nominal
            )),
        },
        "interpretation": (
            "Development-only gate. Passing permits a later sealed confirmation; "
            "it is not confirmatory evidence by itself."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config = load_yaml(_resolved_path(args.config))
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    steps = []
    metadata = []
    for value in args.run_dir:
        directory = _resolved_path(value)
        rows.extend(_read_csv(directory / "episodes.csv"))
        steps.extend(_read_csv(directory / "factorial_steps.csv"))
        metadata.append(json.loads((directory / "metadata.json").read_text(encoding="utf-8")))

    audit = _audit(config, rows, metadata)
    effects = _paired_effects(rows)
    condition_summary = _condition_summary(rows)
    paired_summary = _paired_summary(effects)
    rankings = _strata_rankings(rows)
    gate = _development_gate(config, audit, rows, effects, rankings)

    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "factorial_steps.csv", steps)
    _write_csv(output / "condition_summary.csv", condition_summary)
    _write_csv(output / "paired_effects.csv", effects)
    _write_csv(output / "paired_summary.csv", paired_summary)
    _write_csv(output / "strata_rankings.csv", rankings)
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "development_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"audit": audit, "gate": gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
