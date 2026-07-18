#!/usr/bin/env python3
"""Audit L30 temporal-closing remediation against frozen L29 comparators."""

import argparse
import csv
import json
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


RISK_RENAME = {
    "gated_lcb_nominal": "temporal_gated_lcb_nominal",
    "gated_lcb_icode": "temporal_gated_lcb_icode",
}
RISK_TO_TRADITIONAL = {
    "temporal_gated_lcb_nominal": "traditional_nominal",
    "temporal_gated_lcb_icode": "traditional_icode",
}
RISK_TO_LCB = {
    "temporal_gated_lcb_nominal": "lcb_nominal",
    "temporal_gated_lcb_icode": "lcb_icode",
}
RISK_TO_SPATIAL = {
    "temporal_gated_lcb_nominal": "gated_lcb_nominal",
    "temporal_gated_lcb_icode": "gated_lcb_icode",
}


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _episode_key(row, condition=None):
    return (
        int(row["model_block"]),
        row["scene"],
        row["physics_domain"],
        int(row["episode_seed"]),
        row["condition"] if condition is None else condition,
    )


def _rename_risk(rows):
    output = []
    for row in rows:
        copied = dict(row)
        copied["condition"] = RISK_RENAME[copied["condition"]]
        output.append(copied)
    return output


def _expected_risk_keys(config):
    design = config["rl"]["cross_layer_factorial"]
    scene_names = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    domains = [str(item["name"]) for item in design["physics_domains"]]
    seeds = [int(value) for value in design["development_episode_seeds"]]
    conditions = [RISK_RENAME[str(value)] for value in design["conditions"]]
    return {
        (block, scene, domain, seed, condition)
        for block in range(len(design["model_blocks"]))
        for scene in scene_names
        for domain in domains
        for seed in seeds
        for condition in conditions
    }


def _lookup(rows):
    output = {}
    duplicates = []
    for row in rows:
        key = _episode_key(row)
        if key in output:
            duplicates.append(key)
        output[key] = row
    return output, duplicates


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
            "model_blocks": len({int(value["model_block"]) for value in values}),
            "successes": int(sum(_bool(value["success"]) for value in values)),
            "success_rate": float(np.mean([_bool(value["success"]) for value in values])),
            "collisions": int(sum(_bool(value["collision"]) for value in values)),
            "collision_rate": float(np.mean([_bool(value["collision"]) for value in values])),
            "final_goal_distance_mean": float(np.mean([
                float(value["final_goal_distance"]) for value in values
            ])),
            "rl_gate_alpha_mean": float(np.mean([
                float(value["rl_gate_alpha_mean"]) for value in values
            ])),
            "temporal_closing_alpha_mean": float(np.mean([
                float(value.get("rl_temporal_closing_gate_alpha_mean", 0.0))
                for value in values
            ])),
        })
    return output


def _clean_fallback(risk_steps, baseline_steps):
    baseline = {
        _episode_key(row) + (int(row["step"]),): row
        for row in baseline_steps if row["scene_role"] == "clean"
    }
    numeric = (
        "goal_distance", "collision", "executed_v", "executed_omega",
        "safety_override",
    )
    comparisons = 0
    missing = 0
    mismatches = 0
    maximum = {name: 0.0 for name in numeric}
    maximum["rl_gate_alpha"] = 0.0
    for row in risk_steps:
        if row["scene_role"] != "clean":
            continue
        risk_condition = RISK_RENAME[row["condition"]]
        traditional = RISK_TO_TRADITIONAL[risk_condition]
        key = _episode_key(row, traditional) + (int(row["step"]),)
        match = baseline.get(key)
        if match is None:
            missing += 1
            continue
        comparisons += 1
        for name in numeric:
            difference = abs(float(row[name]) - float(match[name]))
            maximum[name] = max(maximum[name], difference)
            mismatches += int(difference != 0.0)
        gate_alpha = abs(float(row["rl_gate_alpha"]))
        maximum["rl_gate_alpha"] = max(maximum["rl_gate_alpha"], gate_alpha)
        mismatches += int(gate_alpha != 0.0)
        mismatches += int(row["safety_reason"] != match["safety_reason"])

    risk_step_keys = {
        _episode_key(row, RISK_RENAME[row["condition"]]) + (int(row["step"]),)
        for row in risk_steps if row["scene_role"] == "clean"
    }
    for key, row in baseline.items():
        if row["condition"] not in RISK_TO_TRADITIONAL.values():
            continue
        reverse = {
            value: name for name, value in RISK_TO_TRADITIONAL.items()
        }[row["condition"]]
        risk_key = key[:4] + (reverse,) + key[5:]
        if risk_key not in risk_step_keys:
            missing += 1
    return {
        "comparisons": comparisons,
        "missing_step_pairs": missing,
        "mismatch_fields": mismatches,
        "maximum_absolute_difference": maximum,
        "exact": missing == 0 and mismatches == 0,
    }


def _paired_rows(risk_rows, baseline_rows):
    risk_lookup, _ = _lookup(risk_rows)
    baseline_lookup, _ = _lookup(baseline_rows)
    output = []
    for key, risk in risk_lookup.items():
        risk_condition = risk["condition"]
        for comparator_kind, mapping in (
            ("traditional", RISK_TO_TRADITIONAL),
            ("always_lcb", RISK_TO_LCB),
            ("spatial_gate", RISK_TO_SPATIAL),
        ):
            comparator_condition = mapping[risk_condition]
            comparator = baseline_lookup.get(key[:4] + (comparator_condition,))
            if comparator is None:
                continue
            output.append({
                "comparator": comparator_kind,
                "residual": "icode" if risk_condition.endswith("icode") else "nominal",
                "model_block": key[0],
                "scene": key[1],
                "scene_role": risk["scene_role"],
                "physics_domain": key[2],
                "physics_role": risk["physics_role"],
                "episode_seed": key[3],
                "success_difference": int(_bool(risk["success"])) - int(_bool(comparator["success"])),
                "collision_difference": int(_bool(risk["collision"])) - int(_bool(comparator["collision"])),
                "final_distance_improvement_m": float(comparator["final_goal_distance"]) - float(risk["final_goal_distance"]),
            })
    return output


def _risk_icode_effect(risk_rows):
    lookup, _ = _lookup(risk_rows)
    output = []
    for key, nominal in lookup.items():
        if key[-1] != "temporal_gated_lcb_nominal":
            continue
        icode = lookup.get(key[:-1] + ("temporal_gated_lcb_icode",))
        if icode is None:
            continue
        output.append({
            "model_block": key[0],
            "scene": key[1],
            "scene_role": nominal["scene_role"],
            "physics_domain": key[2],
            "physics_role": nominal["physics_role"],
            "episode_seed": key[3],
            "success_difference": int(_bool(icode["success"])) - int(_bool(nominal["success"])),
            "collision_difference": int(_bool(icode["collision"])) - int(_bool(nominal["collision"])),
            "final_distance_improvement_m": float(nominal["final_goal_distance"]) - float(icode["final_goal_distance"]),
        })
    return output


def _blocking_rankings(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["scene_role"] in ("static_blocking", "dynamic_crossing"):
            groups[(row["scene_role"], row["physics_role"])][row["condition"]].append(row)
    output = []
    for (scene, physics), methods in sorted(groups.items()):
        successes = {
            method: int(sum(_bool(value["success"]) for value in values))
            for method, values in methods.items()
        }
        best = max(successes.values())
        combined = "temporal_gated_lcb_icode"
        output.append({
            "scene_role": scene,
            "physics_role": physics,
            "combined_successes": successes[combined],
            "best_successes": best,
            "combined_best_or_tied": successes[combined] == best,
            "best_success_methods": sorted(
                method for method, value in successes.items() if value == best
            ),
        })
    return output


def _eligibility(audit, fallback, paired, rankings):
    dynamic_lcb = [
        row for row in paired
        if row["comparator"] == "always_lcb"
        and row["scene_role"] == "dynamic_crossing"
    ]
    dynamic_by_physics = []
    for physics in ("seen", "unseen"):
        values = [row for row in dynamic_lcb if row["physics_role"] == physics]
        dynamic_by_physics.append({
            "physics_role": physics,
            "pairs": len(values),
            "success_gain": int(sum(row["success_difference"] for row in values)),
            "collision_gain": int(sum(row["collision_difference"] for row in values)),
        })
    static_spatial = [
        row for row in paired
        if row["comparator"] == "spatial_gate"
        and row["scene_role"] == "static_blocking"
    ]
    static_success_gain = int(sum(
        row["success_difference"] for row in static_spatial
    ))
    best_count = int(sum(row["combined_best_or_tied"] for row in rankings))
    checks = {
        "complete_unique_development_matrix": (
            audit["missing_episode_keys"] == 0
            and audit["unexpected_episode_keys"] == 0
            and audit["duplicate_episode_keys"] == 0
        ),
        "protected_seeds_untouched": not audit["protected_seeds_used"],
        "clean_fallback_exact": bool(fallback["exact"]),
        "dynamic_collision_not_worse_than_lcb": all(
            row["pairs"] > 0 and row["collision_gain"] <= 0
            for row in dynamic_by_physics
        ),
        "dynamic_success_not_worse_than_lcb": all(
            row["pairs"] > 0 and row["success_gain"] >= 0
            for row in dynamic_by_physics
        ),
        "static_success_loss_at_most_four": (
            len(static_spatial) > 0 and static_success_gain >= -4
        ),
        "combined_best_or_tied_at_least_three_strata": (
            len(rankings) == 4 and best_count >= 3
        ),
    }
    return {
        "decision": "development_pass" if all(checks.values()) else "development_fail",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "estimands": {
            "dynamic_vs_always_lcb_by_physics": dynamic_by_physics,
            "static_success_gain_vs_spatial_gate": static_success_gain,
            "static_paired_episodes": len(static_spatial),
            "combined_best_or_tied_blocking_strata": best_count,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--risk-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config = load_yaml(_resolved_path(args.config))
    baseline_dir = _resolved_path(args.baseline_dir)
    baseline_episodes = _read_csv(baseline_dir / "episodes.csv")
    baseline_steps = _read_csv(baseline_dir / "factorial_steps.csv")
    raw_risk_episodes, raw_risk_steps, metadata = [], [], []
    for value in args.risk_dir:
        directory = _resolved_path(value)
        raw_risk_episodes.extend(_read_csv(directory / "episodes.csv"))
        raw_risk_steps.extend(_read_csv(directory / "factorial_steps.csv"))
        metadata.append(json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        ))
    risk_episodes = _rename_risk(raw_risk_episodes)
    risk_lookup, duplicates = _lookup(risk_episodes)
    expected = _expected_risk_keys(config)
    observed = set(risk_lookup)
    selected_seeds = {int(row["episode_seed"]) for row in risk_episodes}
    design = config["rl"]["cross_layer_factorial"]
    protected = _protected_previous_seeds().union(
        int(value) for value in design["sealed_confirmation_episode_seeds"]
    )
    audit = {
        "expected_episodes": len(expected),
        "observed_episodes": len(risk_episodes),
        "unique_episode_keys": len(observed),
        "missing_episode_keys": len(expected - observed),
        "unexpected_episode_keys": len(observed - expected),
        "duplicate_episode_keys": len(duplicates),
        "protected_seeds_used": sorted(protected.intersection(selected_seeds)),
        "model_blocks": sorted({int(row["model_block"]) for row in risk_episodes}),
        "metadata_blocks": sorted(int(value["model_block"]) for value in metadata),
    }
    fallback = _clean_fallback(raw_risk_steps, baseline_steps)
    paired = _paired_rows(risk_episodes, baseline_episodes)
    icode = _risk_icode_effect(risk_episodes)
    combined = baseline_episodes + risk_episodes
    rankings = _blocking_rankings(combined)
    gate = _eligibility(audit, fallback, paired, rankings)
    unseen_icode = [row for row in icode if row["physics_role"] == "unseen"]
    gate["icode_temporal_gate_unseen"] = {
        "mean_final_distance_improvement_m": float(np.mean([
            row["final_distance_improvement_m"] for row in unseen_icode
        ])),
        "ci95": _nested_bootstrap([{
            "model_block": row["model_block"],
            "episode_seed": row["episode_seed"],
            "value": row["final_distance_improvement_m"],
        } for row in unseen_icode]),
        "net_success_gain": int(sum(
            row["success_difference"] for row in unseen_icode
        )),
    }
    gate["clean_fallback"] = fallback
    gate["blocking_rankings"] = rankings

    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "temporal_episodes.csv", risk_episodes)
    _write_csv(output / "combined_episodes.csv", combined)
    _write_csv(output / "condition_summary.csv", _condition_summary(combined))
    _write_csv(output / "paired_effects.csv", paired)
    _write_csv(output / "icode_effects.csv", icode)
    _write_csv(output / "blocking_rankings.csv", rankings)
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "development_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"audit": audit, "gate": gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
