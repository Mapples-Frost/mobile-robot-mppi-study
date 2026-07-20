#!/usr/bin/env python3
"""Create compact, auditable tables for the L217 sealed benchmark."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_ORDER = (
    "traditional_mppi",
    "icode_mppi",
    "rl_driven_mppi",
    "simple_combination",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)
OUTCOMES = (
    "success",
    "collision",
    "steps",
    "final_goal_distance",
    "trajectory_length",
    "minimum_clearance",
    "stuck_steps",
    "spin_steps",
    "mean_abs_omega",
    "control_jerk",
    "planner_compute_ms_mean",
    "planner_compute_ms_p95",
    "planner_compute_ms_max",
    "paper_total_rollouts_mean",
)
MECHANISMS = (
    "reliability_hss_enabled_fraction",
    "reliability_authority_mean",
    "reliability_dynamics_confidence_mean",
    "reliability_actor_support_confidence_mean",
    "reliability_guided_fraction_applied_mean",
    "terminal_value_enabled_fraction",
    "terminal_value_authority_mean",
    "terminal_value_dynamics_confidence_mean",
    "terminal_value_critic_confidence_mean",
    "residual_policy_context_enabled_fraction",
    "residual_policy_authority_enabled_fraction",
    "residual_policy_authority_mean",
)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if str(value).lower() in ("true", "false"):
        return float(str(value).lower() == "true")
    return float(value)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows, strata):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in strata)
        grouped[key].append(row)
    output = []
    for key, items in grouped.items():
        entry = dict(zip(strata, key))
        entry.update({
            "episodes": len(items),
            "independent_seeds": len({int(row["seed"]) for row in items}),
        })
        for metric in OUTCOMES:
            entry[metric + "_mean"] = float(np.mean([
                number(row[metric]) for row in items
            ]))
        output.append(entry)
    order = {arm: index for index, arm in enumerate(METHOD_ORDER)}
    return sorted(output, key=lambda item: (
        order[item["benchmark_arm"]],
        item.get("scene", ""),
        item.get("physics_domain", ""),
    ))


def primary_effects(paired):
    output = []
    for metric in OUTCOMES:
        item = paired["full_vs_simple"]["metrics"][metric]
        output.append({
            "comparison": "full_proposed_vs_simple_combination",
            "metric": metric,
            "higher_is_better": item["higher_is_better"],
            "simple_mean": item["control_mean"],
            "full_mean": item["aligned_mean"],
            "favorable_effect": item["favorable_effect"],
            "relative_favorable_change": item["relative_favorable_change"],
            "ci95_low": item["ci95"][0],
            "ci95_high": item["ci95"][1],
            "independent_seed_clusters": paired["full_vs_simple"]["independent_clusters"],
            "bootstrap_samples": paired["full_vs_simple"]["bootstrap_samples"],
        })
    return output


def factorial_effects(factorial):
    output = []
    names = (
        ("icode_main_effect", "value_alignment_main_effect"),
        ("rl_main_effect", "role_aware_hss_main_effect"),
        ("icode_by_rl_interaction", "value_by_hss_interaction"),
    )
    for metric in OUTCOMES:
        effects = factorial[metric]["effects"]
        for internal, label in names:
            item = effects[internal]
            output.append({
                "metric": metric,
                "effect": label,
                "raw_contrast_estimate": item["estimate"],
                "ci95_low": item["ci95"][0],
                "ci95_high": item["ci95"][1],
                "higher_is_better": factorial[metric]["higher_is_better"],
                "favorable": item["favorable"],
                "independent_seed_clusters": factorial[metric]["independent_clusters"],
            })
    return output


def mechanism_summary(rows):
    output = []
    for arm in METHOD_ORDER:
        items = [row for row in rows if row["benchmark_arm"] == arm]
        entry = {"benchmark_arm": arm, "episodes": len(items)}
        for metric in MECHANISMS:
            entry[metric + "_mean"] = float(np.mean([
                number(row[metric]) for row in items
            ]))
        output.append(entry)
    return output


def mechanism_coverage(rows):
    full = [row for row in rows if row["benchmark_arm"] == "full_proposed"]
    average = lambda field: float(np.mean([number(row[field]) for row in full]))
    dynamics_confidence = average("reliability_dynamics_confidence_mean")
    terminal_authority = average("terminal_value_authority_mean")
    coverage = {
        "value_aligned_icode_checkpoint_selected": all(
            int(row["value_alignment"]) == 1 and bool(row["icode_checkpoints"])
            for row in full
        ),
        "rl_prior_enabled": all(bool(row["actor_checkpoint"]) for row in full),
        "role_aware_hss_activated": average("reliability_hss_enabled_fraction") > 0.99,
        "residual_policy_context_activated": average(
            "residual_policy_context_enabled_fraction"
        ) > 0.0,
        "residual_policy_authority_activated": average(
            "residual_policy_authority_enabled_fraction"
        ) > 0.0,
        "terminal_value_enabled": average("terminal_value_enabled_fraction") > 0.99,
        "terminal_value_reliability_weighting_observed": bool(
            dynamics_confidence < 0.99 and terminal_authority < 0.99
        ),
        "mean_dynamics_confidence": dynamics_confidence,
        "mean_hss_authority": average("reliability_authority_mean"),
        "mean_terminal_value_authority": terminal_authority,
        "mean_residual_policy_context_enabled_fraction": average(
            "residual_policy_context_enabled_fraction"
        ),
    }
    required = (
        "value_aligned_icode_checkpoint_selected",
        "rl_prior_enabled",
        "role_aware_hss_activated",
        "residual_policy_context_activated",
        "terminal_value_reliability_weighting_observed",
    )
    coverage["complete_planned_method_coverage"] = all(
        coverage[field] for field in required
    )
    coverage["status"] = (
        "complete" if coverage["complete_planned_method_coverage"]
        else "partial_mechanism_coverage"
    )
    coverage["required_for_complete_claim"] = list(required)
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)
    result = Path(args.result_dir).resolve()
    rows = read_csv(result / "progress.csv")
    paired = json.loads((result / "paired_comparisons.json").read_text(encoding="utf-8"))
    factorial = json.loads((result / "factorial_contrasts.json").read_text(encoding="utf-8"))
    tables = result / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "method_overall_summary.csv", summaries(rows, ("benchmark_arm",)))
    write_csv(
        tables / "method_scene_domain_summary.csv",
        summaries(rows, ("benchmark_arm", "scene", "physics_domain")),
    )
    write_csv(tables / "full_vs_simple_seed_cluster_effects.csv", primary_effects(paired))
    write_csv(tables / "core_factorial_seed_cluster_effects.csv", factorial_effects(factorial))
    write_csv(tables / "mechanism_activation_audit.csv", mechanism_summary(rows))
    coverage = mechanism_coverage(rows)
    (result / "mechanism_coverage_audit.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(tables),
        "tables": sorted(path.name for path in tables.glob("*.csv")),
        "mechanism_coverage": coverage["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
