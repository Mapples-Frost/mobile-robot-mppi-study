#!/usr/bin/env python3
"""Audit and analyze the preregistered L37 bounded-RL x ICODE factorial."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml


CONDITIONS = (
    "traditional_nominal",
    "traditional_icode",
    "bounded_rl_nominal",
    "bounded_rl_icode",
)


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


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


def _key(row, condition=True):
    base = (
        int(row["model_block"]), str(row["scene"]),
        str(row["physics_domain"]), int(row["episode_seed"]),
    )
    return base + ((str(row["condition"]),) if condition else ())


def _expected_keys(config):
    design = config["rl"]["cross_layer_factorial"]
    scenes = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    return {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"]))
        for scene in scenes
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in design["conditions"]
    }


def _paired_effects(rows):
    lookup = {_key(row): row for row in rows}
    effects = []
    bases = sorted({_key(row, condition=False) for row in rows})
    for base in bases:
        values = {condition: lookup[base + (condition,)] for condition in CONDITIONS}
        tn, ti, rn, ri = (values[name] for name in CONDITIONS)

        def binary(row, name):
            return int(_bool(row[name]))

        def distance(row):
            return float(row["final_goal_distance"])

        comparisons = (
            ("bounded_rl_main", "nominal", tn, rn),
            ("bounded_rl_main", "icode", ti, ri),
            ("icode_main", "traditional", tn, ti),
            ("icode_main", "bounded_rl", rn, ri),
            ("combined_vs_baseline", "cross_layer", tn, ri),
        )
        for effect, stratum, first, second in comparisons:
            effects.append({
                "effect": effect,
                "stratum": stratum,
                "model_block": base[0],
                "scene": base[1],
                "physics_domain": base[2],
                "episode_seed": base[3],
                "success_difference": binary(second, "success") - binary(first, "success"),
                "collision_difference": binary(second, "collision") - binary(first, "collision"),
                "goal_distance_improvement_m": distance(first) - distance(second),
            })
        effects.append({
            "effect": "interaction",
            "stratum": "difference_in_differences",
            "model_block": base[0],
            "scene": base[1],
            "physics_domain": base[2],
            "episode_seed": base[3],
            "success_difference": (
                (binary(ri, "success") - binary(ti, "success"))
                - (binary(rn, "success") - binary(tn, "success"))
            ),
            "collision_difference": (
                (binary(ri, "collision") - binary(ti, "collision"))
                - (binary(rn, "collision") - binary(tn, "collision"))
            ),
            "goal_distance_improvement_m": (
                (distance(ti) - distance(ri)) - (distance(tn) - distance(rn))
            ),
        })
    return effects


def _condition_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    output = []
    for condition in CONDITIONS:
        selected = grouped[condition]
        output.append({
            "condition": condition,
            "episodes": len(selected),
            "success_rate": float(np.mean([_bool(row["success"]) for row in selected])),
            "collision_rate": float(np.mean([_bool(row["collision"]) for row in selected])),
            "mean_final_goal_distance_m": float(np.mean([
                float(row["final_goal_distance"]) for row in selected
            ])),
            "mean_planner_compute_ms": float(np.mean([
                float(row["planner_compute_ms_mean"]) for row in selected
            ])),
        })
    return output


def _bootstrap(effects, field, seed, replicates):
    grouped = defaultdict(list)
    for row in effects:
        grouped[int(row["model_block"])].append(float(row[field]))
    blocks = np.asarray(sorted(grouped), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        block_means = []
        for block in sampled_blocks:
            values = np.asarray(grouped[int(block)], dtype=np.float64)
            block_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples[index] = float(np.mean(block_means))
    estimate = float(np.mean([
        np.mean(grouped[int(block)]) for block in blocks
    ]))
    return {
        "estimate": estimate,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
    }


def _block_rankings(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["model_block"])][str(row["condition"])].append(row)
    output = []
    for block, conditions in sorted(grouped.items()):
        stats = {}
        for condition, selected in conditions.items():
            stats[condition] = {
                "successes": int(sum(_bool(row["success"]) for row in selected)),
                "collisions": int(sum(_bool(row["collision"]) for row in selected)),
            }
        combined = stats["bounded_rl_icode"]
        output.append({
            "model_block": block,
            "combined_successes": combined["successes"],
            "combined_collisions": combined["collisions"],
            "combined_success_best_or_tied": bool(
                combined["successes"] >= max(item["successes"] for item in stats.values())
            ),
            "combined_collision_best_or_tied": bool(
                combined["collisions"] <= min(item["collisions"] for item in stats.values())
            ),
            "combined_joint_best_or_tied": bool(
                combined["successes"] >= max(item["successes"] for item in stats.values())
                and combined["collisions"] <= min(item["collisions"] for item in stats.values())
            ),
        })
    return output


def _plot(input_dir, summary, rankings):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })
    labels = ["T–N", "T–I", "RL–N", "RL–I"]
    colors = ["#8C8C8C", "#56B4E9", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.65))
    for axis, field, ylabel, title in (
        (axes[0], "success_rate", "Success rate", "A  Success"),
        (axes[1], "collision_rate", "Collision rate", "B  Collision"),
        (axes[2], "mean_final_goal_distance_m", "Final distance (m)", "C  Terminal error"),
    ):
        values = [row[field] for row in summary]
        axis.bar(np.arange(4), values, color=colors, width=0.68)
        axis.set_xticks(np.arange(4))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        if field.endswith("rate"):
            axis.set_ylim(0.0, 1.0)
    fig.suptitle(
        "L37 bounded RL × ICODE factorial on calibrated dynamic scenes",
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "T: traditional, RL: bounded RL, N: nominal, I: ICODE; three model blocks, shared temporal safety.",
        ha="center", fontsize=7.1,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.91), w_pad=1.1)
    fig.savefig(input_dir / "fig_l37_bounded_rl_icode_factorial.png")
    fig.savefig(input_dir / "fig_l37_bounded_rl_icode_factorial.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    design = config["rl"]["cross_layer_factorial"]
    input_dir = _resolved_path(args.input_dir)
    rows = []
    metadata = []
    for block in range(len(design["model_blocks"])):
        block_dir = input_dir / ("block_%d" % block)
        rows.extend(_read_csv(block_dir / "episodes.csv"))
        metadata.append(json.loads((block_dir / "metadata.json").read_text(encoding="utf-8")))
    expected = _expected_keys(config)
    observed_keys = [_key(row) for row in rows]
    observed_set = set(observed_keys)
    finite_fields = (
        "final_goal_distance", "minimum_clearance", "control_jerk",
        "planner_compute_ms_mean",
    )
    nonfinite = sum(
        any(
            row.get(name) not in (None, "") and not np.isfinite(float(row[name]))
            for name in finite_fields
        ) for row in rows
    )
    protected_used = sorted({
        int(seed) for item in metadata for seed in item.get("previous_protected_seeds_used", ())
    })
    sealed_used = sorted({
        int(seed) for item in metadata for seed in item.get("sealed_confirmation_seeds_used", ())
    })
    integrity = bool(
        observed_set == expected
        and len(observed_keys) == len(observed_set)
        and nonfinite == 0
        and not protected_used
        and not sealed_used
    )
    effects = _paired_effects(rows)
    condition_summary = _condition_summary(rows)
    rankings = _block_rankings(rows)

    def selected(effect, stratum=None):
        return [
            row for row in effects
            if row["effect"] == effect and (stratum is None or row["stratum"] == stratum)
        ]

    combined = selected("combined_vs_baseline")
    bounded = selected("bounded_rl_main")
    icode = selected("icode_main")
    interaction = selected("interaction")
    combined_net_success = int(sum(row["success_difference"] for row in combined))
    combined_net_collision = int(sum(row["collision_difference"] for row in combined))
    bounded_net_success = int(sum(row["success_difference"] for row in bounded))
    icode_distance = float(np.mean([row["goal_distance_improvement_m"] for row in icode]))
    interaction_success = float(np.mean([row["success_difference"] for row in interaction]))
    interaction_distance = float(np.mean([row["goal_distance_improvement_m"] for row in interaction]))
    joint_best_blocks = int(sum(row["combined_joint_best_or_tied"] for row in rankings))
    gate_config = design["gate"]
    gate = {
        "artifact_integrity": integrity,
        "combined_best_or_tied_model_blocks": joint_best_blocks,
        "combined_vs_baseline_net_success_gain": combined_net_success,
        "combined_vs_baseline_net_collision_increase": combined_net_collision,
        "bounded_rl_main_effect_net_success_gain": bounded_net_success,
        "icode_main_effect_goal_distance_improvement_m": icode_distance,
        "interaction_success_difference": interaction_success,
        "interaction_goal_distance_improvement_m": interaction_distance,
    }
    gate["passed"] = bool(
        integrity
        and joint_best_blocks >= int(gate_config["minimum_combined_best_or_tied_model_blocks"])
        and combined_net_success >= int(gate_config["minimum_combined_vs_baseline_net_success_gain"])
        and combined_net_collision <= int(gate_config["maximum_combined_vs_baseline_net_collision_increase"])
        and bounded_net_success >= int(gate_config["minimum_bounded_rl_main_effect_net_success_gain"])
        and icode_distance >= float(gate_config["minimum_icode_main_effect_goal_distance_improvement_m"])
        and (
            interaction_success >= float(gate_config["minimum_interaction_success_difference"])
            or interaction_distance >= float(gate_config["minimum_alternative_interaction_goal_distance_m"])
        )
    )
    seed = int(design["bootstrap_seed"])
    replicates = int(design["bootstrap_replicates"])
    bootstrap = {}
    for index, (name, effect_rows) in enumerate((
        ("combined_vs_baseline", combined),
        ("bounded_rl_main", bounded),
        ("icode_main", icode),
        ("interaction", interaction),
    )):
        bootstrap[name] = {
            field: _bootstrap(effect_rows, field, seed + index * 10 + offset, replicates)
            for offset, field in enumerate((
                "success_difference", "collision_difference", "goal_distance_improvement_m"
            ))
        }
    summary = {
        "design_id": "l37_bounded_rl_icode_factorial_v1",
        "episodes": len(rows),
        "audit": {
            "expected_episodes": len(expected),
            "observed_episodes": len(rows),
            "missing_keys": len(expected - observed_set),
            "unexpected_keys": len(observed_set - expected),
            "duplicate_keys": len(observed_keys) - len(observed_set),
            "nonfinite_rows": int(nonfinite),
            "protected_seeds_used": protected_used,
            "sealed_seeds_used": sealed_used,
        },
        "condition_summary": condition_summary,
        "model_block_rankings": rankings,
        "factorial_effects": {
            "combined_vs_baseline": {
                "net_success_gain": combined_net_success,
                "net_collision_increase": combined_net_collision,
                "mean_goal_distance_improvement_m": float(np.mean([
                    row["goal_distance_improvement_m"] for row in combined
                ])),
            },
            "bounded_rl_main": {
                "net_success_gain": bounded_net_success,
                "net_collision_increase": int(sum(row["collision_difference"] for row in bounded)),
                "mean_goal_distance_improvement_m": float(np.mean([
                    row["goal_distance_improvement_m"] for row in bounded
                ])),
            },
            "icode_main": {
                "net_success_gain": int(sum(row["success_difference"] for row in icode)),
                "net_collision_increase": int(sum(row["collision_difference"] for row in icode)),
                "mean_goal_distance_improvement_m": icode_distance,
            },
            "interaction": {
                "mean_success_difference": interaction_success,
                "mean_collision_difference": float(np.mean([
                    row["collision_difference"] for row in interaction
                ])),
                "mean_goal_distance_improvement_m": interaction_distance,
            },
        },
        "hierarchical_bootstrap": bootstrap,
        "development_gate": gate,
        "interpretation_guard": (
            "L37 uses development seeds and cannot be reported as final confirmation."
        ),
    }
    _write_csv(input_dir / "episodes.csv", rows)
    _write_csv(input_dir / "paired_factorial_effects.csv", effects)
    _write_csv(input_dir / "condition_summary.csv", condition_summary)
    _write_csv(input_dir / "model_block_rankings.csv", rankings)
    (input_dir / "factorial_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(input_dir, condition_summary, rankings)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
