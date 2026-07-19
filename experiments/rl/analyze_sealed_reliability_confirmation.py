#!/usr/bin/env python3
"""Audit the frozen path-and-physics reliability confirmation Gate."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = (
    "ordinary_fixed",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)


def _bool(value):
    value = str(value).strip().lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError("invalid boolean: %r" % value)


def _finite(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("nonfinite %s" % key)
    return value


def _mean(rows, key):
    return sum(_finite(row, key) for row in rows) / float(len(rows))


def _ratio(numerator, denominator):
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _seed_cluster_effect(rows, treatment, reference, seed=20260756):
    indexed = {
        (
            str(row["scene"]),
            str(row["physics_domain"]),
            int(row["seed"]),
            str(row["factorial_arm"]),
        ): row
        for row in rows
    }
    seeds = sorted({key[2] for key in indexed})
    cells = sorted({key[:2] for key in indexed})
    effects = []
    for episode_seed in seeds:
        differences = []
        for scene, domain in cells:
            differences.append(
                _finite(
                    indexed[(scene, domain, episode_seed, treatment)],
                    "cross_track_rmse",
                )
                - _finite(
                    indexed[(scene, domain, episode_seed, reference)],
                    "cross_track_rmse",
                )
            )
        effects.append(float(np.mean(differences)))
    values = np.asarray(effects, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    draws = np.mean(
        values[rng.randint(0, len(values), size=(10000, len(values)))],
        axis=1,
    )
    standard_deviation = float(np.std(values, ddof=1))
    return {
        "treatment": treatment,
        "reference": reference,
        "independent_unit": "seed",
        "clusters": len(values),
        "per_seed_mean_difference": effects,
        "mean_difference": float(np.mean(values)),
        "bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "paired_cohen_dz": (
            float(np.mean(values) / standard_deviation)
            if standard_deviation > 0.0
            else None
        ),
    }


def analyze(
    rows,
    expected_seeds,
    expected_scenes,
    expected_domains,
    unseen_domain="combined_unseen",
    per_cell_tolerance=1.05,
    jerk_tolerance=1.10,
    expected_rollouts=100,
    expected_iterations=2,
):
    expected_seeds = sorted({int(value) for value in expected_seeds})
    expected_scenes = sorted({str(value) for value in expected_scenes})
    expected_domains = sorted({str(value) for value in expected_domains})
    expected_count = (
        len(expected_seeds)
        * len(expected_scenes)
        * len(expected_domains)
        * len(ARMS)
    )
    if len(rows) != expected_count:
        raise ValueError(
            "expected %d episodes, found %d" % (expected_count, len(rows))
        )

    by_arm = defaultdict(list)
    by_cell_arm = defaultdict(list)
    seen = set()
    for row in rows:
        scene = str(row["scene"])
        domain = str(row["physics_domain"])
        seed = int(row["seed"])
        arm = str(row["factorial_arm"])
        if (
            scene not in expected_scenes
            or domain not in expected_domains
            or seed not in expected_seeds
            or arm not in ARMS
        ):
            raise ValueError(
                "unexpected scene/domain/seed/arm: %r"
                % ((scene, domain, seed, arm),)
            )
        key = (scene, domain, seed, arm)
        if key in seen:
            raise ValueError("duplicate confirmation episode: %r" % (key,))
        seen.add(key)
        by_arm[arm].append(row)
        by_cell_arm[(scene, domain, arm)].append(row)

    for scene in expected_scenes:
        for domain in expected_domains:
            for arm in ARMS:
                if len(by_cell_arm[(scene, domain, arm)]) != len(
                    expected_seeds
                ):
                    raise ValueError(
                        "incomplete cell: %s/%s/%s"
                        % (scene, domain, arm)
                    )

    arm_summary = {}
    for arm in ARMS:
        arm_rows = by_arm[arm]
        arm_summary[arm] = {
            "episodes": len(arm_rows),
            "successes": sum(_bool(row["success"]) for row in arm_rows),
            "collisions": sum(
                _bool(row["collision"]) for row in arm_rows
            ),
            "cross_track_rmse_mean": _mean(
                arm_rows, "cross_track_rmse"
            ),
            "control_jerk_mean": _mean(arm_rows, "control_jerk"),
        }

    cell_summary = {}
    stable_cells = 0
    unseen_full = []
    unseen_ordinary = []
    for scene in expected_scenes:
        cell_summary[scene] = {}
        for domain in expected_domains:
            cell = {}
            for arm in ARMS:
                cell_rows = by_cell_arm[(scene, domain, arm)]
                cell[arm] = {
                    "cross_track_rmse_mean": _mean(
                        cell_rows, "cross_track_rmse"
                    ),
                    "successes": sum(
                        _bool(row["success"]) for row in cell_rows
                    ),
                    "collisions": sum(
                        _bool(row["collision"]) for row in cell_rows
                    ),
                }
            ratio = _ratio(
                cell["full_proposed"]["cross_track_rmse_mean"],
                cell["ordinary_fixed"]["cross_track_rmse_mean"],
            )
            cell["full_over_ordinary_cross_track"] = ratio
            stable_cells += ratio <= float(per_cell_tolerance)
            if domain == unseen_domain:
                unseen_full.extend(
                    by_cell_arm[(scene, domain, "full_proposed")]
                )
                unseen_ordinary.extend(
                    by_cell_arm[(scene, domain, "ordinary_fixed")]
                )
            cell_summary[scene][domain] = cell

    full = arm_summary["full_proposed"]
    ordinary = arm_summary["ordinary_fixed"]
    simple = arm_summary["value_fixed"]
    full_over_ordinary = _ratio(
        full["cross_track_rmse_mean"],
        ordinary["cross_track_rmse_mean"],
    )
    full_over_simple = _ratio(
        full["cross_track_rmse_mean"],
        simple["cross_track_rmse_mean"],
    )
    full_over_ordinary_jerk = _ratio(
        full["control_jerk_mean"],
        ordinary["control_jerk_mean"],
    )
    unseen_ratio = _ratio(
        _mean(unseen_full, "cross_track_rmse"),
        _mean(unseen_ordinary, "cross_track_rmse"),
    )
    rollout_values = sorted({
        int(row["rollout_budget_per_decision"]) for row in rows
    })
    iteration_values = sorted({
        int(row["paper_iterations"]) for row in rows
    })
    total_cells = len(expected_scenes) * len(expected_domains)
    checks = {
        "full_success_all": full["successes"] == len(by_arm["full_proposed"]),
        "full_zero_collision": full["collisions"] == 0,
        "pooled_full_beats_ordinary": full_over_ordinary <= 1.0,
        "pooled_full_beats_simple_combination": full_over_simple <= 1.0,
        "five_of_six_cells_stable": stable_cells >= total_cells - 1,
        "unseen_full_beats_ordinary": unseen_ratio <= 1.0,
        "jerk_within_tolerance": (
            full_over_ordinary_jerk <= float(jerk_tolerance)
        ),
        "equal_frozen_budget": (
            rollout_values == [int(expected_rollouts)]
            and iteration_values == [int(expected_iterations)]
        ),
    }
    paired_seed_effects = {
        "full_vs_ordinary": _seed_cluster_effect(
            rows, "full_proposed", "ordinary_fixed", seed=20260756
        ),
        "full_vs_simple_combination": _seed_cluster_effect(
            rows, "full_proposed", "value_fixed", seed=20260757
        ),
    }
    return {
        "schema_version": 1,
        "evidence_class": "sealed_path_physics_confirmation",
        "expected_seeds": expected_seeds,
        "expected_scenes": expected_scenes,
        "expected_domains": expected_domains,
        "episode_count": len(rows),
        "arm_summary": arm_summary,
        "cell_summary": cell_summary,
        "comparisons": {
            "pooled_full_over_ordinary_cross_track": full_over_ordinary,
            "pooled_full_over_simple_cross_track": full_over_simple,
            "pooled_full_over_ordinary_jerk": full_over_ordinary_jerk,
            "unseen_full_over_ordinary_cross_track": unseen_ratio,
            "stable_cells": stable_cells,
            "total_cells": total_cells,
        },
        "frozen_budget": {
            "observed_rollouts": rollout_values,
            "observed_iterations": iteration_values,
        },
        "checks": checks,
        "paired_seed_effects": paired_seed_effects,
        "gate_passed": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-scenes", required=True)
    parser.add_argument("--expected-domains", required=True)
    parser.add_argument("--unseen-domain", default="combined_unseen")
    args = parser.parse_args(argv)
    rows = []
    for value in args.progress.split(","):
        with Path(value.strip()).open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            rows.extend(csv.DictReader(handle))
    result = analyze(
        rows,
        args.expected_seeds.split(","),
        args.expected_scenes.split(","),
        args.expected_domains.split(","),
        unseen_domain=args.unseen_domain,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
