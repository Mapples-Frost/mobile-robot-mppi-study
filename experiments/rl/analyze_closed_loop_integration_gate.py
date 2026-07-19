#!/usr/bin/env python3
"""Audit a preregistered four-arm closed-loop integration Gate."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ARMS = (
    "ordinary_fixed",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)
ADAPTIVE_ARMS = ("ordinary_adaptive", "full_proposed")
LEVEL_COLUMNS = (
    "reliability_low_fraction",
    "reliability_medium_fraction",
    "reliability_high_fraction",
)


def _bool(value):
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError("invalid boolean value: %r" % value)


def _float(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("nonfinite %s" % key)
    return value


def _mean(rows, key):
    if not rows:
        raise ValueError("cannot average an empty arm")
    return sum(_float(row, key) for row in rows) / float(len(rows))


def _ratio(numerator, denominator):
    if denominator < 0.0:
        raise ValueError("ratio denominator must be nonnegative")
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def analyze(
    rows,
    expected_seeds,
    expected_rollouts=100,
    expected_iterations=2,
    tracking_tolerance=1.05,
    jerk_tolerance=1.10,
    level_epsilon=1e-6,
    mechanism_mode="reliability_levels",
):
    """Return a deterministic audit of the frozen development Gate."""

    expected_seeds = sorted({int(seed) for seed in expected_seeds})
    if mechanism_mode not in (
        "reliability_levels",
        "source_competence",
    ):
        raise ValueError("unknown mechanism mode: %s" % mechanism_mode)
    if not expected_seeds:
        raise ValueError("expected seeds must not be empty")
    expected_count = len(expected_seeds) * len(ARMS)
    if len(rows) != expected_count:
        raise ValueError(
            "expected %d episodes, found %d"
            % (expected_count, len(rows))
        )
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        arm = str(row["factorial_arm"])
        seed = int(row["seed"])
        if arm not in ARMS:
            raise ValueError("unexpected factorial arm: %s" % arm)
        if seed not in expected_seeds:
            raise ValueError("unexpected seed: %d" % seed)
        key = (seed, arm)
        if key in seen:
            raise ValueError("duplicate seed-arm episode: %r" % (key,))
        seen.add(key)
        grouped[arm].append(row)
    for arm in ARMS:
        if len(grouped[arm]) != len(expected_seeds):
            raise ValueError("arm %s is incomplete" % arm)

    budget_values = sorted({
        int(row["rollout_budget_per_decision"]) for row in rows
    })
    iteration_values = sorted({
        int(row["paper_iterations"]) for row in rows
    })
    equal_budget = (
        budget_values == [int(expected_rollouts)]
        and iteration_values == [int(expected_iterations)]
    )

    arm_summary = {}
    for arm in ARMS:
        arm_rows = grouped[arm]
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
            "authority_mean": _mean(
                arm_rows, "reliability_authority_mean"
            ),
            "guided_fraction_mean": _mean(
                arm_rows, "reliability_guided_fraction_applied_mean"
            ),
            "reliability_level_fractions": {
                key: _mean(arm_rows, key) for key in LEVEL_COLUMNS
            },
        }
        if mechanism_mode == "source_competence" and arm in ADAPTIVE_ARMS:
            arm_summary[arm].update({
                "actor_competence_mean": _mean(
                    arm_rows, "reliability_actor_competence_mean"
                ),
                "actor_competence_min": min(
                    _float(
                        row, "reliability_actor_competence_min"
                    )
                    for row in arm_rows
                ),
                "actor_competence_max": max(
                    _float(
                        row, "reliability_actor_competence_max"
                    )
                    for row in arm_rows
                ),
                "raw_guided_fraction_mean": _mean(
                    arm_rows,
                    "reliability_guided_fraction_raw_applied_mean",
                ),
            })

    full = arm_summary["full_proposed"]
    ordinary = arm_summary["ordinary_fixed"]
    value = arm_summary["value_fixed"]
    tracking_vs_ordinary = _ratio(
        full["cross_track_rmse_mean"],
        ordinary["cross_track_rmse_mean"],
    )
    tracking_vs_value = _ratio(
        full["cross_track_rmse_mean"],
        value["cross_track_rmse_mean"],
    )
    jerk_vs_ordinary = _ratio(
        full["control_jerk_mean"],
        ordinary["control_jerk_mean"],
    )

    adaptive_diagnostics = {}
    adaptive_exercised = True
    for arm in ADAPTIVE_ARMS:
        summary = arm_summary[arm]
        active_levels = [
            key
            for key, value_at_level in
            summary["reliability_level_fractions"].items()
            if value_at_level > float(level_epsilon)
        ]
        if mechanism_mode == "reliability_levels":
            arm_passed = (
                len(active_levels) >= 2
                and summary["authority_mean"] > 0.0
                and summary["guided_fraction_mean"] > 0.0
            )
            episode_checks = []
        else:
            episode_checks = []
            for row in grouped[arm]:
                competence_min = _float(
                    row, "reliability_actor_competence_min"
                )
                competence_max = _float(
                    row, "reliability_actor_competence_max"
                )
                raw_guided = _float(
                    row,
                    "reliability_guided_fraction_raw_applied_mean",
                )
                episode_checks.append({
                    "seed": int(row["seed"]),
                    "competence_nonconstant": (
                        competence_max - competence_min
                        > float(level_epsilon)
                    ),
                    "guided_changed_from_high": (
                        abs(raw_guided - 0.60) > float(level_epsilon)
                    ),
                })
            arm_passed = (
                summary["authority_mean"] > 0.0
                and summary["guided_fraction_mean"] > 0.0
                and all(
                    item["competence_nonconstant"]
                    and item["guided_changed_from_high"]
                    for item in episode_checks
                )
            )
        adaptive_exercised = adaptive_exercised and arm_passed
        adaptive_diagnostics[arm] = {
            "active_levels": active_levels,
            "authority_nonzero": summary["authority_mean"] > 0.0,
            "guided_fraction_nonzero": (
                summary["guided_fraction_mean"] > 0.0
            ),
            "episode_checks": episode_checks,
            "passed": arm_passed,
        }

    checks = {
        "full_success_all": full["successes"] == len(expected_seeds),
        "full_zero_collision": full["collisions"] == 0,
        "tracking_vs_ordinary_within_tolerance": (
            tracking_vs_ordinary <= float(tracking_tolerance)
        ),
        "tracking_vs_value_within_tolerance": (
            tracking_vs_value <= float(tracking_tolerance)
        ),
        "jerk_vs_ordinary_within_tolerance": (
            jerk_vs_ordinary <= float(jerk_tolerance)
        ),
        "adaptive_mechanism_exercised": adaptive_exercised,
        "equal_frozen_budget": equal_budget,
    }
    return {
        "schema_version": 1,
        "evidence_class": "development_integration_gate",
        "mechanism_mode": mechanism_mode,
        "expected_seeds": expected_seeds,
        "episode_count": len(rows),
        "frozen_budget": {
            "expected_rollouts": int(expected_rollouts),
            "observed_rollouts": budget_values,
            "expected_iterations": int(expected_iterations),
            "observed_iterations": iteration_values,
        },
        "thresholds": {
            "tracking_ratio_max": float(tracking_tolerance),
            "jerk_ratio_max": float(jerk_tolerance),
            "level_epsilon": float(level_epsilon),
        },
        "arm_summary": arm_summary,
        "comparisons": {
            "full_cross_track_over_ordinary": tracking_vs_ordinary,
            "full_cross_track_over_value": tracking_vs_value,
            "full_jerk_over_ordinary": jerk_vs_ordinary,
        },
        "adaptive_diagnostics": adaptive_diagnostics,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-rollouts", type=int, default=100)
    parser.add_argument("--expected-iterations", type=int, default=2)
    parser.add_argument("--tracking-tolerance", type=float, default=1.05)
    parser.add_argument("--jerk-tolerance", type=float, default=1.10)
    parser.add_argument(
        "--mechanism-mode",
        choices=("reliability_levels", "source_competence"),
        default="reliability_levels",
    )
    args = parser.parse_args(argv)
    with Path(args.progress).open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    result = analyze(
        rows,
        [
            int(value)
            for value in args.expected_seeds.split(",")
            if value.strip()
        ],
        expected_rollouts=args.expected_rollouts,
        expected_iterations=args.expected_iterations,
        tracking_tolerance=args.tracking_tolerance,
        jerk_tolerance=args.jerk_tolerance,
        mechanism_mode=args.mechanism_mode,
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
