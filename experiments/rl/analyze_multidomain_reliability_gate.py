#!/usr/bin/env python3
"""Audit the preregistered multi-domain reliability development Gate."""

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


def _bool(value):
    value = str(value).strip().lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError("invalid boolean: %r" % value)


def _float(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("nonfinite %s" % key)
    return value


def _mean(rows, key):
    return sum(_float(row, key) for row in rows) / float(len(rows))


def _ratio(numerator, denominator):
    if denominator < 0.0:
        raise ValueError("negative ratio denominator")
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def analyze(
    rows,
    expected_seeds,
    expected_domains,
    unseen_domain="combined_unseen",
    expected_rollouts=100,
    expected_iterations=2,
    per_domain_tolerance=1.05,
    jerk_tolerance=1.10,
    epsilon=1e-6,
):
    expected_seeds = sorted({int(value) for value in expected_seeds})
    expected_domains = sorted({
        str(value) for value in expected_domains
    })
    if unseen_domain not in expected_domains:
        raise ValueError("unseen domain is absent from expected domains")
    expected_count = (
        len(expected_seeds) * len(expected_domains) * len(ARMS)
    )
    if len(rows) != expected_count:
        raise ValueError(
            "expected %d episodes, found %d"
            % (expected_count, len(rows))
        )

    by_arm = defaultdict(list)
    by_domain_arm = defaultdict(list)
    seen_keys = set()
    for row in rows:
        arm = str(row["factorial_arm"])
        domain = str(row["physics_domain"])
        seed = int(row["seed"])
        if arm not in ARMS:
            raise ValueError("unexpected arm: %s" % arm)
        if domain not in expected_domains:
            raise ValueError("unexpected domain: %s" % domain)
        if seed not in expected_seeds:
            raise ValueError("unexpected seed: %d" % seed)
        key = (domain, seed, arm)
        if key in seen_keys:
            raise ValueError("duplicate domain-seed-arm block: %r" % (key,))
        seen_keys.add(key)
        by_arm[arm].append(row)
        by_domain_arm[(domain, arm)].append(row)

    expected_replicates = len(expected_seeds)
    for domain in expected_domains:
        for arm in ARMS:
            if len(by_domain_arm[(domain, arm)]) != expected_replicates:
                raise ValueError(
                    "incomplete block for %s/%s" % (domain, arm)
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

    domain_summary = {}
    for domain in expected_domains:
        domain_summary[domain] = {}
        for arm in ARMS:
            domain_rows = by_domain_arm[(domain, arm)]
            summary = {
                "episodes": len(domain_rows),
                "successes": sum(
                    _bool(row["success"]) for row in domain_rows
                ),
                "collisions": sum(
                    _bool(row["collision"]) for row in domain_rows
                ),
                "cross_track_rmse_mean": _mean(
                    domain_rows, "cross_track_rmse"
                ),
                "control_jerk_mean": _mean(
                    domain_rows, "control_jerk"
                ),
            }
            if arm in ADAPTIVE_ARMS:
                summary.update({
                    "dynamics_confidence_mean": _mean(
                        domain_rows,
                        "reliability_dynamics_confidence_mean",
                    ),
                    "actor_competence_mean": _mean(
                        domain_rows,
                        "reliability_actor_competence_mean",
                    ),
                    "raw_guided_fraction_mean": _mean(
                        domain_rows,
                        "reliability_guided_fraction_raw_applied_mean",
                    ),
                })
            domain_summary[domain][arm] = summary

    full = arm_summary["full_proposed"]
    ordinary = arm_summary["ordinary_fixed"]
    pooled_tracking_ratio = _ratio(
        full["cross_track_rmse_mean"],
        ordinary["cross_track_rmse_mean"],
    )
    pooled_jerk_ratio = _ratio(
        full["control_jerk_mean"],
        ordinary["control_jerk_mean"],
    )
    domain_ratios = {}
    within_domain_count = 0
    for domain in expected_domains:
        ratio = _ratio(
            domain_summary[domain]["full_proposed"][
                "cross_track_rmse_mean"
            ],
            domain_summary[domain]["ordinary_fixed"][
                "cross_track_rmse_mean"
            ],
        )
        domain_ratios[domain] = ratio
        within_domain_count += ratio <= float(per_domain_tolerance)

    mechanism_episodes = []
    for arm in ADAPTIVE_ARMS:
        for row in by_arm[arm]:
            competence_min = _float(
                row, "reliability_actor_competence_min"
            )
            competence_max = _float(
                row, "reliability_actor_competence_max"
            )
            raw_guided = _float(
                row, "reliability_guided_fraction_raw_applied_mean"
            )
            mechanism_episodes.append({
                "arm": arm,
                "domain": row["physics_domain"],
                "seed": int(row["seed"]),
                "competence_nonconstant": (
                    competence_max - competence_min > float(epsilon)
                ),
                "allocation_changed": (
                    abs(raw_guided - 0.60) > float(epsilon)
                ),
            })
    mechanism_passed = all(
        item["competence_nonconstant"] and item["allocation_changed"]
        for item in mechanism_episodes
    )

    seen_dynamics_confidence = sum(
        domain_summary[domain]["full_proposed"][
            "dynamics_confidence_mean"
        ]
        for domain in expected_domains
        if domain != unseen_domain
    ) / float(len(expected_domains) - 1)
    unseen_dynamics_confidence = domain_summary[unseen_domain][
        "full_proposed"
    ]["dynamics_confidence_mean"]

    rollout_values = sorted({
        int(row["rollout_budget_per_decision"]) for row in rows
    })
    iteration_values = sorted({
        int(row["paper_iterations"]) for row in rows
    })
    checks = {
        "full_success_all": (
            full["successes"] == len(expected_seeds) * len(expected_domains)
        ),
        "full_zero_collision": full["collisions"] == 0,
        "pooled_tracking_directional": pooled_tracking_ratio <= 1.0,
        "three_of_four_domains_stable": within_domain_count >= 3,
        "unseen_tracking_directional": (
            domain_ratios[unseen_domain] <= 1.0
        ),
        "pooled_jerk_within_tolerance": (
            pooled_jerk_ratio <= float(jerk_tolerance)
        ),
        "unseen_dynamics_confidence_lower": (
            unseen_dynamics_confidence < seen_dynamics_confidence
        ),
        "adaptive_mechanism_exercised": mechanism_passed,
        "equal_frozen_budget": (
            rollout_values == [int(expected_rollouts)]
            and iteration_values == [int(expected_iterations)]
        ),
    }
    return {
        "schema_version": 1,
        "evidence_class": "multidomain_development_gate",
        "expected_seeds": expected_seeds,
        "expected_domains": expected_domains,
        "unseen_domain": unseen_domain,
        "episode_count": len(rows),
        "arm_summary": arm_summary,
        "domain_summary": domain_summary,
        "comparisons": {
            "pooled_full_over_ordinary_cross_track": (
                pooled_tracking_ratio
            ),
            "pooled_full_over_ordinary_jerk": pooled_jerk_ratio,
            "full_over_ordinary_cross_track_by_domain": domain_ratios,
            "domains_within_tracking_tolerance": within_domain_count,
            "seen_dynamics_confidence_mean": (
                seen_dynamics_confidence
            ),
            "unseen_dynamics_confidence_mean": (
                unseen_dynamics_confidence
            ),
        },
        "mechanism_episodes": mechanism_episodes,
        "frozen_budget": {
            "observed_rollouts": rollout_values,
            "observed_iterations": iteration_values,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progress",
        required=True,
        help="comma-separated progress CSV shard paths",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-domains", required=True)
    parser.add_argument(
        "--unseen-domain", default="combined_unseen"
    )
    args = parser.parse_args(argv)
    rows = []
    for value in args.progress.split(","):
        with Path(value.strip()).open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            rows.extend(csv.DictReader(handle))
    result = analyze(
        rows,
        [
            int(value)
            for value in args.expected_seeds.split(",")
            if value.strip()
        ],
        [
            value.strip()
            for value in args.expected_domains.split(",")
            if value.strip()
        ],
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
