"""Analyze and integrity-audit the frozen single-obstacle confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_dynamic_actor_single_obstacle_confirmatory import (
    DEFAULT_PROTOCOL,
    _repo_path,
    _sha256,
    _write_json,
    build_schedule,
    validate_protocol,
)


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be a mapping: %s" % path)
    return value


def _bootstrap_mean_ci(values, seed, replicates):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(replicates), dtype=np.float64)
    chunk = 5000
    for start in range(0, int(replicates), chunk):
        stop = min(start + chunk, int(replicates))
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def _paired_summary(values, seed, replicates):
    values = np.asarray(values, dtype=np.float64)
    test = stats.ttest_1samp(values, 0.0)
    standard_error = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, len(values) - 1))
    mean = float(values.mean())
    return {
        "pairs": int(len(values)),
        "mean": mean,
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)),
        "paired_t_two_sided_p": float(test.pvalue),
        "paired_t_95_ci": [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
        "paired_bootstrap_95_ci": _bootstrap_mean_ci(
            values, seed, replicates
        ),
        "positive": int(np.count_nonzero(values > 0.0)),
        "negative": int(np.count_nonzero(values < 0.0)),
        "zero": int(np.count_nonzero(values == 0.0)),
    }


def _mcnemar(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    left_only = int(np.count_nonzero(left & ~right))
    right_only = int(np.count_nonzero(~left & right))
    discordant = left_only + right_only
    pvalue = 1.0 if discordant == 0 else float(
        stats.binomtest(left_only, discordant, 0.5).pvalue
    )
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": pvalue,
    }


def _one_sided_binomial_upper(events, total, confidence=0.95):
    if events >= total:
        return 1.0
    return float(stats.beta.ppf(confidence, events + 1, total - events))


def _holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, (count - index) * float(pvalues[name]))
        adjusted[name] = min(1.0, running)
    return adjusted


def _trajectory_audit(path):
    planner_ms = []
    rollout_values = []
    tracker_enabled = []
    forecast_valid = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            planner_ms.append(float(row["planner_compute_ms"]))
            rollout_values.append(int(float(row["paper_total_rollouts"])))
            tracker_enabled.append(float(row["dynamic_obstacle_tracker_enabled"]))
            forecast_valid.append(float(row["dynamic_obstacle_tracker_forecast_valid"]))
    if not planner_ms:
        raise ValueError("empty trajectory: %s" % path)
    return {
        "planner_ms": planner_ms,
        "rollouts": rollout_values,
        "tracker_enabled": tracker_enabled,
        "forecast_valid": forecast_valid,
    }


def _arm_rows(raw_rows, arm, split=None):
    return [
        row
        for row in raw_rows
        if row["arm"] == arm and (split is None or row["split"] == split)
    ]


def _by_seed(rows):
    result = {}
    for row in rows:
        key = (row["split"], int(row["seed"]))
        if key in result:
            raise ValueError("duplicate arm seed row: %s" % (key,))
        result[key] = row
    return result


def _efficiency(summary, penalty):
    return int(summary["steps"]) if summary["success"] else max(
        int(summary["steps"]), int(penalty)
    )


def _primary_analysis(rows, protocol, split=None):
    source = _by_seed(_arm_rows(rows, "source_actor", split))
    candidate = _by_seed(_arm_rows(rows, "dynamic_actor", split))
    if set(source) != set(candidate):
        raise ValueError("primary arms are not completely paired")
    keys = sorted(source)
    penalty = int(protocol["analysis"]["failure_step_penalty"])
    efficiency = [
        _efficiency(candidate[key]["summary"], penalty)
        - _efficiency(source[key]["summary"], penalty)
        for key in keys
    ]
    distance = [
        float(candidate[key]["summary"]["final_goal_distance_m"])
        - float(source[key]["summary"]["final_goal_distance_m"])
        for key in keys
    ]
    clearance = [
        float(candidate[key]["summary"]["minimum_clearance_m"])
        - float(source[key]["summary"]["minimum_clearance_m"])
        for key in keys
    ]
    success_source = [bool(source[key]["summary"]["success"]) for key in keys]
    success_candidate = [bool(candidate[key]["summary"]["success"]) for key in keys]
    collision_source = [bool(source[key]["summary"]["collision"]) for key in keys]
    collision_candidate = [bool(candidate[key]["summary"]["collision"]) for key in keys]
    candidate_only_collision = sum(
        c and not s for s, c in zip(collision_source, collision_candidate)
    )
    lost_source_success = sum(
        s and not c for s, c in zip(success_source, success_candidate)
    )
    seed = int(protocol["analysis"]["bootstrap_seed"]) + (0 if split is None else (1 if split == "held_out_id" else 2))
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    source_efficiency_mean = float(np.mean([
        _efficiency(source[key]["summary"], penalty) for key in keys
    ]))
    return {
        "pairs": len(keys),
        "source_successes": int(sum(success_source)),
        "candidate_successes": int(sum(success_candidate)),
        "source_collisions": int(sum(collision_source)),
        "candidate_collisions": int(sum(collision_candidate)),
        "candidate_only_collisions": int(candidate_only_collision),
        "lost_source_successes": int(lost_source_success),
        "candidate_only_collision_95_upper": _one_sided_binomial_upper(
            candidate_only_collision, len(keys)
        ),
        "lost_source_success_95_upper": _one_sided_binomial_upper(
            lost_source_success, len(keys)
        ),
        "success_mcnemar": _mcnemar(success_source, success_candidate),
        "collision_mcnemar": _mcnemar(collision_source, collision_candidate),
        "outcome_aware_efficiency_delta_candidate_minus_source": _paired_summary(
            efficiency, seed, replicates
        ),
        "efficiency_noninferiority_margin_steps": float(
            protocol["analysis"]["efficiency_relative_noninferiority_margin"]
        ) * source_efficiency_mean,
        "final_goal_distance_delta_candidate_minus_source_m": _paired_summary(
            distance, seed + 10, replicates
        ),
        "minimum_clearance_delta_candidate_minus_source_m": _paired_summary(
            clearance, seed + 20, replicates
        ),
    }


def _ablation_analysis(rows, protocol):
    full = _by_seed([
        row for row in _arm_rows(rows, "dynamic_actor") if row["ablation_block"]
    ])
    results = {}
    raw_pvalues = {}
    penalty = int(protocol["analysis"]["failure_step_penalty"])
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    base_seed = int(protocol["analysis"]["bootstrap_seed"]) + 100
    for index, arm in enumerate(protocol["design"]["ablation_arms"]):
        ablated = _by_seed(_arm_rows(rows, arm))
        if set(full) != set(ablated) or len(full) != 50:
            raise ValueError("ablation contrast is not complete: %s" % arm)
        differences = [
            _efficiency(full[key]["summary"], penalty)
            - _efficiency(ablated[key]["summary"], penalty)
            for key in sorted(full)
        ]
        summary = _paired_summary(
            differences, base_seed + index, replicates
        )
        results[arm] = {
            "contrast": "full_candidate_minus_ablation",
            "outcome_aware_efficiency_delta": summary,
        }
        raw_pvalues[arm] = summary["paired_t_two_sided_p"]
    adjusted = _holm(raw_pvalues)
    for arm in results:
        results[arm]["holm_adjusted_p"] = adjusted[arm]
    return results


def analyze(protocol_path=DEFAULT_PROTOCOL):
    protocol_path, protocol, registry, _blocks, jobs, verified = validate_protocol(protocol_path)
    output = _repo_path(protocol["output_dir"])
    raw_path = output / "raw_results.json"
    if not raw_path.is_file():
        raise FileNotFoundError("raw confirmatory matrix is incomplete")
    raw = _json(raw_path)
    rows = raw.get("rows", [])
    if len(rows) != len(jobs) or len(rows) != 350:
        raise ValueError("raw result count does not match frozen schedule")
    expected = [
        (job["run_order"], job["split"], job["seed"], job["arm"])
        for job in jobs
    ]
    actual = [
        (row["run_order"], row["split"], row["seed"], row["arm"])
        for row in rows
    ]
    if actual != expected:
        raise ValueError("raw result order does not match frozen schedule")

    primary = _primary_analysis(rows, protocol)
    by_split = {
        split: _primary_analysis(rows, protocol, split)
        for split in ("held_out_id", "held_out_ood")
    }
    ablations = _ablation_analysis(rows, protocol)

    timing = {}
    all_candidate_ms = []
    all_rollouts = []
    tracker_values = []
    forecast_values = []
    trajectory_audits = {}
    for row in rows:
        run_dir = (
            output
            / "runs"
            / str(row["split"])
            / ("seed_%d" % int(row["seed"]))
            / str(row["arm"])
        )
        audit = _trajectory_audit(run_dir / "trajectory.csv")
        trajectory_audits[(row["split"], int(row["seed"]), row["arm"])] = audit
        all_rollouts.extend(audit["rollouts"])
    for split in ("held_out_id", "held_out_ood"):
        split_ms = []
        episode_p95 = []
        for row in _arm_rows(rows, "dynamic_actor", split):
            audit = trajectory_audits[(split, int(row["seed"]), "dynamic_actor")]
            split_ms.extend(audit["planner_ms"])
            episode_p95.append(float(np.percentile(audit["planner_ms"], 95)))
            tracker_values.extend(audit["tracker_enabled"])
            forecast_values.extend(audit["forecast_valid"])
        all_candidate_ms.extend(split_ms)
        timing[split] = {
            "decision_count": len(split_ms),
            "decision_p95_ms": float(np.percentile(split_ms, 95)),
            "decision_deadline_miss_fraction": float(np.mean(np.asarray(split_ms) >= 100.0)),
            "episode_p95_median_ms": float(np.median(episode_p95)),
            "episode_p95_max_ms": float(np.max(episode_p95)),
        }
    timing["pooled"] = {
        "decision_count": len(all_candidate_ms),
        "decision_p95_ms": float(np.percentile(all_candidate_ms, 95)),
        "decision_deadline_miss_fraction": float(np.mean(np.asarray(all_candidate_ms) >= 100.0)),
    }

    gates = protocol["gates"]
    efficiency_ci = primary[
        "outcome_aware_efficiency_delta_candidate_minus_source"
    ]["paired_t_95_ci"]
    checks = {
        "complete_100_primary_pairs": primary["pairs"] == 100,
        "complete_50_pairs_per_ablation": all(
            value["outcome_aware_efficiency_delta"]["pairs"] == 50
            for value in ablations.values()
        ),
        "zero_candidate_only_collisions": primary["candidate_only_collisions"] == 0,
        "zero_lost_source_successes": primary["lost_source_successes"] == 0,
        "candidate_only_collision_noninferior": primary[
            "candidate_only_collision_95_upper"
        ] < float(gates["maximum_pooled_harm_probability"]),
        "lost_source_success_noninferior": primary[
            "lost_source_success_95_upper"
        ] < float(gates["maximum_pooled_harm_probability"]),
        "split_harm_bounds": all(
            by_split[split][name] < float(gates["maximum_split_harm_probability"])
            for split in by_split
            for name in (
                "candidate_only_collision_95_upper",
                "lost_source_success_95_upper",
            )
        ),
        "efficiency_noninferior": efficiency_ci[1]
        <= primary["efficiency_noninferiority_margin_steps"],
        "at_least_one_strict_pooled_improvement": (
            primary["candidate_successes"] > primary["source_successes"]
            or primary["candidate_collisions"] < primary["source_collisions"]
            or primary["outcome_aware_efficiency_delta_candidate_minus_source"]["mean"] < 0.0
        ),
        "fixed_600_rollouts_all_arms": bool(all_rollouts)
        and set(all_rollouts) == {600},
        "pooled_decision_p95_below_100_ms": timing["pooled"]["decision_p95_ms"]
        < float(gates["planner_deadline_ms"]),
        "id_decision_p95_below_100_ms": timing["held_out_id"]["decision_p95_ms"]
        < float(gates["planner_deadline_ms"]),
        "ood_decision_p95_below_100_ms": timing["held_out_ood"]["decision_p95_ms"]
        < float(gates["planner_deadline_ms"]),
        "online_tracker_enabled": bool(tracker_values) and min(tracker_values) >= 1.0,
        "forecast_availability": bool(forecast_values)
        and float(np.mean(forecast_values)) >= float(gates["minimum_forecast_availability"]),
    }
    gate = {"passed": bool(all(checks.values())), "checks": checks}
    superiority = {
        "efficiency_superiority_supported": (
            primary["outcome_aware_efficiency_delta_candidate_minus_source"]["mean"] < 0.0
            and efficiency_ci[1] < 0.0
            and primary["outcome_aware_efficiency_delta_candidate_minus_source"]["paired_t_two_sided_p"] < 0.05
        ),
        "success_superiority_exploratory": primary["candidate_successes"] > primary["source_successes"]
        and primary["success_mcnemar"]["exact_two_sided_p"] < 0.05,
        "collision_superiority_exploratory": primary["candidate_collisions"] < primary["source_collisions"]
        and primary["collision_mcnemar"]["exact_two_sided_p"] < 0.05,
    }

    protocol_sha = _sha256(protocol_path)
    raw_files = sorted(
        path for path in (output / "runs").glob("**/*") if path.is_file()
    )
    manifest = {
        str(path.relative_to(ROOT)): _sha256(path) for path in raw_files
    }
    integrity = {
        "schema_version": 1,
        "status": "pass" if (
            raw.get("protocol_sha256") == protocol_sha
            and raw.get("sealed_seeds_opened") is False
            and len(raw_files) == 350 * 4
        ) else "fail",
        "protocol_sha256": protocol_sha,
        "verified_bindings": verified,
        "raw_file_count": len(raw_files),
        "raw_file_sha256": manifest,
        "raw_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sealed_seeds_opened": False,
        "development_seed_overlap": False,
        "sealed_seed_overlap": False,
    }
    checks["integrity_audit"] = integrity["status"] == "pass"
    gate["passed"] = bool(all(checks.values()))

    analysis = {
        "schema_version": 1,
        "status": "complete",
        "scope": "single_dynamic_obstacle_confirmatory_held_out",
        "primary": primary,
        "by_split": by_split,
        "ablations": ablations,
        "timing": timing,
        "mechanism": {
            "candidate_tracker_enabled_fraction": float(np.mean(tracker_values)),
            "candidate_forecast_valid_fraction": float(np.mean(forecast_values)),
        },
        "formal_superiority": superiority,
        "gate": gate,
        "protocol_sha256": protocol_sha,
        "sealed_seeds_opened": False,
    }
    _write_json(output / "analysis" / "formal_statistics.json", analysis)
    _write_json(output / "analysis" / "gate.json", gate)
    _write_json(output / "integrity_audit.json", integrity)
    print(json.dumps({
        "gate": gate,
        "formal_superiority": superiority,
        "integrity_status": integrity["status"],
    }, indent=2, sort_keys=True))
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    analyze(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
