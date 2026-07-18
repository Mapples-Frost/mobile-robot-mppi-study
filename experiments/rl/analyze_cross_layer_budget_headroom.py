#!/usr/bin/env python3
"""Audit L98 and estimate hindsight headroom for an online budget policy.

This is explicitly exploratory.  It never changes the L98 preregistered Gate.
The per-block oracle uses outcomes that are unavailable online and therefore
serves only as an upper bound and as a label source for the next preregistered
oracle-timescale study.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.run_covariance_context_oracle import _hierarchical_ci


CONDITIONS = (
    "icode_fixed_k50",
    "icode_contextual_k50",
    "icode_fixed_k100",
)
OUTCOMES = (
    "cross_track_rmse",
    "elapsed_s",
    "planner_compute_ms_mean",
    "control_jerk",
    "applied_control_jerk",
)
OBSERVABLE_DIAGNOSTICS = (
    "effective_sample_size_mean",
    "effective_sample_size_min",
    "sample_saturation_fraction_mean",
    "mean_slip_ratio",
    "planner_compute_ms_p95",
)


def _average_tie_ranks(values):
    """Return one-based average ranks without adding a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman_rho(left, right):
    """Compute Spearman's rho; return None when either input is constant."""
    left_rank = _average_tie_ranks(left)
    right_rank = _average_tie_ranks(right)
    if left_rank.size < 2:
        return None
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = np.sqrt(
        np.sum(left_centered ** 2) * np.sum(right_centered ** 2)
    )
    if denominator <= 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row, key):
    value = row.get(key)
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _boolean(row, key):
    return str(row.get(key, "")).strip().lower() in ("1", "true", "yes")


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _index(rows):
    indexed = {}
    for row in rows:
        key = (
            str(row["scene"]), str(row["physics_domain"]), int(row["seed"]),
            str(row["condition"]),
        )
        if key in indexed:
            raise ValueError("duplicate episode key: %r" % (key,))
        indexed[key] = row
    return indexed


def _select_oracle(indexed, rmse_margin_m):
    contexts = sorted({key[:3] for key in indexed})
    decisions = []
    for scene, domain, seed in contexts:
        arms = {
            condition: indexed[(scene, domain, seed, condition)]
            for condition in CONDITIONS
        }
        reference = arms["icode_fixed_k100"]
        eligible = []
        for condition in ("icode_fixed_k50", "icode_contextual_k50"):
            candidate = arms[condition]
            safe = bool(
                _boolean(candidate, "success") >= _boolean(reference, "success")
                and _boolean(candidate, "collision") <= _boolean(reference, "collision")
            )
            precise = bool(
                _number(candidate, "cross_track_rmse")
                <= _number(reference, "cross_track_rmse") + float(rmse_margin_m)
            )
            if safe and precise:
                eligible.append(condition)
        if eligible:
            selected = min(eligible, key=lambda condition: (
                _number(arms[condition], "elapsed_s"),
                _number(arms[condition], "planner_compute_ms_mean"),
                _number(arms[condition], "cross_track_rmse"),
                condition,
            ))
        else:
            selected = "icode_fixed_k100"
        chosen = arms[selected]
        current = arms["icode_contextual_k50"]
        record = {
            "scene": scene,
            "physics_domain": domain,
            "seed": seed,
            "selected_condition": selected,
            "needs_k100": selected == "icode_fixed_k100",
            "eligible_k50_count": len(eligible),
            "eligible_k50_conditions": "|".join(sorted(eligible)),
            "current_excess_rmse_m": (
                _number(current, "cross_track_rmse")
                - _number(reference, "cross_track_rmse")
            ),
        }
        for metric in OUTCOMES:
            record["oracle_" + metric] = _number(chosen, metric)
            record["k100_" + metric] = _number(reference, metric)
            record["current_" + metric] = _number(current, metric)
            record["oracle_minus_k100_" + metric] = (
                _number(chosen, metric) - _number(reference, metric)
            )
            record["oracle_minus_current_" + metric] = (
                _number(chosen, metric) - _number(current, metric)
            )
        for feature in OBSERVABLE_DIAGNOSTICS:
            record["current_" + feature] = _number(current, feature)
        decisions.append(record)
    return decisions


def _contrast(decisions, prefix, bootstrap_seed):
    output = {"paired_blocks": len(decisions)}
    for offset, metric in enumerate(OUTCOMES):
        key = "%s_%s" % (prefix, metric)
        grouped = defaultdict(list)
        for row in decisions:
            grouped[row["scene"] + "__" + row["physics_domain"]].append(
                float(row[key])
            )
        flat = [value for values in grouped.values() for value in values]
        output[metric + "_delta_mean"] = float(np.mean(flat))
        output[metric + "_delta_ci95"] = _hierarchical_ci(
            dict(grouped), int(bootstrap_seed) + offset
        )
    return output


def _majority_accuracy(decisions, group_keys):
    groups = defaultdict(list)
    for row in decisions:
        key = tuple(row[value] for value in group_keys)
        groups[key].append(str(row["selected_condition"]))
    correct = 0
    rules = {}
    for key, labels in groups.items():
        majority = sorted(Counter(labels).items(), key=lambda item: (-item[1], item[0]))[0][0]
        rules["|".join(map(str, key))] = majority
        correct += sum(label == majority for label in labels)
    return {
        "group_keys": list(group_keys),
        "accuracy": float(correct / len(decisions)),
        "rules": rules,
    }


def _diagnostic_associations(decisions):
    output = {}
    target = np.asarray(
        [row["current_excess_rmse_m"] for row in decisions], dtype=np.float64
    )
    for feature in OBSERVABLE_DIAGNOSTICS:
        values = [row["current_" + feature] for row in decisions]
        available = [
            index for index, value in enumerate(values) if value is not None
        ]
        if len(available) < 4 or len({values[index] for index in available}) < 2:
            output[feature] = {
                "available": len(available),
                "spearman_rho": None,
                "p_value_exploratory": None,
            }
            continue
        rho = _spearman_rho(
            [values[index] for index in available], target[available]
        )
        output[feature] = {
            "available": len(available),
            "spearman_rho": rho,
            "p_value_exploratory": None,
            "inference_note": (
                "Exploratory rank association only; no uncorrected small-sample "
                "p-value is reported."
            ),
        }
    return output


def _group_rates(decisions, keys):
    groups = defaultdict(list)
    for row in decisions:
        key = "|".join(str(row[value]) for value in keys)
        groups[key].append(row)
    return {
        key: {
            "blocks": len(values),
            "needs_k100_fraction": float(np.mean([row["needs_k100"] for row in values])),
            "current_excess_rmse_m_mean": float(np.mean([
                row["current_excess_rmse_m"] for row in values
            ])),
        }
        for key, values in sorted(groups.items())
    }


def analyze(rows, rmse_margin_m=0.002, bootstrap_seed=2026072001):
    indexed = _index(rows)
    expected_conditions = set(CONDITIONS)
    contexts = sorted({key[:3] for key in indexed})
    missing = {
        context: sorted(expected_conditions - {
            key[3] for key in indexed if key[:3] == context
        })
        for context in contexts
    }
    missing = {str(key): value for key, value in missing.items() if value}
    if missing:
        raise ValueError("incomplete ICODE blocks: %s" % missing)
    decisions = _select_oracle(indexed, rmse_margin_m)
    selected_counts = Counter(row["selected_condition"] for row in decisions)
    oracle_vs_k100 = _contrast(decisions, "oracle_minus_k100", bootstrap_seed)
    oracle_vs_current = _contrast(
        decisions, "oracle_minus_current", bootstrap_seed + 20
    )
    summary = {
        "schema_version": 1,
        "analysis_type": "exploratory_hindsight_oracle",
        "rows": len(rows),
        "columns": len({key for row in rows for key in row}),
        "icode_blocks": len(decisions),
        "duplicate_episode_keys": len(rows) - len(indexed),
        "rmse_noninferiority_margin_m": float(rmse_margin_m),
        "selected_condition_counts": dict(sorted(selected_counts.items())),
        "needs_k100_fraction": float(np.mean([row["needs_k100"] for row in decisions])),
        "both_k50_eligible_fraction": float(np.mean([
            row["eligible_k50_count"] == 2 for row in decisions
        ])),
        "oracle_vs_fixed_k100": oracle_vs_k100,
        "oracle_vs_current_contextual_k50": oracle_vs_current,
        "route_only_majority": _majority_accuracy(decisions, ("scene",)),
        "route_physics_majority": _majority_accuracy(
            decisions, ("scene", "physics_domain")
        ),
        "by_scene": _group_rates(decisions, ("scene",)),
        "by_physics": _group_rates(decisions, ("physics_domain",)),
        "by_scene_physics": _group_rates(
            decisions, ("scene", "physics_domain")
        ),
        "observable_diagnostic_associations": _diagnostic_associations(decisions),
        "interpretation_boundary": (
            "The oracle uses completed outcomes and is not deployable. Its only "
            "purpose is to test whether a richer online policy has potential headroom."
        ),
    }
    return summary, decisions


def _markdown(summary, source_path):
    k100 = summary["oracle_vs_fixed_k100"]
    current = summary["oracle_vs_current_contextual_k50"]
    lines = [
        "# L98 exploratory budget-headroom EDA",
        "",
        "Source: `%s`  " % source_path,
        "Analysis type: exploratory hindsight oracle; not a confirmatory result",
        "",
        "## Data quality",
        "",
        "- rows: %d; columns: %d; complete ICODE blocks: %d;" % (
            summary["rows"], summary["columns"], summary["icode_blocks"]
        ),
        "- duplicate episode keys: %d;" % summary["duplicate_episode_keys"],
        "- K100 selected in %.1f%% of blocks;" % (
            100.0 * summary["needs_k100_fraction"]
        ),
        "- both K50 candidates eligible in %.1f%% of blocks." % (
            100.0 * summary["both_k50_eligible_fraction"]
        ),
        "",
        "## Hindsight oracle contrasts",
        "",
        "| Contrast | RMSE delta | 95% CI | compute delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
        "| Oracle - fixed K100 | %.6f m | [%.6f, %.6f] | %.3f ms | [%.3f, %.3f] |" % (
            k100["cross_track_rmse_delta_mean"],
            *k100["cross_track_rmse_delta_ci95"],
            k100["planner_compute_ms_mean_delta_mean"],
            *k100["planner_compute_ms_mean_delta_ci95"],
        ),
        "| Oracle - current contextual K50 | %.6f m | [%.6f, %.6f] | %.3f ms | [%.3f, %.3f] |" % (
            current["cross_track_rmse_delta_mean"],
            *current["cross_track_rmse_delta_ci95"],
            current["planner_compute_ms_mean_delta_mean"],
            *current["planner_compute_ms_mean_delta_ci95"],
        ),
        "",
        "## Context sufficiency",
        "",
        "- route-only majority-action accuracy: %.3f;" % (
            summary["route_only_majority"]["accuracy"]
        ),
        "- route + hidden physics-label majority accuracy: %.3f." % (
            summary["route_physics_majority"]["accuracy"]
        ),
        "",
        "A gain from the hidden physics label is evidence that route geometry alone "
        "cannot express the useful decision.  It does not authorize using simulator "
        "labels online; the next study must replace them with causal, measurable "
        "innovation and MPPI diagnostics.",
        "",
        "## Boundary",
        "",
        summary["interpretation_boundary"],
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rmse-margin-m", type=float, default=0.002)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072001)
    args = parser.parse_args(argv)
    rows = _read_csv(args.episodes)
    summary, decisions = analyze(
        rows, args.rmse_margin_m, args.bootstrap_seed
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "headroom_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csv(output / "headroom_block_decisions.csv", decisions)
    with (output / "headroom_eda_report.md").open("w", encoding="utf-8") as handle:
        handle.write(_markdown(summary, args.episodes))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
