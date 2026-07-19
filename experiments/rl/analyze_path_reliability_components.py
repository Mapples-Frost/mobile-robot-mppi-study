#!/usr/bin/env python3
"""Diagnose path-aware reliability components at the episode level.

This is an exploratory diagnostic, not a threshold-selection tool.  It never
changes a calibration artifact and treats episodes, rather than overlapping
rollout windows, as independent units.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


COMPONENTS = (
    "ensemble_disagreement_max",
    "innovation_error_ema",
    "residual_support_confidence_min",
    "actor_ood_score_max",
    "disagreement_confidence",
    "innovation_confidence",
    "dynamics_confidence",
    "actor_confidence",
    "authority",
    "path_cross_track_abs_max",
    "path_curvature_abs_max",
    "path_remaining_mean",
)


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while (
            end < values.size
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.size < 3
        or np.std(left) <= 1e-12
        or np.std(right) <= 1e-12
    ):
        return None
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def _finite_float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s in reliability CSV" % name)
    return value


def load_episode_rows(path):
    grouped = defaultdict(list)
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["split"], row["episode_id"])].append(row)
    if not grouped:
        raise ValueError("reliability CSV contains no windows")
    episodes = []
    for (split, episode_id), rows in sorted(grouped.items()):
        result = {
            "split": split,
            "episode_id": episode_id,
            "seed": int(rows[0]["seed"]),
            "scene": rows[0]["scene"],
            "physics_domain": rows[0]["physics_domain"],
            "windows": len(rows),
            "rollout_error": float(np.mean([
                _finite_float(row, "rollout_error") for row in rows
            ])),
        }
        for component in COMPONENTS:
            result[component] = float(np.mean([
                _finite_float(row, component) for row in rows
            ]))
        episodes.append(result)
    return episodes


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": mean,
        "standard_deviation": float(np.std(values, ddof=1))
        if values.size > 1 else 0.0,
        "coefficient_of_variation": (
            float(np.std(values, ddof=1) / abs(mean))
            if values.size > 1 and abs(mean) > 1e-12 else 0.0
        ),
    }


def summarize_split(rows):
    errors = [row["rollout_error"] for row in rows]
    correlations = {
        component: _spearman(
            [row[component] for row in rows], errors
        )
        for component in COMPONENTS
    }
    component_distributions = {
        component: _distribution([
            row[component] for row in rows
        ])
        for component in COMPONENTS
    }
    by_scene = {}
    for scene in sorted({row["scene"] for row in rows}):
        selected = [row for row in rows if row["scene"] == scene]
        by_scene[scene] = {
            "episodes": len(selected),
            "rollout_error_mean": float(np.mean([
                row["rollout_error"] for row in selected
            ])),
            "authority_mean": float(np.mean([
                row["authority"] for row in selected
            ])),
        }
    by_domain = {}
    for domain in sorted({row["physics_domain"] for row in rows}):
        selected = [
            row for row in rows if row["physics_domain"] == domain
        ]
        by_domain[domain] = {
            "episodes": len(selected),
            "rollout_error_mean": float(np.mean([
                row["rollout_error"] for row in selected
            ])),
            "authority_mean": float(np.mean([
                row["authority"] for row in selected
            ])),
        }
    return {
        "episodes": len(rows),
        "rollout_error": _distribution(errors),
        "spearman_component_vs_rollout_error": correlations,
        "component_distributions": component_distributions,
        "by_scene": by_scene,
        "by_physics_domain": by_domain,
    }


def analyze(name, path):
    episodes = load_episode_rows(path)
    splits = {}
    for split in sorted({row["split"] for row in episodes}):
        splits[split] = summarize_split([
            row for row in episodes if row["split"] == split
        ])
    return {
        "name": name,
        "source": str(Path(path).resolve()),
        "independent_unit": "episode",
        "episode_count": len(episodes),
        "splits": splits,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = {
        "analysis_class": "exploratory_component_diagnostic",
        "thresholds_changed": False,
        "ordinary": analyze("ordinary", args.ordinary),
        "value_aligned": analyze("value_aligned", args.value),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
