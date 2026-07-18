"""Paired checkpoint comparisons for closed-loop controller experiments."""

from collections import defaultdict

import numpy as np


DEFAULT_METRICS = {
    "success": True,
    "collision": False,
    "final_goal_distance": False,
    "control_jerk": False,
    "minimum_clearance": True,
    "stuck_steps": False,
    "spin_steps": False,
    "planner_compute_ms_mean": False,
}


def paired_checkpoint_schedule(seeds, domains, scenes, schedule_seed):
    """Randomize control/aligned order within every repeated-measures block."""

    variants = ("control", "value_aligned")
    rng = np.random.RandomState(int(schedule_seed))
    result = []
    for scene in scenes:
        for domain in domains:
            for seed in seeds:
                block = "%s__%s__seed_%d" % (
                    scene["name"], domain["name"], int(seed)
                )
                for position, index in enumerate(rng.permutation(2)):
                    result.append({
                        "variant": variants[int(index)],
                        "scene": str(scene["name"]),
                        "physics_domain": str(domain["name"]),
                        "seed": int(seed),
                        "block": block,
                        "run_order_within_block": int(position),
                    })
    return result


def _as_float(value, name):
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _selected_index(rows, method, key_fields):
    selected = {}
    for row in rows:
        if str(row["method"]) != str(method):
            continue
        key = tuple(str(row[field]) for field in key_fields)
        if key in selected:
            raise ValueError("duplicate paired cell: %s" % (key,))
        selected[key] = row
    if not selected:
        raise ValueError("no rows found for method %s" % method)
    return selected


def paired_checkpoint_effects(
    control_rows,
    aligned_rows,
    method="simple_combination",
    key_fields=("scene", "physics_domain", "seed"),
    cluster_field="seed",
    metrics=None,
    bootstrap_samples=5000,
    seed=20260719,
):
    """Compare two checkpoints with positive values meaning improvement.

    Scene/domain cells are repeated strata. Bootstrap resampling occurs over
    seed clusters so repeated cells from one simulation seed are never treated
    as independent observations.
    """

    metrics = dict(DEFAULT_METRICS if metrics is None else metrics)
    control = _selected_index(control_rows, method, key_fields)
    aligned = _selected_index(aligned_rows, method, key_fields)
    if set(control) != set(aligned):
        raise ValueError(
            "paired cells differ: missing aligned=%s, missing control=%s"
            % (
                sorted(set(control) - set(aligned)),
                sorted(set(aligned) - set(control)),
            )
        )
    keys = sorted(control)
    clusters = sorted(set(key[key_fields.index(cluster_field)] for key in keys))
    if len(clusters) < 2 and int(bootstrap_samples) > 0:
        raise ValueError("cluster bootstrap requires at least two clusters")

    result = {
        "method": str(method),
        "key_fields": list(key_fields),
        "paired_cells": len(keys),
        "cluster_field": str(cluster_field),
        "independent_clusters": len(clusters),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "metrics": {},
    }
    rng = np.random.RandomState(int(seed))
    for metric, higher_is_better in metrics.items():
        raw = []
        by_cluster = defaultdict(list)
        control_values = []
        aligned_values = []
        for key in keys:
            before = _as_float(control[key][metric], metric)
            after = _as_float(aligned[key][metric], metric)
            effect = after - before if higher_is_better else before - after
            raw.append(effect)
            control_values.append(before)
            aligned_values.append(after)
            by_cluster[key[key_fields.index(cluster_field)]].append(effect)
        cluster_effects = np.asarray(
            [np.mean(by_cluster[name]) for name in clusters],
            dtype=np.float64,
        )
        estimate = float(np.mean(cluster_effects))
        samples = []
        for _ in range(int(bootstrap_samples)):
            indices = rng.randint(0, len(clusters), size=len(clusters))
            samples.append(float(np.mean(cluster_effects[indices])))
        samples = np.asarray(samples, dtype=np.float64)
        baseline_mean = float(np.mean(control_values))
        relative = (
            None
            if abs(baseline_mean) < 1e-12
            else float(estimate / abs(baseline_mean))
        )
        standard_deviation = (
            None
            if cluster_effects.size < 2
            else float(np.std(cluster_effects, ddof=1))
        )
        result["metrics"][metric] = {
            "higher_is_better": bool(higher_is_better),
            "control_mean": baseline_mean,
            "aligned_mean": float(np.mean(aligned_values)),
            "favorable_effect": estimate,
            "relative_favorable_change": relative,
            "favorable": bool(estimate > 0.0),
            "ci95": (
                None
                if samples.size == 0
                else [
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ]
            ),
            "paired_cohen_dz": (
                None
                if (
                    standard_deviation is None
                    or standard_deviation < 1e-12
                )
                else float(estimate / standard_deviation)
            ),
            "per_cluster": [
                float(value) for value in cluster_effects
            ],
            "per_cell": [float(value) for value in raw],
        }
    return result


def gate2_closed_loop_decision(result, tolerance=1e-12):
    """Apply the frozen Gate 2 closed-loop development criterion."""

    metrics = result["metrics"]
    goal = metrics["final_goal_distance"]["favorable_effect"]
    jerk = metrics["control_jerk"]["favorable_effect"]
    success = metrics["success"]["favorable_effect"]
    collision = metrics["collision"]["favorable_effect"]
    positive_primary = goal > tolerance or jerk > tolerance
    no_worsening = all(
        value >= -tolerance for value in (goal, jerk, success, collision)
    )
    return {
        "positive_primary_outcome": bool(positive_primary),
        "goal_distance_not_worse": bool(goal >= -tolerance),
        "control_jerk_not_worse": bool(jerk >= -tolerance),
        "success_not_worse": bool(success >= -tolerance),
        "collision_not_worse": bool(collision >= -tolerance),
        "closed_loop_development_passed": bool(
            positive_primary and no_worsening
        ),
    }
