"""Frozen configuration and seed-cluster statistics for predictor qualification."""

import copy
import itertools
from typing import Mapping, Sequence

import numpy as np


def deep_merge_ood_config(
    base: Mapping[str, object], override: Mapping[str, object]
):
    """Apply a registered OOD overlay without mutating frozen V3."""

    def merge(left, right, path=()):
        result = copy.deepcopy(left)
        for key, value in right.items():
            if key == "base_config_path":
                continue
            if key == "noise_profiles":
                result[key] = copy.deepcopy(value)
            elif (
                key in result
                and isinstance(result[key], Mapping)
                and isinstance(value, Mapping)
            ):
                result[key] = merge(result[key], value, path + (key,))
            else:
                result[key] = copy.deepcopy(value)
        return result

    result = merge(base, override)
    if result["physical_limits"] != base["physical_limits"]:
        raise ValueError("OOD overlay may not change physical limits")
    if result["generator_version"] != base["generator_version"]:
        raise ValueError("OOD overlay may not change generator implementation")
    return result


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Exact two-sided paired randomization test on seed-level differences."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired differences must be a finite nonempty vector")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 2 ** int(values.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        extreme += int(statistic >= observed - 1.0e-15)
    return float(extreme / total)


def cluster_bootstrap_mean_ci(
    differences: Sequence[float],
    seed: int,
    resamples: int = 10000,
    confidence: float = 0.95,
):
    """Percentile CI resampling whole seed-level differences."""
    values = np.asarray(differences, dtype=np.float64)
    resamples = int(resamples)
    confidence = float(confidence)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or resamples <= 0
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("bootstrap inputs are invalid")
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(
        0, values.size, size=(resamples, values.size)
    )
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def paired_seed_summary(
    differences: Sequence[float],
    bootstrap_seed: int,
    bootstrap_resamples: int = 10000,
):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired summary needs at least two finite seeds")
    standard_deviation = float(values.std(ddof=1))
    interval = cluster_bootstrap_mean_ci(
        values, bootstrap_seed, bootstrap_resamples
    )
    return {
        "seed_count": int(values.size),
        "differences": values.tolist(),
        "mean_difference": float(values.mean()),
        "standard_deviation": standard_deviation,
        "median_difference": float(np.median(values)),
        "paired_standardized_effect_dz": (
            None
            if standard_deviation == 0.0
            else float(values.mean() / standard_deviation)
        ),
        "bootstrap_95_ci": list(interval),
        "exact_sign_flip_p": exact_sign_flip_pvalue(values),
    }


def holm_adjust(p_values: Mapping[str, float]):
    """Holm family-wise p-value adjustment with monotonicity enforcement."""
    items = sorted(
        ((str(name), float(value)) for name, value in p_values.items()),
        key=lambda item: item[1],
    )
    if not items or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for _, value in items
    ):
        raise ValueError("Holm p-values are invalid")
    count = len(items)
    adjusted = {}
    running = 0.0
    for rank, (name, value) in enumerate(items):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


__all__ = [
    "cluster_bootstrap_mean_ci",
    "deep_merge_ood_config",
    "exact_sign_flip_pvalue",
    "holm_adjust",
    "paired_seed_summary",
]
