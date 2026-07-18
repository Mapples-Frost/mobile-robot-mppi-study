"""Blocked 2x2 factorial contrasts for ICODE-by-RL experiments."""

from collections import defaultdict

import numpy as np


CELL_METHODS = {
    (0, 0): "traditional_mppi",
    (1, 0): "icode_mppi",
    (0, 1): "rl_driven_mppi",
    (1, 1): "simple_combination",
}


def _finite_float(value, metric, block):
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "%s is not numeric in block %s" % (metric, block)
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            "%s is not finite in block %s" % (metric, block)
        )
    return result


def complete_factorial_blocks(rows, metric):
    """Return one four-cell response vector per randomized complete block."""

    grouped = defaultdict(dict)
    for row in rows:
        block = str(row["block"])
        method = str(row["method"])
        if method in grouped[block]:
            raise ValueError(
                "duplicated method %s in block %s" % (method, block)
            )
        grouped[block][method] = _finite_float(
            row[metric], metric, block
        )
    expected = set(CELL_METHODS.values())
    incomplete = {
        block: sorted(expected - set(cells))
        for block, cells in grouped.items()
        if set(cells) != expected
    }
    if incomplete:
        raise ValueError(
            "factorial blocks are incomplete: %s" % incomplete
        )
    if not grouped:
        raise ValueError("factorial analysis requires at least one block")
    names = sorted(grouped)
    values = np.asarray([
        [
            grouped[block][CELL_METHODS[(0, 0)]],
            grouped[block][CELL_METHODS[(1, 0)]],
            grouped[block][CELL_METHODS[(0, 1)]],
            grouped[block][CELL_METHODS[(1, 1)]],
        ]
        for block in names
    ], dtype=np.float64)
    return names, values


def _contrasts(values):
    traditional, icode, rl, combination = values.T
    return {
        "icode_main_effect": 0.5 * (
            (icode - traditional) + (combination - rl)
        ),
        "rl_main_effect": 0.5 * (
            (rl - traditional) + (combination - icode)
        ),
        "icode_by_rl_interaction": (
            combination - icode - rl + traditional
        ),
        "combination_vs_traditional": combination - traditional,
        "combination_vs_icode": combination - icode,
        "combination_vs_rl": combination - rl,
    }


def blocked_factorial_contrasts(
    rows,
    metric,
    higher_is_better=False,
    bootstrap_samples=5000,
    seed=20260718,
    cluster_key=None,
):
    """Estimate paired effects and resample independent blocks.

    Controller timesteps never enter this calculation.  The resampling unit is
    the declared scene-by-physics-domain-by-seed block.
    """

    blocks, values = complete_factorial_blocks(rows, metric)
    block_effects = _contrasts(values)
    cluster_labels = list(blocks)
    if cluster_key is not None:
        labels_by_block = {}
        for row in rows:
            block = str(row["block"])
            label = str(row[cluster_key])
            existing = labels_by_block.setdefault(block, label)
            if existing != label:
                raise ValueError(
                    "cluster label changes within block %s" % block
                )
        cluster_labels = [labels_by_block[block] for block in blocks]
    unique_clusters = sorted(set(cluster_labels))
    effects = {}
    for name, per_block in block_effects.items():
        effects[name] = np.asarray([
            float(np.mean([
                per_block[index]
                for index, label in enumerate(cluster_labels)
                if label == cluster
            ]))
            for cluster in unique_clusters
        ], dtype=np.float64)
    sample_count = int(bootstrap_samples)
    if sample_count < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    bootstrap = {name: [] for name in effects}
    if sample_count and len(unique_clusters) >= 2:
        rng = np.random.RandomState(int(seed))
        for _ in range(sample_count):
            indices = rng.randint(
                0, len(unique_clusters), size=len(unique_clusters)
            )
            for name, effect in effects.items():
                bootstrap[name].append(float(np.mean(effect[indices])))
    result = {
        "metric": str(metric),
        "higher_is_better": bool(higher_is_better),
        "independent_unit": (
            "factorial block"
            if cluster_key is None
            else "%s cluster; scene/domain blocks are repeated strata"
            % str(cluster_key)
        ),
        "blocks": len(blocks),
        "independent_clusters": len(unique_clusters),
        "cluster_key": cluster_key,
        "bootstrap_samples": sample_count,
        "bootstrap_seed": int(seed),
        "effects": {},
    }
    for name, per_cluster in effects.items():
        estimate = float(np.mean(per_cluster))
        samples = np.asarray(bootstrap[name], dtype=np.float64)
        result["effects"][name] = {
            "estimate": estimate,
            "favorable_direction": (
                "positive" if higher_is_better else "negative"
            ),
            "favorable": bool(
                estimate > 0.0 if higher_is_better else estimate < 0.0
            ),
            "ci95": (
                None
                if samples.size == 0
                else [
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ]
            ),
            "per_block": [
                float(value) for value in block_effects[name]
            ],
            "per_cluster": [float(value) for value in per_cluster],
        }
    return result
