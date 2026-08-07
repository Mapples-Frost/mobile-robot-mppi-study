"""Logging-only diagnostics shared by MPPI optimizer implementations."""

import numpy as np


def reverse_candidate_diagnostics(
    samples,
    costs,
    optimizer_feasible,
    jointly_feasible,
    static_feasible,
    risk_feasible,
    weighting_indices,
    weighting_weights,
    v_index,
    prefix_steps=12,
    weighting_population_kind="unknown",
):
    """Summarize reverse coverage after each optimizer gate.

    The helper reads an already-finalized candidate population and normalized
    weights. It does not mutate either one and has no control-path authority.
    """

    samples = np.asarray(samples)
    costs = np.asarray(costs)
    optimizer_feasible = np.asarray(optimizer_feasible, dtype=bool)
    jointly_feasible = np.asarray(jointly_feasible, dtype=bool)
    static_feasible = np.asarray(static_feasible, dtype=bool)
    risk_feasible = np.asarray(risk_feasible, dtype=bool)
    weighting_indices = np.asarray(weighting_indices, dtype=np.int64)
    weighting_weights = np.asarray(weighting_weights, dtype=np.float64)

    reverse = samples[:, 0, int(v_index)] < 0.0
    prefix_steps = max(1, min(int(prefix_steps), int(samples.shape[1])))
    sustained_reverse = np.all(
        samples[:, :prefix_steps, int(v_index)] < 0.0,
        axis=1,
    )
    reverse_weighted = reverse[weighting_indices]
    sustained_reverse_weighted = sustained_reverse[weighting_indices]
    reverse_optimizer_feasible = reverse & optimizer_feasible

    best_reverse_cost = 0.0
    if np.any(reverse):
        best_reverse_cost = float(np.min(costs[reverse]))

    best_feasible_cost = 0.0
    best_feasible_first_v = 0.0
    if np.any(reverse_optimizer_feasible):
        feasible_indices = np.flatnonzero(reverse_optimizer_feasible)
        best_index = int(
            feasible_indices[np.argmin(costs[feasible_indices])]
        )
        best_feasible_cost = float(costs[best_index])
        best_feasible_first_v = float(samples[best_index, 0, int(v_index)])

    global_min_index = int(np.argmin(costs))
    diagnostics = {
        "reverse_candidate_count": int(np.sum(reverse)),
        "reverse_static_feasible_count": int(
            np.sum(reverse & static_feasible)
        ),
        "reverse_risk_feasible_count": int(
            np.sum(reverse & risk_feasible)
        ),
        "reverse_jointly_feasible_count": int(
            np.sum(reverse & jointly_feasible)
        ),
        "reverse_optimizer_feasible_count": int(
            np.sum(reverse_optimizer_feasible)
        ),
        "reverse_weighted_population_count": int(
            np.sum(reverse_weighted)
        ),
        "reverse_weighted_population_weight_mass": float(
            np.sum(weighting_weights[reverse_weighted])
        ),
        "reverse_best_candidate_cost": best_reverse_cost,
        "reverse_best_optimizer_feasible_cost": best_feasible_cost,
        "reverse_best_optimizer_feasible_first_v": best_feasible_first_v,
        "reverse_sustained_prefix_steps": int(prefix_steps),
        "reverse_sustained_candidate_count": int(
            np.sum(sustained_reverse)
        ),
        "reverse_sustained_optimizer_feasible_count": int(
            np.sum(sustained_reverse & optimizer_feasible)
        ),
        "reverse_sustained_weighted_population_count": int(
            np.sum(sustained_reverse_weighted)
        ),
        "reverse_sustained_weighted_population_weight_mass": float(
            np.sum(weighting_weights[sustained_reverse_weighted])
        ),
        "weighting_population_count": int(weighting_indices.size),
        "weighting_population_kind": str(weighting_population_kind),
        "optimizer_feasible_candidate_count": int(
            np.sum(optimizer_feasible)
        ),
        "jointly_feasible_candidate_count": int(
            np.sum(jointly_feasible)
        ),
        "weight_global_min_was_masked": bool(
            not optimizer_feasible[global_min_index]
        ),
    }
    if weighting_population_kind == "elite":
        # Retain the original Paper-RL naming for the frozen weighting
        # diagnostic while also exposing optimizer-agnostic fields above.
        diagnostics.update({
            "reverse_elite_count": int(np.sum(reverse_weighted)),
            "reverse_elite_weight_mass": float(
                np.sum(weighting_weights[reverse_weighted])
            ),
            "reverse_sustained_elite_count": int(
                np.sum(sustained_reverse_weighted)
            ),
            "reverse_sustained_elite_weight_mass": float(
                np.sum(weighting_weights[sustained_reverse_weighted])
            ),
        })
    return diagnostics
