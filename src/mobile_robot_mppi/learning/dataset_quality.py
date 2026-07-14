"""Quality gates for residual-learning transition datasets."""

from itertools import combinations
from typing import Any, Dict, Mapping

import numpy as np


def _column(dataset, name):
    if isinstance(dataset, Mapping):
        return np.asarray(dataset[name])
    return np.asarray(getattr(dataset, name))


def residual_dataset_quality(dataset, angle_indices=(2,)) -> Dict[str, Any]:
    state = _column(dataset, "state_t").astype(np.float64, copy=False)
    following = _column(dataset, "state_t_plus_1").astype(np.float64, copy=False)
    observed = _column(dataset, "observed_derivative").astype(np.float64, copy=False)
    nominal = _column(dataset, "nominal_derivative").astype(np.float64, copy=False)
    residual = _column(dataset, "residual_target").astype(np.float64, copy=False)
    control = _column(dataset, "control_t").astype(np.float64, copy=False)
    applied = _column(dataset, "applied_control_t").astype(np.float64, copy=False)
    dt = _column(dataset, "dt").astype(np.float64, copy=False).reshape(-1, 1)
    identity_error = observed - nominal - residual
    reconstructed = state + dt * observed
    transition_error = reconstructed - following
    for index in angle_indices:
        transition_error[:, int(index)] = np.arctan2(
            np.sin(transition_error[:, int(index)]),
            np.cos(transition_error[:, int(index)]),
        )
    command_error = applied - control
    episode = _column(dataset, "episode_id")
    return {
        "transition_count": int(state.shape[0]),
        "episode_count": int(np.unique(episode).size),
        "state_dim": int(state.shape[1]),
        "control_dim": int(control.shape[1]),
        "residual_identity_max_abs": float(np.max(np.abs(identity_error))),
        "transition_closure_max_abs": float(np.max(np.abs(transition_error))),
        "residual_rms_per_state": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
        "command_applied_rmse": float(np.sqrt(np.mean(command_error ** 2))),
        "command_applied_mismatch_fraction": float(
            np.mean(np.any(np.abs(command_error) > 1e-12, axis=1))
        ),
    }


def assert_residual_dataset_quality(dataset, angle_indices=(2,), tolerance=1e-9):
    report = residual_dataset_quality(dataset, angle_indices)
    if report["residual_identity_max_abs"] > float(tolerance):
        raise ValueError("residual labels do not equal observed minus nominal derivative")
    if report["transition_closure_max_abs"] > float(tolerance):
        raise ValueError("observed derivatives do not reconstruct the stored transitions")
    return report


def assert_episode_disjoint_splits(splits) -> Dict[str, Any]:
    names = ("train", "validation", "test", "unseen")
    episodes = {
        name: {str(value) for value in np.unique(_column(getattr(splits, name), "episode_id"))}
        for name in names
    }
    for left, right in combinations(names, 2):
        overlap = episodes[left] & episodes[right]
        if overlap:
            raise ValueError("episode leakage between %s and %s: %s" % (
                left, right, sorted(overlap)
            ))
    return {
        "episode_ids": {name: sorted(values) for name, values in episodes.items()},
        "episode_counts": {name: len(values) for name, values in episodes.items()},
    }
