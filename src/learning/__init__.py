"""Dataset and normalization utilities for residual dynamics learning."""

from .normalization import NormalizerBundle, StandardNormalizer
from .residual_dataset import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    DatasetSplits,
    ResidualDataset,
    compute_observed_derivative,
    contiguous_rollout_windows,
    group_split,
    rollout_window_indices,
    split_by_episode,
    wrapped_finite_difference,
)

__all__ = [
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "DatasetSplits",
    "NormalizerBundle",
    "ResidualDataset",
    "StandardNormalizer",
    "compute_observed_derivative",
    "contiguous_rollout_windows",
    "group_split",
    "rollout_window_indices",
    "split_by_episode",
    "wrapped_finite_difference",
]
