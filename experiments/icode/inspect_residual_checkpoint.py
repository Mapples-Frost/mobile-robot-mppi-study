#!/usr/bin/env python3
"""Print a safe, tensor-free JSON summary of a residual checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.learning.checkpointing import load_checkpoint  # noqa: E402
from src.learning.normalization import NormalizerBundle  # noqa: E402


def _plain(value: Any, name: str = "value") -> Any:
    """Convert metadata to JSON values and refuse accidental tensor output."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path, os.PathLike, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, torch.Tensor):
        raise TypeError("{} unexpectedly contains a tensor".format(name))
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), name)
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, "{}.{}".format(name, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _plain(item, "{}[{}]".format(name, index))
            for index, item in enumerate(value)
        ]
    raise TypeError(
        "{} has unsupported type {}".format(name, type(value).__name__)
    )


def _parameter_count(model_state: Mapping[str, Any]) -> int:
    count = 0
    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("model_state.{} is not a tensor".format(name))
        count += int(value.numel())
    return count


def _normalizer_part(normalizer: Any) -> Dict[str, Any]:
    return {
        "feature_dim": int(normalizer.feature_dim),
        "sample_count": int(normalizer.sample_count),
        "epsilon": float(normalizer.epsilon),
        "mean": normalizer.mean.tolist(),
        "scale": normalizer.scale.tolist(),
        "constant_mask": normalizer.constant_mask.tolist(),
    }


def summarize_checkpoint(path: Path) -> Dict[str, Any]:
    """Load ``path`` on CPU with Torch's restricted loader and summarize it."""

    source = Path(path).expanduser().resolve()
    checkpoint = load_checkpoint(
        source,
        map_location="cpu",
        weights_only=True,
        restore_rng=False,
    )
    normalizers = NormalizerBundle.from_state_dict(
        checkpoint["normalizer_bundle_state"]
    )
    model_config = _plain(checkpoint["model_config"], "model_config")
    state_feature_dim = model_config.get(
        "state_feature_dim", normalizers.state.feature_dim
    )
    summary = {
        "checkpoint": str(source),
        "format": checkpoint.get("format"),
        "format_version": int(checkpoint["format_version"]),
        "schema_version": int(checkpoint["schema_version"]),
        "model_class": str(checkpoint["model_class"]),
        "model_type": model_config.get("model_type"),
        "state_dim": int(checkpoint["state_dim"]),
        "state_feature_dim": int(state_feature_dim),
        "control_dim": int(checkpoint["control_dim"]),
        "parameter_count": _parameter_count(checkpoint["model_state"]),
        "model_config": model_config,
        "epoch": int(checkpoint["epoch"]),
        "best_validation_metric": float(checkpoint["best_validation_metric"]),
        "git_sha": str(checkpoint["git_sha"]),
        "config": _plain(checkpoint["config"], "config"),
        "normalizer": {
            "fit_sample_count": int(normalizers.fit_sample_count),
            "encoder_config": _plain(
                normalizers.encoder_config, "normalizer.encoder_config"
            ),
            "state": _normalizer_part(normalizers.state),
            "control": _normalizer_part(normalizers.control),
            "residual": _normalizer_part(normalizers.residual),
        },
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint file to inspect")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.indent < 0:
        print("error: --indent must be non-negative", file=sys.stderr)
        return 1
    try:
        summary = summarize_checkpoint(args.checkpoint)
        rendered = json.dumps(
            summary,
            indent=args.indent,
            sort_keys=True,
            allow_nan=False,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
