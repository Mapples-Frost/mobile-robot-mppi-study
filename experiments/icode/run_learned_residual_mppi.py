#!/usr/bin/env python3
"""Run clean MPPI with a learned residual checkpoint.

The learned planner is constructed exclusively from the nominal unicycle and
the supplied checkpoint.  True disturbance parameters are instantiated later,
inside the execution benchmark, and are never passed to the learned model or
its prediction adapter.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


os.environ.setdefault("MPLBACKEND", "Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.icode.run_oracle_residual_ablation import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_config,
    run_modes,
)
from src.dynamics import (  # noqa: E402
    CombinedDynamics,
    DisturbanceConfig,
    DisturbedUnicycle,
)
from src.learning.checkpointing import load_checkpoint  # noqa: E402
from src.planners.mppi_dynamics_adapter import (  # noqa: E402
    LearnedResidualDynamics,
    MppiDynamicsAdapter,
    build_prediction_dynamics,
)


SUPPORTED_MODEL_TYPES = ("mlp_residual", "icode_residual")
MODEL_TYPE_ALIASES = {
    "mlp": "mlp_residual",
    "mlp_residual": "mlp_residual",
    "icode": "icode_residual",
    "icode_residual": "icode_residual",
}


def _resolve_device(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return str(device)


def _checkpoint_model_type(checkpoint: Mapping[str, Any]) -> str:
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint model_config is missing")
    value = str(config.get("model_type", "")).strip().lower()
    if value not in SUPPORTED_MODEL_TYPES:
        raise ValueError("unsupported checkpoint model_type {!r}".format(value))
    return value


def _select_model_type(requested: str, checkpoint_type: str) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return checkpoint_type
    selected = MODEL_TYPE_ALIASES.get(normalized)
    if selected is None:
        raise ValueError("unsupported --model-type {!r}".format(requested))
    if selected != checkpoint_type:
        raise ValueError(
            "requested model type {} does not match checkpoint {}".format(
                selected, checkpoint_type
            )
        )
    return selected


def _parameter_count(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model_state is missing")
    total = 0
    for value in state.values():
        if not torch.is_tensor(value):
            raise TypeError("checkpoint model_state values must be tensors")
        total += int(value.numel())
    return total


def verify_learned_isolation(
    adapter: MppiDynamicsAdapter,
    checkpoint: Mapping[str, Any],
    model_type: str,
) -> Dict[str, Any]:
    """Fail if the learned planner contains a direct reference to the true plant."""

    combined = getattr(adapter, "dynamics", None)
    if not isinstance(combined, CombinedDynamics):
        raise RuntimeError("learned adapter did not restore CombinedDynamics")
    dynamics = combined.residual_dynamics
    if not isinstance(dynamics, LearnedResidualDynamics):
        raise RuntimeError("combined model does not contain LearnedResidualDynamics")
    for owner_name, owner in (
        ("adapter", adapter),
        ("combined_dynamics", combined),
        ("learned_dynamics", dynamics),
    ):
        for attribute, value in vars(owner).items():
            if isinstance(value, (DisturbedUnicycle, DisturbanceConfig)):
                raise RuntimeError(
                    "{} unexpectedly exposes true plant data through {}".format(
                        owner_name, attribute
                    )
                )
    state = np.asarray([0.25, -0.4, 0.7], dtype=np.float64)
    control = np.asarray([0.6, -0.2], dtype=np.float64)
    first = np.asarray(dynamics.derivative(state, control, time=0.3), dtype=np.float64)
    second = np.asarray(dynamics.derivative(state, control, time=0.3), dtype=np.float64)
    if first.shape != (3,) or not np.all(np.isfinite(first)):
        raise RuntimeError("learned dynamics produced an invalid derivative")
    determinism_error = float(np.max(np.abs(first - second)))
    if not math.isfinite(determinism_error) or determinism_error != 0.0:
        raise RuntimeError("learned inference is not deterministic in evaluation mode")
    if _checkpoint_model_type(checkpoint) != model_type:
        raise RuntimeError("restored model type does not match validated checkpoint type")
    return {
        "passed": True,
        "model_type": model_type,
        "true_parameters_passed_to_planner": False,
        "direct_true_plant_references": 0,
        "repeat_inference_max_absolute_difference": determinism_error,
        "sample_derivative": first.tolist(),
    }


def build_learned_modes(
    checkpoint_path: Path,
    model_type: str,
    device: str,
    integration_method: str,
) -> Dict[str, MppiDynamicsAdapter]:
    """Build planner models without accepting any plant/config argument."""

    nominal = build_prediction_dynamics(
        mode="nominal",
        integration_method=integration_method,
        clone_per_rollout=False,
    )
    learned = build_prediction_dynamics(
        mode=model_type,
        checkpoint_path=checkpoint_path,
        integration_method=integration_method,
        clone_per_rollout=False,
        device=device,
    )
    return {"nominal_mismatch": nominal, model_type: learned}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=("auto", "mlp", "icode", "mlp_residual", "icode_residual"),
        help="validate an explicit architecture or infer it from the checkpoint",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        device = _resolve_device(args.device)
        checkpoint = load_checkpoint(
            checkpoint_path,
            map_location=device,
            weights_only=True,
            restore_rng=False,
        )
        checkpoint_type = _checkpoint_model_type(checkpoint)
        selected_type = _select_model_type(args.model_type, checkpoint_type)
        config = resolve_config(
            load_config(args.config),
            args.config,
            output_dir=args.output_dir,
            smoke=bool(args.smoke),
            headless=bool(args.headless),
        )
        if args.output_dir is None:
            base = Path(config["run"]["output_dir"])
            config["run"]["output_dir"] = str(
                base.parent / "learned_residual_mppi" / selected_type
            )
        # The learned models are fully built here.  The true disturbance config
        # remains encapsulated in run_modes/run_episode and is never an input to
        # this factory.
        modes = build_learned_modes(
            checkpoint_path,
            selected_type,
            device,
            str(config["benchmark"]["integration_method"]),
        )
        isolation = verify_learned_isolation(
            modes[selected_type], checkpoint, selected_type
        )
        provenance = {
            "planner_modes": ["nominal_mismatch", selected_type],
            "checkpoint": str(checkpoint_path),
            "checkpoint_model_type": checkpoint_type,
            "checkpoint_model_class": str(checkpoint["model_class"]),
            "checkpoint_git_sha": str(checkpoint["git_sha"]),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_best_validation_metric": float(
                checkpoint["best_validation_metric"]
            ),
            "checkpoint_parameter_count": _parameter_count(checkpoint),
            "inference_device": device,
            "true_parameters_passed_to_planner": False,
        }
        run_modes(
            config,
            modes,
            identity_check=isolation,
            provenance=provenance,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
