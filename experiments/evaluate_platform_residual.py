#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.learning.models import load_platform_checkpoint
from mobile_robot_mppi.learning.trainer import (
    contiguous_windows,
    load_split,
    rollout_endpoint_error,
    rollout_loss,
)


class ZeroResidual(torch.nn.Module):
    def forward(self, state, control):
        del control
        return torch.zeros_like(state)


def endpoint_metrics(error):
    if error.shape[0] == 0:
        return {"position_rmse": None, "heading_rmse": None, "state_rmse": []}
    state_rmse = torch.sqrt(torch.mean(error ** 2, dim=0))
    return {
        "position_rmse": float(torch.sqrt(torch.mean(torch.sum(error[:, :2] ** 2, dim=1))).cpu()),
        "heading_rmse": float(torch.sqrt(torch.mean(error[:, 2] ** 2)).cpu()),
        "state_rmse": [float(value) for value in state_rmse.cpu()],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--split", choices=("validation", "test", "unseen"), default="test")
    parser.add_argument(
        "--max-windows", type=int, default=0,
        help="maximum contiguous windows per horizon; 0 evaluates all windows",
    )
    args = parser.parse_args(argv)
    if args.max_windows < 0:
        parser.error("--max-windows must be non-negative")
    max_windows = None if args.max_windows == 0 else int(args.max_windows)
    model, payload = load_platform_checkpoint(args.checkpoint, args.device)
    dataset = load_split(args.dataset_dir, args.split)
    if len(dataset["state_t"]) == 0:
        raise ValueError("dataset split is empty: %s" % args.split)
    state = torch.as_tensor(dataset["state_t"], dtype=torch.float32, device=args.device)
    control = torch.as_tensor(dataset["control_t"], dtype=torch.float32, device=args.device)
    target = torch.as_tensor(dataset["residual_target"], dtype=torch.float32, device=args.device)
    with torch.no_grad():
        derivative_rmse = float(torch.sqrt(torch.mean((model(state, control) - target) ** 2)).cpu())
        nominal_derivative_rmse = float(torch.sqrt(torch.mean(target ** 2)).cpu())
        active_mask = model.residual_output_mask.reshape(1, -1)
        active_derivative_rmse = float(torch.sqrt(
            torch.sum(((model(state, control) - target) ** 2) * active_mask)
            / (state.shape[0] * torch.sum(active_mask))
        ).cpu())
        active_nominal_derivative_rmse = float(torch.sqrt(
            torch.sum((target ** 2) * active_mask)
            / (state.shape[0] * torch.sum(active_mask))
        ).cpu())
    dynamics_config = payload["training_config"]["dynamics"]
    metrics = {
        "residual_derivative_rmse": derivative_rmse,
        "nominal_residual_derivative_rmse": nominal_derivative_rmse,
        "active_residual_derivative_rmse": active_derivative_rmse,
        "active_nominal_residual_derivative_rmse": active_nominal_derivative_rmse,
        "residual_output_mask": [
            float(value) for value in model.residual_output_mask.cpu()
        ],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_type": model.model_type,
        "split": args.split,
        "parameter_count": model.parameter_count(),
        "window_selection": "all" if max_windows is None else "prefix",
        "max_windows": max_windows,
    }
    model.eval()
    nominal_model = ZeroResidual().to(args.device).eval()
    for horizon in (int(value) for value in args.horizons.split(",")):
        starts = contiguous_windows(dataset, horizon)
        with torch.no_grad():
            mse = rollout_loss(
                model, dataset, starts, horizon, dynamics_config, args.device,
                max_windows=max_windows,
            )
            nominal_mse = rollout_loss(
                nominal_model, dataset, starts, horizon, dynamics_config, args.device,
                max_windows=max_windows,
            )
            learned_endpoint = rollout_endpoint_error(
                model, dataset, starts, horizon, dynamics_config, args.device,
                max_windows=max_windows,
            )
            nominal_endpoint = rollout_endpoint_error(
                nominal_model, dataset, starts, horizon, dynamics_config, args.device,
                max_windows=max_windows,
            )
        metrics["rollout_rmse_h%d" % horizon] = float(torch.sqrt(mse).cpu())
        metrics["nominal_rollout_rmse_h%d" % horizon] = float(torch.sqrt(nominal_mse).cpu())
        metrics["endpoint_h%d" % horizon] = endpoint_metrics(learned_endpoint)
        metrics["nominal_endpoint_h%d" % horizon] = endpoint_metrics(nominal_endpoint)
        metrics["rollout_windows_h%d" % horizon] = int(starts.size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
