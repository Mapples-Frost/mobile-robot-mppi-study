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
from mobile_robot_mppi.learning.trainer import contiguous_windows, load_split, nominal_derivative, rollout_loss


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizons", default="1,5,10,20")
    args = parser.parse_args(argv)
    model, payload = load_platform_checkpoint(args.checkpoint, args.device)
    dataset = load_split(args.dataset_dir, "test")
    state = torch.as_tensor(dataset["state_t"], dtype=torch.float32, device=args.device)
    control = torch.as_tensor(dataset["control_t"], dtype=torch.float32, device=args.device)
    target = torch.as_tensor(dataset["residual_target"], dtype=torch.float32, device=args.device)
    with torch.no_grad():
        derivative_rmse = float(torch.sqrt(torch.mean((model(state, control) - target) ** 2)).cpu())
    dynamics_config = payload["training_config"]["dynamics"]
    metrics = {"residual_derivative_rmse": derivative_rmse, "checkpoint": str(Path(args.checkpoint).resolve())}
    model.eval()
    for horizon in (int(value) for value in args.horizons.split(",")):
        starts = contiguous_windows(dataset, horizon)
        with torch.no_grad():
            mse = rollout_loss(model, dataset, starts, horizon, dynamics_config, args.device, max_windows=512)
        metrics["rollout_rmse_h%d" % horizon] = float(torch.sqrt(mse).cpu())
        metrics["rollout_windows_h%d" % horizon] = int(starts.size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
