#!/usr/bin/env python3
"""Evaluate the frozen Stage 2 task-aware residual gate."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.learning.models import load_platform_checkpoint
from mobile_robot_mppi.learning.trainer import (
    contiguous_windows,
    load_split,
    rollout_endpoint_error,
)


SEEDS = (20261201, 20261202, 20261203)
HORIZONS = (1, 5, 10, 20, 36)


def _subset(dataset, mask):
    return {key: value[mask] for key, value in dataset.items()}


def _endpoint_position_rmse(model, dataset, horizon, dynamics, device):
    starts = contiguous_windows(dataset, horizon)
    with torch.no_grad():
        error = rollout_endpoint_error(
            model, dataset, starts, horizon, dynamics, device,
            max_windows=None,
        )
    position = float(
        torch.sqrt(torch.mean(torch.sum(error[:, :2] ** 2, dim=1))).cpu()
    )
    state = [
        float(value)
        for value in torch.sqrt(torch.mean(error ** 2, dim=0)).cpu()
    ]
    return {
        "windows": int(starts.size),
        "position_rmse": position,
        "state_rmse": state,
    }


def _active_derivative(model, dataset, device):
    state = torch.as_tensor(
        dataset["state_t"], dtype=torch.float32, device=device
    )
    control = torch.as_tensor(
        dataset["control_t"], dtype=torch.float32, device=device
    )
    target = torch.as_tensor(
        dataset["residual_target"], dtype=torch.float32, device=device
    )
    active = model.residual_output_mask.reshape(1, -1)
    with torch.no_grad():
        prediction = model(state, control)
        learned = torch.sqrt(
            torch.sum((prediction - target) ** 2 * active)
            / (state.shape[0] * torch.sum(active))
        )
        nominal = torch.sqrt(
            torch.sum(target ** 2 * active)
            / (state.shape[0] * torch.sum(active))
        )
    return {
        "learned_rmse": float(learned.cpu()),
        "nominal_rmse": float(nominal.cpu()),
    }


def _evaluate_model(model, dataset, dynamics, device):
    result = {"active_derivative": _active_derivative(model, dataset, device)}
    for horizon in HORIZONS:
        result["h%d" % horizon] = _endpoint_position_rmse(
            model, dataset, horizon, dynamics, device
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--revised-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--maximum-original-regression", type=float, default=0.02
    )
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    test = load_split(dataset_dir, "test")
    unseen = load_split(dataset_dir, "unseen")
    task_mask = (
        np.asarray(test["data_source"]) == "dynamic_obstacle_nominal"
    )
    task_test = _subset(test, task_mask)
    original_test = _subset(test, ~task_mask)
    blocks = []
    for seed in SEEDS:
        base_path = (
            Path(args.base_root)
            / ("l57_icode_high_dynamic_h36_seed%d_v1" % seed)
            / "best.pt"
        )
        revised_path = (
            Path(args.revised_root)
            / ("stage2_task_aware_seed%d_v1" % seed)
            / "best.pt"
        )
        base_model, base_payload = load_platform_checkpoint(
            base_path, args.device
        )
        revised_model, revised_payload = load_platform_checkpoint(
            revised_path, args.device
        )
        dynamics = revised_payload["training_config"]["dynamics"]
        block = {
            "seed": seed,
            "base_checkpoint": str(base_path.resolve()),
            "revised_checkpoint": str(revised_path.resolve()),
            "task_test": {
                "base": _evaluate_model(
                    base_model, task_test, dynamics, args.device
                ),
                "revised": _evaluate_model(
                    revised_model, task_test, dynamics, args.device
                ),
            },
            "original_test": {
                "base": _evaluate_model(
                    base_model, original_test, dynamics, args.device
                ),
                "revised": _evaluate_model(
                    revised_model, original_test, dynamics, args.device
                ),
            },
            "original_unseen": {
                "base": _evaluate_model(
                    base_model, unseen, dynamics, args.device
                ),
                "revised": _evaluate_model(
                    revised_model, unseen, dynamics, args.device
                ),
            },
        }
        task = block["task_test"]
        test_pair = block["original_test"]
        unseen_pair = block["original_unseen"]
        checks = {
            "task_active_derivative_beats_nominal": (
                task["revised"]["active_derivative"]["learned_rmse"]
                < task["revised"]["active_derivative"]["nominal_rmse"]
            ),
            "task_h36_not_worse_than_base": (
                task["revised"]["h36"]["position_rmse"]
                <= task["base"]["h36"]["position_rmse"]
            ),
            "original_test_h36_retained": (
                test_pair["revised"]["h36"]["position_rmse"]
                <= (1.0 + args.maximum_original_regression)
                * test_pair["base"]["h36"]["position_rmse"]
            ),
            "original_unseen_h36_retained": (
                unseen_pair["revised"]["h36"]["position_rmse"]
                <= (1.0 + args.maximum_original_regression)
                * unseen_pair["base"]["h36"]["position_rmse"]
            ),
        }
        serialized = json.dumps(block, allow_nan=False)
        checks["all_metrics_finite"] = all(
            math.isfinite(value)
            for value in _all_numbers(json.loads(serialized))
        )
        block["checks"] = checks
        block["passed"] = bool(all(checks.values()))
        blocks.append(block)
    passing = sum(int(block["passed"]) for block in blocks)
    result = {
        "protocol": "task_aware_residual_retraining_v1",
        "maximum_original_h36_position_regression": (
            args.maximum_original_regression
        ),
        "minimum_passing_blocks": 2,
        "passing_blocks": passing,
        "passed": passing >= 2,
        "task_test_transitions": int(len(task_test["state_t"])),
        "original_test_transitions": int(len(original_test["state_t"])),
        "original_unseen_transitions": int(len(unseen["state_t"])),
        "blocks": blocks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


def _all_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _all_numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


if __name__ == "__main__":
    raise SystemExit(main())
