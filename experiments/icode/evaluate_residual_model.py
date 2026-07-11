#!/usr/bin/env python3
"""Evaluate nominal, oracle, MLP, and ICODE prediction models.

The evaluator reports derivative, one-step, and H-step errors without claiming
that a smoke checkpoint is a trained research result.  Oracle dynamics are
rebuilt from each episode's recorded disturbance metadata; learned models see
only their checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dynamics import DisturbanceConfig, DisturbedUnicycle, NominalUnicycle
from src.dynamics.residual.oracle_residual import OracleResidual
from src.learning.residual_dataset import ResidualDataset
from src.planners.mppi_dynamics_adapter import (
    LearnedResidualDynamics,
    MppiDynamicsAdapter,
    build_prediction_dynamics,
)


ANGLE_INDICES = (2,)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _wrap_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    error[..., 2] = np.arctan2(np.sin(error[..., 2]), np.cos(error[..., 2]))
    return error


def _disturbance(parameters: Any) -> DisturbanceConfig:
    payload = json.loads(str(parameters))
    for key in (
        "world_disturbance_amplitude",
        "world_disturbance_frequency",
        "world_disturbance_phase",
        "state_disturbance_gain",
    ):
        if key in payload:
            payload[key] = tuple(payload[key])
    return DisturbanceConfig(**payload)


def _oracle_adapter(dataset: ResidualDataset, row: int) -> MppiDynamicsAdapter:
    plant = DisturbedUnicycle(_disturbance(dataset.disturbance_parameters[row]))
    return build_prediction_dynamics(
        "oracle_residual",
        true_dynamics=plant,
        integration_method="rk4",
        clone_per_rollout=False,
        start_time=float(dataset.time[row]),
    )


def _learned_residual(path: Path, expected: str, device: str) -> LearnedResidualDynamics:
    return LearnedResidualDynamics.from_checkpoint(
        path,
        device=device,
        expected_model_type=expected,
        expose_icode_components=False,
    )


def _residual_predictions(
    mode: str,
    dataset: ResidualDataset,
    residual: Optional[LearnedResidualDynamics],
    limit: Optional[int],
) -> Tuple[np.ndarray, float]:
    count = len(dataset) if limit is None else min(len(dataset), limit)
    predictions = np.zeros((count, dataset.state_dim), dtype=np.float64)
    nominal = NominalUnicycle()
    started = time.perf_counter()
    for row in range(count):
        if mode == "nominal":
            value = np.zeros(dataset.state_dim, dtype=np.float64)
        elif mode == "oracle":
            plant = DisturbedUnicycle(
                _disturbance(dataset.disturbance_parameters[row])
            )
            value = OracleResidual(plant, nominal).derivative(
                dataset.state_t[row],
                dataset.control_t[row],
                time=float(dataset.time[row]),
            )
        else:
            assert residual is not None
            value = residual.derivative(
                dataset.state_t[row],
                dataset.control_t[row],
                time=float(dataset.time[row]),
            )
        predictions[row] = value
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return predictions, elapsed_ms / max(count, 1)


def _adapter_for_row(
    mode: str,
    dataset: ResidualDataset,
    row: int,
    learned_adapter: Optional[MppiDynamicsAdapter],
) -> MppiDynamicsAdapter:
    if mode == "oracle":
        return _oracle_adapter(dataset, row)
    if learned_adapter is not None:
        learned_adapter.start_time = float(dataset.time[row])
        return learned_adapter
    return MppiDynamicsAdapter(
        NominalUnicycle(),
        integration_method="rk4",
        clone_per_rollout=False,
        start_time=float(dataset.time[row]),
    )


def _prediction_metrics(error: np.ndarray) -> Dict[str, float]:
    flattened = error.reshape(-1, error.shape[-1])
    state_rmse = np.sqrt(np.mean(np.square(flattened), axis=0))
    position = np.linalg.norm(flattened[:, :2], axis=1)
    return {
        "state_rmse": float(np.sqrt(np.mean(np.square(flattened)))),
        "x_rmse": float(state_rmse[0]),
        "y_rmse": float(state_rmse[1]),
        "heading_rmse": float(state_rmse[2]),
        "position_rmse": float(np.sqrt(np.mean(np.square(position)))),
    }


def _evaluate_horizon(
    mode: str,
    dataset: ResidualDataset,
    horizon: int,
    learned_adapter: Optional[MppiDynamicsAdapter],
    maximum_windows: Optional[int],
) -> Dict[str, Any]:
    windows = dataset.rollout_window_indices(
        horizon, require_time_continuity=True, require_constant_dt=True
    )
    if maximum_windows is not None:
        windows = windows[:maximum_windows]
    errors: List[np.ndarray] = []
    started = time.perf_counter()
    for window in windows:
        first = int(window[0])
        dt_values = dataset.dt[window]
        if not np.allclose(dt_values, dt_values[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("rollout window contains non-constant dt")
        adapter = _adapter_for_row(mode, dataset, first, learned_adapter)
        prediction = np.asarray(
            adapter.rollout(
                dataset.state_t[first], dataset.control_t[window], float(dt_values[0])
            )[1:],
            dtype=np.float64,
        )
        target = dataset.state_t_plus_1[window]
        errors.append(_wrap_error(prediction, target))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not errors:
        return {
            "horizon": horizon,
            "window_count": 0,
            "rollout_time_ms_per_window": None,
            **{key: None for key in _prediction_metrics(np.zeros((1, 3)))},
        }
    stacked = np.stack(errors, axis=0)
    result: Dict[str, Any] = {
        "horizon": horizon,
        "window_count": len(errors),
        "rollout_time_ms_per_window": elapsed_ms / len(errors),
    }
    result.update(_prediction_metrics(stacked))
    endpoint = _prediction_metrics(stacked[:, -1, :])
    result.update({"endpoint_{}".format(key): value for key, value in endpoint.items()})
    return result


def evaluate(
    dataset: ResidualDataset,
    split_name: str,
    modes: Mapping[str, Tuple[Optional[LearnedResidualDynamics], Optional[MppiDynamicsAdapter]]],
    horizons: Sequence[int],
    maximum_rows: Optional[int],
    maximum_windows: Optional[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    output: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    count = len(dataset) if maximum_rows is None else min(len(dataset), maximum_rows)
    for mode, (residual, adapter) in modes.items():
        predicted_residual, inference_ms = _residual_predictions(
            mode, dataset, residual, maximum_rows
        )
        residual_error = predicted_residual - dataset.residual_target[:count]
        derivative_rmse = np.sqrt(np.mean(np.square(residual_error), axis=0))
        rollout_results = [
            _evaluate_horizon(
                mode, dataset, horizon, adapter, maximum_windows
            )
            for horizon in horizons
        ]
        output[mode] = {
            "sample_count": count,
            "residual_derivative_rmse": float(
                np.sqrt(np.mean(np.square(residual_error)))
            ),
            "residual_component_rmse": derivative_rmse.tolist(),
            "residual_inference_ms_per_sample": inference_ms,
            "rollouts": rollout_results,
        }
        for item in rollout_results:
            row = {
                "split": split_name,
                "mode": mode,
                "residual_derivative_rmse": output[mode]["residual_derivative_rmse"],
                "residual_inference_ms_per_sample": inference_ms,
            }
            row.update(item)
            rows.append(row)
    return output, rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for mode in sorted({str(row["mode"]) for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        axis.plot(
            [row["horizon"] for row in selected],
            [row["position_rmse"] for row in selected],
            marker="o",
            label=mode,
        )
    axis.set_xlabel("Rollout horizon H")
    axis.set_ylabel("Position RMSE [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _hash(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument("--icode-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    horizons = tuple(int(value) for value in args.horizons.split(","))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes: Dict[str, Tuple[Optional[LearnedResidualDynamics], Optional[MppiDynamicsAdapter]]] = {
        "nominal": (None, None),
        "oracle": (None, None),
    }
    for name, checkpoint in (
        ("mlp_residual", args.mlp_checkpoint),
        ("icode_residual", args.icode_checkpoint),
    ):
        if checkpoint is None:
            continue
        expected = name
        residual = _learned_residual(checkpoint.expanduser().resolve(), expected, args.device)
        adapter = build_prediction_dynamics(
            expected,
            checkpoint_path=checkpoint.expanduser().resolve(),
            integration_method="rk4",
            clone_per_rollout=False,
            device=args.device,
        )
        modes[name] = (residual, adapter)

    maximum_rows = 64 if args.smoke else None
    maximum_windows = 24 if args.smoke else None
    results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    for split in ("test", "unseen"):
        dataset = ResidualDataset.load(dataset_dir / "{}.npz".format(split))
        split_result, rows = evaluate(
            dataset,
            split,
            modes,
            horizons,
            maximum_rows,
            maximum_windows,
        )
        results[split] = split_result
        csv_rows.extend(rows)

    manifest = {
        "schema_version": 1,
        "run_type": "smoke" if args.smoke else "research",
        "git_sha": _git_sha(),
        "dataset_dir": str(dataset_dir),
        "horizons": list(horizons),
        "modes": list(modes),
        "checkpoints": {
            "mlp": None if args.mlp_checkpoint is None else str(args.mlp_checkpoint.resolve()),
            "icode": None if args.icode_checkpoint is None else str(args.icode_checkpoint.resolve()),
        },
        "checkpoint_sha256": {
            "mlp": _hash(args.mlp_checkpoint),
            "icode": _hash(args.icode_checkpoint),
        },
        "limits": {"rows": maximum_rows, "windows_per_horizon": maximum_windows},
        "claim": "Framework smoke only; not a formal ICODE-MPPI result." if args.smoke else "Research run; interpret with the recorded config and checkpoints.",
    }
    _atomic_text(output_dir / "metrics.json", json.dumps(results, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_dir / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "summary.csv", csv_rows)
    _plot(
        output_dir / "rollout_position_rmse.png",
        [row for row in csv_rows if row["split"] == "test"],
        "Residual model rollout error - test - {}".format(manifest["run_type"]),
    )
    print("metrics={}".format(output_dir / "metrics.json"))
    print("summary={}".format(output_dir / "summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
