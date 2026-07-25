"""CSV/JSON/YAML artifact helpers for the probability sandbox."""

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from mobile_robot_mppi.core.config import git_sha

from .motion import ObstacleTrajectory
from .prediction import OnlineForecast


_PROBABILITY_SOURCE_PATHS = (
    "src/mobile_robot_mppi/obstacles/motion.py",
    "src/mobile_robot_mppi/obstacles/state_machine.py",
    "src/mobile_robot_mppi/obstacles/patrol.py",
    "src/mobile_robot_mppi/obstacles/imm.py",
    "src/mobile_robot_mppi/obstacles/change_detection_evaluation.py",
    "src/mobile_robot_mppi/obstacles/collision_risk.py",
    "src/mobile_robot_mppi/obstacles/online_tracking.py",
    "src/mobile_robot_mppi/obstacles/qualification.py",
    "src/mobile_robot_mppi/obstacles/predictor_evaluation.py",
    "src/mobile_robot_mppi/obstacles/prediction.py",
    "src/mobile_robot_mppi/obstacles/artifacts.py",
    "src/mobile_robot_mppi/obstacles/visualization.py",
    "src/mobile_robot_mppi/planning/mppi.py",
    "src/mobile_robot_mppi/perception/legacy_pipeline.py",
    "src/mobile_robot_mppi/evaluation/metrics.py",
    "src/mobile_robot_mppi/runtime/factories.py",
    "src/mobile_robot_mppi/simulation/model_factory.py",
    "src/mobile_robot_mppi/simulation/mujoco_plant.py",
    "src/mobile_robot_mppi/visualization/probability_overlay.py",
    "experiments/dynamic_uncertainty/run_probability_smoke.py",
    "experiments/dynamic_uncertainty/run_obstacle_generator_v2_qualification.py",
    "experiments/dynamic_uncertainty/run_obstacle_generator_v3_qualification.py",
    "experiments/dynamic_uncertainty/show_mujoco_obstacle_program_v3.py",
    "experiments/dynamic_uncertainty/run_ordinary_imm_v3_development.py",
    "experiments/dynamic_uncertainty/run_change_aware_imm_v3_development.py",
    "experiments/dynamic_uncertainty/run_held_out_predictor_qualification.py",
    "experiments/dynamic_uncertainty/run_probabilistic_collision_risk_v1_qualification.py",
    "experiments/dynamic_uncertainty/run_online_v3_tracking_mppi_smoke.py",
    "experiments/dynamic_uncertainty/show_online_v3_risk_aware_closed_loop.py",
    "configs/research/dynamic_uncertainty_probability_smoke.yaml",
    "configs/research/dynamic_obstacle_process_v2.yaml",
    "configs/research/dynamic_obstacle_process_v3.yaml",
    "configs/research/ordinary_imm_v3_development.yaml",
    "configs/research/change_aware_imm_v3_development.yaml",
    "configs/research/change_aware_imm_v3_development_amendment1.yaml",
    "configs/research/change_aware_imm_v3_development_amendment2.yaml",
    "configs/research/dynamic_obstacle_process_v3_ood.yaml",
    "configs/research/held_out_predictor_qualification.yaml",
    "configs/research/probabilistic_collision_risk_v1.yaml",
    "configs/research/mujoco_v3_probabilistic_crossing_smoke.yaml",
    "configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment1.yaml",
    "configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment2.yaml",
    "configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment3.yaml",
    "configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment4.yaml",
    "configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment5.yaml",
    "configs/seeds/dynamic_uncertainty_splits.yaml",
    "docs/experiments/dynamic_uncertainty/PROBABILISTIC_COLLISION_RISK_V1_PREREGISTRATION.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_PREREGISTRATION.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_AMENDMENT1.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_AMENDMENT2.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_AMENDMENT3.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_AMENDMENT4.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_AMENDMENT5.md",
    "docs/experiments/dynamic_uncertainty/ONLINE_V3_TRACKING_AND_MPPI_SMOKE_QUALIFICATION_REPORT.md",
)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return path


def write_yaml(path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(payload), handle, sort_keys=True)
    return path


def write_csv(path, records: Sequence[Mapping[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("CSV writer requires at least one record")
    fieldnames = list(records[0].keys())
    if any(list(record.keys()) != fieldnames for record in records):
        raise ValueError("all CSV records must use the same ordered fields")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(records))
    return path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_provenance(project_root):
    project_root = Path(project_root).resolve()
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        repository_dirty = bool(status.strip())
    except Exception:
        repository_dirty = None
    source_hashes = {}
    for relative in _PROBABILITY_SOURCE_PATHS:
        path = project_root / relative
        if path.is_file():
            source_hashes[relative] = sha256_file(path)
    return {
        "git_sha": git_sha(project_root),
        "repository_dirty": repository_dirty,
        "probability_source_sha256": source_hashes,
    }


def trajectory_records(trajectory: ObstacleTrajectory):
    records = []
    for index, timestamp in enumerate(trajectory.times):
        observed = bool(trajectory.observed_mask[index])
        records.append(
            {
                "time_index": index,
                "time_s": float(timestamp),
                "true_px": float(trajectory.states[index, 0]),
                "true_py": float(trajectory.states[index, 1]),
                "true_vx": float(trajectory.states[index, 2]),
                "true_vy": float(trajectory.states[index, 3]),
                "observation_available": int(observed),
                "observed_px": (
                    float(trajectory.observations[index, 0]) if observed else None
                ),
                "observed_py": (
                    float(trajectory.observations[index, 1]) if observed else None
                ),
                "true_mode": trajectory.true_modes[index],
                "true_change": int(trajectory.change_flags[index]),
            }
        )
    return records


def forecast_records(forecasts: Sequence[OnlineForecast]):
    records = []
    for forecast in forecasts:
        for horizon_index, mean in enumerate(forecast.future.means, start=1):
            covariance = (
                None
                if forecast.future.covariances is None
                else forecast.future.covariances[horizon_index - 1]
            )
            records.append(
                {
                    "time_index": int(forecast.time_index),
                    "time_s": float(forecast.timestamp),
                    "observation_available": int(
                        forecast.observation_available
                    ),
                    "horizon_step": horizon_index,
                    "pred_px": float(mean[0]),
                    "pred_py": float(mean[1]),
                    "pred_vx": float(mean[2]),
                    "pred_vy": float(mean[3]),
                    "cov_xx": None if covariance is None else float(covariance[0, 0]),
                    "cov_xy": None if covariance is None else float(covariance[0, 1]),
                    "cov_yy": None if covariance is None else float(covariance[1, 1]),
                    "cov_vxvx": (
                        None if covariance is None else float(covariance[2, 2])
                    ),
                    "cov_vyvy": (
                        None if covariance is None else float(covariance[3, 3])
                    ),
                }
            )
    return records


def covariance_audit(forecasts: Sequence[OnlineForecast]):
    minimum_eigenvalue = float("inf")
    maximum_asymmetry = 0.0
    covariance_count = 0
    for record in forecasts:
        if record.future.covariances is None:
            continue
        for covariance in record.future.covariances:
            covariance_count += 1
            maximum_asymmetry = max(
                maximum_asymmetry,
                float(np.max(np.abs(covariance - covariance.T))),
            )
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(np.linalg.eigvalsh(covariance).min()),
            )
    if covariance_count == 0:
        return {
            "covariance_count": 0,
            "finite": True,
            "symmetric": True,
            "positive_semidefinite": True,
            "maximum_asymmetry": None,
            "minimum_eigenvalue": None,
        }
    return {
        "covariance_count": covariance_count,
        "finite": True,
        "symmetric": maximum_asymmetry <= 1.0e-10,
        "positive_semidefinite": minimum_eigenvalue >= -1.0e-9,
        "maximum_asymmetry": maximum_asymmetry,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def write_run_artifacts(
    output_dir,
    trajectory: ObstacleTrajectory,
    deterministic_forecasts: Sequence[OnlineForecast],
    kalman_forecasts: Sequence[OnlineForecast],
    resolved_config: Mapping[str, object],
    metrics: Mapping[str, object],
    project_root,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": write_yaml(
            output_dir / "config_resolved.yaml", resolved_config
        ),
        "trajectory": write_csv(
            output_dir / "trajectory.csv", trajectory_records(trajectory)
        ),
        "deterministic": write_csv(
            output_dir / "deterministic_cv_trace.csv",
            forecast_records(deterministic_forecasts),
        ),
        "kalman": write_csv(
            output_dir / "gaussian_cv_kalman_trace.csv",
            forecast_records(kalman_forecasts),
        ),
        "metadata": write_json(output_dir / "obstacle_metadata.json", trajectory.metadata),
    }
    provenance = {
        "schema_version": 1,
        "platform": "windows_native",
        "python": sys.version,
        **repository_provenance(project_root),
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(
        output_dir / "metrics.json",
        dict(metrics, provenance=provenance),
    )
    return provenance


def write_obstacle_run_artifacts(
    output_dir,
    trajectory: ObstacleTrajectory,
    resolved_config: Mapping[str, object],
    metrics: Mapping[str, object],
    project_root,
):
    """Write one obstacle-only run without predictor or controller outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": write_yaml(
            output_dir / "config_resolved.yaml", resolved_config
        ),
        "trajectory": write_csv(
            output_dir / "trajectory.csv", trajectory_records(trajectory)
        ),
        "metadata": write_json(
            output_dir / "obstacle_metadata.json", trajectory.metadata
        ),
    }
    provenance = {
        "schema_version": 2,
        "stage": "obstacle_generator_only",
        "platform": "windows_native",
        "python": sys.version,
        **repository_provenance(project_root),
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(
        output_dir / "metrics.json",
        dict(metrics, provenance=provenance),
    )
    return provenance


def environment_manifest(project_root):
    try:
        import matplotlib

        matplotlib_version = matplotlib.__version__
    except ImportError:
        matplotlib_version = None
    return {
        "schema_version": 1,
        "platform_contract": "windows_native_only",
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib_version,
        **repository_provenance(project_root),
    }


def validate_split_registry(registry: Mapping[str, object]) -> None:
    splits = registry.get("splits", {})
    required = ("development", "held_out_id", "held_out_ood")
    if tuple(splits.keys()) != required:
        raise ValueError("split registry must list development, held_out_id, held_out_ood")
    allocated = set()
    for name in required:
        config = splits[name]
        start = int(config["seed_start"])
        count = int(config["count"])
        if count <= 0:
            raise ValueError("split seed count must be positive")
        seeds = set(range(start, start + count))
        if allocated.intersection(seeds):
            raise ValueError("seed ranges overlap across splits")
        allocated.update(seeds)
    development = splits["development"]
    development_seeds = set(
        range(
            int(development["seed_start"]),
            int(development["seed_start"]) + int(development["count"]),
        )
    )
    smoke = set(int(seed) for seed in development.get("smoke_seeds", ()))
    if len(smoke) != 5 or not smoke.issubset(development_seeds):
        raise ValueError("exactly five smoke seeds must belong to development")
    sealed = registry.get("sealed_registry", {})
    if bool(sealed.get("allowed_in_development", True)):
        raise ValueError("sealed registry must be forbidden in development")


__all__ = [
    "covariance_audit",
    "environment_manifest",
    "forecast_records",
    "repository_provenance",
    "sha256_file",
    "trajectory_records",
    "validate_split_registry",
    "write_csv",
    "write_json",
    "write_obstacle_run_artifacts",
    "write_run_artifacts",
    "write_yaml",
]
