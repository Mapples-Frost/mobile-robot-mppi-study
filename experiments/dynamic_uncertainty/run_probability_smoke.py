"""Run the preregistered 4 x 2 x 5 probability-sandbox smoke test."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml

from mobile_robot_mppi.obstacles.artifacts import (
    covariance_audit,
    environment_manifest,
    validate_split_registry,
    write_csv,
    write_json,
    write_run_artifacts,
    write_yaml,
)
from mobile_robot_mppi.obstacles.motion import (
    PROCESS_NAMES,
    NoiseProfile,
    generate_obstacle_trajectory,
    noise_profiles_from_mapping,
)
from mobile_robot_mppi.obstacles.prediction import (
    DeterministicCVPredictor,
    GaussianCVKalmanPredictor,
    prediction_metrics,
    run_online_forecasts,
)
from mobile_robot_mppi.obstacles.visualization import plot_obstacle_processes


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/research/dynamic_uncertainty_probability_smoke.yaml"
)


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _trajectory_digest(trajectory) -> str:
    digest = hashlib.sha256()
    digest.update(trajectory.times.tobytes())
    digest.update(trajectory.states.tobytes())
    normalized_observations = np.nan_to_num(
        trajectory.observations, nan=9.87654321e37
    )
    digest.update(normalized_observations.tobytes())
    digest.update(trajectory.observed_mask.tobytes())
    digest.update(trajectory.change_flags.tobytes())
    digest.update("|".join(trajectory.true_modes).encode("utf-8"))
    digest.update(
        json.dumps(
            dict(trajectory.metadata), sort_keys=True, allow_nan=False
        ).encode("utf-8")
    )
    return digest.hexdigest()


def build_schedule(
    processes: Sequence[str],
    noise_profiles: Sequence[str],
    seeds: Sequence[int],
    schedule_seed: int,
) -> List[Dict[str, object]]:
    if tuple(processes) != PROCESS_NAMES:
        raise ValueError("smoke must include the four registered processes in order")
    if len(noise_profiles) != 2 or len(set(noise_profiles)) != 2:
        raise ValueError("smoke requires exactly two unique noise profiles")
    if len(seeds) != 5 or len(set(int(seed) for seed in seeds)) != 5:
        raise ValueError("smoke requires exactly five unique development seeds")
    jobs = [
        {
            "process": str(process),
            "noise_profile": str(noise),
            "seed": int(seed),
        }
        for process in processes
        for noise in noise_profiles
        for seed in seeds
    ]
    rng = np.random.RandomState(int(schedule_seed))
    order = rng.permutation(len(jobs))
    scheduled = []
    for run_order, job_index in enumerate(order):
        job = dict(jobs[int(job_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = "{process}::{noise_profile}::seed{seed}".format(
            **job
        )
        scheduled.append(job)
    return scheduled


def _reproducible(
    process: str,
    seed: int,
    profile: NoiseProfile,
    dt: float,
    steps: int,
    expected_digest: str,
) -> bool:
    repeated = generate_obstacle_trajectory(
        process=process,
        seed=seed,
        noise_profile=profile,
        dt=dt,
        steps=steps,
    )
    return _trajectory_digest(repeated) == expected_digest


def _run_one(
    job: Mapping[str, object],
    profiles: Mapping[str, NoiseProfile],
    config: Mapping[str, object],
    output_dir: Path,
) -> Mapping[str, object]:
    process = str(job["process"])
    seed = int(job["seed"])
    profile = profiles[str(job["noise_profile"])]
    trajectory_config = config["trajectory"]
    prediction_config = config["prediction"]
    dt = float(trajectory_config["dt"])
    steps = int(trajectory_config["steps"])
    horizon = int(prediction_config["horizon"])
    trajectory = generate_obstacle_trajectory(
        process=process,
        seed=seed,
        noise_profile=profile,
        dt=dt,
        steps=steps,
    )

    deterministic = DeterministicCVPredictor()
    deterministic_forecasts = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        deterministic,
        horizon=horizon,
        forecast_dt=dt,
    )
    kalman = GaussianCVKalmanPredictor(
        process_acceleration_std=(
            profile.acceleration_std
            * float(prediction_config["kalman_process_std_scale"])
        ),
        observation_std=(
            profile.observation_std
            * float(prediction_config["kalman_observation_std_scale"])
        ),
        initial_velocity_std=float(
            prediction_config["initial_velocity_std"]
        ),
    )
    kalman_forecasts = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        kalman,
        horizon=horizon,
        forecast_dt=dt,
    )

    deterministic_metrics = prediction_metrics(
        deterministic_forecasts, trajectory.states
    )
    kalman_metrics = prediction_metrics(kalman_forecasts, trajectory.states)
    covariance = covariance_audit(kalman_forecasts)
    digest = _trajectory_digest(trajectory)
    reproducible = _reproducible(
        process, seed, profile, dt, steps, digest
    )
    invariants_pass = bool(
        np.isfinite(trajectory.states).all()
        and np.isfinite(
            trajectory.observations[trajectory.observed_mask]
        ).all()
        and covariance["finite"]
        and covariance["symmetric"]
        and covariance["positive_semidefinite"]
        and reproducible
    )
    metrics = {
        "schema_version": 1,
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "trajectory_sha256": digest,
        "trajectory_reproducible": reproducible,
        "observation_count": int(trajectory.observed_mask.sum()),
        "missing_observation_count": int((~trajectory.observed_mask).sum()),
        "deterministic_cv": deterministic_metrics,
        "gaussian_cv_kalman": kalman_metrics,
        "covariance_audit": covariance,
        "invariants_pass": invariants_pass,
    }
    resolved_config = {
        "study_id": config["study_id"],
        "process": process,
        "noise_profile": {
            "name": profile.name,
            "acceleration_std": profile.acceleration_std,
            "observation_std": profile.observation_std,
        },
        "seed": seed,
        "trajectory": dict(trajectory_config),
        "prediction": dict(prediction_config),
        "scope_guards": dict(config["scope_guards"]),
    }
    run_dir = (
        output_dir
        / "runs"
        / process
        / profile.name
        / ("seed_%d" % seed)
    )
    write_run_artifacts(
        run_dir,
        trajectory,
        deterministic_forecasts,
        kalman_forecasts,
        resolved_config,
        metrics,
        project_root=ROOT,
    )
    return {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "steps": steps,
        "observations": int(trajectory.observed_mask.sum()),
        "missing_observations": int((~trajectory.observed_mask).sum()),
        "change_events": int(trajectory.change_flags.sum()),
        "deterministic_ade_m": deterministic_metrics["ade_m"],
        "deterministic_fde_m": deterministic_metrics["fde_m"],
        "kalman_ade_m": kalman_metrics["ade_m"],
        "kalman_fde_m": kalman_metrics["fde_m"],
        "kalman_nll": kalman_metrics["nll"],
        "kalman_coverage_90": kalman_metrics["coverage_90"],
        "minimum_covariance_eigenvalue": covariance["minimum_eigenvalue"],
        "maximum_covariance_asymmetry": covariance["maximum_asymmetry"],
        "trajectory_reproducible": int(reproducible),
        "invariants_pass": int(invariants_pass),
        "run_dir": str(run_dir.relative_to(output_dir)),
    }


def _representative_trajectories(
    profiles: Mapping[str, NoiseProfile],
    seeds: Sequence[int],
    config: Mapping[str, object],
):
    trajectory_config = config["trajectory"]
    profile_name = "medium" if "medium" in profiles else list(profiles)[-1]
    return {
        process: generate_obstacle_trajectory(
            process,
            int(seeds[0]),
            profiles[profile_name],
            dt=float(trajectory_config["dt"]),
            steps=int(trajectory_config["steps"]),
        )
        for process in PROCESS_NAMES
    }


def run_smoke(config_path: Path, output_override=None) -> Mapping[str, object]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if str(config.get("platform")) != "windows_native_only":
        raise ValueError("probability smoke is restricted to native Windows")
    guards = config.get("scope_guards", {})
    required_false = (
        "import_sealed_registry",
        "enable_imm",
        "enable_mppi",
        "enable_rl",
        "enable_icode",
        "enable_hss",
    )
    if any(bool(guards.get(name, True)) for name in required_false):
        raise ValueError("first-stage scope guard was not frozen off")

    registry_path = _resolve(ROOT, str(config["seed_registry_path"]))
    if registry_path.name == "sealed_dynamic_seeds.yaml":
        raise ValueError("development smoke cannot open the sealed registry")
    registry = _load_yaml(registry_path)
    validate_split_registry(registry)
    seeds = [
        int(seed)
        for seed in registry["splits"]["development"]["smoke_seeds"]
    ]
    profiles = noise_profiles_from_mapping(config["noise_profiles"])
    schedule = build_schedule(
        config["processes"],
        list(profiles.keys()),
        seeds,
        int(config["schedule_seed"]),
    )
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(ROOT, str(config["output_dir"])).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(
        output_dir / "manifest.yaml",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "stage": config["stage"],
            "platform": config["platform"],
            "config_path": str(config_path.relative_to(ROOT)),
            "seed_registry_path": str(registry_path.relative_to(ROOT)),
            "expected_runs": 40,
            "processes": list(config["processes"]),
            "noise_profiles": list(profiles.keys()),
            "smoke_seeds": seeds,
            "scope_guards": dict(guards),
        },
    )
    write_json(
        output_dir / "schedule.json",
        {
            "schema_version": 1,
            "schedule_seed": int(config["schedule_seed"]),
            "jobs": schedule,
        },
    )
    write_json(
        output_dir / "environment_manifest.json",
        environment_manifest(ROOT),
    )

    progress = [
        _run_one(job, profiles, config, output_dir) for job in schedule
    ]
    progress.sort(key=lambda row: int(row["run_order"]))
    write_csv(output_dir / "progress.csv", progress)

    representatives = _representative_trajectories(
        profiles, seeds, config
    )
    figure_path = plot_obstacle_processes(
        representatives,
        output_dir / "figures/obstacle_processes_smoke.png",
    )
    keys = [row["experimental_key"] for row in progress]
    all_invariants = all(bool(row["invariants_pass"]) for row in progress)
    development = registry["splits"]["development"]
    development_range = set(
        range(
            int(development["seed_start"]),
            int(development["seed_start"]) + int(development["count"]),
        )
    )
    no_sealed_seed = set(seeds).issubset(development_range)
    required_run_files = (
        "config_resolved.yaml",
        "trajectory.csv",
        "deterministic_cv_trace.csv",
        "gaussian_cv_kalman_trace.csv",
        "obstacle_metadata.json",
        "provenance.json",
        "metrics.json",
    )
    missing_files = []
    for row in progress:
        run_dir = output_dir / str(row["run_dir"])
        for filename in required_run_files:
            if not (run_dir / filename).is_file():
                missing_files.append(str((run_dir / filename).relative_to(output_dir)))
    audit = {
        "schema_version": 1,
        "expected_runs": 40,
        "completed_runs": len(progress),
        "unique_experimental_keys": len(set(keys)),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "all_invariants_pass": all_invariants,
        "all_trajectories_reproducible": all(
            bool(row["trajectory_reproducible"]) for row in progress
        ),
        "sealed_seed_used": not no_sealed_seed,
        "missing_required_files": missing_files,
        "imm_executed": False,
        "closed_loop_mppi_executed": False,
        "rl_icode_hss_executed": False,
    }
    audit["pass"] = bool(
        len(progress) == 40
        and len(set(keys)) == 40
        and all_invariants
        and no_sealed_seed
        and not missing_files
    )
    write_json(output_dir / "integrity_audit.json", audit)

    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "gate_scope": "engineering_smoke_only",
        "gate_status": "PASS" if audit["pass"] else "FAIL",
        "run_count": len(progress),
        "mean_deterministic_ade_m": float(
            np.mean([row["deterministic_ade_m"] for row in progress])
        ),
        "mean_kalman_ade_m": float(
            np.mean([row["kalman_ade_m"] for row in progress])
        ),
        "mean_kalman_coverage_90": float(
            np.mean([row["kalman_coverage_90"] for row in progress])
        ),
        "figure": str(figure_path.relative_to(output_dir)),
        "interpretation": (
            "This smoke validates data, reproducibility, finite predictions, "
            "PSD covariance, and artifact structure. It is not a predictor "
            "accuracy or calibration Gate."
        ),
    }
    write_json(output_dir / "analysis/summary.json", summary)
    return {"output_dir": str(output_dir), "audit": audit, "summary": summary}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_smoke(Path(args.config), output_override=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
