"""Qualify the V2 obstacle generator without running prediction or control."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml

from mobile_robot_mppi.obstacles.artifacts import (
    environment_manifest,
    sha256_file,
    validate_split_registry,
    write_csv,
    write_json,
    write_obstacle_run_artifacts,
    write_yaml,
)
from mobile_robot_mppi.obstacles.motion import (
    PROCESS_NAMES,
    NoiseProfile,
    noise_profiles_from_mapping,
)
from mobile_robot_mppi.obstacles.state_machine import (
    GENERATOR_VERSION,
    audit_state_machine_trajectory,
    build_motion_program,
    generate_state_machine_trajectory,
    validate_v2_config,
)
from mobile_robot_mppi.obstacles.visualization import plot_obstacle_processes


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/research/dynamic_obstacle_process_v2.yaml"


def _load_yaml(path: Path) -> Mapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def trajectory_digest(trajectory) -> str:
    digest = hashlib.sha256()
    for array in (
        trajectory.times,
        trajectory.states,
        np.nan_to_num(trajectory.observations, nan=9.87654321e37),
        trajectory.observed_mask,
        trajectory.change_flags,
    ):
        digest.update(np.asarray(array).tobytes())
    digest.update("|".join(trajectory.true_modes).encode("utf-8"))
    digest.update(
        json.dumps(
            trajectory.metadata, sort_keys=True, allow_nan=False
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
        raise ValueError("qualification requires all four registered processes")
    if len(noise_profiles) != 2 or len(set(noise_profiles)) != 2:
        raise ValueError("qualification requires exactly two noise profiles")
    if len(seeds) != 20 or len(set(int(seed) for seed in seeds)) != 20:
        raise ValueError("qualification requires exactly 20 development seeds")
    jobs = [
        {
            "process": process,
            "noise_profile": noise,
            "seed": int(seed),
        }
        for process in processes
        for noise in noise_profiles
        for seed in seeds
    ]
    order = np.random.RandomState(int(schedule_seed)).permutation(len(jobs))
    result = []
    for run_order, job_index in enumerate(order):
        job = dict(jobs[int(job_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = (
            f"{job['process']}::{job['noise_profile']}::seed{job['seed']}"
        )
        result.append(job)
    return result


def _event_contract(process: str, kinds: Sequence[str]) -> bool:
    kinds = list(kinds)
    turns = sum(kind in ("turn_left", "turn_right") for kind in kinds)
    speeds = sum(kind in ("accelerate", "decelerate") for kind in kinds)
    resumes = sum(kind in ("restart_forward", "reverse") for kind in kinds)
    if process == "noisy_cv":
        return not kinds
    if process == "speed_change":
        return len(kinds) == 3 and speeds == 1 and "stop" in kinds and resumes == 1
    if process == "direction_change":
        return len(kinds) == 2 and turns == 2
    if process == "combined_change":
        return (
            len(kinds) == 5
            and turns >= 1
            and speeds >= 1
            and "stop" in kinds
            and resumes == 1
        )
    return False


def _run_one(
    job: Mapping[str, object],
    profiles: Mapping[str, NoiseProfile],
    config: Mapping[str, object],
    output_dir: Path,
) -> Mapping[str, object]:
    process = str(job["process"])
    seed = int(job["seed"])
    profile = profiles[str(job["noise_profile"])]
    trajectory = generate_state_machine_trajectory(
        process, seed, profile, config
    )
    repeated = generate_state_machine_trajectory(
        process, seed, profile, config
    )
    digest = trajectory_digest(trajectory)
    reproducible = digest == trajectory_digest(repeated)
    audit = audit_state_machine_trajectory(trajectory, config)
    contract = _event_contract(process, audit["event_kinds"])
    expected_dropouts = (
        int(config["observation"]["combined_dropout_count"])
        if process == "combined_change"
        else 0
    )
    dropout_contract = (
        int(audit["dropout_interval_count"]) == expected_dropouts
    )
    invariants = bool(
        reproducible
        and contract
        and dropout_contract
        and audit["finite_truth"]
        and audit["finite_available_observations"]
        and audit["speed_within_limit"]
        and audit["acceleration_within_limit"]
        and audit["yaw_rate_within_limit"]
    )
    metrics = {
        "schema_version": 2,
        "generator_version": GENERATOR_VERSION,
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "trajectory_sha256": digest,
        "trajectory_reproducible": reproducible,
        "event_contract_pass": contract,
        "dropout_contract_pass": dropout_contract,
        "audit": audit,
        "invariants_pass": invariants,
    }
    run_dir = (
        output_dir / "runs" / process / profile.name / f"seed_{seed}"
    )
    write_obstacle_run_artifacts(
        run_dir,
        trajectory,
        {
            "study_id": config["study_id"],
            "generator_version": GENERATOR_VERSION,
            "process": process,
            "noise_profile": {
                "name": profile.name,
                "acceleration_std": profile.acceleration_std,
                "observation_std": profile.observation_std,
            },
            "seed": seed,
            "trajectory": dict(config["trajectory"]),
            "physical_limits": dict(config["physical_limits"]),
            "scope_guards": dict(config["scope_guards"]),
        },
        metrics,
        project_root=ROOT,
    )
    return {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "event_count": int(audit["event_count"]),
        "event_kinds": "|".join(audit["event_kinds"]),
        "dropout_intervals": int(audit["dropout_interval_count"]),
        "missing_observations": int(audit["missing_observation_count"]),
        "maximum_speed_mps": float(audit["maximum_speed_mps"]),
        "maximum_acceleration_mps2": float(
            audit["maximum_acceleration_mps2"]
        ),
        "maximum_moving_yaw_rate_radps": float(
            audit["maximum_moving_yaw_rate_radps"]
        ),
        "minimum_stop_mode_speed_mps": audit["minimum_stop_mode_speed_mps"],
        "trajectory_reproducible": int(reproducible),
        "event_contract_pass": int(contract),
        "dropout_contract_pass": int(dropout_contract),
        "invariants_pass": int(invariants),
        "run_dir": str(run_dir.relative_to(output_dir)),
    }


def _distribution_audit(
    config: Mapping[str, object], seeds: Sequence[int]
) -> Mapping[str, object]:
    kinds = Counter()
    angles = Counter()
    for seed in seeds:
        program = build_motion_program("combined_change", int(seed), config)
        for event in program.events:
            kinds[event.kind] += 1
            if "angle_deg" in event.parameters:
                angles[str(float(event.parameters["angle_deg"]))] += 1
    required_kinds = {
        "turn_left",
        "turn_right",
        "accelerate",
        "decelerate",
        "restart_forward",
        "reverse",
        "stop",
    }
    required_angles = {"-90.0", "-45.0", "45.0", "90.0"}
    return {
        "combined_change_unique_seed_count": len(seeds),
        "event_kind_counts": dict(sorted(kinds.items())),
        "turn_angle_counts": dict(sorted(angles.items())),
        "all_required_event_kinds_observed": required_kinds.issubset(kinds),
        "all_required_turn_angles_observed": required_angles.issubset(angles),
    }


def run_qualification(
    config_path: Path = DEFAULT_CONFIG, output_override=None
) -> Mapping[str, object]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    validate_v2_config(config)
    if str(config.get("platform")) != "windows_native_only":
        raise ValueError("V2 qualification is restricted to native Windows")
    guards = config["scope_guards"]
    if any(bool(value) for value in guards.values()):
        raise ValueError("all algorithm and sealed-registry guards must be false")

    registry_path = _resolve(str(config["seed_registry_path"])).resolve()
    if registry_path.name == "sealed_dynamic_seeds.yaml":
        raise ValueError("qualification cannot open the sealed registry")
    registry = _load_yaml(registry_path)
    validate_split_registry(registry)
    development = registry["splits"]["development"]
    start = int(development["seed_start"])
    count = int(config["development_seed_count"])
    seeds = list(range(start, start + count))
    profiles = noise_profiles_from_mapping(config["noise_profiles"])
    schedule = build_schedule(
        config["processes"],
        list(profiles),
        seeds,
        int(config["schedule_seed"]),
    )
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(str(config["output_dir"])).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 2,
        "study_id": config["study_id"],
        "stage": "obstacle_generator_only",
        "generator_version": GENERATOR_VERSION,
        "platform": config["platform"],
        "config_path": str(config_path.relative_to(ROOT)),
        "seed_registry_path": str(registry_path.relative_to(ROOT)),
        "expected_runs": int(config["qualification"]["required_trajectories"]),
        "development_seeds": seeds,
        "processes": list(config["processes"]),
        "noise_profiles": list(profiles),
        "scope_guards": dict(guards),
    }
    write_yaml(output_dir / "manifest.yaml", manifest)
    write_json(
        output_dir / "schedule.json",
        {"schedule_seed": int(config["schedule_seed"]), "jobs": schedule},
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

    representatives = {
        process: generate_state_machine_trajectory(
            process, seeds[0], profiles["medium"], config
        )
        for process in PROCESS_NAMES
    }
    figure_path = plot_obstacle_processes(
        representatives,
        output_dir / "figures/representative_processes.png",
        obstacle_radius=float(config["trajectory"]["obstacle_radius_m"]),
    )
    distribution = _distribution_audit(config, seeds)
    expected = int(config["qualification"]["required_trajectories"])
    all_pass = (
        len(progress) == expected
        and len({row["experimental_key"] for row in progress}) == expected
        and all(int(row["invariants_pass"]) == 1 for row in progress)
        and distribution["all_required_event_kinds_observed"]
        and distribution["all_required_turn_angles_observed"]
    )
    summary = {
        "schema_version": 2,
        "run_count": len(progress),
        "unique_experimental_keys": len(
            {row["experimental_key"] for row in progress}
        ),
        "passed_runs": sum(
            int(row["invariants_pass"]) for row in progress
        ),
        "maximum_speed_mps": max(
            float(row["maximum_speed_mps"]) for row in progress
        ),
        "maximum_acceleration_mps2": max(
            float(row["maximum_acceleration_mps2"]) for row in progress
        ),
        "maximum_moving_yaw_rate_radps": max(
            float(row["maximum_moving_yaw_rate_radps"]) for row in progress
        ),
        "distribution_audit": distribution,
        "representative_figure": str(figure_path.relative_to(output_dir)),
        "qualification_pass": bool(all_pass),
    }
    write_json(output_dir / "analysis/summary.json", summary)
    summary_markdown = (
        "# V2 obstacle-generator qualification\n\n"
        f"- Runs: {summary['passed_runs']}/{summary['run_count']} passed\n"
        f"- Maximum speed: {summary['maximum_speed_mps']:.6f} m/s\n"
        f"- Maximum acceleration magnitude: "
        f"{summary['maximum_acceleration_mps2']:.6f} m/s^2\n"
        f"- Maximum moving yaw rate: "
        f"{summary['maximum_moving_yaw_rate_radps']:.6f} rad/s\n"
        f"- Required event-kind coverage: "
        f"{distribution['all_required_event_kinds_observed']}\n"
        f"- Required turn-angle coverage: "
        f"{distribution['all_required_turn_angles_observed']}\n"
        f"- Overall qualification: {all_pass}\n"
    )
    analysis_path = output_dir / "analysis/summary.md"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(summary_markdown, encoding="utf-8")

    integrity_paths = (
        output_dir / "manifest.yaml",
        output_dir / "schedule.json",
        output_dir / "environment_manifest.json",
        output_dir / "progress.csv",
        output_dir / "analysis/summary.json",
        output_dir / "analysis/summary.md",
        figure_path,
    )
    integrity = {
        "schema_version": 2,
        "all_expected_files_exist": all(path.is_file() for path in integrity_paths),
        "files": {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in integrity_paths
        },
        "qualification_pass": bool(all_pass),
    }
    write_json(output_dir / "integrity_audit.json", integrity)
    if not all_pass:
        raise RuntimeError("V2 obstacle generator qualification failed")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_qualification(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
