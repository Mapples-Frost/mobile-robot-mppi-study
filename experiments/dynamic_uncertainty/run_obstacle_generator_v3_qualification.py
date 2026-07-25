"""Qualify the recurrent V3 obstacle generator without any prediction/control."""

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
    NoiseProfile,
    noise_profiles_from_mapping,
)
from mobile_robot_mppi.obstacles.patrol import (
    GENERATOR_VERSION_V3,
    V3_PROCESS_NAMES,
    audit_patrol_trajectory,
    generate_patrol_trajectory,
    validate_v3_config,
)
from mobile_robot_mppi.obstacles.visualization import (
    plot_patrol_processes,
    plot_patrol_time_series,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    return value


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def trajectory_digest(trajectory):
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


def build_schedule(processes, noise_profiles, seeds, schedule_seed):
    if tuple(processes) != V3_PROCESS_NAMES:
        raise ValueError("qualification requires the four V3 processes")
    if len(noise_profiles) != 2 or len(set(noise_profiles)) != 2:
        raise ValueError("qualification requires exactly two noise profiles")
    if len(seeds) != 20 or len(set(int(seed) for seed in seeds)) != 20:
        raise ValueError("qualification requires 20 unique development seeds")
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
    permutation = np.random.RandomState(int(schedule_seed)).permutation(
        len(jobs)
    )
    scheduled = []
    for run_order, job_index in enumerate(permutation):
        job = dict(jobs[int(job_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = (
            f"{job['process']}::{job['noise_profile']}::seed{job['seed']}"
        )
        scheduled.append(job)
    return scheduled


def _shuttle_nonperiodic(trajectory):
    departures = [
        event
        for event in trajectory.metadata["events"]
        if event["kind"] == "depart"
    ]
    arrivals = [
        event
        for event in trajectory.metadata["events"]
        if event["kind"] == "waypoint_arrival"
    ]
    speeds = {
        float(event["parameters"]["cruise_speed_mps"])
        for event in departures
    }
    dwells = {
        float(event["parameters"]["dwell_s"]) for event in arrivals
    }
    return len(speeds) >= 2 and len(dwells) >= 2


def _process_contract(process, trajectory, audit, config):
    qualification = config["qualification"]
    if process == "stochastic_cruise":
        return (
            audit["completed_legs"] == 0
            and audit["intervention_count"] == 0
            and audit["dropout_interval_count"] == 0
        )
    if process == "stochastic_shuttle":
        return (
            audit["completed_legs"]
            >= int(qualification["shuttle_minimum_completed_legs"])
            and audit["distinct_waypoints_visited"] == 2
            and audit["round_trip_observed"]
            and _shuttle_nonperiodic(trajectory)
            and audit["dropout_interval_count"] == 0
        )
    if process == "branching_patrol":
        return (
            audit["completed_legs"]
            >= int(qualification["branching_minimum_completed_legs"])
            and audit["distinct_waypoints_visited"]
            >= int(qualification["branching_minimum_distinct_waypoints"])
            and audit["dropout_interval_count"] == 0
        )
    if process == "hybrid_patrol":
        minimum_dropouts = int(
            config["observation"]["hybrid_dropout_count_range"][0]
        )
        return (
            audit["completed_legs"]
            >= int(qualification["hybrid_minimum_completed_legs"])
            and audit["intervention_count"]
            >= int(qualification["hybrid_minimum_interventions"])
            and audit["dropout_interval_count"] >= minimum_dropouts
        )
    return False


def _run_one(job, profiles, config, output_dir):
    process = str(job["process"])
    seed = int(job["seed"])
    profile = profiles[str(job["noise_profile"])]
    trajectory = generate_patrol_trajectory(
        process, seed, profile, config
    )
    repeated = generate_patrol_trajectory(
        process, seed, profile, config
    )
    digest = trajectory_digest(trajectory)
    reproducible = digest == trajectory_digest(repeated)
    audit = audit_patrol_trajectory(trajectory, config)
    contract = _process_contract(process, trajectory, audit, config)
    invariants = bool(
        reproducible
        and contract
        and audit["finite_truth"]
        and audit["finite_available_observations"]
        and audit["speed_within_limit"]
        and audit["acceleration_within_limit"]
        and audit["yaw_rate_within_limit"]
        and audit["teleport_free"]
    )
    metrics = {
        "schema_version": 3,
        "generator_version": GENERATOR_VERSION_V3,
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "trajectory_sha256": digest,
        "trajectory_reproducible": reproducible,
        "process_contract_pass": contract,
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
            "generator_version": GENERATOR_VERSION_V3,
            "process": process,
            "noise_profile": {
                "name": profile.name,
                "acceleration_std": profile.acceleration_std,
                "observation_std": profile.observation_std,
            },
            "seed": seed,
            "trajectory": dict(config["trajectory"]),
            "physical_limits": dict(config["physical_limits"]),
            "navigation": dict(config["navigation"]),
            "scope_guards": dict(config["scope_guards"]),
        },
        metrics,
        ROOT,
    )
    return {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "completed_legs": int(audit["completed_legs"]),
        "distinct_waypoints_visited": int(
            audit["distinct_waypoints_visited"]
        ),
        "waypoint_visits": "|".join(
            str(value) for value in audit["waypoint_visits"]
        ),
        "round_trip_observed": int(audit["round_trip_observed"]),
        "intervention_count": int(audit["intervention_count"]),
        "intervention_kinds": "|".join(audit["intervention_kinds"]),
        "dropout_intervals": int(audit["dropout_interval_count"]),
        "missing_observations": int(audit["missing_observation_count"]),
        "maximum_speed_mps": float(audit["maximum_speed_mps"]),
        "maximum_acceleration_mps2": float(
            audit["maximum_acceleration_mps2"]
        ),
        "maximum_moving_yaw_rate_radps": float(
            audit["maximum_moving_yaw_rate_radps"]
        ),
        "maximum_position_step_m": float(
            audit["maximum_position_step_m"]
        ),
        "trajectory_reproducible": int(reproducible),
        "process_contract_pass": int(contract),
        "invariants_pass": int(invariants),
        "run_dir": str(run_dir.relative_to(output_dir)),
    }


def _distribution_audit(config, seeds, profile):
    intervention_counts = Counter()
    branching_routes = set()
    hybrid_routes = set()
    branching_waypoints = set()
    hybrid_waypoints = set()
    first_intervention_times = set()
    for seed in seeds:
        branching = generate_patrol_trajectory(
            "branching_patrol", int(seed), profile, config
        )
        hybrid = generate_patrol_trajectory(
            "hybrid_patrol", int(seed), profile, config
        )
        branching_visits = tuple(branching.metadata["waypoint_visits"])
        hybrid_visits = tuple(hybrid.metadata["waypoint_visits"])
        branching_routes.add(branching_visits)
        hybrid_routes.add(hybrid_visits)
        branching_waypoints.update(branching_visits)
        hybrid_waypoints.update(hybrid_visits)
        interventions = [
            event
            for event in hybrid.metadata["events"]
            if event["kind"]
            in ("early_retarget", "hesitation_stop", "speed_replan")
        ]
        intervention_counts.update(
            str(event["kind"]) for event in interventions
        )
        first_intervention_times.add(float(interventions[0]["time_s"]))
    expected_branching = len(
        config["processes_config"]["branching_patrol"]["waypoints_m"]
    )
    expected_hybrid = len(
        config["processes_config"]["hybrid_patrol"]["waypoints_m"]
    )
    return {
        "intervention_kind_counts": dict(sorted(intervention_counts.items())),
        "all_intervention_kinds_observed": set(intervention_counts)
        == {"early_retarget", "hesitation_stop", "speed_replan"},
        "unique_branching_route_signatures": len(branching_routes),
        "unique_hybrid_route_signatures": len(hybrid_routes),
        "branching_waypoints_covered": sorted(branching_waypoints),
        "hybrid_waypoints_covered": sorted(hybrid_waypoints),
        "all_branching_waypoints_observed": len(branching_waypoints)
        == expected_branching,
        "all_hybrid_waypoints_observed": len(hybrid_waypoints)
        == expected_hybrid,
        "unique_first_intervention_times": len(first_intervention_times),
        "intervention_times_not_fixed": len(first_intervention_times) >= 10,
    }


def run_qualification(config_path=DEFAULT_CONFIG, output_override=None):
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    validate_v3_config(config)
    if str(config.get("platform")) != "windows_native_only":
        raise ValueError("V3 qualification is restricted to native Windows")
    if any(bool(value) for value in config["scope_guards"].values()):
        raise ValueError("all algorithm and sealed-registry guards must be false")
    registry_path = _resolve(config["seed_registry_path"]).resolve()
    if registry_path.name == "sealed_dynamic_seeds.yaml":
        raise ValueError("qualification cannot open the sealed registry")
    registry = _load_yaml(registry_path)
    validate_split_registry(registry)
    development = registry["splits"]["development"]
    development_set = set(
        range(
            int(development["seed_start"]),
            int(development["seed_start"]) + int(development["count"]),
        )
    )
    start = int(config["development_seed_start"])
    count = int(config["development_seed_count"])
    seeds = list(range(start, start + count))
    if not set(seeds).issubset(development_set):
        raise ValueError("V3 seeds must belong to the development split")
    if set(seeds).intersection(range(730100001, 730100021)):
        raise ValueError("V3 qualification may not reuse V1/V2 smoke seeds")
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
        else _resolve(config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "schema_version": 3,
            "study_id": config["study_id"],
            "stage": "obstacle_generator_only",
            "generator_version": GENERATOR_VERSION_V3,
            "platform": config["platform"],
            "config_path": str(config_path.relative_to(ROOT)),
            "seed_registry_path": str(registry_path.relative_to(ROOT)),
            "expected_runs": int(
                config["qualification"]["required_trajectories"]
            ),
            "development_seeds": seeds,
            "processes": list(config["processes"]),
            "noise_profiles": list(profiles),
            "scope_guards": dict(config["scope_guards"]),
        },
    )
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
        process: generate_patrol_trajectory(
            process, seeds[0], profiles["medium"], config
        )
        for process in V3_PROCESS_NAMES
    }
    path_figure = plot_patrol_processes(
        representatives, output_dir / "figures/representative_paths.png"
    )
    time_figure = plot_patrol_time_series(
        representatives,
        output_dir / "figures/representative_time_series.png",
    )
    distribution = _distribution_audit(
        config, seeds, profiles["medium"]
    )
    expected = int(config["qualification"]["required_trajectories"])
    qualification_pass = bool(
        len(progress) == expected
        and len({row["experimental_key"] for row in progress}) == expected
        and all(int(row["invariants_pass"]) == 1 for row in progress)
        and distribution["all_intervention_kinds_observed"]
        and distribution["all_branching_waypoints_observed"]
        and distribution["all_hybrid_waypoints_observed"]
        and distribution["intervention_times_not_fixed"]
    )
    process_ranges: Dict[str, Dict[str, int]] = {}
    for process in V3_PROCESS_NAMES:
        values = [
            int(row["completed_legs"])
            for row in progress
            if row["process"] == process
        ]
        process_ranges[process] = {
            "minimum_completed_legs": min(values),
            "maximum_completed_legs": max(values),
        }
    summary = {
        "schema_version": 3,
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
        "maximum_position_step_m": max(
            float(row["maximum_position_step_m"]) for row in progress
        ),
        "completed_leg_ranges": process_ranges,
        "distribution_audit": distribution,
        "representative_path_figure": str(
            path_figure.relative_to(output_dir)
        ),
        "representative_time_figure": str(
            time_figure.relative_to(output_dir)
        ),
        "qualification_pass": qualification_pass,
    }
    write_json(output_dir / "analysis/summary.json", summary)
    summary_text = (
        "# V3 recurrent obstacle-generator qualification\n\n"
        f"- Runs: {summary['passed_runs']}/{summary['run_count']} passed\n"
        f"- Maximum speed: {summary['maximum_speed_mps']:.6f} m/s\n"
        f"- Maximum acceleration: "
        f"{summary['maximum_acceleration_mps2']:.6f} m/s²\n"
        f"- Maximum moving yaw rate: "
        f"{summary['maximum_moving_yaw_rate_radps']:.6f} rad/s\n"
        f"- Completed-leg ranges: {process_ranges}\n"
        f"- Unique branching routes: "
        f"{distribution['unique_branching_route_signatures']}\n"
        f"- Unique hybrid routes: "
        f"{distribution['unique_hybrid_route_signatures']}\n"
        f"- Overall qualification: {qualification_pass}\n"
    )
    analysis_md = output_dir / "analysis/summary.md"
    analysis_md.parent.mkdir(parents=True, exist_ok=True)
    analysis_md.write_text(summary_text, encoding="utf-8")
    integrity_paths = (
        output_dir / "manifest.yaml",
        output_dir / "schedule.json",
        output_dir / "environment_manifest.json",
        output_dir / "progress.csv",
        output_dir / "analysis/summary.json",
        analysis_md,
        path_figure,
        time_figure,
    )
    write_json(
        output_dir / "integrity_audit.json",
        {
            "schema_version": 3,
            "all_expected_files_exist": all(
                path.is_file() for path in integrity_paths
            ),
            "files": {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in integrity_paths
            },
            "qualification_pass": qualification_pass,
        },
    )
    if not qualification_pass:
        raise RuntimeError("V3 obstacle generator qualification failed")
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
