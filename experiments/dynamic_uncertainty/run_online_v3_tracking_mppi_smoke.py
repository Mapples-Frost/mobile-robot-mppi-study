"""Run the preregistered paired online V3 tracking and MPPI smoke.

The controller receives only causal sensor/perception outputs.  Frozen V3
truth is regenerated after each episode solely for association-error and
route-crossing audits.
"""

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.obstacles.artifacts import (
    environment_manifest,
    repository_provenance,
    sha256_file,
    write_csv,
    write_json,
    write_yaml,
)
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import generate_patrol_trajectory
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "configs/research/"
    "mujoco_v3_probabilistic_crossing_smoke_amendment17.yaml"
)
SEED_REGISTRY = ROOT / "configs/seeds/dynamic_uncertainty_splits.yaml"


def build_schedule(seeds, arms, schedule_seed):
    seeds = tuple(int(seed) for seed in seeds)
    arms = tuple(str(arm) for arm in arms)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("development seeds must be non-empty and unique")
    if arms != ("risk_disabled", "risk_enabled"):
        raise ValueError(
            "paired smoke arms must be risk_disabled then risk_enabled"
        )
    jobs = [
        {"seed": seed, "arm": arm}
        for seed in seeds
        for arm in arms
    ]
    order = np.random.RandomState(int(schedule_seed)).permutation(len(jobs))
    scheduled = []
    for run_order, job_index in enumerate(order):
        job = dict(jobs[int(job_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = (
            f"seed{job['seed']}::{job['arm']}"
        )
        scheduled.append(job)
    return scheduled


def _load_raw_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    return value


def _validate_scope(config):
    smoke = config["online_smoke"]
    guards = config["scope_guards"]
    required_guards = (
        "obstacle_v3_frozen",
        "change_aware_imm_frozen",
        "collision_risk_v1_frozen",
        "development_seeds_only",
        "paired_complete_episodes",
    )
    if not all(bool(guards.get(name, False)) for name in required_guards):
        raise ValueError("all frozen-scope guards must be true")
    if bool(guards.get("simulator_truth_for_control", True)):
        raise ValueError("simulator truth for control must be false")
    if bool(guards.get("sealed_registry_imported", True)):
        raise ValueError("sealed seed registry must not be imported")

    registry = _load_raw_yaml(SEED_REGISTRY)
    registered = {
        int(value)
        for value in registry["splits"]["development"]["smoke_seeds"]
    }
    selected = {int(value) for value in smoke["development_seeds"]}
    if not selected <= registered:
        raise ValueError(
            "online smoke seeds must be in the registered smoke subset"
        )
    return {
        "registered_smoke_seeds": sorted(registered),
        "selected_smoke_seeds": sorted(selected),
        "sealed_registry_opened": False,
    }


def _obstacle_trajectory(config, seed):
    obstacle = config["scene"]["obstacles"][0]
    motion = obstacle["motion"]
    path = Path(str(motion["config_path"]))
    if not path.is_absolute():
        path = ROOT / path
    obstacle_config = _load_raw_yaml(path)
    profiles = noise_profiles_from_mapping(
        obstacle_config["noise_profiles"]
    )
    return generate_patrol_trajectory(
        str(motion["process"]),
        int(seed),
        profiles[str(motion["noise_profile"])],
        obstacle_config,
    )


def _route_points(config):
    initial = np.asarray(
        config["experiment"]["initial_state"], dtype=np.float64
    )
    goal = np.asarray(config["task"]["position"], dtype=np.float64)
    return initial[:2], goal[:2]


def route_crossing_count(trajectory, route_start, route_goal, duration_s):
    mask = trajectory.times <= float(duration_s) + 1.0e-12
    positions = trajectory.states[mask, :2]
    route_start = np.asarray(route_start, dtype=np.float64)
    route_goal = np.asarray(route_goal, dtype=np.float64)
    route = route_goal - route_start
    route_norm_sq = float(route @ route)
    if route_norm_sq <= 0.0:
        raise ValueError("route endpoints must differ")
    signed = (
        route[0] * (positions[:, 1] - route_start[1])
        - route[1] * (positions[:, 0] - route_start[0])
    )
    changes = signed[:-1] * signed[1:] < 0.0
    alpha = np.divide(
        signed[:-1],
        signed[:-1] - signed[1:],
        out=np.zeros_like(signed[:-1]),
        where=np.abs(signed[:-1] - signed[1:]) > 1.0e-15,
    )
    intersections = (
        positions[:-1]
        + alpha[:, None] * (positions[1:] - positions[:-1])
    )
    route_fraction = (
        (intersections - route_start[None, :]) @ route
    ) / route_norm_sq
    return int(
        np.sum(
            changes
            & (route_fraction >= -1.0e-12)
            & (route_fraction <= 1.0 + 1.0e-12)
        )
    )


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row, name):
    return float(row[name])


def _truth_positions_at(trajectory, timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    return np.column_stack(
        (
            np.interp(
                timestamps, trajectory.times, trajectory.states[:, 0]
            ),
            np.interp(
                timestamps, trajectory.times, trajectory.states[:, 1]
            ),
        )
    )


def _audit_episode(config, seed, arm, run_order, run_dir, trajectory):
    metrics_path = run_dir / "metrics.json"
    trace_path = run_dir / "trajectory.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = _read_csv(trace_path)
    if not rows:
        raise ValueError("episode trajectory is empty")

    dt = float(config["experiment"]["control_dt"])
    control_times = np.maximum(
        0.0,
        np.asarray([_float(row, "time") for row in rows]) - dt,
    )
    truth_xy = _truth_positions_at(trajectory, control_times)
    post_robot_pose = np.asarray(
        [
            (
                _float(row, "x"),
                _float(row, "y"),
                _float(row, "theta"),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    initial_pose = np.asarray(
        config["experiment"]["initial_state"][:3],
        dtype=np.float64,
    )
    pre_robot_pose = np.vstack(
        (initial_pose, post_robot_pose[:-1])
    )
    online_pose = np.asarray(
        [
            (
                _float(row, "planner_observation_x"),
                _float(row, "planner_observation_y"),
                _float(row, "planner_observation_theta"),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    localization_position_errors = np.linalg.norm(
        online_pose[:, :2] - pre_robot_pose[:, :2], axis=1
    )
    localization_heading_errors = np.arctan2(
        np.sin(online_pose[:, 2] - pre_robot_pose[:, 2]),
        np.cos(online_pose[:, 2] - pre_robot_pose[:, 2]),
    )
    radius = float(
        config["perception"]["dynamic_obstacle_tracker"][
            "obstacle_radius_m"
        ]
    )
    visible = (
        np.linalg.norm(truth_xy - pre_robot_pose[:, :2], axis=1)
        - radius
        <= float(config["sensors"]["lidar_range_max"]) + 1.0e-12
    )
    associated = np.asarray(
        [
            bool(round(_float(row, "dynamic_obstacle_tracker_associated")))
            for row in rows
        ]
    )
    measured_xy = np.asarray(
        [
            (
                _float(row, "dynamic_obstacle_tracker_measurement_x"),
                _float(row, "dynamic_obstacle_tracker_measurement_y"),
            )
            for row in rows
        ]
    )
    scored = visible & associated
    relative_world = truth_xy - pre_robot_pose[:, :2]
    angle_delta = online_pose[:, 2] - pre_robot_pose[:, 2]
    cosine = np.cos(angle_delta)
    sine = np.sin(angle_delta)
    expected_xy = online_pose[:, :2] + np.column_stack(
        (
            cosine * relative_world[:, 0]
            - sine * relative_world[:, 1],
            sine * relative_world[:, 0]
            + cosine * relative_world[:, 1],
        )
    )
    errors = np.linalg.norm(
        measured_xy[scored] - expected_xy[scored], axis=1
    )
    measurement_rmse = (
        float(np.sqrt(np.mean(np.square(errors))))
        if errors.size
        else float("inf")
    )

    forecast_valid = np.asarray(
        [
            bool(
                round(
                    _float(
                        row,
                        "dynamic_obstacle_tracker_forecast_valid",
                    )
                )
            )
            for row in rows
        ]
    )
    finite_names = (
        "dynamic_obstacle_tracker_association_distance_m",
        "dynamic_obstacle_tracker_unobserved_duration_s",
        "dynamic_obstacle_tracker_innovation_nis",
        "probabilistic_obstacle_probability_mass",
        "probabilistic_obstacle_union_bound",
        "probabilistic_obstacle_maximum_step_probability",
    )
    diagnostics_finite = all(
        math.isfinite(_float(row, name))
        for row in rows
        for name in finite_names
    )
    expected_risk_flag = arm == "risk_enabled"
    risk_flag_consistent = all(
        bool(round(_float(row, "probabilistic_obstacle_risk_enabled")))
        == expected_risk_flag
        for row in rows
    )
    enabled_forecast_consumed = (
        arm != "risk_enabled"
        or all(
            int(round(_float(row, "probabilistic_obstacle_forecast_count")))
            == 1
            for index, row in enumerate(rows)
            if forecast_valid[index]
        )
    )

    route_start, route_goal = _route_points(config)
    initial_distance = float(np.linalg.norm(route_goal - route_start))
    final_distance = float(metrics["final_goal_distance"])
    completion = float(
        np.clip(
            (initial_distance - final_distance) / initial_distance,
            0.0,
            1.0,
        )
    )
    clearance = metrics.get("minimum_clearance")
    clearance = float(clearance) if clearance is not None else float("-inf")
    planner_times = np.asarray(
        [_float(row, "planner_compute_ms") for row in rows],
        dtype=np.float64,
    )
    duration = float(config["experiment"]["max_steps"]) * dt

    return {
        "run_order": int(run_order),
        "experimental_key": f"seed{int(seed)}::{arm}",
        "seed": int(seed),
        "arm": str(arm),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "steps": int(len(rows)),
        "termination_reason": str(metrics["termination_reason"]),
        "success": int(bool(metrics["success"])),
        "collision": int(bool(metrics["collision"])),
        "final_goal_distance_m": final_distance,
        "completion": completion,
        "minimum_clearance_m": clearance,
        "planner_p95_compute_ms": float(
            np.percentile(planner_times, 95)
        ),
        "route_crossings_40s": route_crossing_count(
            trajectory,
            route_start,
            route_goal,
            duration,
        ),
        "tracker_forecast_availability": float(
            np.mean(forecast_valid)
        ),
        "visible_steps": int(np.sum(visible)),
        "associated_visible_steps": int(np.sum(scored)),
        "visible_measurement_rmse_m": measurement_rmse,
        "maximum_visible_measurement_error_m": (
            float(np.max(errors)) if errors.size else float("inf")
        ),
        "localization_position_rmse_m": float(
            np.sqrt(np.mean(np.square(localization_position_errors)))
        ),
        "localization_final_position_error_m": float(
            localization_position_errors[-1]
        ),
        "localization_heading_rmse_rad": float(
            np.sqrt(np.mean(np.square(localization_heading_errors)))
        ),
        "forecast_contract_valid": int(
            diagnostics_finite
            and risk_flag_consistent
            and enabled_forecast_consumed
        ),
        "diagnostics_finite": int(diagnostics_finite),
        "risk_flag_consistent": int(risk_flag_consistent),
        "enabled_forecast_consumed": int(enabled_forecast_consumed),
    }


def _behavior_divergence(disabled_dir, enabled_dir):
    disabled = _read_csv(disabled_dir / "trajectory.csv")
    enabled = _read_csv(enabled_dir / "trajectory.csv")
    count = min(len(disabled), len(enabled))
    if count <= 0:
        return float("inf"), True
    a = np.asarray(
        [
            (_float(row, "executed_v"), _float(row, "executed_omega"))
            for row in disabled[:count]
        ]
    )
    b = np.asarray(
        [
            (_float(row, "executed_v"), _float(row, "executed_omega"))
            for row in enabled[:count]
        ]
    )
    maximum = float(np.max(np.linalg.norm(a - b, axis=1)))
    divergent = maximum > 1.0e-3 or len(disabled) != len(enabled)
    return maximum, divergent


def _paired_analysis(config, records, output_dir):
    by_key = {
        (int(row["seed"]), str(row["arm"])): row for row in records
    }
    pairs = []
    for seed in config["online_smoke"]["development_seeds"]:
        disabled = by_key[(int(seed), "risk_disabled")]
        enabled = by_key[(int(seed), "risk_enabled")]
        disabled_dir = ROOT / disabled["run_dir"]
        enabled_dir = ROOT / enabled["run_dir"]
        control_delta, divergent = _behavior_divergence(
            disabled_dir, enabled_dir
        )
        pairs.append(
            {
                "seed": int(seed),
                "disabled_collision": int(disabled["collision"]),
                "enabled_collision": int(enabled["collision"]),
                "enabled_collision_regression": int(
                    bool(enabled["collision"])
                    and not bool(disabled["collision"])
                ),
                "disabled_completion": float(disabled["completion"]),
                "enabled_completion": float(enabled["completion"]),
                "enabled_completion_delta": float(
                    enabled["completion"] - disabled["completion"]
                ),
                "disabled_minimum_clearance_m": float(
                    disabled["minimum_clearance_m"]
                ),
                "enabled_minimum_clearance_m": float(
                    enabled["minimum_clearance_m"]
                ),
                "enabled_clearance_delta_m": float(
                    enabled["minimum_clearance_m"]
                    - disabled["minimum_clearance_m"]
                ),
                "maximum_paired_control_delta": control_delta,
                "behaviorally_divergent": int(divergent),
            }
        )

    gate_config = config["online_smoke"]["gate"]
    enabled_records = [
        row for row in records if row["arm"] == "risk_enabled"
    ]
    collision_regressions = int(
        sum(row["enabled_collision_regression"] for row in pairs)
    )
    minimum_completion_delta = float(
        min(row["enabled_completion_delta"] for row in pairs)
    )
    median_clearance_delta = float(
        np.median(
            [row["enabled_clearance_delta_m"] for row in pairs]
        )
    )
    divergence_count = int(
        sum(row["behaviorally_divergent"] for row in pairs)
    )
    gates = {
        "route_crossings": {
            "value": int(
                min(row["route_crossings_40s"] for row in records)
            ),
            "threshold": int(
                gate_config["minimum_route_crossings_per_episode"]
            ),
            "pass": bool(
                min(row["route_crossings_40s"] for row in records)
                >= int(
                    gate_config[
                        "minimum_route_crossings_per_episode"
                    ]
                )
            ),
        },
        "tracker_forecast_availability": {
            "value": float(
                min(
                    row["tracker_forecast_availability"]
                    for row in records
                )
            ),
            "threshold": float(
                gate_config["minimum_tracker_forecast_availability"]
            ),
            "pass": bool(
                min(
                    row["tracker_forecast_availability"]
                    for row in records
                )
                >= float(
                    gate_config[
                        "minimum_tracker_forecast_availability"
                    ]
                )
            ),
        },
        "visible_measurement_rmse_m": {
            "value": float(
                max(
                    row["visible_measurement_rmse_m"]
                    for row in records
                )
            ),
            "threshold": float(
                gate_config["maximum_visible_measurement_rmse_m"]
            ),
            "pass": bool(
                max(
                    row["visible_measurement_rmse_m"]
                    for row in records
                )
                <= float(
                    gate_config[
                        "maximum_visible_measurement_rmse_m"
                    ]
                )
            ),
        },
        "forecast_contract": {
            "value": int(
                sum(row["forecast_contract_valid"] for row in records)
            ),
            "threshold": int(len(records)),
            "pass": bool(
                all(row["forecast_contract_valid"] for row in records)
            ),
        },
        "enabled_collision_regressions": {
            "value": collision_regressions,
            "threshold": int(
                gate_config["maximum_enabled_collision_regressions"]
            ),
            "pass": bool(
                collision_regressions
                <= int(
                    gate_config[
                        "maximum_enabled_collision_regressions"
                    ]
                )
            ),
        },
        "enabled_completion_regression": {
            "value": float(-minimum_completion_delta),
            "threshold": float(
                gate_config["maximum_enabled_completion_regression"]
            ),
            "pass": bool(
                minimum_completion_delta
                >= -float(
                    gate_config[
                        "maximum_enabled_completion_regression"
                    ]
                )
            ),
        },
        "median_clearance_delta_m": {
            "value": median_clearance_delta,
            "threshold": float(
                gate_config[
                    "minimum_enabled_median_clearance_delta_m"
                ]
            ),
            "pass": bool(
                median_clearance_delta
                >= float(
                    gate_config[
                        "minimum_enabled_median_clearance_delta_m"
                    ]
                )
            ),
        },
        "behaviorally_divergent_pairs": {
            "value": divergence_count,
            "threshold": int(
                gate_config["minimum_behaviorally_divergent_pairs"]
            ),
            "pass": bool(
                divergence_count
                >= int(
                    gate_config[
                        "minimum_behaviorally_divergent_pairs"
                    ]
                )
            ),
        },
        "enabled_planner_p95_compute_ms": {
            "value": float(
                max(
                    row["planner_p95_compute_ms"]
                    for row in enabled_records
                )
            ),
            "threshold": float(
                gate_config["maximum_enabled_planner_p95_compute_ms"]
            ),
            "pass": bool(
                max(
                    row["planner_p95_compute_ms"]
                    for row in enabled_records
                )
                <= float(
                    gate_config[
                        "maximum_enabled_planner_p95_compute_ms"
                    ]
                )
            ),
        },
    }
    result = {
        "schema_version": 1,
        "stage": "development_paired_online_smoke",
        "overall_pass": bool(
            all(item["pass"] for item in gates.values())
        ),
        "independent_unit": "complete_episode",
        "truth_use": "posthoc_audit_only",
        "gates": gates,
    }
    write_csv(output_dir / "paired_effects.csv", pairs)
    write_json(output_dir / "gate.json", result)
    return pairs, result


def _write_readme(output_dir, gate_result):
    status = "PASS" if gate_result["overall_pass"] else "FAIL"
    lines = [
        "# Online V3 Tracking and Risk-Aware MPPI Smoke",
        "",
        f"Overall Gate: **{status}**",
        "",
        "This is a paired development smoke, not a formal safety claim.",
        "The controller used causal LaserScan-derived forecasts only. Frozen",
        "V3 truth was regenerated after each run solely for offline audit.",
        "",
        "See `gate.json`, `episode_summary.csv`, `paired_effects.csv`, and",
        "the per-run `runs/` directories for the complete evidence.",
        "",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(config_path=DEFAULT_CONFIG, output_override=None):
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    scope_audit = _validate_scope(config)
    smoke = config["online_smoke"]
    output_dir = Path(
        output_override or (ROOT / str(smoke["output_dir"]))
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty smoke output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(
        smoke["development_seeds"],
        smoke["arms"],
        smoke["schedule_seed"],
    )
    write_yaml(output_dir / "config_resolved.yaml", config)
    write_json(
        output_dir / "schedule.json",
        {
            "schema_version": 1,
            "schedule_seed": int(smoke["schedule_seed"]),
            "paired_by_complete_episode": True,
            "common_random_numbers": True,
            "jobs": schedule,
        },
    )
    write_json(output_dir / "scope_audit.json", scope_audit)
    write_json(
        output_dir / "environment.json", environment_manifest(ROOT)
    )

    trajectories = {
        int(seed): _obstacle_trajectory(config, int(seed))
        for seed in smoke["development_seeds"]
    }
    records = []
    for job in schedule:
        seed = int(job["seed"])
        arm = str(job["arm"])
        run_config = deepcopy(config)
        run_config["experiment"]["seed"] = seed
        run_config["planner"]["seed"] = seed
        run_config["planner"][
            "probabilistic_obstacle_risk_enabled"
        ] = arm == "risk_enabled"
        run_config["experiment"]["name"] = (
            f"online_v3_smoke_seed{seed}_{arm}"
        )
        run_dir = output_dir / "runs" / arm / f"seed_{seed}"
        print(
            f"[{job['run_order'] + 1}/{len(schedule)}] "
            f"seed={seed} arm={arm}",
            flush=True,
        )
        ExperimentRunner(
            run_config,
            ROOT,
            output_dir=run_dir,
            headless=True,
        ).run()
        record = _audit_episode(
            config,
            seed,
            arm,
            job["run_order"],
            run_dir,
            trajectories[seed],
        )
        records.append(record)
        write_csv(
            output_dir / "episode_summary.csv",
            sorted(records, key=lambda item: item["run_order"]),
        )
        print(
            "  "
            f"termination={record['termination_reason']} "
            f"collision={record['collision']} "
            f"completion={record['completion']:.3f} "
            f"availability={record['tracker_forecast_availability']:.3f} "
            f"rmse={record['visible_measurement_rmse_m']:.4f}",
            flush=True,
        )

    records = sorted(records, key=lambda item: item["run_order"])
    write_csv(output_dir / "episode_summary.csv", records)
    pairs, gate_result = _paired_analysis(config, records, output_dir)
    _write_readme(output_dir, gate_result)

    artifact_paths = [
        output_dir / name
        for name in (
            "config_resolved.yaml",
            "schedule.json",
            "scope_audit.json",
            "environment.json",
            "episode_summary.csv",
            "paired_effects.csv",
            "gate.json",
            "README.md",
        )
    ]
    provenance = {
        "schema_version": 1,
        "stage": "development_paired_online_smoke",
        "config_path": str(config_path.relative_to(ROOT)),
        **repository_provenance(ROOT),
        "artifacts": {
            path.name: {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(
        output_dir / "summary.json",
        {
            "overall_pass": gate_result["overall_pass"],
            "episode_count": len(records),
            "pair_count": len(pairs),
            "output_dir": str(output_dir),
        },
    )
    return gate_result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
