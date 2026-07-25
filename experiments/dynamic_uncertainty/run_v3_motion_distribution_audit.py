"""Read-only distribution and saturation audit for the frozen V3 generator."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from mobile_robot_mppi.obstacles.artifacts import write_json
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import (
    V3_PROCESS_NAMES,
    generate_patrol_trajectory,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "research_artifacts/dynamic_obstacle_generator_v3_qualification/"
    "analysis/motion_distribution_audit.json"
)


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _stopped_segment_durations(speeds, threshold, dt):
    stopped = np.asarray(speeds <= threshold, dtype=bool)
    changes = np.diff(np.concatenate(([False], stopped, [False])).astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [float((end - start) * dt) for start, end in zip(starts, ends)]


def run_audit(config_path=DEFAULT_CONFIG, output_path=DEFAULT_OUTPUT):
    config = _load_yaml(config_path)
    profiles = noise_profiles_from_mapping(config["noise_profiles"])
    dt = float(config["trajectory"]["dt"])
    limits = config["physical_limits"]
    process_payload = {}
    global_values = {
        "acceleration_mps2": [],
        "moving_yaw_rate_radps": [],
        "single_step_displacement_m": [],
        "stop_duration_s": [],
        "dwell_duration_s": [],
    }
    global_saturated_acceleration = 0
    global_acceleration_steps = 0
    global_saturated_yaw = 0
    global_yaw_steps = 0
    for process in V3_PROCESS_NAMES:
        values = {name: [] for name in global_values}
        saturated_acceleration = 0
        acceleration_steps = 0
        saturated_yaw = 0
        yaw_steps_count = 0
        for seed in range(730100021, 730100041):
            for profile in profiles.values():
                trajectory = generate_patrol_trajectory(
                    process, seed, profile, config
                )
                velocities = trajectory.states[:, 2:]
                speeds = np.linalg.norm(velocities, axis=1)
                acceleration = np.linalg.norm(
                    np.diff(velocities, axis=0) / dt, axis=1
                )
                speed_change = np.diff(speeds)
                applicable_limit = np.where(
                    speed_change < -1.0e-7,
                    float(limits["maximum_braking_mps2"]),
                    float(limits["maximum_acceleration_mps2"]),
                )
                saturated_acceleration += int(
                    np.sum(acceleration >= 0.995 * applicable_limit)
                )
                acceleration_steps += acceleration.size
                headings = np.unwrap(
                    np.arctan2(velocities[:, 1], velocities[:, 0])
                )
                moving = (
                    speeds[:-1]
                    >= float(limits["yaw_rate_audit_minimum_speed_mps"])
                ) & (
                    speeds[1:]
                    >= float(limits["yaw_rate_audit_minimum_speed_mps"])
                )
                yaw = np.abs(np.diff(headings) / dt)[moving]
                saturated_yaw += int(
                    np.sum(
                        yaw
                        >= 0.995
                        * float(limits["maximum_moving_yaw_rate_radps"])
                    )
                )
                yaw_steps_count += yaw.size
                displacement = np.linalg.norm(
                    np.diff(trajectory.states[:, :2], axis=0), axis=1
                )
                stop_durations = _stopped_segment_durations(
                    speeds,
                    float(limits["stopped_speed_threshold_mps"]),
                    dt,
                )
                dwell_durations = [
                    float(event["parameters"]["dwell_s"])
                    for event in trajectory.metadata["events"]
                    if event["kind"] == "waypoint_arrival"
                ]
                values["acceleration_mps2"].extend(acceleration.tolist())
                values["moving_yaw_rate_radps"].extend(yaw.tolist())
                values["single_step_displacement_m"].extend(
                    displacement.tolist()
                )
                values["stop_duration_s"].extend(stop_durations)
                values["dwell_duration_s"].extend(dwell_durations)
        for name in global_values:
            global_values[name].extend(values[name])
        global_saturated_acceleration += saturated_acceleration
        global_acceleration_steps += acceleration_steps
        global_saturated_yaw += saturated_yaw
        global_yaw_steps += yaw_steps_count
        process_payload[process] = {
            "distributions": {
                name: distribution_summary(samples)
                for name, samples in values.items()
            },
            "acceleration_saturation_occupancy": float(
                saturated_acceleration / acceleration_steps
            ),
            "yaw_rate_saturation_occupancy": float(
                saturated_yaw / yaw_steps_count
            ),
        }
    payload = {
        "schema_version": 1,
        "generator_version": config["generator_version"],
        "trajectory_count": 160,
        "saturation_definition": (
            "absolute value >= 99.5% of the applicable registered limit"
        ),
        "overall": {
            "distributions": {
                name: distribution_summary(samples)
                for name, samples in global_values.items()
            },
            "acceleration_saturation_occupancy": float(
                global_saturated_acceleration / global_acceleration_steps
            ),
            "yaw_rate_saturation_occupancy": float(
                global_saturated_yaw / global_yaw_steps
            ),
        },
        "by_process": process_payload,
    }
    output_path = Path(output_path)
    write_json(output_path, payload)
    markdown = output_path.with_suffix(".md")
    overall = payload["overall"]
    markdown.write_text(
        "# V3 motion-distribution audit\n\n"
        f"- Trajectories: {payload['trajectory_count']}\n"
        f"- Acceleration saturation occupancy: "
        f"{overall['acceleration_saturation_occupancy']:.6f}\n"
        f"- Moving yaw-rate saturation occupancy: "
        f"{overall['yaw_rate_saturation_occupancy']:.6f}\n"
        f"- Acceleration p95/p99/max: "
        f"{overall['distributions']['acceleration_mps2']['p95']:.6f} / "
        f"{overall['distributions']['acceleration_mps2']['p99']:.6f} / "
        f"{overall['distributions']['acceleration_mps2']['max']:.6f} m/s²\n"
        f"- Yaw-rate p95/p99/max: "
        f"{overall['distributions']['moving_yaw_rate_radps']['p95']:.6f} / "
        f"{overall['distributions']['moving_yaw_rate_radps']['p99']:.6f} / "
        f"{overall['distributions']['moving_yaw_rate_radps']['max']:.6f} rad/s\n",
        encoding="utf-8",
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run_audit(args.config, args.output)
    print(json.dumps(payload["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
