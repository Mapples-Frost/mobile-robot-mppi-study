#!/usr/bin/env python3
"""Audit the preregistered L56 fixed-plant high-dynamic data gate."""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _episode_key(row):
    return (
        str(row["scene"]), str(row["physics_domain"]),
        int(row["episode_seed"]), str(row["condition"]),
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _finite_array(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite values in %s" % field)
    return values


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    specification = config["residual_dataset"]
    gate_config = specification["data_gate"]
    block = int(specification.get("source_block", 0))
    condition = str(specification.get("source_condition", "traditional_nominal"))
    block_dir = Path(input_dir) / ("block_%d" % block)
    episodes = _read_csv(block_dir / "episodes.csv")
    steps = _read_csv(block_dir / "factorial_steps.csv")
    metadata = json.loads(
        (block_dir / "metadata.json").read_text(encoding="utf-8")
    )

    scenes = []
    for item in design["scenes"]:
        scene_config = load_yaml(_resolved(item["path"]))
        scenes.append(str(scene_config["scene"]["name"]))
    domains = [str(item["name"]) for item in design["physics_domains"]]
    seeds = [int(value) for value in design["development_episode_seeds"]]
    expected = {
        (scene, domain, seed, condition)
        for scene in scenes for domain in domains for seed in seeds
    }
    episode_counts = Counter(_episode_key(row) for row in episodes)
    duplicate_episode_keys = sorted(
        key for key, count in episode_counts.items() if count != 1
    )
    observed = set(episode_counts)

    step_counts = Counter(
        _episode_key(row) + (int(row["step"]),) for row in steps
    )
    duplicate_step_keys = sum(count - 1 for count in step_counts.values() if count > 1)
    step_episode_keys = {_episode_key(row) for row in steps}

    per_scene = []
    for scene in scenes:
        selected = [row for row in episodes if str(row["scene"]) == scene]
        per_scene.append({
            "scene": scene,
            "episodes": len(selected),
            "success_rate": float(np.mean([
                _as_bool(row["success"]) for row in selected
            ])) if selected else 0.0,
            "collisions": int(sum(_as_bool(row["collision"]) for row in selected)),
            "mean_steps": float(np.mean([
                int(row["steps"]) for row in selected
            ])) if selected else 0.0,
        })

    executed_v = _finite_array(steps, "executed_v")
    executed_omega = _finite_array(steps, "executed_omega")
    applied_v = _finite_array(steps, "applied_v")
    applied_omega = _finite_array(steps, "applied_omega")
    state_v = _finite_array(steps, "v")
    state_omega = _finite_array(steps, "omega")
    safety_override = _finite_array(steps, "safety_override")
    run_configs = []
    missing_run_configs = []
    for scene, domain, seed, expected_condition in sorted(expected):
        path = (
            block_dir / "runs" / scene / domain / expected_condition
            / ("seed_%d" % seed) / "config_resolved.yaml"
        )
        if not path.exists():
            missing_run_configs.append(str(path))
        else:
            run_configs.append(load_yaml(path))
    plant_contracts = {_canonical(item["plant"]) for item in run_configs}
    action_contracts = {_canonical(item["action_space"]) for item in run_configs}
    sensor_contracts = {_canonical(item["sensors"]) for item in run_configs}
    isolated_sensor_contract = bool(
        run_configs
        and all(item["sensors"].get("pose_source") == "ground_truth"
                and item["sensors"].get("twist_source") == "ground_truth"
                and float(item["sensors"].get("latency", 0.0)) == 0.0
                and not any(float(value) != 0.0 for value in item["sensors"].get(
                    "odom_noise_std", (0.0, 0.0, 0.0)
                ))
                and not any(float(value) != 0.0 for value in item["sensors"].get(
                    "twist_noise_std", (0.0, 0.0)
                ))
                for item in run_configs)
    )
    # The factorial root is a scheduling config; scene files provide the
    # actual action space.  Derive saturation bounds from the resolved runtime
    # contract rather than from an inherited parent experiment.
    action = run_configs[0]["action_space"] if run_configs else config["action_space"]
    v_index = list(action["names"]).index("v_cmd")
    omega_index = list(action["names"]).index("omega_cmd")
    v_upper = float(action["upper"][v_index])
    omega_lower = float(action["lower"][omega_index])
    omega_upper = float(action["upper"][omega_index])
    tolerance = 1e-9

    success_rate = float(np.mean([
        _as_bool(row["success"]) for row in episodes
    ])) if episodes else 0.0
    collisions = int(sum(_as_bool(row["collision"]) for row in episodes))
    metrics = {
        "episodes": len(episodes),
        "steps": len(steps),
        "success_rate": success_rate,
        "minimum_per_scene_success_rate": min(
            row["success_rate"] for row in per_scene
        ) if per_scene else 0.0,
        "collisions": collisions,
        "safety_override_fraction": float(np.mean(safety_override)),
        "applied_v_q95": float(np.quantile(applied_v, 0.95)),
        "abs_applied_omega_q95": float(np.quantile(np.abs(applied_omega), 0.95)),
        "state_v_std": float(np.std(state_v)),
        "state_omega_std": float(np.std(state_omega)),
        "executed_v_upper_saturation_fraction": float(np.mean(
            executed_v >= v_upper - tolerance
        )),
        "executed_omega_saturation_fraction": float(np.mean(
            (executed_omega <= omega_lower + tolerance)
            | (executed_omega >= omega_upper - tolerance)
        )),
    }
    metadata_integrity = bool(
        int(metadata.get("model_block", -1)) == block
        and sorted(int(value) for value in metadata.get("episode_seeds", []))
        == sorted(seeds)
        and metadata.get("conditions") == [condition]
        and not metadata.get("sealed_confirmation_seeds_used")
        and not metadata.get("previous_protected_seeds_used")
        and int(metadata.get("episodes", -1)) == len(expected)
    )
    artifact_integrity = bool(
        observed == expected
        and not duplicate_episode_keys
        and step_episode_keys == expected
        and duplicate_step_keys == 0
        and not missing_run_configs
        and len(plant_contracts) == 1
        and len(action_contracts) == 1
        and len(sensor_contracts) == 1
        and isolated_sensor_contract
        and metadata_integrity
        and len(episodes) == int(gate_config["expected_episodes"])
    )
    checks = {
        "artifact_integrity": artifact_integrity,
        "minimum_success_rate": metrics["success_rate"]
        >= float(gate_config["minimum_success_rate"]),
        "minimum_per_scene_success_rate": metrics["minimum_per_scene_success_rate"]
        >= float(gate_config["minimum_per_scene_success_rate"]),
        "maximum_collisions": metrics["collisions"]
        <= int(gate_config["maximum_collisions"]),
        "maximum_safety_override_fraction": metrics["safety_override_fraction"]
        <= float(gate_config["maximum_safety_override_fraction"]),
        "minimum_applied_v_q95": metrics["applied_v_q95"]
        >= float(gate_config["minimum_applied_v_q95"]),
        "minimum_abs_applied_omega_q95": metrics["abs_applied_omega_q95"]
        >= float(gate_config["minimum_abs_applied_omega_q95"]),
        "minimum_state_v_std": metrics["state_v_std"]
        >= float(gate_config["minimum_state_v_std"]),
        "minimum_state_omega_std": metrics["state_omega_std"]
        >= float(gate_config["minimum_state_omega_std"]),
        "maximum_v_upper_saturation_fraction": metrics[
            "executed_v_upper_saturation_fraction"
        ] <= float(gate_config["maximum_v_upper_saturation_fraction"]),
        "maximum_omega_saturation_fraction": metrics[
            "executed_omega_saturation_fraction"
        ] <= float(gate_config["maximum_omega_saturation_fraction"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        checks["finite_metrics"] = False
    else:
        checks["finite_metrics"] = True
    return {
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
        "per_scene": per_scene,
        "integrity": {
            "expected_episode_keys": len(expected),
            "observed_episode_keys": len(observed),
            "duplicate_episode_keys": len(duplicate_episode_keys),
            "duplicate_step_keys": duplicate_step_keys,
            "missing_run_configs": len(missing_run_configs),
            "plant_contract_count": len(plant_contracts),
            "action_contract_count": len(action_contracts),
            "sensor_contract_count": len(sensor_contracts),
            "isolated_sensor_contract": isolated_sensor_contract,
            "metadata_integrity": metadata_integrity,
        },
        "thresholds": dict(gate_config),
        "interpretation_guard": (
            "This gate validates data suitability only; it is not evidence "
            "that ICODE improves prediction or closed-loop control."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    result = summarize(config, _resolved(args.input_dir))
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "high_dynamic_data_gate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "high_dynamic_data_gate_per_scene.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("scene", "episodes", "success_rate", "collisions", "mean_steps")
        )
        writer.writeheader()
        writer.writerows(result["per_scene"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
