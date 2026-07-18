#!/usr/bin/env python3
"""Independent three-arm confirmation for the retained ICODE + RL package."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from experiments.rl.run_contextual_covariance_jerk_screening import _blocked_schedule
from experiments.rl.run_cross_layer_sample_efficiency_factorial import (
    METRICS,
    _configure_arm,
    _descriptives,
    _paired_contrast,
)
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit(rows, spec):
    names = {str(arm["name"]) for arm in spec["arms"]}
    expected_blocks = (
        len(spec["scenes"]) * len(spec["physics_domains"])
        * len(spec["seeds"])
    )
    blocks = {}
    keys = []
    for row in rows:
        block = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        blocks.setdefault(block, set()).add(str(row["condition"]))
        keys.append(block + (str(row["condition"]),))
    finite = all(
        np.isfinite(float(row[metric]))
        for row in rows for metric in METRICS
    )
    positions = {
        name: [
            sum(
                str(row["condition"]) == name
                and int(row["run_position"]) == position
                for row in rows
            )
            for position in range(len(names))
        ]
        for name in sorted(names)
    }
    counts = [value for values in positions.values() for value in values]
    balanced = bool(counts and max(counts) == min(counts))
    return {
        "passed": bool(
            len(rows) == expected_blocks * len(names)
            and len(set(keys)) == len(rows)
            and len(blocks) == expected_blocks
            and all(value == names for value in blocks.values())
            and finite and balanced
        ),
        "expected_rows": expected_blocks * len(names),
        "observed_rows": len(rows),
        "expected_blocks": expected_blocks,
        "complete_blocks": sum(value == names for value in blocks.values()),
        "finite_metrics": bool(finite),
        "balanced_run_positions": balanced,
        "run_position_counts": positions,
        "successes": sum(bool(row["success"]) for row in rows),
        "collisions": sum(bool(row["collision"]) for row in rows),
    }


def _safety(contrast):
    return bool(
        contrast["success_delta_mean"] >= 0.0
        and contrast["collision_delta_mean"] <= 0.0
    )


def _gates(contrasts, descriptives, spec):
    package = contrasts["proposed_vs_nominal"]
    half = contrasts["proposed_vs_icode_k100"]
    dynamics = contrasts["icode_vs_nominal_k100"]
    h1 = {
        "safety": _safety(package),
        "rmse_superiority": package["cross_track_rmse_delta_ci95"][1] < 0.0,
        "elapsed_superiority": package["elapsed_s_delta_ci95"][1] < 0.0,
        "absolute_compute": descriptives[str(spec.get(
            "proposed_condition", "proposed_icode_contextual_k50"
        ))][
            "planner_compute_ms_mean"
        ]["mean"] <= float(spec["planner_mean_deadline_ms"]),
    }
    if bool(spec.get("require_proposed_rmse_superiority", False)):
        rmse_check = half["cross_track_rmse_delta_ci95"][1] < 0.0
    else:
        rmse_check = half["cross_track_rmse_delta_ci95"][1] <= float(
            spec["cross_track_noninferiority_margin_m"]
        )
    compute_margin = spec.get("planner_compute_noninferiority_margin_ms")
    if compute_margin is None:
        compute_check = half["planner_compute_ms_mean_delta_ci95"][1] < 0.0
    else:
        compute_check = half["planner_compute_ms_mean_delta_ci95"][1] <= float(
            compute_margin
        )
    h2 = {
        "safety": _safety(half),
        "rmse_requirement": bool(rmse_check),
        "issued_jerk_noninferiority": half["control_jerk_delta_ci95"][1]
        <= float(spec["issued_jerk_noninferiority_margin"]),
        "applied_jerk_noninferiority": half[
            "applied_control_jerk_delta_ci95"
        ][1] <= float(spec["applied_jerk_noninferiority_margin"]),
        "elapsed_superiority": half["elapsed_s_delta_ci95"][1] < 0.0,
        "compute_requirement": bool(compute_check),
    }
    h3 = {
        "safety": _safety(dynamics),
        "rmse_superiority": dynamics["cross_track_rmse_delta_ci95"][1] < 0.0,
    }
    return {
        "h1_package_vs_nominal": {"passed": all(h1.values()), "checks": h1},
        "h2_policy_vs_icode_k100": {
            "passed": all(h2.values()), "checks": h2
        },
        "h3_icode_contribution": {"passed": all(h3.values()), "checks": h3},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    base = load_yaml(ROOT / spec["base_config"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = _blocked_schedule(spec)
    with (output / "schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2, sort_keys=True)
    with (output / "design_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    selected_blocks = sorted({int(item["block_index"]) for item in schedule})
    if args.max_blocks is not None:
        selected_blocks = selected_blocks[:int(args.max_blocks)]
    selected_blocks = set(selected_blocks)
    rows = []
    for item in schedule:
        if int(item["block_index"]) not in selected_blocks:
            continue
        arm = item["arm"]
        domain = item["physics_domain"]
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        config = _configure_arm(config, arm, spec)
        config["experiment"]["seed"] = int(item["seed"])
        config["experiment"]["name"] = "l107_%s_%s_%s_%d" % (
            config["scene"]["name"], domain["name"], arm["name"], item["seed"]
        )
        run_dir = (
            output / "runs" / domain["name"] / config["scene"]["name"]
            / arm["name"] / ("seed_%d" % item["seed"])
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                episode = json.load(handle)
        else:
            episode = ExperimentRunner(
                copy.deepcopy(config), ROOT, run_dir, headless=True
            ).run().summary
        row = dict(episode)
        row.update({
            "block_index": int(item["block_index"]),
            "run_position": int(item["run_position"]),
            "scene": str(config["scene"]["name"]),
            "physics_domain": str(domain["name"]),
            "condition": str(arm["name"]),
            "seed": int(item["seed"]),
            "elapsed_s": float(episode["steps"])
            * float(config["experiment"]["control_dt"]),
        })
        rows.append(row)
        _write_csv(output / "episodes.csv", rows)

    if len(rows) != len(schedule):
        print(json.dumps({
            "complete": False,
            "completed_runs": len(rows),
            "expected_runs": len(schedule),
        }, indent=2))
        return 0

    bootstrap = int(spec["bootstrap_seed"])
    proposed = str(spec.get(
        "proposed_condition", "proposed_icode_contextual_k50"
    ))
    nominal = str(spec.get("nominal_condition", "nominal_fixed_k100"))
    icode = str(spec.get("icode_condition", "icode_fixed_k100"))
    contrasts = {
        "proposed_vs_nominal": _paired_contrast(
            rows, proposed, nominal, bootstrap,
        ),
        "proposed_vs_icode_k100": _paired_contrast(
            rows, proposed, icode, bootstrap + 100,
        ),
        "icode_vs_nominal_k100": _paired_contrast(
            rows, icode, nominal, bootstrap + 200,
        ),
    }
    descriptives = _descriptives(rows)
    gates = _gates(contrasts, descriptives, spec)
    audit = _audit(rows, spec)
    summary = {
        "schema_version": 1,
        "design_id": str(spec["design_id"]),
        "phase": str(spec["phase"]),
        "completed_runs": len(rows),
        "audit": audit,
        "descriptives": descriptives,
        "contrasts": contrasts,
        "gates": gates,
        "primary_gate_passed": bool(
            audit["passed"] and all(value["passed"] for value in gates.values())
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
