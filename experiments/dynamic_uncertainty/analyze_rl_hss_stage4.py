"""Audit and summarize the preregistered Stage 4 RL/HSS matrix.

The analyzer is intentionally descriptive.  Complete MuJoCo episodes are the
independent units, while the three residual checkpoints remain model blocks
rather than extra environment replicates.  A preregistered collision stop is
therefore retained as an incomplete matrix instead of being resumed or imputed.
"""

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    REQUIRED_RUN_FILES,
    _resolve,
    _run_dir,
    _sha256,
    build_schedule,
    configure_job,
    validate_job_config,
    validate_protocol,
)
from mobile_robot_mppi.core.config import load_yaml


DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_uncertainty_rl_hss_stage4_amendment1.yaml"
)
RESULT_REPORT = (
    ROOT
    / "docs"
    / "experiments"
    / "dynamic_uncertainty"
    / "RL_HSS_STAGE4_RESULT.md"
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return value


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty episode summary")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_binding(path):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _number(value, default=0.0):
    if value in (None, ""):
        return float(default)
    return float(value)


def _flag(value):
    if isinstance(value, bool):
        return value
    return _number(value) > 0.5


def _fraction(rows, name):
    return float(np.mean([_flag(row.get(name)) for row in rows]))


def _finite_values(rows, name):
    values = [
        _number(row[name])
        for row in rows
        if row.get(name) not in (None, "")
    ]
    return np.asarray(values, dtype=np.float64)


def _canonical_config(config):
    """Remove the host-dependent source-path annotation before comparison."""

    value = deepcopy(config)
    value.pop("_config_path", None)
    return value


def _dynamics_name(job):
    if job["condition"] == "nominal":
        return "nominal"
    return "residual_block_%d" % int(job["model_block"])


def _trajectory_audit(rows, job):
    if not rows:
        raise ValueError("trajectory is empty for %s" % job["experimental_key"])
    names = set(rows[0])
    required_probability = {
        "probabilistic_obstacle_candidate_feasible_fraction",
        "probabilistic_obstacle_forecast_count",
        "probabilistic_obstacle_hard_violation",
        "probabilistic_obstacle_maximum_step_probability",
        "probabilistic_obstacle_risk_enabled",
    }
    required_scan = {"executed_v", "executed_omega", "safety_reason"}
    probability_values = _finite_values(
        rows, "probabilistic_obstacle_maximum_step_probability"
    )
    result = {
        "row_count": len(rows),
        "probability_diagnostics_present": required_probability <= names,
        "scan_guard_diagnostics_present": required_scan <= names,
        "probabilistic_risk_enabled_fraction": _fraction(
            rows, "probabilistic_obstacle_risk_enabled"
        ),
        "tracker_forecast_valid_fraction": _fraction(
            rows, "dynamic_obstacle_tracker_forecast_valid"
        ),
        "tracker_forecast_availability_mean": float(
            np.mean(
                _finite_values(
                    rows, "dynamic_obstacle_tracker_forecast_availability"
                )
            )
        ),
        "maximum_selected_probability": (
            float(np.max(probability_values)) if probability_values.size else None
        ),
        "hard_violation_fraction": _fraction(
            rows, "probabilistic_obstacle_hard_violation"
        ),
        "fail_closed_fraction": _fraction(
            rows, "probabilistic_obstacle_fail_closed"
        ),
        "scan_guard_intervention_fraction": float(
            np.mean(
                [
                    str(row.get("safety_reason", ""))
                    not in ("", "front_clear")
                    for row in rows
                ]
            )
        ),
        "residual_shield_enabled_fraction": _fraction(
            rows, "residual_safety_shield_enabled"
        ),
    }
    if bool(job["rl_hss_enabled"]):
        rollouts = _finite_values(rows, "paper_total_rollouts")
        candidates = _finite_values(rows, "paper_candidates_per_iteration")
        result.update(
            {
                "hss_enabled_fraction": _fraction(
                    rows, "reliability_hss_enabled"
                ),
                "paper_total_rollouts_minimum": float(np.min(rollouts)),
                "paper_total_rollouts_maximum": float(np.max(rollouts)),
                "paper_candidates_per_iteration_minimum": float(
                    np.min(candidates)
                ),
                "paper_candidates_per_iteration_maximum": float(
                    np.max(candidates)
                ),
            }
        )
    else:
        result.update(
            {
                "hss_enabled_fraction": _fraction(
                    rows, "reliability_hss_enabled"
                ),
                "paper_total_rollouts_minimum": 0.0,
                "paper_total_rollouts_maximum": 0.0,
                "paper_candidates_per_iteration_minimum": 0.0,
                "paper_candidates_per_iteration_maximum": 0.0,
            }
        )
    return result


def _episode_record(job, run_dir, base_config, protocol, stage3_protocol):
    config = _yaml(run_dir / "config_resolved.yaml")
    metrics = _json(run_dir / "metrics.json")
    trajectory = _read_csv(run_dir / "trajectory.csv")
    expected = configure_job(base_config, job, protocol, stage3_protocol)
    validation_error = None
    try:
        validate_job_config(
            config, job, protocol, stage3_protocol, base_config
        )
    except ValueError as error:
        validation_error = str(error)
    audit = _trajectory_audit(trajectory, job)
    start = np.asarray(config["experiment"]["initial_state"][:2], dtype=np.float64)
    goal = np.asarray(config["task"]["position"], dtype=np.float64)
    initial_goal_distance = float(np.linalg.norm(goal - start))
    final_goal_distance = float(metrics["final_goal_distance"])
    completion = float(
        np.clip(
            (initial_goal_distance - final_goal_distance)
            / initial_goal_distance,
            0.0,
            1.0,
        )
    )
    return {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "dynamics": _dynamics_name(job),
        "condition": str(job["condition"]),
        "model_block": int(job["model_block"]),
        "checkpoint_seed": int(job["checkpoint_seed"]),
        "episode_seed": int(job["episode_seed"]),
        "rl_hss_enabled": bool(job["rl_hss_enabled"]),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "success": bool(metrics.get("success", False)),
        "collision": bool(metrics.get("collision", False)),
        "termination_reason": str(metrics.get("termination_reason", "")),
        "steps": int(metrics["steps"]),
        "completion": completion,
        "final_goal_distance_m": final_goal_distance,
        "minimum_clearance_m": float(metrics["minimum_clearance"]),
        "planner_p95_compute_ms": float(metrics["planner_compute_ms_p95"]),
        "planner_deadline_miss_rate": float(
            metrics.get("planner_deadline_miss_rate", 0.0)
        ),
        "paper_actor_baseline_first_action_l2_delta_mean": float(
            metrics.get("paper_actor_baseline_first_action_l2_delta_mean", 0.0)
        ),
        "paper_guided_minus_gaussian_cost_min_mean": float(
            metrics.get("paper_guided_minus_gaussian_cost_min_mean", 0.0)
        ),
        "reliability_authority_mean": float(
            metrics.get("reliability_authority_mean", 0.0)
        ),
        "reliability_dynamics_confidence_mean": float(
            metrics.get("reliability_dynamics_confidence_mean", 0.0)
        ),
        "reliability_proposal_fallback_fraction_mean": float(
            metrics.get("reliability_proposal_fallback_fraction_mean", 0.0)
        ),
        "residual_safety_shield_acceptance_fraction": float(
            metrics.get("residual_safety_shield_acceptance_fraction", 0.0)
        ),
        "resolved_config_matches_frozen_generator": (
            _canonical_config(config) == _canonical_config(expected)
        ),
        "runtime_contract_valid": validation_error is None,
        "runtime_contract_error": validation_error,
        **audit,
    }


def _paired_records(records):
    cells = {}
    for record in records:
        key = (
            record["dynamics"],
            record["model_block"],
            record["episode_seed"],
        )
        cells.setdefault(key, {})[bool(record["rl_hss_enabled"])] = record
    paired = []
    unpaired = []
    for key, arms in sorted(cells.items()):
        if set(arms) != {False, True}:
            unpaired.append(
                {
                    "dynamics": key[0],
                    "model_block": key[1],
                    "episode_seed": key[2],
                    "available_arms": [
                        "rl_hss_on" if value else "rl_hss_off"
                        for value in sorted(arms)
                    ],
                }
            )
            continue
        off = arms[False]
        on = arms[True]
        paired.append(
            {
                "dynamics": key[0],
                "model_block": key[1],
                "episode_seed": key[2],
                "rl_off_success": off["success"],
                "rl_on_success": on["success"],
                "success_delta": int(on["success"]) - int(off["success"]),
                "rl_off_collision": off["collision"],
                "rl_on_collision": on["collision"],
                "collision_delta": int(on["collision"])
                - int(off["collision"]),
                "rl_off_final_goal_distance_m": off["final_goal_distance_m"],
                "rl_on_final_goal_distance_m": on["final_goal_distance_m"],
                "final_goal_distance_delta_m": (
                    on["final_goal_distance_m"]
                    - off["final_goal_distance_m"]
                ),
                "rl_off_completion": off["completion"],
                "rl_on_completion": on["completion"],
                "completion_delta": on["completion"] - off["completion"],
                "rl_off_minimum_clearance_m": off["minimum_clearance_m"],
                "rl_on_minimum_clearance_m": on["minimum_clearance_m"],
                "minimum_clearance_delta_m": (
                    on["minimum_clearance_m"] - off["minimum_clearance_m"]
                ),
                "rl_off_planner_p95_ms": off["planner_p95_compute_ms"],
                "rl_on_planner_p95_ms": on["planner_p95_compute_ms"],
                "planner_p95_delta_ms": (
                    on["planner_p95_compute_ms"]
                    - off["planner_p95_compute_ms"]
                ),
            }
        )
    return paired, unpaired


def summarize_pairs(records):
    """Return raw values and medians without treating blocks as replicates."""

    if not records:
        return {"pair_count": 0}
    delta_names = (
        "success_delta",
        "collision_delta",
        "final_goal_distance_delta_m",
        "completion_delta",
        "minimum_clearance_delta_m",
        "planner_p95_delta_ms",
    )
    result = {
        "pair_count": len(records),
        "obstacle_seed_count": len({row["episode_seed"] for row in records}),
        "rl_off_success_count": sum(row["rl_off_success"] for row in records),
        "rl_on_success_count": sum(row["rl_on_success"] for row in records),
        "rl_off_collision_count": sum(
            row["rl_off_collision"] for row in records
        ),
        "rl_on_collision_count": sum(
            row["rl_on_collision"] for row in records
        ),
    }
    for name in delta_names:
        values = [float(row[name]) for row in records]
        result[name] = {
            "raw": values,
            "median": float(np.median(values)),
        }
    return result


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _collision_forensics(record):
    rows = _read_csv(ROOT / record["run_dir"] / "trajectory.csv")
    first_hard = next(
        (
            row
            for row in rows
            if _flag(row.get("probabilistic_obstacle_hard_violation"))
        ),
        None,
    )
    terminal_hard_index = len(rows)
    while terminal_hard_index > 0 and _flag(
        rows[terminal_hard_index - 1].get(
            "probabilistic_obstacle_hard_violation"
        )
    ):
        terminal_hard_index -= 1
    terminal_hard = (
        rows[terminal_hard_index] if terminal_hard_index < len(rows) else None
    )
    zero_suffix = 0
    for row in reversed(rows):
        if abs(_number(row.get("executed_v"))) <= 1e-12 and abs(
            _number(row.get("executed_omega"))
        ) <= 1e-12:
            zero_suffix += 1
        else:
            break
    fields = (
        "time",
        "x",
        "y",
        "theta",
        "executed_v",
        "executed_omega",
        "safety_reason",
        "temporal_scan_ttc_s",
        "probabilistic_obstacle_maximum_step_probability",
        "probabilistic_obstacle_hard_violation",
        "probabilistic_obstacle_active_fallback_used",
        "probabilistic_obstacle_fail_closed",
        "residual_safety_shield_selected_source",
        "reliability_authority",
        "reliability_dynamics_confidence",
    )
    return {
        "experimental_key": record["experimental_key"],
        "steps": record["steps"],
        "minimum_clearance_m": record["minimum_clearance_m"],
        "final_goal_distance_m": record["final_goal_distance_m"],
        "first_hard_violation_time_s": (
            _number(first_hard.get("time")) if first_hard else None
        ),
        "terminal_hard_violation_onset_time_s": (
            _number(terminal_hard.get("time")) if terminal_hard else None
        ),
        "zero_command_suffix_steps": zero_suffix,
        "last_ten_control_records": [
            {name: row.get(name) for name in fields} for row in rows[-10:]
        ],
    }


def _gate(name, passed, value, threshold=None, detail=None):
    result = {"pass": bool(passed), "value": value}
    if threshold is not None:
        result["threshold"] = threshold
    if detail is not None:
        result["detail"] = detail
    return name, result


def run(protocol_path=DEFAULT_PROTOCOL, output_dir=None):
    protocol_path = Path(protocol_path).resolve()
    protocol = _yaml(protocol_path)
    protocol_audit = validate_protocol(protocol)
    stage3_protocol = _yaml(_resolve(protocol["stage3_protocol"]))
    base_config = load_yaml(_resolve(protocol["base_config"]))
    schedule = build_schedule(protocol, stage3_protocol)
    artifact_root = Path(output_dir or _resolve(protocol["output_dir"])).resolve()
    progress = _json(artifact_root / "progress.json")
    manifest = _json(artifact_root / "run_manifest.json")

    records = []
    missing = []
    partial = []
    for job in schedule:
        run_dir = _run_dir(artifact_root, job)
        present = [name for name in REQUIRED_RUN_FILES if (run_dir / name).is_file()]
        if len(present) != len(REQUIRED_RUN_FILES):
            missing.append(str(job["experimental_key"]))
            if run_dir.exists():
                partial.append(
                    {
                        "experimental_key": str(job["experimental_key"]),
                        "present_files": present,
                    }
                )
            continue
        records.append(
            _episode_record(
                job, run_dir, base_config, protocol, stage3_protocol
            )
        )
    expected_metric_paths = {
        (_run_dir(artifact_root, job) / "metrics.json").resolve()
        for job in schedule
    }
    observed_metric_paths = {
        path.resolve() for path in (artifact_root / "runs").rglob("metrics.json")
    }
    unexpected_metrics = sorted(
        str(path.relative_to(ROOT))
        for path in observed_metric_paths - expected_metric_paths
    )
    records.sort(key=lambda row: row["run_order"])
    collisions = [row for row in records if row["collision"]]
    rl_on = [row for row in records if row["rl_hss_enabled"]]
    rl_off = [row for row in records if not row["rl_hss_enabled"]]
    residual = [row for row in records if row["condition"] == "icode_residual"]
    nominal = [row for row in records if row["condition"] == "nominal"]
    completed_orders = [row["run_order"] for row in records]
    expected_prefix = list(range(len(records)))
    collision_key = collisions[0]["experimental_key"] if collisions else None
    progress_collision_key = progress.get("early_stop", {}).get(
        "experimental_key"
    )

    gate_pairs = [
        _gate(
            "frozen_hash_bindings_revalidated",
            True,
            {
                "implementation_file_count": len(
                    protocol_audit["implementation_contract"]
                ),
                "protocol_revision": protocol_audit["protocol_revision"],
            },
            {
                "implementation_file_count": len(
                    protocol["implementation_contract"]["files"]
                ),
                "protocol_revision": int(protocol["protocol_revision"]),
            },
        ),
        _gate(
            "maximum_collision_count",
            len(collisions) <= int(protocol["gates"]["maximum_collision_count"]),
            len(collisions),
            int(protocol["gates"]["maximum_collision_count"]),
        ),
        _gate(
            "matrix_completion",
            len(records) == len(schedule),
            len(records),
            len(schedule),
            "Two RL/HSS-on cells remain intentionally unrun after the collision stop."
            if len(records) != len(schedule)
            else None,
        ),
        _gate(
            "preregistered_first_collision_stop_enforced",
            bool(collisions)
            and len(collisions) == 1
            and completed_orders == expected_prefix
            and progress.get("status") == "stopped_by_preregistered_rule"
            and progress_collision_key == collision_key,
            progress.get("status"),
            "stopped_by_preregistered_rule",
        ),
        _gate(
            "artifact_integrity",
            not partial
            and not unexpected_metrics
            and len(records) == int(progress["completed_jobs"])
            and len(missing) == int(progress["remaining_jobs"]),
            {
                "complete": len(records),
                "missing": len(missing),
                "partial": len(partial),
                "unexpected": len(unexpected_metrics),
            },
            {"complete": 22, "missing": 2, "partial": 0, "unexpected": 0},
        ),
        _gate(
            "all_resolved_configs_match_frozen_generator",
            all(
                row["resolved_config_matches_frozen_generator"]
                and row["runtime_contract_valid"]
                for row in records
            ),
            sum(
                row["resolved_config_matches_frozen_generator"]
                and row["runtime_contract_valid"]
                for row in records
            ),
            len(records),
        ),
        _gate(
            "rl_off_exact_stage3_configuration",
            all(
                row["resolved_config_matches_frozen_generator"]
                and row["runtime_contract_valid"]
                for row in rl_off
            ),
            sum(
                row["resolved_config_matches_frozen_generator"]
                and row["runtime_contract_valid"]
                for row in rl_off
            ),
            len(rl_off),
        ),
        _gate(
            "hss_enabled_in_every_completed_rl_on_episode",
            bool(rl_on)
            and all(abs(row["hss_enabled_fraction"] - 1.0) <= 1e-12 for row in rl_on),
            min(row["hss_enabled_fraction"] for row in rl_on),
            1.0,
        ),
        _gate(
            "fixed_candidate_budget",
            bool(rl_on)
            and all(
                row["paper_total_rollouts_minimum"] == 600.0
                and row["paper_total_rollouts_maximum"] == 600.0
                and row["paper_candidates_per_iteration_minimum"] == 300.0
                and row["paper_candidates_per_iteration_maximum"] == 300.0
                for row in rl_on
            ),
            {
                "total_rollouts_minimum": min(
                    row["paper_total_rollouts_minimum"] for row in rl_on
                ),
                "total_rollouts_maximum": max(
                    row["paper_total_rollouts_maximum"] for row in rl_on
                ),
                "candidates_per_iteration_minimum": min(
                    row["paper_candidates_per_iteration_minimum"] for row in rl_on
                ),
                "candidates_per_iteration_maximum": max(
                    row["paper_candidates_per_iteration_maximum"] for row in rl_on
                ),
            },
            {"total_rollouts": 600, "candidates_per_iteration": 300},
        ),
        _gate(
            "scan_guard_present",
            all(
                row["scan_guard_diagnostics_present"]
                and row["runtime_contract_valid"]
                for row in records
            ),
            sum(row["scan_guard_diagnostics_present"] for row in records),
            len(records),
        ),
        _gate(
            "residual_shield_present",
            bool(residual)
            and all(
                abs(row["residual_shield_enabled_fraction"] - 1.0) <= 1e-12
                for row in residual
            ),
            min(row["residual_shield_enabled_fraction"] for row in residual),
            1.0,
        ),
        _gate(
            "nominal_cells_not_masquerading_as_residual_shield",
            bool(nominal)
            and all(
                abs(row["residual_shield_enabled_fraction"]) <= 1e-12
                for row in nominal
            ),
            max(row["residual_shield_enabled_fraction"] for row in nominal),
            0.0,
        ),
        _gate(
            "paper_probabilistic_risk_parity",
            bool(rl_on)
            and all(
                row["probability_diagnostics_present"]
                and abs(row["probabilistic_risk_enabled_fraction"] - 1.0)
                <= 1e-12
                for row in rl_on
            ),
            {
                "diagnostic_episode_count": sum(
                    row["probability_diagnostics_present"] for row in rl_on
                ),
                "minimum_enabled_fraction": min(
                    row["probabilistic_risk_enabled_fraction"] for row in rl_on
                ),
            },
            {"diagnostic_episode_count": len(rl_on), "enabled_fraction": 1.0},
        ),
        _gate(
            "sealed_seeds_closed",
            not bool(progress.get("sealed_seeds_opened", True))
            and not bool(manifest.get("sealed_seeds_opened", True))
            and set(row["episode_seed"] for row in records)
            <= set(protocol["design"]["obstacle_process_seeds"]),
            {
                "progress": progress.get("sealed_seeds_opened"),
                "manifest": manifest.get("sealed_seeds_opened"),
                "observed_seeds": sorted(
                    {row["episode_seed"] for row in records}
                ),
            },
            False,
        ),
    ]
    gates = dict(gate_pairs)
    mechanism_gate_names = (
        "artifact_integrity",
        "frozen_hash_bindings_revalidated",
        "all_resolved_configs_match_frozen_generator",
        "rl_off_exact_stage3_configuration",
        "hss_enabled_in_every_completed_rl_on_episode",
        "fixed_candidate_budget",
        "scan_guard_present",
        "residual_shield_present",
        "nominal_cells_not_masquerading_as_residual_shield",
        "paper_probabilistic_risk_parity",
        "sealed_seeds_closed",
    )
    gate_result = {
        "schema_version": 1,
        "stage": "dynamic_uncertainty_rl_hss_stage4_amendment1_development",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "independent_unit": "complete_episode",
        "model_block_unit": "residual_checkpoint_seed",
        "status": progress["status"],
        "completed_episode_count": len(records),
        "scheduled_episode_count": len(schedule),
        "success_count": sum(row["success"] for row in records),
        "collision_count": len(collisions),
        "mechanism_integrity_pass": all(
            gates[name]["pass"] for name in mechanism_gate_names
        ),
        "overall_pass": all(gate["pass"] for gate in gates.values()),
        "decision": "fail_safety_gate_stopped_by_preregistered_rule",
        "gates": gates,
        "collision_forensics": [
            _collision_forensics(row) for row in collisions
        ],
        "missing_experimental_keys": missing,
        "partial_runs": partial,
        "unexpected_metrics": unexpected_metrics,
        "sealed_seeds_opened": False,
    }

    paired, unpaired = _paired_records(records)
    groups = {"all_complete_pairs": summarize_pairs(paired)}
    for dynamics in sorted({row["dynamics"] for row in paired}):
        groups[dynamics] = summarize_pairs(
            [row for row in paired if row["dynamics"] == dynamics]
        )
    paired_result = {
        "schema_version": 1,
        "analysis": "episode_level_common_random_numbers_pairing",
        "independent_unit": "complete_episode",
        "model_blocks_are_environment_replicates": False,
        "inferential_tests_run": False,
        "reason_no_inferential_test": (
            "Only three development obstacle seeds were registered and the matrix "
            "stopped early; raw paired values and medians take precedence."
        ),
        "complete_pair_count": len(paired),
        "unpaired_cell_count": len(unpaired),
        "unpaired_cells": unpaired,
        "groups": groups,
        "rl_on_mechanism_summary": {
            "episode_count": len(rl_on),
            "reliability_authority_mean": _distribution(
                [row["reliability_authority_mean"] for row in rl_on]
            ),
            "reliability_dynamics_confidence_mean": _distribution(
                [row["reliability_dynamics_confidence_mean"] for row in rl_on]
            ),
            "reliability_proposal_fallback_fraction_mean": _distribution(
                [
                    row["reliability_proposal_fallback_fraction_mean"]
                    for row in rl_on
                ]
            ),
            "paper_actor_baseline_first_action_l2_delta_mean": _distribution(
                [
                    row["paper_actor_baseline_first_action_l2_delta_mean"]
                    for row in rl_on
                ]
            ),
            "paper_guided_minus_gaussian_cost_min_mean": _distribution(
                [
                    row["paper_guided_minus_gaussian_cost_min_mean"]
                    for row in rl_on
                ]
            ),
            "guided_minimum_cost_worse_than_gaussian_episode_count": sum(
                row["paper_guided_minus_gaussian_cost_min_mean"] > 0.0
                for row in rl_on
            ),
            "planner_p95_compute_ms": _distribution(
                [row["planner_p95_compute_ms"] for row in rl_on]
            ),
        },
        "records": paired,
        "sealed_seeds_opened": False,
    }

    episode_rows = []
    for row in records:
        episode_rows.append(
            {
                key: int(value) if isinstance(value, bool) else value
                for key, value in row.items()
                if key
                not in {
                    "runtime_contract_error",
                }
            }
        )
    _write_csv(artifact_root / "episode_summary.csv", episode_rows)
    _write_json(artifact_root / "paired_analysis.json", paired_result)
    _write_json(artifact_root / "gate.json", gate_result)
    failed_root = _resolve(protocol["amendment"]["failed_launch"]["output_dir"])
    episode_artifact_paths = sorted(
        path
        for path in (artifact_root / "runs").rglob("*")
        if path.is_file()
    )
    original_failure_paths = sorted(
        path for path in failed_root.rglob("*") if path.is_file()
    )
    top_level_paths = [
        protocol_path,
        _resolve(protocol["preregistration"]),
        _resolve(protocol["amendment"]["failed_launch"]["protocol"]),
        ROOT / "stage4_rl_hss.stdout.log",
        ROOT / "stage4_rl_hss.stderr.log",
        ROOT / "stage4_rl_hss_amendment1.stdout.log",
        ROOT / "stage4_rl_hss_amendment1.stderr.log",
        artifact_root / "run_manifest.json",
        artifact_root / "preflight_audit.json",
        artifact_root / "protocol_resolved.yaml",
        artifact_root / "schedule.json",
        artifact_root / "progress.json",
        artifact_root / "episode_summary.csv",
        artifact_root / "paired_analysis.json",
        artifact_root / "gate.json",
        Path(__file__).resolve(),
        RESULT_REPORT,
    ]
    result_manifest = {
        "schema_version": 1,
        "stage": "dynamic_uncertainty_rl_hss_stage4_amendment1_development",
        "status": progress["status"],
        "top_level_bindings": [_file_binding(path) for path in top_level_paths],
        "episode_artifact_bindings": [
            _file_binding(path) for path in episode_artifact_paths
        ],
        "original_failure_artifact_bindings": [
            _file_binding(path) for path in original_failure_paths
        ],
        "episode_artifact_file_count": len(episode_artifact_paths),
        "original_failure_artifact_file_count": len(original_failure_paths),
        "sealed_seeds_opened": False,
    }
    _write_json(artifact_root / "result_manifest.json", result_manifest)
    return {
        "gate": gate_result,
        "paired_analysis": paired_result,
        "episode_summary": episode_rows,
        "result_manifest": result_manifest,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = run(args.protocol, args.output_dir)
    print(
        json.dumps(
            {
                "completed_episode_count": result["gate"][
                    "completed_episode_count"
                ],
                "collision_count": result["gate"]["collision_count"],
                "mechanism_integrity_pass": result["gate"][
                    "mechanism_integrity_pass"
                ],
                "overall_pass": result["gate"]["overall_pass"],
                "complete_pair_count": result["paired_analysis"][
                    "complete_pair_count"
                ],
                "decision": result["gate"]["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
