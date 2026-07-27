"""Analyze the frozen paired closed-loop maneuver Actor Gate D."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_complex_supervised_maneuver_actor_gate_d import (
    DEFAULT_PROTOCOL,
    _load_protocol,
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row, key, default=0.0):
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _truth(row, key):
    return _number(row, key, 0.0) > 0.5


def _metric(metrics, canonical_name, *legacy_names):
    for name in (canonical_name, *legacy_names):
        if name in metrics:
            return metrics[name]
    raise KeyError(
        "missing metric %s (accepted legacy aliases: %s)"
        % (canonical_name, ", ".join(legacy_names) or "none")
    )


def _normalized_pair_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config["planner"]["paper_rl_driven"].pop(
        "supervised_maneuver_actor", None
    )
    config["experiment"].pop("name", None)
    config.pop("gate_d_contract", None)
    return config


def _git_changed_files(first_sha, second_sha):
    result = subprocess.run(
        ["git", "diff", "--name-only", str(first_sha), str(second_sha)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        value.strip().replace("\\", "/")
        for value in result.stdout.splitlines()
        if value.strip()
    ]


def _non_artifact_changes(first_sha, second_sha):
    return [
        value for value in _git_changed_files(first_sha, second_sha)
        if not value.startswith("research_artifacts/")
    ]


def _pair_revision_audit(input_root, map_name, pair):
    control = pair["frozen_full"]
    treatment = pair["full_plus_bootstrap_actor"]
    control_sha = control["provenance"]["git_sha"]
    treatment_sha = treatment["provenance"]["git_sha"]
    if control_sha == treatment_sha:
        return {
            "status": "pass",
            "basis": "identical_git_sha",
            "control_git_sha": control_sha,
            "treatment_git_sha": treatment_sha,
            "non_artifact_changes": [],
        }

    changed = _non_artifact_changes(control_sha, treatment_sha)
    if not changed:
        return {
            "status": "pass",
            "basis": "artifact_only_commits_between_arms",
            "control_git_sha": control_sha,
            "treatment_git_sha": treatment_sha,
            "non_artifact_changes": [],
        }

    if str(map_name) != "chapter1":
        raise ValueError(
            "Gate D pair changed runtime code between arms: %s"
            % ", ".join(changed)
        )
    audit_path = (
        Path(input_root)
        / "equivalence_audit"
        / "equivalence_result.json"
    )
    audit = _read_json(audit_path)
    checks = audit.get("checks", {})
    if not (
        audit.get("status") == "pass"
        and audit.get("pre_patch_git_sha") == control_sha
        and all(bool(value) for value in checks.values())
    ):
        raise ValueError("Chapter 1 interface equivalence audit is invalid")
    post_patch_sha = audit["post_patch_git_sha"]
    post_patch_changes = _non_artifact_changes(
        post_patch_sha, treatment_sha
    )
    if post_patch_changes:
        raise ValueError(
            "Chapter 1 runtime changed after equivalence audit: %s"
            % ", ".join(post_patch_changes)
        )
    return {
        "status": "pass",
        "basis": "preoutcome_actor_off_equivalence_audit",
        "control_git_sha": control_sha,
        "treatment_git_sha": treatment_sha,
        "equivalence_post_patch_git_sha": post_patch_sha,
        "non_artifact_changes": changed,
        "post_audit_non_artifact_changes": [],
    }


def _first_time(rows, predicate):
    for row in rows:
        if predicate(row):
            return _number(row, "time")
    return None


def _persistent_conflict_time(rows):
    active = [
        (
            _truth(row, "probabilistic_obstacle_temporal_emergency_triggered")
            or _truth(
                row,
                "probabilistic_obstacle_emergency_critical_distance_triggered",
            )
        )
        for row in rows
    ]
    for index in range(max(0, len(active) - 1)):
        if active[index] and active[index + 1]:
            return _number(rows[index], "time")
    return None


def _cold_start_summary(rows, end_s):
    early = [row for row in rows if _number(row, "time") < float(end_s)]
    first_conflict = _persistent_conflict_time(rows)
    conflict_row = None
    if first_conflict is not None:
        conflict_row = min(
            rows, key=lambda row: abs(_number(row, "time") - first_conflict)
        )
    forecast_available = [
        _truth(row, "dynamic_obstacle_tracker_forecast_valid")
        for row in early
    ]
    return {
        "first_dynamic_observation_time_s": _first_time(
            rows,
            lambda row: _number(
                row, "dynamic_obstacle_tracker_observation_count"
            ) > 0,
        ),
        "first_imm_initialized_time_s": _first_time(
            rows,
            lambda row: _truth(
                row, "dynamic_obstacle_tracker_imm_initialized"
            ),
        ),
        "first_forecast_available_time_s": _first_time(
            rows,
            lambda row: _truth(
                row, "dynamic_obstacle_tracker_forecast_valid"
            ),
        ),
        "first_persistent_conflict_time_s": first_conflict,
        "history_frames_at_first_persistent_conflict": (
            None
            if conflict_row is None
            else int(_number(
                conflict_row, "dynamic_obstacle_tracker_history_length"
            ))
        ),
        "early_window_forecast_availability_fraction": (
            float(np.mean(forecast_available))
            if forecast_available else 0.0
        ),
        "early_window_risk_unavailable_rejection_count": int(sum(
            (
                not _truth(
                    row, "dynamic_obstacle_tracker_forecast_valid"
                )
                and _truth(
                    row,
                    "probabilistic_obstacle_temporal_emergency_triggered",
                )
            )
            for row in early
        )),
        "early_window_safety_forced_stop_count": int(sum(
            _truth(row, "safety_override")
            and abs(_number(row, "executed_v")) < 0.02
            and abs(_number(row, "executed_omega")) < 0.05
            for row in early
        )),
        "stratum": (
            "no_persistent_conflict"
            if first_conflict is None
            else "early_conflict_before_1_5s"
            if first_conflict < float(end_s)
            else "later_conflict_at_or_after_1_5s"
        ),
    }


def _episode_record(directory, map_name, seed, arm, end_s):
    metrics = _read_json(Path(directory) / "metrics.json")
    rows = _read_csv(Path(directory) / "trajectory.csv")
    rollout_counts = sorted({
        int(_number(row, "paper_total_rollouts")) for row in rows
    })
    active_budget_valid = all(
        (
            int(_number(row, "paper_total_rollouts")) == 600
            if str(row.get("optimizer", "")) == "paper_rl_driven"
            else int(_number(row, "paper_total_rollouts")) == 0
        )
        for row in rows
    )
    return {
        "map": str(map_name),
        "seed": int(seed),
        "arm": str(arm),
        "success": bool(metrics["success"]),
        "collision": bool(metrics["collision"]),
        "safe_success": bool(metrics["success"] and not metrics["collision"]),
        "termination_reason": str(metrics["termination_reason"]),
        "steps": int(metrics["steps"]),
        "final_goal_distance": float(metrics["final_goal_distance"]),
        "path_length": float(
            _metric(metrics, "trajectory_length", "path_length")
        ),
        "minimum_clearance": float(metrics["minimum_clearance"]),
        "safety_overrides": int(
            _metric(metrics, "safety_interventions", "safety_overrides")
        ),
        "paper_total_rollouts_mean": float(
            metrics["paper_total_rollouts_mean"]
        ),
        "per_cycle_rollout_counts": rollout_counts,
        "active_paper_optimizer_budget_valid": bool(active_budget_valid),
        "missing_forecast_deterministic_stop_steps": int(sum(
            str(row.get("optimizer", "")) == "standard"
            and int(_number(row, "paper_total_rollouts")) == 0
            for row in rows
        )),
        "supervised_proposal_count_total": int(
            metrics.get("supervised_proposal_count_total", 0)
        ),
        "supervised_risk_feasible_count_total": int(
            metrics.get("supervised_risk_feasible_count_total", 0)
        ),
        "supervised_elite_count_total": int(
            metrics.get("supervised_elite_count_total", 0)
        ),
        "supervised_selected_count_total": int(
            metrics.get("supervised_selected_count_total", 0)
        ),
        "supervised_added_rollout_count_total": int(
            metrics.get("supervised_added_rollout_count_total", 0)
        ),
        "provenance": metrics["provenance"],
        "cold_start": _cold_start_summary(rows, end_s),
    }


def gate_decision(protocol, pairs):
    treatment = "full_plus_bootstrap_actor"
    control = "frozen_full"
    collision_regressions = sum(
        pair[treatment]["collision"] and not pair[control]["collision"]
        for pair in pairs
    )
    success_losses = sum(
        pair[control]["safe_success"] and not pair[treatment]["safe_success"]
        for pair in pairs
    )
    success_gains = sum(
        pair[treatment]["safe_success"] and not pair[control]["safe_success"]
        for pair in pairs
    )
    goal_gains = [
        pair[control]["final_goal_distance"]
        - pair[treatment]["final_goal_distance"]
        for pair in pairs
    ]
    actor_maps = sum(
        pair[treatment]["supervised_proposal_count_total"] > 0
        for pair in pairs
    )
    risk_maps = sum(
        pair[treatment]["supervised_risk_feasible_count_total"] > 0
        for pair in pairs
    )
    elite_or_selected = sum(
        pair[treatment]["supervised_elite_count_total"]
        + pair[treatment]["supervised_selected_count_total"]
        for pair in pairs
    )
    fallback = protocol["gate"]["fallback_if_both_arms_safe_but_incomplete"]
    fallback_applicable = all(
        not pair[control]["safe_success"]
        and not pair[treatment]["safe_success"]
        and not pair[control]["collision"]
        and not pair[treatment]["collision"]
        for pair in pairs
    )
    fallback_pass = bool(
        fallback_applicable
        and sum(value > 0.0 for value in goal_gains)
        >= int(fallback[
            "minimum_maps_with_positive_terminal_goal_distance_gain"
        ])
        and float(sum(goal_gains))
        >= float(fallback["minimum_total_terminal_goal_distance_gain_m"])
    )
    task_improvement = bool(
        success_gains
        >= int(protocol["gate"]["minimum_actor_success_gains"])
        or fallback_pass
    )
    checks = {
        "maximum_actor_collision_regressions": (
            collision_regressions
            <= int(protocol["gate"]["maximum_actor_collision_regressions"])
        ),
        "maximum_actor_success_losses": (
            success_losses
            <= int(protocol["gate"]["maximum_actor_success_losses"])
        ),
        "task_improvement": task_improvement,
        "minimum_maps_with_nonzero_supervised_proposals": (
            actor_maps
            >= int(protocol["gate"][
                "minimum_maps_with_nonzero_supervised_proposals"
            ])
        ),
        "minimum_maps_with_risk_feasible_supervised_proposals": (
            risk_maps
            >= int(protocol["gate"][
                "minimum_maps_with_risk_feasible_supervised_proposals"
            ])
        ),
        "minimum_total_supervised_elite_or_selected_count": (
            elite_or_selected
            >= int(protocol["gate"][
                "minimum_total_supervised_elite_or_selected_count"
            ])
        ),
        "same_rollout_budget": all(
            pair[control]["active_paper_optimizer_budget_valid"]
            and pair[treatment]["active_paper_optimizer_budget_valid"]
            and set(pair[control]["per_cycle_rollout_counts"]).issubset(
                set(protocol["online_interface"][
                    "permitted_per_cycle_rollout_counts"
                ])
            )
            and set(pair[treatment]["per_cycle_rollout_counts"]).issubset(
                set(protocol["online_interface"][
                    "permitted_per_cycle_rollout_counts"
                ])
            )
            and pair[treatment]["supervised_added_rollout_count_total"] == 0
            for pair in pairs
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "collision_regressions": int(collision_regressions),
        "success_losses": int(success_losses),
        "success_gains": int(success_gains),
        "terminal_goal_distance_gains_m": goal_gains,
        "fallback_applicable": fallback_applicable,
        "fallback_pass": fallback_pass,
        "maps_with_supervised_proposals": int(actor_maps),
        "maps_with_risk_feasible_supervised_proposals": int(risk_maps),
        "total_supervised_elite_or_selected_count": int(elite_or_selected),
    }


def analyze(protocol_path, input_root, output):
    protocol = _load_protocol(protocol_path)
    input_root = Path(input_root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Gate D analysis output already contains evidence")
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    pairs = []
    pair_revision_audits = []
    end_s = float(
        protocol["prediction_cold_start_diagnostics"][
            "early_window_end_s"
        ]
    )
    for block in protocol["paired_blocks"]:
        pair = {}
        for arm in block["order"]:
            directory = (
                input_root
                / ("%s_seed%d" % (block["map"], int(block["seed"])))
                / arm
            )
            if not (directory / "metrics.json").is_file():
                raise FileNotFoundError("Gate D episode missing: %s" % directory)
            record = _episode_record(
                directory, block["map"], block["seed"], arm, end_s
            )
            episodes.append(record)
            pair[arm] = record
        if set(pair) != {"frozen_full", "full_plus_bootstrap_actor"}:
            raise ValueError("Gate D paired block is incomplete")
        control_directory = (
            input_root
            / ("%s_seed%d" % (block["map"], int(block["seed"])))
            / "frozen_full"
        )
        treatment_directory = (
            input_root
            / ("%s_seed%d" % (block["map"], int(block["seed"])))
            / "full_plus_bootstrap_actor"
        )
        if _normalized_pair_config(
            control_directory / "config_resolved.yaml"
        ) != _normalized_pair_config(
            treatment_directory / "config_resolved.yaml"
        ):
            raise ValueError(
                "Gate D pair differs outside the proposal-only interface"
            )
        pair_revision_audits.append(
            {
                "map": block["map"],
                "seed": int(block["seed"]),
                **_pair_revision_audit(input_root, block["map"], pair),
            }
        )
        pairs.append(pair)
    decision = gate_decision(protocol, pairs)
    result = {
        **decision,
        "scope": protocol["scope"],
        "episode_count": len(episodes),
        "pair_count": len(pairs),
        "pair_revision_audits": pair_revision_audits,
        "episodes": episodes,
        "cold_start_is_diagnostic_only": True,
        "statistical_claim_authorized": False,
        "formal_experiment_authorized": False,
    }
    result_path = output / "gate_d_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_copy = output / "protocol_resolved.yaml"
    protocol_copy.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (result_path, protocol_copy)
        },
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.protocol, args.input, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
