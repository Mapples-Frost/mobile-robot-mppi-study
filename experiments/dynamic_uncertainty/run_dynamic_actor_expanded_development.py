"""Run the frozen 24-pair expanded dynamic-Actor development protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_actor_v5a6_samecycle_expanded_development.yaml"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_protocol(path):
    protocol_path = Path(path).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("expanded development protocol is not frozen")
    if protocol.get("sealed_seeds_opened") is not False:
        raise ValueError("sealed-seed boundary is not closed")
    schedule = [
        {"batch": batch["name"], **entry}
        for batch in protocol["batches"]
        for entry in batch["schedule"]
    ]
    seeds = [int(entry["seed"]) for entry in schedule]
    if len(schedule) != 24 or len(set(seeds)) != 24:
        raise ValueError("protocol must contain 24 unique paired seeds")
    if not all(730100001 <= seed <= 730100300 for seed in seeds):
        raise ValueError("expanded protocol seed escaped development-only range")
    for batch in protocol["batches"]:
        orders = [entry["arm_order"] for entry in batch["schedule"]]
        if sorted(orders) != ["candidate_first"] * 4 + ["source_first"] * 4:
            raise ValueError("each batch must balance four runs per arm order")
    return protocol_path, protocol, schedule


def _verify_bindings(protocol):
    verified = {}
    for name, binding in protocol["bindings"].items():
        path = _resolve_repo_path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        expected = str(binding["sha256"]).lower()
        if actual != expected:
            raise ValueError(f"binding hash mismatch for {name}")
        verified[name] = {"path": str(path.relative_to(ROOT)), "sha256": actual}
    return verified


def _arm_metrics(record, arm):
    return record["results"][arm]


def _probe_arguments_and_expectations(protocol):
    settings = dict(protocol.get("probe_settings", {}))
    if not settings:
        return [], {}
    allowed = {
        "completion_handover_full_fallback_distance",
        "completion_handover_full_rl_distance",
        "vetted_reactive_escape",
        "dynamic_escape_min_probability_mass_relative_improvement",
        "probabilistic_emergency_trigger_ttc_s",
        "probabilistic_emergency_intent_hold_steps",
        "probabilistic_emergency_pareto_forward_commit",
        "probabilistic_emergency_forward_risk_ceiling",
        "probabilistic_emergency_forward_mass_ceiling",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            "unknown expanded-development probe settings: "
            + ", ".join(sorted(unknown))
        )
    arguments = []
    expectations = {}
    scalar_options = {
        "completion_handover_full_fallback_distance": (
            "--completion-handover-full-fallback-distance"
        ),
        "completion_handover_full_rl_distance": (
            "--completion-handover-full-rl-distance"
        ),
        "dynamic_escape_min_probability_mass_relative_improvement": (
            "--dynamic-escape-min-probability-mass-relative-improvement"
        ),
        "probabilistic_emergency_trigger_ttc_s": (
            "--probabilistic-emergency-trigger-ttc-s"
        ),
        "probabilistic_emergency_intent_hold_steps": (
            "--probabilistic-emergency-intent-hold-steps"
        ),
        "probabilistic_emergency_forward_risk_ceiling": (
            "--probabilistic-emergency-forward-risk-ceiling"
        ),
        "probabilistic_emergency_forward_mass_ceiling": (
            "--probabilistic-emergency-forward-mass-ceiling"
        ),
    }
    for key, option in scalar_options.items():
        if key in settings:
            arguments.extend((option, str(settings[key])))
            expectations[key] = settings[key]
    flag_options = {
        "vetted_reactive_escape": "--vetted-reactive-escape",
        "probabilistic_emergency_pareto_forward_commit": (
            "--probabilistic-emergency-pareto-forward-commit"
        ),
    }
    for key, option in flag_options.items():
        enabled = bool(settings.get(key, False))
        if enabled:
            arguments.append(option)
        expectations[key] = enabled
    return arguments, expectations


def _aggregate(records, failure_step_penalty=0):
    source = [_arm_metrics(record, "source_actor") for record in records]
    candidate = [_arm_metrics(record, "dynamic_actor") for record in records]
    failure_step_penalty = int(failure_step_penalty)
    if failure_step_penalty < 0:
        raise ValueError("failure step penalty must be non-negative")

    def efficiency_steps(row):
        steps = int(row["steps"])
        if failure_step_penalty and not bool(row["success"]):
            return max(steps, failure_step_penalty)
        return steps

    source_distance = sum(x["final_goal_distance_m"] for x in source) / len(source)
    candidate_distance = (
        sum(x["final_goal_distance_m"] for x in candidate) / len(candidate)
    )
    return {
        "pairs": len(records),
        "source_successes": sum(bool(x["success"]) for x in source),
        "candidate_successes": sum(bool(x["success"]) for x in candidate),
        "source_collisions": sum(bool(x["collision"]) for x in source),
        "candidate_collisions": sum(bool(x["collision"]) for x in candidate),
        "new_candidate_collisions": sum(
            bool(candidate_row["collision"]) and not bool(source_row["collision"])
            for source_row, candidate_row in zip(source, candidate)
        ),
        "prevented_source_collisions": sum(
            bool(source_row["collision"]) and not bool(candidate_row["collision"])
            for source_row, candidate_row in zip(source, candidate)
        ),
        "lost_source_successes": sum(
            bool(source_row["success"]) and not bool(candidate_row["success"])
            for source_row, candidate_row in zip(source, candidate)
        ),
        "rescued_source_failures": sum(
            not bool(source_row["success"]) and bool(candidate_row["success"])
            for source_row, candidate_row in zip(source, candidate)
        ),
        "source_total_steps": sum(int(x["steps"]) for x in source),
        "candidate_total_steps": sum(int(x["steps"]) for x in candidate),
        "source_efficiency_steps": sum(efficiency_steps(x) for x in source),
        "candidate_efficiency_steps": sum(
            efficiency_steps(x) for x in candidate
        ),
        "failure_step_penalty": failure_step_penalty,
        "source_mean_final_distance_m": source_distance,
        "candidate_mean_final_distance_m": candidate_distance,
        "source_filter_enabled_min": min(
            float(x["same_cycle_filter_enabled_fraction"]) for x in source
        ),
        "candidate_filter_enabled_min": min(
            float(x["same_cycle_filter_enabled_fraction"]) for x in candidate
        ),
        "source_filter_iterations_total": sum(
            int(x["same_cycle_filter_iterations_total"]) for x in source
        ),
        "candidate_filter_iterations_total": sum(
            int(x["same_cycle_filter_iterations_total"]) for x in candidate
        ),
        "source_filtered_candidates_total": sum(
            int(x["same_cycle_filtered_candidates_total"]) for x in source
        ),
        "candidate_filtered_candidates_total": sum(
            int(x["same_cycle_filtered_candidates_total"]) for x in candidate
        ),
    }


def _nonregressive(summary, use_outcome_aware_efficiency=False):
    source_steps = (
        summary["source_efficiency_steps"]
        if use_outcome_aware_efficiency
        else summary["source_total_steps"]
    )
    candidate_steps = (
        summary["candidate_efficiency_steps"]
        if use_outcome_aware_efficiency
        else summary["candidate_total_steps"]
    )
    return (
        summary["candidate_successes"] >= summary["source_successes"]
        and summary["candidate_collisions"] <= summary["source_collisions"]
        and candidate_steps <= source_steps
        and summary["candidate_mean_final_distance_m"]
        <= summary["source_mean_final_distance_m"]
    )


def _evaluate_gate(
    pooled,
    batches,
    protocol_integrity,
    use_outcome_aware_efficiency=False,
    efficiency_relative_noninferiority_margin=0.0,
):
    source_steps = (
        pooled["source_efficiency_steps"]
        if use_outcome_aware_efficiency
        else pooled["source_total_steps"]
    )
    candidate_steps = (
        pooled["candidate_efficiency_steps"]
        if use_outcome_aware_efficiency
        else pooled["candidate_total_steps"]
    )
    efficiency_limit = source_steps * (
        1.0 + float(efficiency_relative_noninferiority_margin)
    )
    checks = {
        "complete_24_pairs": pooled["pairs"] == 24,
        "zero_new_candidate_collisions": pooled["new_candidate_collisions"] == 0,
        "zero_lost_source_successes": pooled["lost_source_successes"] == 0,
        "completion_noninferior": (
            pooled["candidate_successes"] >= pooled["source_successes"]
        ),
        "total_steps_noninferior": (
            candidate_steps <= efficiency_limit
        ),
        "mean_final_distance_noninferior": (
            pooled["candidate_mean_final_distance_m"]
            <= pooled["source_mean_final_distance_m"]
        ),
        "strict_pooled_improvement": (
            pooled["candidate_successes"] > pooled["source_successes"]
            or candidate_steps < source_steps
            or pooled["candidate_mean_final_distance_m"]
            < pooled["source_mean_final_distance_m"]
        ),
        "at_least_two_nonregressive_batches": (
            sum(bool(batch["nonregressive"]) for batch in batches.values()) >= 2
        ),
        "protocol_integrity": bool(protocol_integrity),
        "same_cycle_filter_enabled_all_arms": (
            pooled["source_filter_enabled_min"] >= 1.0
            and pooled["candidate_filter_enabled_min"] >= 1.0
        ),
        "same_cycle_filter_exercised": (
            pooled["source_filter_iterations_total"] > 0
            and pooled["candidate_filter_iterations_total"] > 0
            and pooled["source_filtered_candidates_total"] > 0
            and pooled["candidate_filtered_candidates_total"] > 0
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args(argv)
    protocol_path, protocol, schedule = _load_protocol(args.protocol)
    bindings = _verify_bindings(protocol)
    probe_arguments, probe_expectations = (
        _probe_arguments_and_expectations(protocol)
    )
    output = _resolve_repo_path(protocol["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = _sha256(protocol_path)
    _write_json(output / "protocol_snapshot.json", {
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "verified_bindings": bindings,
    })

    completed = []
    for index, entry in enumerate(schedule, start=1):
        seed = int(entry["seed"])
        pair_output = output / f"{entry['batch']}_seed{seed}"
        result_path = pair_output / "result.json"
        if not result_path.is_file():
            command = [
                sys.executable,
                str(ROOT / protocol["probe_script"]),
                "--seed", str(seed),
                "--source", protocol["bindings"]["source_checkpoint"]["path"],
                "--candidate", protocol["bindings"]["candidate_checkpoint"]["path"],
                "--output-dir", str(pair_output),
                "--proposal-gate-mode", "same_cycle_filter",
                "--proposal-only",
                "--arm-order", entry["arm_order"],
            ] + probe_arguments
            subprocess.run(command, cwd=ROOT, check=True)
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            int(record["episode_seed"]) != seed
            or record["arm_order"] != entry["arm_order"]
            or record["proposal_gate_mode"] != "same_cycle_filter"
            or record["proposal_only"] is not True
            or record["sealed_seeds_opened"] is not False
            or any(
                record.get(key) != value
                for key, value in probe_expectations.items()
            )
        ):
            raise ValueError(f"pair integrity failure for seed {seed}")
        completed.append({"batch": entry["batch"], "seed": seed, "result": record})
        _write_json(output / "progress.json", {
            "status": "running" if index < len(schedule) else "aggregating",
            "completed_pairs": index,
            "total_pairs": len(schedule),
            "current_batch": entry["batch"],
            "last_completed_seed": seed,
            "protocol_sha256": protocol_sha256,
        })

    records = [entry["result"] for entry in completed]
    gate_settings = dict(protocol.get("gates", {}))
    failure_step_penalty = int(
        gate_settings.get("failure_step_penalty", 0)
    )
    efficiency_margin = float(
        gate_settings.get(
            "efficiency_relative_noninferiority_margin",
            0.0,
        )
    )
    if failure_step_penalty < 0 or not 0.0 <= efficiency_margin < 1.0:
        raise ValueError("invalid outcome-aware efficiency gate settings")
    use_outcome_aware_efficiency = failure_step_penalty > 0
    batch_summaries = {}
    for batch in (item["name"] for item in protocol["batches"]):
        batch_records = [
            entry["result"] for entry in completed if entry["batch"] == batch
        ]
        summary = _aggregate(
            batch_records,
            failure_step_penalty=failure_step_penalty,
        )
        summary["nonregressive"] = _nonregressive(
            summary,
            use_outcome_aware_efficiency=use_outcome_aware_efficiency,
        )
        batch_summaries[batch] = summary
    pooled = _aggregate(
        records,
        failure_step_penalty=failure_step_penalty,
    )
    protocol_integrity = all(
        record["source_checkpoint"]
        == protocol["bindings"]["source_checkpoint"]["path"]
        and record["candidate_checkpoint"]
        == protocol["bindings"]["candidate_checkpoint"]["path"]
        for record in records
    )
    gate = _evaluate_gate(
        pooled,
        batch_summaries,
        protocol_integrity=protocol_integrity,
        use_outcome_aware_efficiency=use_outcome_aware_efficiency,
        efficiency_relative_noninferiority_margin=efficiency_margin,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "expanded_paired_dynamic_actor_development",
        "sealed_seeds_opened": False,
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "verified_bindings": bindings,
        "efficiency_gate": {
            "outcome_aware": use_outcome_aware_efficiency,
            "failure_step_penalty": failure_step_penalty,
            "relative_noninferiority_margin": efficiency_margin,
        },
        "batch_results": batch_summaries,
        "pooled_results": pooled,
        "gate": gate,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "gate.json", gate)
    _write_json(output / "progress.json", {
        "status": "complete",
        "completed_pairs": len(schedule),
        "total_pairs": len(schedule),
        "gate_passed": gate["passed"],
        "protocol_sha256": protocol_sha256,
    })
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
