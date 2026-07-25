"""Audit and summarize the completed expanded dynamic-Actor development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_actor_v5a6_samecycle_expanded_development.yaml"
)
ARTIFACT = (
    ROOT
    / "research_artifacts"
    / "dynamic_actor_v5a6_u000250_samecycle_expanded_development"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _aggregate_rows(rows):
    source = [row["source"] for row in rows]
    candidate = [row["candidate"] for row in rows]
    step_deltas = [row["step_delta"] for row in rows]
    distance_deltas = [row["distance_delta_m"] for row in rows]
    return {
        "pairs": len(rows),
        "source_successes": sum(bool(x["success"]) for x in source),
        "candidate_successes": sum(bool(x["success"]) for x in candidate),
        "source_collisions": sum(bool(x["collision"]) for x in source),
        "candidate_collisions": sum(bool(x["collision"]) for x in candidate),
        "source_total_steps": sum(int(x["steps"]) for x in source),
        "candidate_total_steps": sum(int(x["steps"]) for x in candidate),
        "source_mean_final_distance_m": statistics.fmean(
            float(x["final_goal_distance_m"]) for x in source
        ),
        "candidate_mean_final_distance_m": statistics.fmean(
            float(x["final_goal_distance_m"]) for x in candidate
        ),
        "median_step_delta_candidate_minus_source": statistics.median(step_deltas),
        "mean_step_delta_candidate_minus_source": statistics.fmean(step_deltas),
        "step_better_pairs": sum(delta < 0 for delta in step_deltas),
        "step_equal_pairs": sum(delta == 0 for delta in step_deltas),
        "step_worse_pairs": sum(delta > 0 for delta in step_deltas),
        "median_distance_delta_m_candidate_minus_source": statistics.median(
            distance_deltas
        ),
        "mean_distance_delta_m_candidate_minus_source": statistics.fmean(
            distance_deltas
        ),
        "distance_better_pairs": sum(delta < 0 for delta in distance_deltas),
        "distance_equal_pairs": sum(delta == 0 for delta in distance_deltas),
        "distance_worse_pairs": sum(delta > 0 for delta in distance_deltas),
        "source_planner_p95_max_ms": max(
            float(x["planner_p95_ms"]) for x in source
        ),
        "candidate_planner_p95_max_ms": max(
            float(x["planner_p95_ms"]) for x in candidate
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--artifact")
    args = parser.parse_args(argv)
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    artifact = (
        Path(args.artifact).resolve()
        if args.artifact
        else _resolve_repo_path(protocol["output_dir"])
    )
    snapshot = _load_json(artifact / "protocol_snapshot.json")
    summary = _load_json(artifact / "summary.json")
    gate = _load_json(artifact / "gate.json")
    schedule = [
        {"batch": batch["name"], **entry}
        for batch in protocol["batches"]
        for entry in batch["schedule"]
    ]
    issues = []
    rows = []
    for entry in schedule:
        seed = int(entry["seed"])
        pair = artifact / f"{entry['batch']}_seed{seed}"
        result_path = pair / "result.json"
        if not result_path.is_file():
            issues.append(f"missing result for {seed}")
            continue
        result = _load_json(result_path)
        if result.get("episode_seed") != seed:
            issues.append(f"seed mismatch for {seed}")
        if result.get("arm_order") != entry["arm_order"]:
            issues.append(f"arm order mismatch for {seed}")
        if result.get("sealed_seeds_opened") is not False:
            issues.append(f"sealed boundary mismatch for {seed}")
        if result.get("proposal_gate_mode") != "same_cycle_filter":
            issues.append(f"gate mode mismatch for {seed}")
        if result.get("proposal_only") is not True:
            issues.append(f"proposal-only mismatch for {seed}")
        for key, expected in protocol.get("probe_settings", {}).items():
            if result.get(key) != expected:
                issues.append(f"probe setting mismatch for {seed}: {key}")
        for arm, expected_checkpoint in (
            ("source_actor", protocol["bindings"]["source_checkpoint"]["path"]),
            ("dynamic_actor", protocol["bindings"]["candidate_checkpoint"]["path"]),
        ):
            run = pair / "runs" / arm
            required = (
                "config_resolved.yaml",
                "metrics.json",
                "provenance.json",
                "trajectory.csv",
            )
            for name in required:
                if not (run / name).is_file():
                    issues.append(f"missing {name} for {seed}/{arm}")
            config = yaml.safe_load(
                (run / "config_resolved.yaml").read_text(encoding="utf-8")
            )
            metrics = _load_json(run / "metrics.json")
            paper = config["planner"]["paper_rl_driven"]
            checkpoint = Path(config["rl"]["checkpoint"])
            if str(checkpoint).replace("\\", "/").endswith(expected_checkpoint) is False:
                issues.append(f"checkpoint mismatch for {seed}/{arm}")
            if config["experiment"]["seed"] != seed:
                issues.append(f"resolved seed mismatch for {seed}/{arm}")
            if paper.get("same_cycle_guided_cost_filter") is not True:
                issues.append(f"same-cycle filter disabled for {seed}/{arm}")
            if float(paper.get("same_cycle_guided_relative_margin")) != 0.0:
                issues.append(f"same-cycle margin mismatch for {seed}/{arm}")
            if float(paper.get("terminal_value_weight")) != 0.0:
                issues.append(f"terminal Critic enabled for {seed}/{arm}")
            if int(config["planner"]["num_samples"]) * int(paper["iterations"]) != 600:
                issues.append(f"rollout budget mismatch for {seed}/{arm}")
            if float(metrics.get("paper_total_rollouts_mean", 0.0)) != 600.0:
                issues.append(f"realized rollout budget mismatch for {seed}/{arm}")
            arm_summary = result["results"][arm]
            if float(arm_summary["same_cycle_filter_enabled_fraction"]) != 1.0:
                issues.append(f"filter not enabled throughout {seed}/{arm}")
            if int(arm_summary["same_cycle_filter_iterations_total"]) <= 0:
                issues.append(f"filter not exercised for {seed}/{arm}")
        source = result["results"]["source_actor"]
        candidate = result["results"]["dynamic_actor"]
        rows.append({
            "batch": entry["batch"],
            "seed": seed,
            "arm_order": entry["arm_order"],
            "source": source,
            "candidate": candidate,
            "success_delta": int(candidate["success"]) - int(source["success"]),
            "collision_delta": int(candidate["collision"]) - int(source["collision"]),
            "step_delta": int(candidate["steps"]) - int(source["steps"]),
            "distance_delta_m": (
                float(candidate["final_goal_distance_m"])
                - float(source["final_goal_distance_m"])
            ),
        })

    current_bindings = {}
    for name, binding in protocol["bindings"].items():
        path = ROOT / binding["path"]
        actual = _sha256(path)
        current_bindings[name] = actual
        if actual != binding["sha256"]:
            issues.append(f"binding hash mismatch: {name}")
    protocol_sha256 = _sha256(protocol_path)
    if protocol_sha256 != snapshot["protocol_sha256"]:
        issues.append("protocol snapshot hash mismatch")
    if protocol_sha256 != summary["protocol_sha256"]:
        issues.append("summary protocol hash mismatch")
    if gate != summary["gate"]:
        issues.append("gate and summary disagreement")
    if len(rows) != 24:
        issues.append(f"expected 24 rows, found {len(rows)}")

    raw_files = sorted(
        path for path in artifact.rglob("*")
        if path.is_file()
        and path.name not in {"integrity_audit.json", "paired_analysis.json"}
    )
    manifest = {
        str(path.relative_to(ROOT)): _sha256(path) for path in raw_files
    }
    audit = {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "complete_pairs": len(rows),
        "scheduled_pairs": len(schedule),
        "raw_file_count": len(raw_files),
        "raw_file_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "raw_file_manifest": manifest,
        "protocol_sha256": protocol_sha256,
        "binding_hashes": current_bindings,
        "sealed_seeds_opened": False,
    }
    aggregate = _aggregate_rows(rows)
    source_steps = aggregate["source_total_steps"]
    source_distance = aggregate["source_mean_final_distance_m"]
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "gate_passed": bool(gate["passed"]),
        "pooled": aggregate,
        "relative_changes": {
            "total_steps_fraction": (
                aggregate["candidate_total_steps"] - source_steps
            ) / source_steps,
            "mean_final_distance_fraction": (
                aggregate["candidate_mean_final_distance_m"] - source_distance
            ) / source_distance,
        },
        "new_candidate_collision_seeds": [
            row["seed"] for row in rows
            if row["candidate"]["collision"] and not row["source"]["collision"]
        ],
        "shared_collision_seeds": [
            row["seed"] for row in rows
            if row["candidate"]["collision"] and row["source"]["collision"]
        ],
        "lost_source_success_seeds": [
            row["seed"] for row in rows
            if row["source"]["success"] and not row["candidate"]["success"]
        ],
        "rescued_source_failure_seeds": [
            row["seed"] for row in rows
            if not row["source"]["success"] and row["candidate"]["success"]
        ],
        "by_arm_order": {
            order: _aggregate_rows([row for row in rows if row["arm_order"] == order])
            for order in ("source_first", "candidate_first")
        },
        "rows": rows,
    }
    _write_json(artifact / "integrity_audit.json", audit)
    _write_json(artifact / "paired_analysis.json", analysis)
    print(json.dumps({
        "integrity_status": audit["status"],
        "issues": len(issues),
        "gate_passed": analysis["gate_passed"],
        "pooled": aggregate,
        "new_collision_seeds": analysis["new_candidate_collision_seeds"],
        "lost_success_seeds": analysis["lost_source_success_seeds"],
        "rescued_failure_seeds": analysis["rescued_source_failure_seeds"],
    }, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
