"""Run the sealed, controller-independent held-out qualification for v5.

This is an engineering qualification, not formal effect estimation.  The seed
registry is selected and sealed using only the frozen ghost-path conflict
certificate before either controller is executed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_paper_v4 as v4,
)
from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_v5_recovery_development as development,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner

DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "single_dynamic_obstacle_v5_held_out_qualification.yaml"
)


def _protocol(protocol_path=DEFAULT_PROTOCOL):
    path = development._repo_path(protocol_path)
    protocol = v4._load_yaml(path)
    if protocol.get("status") != "frozen_before_held_out_qualification":
        raise ValueError("v5 held-out qualification protocol is not frozen")
    if protocol.get("scope") != "engineering_held_out_qualification_not_effect_estimation":
        raise ValueError("v5 held-out qualification scope changed")
    if bool(protocol["design"].get("formal_effect_estimation_authorized", True)):
        raise ValueError("held-out qualification cannot authorize effect estimation")
    return path, protocol


def _validate_parent(protocol):
    result_path = development._repo_path(protocol["candidate_development_result"])
    manifest_path = development._repo_path(
        protocol["candidate_development_manifest"]
    )
    if development._sha256(result_path) != protocol[
        "candidate_development_result_sha256"
    ]:
        raise RuntimeError("candidate development result hash mismatch")
    if development._sha256(manifest_path) != protocol[
        "candidate_development_manifest_sha256"
    ]:
        raise RuntimeError("candidate development manifest hash mismatch")
    result = development._load_json(result_path)
    manifest = development._load_json(manifest_path)
    if result.get("status") != "development_gate_pass":
        raise RuntimeError("candidate development gate did not pass")
    if manifest.get("bundle_sha256") != protocol[
        "candidate_development_bundle_sha256"
    ]:
        raise RuntimeError("candidate development bundle hash mismatch")
    candidate_protocol = v4._load_yaml(
        development._repo_path(protocol["candidate_development_protocol"])
    )
    if candidate_protocol.get("candidate_overrides") != protocol.get(
        "candidate_overrides"
    ):
        raise RuntimeError("held-out candidate differs from frozen C3 candidate")


def _source(protocol):
    source_path = development._repo_path(protocol["source_protocol"])
    if development._sha256(source_path) != protocol["source_protocol_sha256"]:
        raise RuntimeError("source protocol hash mismatch")
    _, source_protocol, id_config, ood_config, verified = v4.validate_protocol(
        source_path
    )
    return source_path, source_protocol, id_config, ood_config, verified


def _seed_in_ranges(seed, ranges):
    return any(
        int(item["start"]) <= int(seed) <= int(item["end"])
        for item in ranges
    )


def build_registry(protocol, source_protocol, id_config, ood_config):
    selection = protocol["seed_selection"]
    needed = int(selection["seeds_per_split"])
    domain_configs = {"id": id_config, "ood": ood_config}
    certificates = {}
    for split in ("id", "ood"):
        certificates[split] = v4._first_certified(
            selection["candidate_start"][split],
            selection["candidate_count_per_split"],
            needed,
            split,
            source_protocol,
            domain_configs[split],
        )
    selected = {
        int(item["seed"])
        for split in ("id", "ood")
        for item in certificates[split]
    }
    if len(selected) != needed * 2:
        raise RuntimeError("held-out selected seeds are not unique")
    if any(_seed_in_ranges(seed, selection["forbidden_seed_ranges"]) for seed in selected):
        raise RuntimeError("held-out seed overlaps a forbidden range")
    if any(_seed_in_ranges(seed, selection["reserved_future_formal_ranges"]) for seed in selected):
        raise RuntimeError("held-out seed overlaps a future formal range")

    arms = list(protocol["design"]["paired_arms"])
    if arms != ["V4_full_frozen", "V5_dual_horizon_full"]:
        raise RuntimeError("held-out paired-arm contract changed")
    rng = random.Random(int(protocol["design"]["schedule_seed"]))
    blocks = []
    for split_index, split in enumerate(("id", "ood")):
        first = rng.randrange(2)
        for index, certificate in enumerate(certificates[split]):
            offset = (first + index) % 2
            arm_order = arms[offset:] + arms[:offset]
            blocks.append({
                "split": split,
                "seed": int(certificate["seed"]),
                "model_block": (index + split_index) % 3,
                "stratum": "controller_independent_conflict_certificate",
                "arm_order": arm_order,
                "certificate": certificate,
            })
    rng.shuffle(blocks)
    for index, block in enumerate(blocks, start=1):
        block["block_order"] = index
    return {
        "schema_version": 1,
        "status": "sealed_before_held_out_qualification_execution",
        "scope": protocol["scope"],
        "outcome_selection_used": False,
        "controller_outcomes_opened": False,
        "selection_rule": (
            "ascending_first_16_per_split_with_prefrozen_"
            "controller_independent_conflict_certificate"
        ),
        "candidate_ranges": deepcopy(selection["candidate_start"]),
        "candidate_count_per_split": int(selection["candidate_count_per_split"]),
        "splits": certificates,
        "schedule": blocks,
        "schedule_sha256": development._canonical_sha256(blocks),
    }


def seal_registry(protocol_path=DEFAULT_PROTOCOL):
    protocol_path, protocol = _protocol(protocol_path)
    _validate_parent(protocol)
    _, source_protocol, id_config, ood_config, _ = _source(protocol)
    path = development._repo_path(protocol["sealed_registry"])
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError("held-out registry already exists: %s" % path)
    registry = build_registry(protocol, source_protocol, id_config, ood_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    v4._write_yaml(path, registry)
    digest = development._sha256(path)
    sidecar.write_text("%s  %s\n" % (digest, path.name), encoding="ascii")
    return {
        "registry": str(path.relative_to(ROOT)),
        "registry_sha256": digest,
        "schedule_sha256": registry["schedule_sha256"],
        "seed_blocks": len(registry["schedule"]),
        "episode_jobs": len(registry["schedule"]) * 2,
    }


def validate(protocol_path=DEFAULT_PROTOCOL):
    protocol_path, protocol = _protocol(protocol_path)
    _validate_parent(protocol)
    source_path, source_protocol, id_config, ood_config, verified = _source(
        protocol
    )
    path = development._repo_path(protocol["sealed_registry"])
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("sealed held-out registry or sidecar is missing")
    expected_file_hash = sidecar.read_text(encoding="ascii").split()[0].lower()
    actual_file_hash = development._sha256(path)
    if actual_file_hash != expected_file_hash:
        raise RuntimeError("held-out registry file hash mismatch")
    registry = v4._load_yaml(path)
    rebuilt = build_registry(protocol, source_protocol, id_config, ood_config)
    if registry != rebuilt:
        raise RuntimeError("held-out registry differs from deterministic rebuild")
    if registry["schedule_sha256"] != development._canonical_sha256(
        registry["schedule"]
    ):
        raise RuntimeError("held-out schedule hash mismatch")
    if len(registry["schedule"]) != int(protocol["design"]["paired_blocks"]):
        raise RuntimeError("held-out block count mismatch")
    orders = [tuple(block["arm_order"]) for block in registry["schedule"]]
    for split in ("id", "ood"):
        split_orders = [
            tuple(block["arm_order"])
            for block in registry["schedule"]
            if block["split"] == split
        ]
        if split_orders.count(("V4_full_frozen", "V5_dual_horizon_full")) != 8:
            raise RuntimeError("held-out arm order is not balanced within split")
        if split_orders.count(("V5_dual_horizon_full", "V4_full_frozen")) != 8:
            raise RuntimeError("held-out arm order is not balanced within split")
    if len(orders) != 32:
        raise RuntimeError("held-out schedule is incomplete")
    return {
        "protocol_path": protocol_path,
        "protocol": protocol,
        "source_path": source_path,
        "source_protocol": source_protocol,
        "registry_path": path,
        "registry_sha256": actual_file_hash,
        "registry": registry,
        "verified_checkpoints": verified,
    }


def _qualification_config(protocol, source_protocol, block, arm):
    config, changes = development._configure_arm(
        protocol, source_protocol, block, arm
    )
    config["experiment"].update({
        "name": "v5_held_out_qualification__%s__seed%d__%s"
        % (block["split"], int(block["seed"]), arm),
        "paper_v5_held_out_qualification_arm": arm,
    })
    config["scope_guards"].update({
        "development_seeds_only": False,
        "paper_v5_development": False,
        "paper_v5_held_out_qualification": True,
        "formal_effect_estimation": False,
    })
    return config, changes


def preflight(protocol_path=DEFAULT_PROTOCOL):
    validated = validate(protocol_path)
    protocol = validated["protocol"]
    source_protocol = validated["source_protocol"]
    block = validated["registry"]["schedule"][0]
    audits = {}
    for arm in protocol["design"]["paired_arms"]:
        config, changes = _qualification_config(
            protocol, source_protocol, block, arm
        )
        total = int(config["planner"]["num_samples"]) * int(
            config["planner"]["paper_rl_driven"]["iterations"]
        )
        if total != 600:
            raise RuntimeError("held-out preflight rollout budget changed")
        audits[arm] = {
            "change_signature": changes,
            "resolved_config_sha256": development._canonical_sha256(config),
            "total_rollouts_per_decision": total,
        }
    return {
        "status": "preflight_pass",
        "registry_sha256": validated["registry_sha256"],
        "schedule_sha256": validated["registry"]["schedule_sha256"],
        "seed_blocks": 32,
        "episode_jobs": 64,
        "arm_audits": audits,
    }


def _run_block(payload):
    protocol, source_protocol, block, output = payload
    development._configure_threads()
    completed = []
    for arm in block["arm_order"]:
        run_dir = development._run_dir(output, block, arm)
        if run_dir.exists():
            raise FileExistsError(
                "held-out qualification artifact already exists: %s" % run_dir
            )
        config, changes = _qualification_config(
            protocol, source_protocol, block, arm
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        development._write_json(run_dir / "v5_qualification_job.json", {
            "arm": arm,
            "block": block,
            "change_signature": changes,
            "resolved_config_sha256": development._canonical_sha256(config),
        })
        ExperimentRunner(config, ROOT, output_dir=run_dir, headless=True).run()
        if not v4._complete(run_dir):
            raise RuntimeError(
                "incomplete held-out qualification episode: %s" % run_dir
            )
        completed.append(arm)
    return {"seed": int(block["seed"]), "arms_completed": completed}


def _pairs_and_rows(protocol, source_protocol, output, blocks):
    rows = []
    pairs = []
    candidate = development._candidate_arm(protocol)
    for block in blocks:
        arm_rows = {
            arm: development._episode_row(output, block, arm, source_protocol)
            for arm in protocol["design"]["paired_arms"]
        }
        rows.extend(arm_rows.values())
        control = arm_rows["V4_full_frozen"]
        treatment = arm_rows[candidate]
        pairs.append({
            "split": block["split"],
            "seed": int(block["seed"]),
            "v4_success": control["success"],
            "v5_success": treatment["success"],
            "v4_collision": control["collision"],
            "v5_collision": treatment["collision"],
            "success_delta": int(treatment["success"]) - int(control["success"]),
            "collision_delta": int(treatment["collision"]) - int(control["collision"]),
            "minimum_clearance_delta": (
                treatment["minimum_clearance"] - control["minimum_clearance"]
            ),
            "conflict_q05_clearance_delta": (
                treatment["conflict_q05_clearance"]
                - control["conflict_q05_clearance"]
            ),
            "final_goal_distance_delta": (
                treatment["final_goal_distance"]
                - control["final_goal_distance"]
            ),
            "zero_speed_risk_steps_delta": (
                treatment["zero_speed_risk_steps"]
                - control["zero_speed_risk_steps"]
            ),
            "stuck_steps_delta": treatment["stuck_steps"] - control["stuck_steps"],
        })
    return rows, pairs


def _relative_reduction(control, treatment, key):
    baseline = float(sum(float(row[key]) for row in control))
    candidate = float(sum(float(row[key]) for row in treatment))
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def _analyze(protocol, source_protocol, output, blocks):
    rows, pairs = _pairs_and_rows(protocol, source_protocol, output, blocks)
    candidate = development._candidate_arm(protocol)
    control = [row for row in rows if row["arm"] == "V4_full_frozen"]
    treatment = [row for row in rows if row["arm"] == candidate]
    conversions = sum(
        (not pair["v4_success"]) and pair["v5_success"] for pair in pairs
    )
    losses = sum(
        pair["v4_success"] and (not pair["v5_success"]) for pair in pairs
    )
    new_collisions = sum(
        (not pair["v4_collision"]) and pair["v5_collision"] for pair in pairs
    )
    v4_collisions = sum(row["collision"] for row in control)
    v5_collisions = sum(row["collision"] for row in treatment)
    net_gain = int(sum(pair["success_delta"] for pair in pairs))
    split_gains = {
        split: int(sum(
            pair["success_delta"] for pair in pairs if pair["split"] == split
        ))
        for split in ("id", "ood")
    }
    mean_clearance_delta = float(np.mean([
        pair["minimum_clearance_delta"] for pair in pairs
    ]))
    mean_conflict_q05_delta = float(np.mean([
        pair["conflict_q05_clearance_delta"] for pair in pairs
    ]))
    zero_reduction = _relative_reduction(
        control, treatment, "zero_speed_risk_steps"
    )
    stuck_reduction = _relative_reduction(control, treatment, "stuck_steps")
    direction_increase = int(
        sum(row["direction_switch_count"] for row in treatment)
        - sum(row["direction_switch_count"] for row in control)
    )
    oscillation_increase = int(
        sum(row["three_phase_oscillation_count"] for row in treatment)
        - sum(row["three_phase_oscillation_count"] for row in control)
    )
    rollout_exact = all(
        abs(float(row["paper_total_rollouts_mean"]) - 600.0) <= 1.0e-9
        for row in rows
    )
    gate = protocol["qualification_go_no_go"]
    checks = {
        "minimum_paired_safe_success_net_gain": net_gain
        >= int(gate["minimum_paired_safe_success_net_gain"]),
        "minimum_safe_noncompletion_conversions": conversions
        >= int(gate["minimum_safe_noncompletion_conversions"]),
        "no_lost_v4_successes": losses <= int(gate["maximum_lost_v4_successes"]),
        "no_new_paired_collisions": new_collisions
        <= int(gate["maximum_new_paired_collisions"]),
        "collision_count_not_worse": v5_collisions <= v4_collisions,
        "split_safe_success_not_worse": all(value >= 0 for value in split_gains.values()),
        "mean_minimum_clearance_preserved": mean_clearance_delta
        >= -float(gate["maximum_mean_minimum_clearance_loss_m"]),
        "mean_conflict_q05_clearance_preserved": mean_conflict_q05_delta
        >= -float(gate["maximum_mean_conflict_q05_clearance_loss_m"]),
        "zero_speed_risk_bounded": zero_reduction
        >= float(gate["minimum_relative_zero_speed_risk_reduction"]),
        "stuck_steps_bounded": stuck_reduction
        >= float(gate["minimum_relative_stuck_step_reduction"]),
        "direction_switches_bounded": direction_increase
        <= int(gate["maximum_total_direction_switch_increase"]),
        "three_phase_oscillation_not_worse": oscillation_increase
        <= int(gate["maximum_total_three_phase_oscillation_increase"]),
        "rollout_budget_exact": rollout_exact,
    }
    return {
        "schema_version": 1,
        "status": "qualification_pass" if all(checks.values()) else "qualification_fail",
        "scope": protocol["scope"],
        "formal_effect_estimation_authorized": False,
        "candidate_authorized_for_formal_protocol_freeze": all(checks.values()),
        "paired_blocks": len(pairs),
        "episode_jobs": len(rows),
        "safe_success_conversions": int(conversions),
        "lost_v4_successes": int(losses),
        "paired_safe_success_net_gain": net_gain,
        "split_safe_success_net_gain": split_gains,
        "new_paired_collisions": int(new_collisions),
        "v4_collisions": int(v4_collisions),
        "v5_collisions": int(v5_collisions),
        "mean_minimum_clearance_delta_m": mean_clearance_delta,
        "mean_conflict_q05_clearance_delta_m": mean_conflict_q05_delta,
        "relative_zero_speed_risk_reduction": zero_reduction,
        "relative_stuck_step_reduction": stuck_reduction,
        "total_direction_switch_increase": direction_increase,
        "total_three_phase_oscillation_increase": oscillation_increase,
        "checks": checks,
        "arm_summaries": {
            "V4_full_frozen": {
                "safe_successes": int(sum(row["success"] for row in control)),
                "collisions": int(v4_collisions),
            },
            candidate: {
                "safe_successes": int(sum(row["success"] for row in treatment)),
                "collisions": int(v5_collisions),
            },
        },
        "pairs": pairs,
        "episodes": rows,
    }


def _manifest(output):
    hashes = {}
    for path in sorted(Path(output).rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            hashes[str(path.relative_to(output)).replace("\\", "/")] = (
                development._sha256(path)
            )
    manifest = {
        "schema_version": 1,
        "file_count": len(hashes),
        "files": hashes,
        "bundle_sha256": development._canonical_sha256(hashes),
    }
    development._write_json(Path(output) / "artifact_manifest.json", manifest)
    return manifest


def run(protocol_path=DEFAULT_PROTOCOL, workers=None):
    validated = validate(protocol_path)
    protocol_path = validated["protocol_path"]
    protocol = validated["protocol"]
    source_protocol = validated["source_protocol"]
    registry = validated["registry"]
    output = development._repo_path(protocol["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("held-out qualification output is non-empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    worker_count = int(
        workers if workers is not None else protocol["design"]["workers"]
    )
    execution = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "implementation_git_sha": git_sha(ROOT),
        "protocol_sha256": development._sha256(protocol_path),
        "registry_sha256": validated["registry_sha256"],
        "schedule_sha256": registry["schedule_sha256"],
        "workers": worker_count,
        "outcomes_opened": False,
        "blocks": registry["schedule"],
    }
    development._write_json(output / "execution_schedule.json", execution)
    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _run_block,
                (protocol, source_protocol, block, str(output)),
            ): block
            for block in registry["schedule"]
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                report = future.result()
                completed += 1
                development._write_json(
                    output / "worker_reports" / (
                        "block_%02d.json" % int(block["block_order"])
                    ),
                    report,
                )
                print(
                    "[block %d/%d] %s seed=%d complete"
                    % (completed, len(registry["schedule"]), block["split"], int(block["seed"])),
                    flush=True,
                )
            except Exception as error:
                failures.append({
                    "split": block["split"],
                    "seed": int(block["seed"]),
                    "error": "".join(traceback.format_exception(
                        type(error), error, error.__traceback__
                    )),
                })
            development._write_json(output / "progress.json", {
                "status": "failed" if failures else (
                    "complete_structural" if completed == len(registry["schedule"])
                    else "running"
                ),
                "completed_blocks": completed,
                "total_blocks": len(registry["schedule"]),
                "completed_episode_jobs": completed * 2,
                "total_episode_jobs": len(registry["schedule"]) * 2,
                "failures": failures,
                "outcomes_opened": False,
            })
    if failures:
        raise RuntimeError("held-out qualification failed structurally; no automatic retry")

    result = _analyze(protocol, source_protocol, output, registry["schedule"])
    result.update({
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": development._sha256(protocol_path),
        "registry": str(validated["registry_path"].relative_to(ROOT)),
        "registry_sha256": validated["registry_sha256"],
        "schedule_sha256": registry["schedule_sha256"],
        "episode_implementation_git_sha": execution["implementation_git_sha"],
        "analysis_git_sha": git_sha(ROOT),
    })
    development._write_json(output / "qualification_result.json", result)
    development._write_json(output / "progress.json", {
        "status": result["status"],
        "completed_blocks": len(registry["schedule"]),
        "total_blocks": len(registry["schedule"]),
        "completed_episode_jobs": len(registry["schedule"]) * 2,
        "total_episode_jobs": len(registry["schedule"]) * 2,
        "failures": [],
        "outcomes_opened": True,
    })
    manifest = _manifest(output)
    return result, manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal-registry", "validate", "preflight", "run"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "seal-registry":
        payload = seal_registry(args.protocol)
    elif args.command == "validate":
        checked = validate(args.protocol)
        payload = {
            "status": "validation_pass",
            "registry_sha256": checked["registry_sha256"],
            "schedule_sha256": checked["registry"]["schedule_sha256"],
            "seed_blocks": len(checked["registry"]["schedule"]),
        }
    elif args.command == "preflight":
        payload = preflight(args.protocol)
    else:
        result, manifest = run(args.protocol, args.workers)
        payload = {
            "status": result["status"],
            "checks": result["checks"],
            "bundle_sha256": manifest["bundle_sha256"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
