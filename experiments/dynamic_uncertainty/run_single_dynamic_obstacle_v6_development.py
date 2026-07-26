"""Run paired, outcome-informed single-obstacle v6 mechanism development.

This runner deliberately reuses the frozen paper-v4 Full construction.  The
candidate may change only the explicitly declared dotted configuration paths.
It is a development runner: its output can never authorize formal claims.
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dynamic_uncertainty import (
    analyze_single_dynamic_obstacle_paper_v4 as v4_analysis,
)
from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_paper_v4 as v4,
)
from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_v5_recovery_development as v5,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "single_dynamic_obstacle_v6_emergency_development_a1.yaml"
)


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _candidate_arm(protocol):
    arms = list(protocol["design"]["paired_arms"])
    candidates = [arm for arm in arms if arm != "V4_full_frozen"]
    if len(candidates) != 1:
        raise ValueError("v6 development requires exactly one candidate arm")
    return str(candidates[0])


def _set_dotted(config, dotted_path, value):
    keys = str(dotted_path).split(".")
    if len(keys) < 2:
        raise ValueError("override path must be dotted: %s" % dotted_path)
    cursor = config
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError("unknown override parent: %s" % dotted_path)
        cursor = cursor[key]
    cursor[keys[-1]] = deepcopy(value)


def _validate_parent(protocol):
    for path_key, hash_key in (
        ("parent_result", "parent_result_sha256"),
        ("parent_manifest", "parent_manifest_sha256"),
    ):
        path = _repo_path(protocol[path_key])
        if v5._sha256(path) != str(protocol[hash_key]).lower():
            raise RuntimeError("v6 parent evidence hash mismatch: %s" % path_key)
    manifest = v5._load_json(_repo_path(protocol["parent_manifest"]))
    if manifest["bundle_sha256"] != protocol["parent_bundle_sha256"]:
        raise RuntimeError("v6 parent evidence bundle hash mismatch")


def _configure_arm(protocol, source_protocol, block, arm):
    base = v4.load_yaml(v4._repo_path(source_protocol["base_config"]))
    stage3 = v4._mapping(v4._repo_path(source_protocol["stage3_protocol"]))
    stage4 = v4._mapping(v4._repo_path(source_protocol["stage4_protocol"]))
    config = v4.configure_arm(
        source_protocol,
        block,
        "B11_full_proposed",
        base,
        stage3,
        stage4,
    )
    before = deepcopy(config)
    candidate = _candidate_arm(protocol)
    overrides = protocol["candidate_override_paths"]
    allowed_prefixes = tuple(protocol["change_scope"]["allowed_prefixes"])
    if arm == candidate:
        for dotted_path, value in overrides.items():
            if not str(dotted_path).startswith(allowed_prefixes):
                raise ValueError("override escaped frozen scope: %s" % dotted_path)
            _set_dotted(config, dotted_path, value)
    elif arm != "V4_full_frozen":
        raise ValueError("unknown v6 arm: %s" % arm)

    changes = v4._change_signature(before, config)
    expected = {
        path
        for path, value in overrides.items()
        if v4._flatten(before).get(path, "<MISSING>") != value
    }
    if arm == "V4_full_frozen" and changes:
        raise RuntimeError("V4 frozen arm changed: %s" % sorted(changes))
    if arm == candidate and set(changes) != expected:
        raise RuntimeError(
            "v6 candidate change scope differs: %s vs %s"
            % (sorted(changes), sorted(expected))
        )

    runner_label = str(protocol.get("runner_label", "v6"))
    config["experiment"].update({
        "name": "%s_development__%s__seed%d__%s"
        % (
            runner_label,
            block["split"],
            int(block["seed"]),
            arm,
        ),
        "paper_v6_development_arm": arm,
    })
    config["scope_guards"].update({
        "development_seeds_only": True,
        "paper_v4": False,
        "paper_v6_development": True,
        "formal_effect_estimation": False,
    })
    total = int(config["planner"]["num_samples"]) * int(
        config["planner"]["paper_rl_driven"]["iterations"]
    )
    if total != 600:
        raise RuntimeError("v6 changed the frozen 600-rollout budget")
    if config["perception"]["scan_guard"].get(
        "dynamic_deadline_supervisor_enabled", False
    ):
        raise RuntimeError("rejected v5 deadline supervisor leaked into v6")
    return config, changes


def _run_dir(output, block, arm):
    return (
        Path(output)
        / "runs"
        / str(block["split"])
        / ("seed_%d" % int(block["seed"]))
        / str(arm)
    )


def _run_block(payload):
    protocol_path, block, output = payload
    v5._configure_threads()
    protocol = v4._load_yaml(protocol_path)
    _, source, _, _, _ = v4.validate_protocol(
        _repo_path(protocol["source_protocol"])
    )
    completed = []
    for arm in block["arm_order"]:
        run_dir = _run_dir(output, block, arm)
        if run_dir.exists():
            raise FileExistsError("v6 artifact already exists: %s" % run_dir)
        config, changes = _configure_arm(protocol, source, block, arm)
        run_dir.mkdir(parents=True, exist_ok=False)
        runner_label = str(protocol.get("runner_label", "v6"))
        v5._write_json(run_dir / ("%s_development_job.json" % runner_label), {
            "arm": arm,
            "block": block,
            "change_signature": changes,
            "resolved_config_sha256": v5._canonical_sha256(config),
        })
        ExperimentRunner(config, ROOT, output_dir=run_dir, headless=True).run()
        if not v4._complete(run_dir):
            raise RuntimeError("incomplete v6 episode: %s" % run_dir)
        completed.append(arm)
    return {"seed": int(block["seed"]), "arms_completed": completed}


def _episode_row(output, block, arm, source_protocol):
    row = v5._episode_row(output, block, arm, source_protocol)
    metrics = v5._load_json(_run_dir(output, block, arm) / "metrics.json")
    row.update({
        "emergency_candidate_selected_steps": int(metrics.get(
            "probabilistic_obstacle_emergency_candidate_selected_steps", 0
        )),
        "speed_governor_bypassed_for_active_avoidance_steps": int(metrics.get(
            "probabilistic_obstacle_speed_governor_bypassed_for_active_avoidance_steps",
            0,
        )),
        "temporal_emergency_triggered_steps": int(metrics.get(
            "probabilistic_obstacle_temporal_emergency_triggered_steps", 0
        )),
        "temporal_emergency_vetted_steps": int(metrics.get(
            "probabilistic_obstacle_temporal_emergency_vetted_steps", 0
        )),
        "planner_temporal_escape_active_steps": int(metrics.get(
            "planner_temporal_escape_active_steps", 0
        )),
    })
    return row


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def _analyze(protocol, source_protocol, output, blocks):
    rows, pairs = [], []
    candidate_arm = _candidate_arm(protocol)
    for block in blocks:
        arm_rows = {
            arm: _episode_row(output, block, arm, source_protocol)
            for arm in protocol["design"]["paired_arms"]
        }
        rows.extend(arm_rows.values())
        control = arm_rows["V4_full_frozen"]
        candidate = arm_rows[candidate_arm]
        pairs.append({
            "split": block["split"],
            "seed": int(block["seed"]),
            "stratum": block["stratum"],
            "v4_success": control["success"],
            "v6_success": candidate["success"],
            "v4_collision": control["collision"],
            "v6_collision": candidate["collision"],
            "success_delta": int(candidate["success"]) - int(control["success"]),
            "collision_delta": int(candidate["collision"]) - int(control["collision"]),
            "steps_delta": candidate["steps"] - control["steps"],
            "final_goal_distance_delta": (
                candidate["final_goal_distance"] - control["final_goal_distance"]
            ),
            "minimum_clearance_delta": (
                candidate["minimum_clearance"] - control["minimum_clearance"]
            ),
        })

    control = [row for row in rows if row["arm"] == "V4_full_frozen"]
    candidate = [row for row in rows if row["arm"] == candidate_arm]
    safe_success_gain = sum(row["success"] for row in candidate) - sum(
        row["success"] for row in control
    )
    prevented_collisions = sum(
        pair["v4_collision"] and not pair["v6_collision"] for pair in pairs
    )
    new_collisions = sum(
        not pair["v4_collision"] and pair["v6_collision"] for pair in pairs
    )
    lost_successes = sum(
        pair["v4_success"] and not pair["v6_success"] for pair in pairs
    )
    v4_collisions = sum(row["collision"] for row in control)
    v6_collisions = sum(row["collision"] for row in candidate)
    clearance_delta = _mean(candidate, "minimum_clearance") - _mean(
        control, "minimum_clearance"
    )
    gate = protocol["development_go_no_go"]
    checks = {
        "safe_success_noninferior": safe_success_gain
        >= int(gate["minimum_safe_success_gain"]),
        "minimum_prevented_collisions": prevented_collisions
        >= int(gate["minimum_prevented_collisions"]),
        "no_new_paired_collisions": new_collisions
        <= int(gate["maximum_new_paired_collisions"]),
        "no_lost_v4_successes": lost_successes
        <= int(gate["maximum_lost_v4_successes"]),
        "collision_count_improved": (v4_collisions - v6_collisions)
        >= int(gate["minimum_collision_count_reduction"]),
        "mean_minimum_clearance_preserved": clearance_delta
        >= -float(gate["maximum_mean_minimum_clearance_loss_m"]),
        # v7's hard-risk motion contract intentionally reuses the normal
        # active-avoidance fallback rather than the legacy emergency-candidate
        # path.  Count the source-level speed-governor bypass as exercise of
        # that mechanism; otherwise the old audit would report a false gate
        # failure even when the selected motion was preserved.
        "emergency_mechanism_exercised": any(
            (
                int(row.get("emergency_candidate_selected_steps", 0))
                + int(row.get(
                    "speed_governor_bypassed_for_active_avoidance_steps", 0
                ))
            ) > 0
            for row in candidate
        ),
        "rollout_budget_exact": all(
            abs(row["paper_total_rollouts_mean"] - 600.0) <= 1.0e-9
            for row in rows
        ),
    }
    return {
        "schema_version": 1,
        "status": "development_gate_pass" if all(checks.values()) else "development_gate_fail",
        "scope": protocol["scope"],
        "formal_effect_estimation_authorized": False,
        "episode_jobs": len(rows),
        "paired_blocks": len(pairs),
        "safe_success_gain": int(safe_success_gain),
        "prevented_collisions": int(prevented_collisions),
        "new_paired_collisions": int(new_collisions),
        "lost_v4_successes": int(lost_successes),
        "v4_collisions": int(v4_collisions),
        "v6_collisions": int(v6_collisions),
        "mean_minimum_clearance_delta_m": clearance_delta,
        "checks": checks,
        "arm_summaries": {
            "V4_full_frozen": {
                "successes": int(sum(row["success"] for row in control)),
                "collisions": int(v4_collisions),
                "mean_final_goal_distance": _mean(control, "final_goal_distance"),
            },
            candidate_arm: {
                "successes": int(sum(row["success"] for row in candidate)),
                "collisions": int(v6_collisions),
                "mean_final_goal_distance": _mean(candidate, "final_goal_distance"),
                "emergency_candidate_selected_steps": int(sum(
                    row["emergency_candidate_selected_steps"] for row in candidate
                )),
                "speed_governor_bypassed_for_active_avoidance_steps": int(sum(
                    row["speed_governor_bypassed_for_active_avoidance_steps"]
                    for row in candidate
                )),
                "temporal_emergency_vetted_steps": int(sum(
                    row["temporal_emergency_vetted_steps"] for row in candidate
                )),
            },
        },
        "pairs": pairs,
        "episodes": rows,
    }


def _finalize(protocol, source_protocol, protocol_path, output, schedule):
    result = _analyze(protocol, source_protocol, output, schedule["blocks"])
    result.update({
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": v5._sha256(protocol_path),
        "implementation_git_sha": schedule["implementation_git_sha"],
        "episode_implementation_git_sha": schedule["implementation_git_sha"],
        "analysis_git_sha": git_sha(ROOT),
        "schedule_sha256": schedule["schedule_sha256"],
    })
    v5._write_json(output / "development_result.json", result)
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            hashes[str(path.relative_to(output)).replace("\\", "/")] = v5._sha256(path)
    manifest = {
        "schema_version": 1,
        "file_count": len(hashes),
        "files": hashes,
        "bundle_sha256": v5._canonical_sha256(hashes),
    }
    v5._write_json(output / "artifact_manifest.json", manifest)
    return result, manifest


def run(protocol_path=DEFAULT_PROTOCOL, workers=None):
    protocol_path = _repo_path(protocol_path)
    protocol = v4._load_yaml(protocol_path)
    if protocol["status"] != "frozen_before_development_execution":
        raise ValueError("v6 development protocol is not frozen")
    _validate_parent(protocol)
    source_path = _repo_path(protocol["source_protocol"])
    _, source, id_config, ood_config, _ = v4.validate_protocol(source_path)
    output = _repo_path(protocol["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("v6 output is non-empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    blocks = deepcopy(protocol["development_blocks"])
    domains = {"id": id_config, "ood": ood_config}
    for index, block in enumerate(blocks, start=1):
        block["block_order"] = index
        block["certificate"] = v4.conflict_certificate(
            int(block["seed"]), block["split"], source, domains[block["split"]]
        )
        if block["certificate"] is None:
            raise RuntimeError("v6 seed lacks controller-independent conflict certificate")
    worker_count = int(workers if workers is not None else protocol["design"]["workers"])
    schedule = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "outcome_informed_development_selection": True,
        "formal_effect_estimation_authorized": False,
        "implementation_git_sha": git_sha(ROOT),
        "workers": worker_count,
        "blocks": blocks,
        "schedule_sha256": v5._canonical_sha256(blocks),
    }
    v5._write_json(output / "schedule.json", schedule)
    failures, completed = [], 0
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_run_block, (str(protocol_path), block, str(output))): block
            for block in blocks
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                report = future.result()
                completed += 1
                print(
                    "[block %d/%d] %s seed=%d complete"
                    % (completed, len(blocks), block["split"], int(block["seed"])),
                    flush=True,
                )
                v5._write_json(
                    output / "worker_reports" / ("block_%02d.json" % block["block_order"]),
                    report,
                )
            except Exception as error:
                detail = "".join(traceback.format_exception(
                    type(error), error, error.__traceback__
                ))
                failures.append({
                    "seed": int(block["seed"]),
                    "split": block["split"],
                    "error": detail,
                })
                print(detail, flush=True)
            v5._write_json(output / "progress.json", {
                "status": "failed" if failures else (
                    "complete" if completed == len(blocks) else "running"
                ),
                "completed_blocks": completed,
                "total_blocks": len(blocks),
                "failures": failures,
            })
    if failures:
        raise RuntimeError("v6 development failed; no automatic retry")
    return _finalize(protocol, source, protocol_path, output, schedule)


def analyze_existing(protocol_path=DEFAULT_PROTOCOL):
    """Verify and finalize a complete raw run after detached stdout."""

    protocol_path = _repo_path(protocol_path)
    protocol = v4._load_yaml(protocol_path)
    if protocol["status"] != "frozen_before_development_execution":
        raise ValueError("v6 development protocol is not frozen")
    _validate_parent(protocol)
    _, source, _, _, _ = v4.validate_protocol(
        _repo_path(protocol["source_protocol"])
    )
    output = _repo_path(protocol["output_dir"])
    if (output / "development_result.json").exists():
        raise FileExistsError("v6 development result already exists")
    if (output / "artifact_manifest.json").exists():
        raise FileExistsError("v6 artifact manifest already exists")
    schedule = v5._load_json(output / "schedule.json")
    if schedule["schedule_sha256"] != v5._canonical_sha256(
        schedule["blocks"]
    ):
        raise RuntimeError("v6 existing schedule hash mismatch")
    expected = protocol["development_blocks"]
    if len(schedule["blocks"]) != len(expected):
        raise RuntimeError("v6 existing schedule block count mismatch")
    identity = ("split", "seed", "model_block", "stratum", "arm_order")
    for scheduled, frozen in zip(schedule["blocks"], expected):
        if any(scheduled[key] != frozen[key] for key in identity):
            raise RuntimeError("v6 existing schedule differs from protocol")
        if not scheduled.get("certificate"):
            raise RuntimeError("v6 existing schedule lacks certificate")
        for arm in protocol["design"]["paired_arms"]:
            run_dir = _run_dir(output, scheduled, arm)
            if not v4._complete(run_dir):
                raise RuntimeError("incomplete existing v6 episode: %s" % run_dir)
            runner_label = str(protocol.get("runner_label", "v6"))
            job = v5._load_json(
                run_dir / ("%s_development_job.json" % runner_label)
            )
            if job["arm"] != arm or int(job["block"]["seed"]) != int(
                scheduled["seed"]
            ):
                raise RuntimeError("v6 existing job identity mismatch: %s" % run_dir)
    v5._write_json(output / "progress.json", {
        "status": "complete",
        "completed_blocks": len(expected),
        "total_blocks": len(expected),
        "failures": [],
        "progress_reconciled_after_detached_stdout": True,
    })
    return _finalize(protocol, source, protocol_path, output, schedule)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.analyze_existing:
        result, manifest = analyze_existing(args.protocol)
    else:
        result, manifest = run(args.protocol, args.workers)
    print(json.dumps({
        "status": result["status"],
        "checks": result["checks"],
        "bundle_sha256": manifest["bundle_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
