"""Run an outcome-informed, non-confirmatory v5 development panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner

from experiments.dynamic_uncertainty import (
    analyze_single_dynamic_obstacle_paper_v4 as v4_analysis,
)
from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_paper_v4 as v4,
)

DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "single_dynamic_obstacle_v5_recovery_development_a5.yaml"
)


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _candidate_arm(development):
    arms = list(development["design"]["paired_arms"])
    candidates = [arm for arm in arms if arm != "V4_full_frozen"]
    if len(candidates) != 1:
        raise ValueError("development design must contain one candidate arm")
    return str(candidates[0])


def _candidate_overrides(development):
    if "candidate_overrides" in development:
        return development["candidate_overrides"]
    return development["v5_recovery_overrides"]


def _validate_parent_attempt(development):
    if "parent_development_result" not in development:
        return
    result_path = _repo_path(development["parent_development_result"])
    manifest_path = _repo_path(development["parent_development_manifest"])
    if _sha256(result_path) != development[
        "parent_development_result_sha256"
    ]:
        raise RuntimeError("parent development result hash mismatch")
    if _sha256(manifest_path) != development[
        "parent_development_manifest_sha256"
    ]:
        raise RuntimeError("parent development manifest hash mismatch")
    parent_manifest = _load_json(manifest_path)
    if parent_manifest["bundle_sha256"] != development[
        "parent_development_bundle_sha256"
    ]:
        raise RuntimeError("parent development bundle hash mismatch")


def _configure_threads():
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _run_dir(output, block, arm):
    return (
        Path(output)
        / "runs"
        / str(block["split"])
        / ("seed_%d" % int(block["seed"]))
        / str(arm)
    )


def _configure_arm(development, source_protocol, block, arm):
    base = load_yaml(v4._repo_path(source_protocol["base_config"]))
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
    candidate_arm = _candidate_arm(development)
    overrides = _candidate_overrides(development)
    if arm == candidate_arm:
        guard = config["perception"]["scan_guard"]
        for key, value in overrides.items():
            guard[str(key)] = deepcopy(value)
    elif arm != "V4_full_frozen":
        raise ValueError("unknown development arm: %s" % arm)

    changes = v4._change_signature(before, config)
    expected = {
        "perception.scan_guard.%s" % key
        for key in overrides
        if before["perception"]["scan_guard"].get(key, "<MISSING>")
        != overrides[key]
    }
    if arm == "V4_full_frozen" and changes:
        raise RuntimeError("V4 frozen arm changed: %s" % sorted(changes))
    if arm == candidate_arm and set(changes) != expected:
        raise RuntimeError(
            "v5 candidate change scope differs: %s vs %s"
            % (sorted(changes), sorted(expected))
        )

    config["experiment"].update({
        "name": "v5_recovery_development__%s__seed%d__%s"
        % (block["split"], int(block["seed"]), arm),
        "paper_v5_development_arm": arm,
    })
    config["scope_guards"].update({
        "development_seeds_only": True,
        "paper_v4": False,
        "paper_v5_development": True,
    })
    total = int(config["planner"]["num_samples"]) * int(
        config["planner"]["paper_rl_driven"]["iterations"]
    )
    if total != 600:
        raise RuntimeError("development arm changed the 600-rollout budget")
    return config, changes


def _run_block(payload):
    protocol_path, block, output = payload
    _configure_threads()
    development = v4._load_yaml(protocol_path)
    _, source_protocol, _, _, _ = v4.validate_protocol(
        _repo_path(development["source_protocol"])
    )
    completed = []
    for arm in block["arm_order"]:
        run_dir = _run_dir(output, block, arm)
        if run_dir.exists():
            raise FileExistsError("development artifact already exists: %s" % run_dir)
        config, changes = _configure_arm(
            development, source_protocol, block, arm
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "v5_development_job.json", {
            "arm": arm,
            "block": block,
            "change_signature": changes,
            "resolved_config_sha256": _canonical_sha256(config),
        })
        ExperimentRunner(
            config, ROOT, output_dir=run_dir, headless=True
        ).run()
        if not v4._complete(run_dir):
            raise RuntimeError("incomplete development episode: %s" % run_dir)
        completed.append(arm)
    return {"seed": int(block["seed"]), "arms_completed": completed}


def _episode_row(output, block, arm, source_protocol):
    run_dir = _run_dir(output, block, arm)
    metrics = _load_json(run_dir / "metrics.json")
    trajectory = v4_analysis._read_trajectory(run_dir / "trajectory.csv")
    behavior = v4_analysis._behavior_metrics(
        trajectory,
        block,
        source_protocol,
        float(metrics["minimum_clearance"]),
    )
    row = {
        "arm": arm,
        "split": block["split"],
        "seed": int(block["seed"]),
        "model_block": int(block["model_block"]),
        "stratum": block["stratum"],
        "success": bool(metrics["success"]),
        "collision": bool(metrics["collision"]),
        "steps": int(metrics["steps"]),
        "final_goal_distance": float(metrics["final_goal_distance"]),
        "minimum_clearance": float(metrics["minimum_clearance"]),
        "stuck_steps": int(metrics.get("stuck_steps", 0)),
        "planner_compute_ms_p95_health_only": float(
            metrics.get("planner_compute_ms_p95", 0.0)
        ),
        "paper_total_rollouts_mean": float(
            metrics.get("paper_total_rollouts_mean", 0.0)
        ),
        "dynamic_recovery_active_steps": int(
            metrics.get("dynamic_recovery_active_steps", 0)
        ),
        "dynamic_recovery_advance_steps": int(
            metrics.get("dynamic_recovery_advance_steps", 0)
        ),
        "dynamic_recovery_forward_commit_steps": int(
            metrics.get("dynamic_recovery_forward_commit_steps", 0)
        ),
        "dynamic_recovery_speed_floor_mean": float(
            metrics.get("dynamic_recovery_speed_floor_mean", 0.0)
        ),
        "dynamic_deadline_supervisor_active_steps": int(
            metrics.get("dynamic_deadline_supervisor_active_steps", 0)
        ),
        "dynamic_deadline_required_speed_max_active": float(
            metrics.get(
                "dynamic_deadline_required_speed_max_active", 0.0
            )
        ),
        "dynamic_deadline_speed_floor_mean_active": float(
            metrics.get(
                "dynamic_deadline_speed_floor_mean_active", 0.0
            )
        ),
    }
    row.update(behavior)
    return row


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_optional(rows, key):
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))


def _relative_reduction(control, treatment, key):
    baseline = float(sum(float(row[key]) for row in control))
    candidate = float(sum(float(row[key]) for row in treatment))
    return float((baseline - candidate) / baseline) if baseline > 0.0 else 0.0


def _analyze(development, source_protocol, output, blocks):
    rows = []
    pairs = []
    for block in blocks:
        arm_rows = {
            arm: _episode_row(output, block, arm, source_protocol)
            for arm in development["design"]["paired_arms"]
        }
        rows.extend(arm_rows.values())
        control = arm_rows["V4_full_frozen"]
        treatment = arm_rows[_candidate_arm(development)]
        pairs.append({
            "split": block["split"],
            "seed": int(block["seed"]),
            "stratum": block["stratum"],
            "v4_success": control["success"],
            "v5_success": treatment["success"],
            "v4_collision": control["collision"],
            "v5_collision": treatment["collision"],
            "success_delta": int(treatment["success"]) - int(control["success"]),
            "collision_delta": int(treatment["collision"]) - int(control["collision"]),
            "steps_delta": treatment["steps"] - control["steps"],
            "final_goal_distance_delta": (
                treatment["final_goal_distance"]
                - control["final_goal_distance"]
            ),
            "minimum_clearance_delta": (
                treatment["minimum_clearance"]
                - control["minimum_clearance"]
            ),
            "zero_speed_risk_steps_delta": (
                treatment["zero_speed_risk_steps"]
                - control["zero_speed_risk_steps"]
            ),
            "stuck_steps_delta": (
                treatment["stuck_steps"] - control["stuck_steps"]
            ),
            "release_delay_max_s_delta": (
                treatment["release_delay_max_s"]
                - control["release_delay_max_s"]
            ),
        })

    control = [row for row in rows if row["arm"] == "V4_full_frozen"]
    candidate_arm = _candidate_arm(development)
    treatment = [row for row in rows if row["arm"] == candidate_arm]
    challenge_pairs = [
        pair for pair in pairs if "safe_noncompletion" in pair["stratum"]
    ]
    conversions = sum(
        not pair["v4_success"] and pair["v5_success"]
        for pair in challenge_pairs
    )
    lost_successes = sum(
        pair["v4_success"] and not pair["v5_success"] for pair in pairs
    )
    v5_collisions = sum(row["collision"] for row in treatment)
    v4_collisions = sum(row["collision"] for row in control)
    new_paired_collisions = sum(
        not pair["v4_collision"] and pair["v5_collision"]
        for pair in pairs
    )
    zero_reduction = _relative_reduction(
        control, treatment, "zero_speed_risk_steps"
    )
    stuck_reduction = _relative_reduction(control, treatment, "stuck_steps")
    mean_clearance_delta = _mean(treatment, "minimum_clearance") - _mean(
        control, "minimum_clearance"
    )
    rollout_pass = all(
        abs(row["paper_total_rollouts_mean"] - 600.0) <= 1.0e-9
        for row in rows
    )
    gate = development["development_go_no_go"]
    minimum_conversions = int(gate.get(
        "minimum_challenge_conversions",
        gate.get("minimum_recovery_challenge_conversions", 0),
    ))
    conversion_check_name = (
        "challenge_conversions"
        if "minimum_challenge_conversions" in gate
        else "recovery_challenge_conversions"
    )
    checks = {
        conversion_check_name: conversions >= minimum_conversions,
        "no_lost_v4_successes": lost_successes
        <= int(gate["maximum_lost_v4_successes"]),
        "zero_speed_risk_reduction": zero_reduction
        >= float(gate["minimum_relative_zero_speed_risk_reduction"]),
        "stuck_step_reduction": stuck_reduction
        >= float(gate["minimum_relative_stuck_step_reduction"]),
        "mean_minimum_clearance_preserved": mean_clearance_delta
        >= -float(gate["maximum_mean_minimum_clearance_loss_m"]),
        "rollout_budget_exact": rollout_pass,
    }
    if "maximum_v5_collisions" in gate:
        checks["zero_v5_collisions"] = (
            v5_collisions <= int(gate["maximum_v5_collisions"])
        )
    if "maximum_new_paired_collisions" in gate:
        checks["no_new_paired_collisions"] = (
            new_paired_collisions
            <= int(gate["maximum_new_paired_collisions"])
        )
    if gate.get("require_collision_count_not_worse", False):
        checks["collision_count_not_worse"] = (
            v5_collisions <= v4_collisions
        )
    final_distance_delta = _mean(
        treatment, "final_goal_distance"
    ) - _mean(control, "final_goal_distance")
    direction_switch_increase = int(sum(
        row.get("direction_switch_count", 0) for row in treatment
    ) - sum(row.get("direction_switch_count", 0) for row in control))
    oscillation_increase = int(sum(
        row.get("three_phase_oscillation_count", 0) for row in treatment
    ) - sum(
        row.get("three_phase_oscillation_count", 0) for row in control
    ))
    if "minimum_mean_final_goal_distance_reduction_m" in gate:
        checks["mean_final_goal_distance_reduced"] = (
            final_distance_delta
            <= -float(gate[
                "minimum_mean_final_goal_distance_reduction_m"
            ])
        )
    if "maximum_total_direction_switch_increase" in gate:
        checks["direction_switches_bounded"] = (
            direction_switch_increase
            <= int(gate["maximum_total_direction_switch_increase"])
        )
    if "maximum_total_three_phase_oscillation_increase" in gate:
        checks["three_phase_oscillation_not_worse"] = (
            oscillation_increase
            <= int(gate[
                "maximum_total_three_phase_oscillation_increase"
            ])
        )
    return {
        "schema_version": 1,
        "status": "development_gate_pass" if all(checks.values()) else "development_gate_fail",
        "scope": development["scope"],
        "formal_effect_estimation_authorized": False,
        "episode_jobs": len(rows),
        "paired_blocks": len(pairs),
        "recovery_challenge_conversions": int(conversions),
        "lost_v4_successes": int(lost_successes),
        "v5_collisions": int(v5_collisions),
        "v4_collisions": int(v4_collisions),
        "new_paired_collisions": int(new_paired_collisions),
        "relative_zero_speed_risk_reduction": zero_reduction,
        "relative_stuck_step_reduction": stuck_reduction,
        "mean_minimum_clearance_delta_m": mean_clearance_delta,
        "mean_final_goal_distance_delta_m": final_distance_delta,
        "total_direction_switch_increase": direction_switch_increase,
        "total_three_phase_oscillation_increase": oscillation_increase,
        "checks": checks,
        "arm_summaries": {
            "V4_full_frozen": {
                "successes": int(sum(row["success"] for row in control)),
                "collisions": int(sum(row["collision"] for row in control)),
                "mean_steps": _mean(control, "steps"),
                "mean_zero_speed_risk_steps": _mean(
                    control, "zero_speed_risk_steps"
                ),
                "mean_stuck_steps": _mean(control, "stuck_steps"),
                "mean_release_delay_max_s": _mean(
                    control, "release_delay_max_s"
                ),
                "mean_minimum_clearance": _mean(
                    control, "minimum_clearance"
                ),
            },
            candidate_arm: {
                "successes": int(sum(row["success"] for row in treatment)),
                "collisions": int(sum(row["collision"] for row in treatment)),
                "mean_steps": _mean(treatment, "steps"),
                "mean_zero_speed_risk_steps": _mean(
                    treatment, "zero_speed_risk_steps"
                ),
                "mean_stuck_steps": _mean(treatment, "stuck_steps"),
                "mean_release_delay_max_s": _mean(
                    treatment, "release_delay_max_s"
                ),
                "mean_minimum_clearance": _mean(
                    treatment, "minimum_clearance"
                ),
                "mean_dynamic_recovery_advance_steps": _mean(
                    treatment, "dynamic_recovery_advance_steps"
                ),
                "mean_dynamic_recovery_forward_commit_steps": _mean(
                    treatment, "dynamic_recovery_forward_commit_steps"
                ),
                "mean_dynamic_deadline_supervisor_active_steps": _mean_optional(
                    treatment,
                    "dynamic_deadline_supervisor_active_steps",
                ),
                "maximum_dynamic_deadline_required_speed": float(max(
                    row.get(
                        "dynamic_deadline_required_speed_max_active", 0.0
                    )
                    for row in treatment
                )),
            },
        },
        "pairs": pairs,
        "episodes": rows,
    }


def _finalize(
    development,
    source_protocol,
    protocol_path,
    source_path,
    source_analysis_sha256,
    source_analysis_manifest_sha256,
    output,
    schedule,
):
    result = _analyze(
        development, source_protocol, output, schedule["blocks"]
    )
    episode_git_sha = str(schedule["implementation_git_sha"])
    result.update({
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "source_protocol_sha256": _sha256(source_path),
        "source_analysis_sha256": source_analysis_sha256,
        "source_analysis_manifest_sha256": (
            source_analysis_manifest_sha256
        ),
        "implementation_git_sha": episode_git_sha,
        "episode_implementation_git_sha": episode_git_sha,
        "analysis_git_sha": git_sha(ROOT),
        "schedule_sha256": str(schedule["schedule_sha256"]),
    })
    _write_json(output / "development_result.json", result)
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            hashes[str(path.relative_to(output)).replace("\\", "/")] = (
                _sha256(path)
            )
    manifest = {
        "schema_version": 1,
        "file_count": len(hashes),
        "files": hashes,
        "bundle_sha256": _canonical_sha256(hashes),
    }
    _write_json(output / "artifact_manifest.json", manifest)
    return result, manifest


def run(protocol_path=DEFAULT_PROTOCOL, workers=None):
    protocol_path = _repo_path(protocol_path)
    development = v4._load_yaml(protocol_path)
    if development["status"] != "frozen_before_development_execution":
        raise ValueError("v5 development protocol is not frozen")
    _validate_parent_attempt(development)
    source_path = _repo_path(development["source_protocol"])
    source_analysis = _repo_path(development["source_analysis"])
    source_analysis_manifest = _repo_path(
        development["source_analysis_manifest"]
    )
    source_analysis_sha256 = _sha256(source_analysis)
    source_analysis_manifest_sha256 = _sha256(source_analysis_manifest)
    if source_analysis_sha256 != development["source_analysis_sha256"]:
        raise RuntimeError("v4 development-source episode table hash mismatch")
    if (
        source_analysis_manifest_sha256
        != development["source_analysis_manifest_sha256"]
    ):
        raise RuntimeError("v4 development-source manifest hash mismatch")
    _, source_protocol, id_config, ood_config, _ = v4.validate_protocol(
        source_path
    )
    output = _repo_path(development["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("development output is non-empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    blocks = deepcopy(development["development_blocks"])
    domain_configs = {"id": id_config, "ood": ood_config}
    for index, block in enumerate(blocks, start=1):
        block["block_order"] = index
        block["certificate"] = v4.conflict_certificate(
            int(block["seed"]),
            block["split"],
            source_protocol,
            domain_configs[block["split"]],
        )
        if block["certificate"] is None:
            raise RuntimeError("development seed lacks conflict certificate")
    worker_count = int(
        workers if workers is not None else development["design"]["workers"]
    )
    schedule = {
        "schema_version": 1,
        "scope": development["scope"],
        "outcome_informed_development_selection": True,
        "formal_effect_estimation_authorized": False,
        "implementation_git_sha": git_sha(ROOT),
        "source_analysis_sha256": source_analysis_sha256,
        "source_analysis_manifest_sha256": source_analysis_manifest_sha256,
        "workers": worker_count,
        "blocks": blocks,
        "schedule_sha256": _canonical_sha256(blocks),
    }
    _write_json(output / "schedule.json", schedule)
    failures = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _run_block,
                (str(protocol_path), block, str(output)),
            ): block
            for block in blocks
        }
        completed = 0
        for future in as_completed(futures):
            block = futures[future]
            try:
                result = future.result()
                completed += 1
                print(
                    "[block %d/%d] %s seed=%d complete"
                    % (
                        completed,
                        len(blocks),
                        block["split"],
                        int(block["seed"]),
                    ),
                    flush=True,
                )
                _write_json(
                    output
                    / "worker_reports"
                    / ("block_%02d.json" % int(block["block_order"])),
                    result,
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
            _write_json(output / "progress.json", {
                "status": "failed" if failures else (
                    "complete" if completed == len(blocks) else "running"
                ),
                "completed_blocks": completed,
                "total_blocks": len(blocks),
                "failures": failures,
            })
    if failures:
        raise RuntimeError("v5 development failed; no automatic retry")

    return _finalize(
        development,
        source_protocol,
        protocol_path,
        source_path,
        source_analysis_sha256,
        source_analysis_manifest_sha256,
        output,
        schedule,
    )


def analyze_existing(protocol_path=DEFAULT_PROTOCOL):
    protocol_path = _repo_path(protocol_path)
    development = v4._load_yaml(protocol_path)
    if development["status"] != "frozen_before_development_execution":
        raise ValueError("v5 development protocol is not frozen")
    _validate_parent_attempt(development)
    source_path = _repo_path(development["source_protocol"])
    source_analysis = _repo_path(development["source_analysis"])
    source_analysis_manifest = _repo_path(
        development["source_analysis_manifest"]
    )
    source_analysis_sha256 = _sha256(source_analysis)
    source_analysis_manifest_sha256 = _sha256(source_analysis_manifest)
    if source_analysis_sha256 != development["source_analysis_sha256"]:
        raise RuntimeError("v4 development-source episode table hash mismatch")
    if (
        source_analysis_manifest_sha256
        != development["source_analysis_manifest_sha256"]
    ):
        raise RuntimeError("v4 development-source manifest hash mismatch")
    _, source_protocol, _, _, _ = v4.validate_protocol(source_path)
    output = _repo_path(development["output_dir"])
    if (output / "development_result.json").exists():
        raise FileExistsError("development result already exists")
    if (output / "artifact_manifest.json").exists():
        raise FileExistsError("development manifest already exists")
    schedule = _load_json(output / "schedule.json")
    if schedule["schedule_sha256"] != _canonical_sha256(
        schedule["blocks"]
    ):
        raise RuntimeError("development schedule hash mismatch")
    expected_blocks = development["development_blocks"]
    if len(schedule["blocks"]) != len(expected_blocks):
        raise RuntimeError("development schedule block count mismatch")
    identity_keys = ("split", "seed", "model_block", "stratum", "arm_order")
    for scheduled, expected in zip(schedule["blocks"], expected_blocks):
        if any(scheduled[key] != expected[key] for key in identity_keys):
            raise RuntimeError("development schedule differs from protocol")
        if not scheduled.get("certificate"):
            raise RuntimeError("development schedule lacks certificate")
        for arm in development["design"]["paired_arms"]:
            run_dir = _run_dir(output, scheduled, arm)
            if not v4._complete(run_dir):
                raise RuntimeError(
                    "incomplete existing development episode: %s" % run_dir
                )
    progress = _load_json(output / "progress.json")
    if (
        progress.get("status") != "complete"
        or int(progress.get("completed_blocks", -1)) != len(expected_blocks)
        or progress.get("failures")
    ):
        raise RuntimeError("existing development progress is not complete")
    return _finalize(
        development,
        source_protocol,
        protocol_path,
        source_path,
        source_analysis_sha256,
        source_analysis_manifest_sha256,
        output,
        schedule,
    )


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
