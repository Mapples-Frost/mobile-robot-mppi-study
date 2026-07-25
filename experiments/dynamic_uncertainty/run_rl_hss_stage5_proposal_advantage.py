"""Preflight and run the Stage 5 proposal-advantage mechanism probe.

Execution is opt-in through ``--execute``.  The probe is nominal-dynamics only,
uses fresh development seeds, blocks three arms within obstacle seed, and stops
at the first collision.  It never opens sealed seeds.
"""

import argparse
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
    _complete,
    _development_seed_guard,
    _json,
    _mapping,
    _resolve,
    _sha256,
    _verify_file,
    _write_json,
    configure_job as configure_stage4_job,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development.yaml"
)
ARMS = ("rl_hss_off", "rl_hss_shadow", "rl_hss_advantage_veto")


def validate_protocol(protocol):
    if str(protocol.get("status")) != "preregistered_development_not_started":
        raise ValueError("Stage 5 protocol is not in the frozen pre-run state")
    design = protocol["design"]
    if tuple(design.get("arms", ())) != ARMS:
        raise ValueError("Stage 5 arms are not frozen")
    if bool(design.get("sealed_seeds_authorized", True)):
        raise ValueError("sealed seeds are forbidden in Stage 5 development")
    if not bool(design.get("common_random_numbers", False)):
        raise ValueError("Stage 5 requires common random numbers")
    if str(design.get("scope")) != "nominal_dynamics_mechanism_probe":
        raise ValueError("Stage 5 scope must remain nominal-only")
    seeds = [int(value) for value in design["obstacle_process_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage 5 requires three unique development seeds")
    seed_audit = _development_seed_guard(seeds)
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    old_seeds = set(int(value) for value in stage4["design"]["obstacle_process_seeds"])
    if old_seeds.intersection(seeds):
        raise ValueError("Stage 5 prospective seeds overlap Stage 4")
    budget = protocol["candidate_budget"]
    total = int(budget["total_rollouts_per_controller_decision"])
    if (
        int(budget["rl_candidates_per_iteration"])
        * int(budget["rl_iterations"])
        != total
        or int(budget["standard_candidates_per_iteration"]) != total
        or bool(budget.get("additional_candidates_authorized", True))
    ):
        raise ValueError("Stage 5 candidate budget is not matched")
    gates = protocol["proposal_advantage_gate"]
    shadow = gates["shadow"]
    active = gates["active"]
    common = {
        "enabled": True,
        "relative_disadvantage_margin": 0.0,
        "consecutive_disadvantages": 3,
    }
    if shadow != {**common, "mode": "shadow"}:
        raise ValueError("Stage 5 shadow gate is not frozen")
    if active != {**common, "mode": "episode_latched_veto"}:
        raise ValueError("Stage 5 active gate is not frozen")
    source_audit = [
        _verify_file(row["path"], row["sha256"], "Stage 5 frozen source")
        for row in protocol.get("frozen_sources", ())
    ]
    implementation_audit = [
        _verify_file(
            row["path"], row["sha256"], "Stage 5 implementation"
        )
        for row in protocol.get("implementation_contract", {}).get(
            "files", ()
        )
    ]
    if not source_audit or not implementation_audit:
        raise ValueError("Stage 5 forensic file bindings are incomplete")
    return {
        "schema_version": 1,
        "protocol_valid": True,
        "sealed_seeds_opened": False,
        "development_seed_guard": seed_audit,
        "frozen_sources": source_audit,
        "implementation_contract": implementation_audit,
        "candidate_budget_per_controller_decision": total,
    }


def build_schedule(protocol):
    design = protocol["design"]
    seeds = [int(value) for value in design["obstacle_process_seeds"]]
    rng = np.random.RandomState(int(design["schedule_seed"]))
    schedule = []
    for seed_index in rng.permutation(len(seeds)):
        seed = seeds[int(seed_index)]
        for within_block, arm_index in enumerate(rng.permutation(len(ARMS))):
            arm = ARMS[int(arm_index)]
            schedule.append({
                "run_order": len(schedule),
                "within_obstacle_block_order": int(within_block),
                "episode_seed": seed,
                "arm": arm,
                "experimental_key": "%s::seed%d" % (arm, seed),
            })
    return schedule


def configure_job(base_config, job, protocol, stage3_protocol, stage4_protocol):
    arm = str(job["arm"])
    stage4_job = {
        "condition": "nominal",
        "episode_seed": int(job["episode_seed"]),
        "model_block": -1,
        "rl_hss_enabled": arm != "rl_hss_off",
    }
    config = configure_stage4_job(
        base_config,
        stage4_job,
        stage4_protocol,
        stage3_protocol,
    )
    config = deepcopy(config)
    if arm != "rl_hss_off":
        mode = "shadow" if arm == "rl_hss_shadow" else "active"
        paper = config["planner"]["paper_rl_driven"]
        paper["proposal_advantage_gate"] = deepcopy(
            protocol["proposal_advantage_gate"][mode]
        )
    config["experiment"]["name"] += "__stage5_%s" % arm
    config["experiment"]["stage5_proposal_advantage_arm"] = arm
    config["experiment"]["stage5_sealed_seeds_opened"] = False
    return config


def validate_job_config(config, job, protocol, base_config):
    arm = str(job["arm"])
    for section in (
        "task", "state_space", "action_space", "plant", "scene", "sensors",
        "perception",
    ):
        if config[section] != base_config[section]:
            raise ValueError("Stage 5 changed frozen section: %s" % section)
    planner = config["planner"]
    total = int(
        protocol["candidate_budget"]["total_rollouts_per_controller_decision"]
    )
    if planner.get("prediction_mode") != "nominal":
        raise ValueError("Stage 5 mechanism probe must use nominal dynamics")
    if arm == "rl_hss_off":
        if (
            planner.get("optimizer", "standard") != "standard"
            or planner.get("sampling_prior", "goal_warm_start")
            != "goal_warm_start"
            or int(planner["num_samples"]) != total
            or bool(config.get("rl", {}).get("enabled", False))
        ):
            raise ValueError("Stage 5 RL-off arm changed the Stage 3 baseline")
    else:
        paper = planner.get("paper_rl_driven", {})
        expected_mode = (
            "shadow" if arm == "rl_hss_shadow" else "episode_latched_veto"
        )
        gate = paper.get("proposal_advantage_gate", {})
        if (
            planner.get("optimizer") != "paper_rl_driven"
            or int(planner["num_samples"]) * int(paper.get("iterations", 0))
            != total
            or not bool(paper.get("reliability", {}).get("enabled", False))
            or bool(
                paper.get("reliability", {}).get(
                    "source_competence_enabled", False
                )
            )
            or gate.get("mode") != expected_mode
            or not bool(gate.get("enabled", False))
            or int(gate.get("consecutive_disadvantages", 0)) != 3
            or float(gate.get("relative_disadvantage_margin", -1.0)) != 0.0
        ):
            raise ValueError("Stage 5 RL/HSS arm violates the frozen treatment")
    return True


def _run_dir(output_dir, job):
    return Path(output_dir) / "runs" / str(job["arm"]) / (
        "seed_%d" % int(job["episode_seed"])
    )


def run(
    protocol_path=DEFAULT_PROTOCOL,
    *,
    execute=False,
    output_override=None,
    max_jobs=None,
):
    protocol_path = Path(protocol_path).resolve()
    protocol = _mapping(protocol_path)
    audit = validate_protocol(protocol)
    schedule = build_schedule(protocol)
    if not execute:
        return {**audit, "schedule": schedule, "formal_matrix_started": False}
    output_dir = Path(output_override or _resolve(protocol["output_dir"])).resolve()
    if ROOT not in output_dir.parents:
        raise ValueError("Stage 5 output must remain inside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = _sha256(protocol_path)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        if manifest.get("protocol_sha256") != protocol_hash:
            raise ValueError("cannot resume Stage 5 with a changed protocol")
    else:
        _write_json(manifest_path, {
            "schema_version": 1,
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": protocol_hash,
            "formal_matrix_started": True,
            "sealed_seeds_opened": False,
        })
        _write_json(output_dir / "preflight_audit.json", audit)
        (output_dir / "protocol_resolved.yaml").write_text(
            yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
        )
    schedule_payload = {
        "schema_version": 1,
        "schedule_seed": int(protocol["design"]["schedule_seed"]),
        "jobs": schedule,
    }
    schedule_path = output_dir / "schedule.json"
    if schedule_path.is_file() and _json(schedule_path) != schedule_payload:
        raise ValueError("cannot resume Stage 5 with a changed schedule")
    _write_json(schedule_path, schedule_payload)
    base_config = load_yaml(_resolve(protocol["base_config"]))
    stage3_protocol = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4_protocol = _mapping(_resolve(protocol["stage4_protocol"]))
    completed_this_call = 0
    stopped = None
    for job in schedule:
        run_dir = _run_dir(output_dir, job)
        if _complete(run_dir):
            continue
        if max_jobs is not None and completed_this_call >= int(max_jobs):
            break
        config = configure_job(
            base_config, job, protocol, stage3_protocol, stage4_protocol
        )
        validate_job_config(config, job, protocol, base_config)
        print(
            "[%d/%d] %s" % (
                int(job["run_order"]) + 1,
                len(schedule),
                job["experimental_key"],
            ),
            flush=True,
        )
        ExperimentRunner(config, ROOT, output_dir=run_dir, headless=True).run()
        completed_this_call += 1
        metrics = _json(run_dir / "metrics.json")
        if bool(metrics.get("collision", False)) and bool(
            protocol["design"].get("stop_on_first_collision", False)
        ):
            stopped = {
                "reason": "preregistered_first_collision_stop",
                "experimental_key": job["experimental_key"],
            }
            break
    completed = [job for job in schedule if _complete(_run_dir(output_dir, job))]
    progress = {
        "schema_version": 1,
        "status": (
            "stopped_by_preregistered_rule"
            if stopped is not None
            else ("complete" if len(completed) == len(schedule) else "incomplete")
        ),
        "completed_jobs": len(completed),
        "total_jobs": len(schedule),
        "remaining_jobs": len(schedule) - len(completed),
        "sealed_seeds_opened": False,
    }
    if stopped is not None:
        progress["early_stop"] = stopped
    _write_json(output_dir / "progress.json", progress)
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args(argv)
    if args.max_jobs is not None and not args.execute:
        parser.error("--max-jobs requires --execute")
    result = run(
        args.protocol,
        execute=bool(args.execute),
        output_override=args.output_dir,
        max_jobs=args.max_jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
