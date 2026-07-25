"""Preflight and resumable runner for the preregistered Stage 4 RL/HSS matrix.

The command is preflight-only unless ``--execute`` is supplied explicitly.
This prevents interface verification from accidentally starting the 24-episode
MuJoCo development matrix.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_residual_dynamics_stage1 import (
    configure_condition as configure_stage3_condition,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_uncertainty_rl_hss_stage4_amendment1.yaml"
)
REQUIRED_RUN_FILES = (
    "config_resolved.yaml",
    "metrics.json",
    "trajectory.csv",
)


def _mapping(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return value


def _resolve(value):
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(value, expected, label):
    path = _resolve(value)
    if not path.is_file():
        raise FileNotFoundError("%s missing: %s" % (label, path))
    actual = _sha256(path)
    if actual != str(expected).lower():
        raise ValueError("%s hash mismatch: %s" % (label, path))
    return {"path": str(path.relative_to(ROOT)), "sha256": actual}


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be a mapping: %s" % path)
    return value


def _development_seed_guard(seeds):
    registry = _mapping(ROOT / "configs/seeds/dynamic_uncertainty_splits.yaml")
    development = registry["splits"]["development"]
    start = int(development["seed_start"])
    stop = start + int(development["count"])
    invalid = [int(seed) for seed in seeds if not start <= int(seed) < stop]
    if invalid:
        raise ValueError(
            "Stage 4 seeds are outside the registered development split: %s"
            % invalid
        )
    return {"registry": "dynamic_uncertainty_probability_program", "range": [start, stop]}


def validate_protocol(protocol):
    """Fail closed before any MuJoCo episode or output directory is created."""

    revision = int(protocol.get("protocol_revision", 0))
    if revision not in (1, 2):
        raise ValueError("unsupported Stage 4 protocol revision")
    expected_status = {
        1: "preregistered_development_not_started",
        2: "preregistered_amendment1_development_not_started",
    }[revision]
    if str(protocol.get("status")) != expected_status:
        raise ValueError("Stage 4 protocol is not in the frozen pre-run state")
    amendment_audit = None
    implementation_audit = []
    if revision == 2:
        amendment = protocol.get("amendment", {})
        if str(amendment.get("reason")) != (
            "paper_rl_probabilistic_risk_interface_gap"
        ):
            raise ValueError("Stage 4 Amendment 1 reason is not frozen")
        failed = amendment.get("failed_launch", {})
        failed_protocol = _verify_file(
            failed.get("protocol"),
            failed.get("protocol_sha256"),
            "Stage 4 failed-launch protocol",
        )
        failed_manifest = _verify_file(
            failed.get("manifest"),
            failed.get("manifest_sha256"),
            "Stage 4 failed-launch manifest",
        )
        failed_stderr = _verify_file(
            failed.get("stderr"),
            failed.get("stderr_sha256"),
            "Stage 4 failed-launch stderr",
        )
        manifest = _json(_resolve(failed.get("manifest")))
        if (
            manifest.get("formal_matrix_started") is not True
            or bool(manifest.get("sealed_seeds_opened", True))
            or manifest.get("protocol_sha256")
            != str(failed.get("protocol_sha256")).lower()
        ):
            raise ValueError("Stage 4 failed-launch manifest is inconsistent")
        failed_output = _resolve(failed.get("output_dir"))
        completed_metrics = list(failed_output.glob("runs/**/metrics.json"))
        if completed_metrics or int(failed.get("completed_episodes", -1)) != 0:
            raise ValueError("Stage 4 failed launch contains completed episodes")
        if _resolve(protocol["output_dir"]) == failed_output:
            raise ValueError("Stage 4 Amendment 1 would overwrite failed launch")
        amendment_audit = {
            "reason": str(amendment["reason"]),
            "failed_protocol": failed_protocol,
            "failed_manifest": failed_manifest,
            "failed_stderr": failed_stderr,
            "completed_episodes": 0,
        }

        implementation = protocol.get("implementation_contract", {})
        if str(implementation.get("contract")) != (
            "paper_probabilistic_risk_parity_v1"
        ):
            raise ValueError("Stage 4 Amendment 1 implementation is not frozen")
        rows = implementation.get("files", ())
        required_paths = {
            "src/mobile_robot_mppi/planning/mppi.py",
            "src/mobile_robot_mppi/planning/rl_driven_mppi.py",
            "src/mobile_robot_mppi/runtime/factories.py",
            "src/mobile_robot_mppi/rl/paper_policy.py",
            "experiments/dynamic_uncertainty/run_rl_hss_stage4.py",
        }
        declared_paths = {str(row.get("path")) for row in rows}
        if declared_paths != required_paths:
            raise ValueError("Stage 4 implementation file set is incomplete")
        implementation_audit = [
            _verify_file(
                row["path"], row["sha256"], "Stage 4 implementation"
            )
            for row in rows
        ]
    design = protocol["design"]
    if str(design["independent_unit"]) != "complete_episode":
        raise ValueError("Stage 4 independent unit must be a complete episode")
    if bool(design.get("sealed_seeds_authorized", True)):
        raise ValueError("sealed seeds are not authorized in Stage 4 development")
    if not bool(design.get("common_random_numbers", False)):
        raise ValueError("Stage 4 requires common random numbers")
    seeds = [int(value) for value in design["obstacle_process_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage 4 requires three unique obstacle seeds")
    seed_audit = _development_seed_guard(seeds)

    sources = protocol["frozen_sources"]
    base_audit = _verify_file(
        protocol["base_config"],
        sources["base_config_sha256"],
        "Stage 3 base config",
    )
    stage3_audit = _verify_file(
        protocol["stage3_protocol"],
        sources["stage3_protocol_sha256"],
        "Stage 3 protocol",
    )
    gate_audit = _verify_file(
        sources["stage3_gate"]["path"],
        sources["stage3_gate"]["sha256"],
        "Stage 3 gate",
    )
    equivalence_audit = _verify_file(
        sources["stage3_sequential_equivalence"]["path"],
        sources["stage3_sequential_equivalence"]["sha256"],
        "Stage 3 sequential equivalence audit",
    )
    gate = _json(_resolve(sources["stage3_gate"]["path"]))
    equivalence = _json(
        _resolve(sources["stage3_sequential_equivalence"]["path"])
    )
    if gate.get("overall_pass") is not True:
        raise ValueError("Stage 3 gate is not passed")
    if equivalence.get("all_outcomes_match") is not True:
        raise ValueError("Stage 3 sequential equivalence is not passed")
    if bool(gate.get("sealed_seeds_opened", True)) or bool(
        equivalence.get("sealed_seeds_opened", True)
    ):
        raise ValueError("frozen Stage 3 evidence reports opened sealed seeds")

    stage3_protocol = _mapping(_resolve(protocol["stage3_protocol"]))
    if stage3_protocol["base_config"] != protocol["base_config"]:
        raise ValueError("Stage 4 does not point to the Stage 3 base config")
    stage3_seeds = [
        int(value)
        for value in stage3_protocol["design"]["development_episode_seeds"]
    ]
    if stage3_seeds != seeds:
        raise ValueError("Stage 4 obstacle seeds differ from Stage 3")
    if bool(stage3_protocol["design"].get("sealed_seeds_authorized", True)):
        raise ValueError("source Stage 3 protocol authorizes sealed seeds")

    actor = protocol["actor"]
    actor_audit = _verify_file(
        actor["checkpoint"], actor["sha256"], "frozen point-goal Actor"
    )
    if str(actor.get("action_adapter", "")) != "frozen_actor_subspace_v1":
        raise ValueError("Stage 4 Actor action adapter is not frozen")
    from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint

    actor_payload = load_sac_checkpoint(
        _resolve(actor["checkpoint"]), map_location="cpu"
    )
    saved_action = actor_payload.get("action_spec", {})
    declared_actor_lower = np.asarray(
        actor["checkpoint_action_lower"], dtype=np.float64
    )
    declared_actor_upper = np.asarray(
        actor["checkpoint_action_upper"], dtype=np.float64
    )
    if (
        str(actor_payload.get("action_mode")) != "direct_control"
        or tuple(saved_action.get("names", ())) != ("v_cmd", "omega_cmd")
        or not np.array_equal(
            np.asarray(saved_action.get("lower"), dtype=np.float64),
            declared_actor_lower,
        )
        or not np.array_equal(
            np.asarray(saved_action.get("upper"), dtype=np.float64),
            declared_actor_upper,
        )
    ):
        raise ValueError("frozen Actor checkpoint action contract changed")
    actor_audit["action_mode"] = "direct_control"
    actor_audit["action_lower"] = declared_actor_lower.tolist()
    actor_audit["action_upper"] = declared_actor_upper.tolist()
    hss = protocol["hss"]
    hss_evidence = {}
    for name in ("calibration_summary", "calibration_config", "stress_gate"):
        row = hss[name]
        hss_evidence[name] = _verify_file(
            row["path"], row["sha256"], "HSS %s" % name
        )
    stress = _json(_resolve(hss["stress_gate"]["path"]))
    if stress.get("gate_passed") is not True:
        raise ValueError("frozen HSS stress gate is not passed")
    sidecar_members = []
    for index, member in enumerate(hss["sidecar"]["members"]):
        sidecar_members.append(
            _verify_file(
                member["checkpoint"],
                member["sha256"],
                "HSS sidecar member %d" % index,
            )
        )
    if len(sidecar_members) != 3:
        raise ValueError("Stage 4 requires all three frozen HSS members")

    blocks = stage3_protocol["residual_model_blocks"]
    if len(blocks) != 3:
        raise ValueError("Stage 4 requires all three Stage 3 residual blocks")
    residual_blocks = [
        _verify_file(
            block["checkpoint"],
            block["sha256"],
            "Stage 3 residual block %d" % index,
        )
        for index, block in enumerate(blocks)
    ]
    budget = protocol["candidate_budget"]
    total = int(budget["total_rollouts_per_controller_decision"])
    if (
        total != 600
        or int(budget["rl_iterations"])
        * int(budget["rl_candidates_per_iteration"])
        != total
        or int(budget["standard_iterations"])
        * int(budget["standard_candidates_per_iteration"])
        != total
        or bool(budget.get("additional_candidates_authorized", True))
    ):
        raise ValueError("Stage 4 candidate-budget contract is invalid")

    schedule = build_schedule(protocol, stage3_protocol=stage3_protocol)
    if len(schedule) != int(design["expected_episode_count"]):
        raise ValueError("Stage 4 schedule size differs from preregistration")
    base_config = load_yaml(_resolve(protocol["base_config"]))
    actor_lower = declared_actor_lower
    actor_upper = declared_actor_upper
    controller_lower = np.asarray(
        base_config["action_space"]["lower"], dtype=np.float64
    )
    controller_upper = np.asarray(
        base_config["action_space"]["upper"], dtype=np.float64
    )
    if (
        actor_lower.shape != controller_lower.shape
        or actor_upper.shape != controller_upper.shape
        or np.any(actor_lower < controller_lower - 1.0e-12)
        or np.any(actor_upper > controller_upper + 1.0e-12)
    ):
        raise ValueError("frozen Actor bounds are not a controller subspace")
    for job in schedule:
        config = configure_job(base_config, job, protocol, stage3_protocol)
        validate_job_config(config, job, protocol, stage3_protocol, base_config)
    return {
        "schema_version": 1,
        "protocol_revision": revision,
        "status": "preflight_passed_matrix_not_started",
        "episode_count": len(schedule),
        "sealed_seeds_opened": False,
        "seed_audit": seed_audit,
        "base_config": base_audit,
        "stage3_protocol": stage3_audit,
        "stage3_gate": gate_audit,
        "stage3_sequential_equivalence": equivalence_audit,
        "actor": actor_audit,
        "hss_evidence": hss_evidence,
        "hss_sidecar_members": sidecar_members,
        "residual_blocks": residual_blocks,
        "candidate_budget_per_controller_decision": total,
        "amendment": amendment_audit,
        "implementation_contract": implementation_audit,
    }


def build_schedule(protocol, stage3_protocol=None):
    stage3_protocol = stage3_protocol or _mapping(
        _resolve(protocol["stage3_protocol"])
    )
    seeds = [int(value) for value in protocol["design"]["obstacle_process_seeds"]]
    rng = np.random.RandomState(int(protocol["design"]["schedule_seed"]))
    scheduled = []
    for seed_index in rng.permutation(len(seeds)):
        seed = seeds[int(seed_index)]
        cells = []
        for rl_hss_enabled in (False, True):
            cells.append({
                "condition": "nominal",
                "episode_seed": seed,
                "model_block": -1,
                "checkpoint_seed": -1,
                "rl_hss_enabled": rl_hss_enabled,
            })
        for block_index, block in enumerate(
            stage3_protocol["residual_model_blocks"]
        ):
            for rl_hss_enabled in (False, True):
                cells.append({
                    "condition": "icode_residual",
                    "episode_seed": seed,
                    "model_block": int(block_index),
                    "checkpoint_seed": int(block["seed"]),
                    "rl_hss_enabled": rl_hss_enabled,
                })
        for within_block, index in enumerate(rng.permutation(len(cells))):
            job = dict(cells[int(index)])
            job["within_obstacle_block_order"] = int(within_block)
            job["run_order"] = len(scheduled)
            dynamics = (
                "nominal"
                if job["condition"] == "nominal"
                else "residual_block%d" % int(job["model_block"])
            )
            job["experimental_key"] = "%s::seed%d::rl_hss_%s" % (
                dynamics,
                seed,
                "on" if job["rl_hss_enabled"] else "off",
            )
            scheduled.append(job)
    return scheduled


def _sidecar_mapping(protocol):
    sidecar = protocol["hss"]["sidecar"]
    return {
        "contract": str(sidecar["contract"]),
        "checkpoints": [
            str(member["checkpoint"]) for member in sidecar["members"]
        ],
        "device": str(sidecar["device"]),
        "torch_num_threads": int(sidecar["torch_num_threads"]),
        "use_torchscript": bool(sidecar["use_torchscript"]),
        "ensemble": deepcopy(sidecar["ensemble"]),
    }


def configure_job(base_config, job, protocol, stage3_protocol=None):
    """Build one cell; RL-off delegates exactly to the qualified Stage 3 path."""

    stage3_protocol = stage3_protocol or _mapping(
        _resolve(protocol["stage3_protocol"])
    )
    stage3_job = {
        "condition": str(job["condition"]),
        "episode_seed": int(job["episode_seed"]),
        "model_block": int(job["model_block"]),
    }
    config = configure_stage3_condition(
        base_config, stage3_job, stage3_protocol
    )
    if not bool(job["rl_hss_enabled"]):
        return config

    config = deepcopy(config)
    budget = protocol["candidate_budget"]
    planner = config["planner"]
    paper = deepcopy(protocol["paper_rl_driven"])
    paper["reliability"] = deepcopy(protocol["hss"]["runtime"])
    paper["reliability_sidecar"] = _sidecar_mapping(protocol)
    planner.update({
        "optimizer": "paper_rl_driven",
        "sampling_prior": "paper_direct_rl",
        "num_samples": int(budget["rl_candidates_per_iteration"]),
        "importance_sampling_correction": False,
        "paper_rl_driven": paper,
    })
    if job["condition"] == "icode_residual":
        shield = deepcopy(planner["residual_safety_shield"])
        shield["rl_hss_integration"] = "matched_controllers_v1"
        planner["residual_safety_shield"] = shield
    actor = protocol["actor"]
    config["rl"] = {
        "enabled": True,
        "integration": "paper_direct_control",
        "policy_id": str(actor["policy_id"]),
        "checkpoint": str(actor["checkpoint"]),
        "device": str(actor["device"]),
        "torch_num_threads": int(actor["torch_num_threads"]),
        "allow_actor_action_subspace": True,
    }
    config["experiment"]["name"] += "__rl_hss_on"
    config["experiment"]["stage4_rl_hss_enabled"] = True
    config["experiment"]["rollout_budget_per_controller_decision"] = int(
        budget["total_rollouts_per_controller_decision"]
    )
    return config


def validate_job_config(config, job, protocol, stage3_protocol, base_config):
    """Assert that treatment injection cannot bypass frozen safety contracts."""

    for section in ("task", "state_space", "action_space", "plant", "scene", "sensors", "perception"):
        if config[section] != base_config[section]:
            raise ValueError("Stage 4 changed frozen section: %s" % section)
    if config["perception"].get("scan_guard") != base_config["perception"].get(
        "scan_guard"
    ):
        raise ValueError("Stage 4 changed or bypassed scan_guard")
    planner = config["planner"]
    base_planner = base_config["planner"]
    frozen_planner_fields = {"horizon"}
    frozen_planner_fields.update(
        name
        for name in base_planner
        if str(name).startswith("probabilistic_obstacle_")
    )
    for name in sorted(frozen_planner_fields):
        if planner.get(name) != base_planner.get(name):
            raise ValueError("Stage 4 changed frozen planner field: %s" % name)
    if job["condition"] == "nominal":
        if planner["prediction_mode"] != "nominal" or "checkpoint" in planner:
            raise ValueError("nominal Stage 4 cell contains residual dynamics")
        if "residual_safety_shield" in planner:
            raise ValueError("nominal Stage 4 cell unexpectedly contains shield")
    else:
        block = stage3_protocol["residual_model_blocks"][int(job["model_block"])]
        if (
            planner["prediction_mode"] != "icode_residual"
            or planner["checkpoint"] != block["checkpoint"]
        ):
            raise ValueError("residual Stage 4 cell changed its model block")
        shield = planner.get("residual_safety_shield", {})
        if not bool(shield.get("enabled", False)):
            raise ValueError("residual Stage 4 cell bypassed the shield")
    total = int(protocol["candidate_budget"]["total_rollouts_per_controller_decision"])
    if bool(job["rl_hss_enabled"]):
        paper = planner.get("paper_rl_driven", {})
        if (
            planner.get("optimizer") != "paper_rl_driven"
            or planner.get("sampling_prior") != "paper_direct_rl"
            or int(planner["num_samples"]) * int(paper.get("iterations", 0))
            != total
            or bool(planner.get("importance_sampling_correction", True))
            or not bool(config.get("rl", {}).get("enabled", False))
            or not bool(
                config.get("rl", {}).get("allow_actor_action_subspace", False)
            )
            or not bool(paper.get("reliability", {}).get("enabled", False))
            or not paper.get("reliability_sidecar")
        ):
            raise ValueError("RL/HSS cell violates MPPI or budget contracts")
        if job["condition"] == "icode_residual" and (
            planner["residual_safety_shield"].get("rl_hss_integration")
            != "matched_controllers_v1"
        ):
            raise ValueError("residual RL/HSS cell lacks matched shield controllers")
    else:
        if (
            planner.get("optimizer", "standard") != "standard"
            or planner.get("sampling_prior", "goal_warm_start")
            != "goal_warm_start"
            or int(planner["num_samples"]) != total
            or bool(config.get("rl", {}).get("enabled", False))
        ):
            raise ValueError("RL-off cell no longer reproduces Stage 3")
    return True


def _run_dir(output_dir, job):
    dynamics = (
        "nominal"
        if job["condition"] == "nominal"
        else "residual_block_%d" % int(job["model_block"])
    )
    rl = "rl_hss_on" if job["rl_hss_enabled"] else "rl_hss_off"
    return output_dir / "runs" / dynamics / rl / (
        "seed_%d" % int(job["episode_seed"])
    )


def _complete(run_dir):
    return all((Path(run_dir) / name).is_file() for name in REQUIRED_RUN_FILES)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(protocol_path=DEFAULT_PROTOCOL, *, execute=False, output_override=None, max_jobs=None):
    protocol_path = Path(protocol_path).resolve()
    protocol = _mapping(protocol_path)
    audit = validate_protocol(protocol)
    stage3_protocol = _mapping(_resolve(protocol["stage3_protocol"]))
    schedule = build_schedule(protocol, stage3_protocol)
    if not execute:
        return {**audit, "schedule": schedule, "formal_matrix_started": False}

    output_dir = Path(output_override or _resolve(protocol["output_dir"])).resolve()
    if ROOT not in output_dir.parents:
        raise ValueError("Stage 4 output must remain inside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = _sha256(protocol_path)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        if manifest.get("protocol_sha256") != protocol_hash:
            raise ValueError("cannot resume Stage 4 with a changed protocol")
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
        raise ValueError("cannot resume Stage 4 with a changed schedule")
    _write_json(schedule_path, schedule_payload)

    base_config = load_yaml(_resolve(protocol["base_config"]))
    completed_this_call = 0
    stopped = None
    for job in schedule:
        run_dir = _run_dir(output_dir, job)
        if _complete(run_dir):
            continue
        if max_jobs is not None and completed_this_call >= int(max_jobs):
            break
        config = configure_job(base_config, job, protocol, stage3_protocol)
        validate_job_config(config, job, protocol, stage3_protocol, base_config)
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
