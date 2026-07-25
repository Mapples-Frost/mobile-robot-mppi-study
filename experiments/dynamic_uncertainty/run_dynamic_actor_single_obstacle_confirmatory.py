"""Preflight and execute the frozen single-obstacle confirmatory matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_dynamic_actor_mujoco_probe import (
    _extended_summary,
)
from experiments.dynamic_uncertainty.run_rl_hss_combined_safety_probe import (
    configure_combined_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import _mapping, _resolve
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.obstacles.patrol import validate_v3_config
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_actor_single_obstacle_confirmatory.yaml"
)
PRIMARY_ARMS = ("source_actor", "dynamic_actor")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return value


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be a mapping: %s" % path)
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _repo_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("confirmatory path escaped repository: %s" % path)
    return path


def _split_range(registry, name):
    split = registry["splits"][name]
    start = int(split["seed_start"])
    return range(start, start + int(split["count"]))


def build_schedule(protocol, registry):
    """Build balanced seed blocks without evaluating any trajectory."""

    design = protocol["design"]
    rng = random.Random(int(design["schedule_randomization_seed"]))
    arm_names = list(protocol["arms"])
    ablation_arms = list(design["ablation_arms"])
    if arm_names[:2] != list(PRIMARY_ARMS):
        raise ValueError("primary arm order is not frozen")
    if sorted(arm_names[2:]) != sorted(ablation_arms):
        raise ValueError("ablation arm declaration mismatch")
    blocks = []
    for split_index, split_name in enumerate(("held_out_id", "held_out_ood")):
        split_design = design["splits"][split_name]
        registered = _split_range(registry, split_name)
        start = int(split_design["seed_start"])
        count = int(split_design["seed_count"])
        seeds = list(range(start, start + count))
        if count != 50 or not set(seeds).issubset(set(registered)):
            raise ValueError("confirmatory split must contain 50 registered seeds")
        rng.shuffle(seeds)
        ablation_count = int(split_design["ablation_seed_count"])
        if ablation_count != 25:
            raise ValueError("each split must contain 25 ablation seed blocks")

        latin_base = list(arm_names)
        rng.shuffle(latin_base)
        rotations = list(range(len(latin_base))) * (
            ablation_count // len(latin_base)
        )
        rng.shuffle(rotations)
        for seed, rotation in zip(seeds[:ablation_count], rotations):
            sequence = latin_base[rotation:] + latin_base[:rotation]
            blocks.append({
                "split": split_name,
                "seed": int(seed),
                "ablation_block": True,
                "arm_sequence": sequence,
                "obstacle_noise_profile": (
                    "medium" if split_name == "held_out_id" else "process_shift"
                ),
            })

        remaining = seeds[ablation_count:]
        first_count = len(remaining) // 2 + int(split_index == 0)
        orders = (["source_first"] * first_count) + (
            ["candidate_first"] * (len(remaining) - first_count)
        )
        rng.shuffle(orders)
        for seed, order in zip(remaining, orders):
            sequence = (
                list(PRIMARY_ARMS)
                if order == "source_first"
                else list(reversed(PRIMARY_ARMS))
            )
            blocks.append({
                "split": split_name,
                "seed": int(seed),
                "ablation_block": False,
                "arm_sequence": sequence,
                "obstacle_noise_profile": (
                    "medium" if split_name == "held_out_id" else "process_shift"
                ),
            })
    rng.shuffle(blocks)
    jobs = []
    for block_index, block in enumerate(blocks, start=1):
        block["block_order"] = block_index
        for arm_position, arm in enumerate(block["arm_sequence"], start=1):
            jobs.append({
                "run_order": len(jobs) + 1,
                "block_order": block_index,
                "arm_position": arm_position,
                "split": block["split"],
                "seed": block["seed"],
                "ablation_block": block["ablation_block"],
                "arm": arm,
                "obstacle_noise_profile": block["obstacle_noise_profile"],
            })
    return blocks, jobs


def _verify_bindings(protocol):
    verified = {}
    for name, binding in protocol["bindings"].items():
        path = _repo_path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError("frozen binding missing: %s" % path)
        actual = _sha256(path)
        if actual != str(binding["sha256"]).lower():
            raise ValueError("frozen binding hash mismatch: %s" % name)
        verified[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
        }
    return verified


def validate_protocol(protocol_path):
    protocol_path = _repo_path(protocol_path)
    protocol = _load_yaml(protocol_path)
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("confirmatory protocol is not frozen")
    if protocol.get("sealed_seeds_opened") is not False:
        raise ValueError("sealed-seed boundary is not closed")
    registry = _load_yaml(_repo_path(protocol["seed_registry_path"]))
    sealed = _load_yaml(_repo_path(protocol["sealed_registry_path"]))
    blocks, jobs = build_schedule(protocol, registry)
    if len(blocks) != 100 or len(jobs) != 350:
        raise ValueError("confirmatory schedule size changed")
    seeds = {int(block["seed"]) for block in blocks}
    development = set(_split_range(registry, "development"))
    if seeds & development or seeds & set(int(x) for x in sealed["seeds"]):
        raise ValueError("confirmatory seeds overlap development or sealed seeds")
    if len(seeds) != 100:
        raise ValueError("confirmatory seed blocks are not unique")
    verified = _verify_bindings(protocol)
    id_config = _load_yaml(_repo_path(protocol["domains"]["held_out_id"]["obstacle_process_config"]))
    ood_config = _load_yaml(_repo_path(protocol["domains"]["held_out_ood"]["obstacle_process_config"]))
    validate_v3_config(id_config)
    validate_v3_config(ood_config)
    if ood_config["physical_limits"] != id_config["physical_limits"]:
        raise ValueError("OOD changed frozen physical limits")
    return protocol_path, protocol, registry, blocks, jobs, verified


def _profile_settings(protocol, arm_name):
    arm = protocol["arms"][arm_name]
    profile = deepcopy(protocol["controller_profiles"][arm["controller_profile"]])
    return arm, profile


def configure_arm(protocol, job, base, stage3, stage4):
    arm, settings = _profile_settings(protocol, job["arm"])
    config = configure_combined_job(
        base, stage3, stage4, "combined_veto", int(job["seed"])
    )
    planner = config["planner"]
    paper = planner["paper_rl_driven"]
    mode = str(settings["proposal_gate_mode"])
    if mode in ("shadow", "same_cycle_filter"):
        paper["proposal_advantage_gate"] = {
            "enabled": True,
            "mode": "shadow",
            "relative_disadvantage_margin": 0.0,
            "consecutive_disadvantages": 3,
        }
        paper["standard_fallback_on_advantage_veto"] = False
        paper["same_cycle_guided_cost_filter"] = mode == "same_cycle_filter"
        paper["same_cycle_guided_relative_margin"] = 0.0
    else:
        raise ValueError("unsupported confirmatory proposal gate mode")
    paper["terminal_value_weight"] = 0.0
    paper.update({
        "completion_handover_full_fallback_distance": float(
            settings["completion_handover_full_fallback_distance"]
        ),
        "completion_handover_full_rl_distance": float(
            settings["completion_handover_full_rl_distance"]
        ),
    })
    planner.update({
        "probabilistic_obstacle_emergency_candidate_trigger_ttc_s": float(
            settings["probabilistic_emergency_trigger_ttc_s"]
        ),
        "probabilistic_obstacle_emergency_candidate_intent_hold_steps": int(
            settings["probabilistic_emergency_intent_hold_steps"]
        ),
        "probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled": bool(
            settings["probabilistic_emergency_pareto_forward_commit"]
        ),
    })
    config["perception"]["scan_guard"].update({
        "dynamic_escape_use_vetted_planner_control": bool(
            settings["vetted_reactive_escape"]
        ),
        "dynamic_escape_min_probability_mass_relative_improvement": float(
            settings["dynamic_escape_min_probability_mass_relative_improvement"]
        ),
    })
    domain = protocol["domains"][job["split"]]
    motion = config["scene"]["obstacles"][0]["motion"]
    motion["config_path"] = str(
        _repo_path(domain["obstacle_process_config"])
    )
    motion["noise_profile"] = str(job["obstacle_noise_profile"])
    checkpoint = _repo_path(arm["checkpoint"])
    config["rl"]["checkpoint"] = str(checkpoint)
    config["rl"]["policy_id"] = str(job["arm"])
    config["experiment"].update({
        "name": "single_obstacle_confirmatory__%s__seed%d__%s"
        % (job["split"], int(job["seed"]), job["arm"]),
        "single_obstacle_confirmatory": True,
        "confirmatory_split": str(job["split"]),
        "confirmatory_arm": str(job["arm"]),
        "sealed_seeds_opened": False,
    })
    config["scope_guards"].update({
        "development_seeds_only": False,
        "held_out_confirmatory": True,
        "simulator_truth_for_control": False,
        "exact_ground_truth_pose_for_control": False,
        "sealed_registry_imported": False,
    })
    if len(config["scene"]["obstacles"]) != 1:
        raise ValueError("confirmatory scene must contain exactly one obstacle")
    if int(planner["horizon"]) != 36:
        raise ValueError("confirmatory horizon changed")
    total = int(planner["num_samples"]) * int(paper["iterations"])
    if total != int(protocol["design"]["total_rollouts_per_controller_decision"]):
        raise ValueError("confirmatory rollout budget changed")
    return config


def _run_dir(output, job):
    return (
        output
        / "runs"
        / str(job["split"])
        / ("seed_%d" % int(job["seed"]))
        / str(job["arm"])
    )


def _completed(run_dir):
    return (run_dir / "metrics.json").is_file() and (
        run_dir / "provenance.json"
    ).is_file()


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run(protocol_path=DEFAULT_PROTOCOL, *, execute=False, resume=False):
    (
        protocol_path,
        protocol,
        _registry,
        blocks,
        jobs,
        verified,
    ) = validate_protocol(protocol_path)
    protocol_sha = _sha256(protocol_path)
    preflight = {
        "status": "preflight_passed",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "seed_blocks": len(blocks),
        "episode_jobs": len(jobs),
        "primary_pairs": 100,
        "ablation_pairs_per_contrast": 50,
        "verified_bindings": verified,
        "sealed_seeds_opened": False,
    }
    if not execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return preflight

    output = _repo_path(protocol["output_dir"])
    snapshot_path = output / "protocol_snapshot.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("confirmatory output already exists; use no overwrite")
    if resume:
        if not snapshot_path.is_file():
            raise ValueError("resume requires an existing protocol snapshot")
        snapshot = _load_json(snapshot_path)
        if snapshot.get("protocol_sha256") != protocol_sha:
            raise ValueError("resume protocol hash mismatch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(snapshot_path, {
            "protocol_path": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": protocol_sha,
            "protocol": protocol,
        })
        _write_json(output / "schedule.json", {
            "schedule_randomization_seed": protocol["design"]["schedule_randomization_seed"],
            "blocks": blocks,
            "jobs": jobs,
        })
        _write_json(output / "environment_manifest.json", {
            "git_head": _git_head(),
            "platform": "windows_native",
            "python": sys.executable,
            "protocol_sha256": protocol_sha,
            "verified_bindings": verified,
            "sealed_seeds_opened": False,
        })

    base = load_yaml(_repo_path(protocol["base_config"]))
    stage3 = _mapping(_repo_path(protocol["stage3_protocol"]))
    stage4 = _mapping(_repo_path(protocol["stage4_protocol"]))
    completed = 0
    for job in jobs:
        run_dir = _run_dir(output, job)
        if _completed(run_dir):
            if not resume:
                raise ValueError("unexpected pre-existing complete arm")
            completed += 1
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(
                "incomplete arm artifact retained; automatic retry forbidden: %s"
                % run_dir
            )
        config = configure_arm(protocol, job, base, stage3, stage4)
        print(json.dumps({"stage": "confirmatory_episode", **job}, sort_keys=True), flush=True)
        ExperimentRunner(config, ROOT, output_dir=run_dir, headless=True).run()
        if not _completed(run_dir):
            raise RuntimeError("confirmatory episode did not produce complete artifacts")
        completed += 1
        _write_json(output / "progress.json", {
            "status": "running" if completed < len(jobs) else "raw_complete",
            "completed_episode_jobs": completed,
            "total_episode_jobs": len(jobs),
            "last_job": job,
            "protocol_sha256": protocol_sha,
        })

    rows = []
    for job in jobs:
        metrics = _load_json(_run_dir(output, job) / "metrics.json")
        rows.append({**job, "summary": _extended_summary(metrics)})
    _write_json(output / "raw_results.json", {
        "schema_version": 1,
        "status": "raw_complete",
        "scope": "single_dynamic_obstacle_confirmatory",
        "protocol_sha256": protocol_sha,
        "sealed_seeds_opened": False,
        "rows": rows,
    })
    result = dict(preflight)
    result.update({"status": "raw_complete", "output_dir": str(output.relative_to(ROOT))})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.protocol, execute=bool(args.execute), resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
