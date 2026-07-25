"""Freeze, qualify, seal, and execute the single-obstacle paper-v4 matrix.

The formal registry cannot be created until a complete qualification report
exists.  A seed is eligible only when a pre-controller ghost trajectory has a
physical overlap with the obstacle truth; controller outcomes are never used
for seed selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import traceback
import platform

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for _value in (ROOT, ROOT / "src"):
    if str(_value) not in sys.path:
        sys.path.insert(0, str(_value))

from experiments.dynamic_uncertainty.run_dynamic_actor_single_obstacle_confirmatory import (
    _profile_settings,
)
from experiments.dynamic_uncertainty.run_rl_hss_combined_safety_probe import (
    configure_combined_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    _mapping,
    configure_job as configure_stage4_job,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import generate_patrol_trajectory, validate_v3_config
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = ROOT / "configs/research/single_dynamic_obstacle_paper_v4.yaml"
REQUIRED_RUN_FILES = ("metrics.json", "provenance.json")


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


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repo_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _output_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _complete(run_dir):
    return all((Path(run_dir) / name).is_file() for name in REQUIRED_RUN_FILES)


def _verify_checkpoint_bindings(protocol):
    verified = {}
    bindings = (
        [protocol["checkpoints"]["actor"]]
        + list(protocol["checkpoints"]["residual_blocks"])
        + list(protocol["checkpoints"]["hss_sidecars"])
    )
    for binding in bindings:
        path = _repo_path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError("checkpoint missing: %s" % path)
        actual = _sha256(path)
        expected = str(binding["sha256"]).lower()
        if actual != expected:
            raise ValueError("checkpoint SHA-256 mismatch: %s" % path)
        verified[str(path.relative_to(ROOT))] = actual
    return verified


def validate_protocol(protocol_path=DEFAULT_PROTOCOL):
    protocol_path = _repo_path(protocol_path)
    protocol = _load_yaml(protocol_path)
    if protocol.get("status") != "frozen_before_qualification":
        raise ValueError("paper-v4 protocol is not frozen before qualification")
    design = protocol["design"]
    if (
        int(design["core_seed_count"]) != 360
        or int(design["ablation_seed_count"]) != 140
        or int(design["total_episode_jobs"]) != 1860
    ):
        raise ValueError("paper-v4 episode matrix changed")
    if len(design["core_arms"]) != 4 or len(design["ablation_arms"]) != 3:
        raise ValueError("paper-v4 arm count changed")
    if int(design["total_rollouts_per_decision"]) != 600:
        raise ValueError("rollout budget changed")
    id_config = _load_yaml(_repo_path(protocol["id_obstacle_process"]))
    ood_config = _load_yaml(_repo_path(protocol["ood_obstacle_process"]))
    validate_v3_config(id_config)
    validate_v3_config(ood_config)
    if id_config["physical_limits"] != ood_config["physical_limits"]:
        raise ValueError("OOD configuration changed frozen physical limits")
    verified = _verify_checkpoint_bindings(protocol)
    return protocol_path, protocol, id_config, ood_config, verified


def conflict_certificate(seed, split, protocol, obstacle_config):
    """Return a controller-independent spatiotemporal collision certificate."""

    contract = protocol["conflict_certificate"]
    profile_name = "medium" if split == "id" else "process_shift"
    noise = noise_profiles_from_mapping(obstacle_config["noise_profiles"])[
        profile_name
    ]
    trajectory = generate_patrol_trajectory(
        str(contract["process"]), int(seed), noise, obstacle_config
    )
    start = np.asarray(contract["robot_start_xy"], dtype=np.float64)
    goal = np.asarray(contract["robot_goal_xy"], dtype=np.float64)
    displacement = goal - start
    distance = float(np.linalg.norm(displacement))
    direction = displacement / distance
    traveled = np.minimum(
        float(contract["ghost_speed_mps"]) * trajectory.times, distance
    )
    ghost = start[None, :] + traveled[:, None] * direction[None, :]
    separations = np.linalg.norm(trajectory.states[:, :2] - ghost, axis=1)
    collision_threshold = float(contract["collision_distance_m"])
    conflict_threshold = float(contract["conflict_window_distance_m"])
    collision_indices = np.flatnonzero(separations <= collision_threshold + 1e-12)
    if collision_indices.size == 0:
        return None
    conflict_indices = np.flatnonzero(separations <= conflict_threshold + 1e-12)
    intervals = []
    if conflict_indices.size:
        groups = np.split(
            conflict_indices, np.flatnonzero(np.diff(conflict_indices) > 1) + 1
        )
        padding = float(contract["window_padding_s"])
        raw = [
            [
                max(0.0, float(trajectory.times[group[0]]) - padding),
                min(float(trajectory.times[-1]), float(trajectory.times[group[-1]]) + padding),
            ]
            for group in groups
        ]
        merge_gap = float(contract["merge_gap_s"])
        for begin, end in raw:
            if intervals and begin - intervals[-1][1] <= merge_gap + 1e-12:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([begin, end])
    minimum_index = int(np.argmin(separations))
    truth_digest = hashlib.sha256(
        np.asarray(trajectory.states, dtype="<f8").tobytes()
    ).hexdigest()
    return {
        "seed": int(seed),
        "split": str(split),
        "obstacle_process": str(contract["process"]),
        "noise_profile": profile_name,
        "minimum_ghost_obstacle_center_distance_m": float(separations[minimum_index]),
        "minimum_distance_time_s": float(trajectory.times[minimum_index]),
        "collision_threshold_m": collision_threshold,
        "conflict_window_threshold_m": conflict_threshold,
        "conflict_windows_s": intervals,
        "obstacle_truth_sha256": truth_digest,
        "certificate_rule": "ghost_constant_speed_center_distance_le_sum_radii",
    }


def _first_certified(start, count, needed, split, protocol, obstacle_config):
    selected = []
    for seed in range(int(start), int(start) + int(count)):
        certificate = conflict_certificate(seed, split, protocol, obstacle_config)
        if certificate is not None:
            selected.append(certificate)
            if len(selected) == int(needed):
                break
    if len(selected) != int(needed):
        raise RuntimeError(
            "%s conflict pool yielded %d/%d certified seeds"
            % (split, len(selected), int(needed))
        )
    return selected


def build_qualification_registry(protocol, id_config, ood_config):
    q = protocol["qualification"]
    needed = int(q["seeds_per_split"])
    certificates = {}
    for split, config in (("id", id_config), ("ood", ood_config)):
        certificates[split] = _first_certified(
            q["seed_candidate_start"][split],
            q["seed_candidate_count"],
            needed,
            split,
            protocol,
            config,
        )
    return {
        "schema_version": 1,
        "scope": "paper_v4_qualification_only",
        "outcome_selection_used": False,
        "splits": certificates,
    }


def _cyclic_sequence(arms, offset):
    values = list(arms)
    offset = int(offset) % len(values)
    return values[offset:] + values[:offset]


def build_schedule(registry, protocol, qualification=False):
    design = protocol["design"]
    core = list(design["core_arms"])
    ablations = list(design["ablation_arms"])
    rng = random.Random(int(design["schedule_seed"]) + int(qualification))
    blocks = []
    for split_index, split in enumerate(("id", "ood")):
        entries = list(registry["splits"][split])
        if qualification:
            ablation_seeds = {int(item["seed"]) for item in entries}
        else:
            ablation_seeds = set(
                int(value) for value in registry["ablation_subset"][split]
            )
        base4 = core[:]
        base7 = core + ablations
        rng.shuffle(base4)
        rng.shuffle(base7)
        rotation_counters = {4: 0, 7: 0}
        for index, item in enumerate(entries):
            seed = int(item["seed"])
            arms = base7 if seed in ablation_seeds else base4
            offset = rotation_counters[len(arms)] + (
                0 if split_index == 0 else len(arms) // 2
            )
            rotation_counters[len(arms)] += 1
            blocks.append({
                "split": split,
                "seed": seed,
                "model_block": (index + split_index) % 3,
                "ablation_block": seed in ablation_seeds,
                "arm_sequence": _cyclic_sequence(arms, offset),
                "certificate": item,
            })
    rng.shuffle(blocks)
    for index, block in enumerate(blocks, start=1):
        block["block_order"] = index
    return blocks


def _copy_probability_template(config, template):
    planner = config["planner"]
    for key, value in template["planner"].items():
        if str(key).startswith("probabilistic_obstacle_"):
            planner[key] = deepcopy(value)
    scan = config["perception"]["scan_guard"]
    for key, value in template["perception"]["scan_guard"].items():
        if str(key).startswith("dynamic_"):
            scan[key] = deepcopy(value)
    config["perception"]["dynamic_obstacle_tracker"] = deepcopy(
        template["perception"]["dynamic_obstacle_tracker"]
    )
    config["perception"]["temporal_scan_guard"] = deepcopy(
        template["perception"]["temporal_scan_guard"]
    )


def _set_probability_enabled(config, enabled, predictor):
    tracker = config["perception"]["dynamic_obstacle_tracker"]
    tracker["enabled"] = bool(enabled)
    tracker["predictor_mode"] = (
        str(predictor) if bool(enabled) else "change_aware"
    )
    config["perception"]["temporal_scan_guard"]["enabled"] = bool(enabled)
    for key in list(config["planner"]):
        if str(key).startswith("probabilistic_obstacle_") and str(key).endswith(
            "_enabled"
        ):
            config["planner"][key] = bool(enabled)
    config["planner"]["probabilistic_obstacle_risk_enabled"] = bool(enabled)
    scan = config["perception"]["scan_guard"]
    for key in list(scan):
        if str(key).startswith("dynamic_") and str(key).endswith("_enabled"):
            scan[key] = bool(enabled)


def configure_arm(protocol, block, arm_name, base, stage3, stage4):
    contract = protocol["arm_contracts"][arm_name]
    learning = bool(contract["learning"])
    icode = bool(contract["icode"])
    job = {
        "condition": "icode_residual" if icode else "nominal",
        "episode_seed": int(block["seed"]),
        "model_block": int(block["model_block"]) if icode else -1,
        "rl_hss_enabled": learning,
    }
    config = configure_stage4_job(base, job, stage4, stage3)
    probability_template = configure_combined_job(
        base, stage3, stage4, "combined_veto", int(block["seed"])
    )
    _copy_probability_template(config, probability_template)
    if learning:
        paper = config["planner"]["paper_rl_driven"]
        template_paper = probability_template["planner"]["paper_rl_driven"]
        for key in (
            "proposal_advantage_gate",
            "standard_fallback_on_advantage_veto",
            "same_cycle_guided_cost_filter",
            "same_cycle_guided_relative_margin",
        ):
            if key in template_paper:
                paper[key] = deepcopy(template_paper[key])
        paper["proposal_advantage_gate"] = {
            "enabled": True,
            "mode": "shadow",
            "relative_disadvantage_margin": 0.0,
            "consecutive_disadvantages": 3,
        }
        paper["standard_fallback_on_advantage_veto"] = False
        paper["same_cycle_guided_cost_filter"] = True
        paper["same_cycle_guided_relative_margin"] = 0.0
        paper["terminal_value_weight"] = 0.0
        paper["completion_handover_full_fallback_distance"] = 0.4
        paper["completion_handover_full_rl_distance"] = 0.8
        actor = protocol["checkpoints"]["actor"]
        config["rl"]["checkpoint"] = str(_repo_path(actor["path"]))
        config["rl"]["policy_id"] = "dynamic_actor_v5a6_u000250"
        if contract["hss"] == "fixed_0.30":
            paper["reliability"]["enabled"] = True
            paper["reliability"]["low_guided_fraction"] = 0.30
            paper["reliability"]["medium_guided_fraction"] = 0.30
            paper["reliability"]["high_guided_fraction"] = 0.30
            paper["guided_fraction"] = 0.30
    if icode:
        shield = config["planner"]["residual_safety_shield"]
        shield["deterministic_clearance_fallback_enabled"] = True
        shield["maximum_nominal_clearance_regression_m"] = 0.0
    _set_probability_enabled(
        config, bool(contract["probability"]), contract["predictor"]
    )
    domain_path = (
        protocol["id_obstacle_process"]
        if block["split"] == "id"
        else protocol["ood_obstacle_process"]
    )
    motion = config["scene"]["obstacles"][0]["motion"]
    motion["config_path"] = str(_repo_path(domain_path))
    motion["noise_profile"] = "medium" if block["split"] == "id" else "process_shift"
    config["planner"]["device"] = "cpu"
    config["planner"]["residual_cuda_graph_enabled"] = False
    config["planner"]["residual_device_rollout_enabled"] = False
    if learning:
        config["rl"]["device"] = "cpu"
        config["rl"]["torch_num_threads"] = 1
        sidecar = config["planner"]["paper_rl_driven"].get(
            "reliability_sidecar", {}
        )
        sidecar["device"] = "cpu"
        sidecar["torch_num_threads"] = 1
    config["experiment"].update({
        "name": "paper_v4__%s__seed%d__%s"
        % (block["split"], int(block["seed"]), arm_name),
        "seed": int(block["seed"]),
        "paper_v4_arm": arm_name,
        "paper_v4_split": block["split"],
        "paper_v4_model_block": int(block["model_block"]),
    })
    config["scope_guards"].update({
        "development_seeds_only": False,
        "paper_v4": True,
        "simulator_truth_for_control": False,
        "exact_ground_truth_pose_for_control": False,
    })
    if len(config["scene"]["obstacles"]) != 1:
        raise ValueError("paper-v4 requires exactly one moving obstacle")
    paper_iterations = int(
        config["planner"].get("paper_rl_driven", {}).get("iterations", 1)
    )
    total = int(config["planner"]["num_samples"]) * paper_iterations
    if total != int(protocol["design"]["total_rollouts_per_decision"]):
        raise ValueError("arm %s changed the 600-rollout budget" % arm_name)
    return config


def _flatten(value, prefix=""):
    rows = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            rows.update(_flatten(value[key], path))
    elif isinstance(value, (list, tuple)):
        rows[prefix] = json.loads(json.dumps(value))
    else:
        rows[prefix] = value
    return rows


def _change_signature(before, after):
    left, right = _flatten(before), _flatten(after)
    ignored = (
        "experiment.name",
        "experiment.paper_v4_arm",
        "scope_guards.",
    )
    result = {}
    for key in sorted(set(left) | set(right)):
        if key in ignored or any(key.startswith(item) for item in ignored if item.endswith(".")):
            continue
        pair = (left.get(key, "<MISSING>"), right.get(key, "<MISSING>"))
        if pair[0] != pair[1]:
            result[key] = pair
    return result


def factor_separability_audit(protocol):
    base = load_yaml(_repo_path(protocol["base_config"]))
    stage3 = _mapping(_repo_path(protocol["stage3_protocol"]))
    stage4 = _mapping(_repo_path(protocol["stage4_protocol"]))
    block = {"split": "id", "seed": 730100001, "model_block": 0}
    configs = {
        arm: configure_arm(protocol, block, arm, base, stage3, stage4)
        for arm in protocol["design"]["core_arms"]
    }
    learning_without_p = _change_signature(
        configs["B00_strong_nominal_mppi"], configs["B10_learning_only"]
    )
    learning_with_p = _change_signature(
        configs["B01_probability_only"], configs["B11_full_proposed"]
    )
    probability_without_l = _change_signature(
        configs["B00_strong_nominal_mppi"], configs["B01_probability_only"]
    )
    probability_with_l = _change_signature(
        configs["B10_learning_only"], configs["B11_full_proposed"]
    )
    passed = (
        learning_without_p == learning_with_p
        and probability_without_l == probability_with_l
    )
    audit = {
        "status": "pass" if passed else "fail",
        "learning_delta_sha256": _canonical_sha256(learning_without_p),
        "probability_delta_sha256": _canonical_sha256(probability_without_l),
        "learning_changed_paths": sorted(learning_without_p),
        "probability_changed_paths": sorted(probability_without_l),
    }
    if not passed:
        audit["learning_factor_mismatch"] = {
            "without_probability": learning_without_p,
            "with_probability": learning_with_p,
        }
        audit["probability_factor_mismatch"] = {
            "without_learning": probability_without_l,
            "with_learning": probability_with_l,
        }
        raise ValueError("L/P factor separability audit failed")
    return audit


def component_construction_audit(protocol):
    """Instantiate and run one decision for every arm before qualification."""

    import gc

    base = load_yaml(_repo_path(protocol["base_config"]))
    stage3 = _mapping(_repo_path(protocol["stage3_protocol"]))
    stage4 = _mapping(_repo_path(protocol["stage4_protocol"]))
    block = {"split": "id", "seed": 730100001, "model_block": 0}
    rows = []
    with tempfile.TemporaryDirectory(prefix="paper_v4_preflight_") as temporary:
        for arm in protocol["arm_contracts"]:
            config = configure_arm(protocol, block, arm, base, stage3, stage4)
            config["experiment"]["max_steps"] = 1
            run_dir = Path(temporary) / arm
            runner = ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True
            )
            controller_type = type(runner.components["controller"]).__name__
            prediction_mode = runner.components["prediction_mode"]
            runner.run()
            if not _complete(run_dir):
                raise RuntimeError(
                    "one-step component preflight incomplete: %s" % arm
                )
            metrics = _load_json(run_dir / "metrics.json")
            if bool(protocol["arm_contracts"][arm]["learning"]):
                if abs(float(metrics.get("paper_total_rollouts_mean", 0.0)) - 600.0) > 1e-9:
                    raise RuntimeError(
                        "one-step preflight changed learned rollout budget: %s"
                        % arm
                    )
            rows.append({
                "arm": arm,
                "controller_type": controller_type,
                "prediction_mode": prediction_mode,
                "one_step_runtime": "pass",
            })
            del runner
            gc.collect()
    return {"status": "pass", "arms": rows}


def _run_dir(output, block, arm):
    return (
        Path(output)
        / "runs"
        / str(block["split"])
        / ("seed_%d" % int(block["seed"]))
        / str(arm)
    )


def _configure_worker(worker_count, logical_cpu_count, cpus_per_worker):
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
    affinity = []
    try:
        import multiprocessing
        import psutil
        identity = multiprocessing.current_process()._identity
        index = (int(identity[0]) - 1) % int(worker_count) if identity else 0
        start = (index * int(cpus_per_worker)) % int(logical_cpu_count)
        affinity = [
            (start + offset) % int(logical_cpu_count)
            for offset in range(int(cpus_per_worker))
        ]
        psutil.Process().cpu_affinity(affinity)
    except Exception:
        affinity = []
    return affinity


def _run_block(payload):
    protocol_path, block, output, worker_count = payload
    _, protocol, _, _, _ = validate_protocol(protocol_path)
    execution = protocol["execution"]
    affinity = _configure_worker(
        worker_count,
        execution["logical_cpu_count"],
        execution["logical_cpus_per_worker"],
    )
    base = load_yaml(_repo_path(protocol["base_config"]))
    stage3 = _mapping(_repo_path(protocol["stage3_protocol"]))
    stage4 = _mapping(_repo_path(protocol["stage4_protocol"]))
    started = time.perf_counter()
    completed = []
    for arm in block["arm_sequence"]:
        run_dir = _run_dir(output, block, arm)
        if _complete(run_dir):
            completed.append(arm)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                "incomplete artifact retained; automatic retry forbidden: %s"
                % run_dir
            )
        config = configure_arm(protocol, block, arm, base, stage3, stage4)
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "paper_v4_job.json", {
            "block": block,
            "arm": arm,
            "resolved_config_sha256": _canonical_sha256(config),
            "worker_affinity": affinity,
        })
        ExperimentRunner(config, ROOT, output_dir=run_dir, headless=True).run()
        if not _complete(run_dir):
            raise RuntimeError("episode artifacts incomplete: %s" % run_dir)
        completed.append(arm)
    rss = None
    try:
        import psutil
        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    return {
        "split": block["split"],
        "seed": int(block["seed"]),
        "arms_completed": completed,
        "elapsed_s": time.perf_counter() - started,
        "worker_affinity": affinity,
        "worker_rss_bytes": rss,
    }


def execute_blocks(protocol_path, protocol, blocks, output, workers, scope):
    output = _output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "schedule.json"
    schedule_payload = {
        "schema_version": 1,
        "scope": scope,
        "workers": int(workers),
        "blocks": blocks,
        "schedule_sha256": _canonical_sha256(blocks),
    }
    if schedule_path.is_file() and _load_json(schedule_path) != schedule_payload:
        raise ValueError("existing output has a different frozen schedule")
    _write_json(schedule_path, schedule_payload)
    remaining = [
        block
        for block in blocks
        if not all(_complete(_run_dir(output, block, arm)) for arm in block["arm_sequence"])
    ]
    completed = len(blocks) - len(remaining)
    failures = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(
                _run_block,
                (str(protocol_path), block, str(output), int(workers)),
            ): block
            for block in remaining
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                result = future.result()
                completed += 1
                print(
                    "[block %d/%d] %s seed=%d complete (%d arms, %.1fs)"
                    % (
                        completed,
                        len(blocks),
                        block["split"],
                        int(block["seed"]),
                        len(result["arms_completed"]),
                        float(result["elapsed_s"]),
                    ),
                    flush=True,
                )
                _write_json(
                    output / "worker_reports" / ("block_%04d.json" % block["block_order"]),
                    result,
                )
            except Exception as error:
                detail = "".join(traceback.format_exception(
                    type(error), error, error.__traceback__
                ))
                print(
                    "[block failure] %s seed=%d:\n%s"
                    % (block["split"], int(block["seed"]), detail),
                    flush=True,
                )
                failures.append({
                    "block_order": block["block_order"],
                    "split": block["split"],
                    "seed": block["seed"],
                    "error": detail,
                })
            _write_json(output / "progress.json", {
                "status": "failed" if failures else (
                    "complete" if completed == len(blocks) else "running"
                ),
                "completed_blocks": completed,
                "total_blocks": len(blocks),
                "completed_episode_jobs": sum(
                    len(item["arm_sequence"])
                    for item in blocks
                    if all(_complete(_run_dir(output, item, arm)) for arm in item["arm_sequence"])
                ),
                "total_episode_jobs": sum(len(item["arm_sequence"]) for item in blocks),
                "failures": failures,
                "outcomes_opened": False,
            })
    if failures:
        raise RuntimeError("%d seed blocks failed; see progress.json" % len(failures))
    return output


def qualification_audit(protocol, blocks, output):
    output = Path(output)
    missing = []
    integrity_failures = []
    for block in blocks:
        for arm in block["arm_sequence"]:
            run_dir = _run_dir(output, block, arm)
            if not _complete(run_dir):
                missing.append({"seed": block["seed"], "arm": arm})
                continue
            metrics = _load_json(run_dir / "metrics.json")
            if arm in (
                "B10_learning_only",
                "B11_full_proposed",
                "A_full_ordinary_imm",
                "A_no_icode",
                "A_fixed_hss",
            ):
                rollouts = float(metrics.get("paper_total_rollouts_mean", 0.0))
                if abs(rollouts - 600.0) > 1e-9:
                    integrity_failures.append({
                        "seed": block["seed"], "arm": arm,
                        "reason": "paper_total_rollouts_mean", "value": rollouts,
                    })
            for required in ("success", "collision", "steps", "final_goal_distance"):
                if required not in metrics:
                    integrity_failures.append({
                        "seed": block["seed"], "arm": arm,
                        "reason": "missing_metric", "value": required,
                    })
    reports = [
        _load_json(path)
        for path in sorted((output / "worker_reports").glob("block_*.json"))
    ]
    peak_rss = max(
        (int(item["worker_rss_bytes"]) for item in reports if item.get("worker_rss_bytes") is not None),
        default=0,
    )
    status = "pass" if not missing and not integrity_failures else "fail"
    report = {
        "schema_version": 1,
        "status": status,
        "scope": "engineering_qualification_not_effect_estimation",
        "seed_blocks": len(blocks),
        "episode_jobs": sum(len(block["arm_sequence"]) for block in blocks),
        "missing": missing,
        "integrity_failures": integrity_failures,
        "peak_worker_rss_bytes": peak_rss,
        "outcomes_used_for_seed_selection": False,
        "formal_registry_authorized": status == "pass",
    }
    _write_json(output / "qualification_report.json", report)
    return report


def seal_registry(protocol_path, protocol, id_config, ood_config):
    qualification_output = _output_path(protocol["qualification_output"])
    report_path = qualification_output / "qualification_report.json"
    if not report_path.is_file() or _load_json(report_path).get("status") != "pass":
        raise ValueError("formal registry requires a passing qualification report")
    selection = protocol["formal_seed_selection"]
    needed = int(protocol["design"]["core_seed_count_per_split"])
    splits = {
        "id": _first_certified(
            selection["id_candidate_start"], selection["candidate_count_per_split"],
            needed, "id", protocol, id_config,
        ),
        "ood": _first_certified(
            selection["ood_candidate_start"], selection["candidate_count_per_split"],
            needed, "ood", protocol, ood_config,
        ),
    }
    rng = random.Random(int(protocol["design"]["ablation_subset_seed"]))
    subset = {}
    for split in ("id", "ood"):
        seeds = [int(item["seed"]) for item in splits[split]]
        subset[split] = sorted(rng.sample(
            seeds, int(protocol["design"]["ablation_seed_count_per_split"])
        ))
    registry = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "sealed_before_formal_execution",
        "selection_used_controller_outcomes": False,
        "selection_rule": selection["selection_rule"],
        "splits": splits,
        "ablation_subset": subset,
        "ablation_subset_sha256": _canonical_sha256(subset),
    }
    blocks = build_schedule(registry, protocol, qualification=False)
    registry["schedule"] = blocks
    registry["schedule_sha256"] = _canonical_sha256(blocks)
    if sum(len(block["arm_sequence"]) for block in blocks) != 1860:
        raise RuntimeError("sealed schedule is not 1860 episodes")
    path = _repo_path(protocol["formal_registry"])
    if path.exists():
        raise FileExistsError("formal registry already exists: %s" % path)
    _write_yaml(path, registry)
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        "%s  %s\n" % (digest, path.name), encoding="ascii"
    )
    return {"registry": str(path), "sha256": digest, "episode_jobs": 1860}


def _execution_manifest(protocol_path, protocol, verified, registry_path=None):
    files = [
        protocol_path,
        ROOT / protocol["preregistration"],
        ROOT / protocol["analysis_amendment"],
        Path(__file__),
        ROOT / "experiments/dynamic_uncertainty/analyze_single_dynamic_obstacle_paper_v4.py",
        _repo_path(protocol["base_config"]),
        _repo_path(protocol["stage3_protocol"]),
        _repo_path(protocol["stage4_protocol"]),
        _repo_path(protocol["id_obstacle_process"]),
        _repo_path(protocol["ood_obstacle_process"]),
    ]
    # Freeze the complete executable Python closure rather than a hand-written
    # subset.  The paper runner imports configuration helpers from several
    # experiment modules, and ExperimentRunner reaches metrics, perception,
    # simulation and safety code indirectly.  Hashing the trees prevents an
    # unlisted transitive dependency from changing between formal blocks.
    files.extend(sorted((ROOT / "src/mobile_robot_mppi").rglob("*.py")))
    files.extend(sorted(
        (ROOT / "experiments/dynamic_uncertainty").glob("*.py")
    ))
    files.extend(sorted(
        (ROOT / "mppi_hardware_bridge/scripts").glob("*.py")
    ))
    for external in (
        Path("C:/Research/bootstrap/bootstrap_status.json"),
        Path("C:/Research/bootstrap/python_inventory.json"),
        Path("C:/Research/bootstrap/windows_inventory.json"),
        Path("C:/Research/bootstrap/pip_freeze.txt"),
    ):
        if external.is_file():
            files.append(external)
    if registry_path is not None:
        files.append(Path(registry_path))
    hashes = {}
    for path in dict.fromkeys(Path(item).resolve() for item in files):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError("manifest binding missing: %s" % path)
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        hashes[name] = _sha256(path)
    hashes.update(verified)
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    for module_name in ("numpy", "scipy", "torch", "mujoco", "psutil", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = None
    return {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "files": hashes,
        "manifest_sha256": _canonical_sha256(hashes),
        "runtime": versions,
        "runtime_sha256": _canonical_sha256(versions),
        "formal_experiment_started": False,
    }


def preflight(protocol_path=DEFAULT_PROTOCOL):
    protocol_path, protocol, id_config, ood_config, verified = validate_protocol(protocol_path)
    qualification_registry = build_qualification_registry(protocol, id_config, ood_config)
    blocks = build_schedule(qualification_registry, protocol, qualification=True)
    factor_audit = factor_separability_audit(protocol)
    component_audit = component_construction_audit(protocol)
    manifest = _execution_manifest(protocol_path, protocol, verified)
    result = {
        "status": "preflight_pass",
        "qualification_seed_blocks": len(blocks),
        "qualification_episode_jobs": sum(len(block["arm_sequence"]) for block in blocks),
        "factor_separability": factor_audit,
        "component_construction": component_audit,
        "execution_manifest": manifest,
        "formal_experiment_started": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_qualification(protocol_path, workers):
    protocol_path, protocol, id_config, ood_config, verified = validate_protocol(protocol_path)
    registry = build_qualification_registry(protocol, id_config, ood_config)
    blocks = build_schedule(registry, protocol, qualification=True)
    output = _output_path(protocol["qualification_output"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "qualification_registry.json", registry)
    _write_json(output / "factor_separability.json", factor_separability_audit(protocol))
    _write_json(output / "execution_manifest.json", _execution_manifest(protocol_path, protocol, verified))
    execute_blocks(protocol_path, protocol, blocks, output, workers, "paper_v4_qualification")
    report = qualification_audit(protocol, blocks, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise RuntimeError("paper-v4 qualification failed")
    return report


def run_formal(protocol_path, workers):
    protocol_path, protocol, _, _, verified = validate_protocol(protocol_path)
    registry_path = _repo_path(protocol["formal_registry"])
    if not registry_path.is_file():
        raise FileNotFoundError("sealed formal registry is missing")
    registry = _load_yaml(registry_path)
    if registry.get("status") != "sealed_before_formal_execution":
        raise ValueError("formal registry is not sealed")
    blocks = registry["schedule"]
    if _canonical_sha256(blocks) != registry["schedule_sha256"]:
        raise ValueError("sealed schedule hash mismatch")
    output = _output_path(protocol["formal_output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "execution_manifest.json"
    manifest = _execution_manifest(protocol_path, protocol, verified, registry_path)
    manifest["formal_experiment_started"] = True
    if manifest_path.is_file() and _load_json(manifest_path) != manifest:
        raise ValueError("formal execution manifest changed")
    _write_json(manifest_path, manifest)
    return execute_blocks(
        protocol_path, protocol, blocks, output, workers, "paper_v4_formal"
    )


def status(protocol_path=DEFAULT_PROTOCOL):
    _, protocol, _, _, _ = validate_protocol(protocol_path)
    payload = {}
    for name, output in (
        ("qualification", protocol["qualification_output"]),
        ("formal", protocol["formal_output"]),
    ):
        progress = _output_path(output) / "progress.json"
        payload[name] = _load_json(progress) if progress.is_file() else {
            "status": "not_started"
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "qualify", "seal", "formal", "status")
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.workers not in (8, 12, 16):
        parser.error("--workers must be 8, 12, or 16")
    if args.command == "preflight":
        preflight(args.protocol)
    elif args.command == "qualify":
        run_qualification(args.protocol, args.workers)
    elif args.command == "seal":
        protocol_path, protocol, id_config, ood_config, _ = validate_protocol(args.protocol)
        print(json.dumps(
            seal_registry(protocol_path, protocol, id_config, ood_config),
            indent=2, sort_keys=True,
        ))
    elif args.command == "formal":
        run_formal(args.protocol, args.workers)
    else:
        status(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
