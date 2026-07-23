#!/usr/bin/env python3
"""Run the preregistered L285 frozen-Actor proposal development Gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_final_paper_benchmark import (
    _git_sha,
    _load_reliability_with_gate,
    _manifest_paths,
    _resolve_manifest_path,
    _sha256,
    _write_csv,
    build_arm_config,
    load_benchmark_manifest,
    resolve_scene_max_steps,
)
from experiments.rl.run_gate1_simple_combination import (
    load_physics_domains,
    load_scenes,
)
from experiments.rl.run_full_proposed_factorial import path_tracking_metrics
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_CONFIG = ROOT / "configs/research/tracking_l285_frozen_actor_proposal_gate.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l285_frozen_actor_proposal_gate"
L285_ARMS = (
    "icode_value_control",
    "source_full",
    "l276_full",
    "l281_full",
    "l284_full",
)
FULL_VARIANTS = ("source_full", "l276_full", "l281_full", "l284_full")
NUMERIC_METRICS = (
    "path_completion_ratio",
    "cross_track_rmse",
    "final_goal_distance",
    "return",
    "reliability_proposal_authority_mean",
    "reliability_proposal_fallback_fraction_mean",
    "rl_elite_fraction_mean",
)


def _read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def _as_float(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("L285 non-finite metric %s" % key)
    return value


def _experimental_key(row):
    return (
        int(row["seed"]),
        str(row["scene"]),
        str(row["physics_domain"]),
        str(row["experimental_arm"]),
    )


def frozen_schedule(seeds, scenes, domains, schedule_seed, arms=L285_ARMS):
    rng = np.random.RandomState(int(schedule_seed))
    jobs = []
    for seed in seeds:
        for scene in scenes:
            for domain in domains:
                block = "%s::%s::seed%d" % (
                    scene["name"], domain["name"], int(seed)
                )
                order = list(arms)
                rng.shuffle(order)
                for within, arm in enumerate(order):
                    jobs.append({
                        "seed": int(seed),
                        "scene": str(scene["name"]),
                        "physics_domain": str(domain["name"]),
                        "block": block,
                        "experimental_arm": str(arm),
                        "run_order_within_block": int(within),
                    })
    for index, job in enumerate(jobs):
        job["global_run_order"] = int(index)
    return jobs


def _load_and_validate_evidence(frozen):
    evidence = {}
    for name, spec in frozen["source_evidence"].items():
        path = _resolve_manifest_path(spec["path"])
        if _sha256(path) != str(spec["sha256"]):
            raise ValueError("L285 %s evidence SHA256 mismatch" % name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decision") != spec["required_decision"]:
            raise ValueError("L285 %s evidence decision mismatch" % name)
        evidence[name] = {"path": str(path), "sha256": _sha256(path)}
    return evidence


def _checkpoint_table(frozen):
    source = frozen["source_actor"]
    source_path = _resolve_manifest_path(source["path"])
    if _sha256(source_path) != str(source["sha256"]):
        raise ValueError("L285 source Actor SHA256 mismatch")
    table = {}
    paired = {}
    for block in frozen["paired_blocks"]:
        seed = int(block["evaluation_seed"])
        if seed in table:
            raise ValueError("L285 duplicate evaluation seed")
        variants = {"source": str(source_path)}
        metadata = {"source": {"actor_seed": None, "sha256": _sha256(source_path)}}
        for variant in ("l276", "l281", "l284"):
            spec = block["checkpoints"][variant]
            path = _resolve_manifest_path(spec["path"])
            observed = _sha256(path)
            if observed != str(spec["sha256"]):
                raise ValueError("L285 %s checkpoint SHA256 mismatch" % variant)
            variants[variant] = str(path)
            metadata[variant] = {
                "actor_seed": int(block["actor_seeds"][variant]),
                "sha256": observed,
            }
        table[seed] = variants
        paired[seed] = metadata
    if set(table) != {int(seed) for seed in frozen["development_seeds"]}:
        raise ValueError("L285 paired block seed set mismatch")
    return table, paired


def _aggregate(rows, arm):
    selected = [row for row in rows if row["experimental_arm"] == arm]
    result = {}
    for seed in sorted({int(row["seed"]) for row in selected}):
        group = [row for row in selected if int(row["seed"]) == seed]
        if len(group) != 3 or len({row["scene"] for row in group}) != 3:
            raise ValueError("L285 seed-arm does not contain three scenes")
        result[seed] = {
            key: float(np.mean([_as_float(row, key) for row in group]))
            for key in NUMERIC_METRICS
        }
        result[seed].update({
            "collisions": int(sum(_as_bool(row["collision"]) for row in group)),
            "boundary_violation_episodes": int(sum(
                _as_float(row, "boundary_violation_steps") > 0.0 for row in group
            )),
            "boundary_safe_successes": int(sum(
                _as_bool(row["boundary_safe_success"]) for row in group
            )),
        })
    return result


def _paired_effects(treatment, control):
    if set(treatment) != set(control):
        raise ValueError("L285 paired seed set mismatch")
    effects = []
    for seed in sorted(treatment):
        current, baseline = treatment[seed], control[seed]
        effects.append({
            "seed": int(seed),
            "completion_change": current["path_completion_ratio"] - baseline["path_completion_ratio"],
            "cte_improvement": baseline["cross_track_rmse"] - current["cross_track_rmse"],
            "goal_distance_improvement": baseline["final_goal_distance"] - current["final_goal_distance"],
            "return_change": current["return"] - baseline["return"],
            "collision_increase": current["collisions"] - baseline["collisions"],
            "boundary_violation_episode_increase": current["boundary_violation_episodes"] - baseline["boundary_violation_episodes"],
            "boundary_safe_success_change": current["boundary_safe_successes"] - baseline["boundary_safe_successes"],
        })
    return effects


def evaluate(rows, frozen, engineering):
    expected = len(frozen["development_seeds"]) * 3 * len(L285_ARMS)
    keys = [_experimental_key(row) for row in rows]
    complete = len(rows) == expected and len(set(keys)) == expected
    for row in rows:
        for key in NUMERIC_METRICS + ("boundary_violation_steps",):
            _as_float(row, key)
    aggregates = {arm: _aggregate(rows, arm) for arm in L285_ARMS}
    primary = _paired_effects(aggregates["l276_full"], aggregates["source_full"])
    versus_icode = _paired_effects(
        aggregates["l276_full"], aggregates["icode_value_control"]
    )
    descriptive = {
        arm: _paired_effects(aggregates[arm], aggregates["source_full"])
        for arm in ("l281_full", "l284_full")
    }
    scene_values = defaultdict(list)
    lookup = {
        (row["experimental_arm"], int(row["seed"]), row["scene"]): row
        for row in rows
    }
    for seed in frozen["development_seeds"]:
        for scene in sorted({row["scene"] for row in rows}):
            source = lookup[("source_full", int(seed), scene)]
            l276 = lookup[("l276_full", int(seed), scene)]
            scene_values[scene].append(
                _as_float(source, "cross_track_rmse")
                - _as_float(l276, "cross_track_rmse")
            )
    scene_cte = {
        scene: float(np.mean(values))
        for scene, values in sorted(scene_values.items())
    }
    gate = frozen["gate"]
    completion = [row["completion_change"] for row in primary]
    cte = [row["cte_improvement"] for row in primary]
    goal = [row["goal_distance_improvement"] for row in primary]
    completion_vs_icode = [row["completion_change"] for row in versus_icode]
    improved_seeds = sum(c > 0.0 or g > 0.0 for c, g in zip(cte, goal))
    l276_aggregate = aggregates["l276_full"]
    authorities = [
        l276_aggregate[seed]["reliability_proposal_authority_mean"]
        for seed in sorted(l276_aggregate)
    ]
    elite_fractions = [
        l276_aggregate[seed]["rl_elite_fraction_mean"]
        for seed in sorted(l276_aggregate)
    ]
    safety_primary = all(
        row["collision_increase"] <= int(gate["maximum_collision_increase"])
        and row["boundary_violation_episode_increase"]
        <= int(gate["maximum_boundary_violation_episode_increase"])
        for row in primary
    )
    safety_icode = all(
        row["collision_increase"] <= int(gate["maximum_collision_increase"])
        and row["boundary_violation_episode_increase"]
        <= int(gate["maximum_boundary_violation_episode_increase"])
        for row in versus_icode
    )
    checks = {
        "engineering_complete": bool(complete and all(engineering.values())),
        "completion_regression_bound_each_seed": all(
            value >= -float(gate["maximum_per_seed_mean_completion_regression"])
            for value in completion
        ),
        "median_completion_nonnegative": float(np.median(completion))
        >= float(gate["minimum_median_completion_change"]),
        "cte_or_goal_effect": (
            float(np.median(cte)) >= float(gate["minimum_median_cte_improvement"])
            or float(np.median(goal))
            >= float(gate["minimum_median_goal_distance_improvement"])
        ),
        "seed_direction_coverage": improved_seeds
        >= int(gate["minimum_seeds_with_cte_or_goal_improvement"]),
        "scene_cte_coverage": sum(value > 0.0 for value in scene_cte.values())
        >= int(gate["minimum_scenes_with_cte_improvement"]),
        "safety_noninferior_to_source": safety_primary,
        "safety_noninferior_to_icode": safety_icode,
        "completion_noninferior_to_icode_each_seed": all(
            value >= -float(gate["maximum_per_seed_completion_regression_vs_icode"])
            for value in completion_vs_icode
        ),
        "median_completion_nonnegative_vs_icode": float(np.median(completion_vs_icode))
        >= float(gate["minimum_median_completion_change_vs_icode"]),
        "proposal_mechanism_active": (
            float(np.median(authorities))
            >= float(gate["minimum_median_proposal_authority"])
            and float(np.median(elite_fractions))
            >= float(gate["minimum_median_rl_elite_fraction"])
        ),
    }
    metrics = {
        "primary_l276_vs_source": primary,
        "l276_vs_icode": versus_icode,
        "descriptive_vs_source": descriptive,
        "median_completion_change": float(np.median(completion)),
        "median_cte_improvement": float(np.median(cte)),
        "median_goal_distance_improvement": float(np.median(goal)),
        "median_completion_change_vs_icode": float(np.median(completion_vs_icode)),
        "scene_cte_improvements": scene_cte,
        "seeds_with_cte_or_goal_improvement": int(improved_seeds),
        "median_proposal_authority": float(np.median(authorities)),
        "median_rl_elite_fraction": float(np.median(elite_fractions)),
        "aggregates": aggregates,
    }
    passed = bool(all(checks.values()))
    screen_only = frozen.get("study_mode") == "resource_limited_direction_screen"
    return {
        "protocol": "L285",
        "status": "complete",
        "study_mode": frozen.get("study_mode", "confirmatory_gate"),
        "screen_pass": passed if screen_only else None,
        "gate_pass": False if screen_only else passed,
        "decision": (
            ("frozen_l276_actor_proposal_screen_promising" if passed
             else "frozen_l276_actor_proposal_screen_not_promising")
            if screen_only else
            ("frozen_l276_actor_proposal_gate_pass" if passed
             else "frozen_l276_actor_proposal_gate_fail")
        ),
        "checks": checks,
        "metrics": metrics,
        "larger_validation_preregistration_authorized": passed,
        "early_stopping_probe_authorized": not passed,
        "final_map_evaluation_authorized": False,
        "descriptive_arms_cannot_override_primary_gate": True,
    }


def run(config_path, output, resume=False, max_episodes=None):
    manifest_path = Path(config_path).resolve()
    manifest = load_benchmark_manifest(manifest_path)
    frozen = dict(manifest["final_benchmark"])
    if frozen.get("status") != "preregistered" or frozen.get("protocol") != "L285":
        raise ValueError("L285 requires a preregistered L285 manifest")
    if tuple(frozen["l285_arms"]) != L285_ARMS:
        raise ValueError("L285 arm contract drifted")
    contract_payload = dict(frozen)
    contract_payload.pop("forbidden_tokens", None)
    serialized = json.dumps(contract_payload, sort_keys=True).lower()
    for token in frozen["forbidden_tokens"]:
        if str(token).lower() in serialized:
            raise ValueError("L285 manifest contains forbidden token %s" % token)
    evidence = _load_and_validate_evidence(frozen)
    checkpoints, paired_metadata = _checkpoint_table(frozen)

    base = load_yaml(_resolve_manifest_path(frozen["base_config"]))
    domains = load_physics_domains(
        _resolve_manifest_path(frozen["physics_domain_config"]),
        tuple(frozen["physics_domains"]),
        (),
    )
    scenes = load_scenes(base, tuple(frozen["scene_configs"]))
    if any(scene["config"].get("task", {}).get("type") != "polyline" for scene in scenes):
        raise ValueError("L285 requires polyline development scenes")
    max_steps = resolve_scene_max_steps(frozen, scenes)
    seeds = tuple(int(seed) for seed in frozen["development_seeds"])
    schedule = frozen_schedule(
        seeds, scenes, domains, int(frozen["schedule_seed"]), L285_ARMS
    )
    ordinary_checkpoints = _manifest_paths(frozen["ordinary_checkpoints"])
    value_checkpoints = _manifest_paths(frozen["value_checkpoints"])
    ordinary_reliability = _load_reliability_with_gate(
        _resolve_manifest_path(frozen["ordinary_calibration_summary"]),
        _resolve_manifest_path(frozen["ordinary_calibration_config"]),
        _resolve_manifest_path(frozen["ordinary_calibration_gate_evidence"]),
    )
    value_reliability = _load_reliability_with_gate(
        _resolve_manifest_path(frozen["value_calibration_summary"]),
        _resolve_manifest_path(frozen["value_calibration_config"]),
        _resolve_manifest_path(frozen["value_calibration_gate_evidence"]),
    )

    output = Path(output).resolve()
    progress_path = output / "progress.csv"
    if output.exists() and not resume:
        raise FileExistsError("L285 output exists; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    existing_schedule = output / "schedule.json"
    if existing_schedule.is_file():
        if json.loads(existing_schedule.read_text(encoding="utf-8")) != schedule:
            raise ValueError("L285 resume schedule mismatch")
    else:
        existing_schedule.write_text(
            json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    rows = _read_csv(progress_path)
    completed = {_experimental_key(row) for row in rows}
    if len(completed) != len(rows):
        raise ValueError("L285 progress contains duplicate keys")

    scene_by_name = {scene["name"]: scene for scene in scenes}
    domain_by_name = {domain["name"]: domain for domain in domains}
    episodes_run = 0
    for job in schedule:
        key = (
            int(job["seed"]), job["scene"], job["physics_domain"],
            job["experimental_arm"],
        )
        if key in completed:
            continue
        if max_episodes is not None and episodes_run >= int(max_episodes):
            break
        arm = job["experimental_arm"]
        if arm == "icode_value_control":
            benchmark_arm = "value_fixed"
            actor_variant = "source"
        else:
            benchmark_arm = "full_proposed"
            actor_variant = arm.removesuffix("_full")
        actor = checkpoints[int(job["seed"])][actor_variant]
        benchmark_job = dict(job, arm=benchmark_arm)
        config, flags, used_checkpoints, calibration = build_arm_config(
            scene_by_name[job["scene"]]["config"],
            benchmark_job,
            actor,
            ordinary_checkpoints,
            value_checkpoints,
            ordinary_reliability,
            value_reliability,
            int(frozen["total_rollouts"]),
            int(frozen["iterations"]),
            domain_by_name[job["physics_domain"]],
            int(max_steps[job["scene"]]),
            float(frozen["terminal_guidance_radius"]),
            float(frozen["terminal_guided_fraction_floor"]),
            float(frozen.get("completion_handover_full_fallback_distance", 0.0)),
            float(frozen.get("completion_handover_full_rl_distance", 0.0)),
            frozen.get("planner_overrides", {}),
            frozen.get("sensor_overrides", {}),
            frozen.get("reliability_overrides", {}),
            coupled_actor_checkpoint=actor,
            coupled_rl_overrides=frozen.get("coupled_rl_overrides", {}),
            paper_rl_driven_overrides=frozen.get("paper_rl_driven_overrides", {}),
            scan_guard_overrides=frozen.get("scan_guard_overrides", {}),
        )
        config["experiment"]["name"] = "l285_%s_%s_seed%d" % (
            arm, job["scene"], int(job["seed"])
        )
        run_dir = output / "runs" / arm / config["experiment"]["name"]
        if run_dir.exists():
            raise FileExistsError("L285 incomplete run directory blocks exact resume")
        experiment = ExperimentRunner(config, ROOT, run_dir, headless=True).run()
        row = dict(experiment.summary)
        tracking = path_tracking_metrics(
            run_dir / "trajectory.csv",
            config["task"]["points"],
            config["task"].get("completion_corridor", 0.75),
        )
        if not math.isclose(
            float(row["cross_track_rmse"]),
            tracking["path_cross_track_rmse_recomputed"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("L285 trajectory and episode metrics disagree")
        row.update(tracking)
        row.update({
            "method": arm,
            "experimental_arm": arm,
            "runtime_benchmark_arm": benchmark_arm,
            "actor_variant": actor_variant,
            "actor_checkpoint": actor if flags["use_rl"] else "",
            "actor_checkpoint_sha256": _sha256(Path(actor)) if flags["use_rl"] else "",
            "seed": int(job["seed"]),
            "scene": job["scene"],
            "scene_source": scene_by_name[job["scene"]]["source"],
            "physics_domain": job["physics_domain"],
            "physics_domain_role": domain_by_name[job["physics_domain"]].get("role", "unknown"),
            "block": job["block"],
            "run_order_within_block": int(job["run_order_within_block"]),
            "global_run_order": int(job["global_run_order"]),
            "icode_checkpoints": "|".join(used_checkpoints),
            "reliability_calibration_summary": calibration["summary_path"],
            "rollout_budget_per_decision": int(frozen["total_rollouts"]),
            "max_steps_budget": int(max_steps[job["scene"]]),
        })
        rows.append(row)
        completed.add(key)
        episodes_run += 1
        _write_csv(progress_path, rows)
        print(json.dumps({
            "completed_episodes": len(rows),
            "expected_episodes": len(schedule),
            "experimental_arm": arm,
            "scene": job["scene"],
            "seed": int(job["seed"]),
        }, sort_keys=True), flush=True)

    provenance = {
        "protocol": "L285",
        "status": "development_preregistered",
        "git_sha": _git_sha(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "evidence": evidence,
        "checkpoint_metadata": paired_metadata,
        "development_seeds": list(seeds),
        "independent_unit": "paired_actor_environment_seed_cluster",
        "repeated_strata": ["scene", "physics_domain"],
        "arms": list(L285_ARMS),
        "scenes": [{"name": scene["name"], "source": scene["source"]} for scene in scenes],
        "physics_domains": domains,
        "schedule_seed": int(frozen["schedule_seed"]),
        "max_steps_by_scene": max_steps,
        "no_training": True,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(rows) == len(schedule):
        engineering = {
            "schedule_complete": len({_experimental_key(row) for row in rows}) == len(schedule),
            "checkpoint_hashes_verified": True,
            "evidence_hashes_verified": True,
            "no_training": True,
            "windows_native_contract": sys.platform == "win32",
        }
        summary = evaluate(rows, frozen, engineering)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    return {"status": "partial", "completed_episodes": len(rows), "expected_episodes": len(schedule)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    run(Path(args.config), Path(args.output_dir), args.resume, args.max_episodes)


if __name__ == "__main__":
    main()
