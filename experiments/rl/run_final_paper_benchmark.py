#!/usr/bin/env python3
"""Run the frozen seven-arm paper benchmark with seed-cluster inference.

The benchmark separates contextual baselines from the confirmatory 2x2
factorial.  Scene/domain repetitions are strata; the simulation seed is the
independent unit used by every bootstrap confidence interval.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.run_full_proposed_factorial import (
    _factorial,
    _git_sha,
    _load_reliability,
    _paired,
    _parse_paths,
    _sha256,
    _write_csv,
    metrics_for_profile,
    path_tracking_metrics,
)
from experiments.rl.run_gate1_simple_combination import (
    load_physics_domains,
    load_scenes,
    method_config,
    parse_ints,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner

ARMS = (
    "traditional_mppi",
    "icode_mppi",
    "rl_driven_mppi",
    "simple_combination",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)
CORE_ARMS = (
    "simple_combination",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)
CORE_FACTORIAL_METHOD = {
    "simple_combination": "traditional_mppi",
    "value_fixed": "icode_mppi",
    "ordinary_adaptive": "rl_driven_mppi",
    "full_proposed": "simple_combination",
}


def final_schedule(seeds, domains, scenes, schedule_seed):
    """Randomize all seven arms inside each seed-scene-domain block."""

    import numpy as np

    rng = np.random.RandomState(int(schedule_seed))
    jobs = []
    for seed in seeds:
        for scene in scenes:
            for domain in domains:
                block = "%s::%s::seed%d" % (
                    scene["name"], domain["name"], int(seed)
                )
                order = list(ARMS)
                rng.shuffle(order)
                for within, arm in enumerate(order):
                    jobs.append({
                        "seed": int(seed),
                        "scene": str(scene["name"]),
                        "physics_domain": str(domain["name"]),
                        "block": block,
                        "arm": arm,
                        "run_order_within_block": int(within),
                    })
    for index, job in enumerate(jobs):
        job["global_run_order"] = int(index)
    return jobs


def _arm_flags(arm):
    if arm not in ARMS:
        raise ValueError("unknown final benchmark arm: %s" % arm)
    return {
        "use_icode": arm not in (
            "traditional_mppi", "rl_driven_mppi"
        ),
        "use_rl": arm not in (
            "traditional_mppi", "icode_mppi"
        ),
        "value_aligned": arm in ("value_fixed", "full_proposed"),
        "adaptive_hss": arm in (
            "ordinary_adaptive", "full_proposed"
        ),
    }


def build_arm_config(
    base,
    job,
    actor_checkpoint,
    ordinary_checkpoints,
    value_checkpoints,
    ordinary_reliability,
    value_reliability,
    total_rollouts,
    iterations,
    physics_domain,
    max_steps,
    terminal_guidance_radius,
    terminal_guided_fraction_floor,
):
    """Build one frozen arm without allowing cross-arm parameter leakage."""

    arm = str(job["arm"])
    flags = _arm_flags(arm)
    checkpoints = (
        value_checkpoints
        if flags["value_aligned"]
        else ordinary_checkpoints
    )
    calibration = (
        value_reliability
        if flags["value_aligned"]
        else ordinary_reliability
    )
    config = method_config(
        base,
        arm,
        flags["use_icode"],
        flags["use_rl"],
        actor_checkpoint,
        checkpoints[0],
        total_rollouts,
        iterations,
        job["seed"],
        physics_domain,
    )
    planner = config["planner"]
    if flags["use_icode"]:
        planner["checkpoints"] = list(checkpoints)
        planner.pop("checkpoint", None)
        ensemble = calibration["ensemble"]
        planner["residual_ensemble"] = {
            "disagreement_scales": list(
                ensemble["disagreement_scales"]
            ),
            "innovation_scales": list(ensemble["innovation_scales"]),
            "innovation_decay": float(ensemble["innovation_decay"]),
            "support_soft_z": float(ensemble["support_soft_z"]),
            "support_hard_z": float(ensemble["support_hard_z"]),
        }
    if flags["use_rl"]:
        reliability = dict(calibration["runtime"])
        reliability["enabled"] = bool(flags["adaptive_hss"])
        planner["paper_rl_driven"]["reliability"] = reliability
        planner["paper_rl_driven"]["terminal_guidance_radius"] = (
            float(terminal_guidance_radius)
            if flags["adaptive_hss"]
            else 0.0
        )
        planner["paper_rl_driven"][
            "terminal_guided_fraction_floor"
        ] = (
            float(terminal_guided_fraction_floor)
            if flags["adaptive_hss"]
            else 0.0
        )
        planner["paper_rl_driven"].pop("conservative_terminal", None)
    if int(max_steps) > 0:
        config["experiment"]["max_steps"] = int(max_steps)
    config["experiment"]["final_benchmark_arm"] = arm
    return config, flags, checkpoints, calibration


def _core_factorial_rows(rows):
    result = []
    for row in rows:
        arm = str(row["benchmark_arm"])
        if arm not in CORE_ARMS:
            continue
        mapped = dict(row)
        mapped["method"] = CORE_FACTORIAL_METHOD[arm]
        result.append(mapped)
    return result


def _comparison(rows, before, after, label, samples, seed, metrics):
    control = [
        dict(row, method=label)
        for row in rows
        if row["benchmark_arm"] == before
    ]
    aligned = [
        dict(row, method=label)
        for row in rows
        if row["benchmark_arm"] == after
    ]
    return _paired(
        control,
        aligned,
        label,
        samples,
        seed,
        metrics,
    )


def _analyse(rows, samples, seed, metrics):
    comparisons = (
        ("icode_vs_traditional", "traditional_mppi", "icode_mppi"),
        ("rl_vs_traditional", "traditional_mppi", "rl_driven_mppi"),
        (
            "simple_vs_traditional",
            "traditional_mppi",
            "simple_combination",
        ),
        ("simple_vs_icode", "icode_mppi", "simple_combination"),
        ("simple_vs_rl", "rl_driven_mppi", "simple_combination"),
        ("value_at_fixed", "simple_combination", "value_fixed"),
        (
            "hss_at_ordinary",
            "simple_combination",
            "ordinary_adaptive",
        ),
        ("full_vs_simple", "simple_combination", "full_proposed"),
        ("full_vs_icode", "icode_mppi", "full_proposed"),
        ("full_vs_rl", "rl_driven_mppi", "full_proposed"),
    )
    paired = {}
    for index, (label, before, after) in enumerate(comparisons):
        paired[label] = _comparison(
            rows,
            before,
            after,
            label,
            samples,
            int(seed) + index,
            metrics,
        )
    core = _core_factorial_rows(rows)
    return paired, _factorial(core, samples, int(seed) + 100, metrics)


def _resolve_manifest_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else (ROOT / path).resolve()


def _manifest_paths(values):
    return [str(_resolve_manifest_path(value)) for value in values]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or "final_benchmark" not in manifest:
        raise ValueError("manifest must define final_benchmark")
    frozen = dict(manifest["final_benchmark"])
    if not args.qualification and frozen.get("status") != "preregistered":
        raise ValueError("formal benchmark requires status=preregistered")
    base_path = _resolve_manifest_path(frozen["base_config"])
    domain_path = _resolve_manifest_path(frozen["physics_domain_config"])
    actor = str(_resolve_manifest_path(frozen["actor_checkpoint"]))
    ordinary_checkpoints = _manifest_paths(
        frozen["ordinary_checkpoints"]
    )
    value_checkpoints = _manifest_paths(frozen["value_checkpoints"])
    ordinary_reliability = _load_reliability(
        _resolve_manifest_path(frozen["ordinary_calibration_summary"]),
        _resolve_manifest_path(frozen["ordinary_calibration_config"]),
    )
    value_reliability = _load_reliability(
        _resolve_manifest_path(frozen["value_calibration_summary"]),
        _resolve_manifest_path(frozen["value_calibration_config"]),
    )
    base = load_yaml(base_path)
    seeds = parse_ints(args.seeds)
    domains = load_physics_domains(
        domain_path, tuple(frozen["physics_domains"]), ()
    )
    scenes = load_scenes(base, tuple(frozen["scene_configs"]))
    profile = str(frozen.get("metric_profile", "point_goal"))
    metrics = metrics_for_profile(profile)
    if profile == "path_tracking":
        invalid = [
            item["name"]
            for item in scenes
            if item["config"].get("task", {}).get("type") != "polyline"
        ]
        if invalid:
            raise ValueError(
                "path_tracking profile requires polyline scenes: %s"
                % ", ".join(invalid)
            )
    schedule = final_schedule(
        seeds,
        domains,
        scenes,
        int(frozen["schedule_seed"]),
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_by_name = {item["name"]: item for item in scenes}
    domain_by_name = {item["name"]: item for item in domains}
    rows = []
    for job in schedule:
        config, flags, checkpoints, calibration = build_arm_config(
            scene_by_name[job["scene"]]["config"],
            job,
            actor,
            ordinary_checkpoints,
            value_checkpoints,
            ordinary_reliability,
            value_reliability,
            int(frozen["total_rollouts"]),
            int(frozen["iterations"]),
            domain_by_name[job["physics_domain"]],
            int(frozen["max_steps"]),
            float(frozen["terminal_guidance_radius"]),
            float(frozen["terminal_guided_fraction_floor"]),
        )
        arm = str(job["arm"])
        run_dir = output / "runs" / arm / config["experiment"]["name"]
        experiment = ExperimentRunner(
            config, ROOT, run_dir, headless=True
        ).run()
        row = dict(experiment.summary)
        if profile == "path_tracking":
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
                raise ValueError("trajectory and episode metrics disagree")
            row.update(tracking)
        row.update({
            "method": arm,
            "benchmark_arm": arm,
            "value_alignment": int(flags["value_aligned"]),
            "adaptive_hss": int(flags["adaptive_hss"]),
            "scene": str(job["scene"]),
            "scene_source": scene_by_name[job["scene"]]["source"],
            "physics_domain": str(job["physics_domain"]),
            "physics_domain_role": str(
                domain_by_name[job["physics_domain"]].get(
                    "role", "unknown"
                )
            ),
            "seed": int(job["seed"]),
            "block": str(job["block"]),
            "run_order_within_block": int(
                job["run_order_within_block"]
            ),
            "global_run_order": int(job["global_run_order"]),
            "actor_checkpoint": actor if flags["use_rl"] else "",
            "icode_checkpoints": (
                "|".join(checkpoints) if flags["use_icode"] else ""
            ),
            "reliability_calibration_summary": (
                calibration["summary_path"] if flags["use_icode"] else ""
            ),
            "rollout_budget_per_decision": int(
                frozen["total_rollouts"]
            ),
            "paper_iterations": (
                int(frozen["iterations"]) if flags["use_rl"] else 1
            ),
            "metric_profile": profile,
            "qualification": int(bool(args.qualification)),
        })
        rows.append(row)
        _write_csv(output / "progress.csv", rows)

    for arm in ARMS:
        _write_csv(
            output / ("%s_episodes.csv" % arm),
            [row for row in rows if row["benchmark_arm"] == arm],
        )
    provenance = {
        "status": (
            "pipeline_qualification"
            if args.qualification
            else "formal_preregistered_benchmark"
        ),
        "git_sha": _git_sha(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "actor_checkpoint": {"path": actor, "sha256": _sha256(actor)},
        "ordinary_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in ordinary_checkpoints
        ],
        "value_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in value_checkpoints
        ],
        "seeds": list(seeds),
        "independent_unit": "seed",
        "repeated_strata": ["scene", "physics_domain"],
        "arms": list(ARMS),
        "scenes": [
            {"name": item["name"], "source": item["source"]}
            for item in scenes
        ],
        "physics_domains": domains,
        "schedule_seed": int(frozen["schedule_seed"]),
        "metric_profile": profile,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if len(seeds) >= 2:
        paired, factorial = _analyse(
            rows,
            int(args.bootstrap_samples),
            int(frozen["schedule_seed"]),
            metrics,
        )
        (output / "paired_comparisons.json").write_text(
            json.dumps(paired, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "factorial_contrasts.json").write_text(
            json.dumps(factorial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "output_dir": str(output),
        "episodes": len(rows),
        "independent_seeds": len(seeds),
        "blocks": len(schedule) // len(ARMS),
        "status": provenance["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
