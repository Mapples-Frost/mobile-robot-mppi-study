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
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_effects,
)
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


def load_benchmark_manifest(path, _visited=None):
    """Load composable benchmark manifests without experiment validation."""

    path = Path(path).resolve()
    visited = set() if _visited is None else set(_visited)
    if path in visited:
        raise ValueError("cyclic benchmark manifest include: %s" % path)
    visited.add(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("benchmark manifest must be a mapping")
    includes = data.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged = {}
    for include in includes:
        include_path = (path.parent / str(include)).resolve()
        merged = deep_merge(
            merged,
            load_benchmark_manifest(include_path, visited),
        )
    return deep_merge(merged, data)


def final_schedule(seeds, domains, scenes, schedule_seed, arms=ARMS):
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
                order = list(arms)
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
    completion_handover_full_fallback_distance=0.0,
    completion_handover_full_rl_distance=0.0,
    planner_overrides=None,
    sensor_overrides=None,
    reliability_overrides=None,
    coupled_actor_checkpoint=None,
    coupled_rl_overrides=None,
    paper_rl_driven_overrides=None,
    scan_guard_overrides=None,
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
    selected_actor = actor_checkpoint
    if (
        flags["use_icode"]
        and flags["use_rl"]
        and coupled_actor_checkpoint
    ):
        selected_actor = str(coupled_actor_checkpoint)
    config = method_config(
        base,
        arm,
        flags["use_icode"],
        flags["use_rl"],
        selected_actor,
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
        if flags["adaptive_hss"]:
            reliability.update(dict(reliability_overrides or {}))
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
        planner["paper_rl_driven"][
            "completion_handover_full_fallback_distance"
        ] = (
            float(completion_handover_full_fallback_distance)
            if flags["adaptive_hss"]
            else 0.0
        )
        planner["paper_rl_driven"][
            "completion_handover_full_rl_distance"
        ] = (
            float(completion_handover_full_rl_distance)
            if flags["adaptive_hss"]
            else 0.0
        )
        planner["paper_rl_driven"].pop("conservative_terminal", None)
        if paper_rl_driven_overrides:
            planner["paper_rl_driven"] = deep_merge(
                planner["paper_rl_driven"],
                dict(paper_rl_driven_overrides),
            )
        if (
            flags["use_icode"]
            and coupled_actor_checkpoint
            and coupled_rl_overrides
        ):
            config["rl"] = deep_merge(
                config["rl"], dict(coupled_rl_overrides)
            )
    planner.update(dict(planner_overrides or {}))
    config.setdefault("sensors", {}).update(dict(sensor_overrides or {}))
    config.setdefault("perception", {}).setdefault("scan_guard", {}).update(
        dict(scan_guard_overrides or {})
    )
    if int(max_steps) > 0:
        config["experiment"]["max_steps"] = int(max_steps)
    config["experiment"]["final_benchmark_arm"] = arm
    return config, flags, checkpoints, calibration


def validate_required_arm_config(config, arm, contracts):
    """Fail closed when a preregistered treatment was not injected.

    Contracts use dotted paths so an experiment manifest can assert the
    resolved runtime configuration before MuJoCo starts.  This prevents a
    development run from silently becoming a no-treatment duplicate because
    an optional nested planner feature retained its default value.
    """

    required = dict((contracts or {}).get(str(arm), {}))
    for dotted_path, expected in required.items():
        current = config
        for key in str(dotted_path).split("."):
            if not isinstance(current, dict) or key not in current:
                raise ValueError(
                    "required arm config is missing %s:%s"
                    % (arm, dotted_path)
                )
            current = current[key]
        if current != expected:
            raise ValueError(
                "required arm config mismatch %s:%s: expected %r, got %r"
                % (arm, dotted_path, expected, current)
            )


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
    return paired_checkpoint_effects(
        control,
        aligned,
        method=label,
        metrics=metrics,
        bootstrap_samples=int(samples),
        seed=int(seed),
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


def resolve_benchmark_seeds(
    cli_seeds,
    frozen,
    qualification,
    shard_index=0,
    shard_count=1,
):
    """Bind formal seeds to the preregistration and derive safe shards.

    Qualification runs intentionally accept an explicit development subset.
    Formal runs do not: their complete seed list lives in the committed
    manifest, and an optional parallel shard is a deterministic strided view
    of that list.  This prevents a formal invocation from silently replacing
    or cherry-picking sealed seeds.
    """

    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if shard_count <= 0:
        raise ValueError("formal shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "formal shard index must satisfy 0 <= index < count"
        )
    provided = tuple(parse_ints(cli_seeds)) if str(cli_seeds).strip() else ()
    if qualification:
        if shard_count != 1 or shard_index != 0:
            raise ValueError("qualification runs do not use formal shards")
        if not provided:
            raise ValueError("qualification run requires explicit seeds")
        return provided, provided

    sealed = tuple(int(seed) for seed in frozen.get(
        "sealed_seeds", frozen.get("formal_seeds", ())
    ))
    if not sealed:
        raise ValueError("formal manifest must define sealed_seeds")
    if len(set(sealed)) != len(sealed):
        raise ValueError("formal manifest sealed_seeds must be unique")
    selected = sealed[shard_index::shard_count]
    if not selected:
        raise ValueError("formal shard selects no sealed seeds")
    if provided and provided != selected:
        raise ValueError(
            "formal CLI seeds do not match the preregistered shard"
        )
    return selected, sealed


def resolve_scene_max_steps(frozen, scenes):
    """Resolve a positive episode budget for every selected scene.

    Existing manifests keep their historical scalar ``max_steps`` behavior.
    A manifest that opts into ``max_steps_by_scene`` is fail-closed: every
    selected scene must have an explicit positive budget.  This prevents a
    misspelled scene name from silently falling back to an incomparable
    episode horizon.
    """

    mapping = frozen.get("max_steps_by_scene")
    if mapping is None:
        value = int(frozen["max_steps"])
        if value <= 0:
            raise ValueError("final_benchmark.max_steps must be positive")
        return {str(scene["name"]): value for scene in scenes}
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "final_benchmark.max_steps_by_scene must be a non-empty mapping"
        )
    parsed = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("max_steps_by_scene keys must be scene names")
        parsed_value = int(value)
        if parsed_value <= 0:
            raise ValueError(
                "max_steps_by_scene[%s] must be positive" % name
            )
        parsed[str(name)] = parsed_value
    selected_names = tuple(str(scene["name"]) for scene in scenes)
    missing = sorted(set(selected_names) - set(parsed))
    if missing:
        raise ValueError(
            "max_steps_by_scene is missing selected scenes: %s"
            % ", ".join(missing)
        )
    return {name: parsed[name] for name in selected_names}


def _manifest_paths(values):
    return [str(_resolve_manifest_path(value)) for value in values]


def _load_reliability_with_gate(summary_path, config_path, evidence_path):
    """Load calibration only when an external frozen-gate audit passed."""

    summary_path = Path(summary_path).resolve()
    reliability = _load_reliability(
        summary_path, config_path, allow_failed_calibration=True
    )
    evidence_path = Path(evidence_path).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("gate_passed") is not True:
        raise ValueError(
            "external reliability evidence did not pass: %s"
            % evidence_path
        )
    expected = str(evidence.get("calibration_summary_sha256", ""))
    observed = _sha256(summary_path)
    if expected != observed:
        raise ValueError(
            "external reliability evidence does not bind calibration: %s"
            % summary_path
        )
    reliability["gate_evidence_path"] = str(evidence_path)
    reliability["gate_evidence_sha256"] = _sha256(evidence_path)
    return reliability


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        default="",
        help=(
            "development seeds for --qualification; formal seeds are "
            "loaded from the preregistered manifest"
        ),
    )
    parser.add_argument("--formal-shard-index", type=int, default=0)
    parser.add_argument("--formal-shard-count", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument(
        "--arms",
        default="",
        help="qualification-only comma-separated arm subset",
    )
    parser.add_argument(
        "--scene-configs",
        default="",
        help="qualification-only comma-separated scene override",
    )
    parser.add_argument(
        "--physics-domains",
        default="",
        help="qualification-only comma-separated domain override",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest = load_benchmark_manifest(manifest_path)
    if not isinstance(manifest, dict) or "final_benchmark" not in manifest:
        raise ValueError("manifest must define final_benchmark")
    frozen = dict(manifest["final_benchmark"])
    if not args.qualification and frozen.get("status") != "preregistered":
        raise ValueError("formal benchmark requires status=preregistered")
    if (
        not args.qualification
        and frozen.get("bootstrap_samples") is not None
        and int(args.bootstrap_samples)
        != int(frozen["bootstrap_samples"])
    ):
        raise ValueError(
            "formal bootstrap count must match the preregistered manifest"
        )
    if not args.qualification and any((
        args.arms, args.scene_configs, args.physics_domains
    )):
        raise ValueError("formal benchmark forbids qualification overrides")
    selected_arms = tuple(
        item.strip() for item in args.arms.split(",") if item.strip()
    ) or ARMS
    unknown_arms = sorted(set(selected_arms) - set(ARMS))
    if unknown_arms:
        raise ValueError(
            "unknown qualification arms: %s" % ", ".join(unknown_arms)
        )
    if len(set(selected_arms)) != len(selected_arms):
        raise ValueError("qualification arms must be unique")
    base_path = _resolve_manifest_path(frozen["base_config"])
    domain_path = _resolve_manifest_path(frozen["physics_domain_config"])
    actor = str(_resolve_manifest_path(frozen["actor_checkpoint"]))
    coupled_actor_value = frozen.get("coupled_actor_checkpoint")
    coupled_actor = (
        str(_resolve_manifest_path(coupled_actor_value))
        if coupled_actor_value
        else actor
    )
    ordinary_checkpoints = _manifest_paths(
        frozen["ordinary_checkpoints"]
    )
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
    base = load_yaml(base_path)
    seeds, sealed_seeds = resolve_benchmark_seeds(
        args.seeds,
        frozen,
        bool(args.qualification),
        args.formal_shard_index,
        args.formal_shard_count,
    )
    selected_domains = tuple(
        item.strip()
        for item in args.physics_domains.split(",")
        if item.strip()
    ) or tuple(frozen["physics_domains"])
    selected_scene_paths = tuple(
        item.strip()
        for item in args.scene_configs.split(",")
        if item.strip()
    ) or tuple(frozen["scene_configs"])
    domains = load_physics_domains(
        domain_path, selected_domains, ()
    )
    scenes = load_scenes(base, selected_scene_paths)
    max_steps_by_scene = resolve_scene_max_steps(frozen, scenes)
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
    full_schedule = final_schedule(
        sealed_seeds,
        domains,
        scenes,
        int(frozen["schedule_seed"]),
        selected_arms,
    )
    selected_seed_set = set(seeds)
    schedule = [
        job for job in full_schedule
        if int(job["seed"]) in selected_seed_set
    ]
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
            int(max_steps_by_scene[job["scene"]]),
            float(frozen["terminal_guidance_radius"]),
            float(frozen["terminal_guided_fraction_floor"]),
            float(frozen.get(
                "completion_handover_full_fallback_distance", 0.0
            )),
            float(frozen.get(
                "completion_handover_full_rl_distance", 0.0
            )),
            frozen.get("planner_overrides", {}),
            frozen.get("sensor_overrides", {}),
            frozen.get("reliability_overrides", {}),
            coupled_actor_checkpoint=coupled_actor,
            coupled_rl_overrides=frozen.get(
                "coupled_rl_overrides", {}
            ),
            paper_rl_driven_overrides=frozen.get(
                "paper_rl_driven_overrides", {}
            ),
            scan_guard_overrides=frozen.get("scan_guard_overrides", {}),
        )
        arm = str(job["arm"])
        validate_required_arm_config(
            config,
            arm,
            frozen.get("required_arm_config", {}),
        )
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
            "actor_checkpoint": (
                str(config.get("rl", {}).get("checkpoint", ""))
                if flags["use_rl"] else ""
            ),
            "icode_checkpoints": (
                "|".join(checkpoints) if flags["use_icode"] else ""
            ),
            "reliability_calibration_summary": (
                calibration["summary_path"] if flags["use_icode"] else ""
            ),
            "rollout_budget_per_decision": int(
                frozen["total_rollouts"]
            ),
            "max_steps_budget": int(max_steps_by_scene[job["scene"]]),
            "paper_iterations": (
                int(frozen["iterations"]) if flags["use_rl"] else 1
            ),
            "metric_profile": profile,
            "qualification": int(bool(args.qualification)),
        })
        rows.append(row)
        _write_csv(output / "progress.csv", rows)

    for arm in selected_arms:
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
        "coupled_actor_checkpoint": {
            "path": coupled_actor,
            "sha256": _sha256(coupled_actor),
        },
        "ordinary_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in ordinary_checkpoints
        ],
        "value_checkpoints": [
            {"path": path, "sha256": _sha256(path)}
            for path in value_checkpoints
        ],
        "ordinary_reliability_gate": {
            "path": ordinary_reliability["gate_evidence_path"],
            "sha256": ordinary_reliability["gate_evidence_sha256"],
        },
        "value_reliability_gate": {
            "path": value_reliability["gate_evidence_path"],
            "sha256": value_reliability["gate_evidence_sha256"],
        },
        "seeds": list(seeds),
        "sealed_seeds": list(sealed_seeds),
        "formal_shard": {
            "index": int(args.formal_shard_index),
            "count": int(args.formal_shard_count),
        },
        "independent_unit": "seed",
        "repeated_strata": ["scene", "physics_domain"],
        "arms": list(selected_arms),
        "scenes": [
            {"name": item["name"], "source": item["source"]}
            for item in scenes
        ],
        "physics_domains": domains,
        "schedule_seed": int(frozen["schedule_seed"]),
        "metric_profile": profile,
        "max_steps_by_scene": dict(max_steps_by_scene),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if len(seeds) >= 2 and set(selected_arms) == set(ARMS):
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
        "blocks": len(schedule) // len(selected_arms),
        "status": provenance["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
