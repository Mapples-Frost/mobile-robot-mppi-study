#!/usr/bin/env python3
"""Paired Gate 1 factorial for paper-faithful ICODE + RL-Driven MPPI."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.evaluation.factorial import (
    blocked_factorial_contrasts,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


METHODS = (
    ("traditional_mppi", False, False),
    ("icode_mppi", True, False),
    ("rl_driven_mppi", False, True),
    ("simple_combination", True, True),
)


def parse_ints(value):
    values = tuple(int(item) for item in str(value).split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise ValueError("seeds must be non-negative integers")
    return values


def method_config(
    base,
    method,
    use_icode,
    use_rl,
    actor_checkpoint,
    icode_checkpoint,
    total_rollouts,
    iterations,
    seed,
    physics_domain=None,
):
    """Create one auditable cell with an equal model-rollout budget."""

    total_rollouts = int(total_rollouts)
    iterations = int(iterations)
    if total_rollouts <= 0 or iterations <= 0:
        raise ValueError("rollout budget and iterations must be positive")
    if use_rl and total_rollouts % iterations:
        raise ValueError(
            "paper cells require total_rollouts divisible by iterations"
        )
    config = copy.deepcopy(base)
    domain = dict(physics_domain or {
        "name": "embedded",
        "role": "embedded",
        "plant_override": {},
    })
    config = deep_merge(
        config, {"plant": domain.get("plant_override", {})}
    )
    config.setdefault("experiment", {})["seed"] = int(seed)
    scene_name = str(
        config.get("scene", {}).get("name", "embedded")
    )
    config["experiment"]["name"] = "%s__%s__%s__seed%d" % (
        method,
        scene_name,
        str(domain["name"]),
        int(seed),
    )
    planner = config.setdefault("planner", {})
    planner["prediction_mode"] = (
        "icode_residual" if use_icode else "nominal"
    )
    if use_icode:
        if not icode_checkpoint:
            raise ValueError("ICODE cells require an ICODE checkpoint")
        planner["checkpoint"] = str(icode_checkpoint)
        planner["integrator"] = "rk4"
    else:
        planner.pop("checkpoint", None)
    config.setdefault("memory", {})["enable"] = False

    if use_rl:
        if not actor_checkpoint:
            raise ValueError("RL cells require a direct Actor checkpoint")
        planner.update({
            "optimizer": "paper_rl_driven",
            "sampling_prior": "paper_direct_rl",
            "num_samples": total_rollouts // iterations,
            "importance_sampling_correction": False,
            "paper_rl_driven": {
                "iterations": iterations,
                "guided_fraction": 0.30,
                "elite_fraction": 0.20,
                "covariance_smoothing": 0.50,
                "covariance_min_scale": 0.25,
                "covariance_max_scale": 2.00,
                "terminal_value_weight": 1.0,
                "terminal_critic_source": "target",
            },
        })
        config["rl"] = {
            "enabled": True,
            "integration": "paper_direct_control",
            "policy_id": "gate1_direct_control_actor",
            "checkpoint": str(actor_checkpoint),
            "device": "cpu",
            "torch_num_threads": 1,
        }
    else:
        planner.update({
            "optimizer": "standard",
            "sampling_prior": "goal_warm_start",
            "num_samples": total_rollouts,
        })
        planner.pop("paper_rl_driven", None)
        config["rl"] = {"enabled": False}
    config["experiment"]["gate1_method"] = str(method)
    config["experiment"]["rollout_budget_per_decision"] = total_rollouts
    config["experiment"]["paper_iterations"] = iterations if use_rl else 1
    config["experiment"]["physics_domain"] = str(domain["name"])
    config["experiment"]["physics_domain_role"] = str(
        domain.get("role", "unknown")
    )
    return config


def aggregate(rows):
    result = []
    for method in [item[0] for item in METHODS]:
        values = [row for row in rows if row["method"] == method]
        if not values:
            continue
        result.append({
            "method": method,
            "seeds": len(values),
            "success_rate": float(np.mean([
                float(row["success"]) for row in values
            ])),
            "collision_rate": float(np.mean([
                float(row["collision"]) for row in values
            ])),
            "final_goal_distance_mean": float(np.mean([
                float(row["final_goal_distance"]) for row in values
            ])),
            "planner_compute_ms_mean": float(np.mean([
                float(row["planner_compute_ms_mean"]) for row in values
            ])),
            "planner_compute_ms_p95_mean": float(np.mean([
                float(row["planner_compute_ms_p95"]) for row in values
            ])),
            "control_jerk_mean": float(np.mean([
                float(row["control_jerk"]) for row in values
            ])),
            "minimum_clearance_min": float(np.min([
                float(row["minimum_clearance"]) for row in values
                if row["minimum_clearance"] is not None
            ])),
        })
    return result


def aggregate_by_domain(rows):
    result = []
    domains = sorted({str(row["physics_domain"]) for row in rows})
    for domain in domains:
        subset = [
            row for row in rows if str(row["physics_domain"]) == domain
        ]
        for summary in aggregate(subset):
            summary["physics_domain"] = domain
            result.append(summary)
    return result


def aggregate_by_scene_domain(rows):
    result = []
    groups = sorted({
        (str(row["scene"]), str(row["physics_domain"]))
        for row in rows
    })
    for scene, domain in groups:
        subset = [
            row for row in rows
            if str(row["scene"]) == scene
            and str(row["physics_domain"]) == domain
        ]
        for summary in aggregate(subset):
            summary["scene"] = scene
            summary["physics_domain"] = domain
            result.append(summary)
    return result


def load_physics_domains(path=None, names=None, roles=None):
    if path is None:
        return [{
            "name": "embedded",
            "role": "embedded",
            "plant_override": {},
        }]
    specification = load_yaml(path)
    domains = [
        dict(item)
        for item in specification.get(
            "physics_domains", {}
        ).get("domains", ())
    ]
    selected_names = set(names or ())
    selected_roles = set(roles or ())
    if selected_names:
        domains = [
            item for item in domains
            if str(item.get("name")) in selected_names
        ]
    if selected_roles:
        domains = [
            item for item in domains
            if str(item.get("role", "unknown")) in selected_roles
        ]
    if not domains:
        raise ValueError("physics-domain selection is empty")
    found = {str(item.get("name")) for item in domains}
    if selected_names - found:
        raise ValueError(
            "unknown physics domains: %s"
            % ", ".join(sorted(selected_names - found))
        )
    return domains


def load_scenes(base, paths=None):
    if not paths:
        return [{
            "name": str(
                base.get("scene", {}).get("name", "embedded")
            ),
            "source": None,
            "config": copy.deepcopy(base),
        }]
    result = []
    for path in paths:
        config = load_yaml(path)
        result.append({
            "name": str(
                config.get("scene", {}).get(
                    "name", Path(path).stem
                )
            ),
            "source": str(Path(path).resolve()),
            "config": config,
        })
    names = [item["name"] for item in result]
    if len(set(names)) != len(names):
        raise ValueError("scene names must be unique")
    return result


def randomized_block_schedule(
    seeds, schedule_seed, physics_domains=None, scenes=None
):
    """Randomize methods within each scene-by-domain-by-seed block."""

    rng = np.random.RandomState(int(schedule_seed))
    domains = physics_domains or [{
        "name": "embedded",
        "role": "embedded",
        "plant_override": {},
    }]
    scene_specs = scenes or [{
        "name": "embedded",
        "source": None,
    }]
    schedule = []
    for scene in scene_specs:
        scene_name = str(scene["name"])
        for domain in domains:
            domain_name = str(domain["name"])
            for seed in seeds:
                order = rng.permutation(len(METHODS))
                for position, index in enumerate(order):
                    method, use_icode, use_rl = METHODS[int(index)]
                    schedule.append({
                        "seed": int(seed),
                        "scene": scene_name,
                        "scene_source": scene.get("source"),
                        "physics_domain": domain_name,
                        "physics_domain_role": str(
                            domain.get("role", "unknown")
                        ),
                        "physics_domain_spec": copy.deepcopy(domain),
                        "block": "%s__%s__seed_%d" % (
                            scene_name, domain_name, int(seed)
                        ),
                        "run_order_within_block": int(position),
                        "method": method,
                        "use_icode": bool(use_icode),
                        "use_rl": bool(use_rl),
                    })
    return schedule


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/research/mujoco_strong_mppi_baseline.yaml"),
    )
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--icode-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--total-rollouts", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--physics-domain-config")
    parser.add_argument(
        "--scene-configs",
        default="",
        help="comma-separated scene configs; empty uses --config",
    )
    parser.add_argument(
        "--physics-domains",
        default="",
        help="comma-separated domain names; empty selects every domain",
    )
    parser.add_argument(
        "--physics-domain-roles",
        default="",
        help="comma-separated roles applied after the name filter",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    base = load_yaml(args.config)
    seeds = parse_ints(args.seeds)
    if args.smoke:
        seeds = seeds[:1]
    domain_names = tuple(
        value.strip()
        for value in args.physics_domains.split(",")
        if value.strip()
    )
    domain_roles = tuple(
        value.strip()
        for value in args.physics_domain_roles.split(",")
        if value.strip()
    )
    domains = load_physics_domains(
        args.physics_domain_config, domain_names, domain_roles
    )
    scene_paths = tuple(
        value.strip()
        for value in args.scene_configs.split(",")
        if value.strip()
    )
    scenes = load_scenes(base, scene_paths)
    if args.smoke:
        domains = domains[:1]
        scenes = scenes[:1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    schedule = randomized_block_schedule(
        seeds, args.schedule_seed, domains, scenes
    )
    scene_by_name = {item["name"]: item for item in scenes}
    for job in schedule:
        config = method_config(
            scene_by_name[job["scene"]]["config"],
            job["method"],
            job["use_icode"],
            job["use_rl"],
            args.actor_checkpoint,
            args.icode_checkpoint,
            args.total_rollouts,
            args.iterations,
            job["seed"],
            job["physics_domain_spec"],
        )
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = int(args.max_steps)
        if args.smoke:
            config["experiment"]["max_steps"] = min(
                int(config["experiment"]["max_steps"]), 8
            )
            config["planner"]["horizon"] = min(
                int(config["planner"]["horizon"]), 8
            )
        run_dir = output / "runs" / config["experiment"]["name"]
        experiment = ExperimentRunner(
            config, ROOT, run_dir, headless=True
        ).run()
        row = dict(experiment.summary)
        row.update({
            "method": job["method"],
            "scene": job["scene"],
            "scene_source": job["scene_source"],
            "seed": int(job["seed"]),
            "block": job["block"],
            "run_order_within_block": int(
                job["run_order_within_block"]
            ),
            "use_icode": bool(job["use_icode"]),
            "use_rl": bool(job["use_rl"]),
            "physics_domain": job["physics_domain"],
            "physics_domain_role": job["physics_domain_role"],
            "rollout_budget_per_decision": int(args.total_rollouts),
            "paper_iterations": int(
                args.iterations if job["use_rl"] else 1
            ),
            "actor_checkpoint": str(Path(args.actor_checkpoint).resolve()),
            "icode_checkpoint": str(Path(args.icode_checkpoint).resolve()),
        })
        rows.append(row)

    summary = aggregate(rows)
    domain_summary = aggregate_by_domain(rows)
    scene_domain_summary = aggregate_by_scene_domain(rows)
    factorial_contrasts = {
        metric: blocked_factorial_contrasts(
            rows,
            metric,
            higher_is_better=higher_is_better,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.schedule_seed + offset,
        )
        for offset, (metric, higher_is_better) in enumerate((
            ("success", True),
            ("final_goal_distance", False),
            ("control_jerk", False),
            ("minimum_clearance", True),
            ("planner_compute_ms_mean", False),
        ))
    }
    with (output / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output / "domain_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(domain_summary[0])
        )
        writer.writeheader()
        writer.writerows(domain_summary)
    with (output / "scene_domain_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(scene_domain_summary[0])
        )
        writer.writeheader()
        writer.writerows(scene_domain_summary)
    with (output / "factorial_contrasts.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            factorial_contrasts, handle, indent=2, sort_keys=True
        )
        handle.write("\n")
    payload = {
        "gate": "Gate 1 paper-faithful Simple Combination",
        "formal_result": not bool(args.smoke),
        "source_config": str(Path(args.config).resolve()),
        "seeds": list(seeds),
        "total_rollouts_per_decision": int(args.total_rollouts),
        "paper_iterations": int(args.iterations),
        "physics_domain_config": (
            None
            if args.physics_domain_config is None
            else str(Path(args.physics_domain_config).resolve())
        ),
        "physics_domains": [
            {
                "name": str(item["name"]),
                "role": str(item.get("role", "unknown")),
                "plant_override": copy.deepcopy(
                    item.get("plant_override", {})
                ),
            }
            for item in domains
        ],
        "scenes": [
            {
                "name": str(item["name"]),
                "source": item["source"],
            }
            for item in scenes
        ],
        "randomization": {
            "design": "randomized_complete_block",
            "block": "paired simulation seed",
            "schedule_seed": int(args.schedule_seed),
            "independent_replicate": "seed x physics domain",
            "schedule": schedule,
        },
        "episodes": rows,
        "summary": summary,
        "domain_summary": domain_summary,
        "scene_domain_summary": scene_domain_summary,
        "factorial_contrasts": factorial_contrasts,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
