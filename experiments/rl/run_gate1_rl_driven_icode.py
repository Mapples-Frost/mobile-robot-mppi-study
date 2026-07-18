#!/usr/bin/env python3
"""Run the preregistered Gate-1 RL-Driven ICODE-MPPI factorial."""

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, git_sha, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config


CONDITIONS = {
    "icode_mppi": {
        "optimizer": "standard",
        "rl": False,
        "terminal": False,
    },
    "direct_rl_icode": {
        "optimizer": "standard",
        "rl": True,
        "terminal": False,
    },
    "hybrid_no_rl_icode": {
        "optimizer": "rl_driven",
        "rl": False,
        "terminal": False,
    },
    "hybrid_policy_icode": {
        "optimizer": "rl_driven",
        "rl": True,
        "terminal": False,
    },
    "hybrid_lcb_policy_icode": {
        "optimizer": "rl_driven",
        "rl": True,
        "terminal": False,
        "correction_gate": "lcb",
    },
    "rl_driven_icode": {
        "optimizer": "rl_driven",
        "rl": True,
        "terminal": True,
    },
    "critic_only_icode": {
        "optimizer": "rl_driven",
        "rl": True,
        "terminal": True,
        "fractions": (0.0, 0.50, 0.50),
    },
    "sparse_policy_icode": {
        "optimizer": "rl_driven",
        "rl": True,
        "terminal": False,
        "fractions": (0.05, 0.475, 0.475),
    },
    "sparse_rl_driven_icode": {
        "optimizer": "rl_driven",
        "rl": True,
        "terminal": True,
        "fractions": (0.05, 0.475, 0.475),
    },
}


def _resolved(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty Gate-1 CSV")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_ints(value, default):
    if value is None:
        result = [int(item) for item in default]
    else:
        result = [int(item) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("episode seeds must be nonempty and unique")
    return result


def _parse_conditions(value, configured):
    result = list(configured) if value is None else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    if not result or len(result) != len(set(result)):
        raise ValueError("conditions must be nonempty and unique")
    unknown = sorted(set(result).difference(CONDITIONS))
    if unknown:
        raise ValueError("unknown Gate-1 conditions: %s" % unknown)
    return result


def _condition_config(base, design, block, condition, seed):
    spec = CONDITIONS[condition]
    scene_path = _resolved(design["scene_config"])
    config = copy.deepcopy(_apply_scene_config(base, scene_path))
    domain = dict(design["physics_domain"])
    config["plant"] = deep_merge(
        config["plant"], domain.get("plant_override", {})
    )
    config["experiment"].update({
        "name": "gate1_%s_block%d_seed%d" % (
            condition, int(block["block"]), int(seed)
        ),
        "seed": int(seed),
        "physics_domain": str(domain["name"]),
        "physics_domain_role": str(domain.get("role", "unknown")),
    })
    config.setdefault("memory", {})["enable"] = False
    planner = config["planner"]
    if spec["rl"]:
        sampling_prior = "rl"
    elif spec["optimizer"] == "rl_driven":
        sampling_prior = "hybrid_baseline"
    else:
        sampling_prior = "goal_warm_start"
    planner.update({
        "prediction_mode": "icode_residual",
        "checkpoint": str(block["icode_checkpoint"]),
        "optimizer": spec["optimizer"],
        "sampling_prior": sampling_prior,
        "importance_sampling_correction": spec["optimizer"] == "standard",
    })
    if spec["optimizer"] == "rl_driven":
        planner.setdefault("rl_driven", {})["terminal_value_weight"] = (
            float(base["planner"]["rl_driven"]["terminal_value_weight"])
            if spec["terminal"] else 0.0
        )
        if "fractions" in spec:
            rl_fraction, shifted_fraction, base_fraction = spec["fractions"]
            planner["rl_driven"].update({
                "rl_fraction": float(rl_fraction),
                "shifted_fraction": float(shifted_fraction),
                "base_fraction": float(base_fraction),
            })
    else:
        planner.pop("rl_driven", None)
    config.setdefault("rl", {}).update({
        "enabled": bool(spec["rl"]),
        "checkpoint": str(block["rl_checkpoint"]) if spec["rl"] else None,
        "policy_id": "gate1_%s" % condition,
    })
    config["rl"].setdefault("gate", {}).update({
        "mode": "none",
        "near_goal_fallback_enabled": False,
        "fallback_after_safety_override": False,
        "correction_advantage_gate_mode": spec.get(
            "correction_gate", "none"
        ),
        "correction_advantage_critic_source": "target",
        "correction_advantage_threshold": 0.0,
        "correction_advantage_uncertainty_multiplier": 2.0,
    })
    config["factorial"] = {
        "condition": condition,
        "policy_source": "rl" if spec["rl"] else "none",
        "optimizer": spec["optimizer"],
        "terminal_value": bool(spec["terminal"]),
        "prediction_model": "icode",
    }
    return config


def _task_cost(summary, design):
    weights = dict(design["task_cost"])
    return (
        float(weights["final_goal_distance_weight"])
        * float(summary["final_goal_distance"])
        + float(weights["collision_penalty"])
        * float(bool(summary["collision"]))
        + float(weights["control_jerk_weight"])
        * float(summary["control_jerk"])
        + float(weights["step_weight"])
        * float(summary["steps"])
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/rl/gate1_rl_driven_icode_factorial.yaml"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--model-blocks")
    parser.add_argument("--episode-seeds")
    parser.add_argument("--conditions")
    parser.add_argument("--scene-config")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--allow-sealed-confirmation", action="store_true")
    args = parser.parse_args(argv)

    base = load_yaml(args.config)
    design = copy.deepcopy(base["rl"]["gate1_factorial"])
    if args.scene_config is not None:
        design["scene_config"] = str(_resolved(args.scene_config))
    configured_blocks = []
    for index, raw in enumerate(design["model_blocks"], start=1):
        block = dict(raw)
        block["block"] = index
        block["rl_checkpoint"] = _resolved(block["rl_checkpoint"])
        block["icode_checkpoint"] = _resolved(block["icode_checkpoint"])
        for name in ("rl_checkpoint", "icode_checkpoint"):
            if not block[name].exists():
                raise FileNotFoundError(block[name])
        configured_blocks.append(block)
    selected_blocks = _parse_ints(
        args.model_blocks,
        [item["block"] for item in configured_blocks],
    )
    blocks = [
        item for item in configured_blocks if item["block"] in selected_blocks
    ]
    if len(blocks) != len(selected_blocks):
        raise ValueError("model block selection is out of range")
    seeds = _parse_ints(
        args.episode_seeds, design["development_episode_seeds"]
    )
    sealed = set(int(item) for item in design["sealed_confirmation_episode_seeds"])
    if sealed.intersection(seeds) and not args.allow_sealed_confirmation:
        raise ValueError("sealed confirmation seeds require explicit approval")
    conditions = _parse_conditions(args.conditions, design["conditions"])
    output = _resolved(
        args.output_dir or base["experiment"]["output_dir"]
    )
    output.mkdir(parents=True, exist_ok=True)

    schedule = [
        (block, seed, condition)
        for block in blocks
        for seed in seeds
        for condition in conditions
    ]
    random.Random(int(design["schedule_seed"])).shuffle(schedule)
    schedule_rows = [{
        "run_order": index,
        "model_block": int(block["block"]),
        "rl_training_seed": int(block["rl_seed"]),
        "icode_training_seed": int(block["icode_seed"]),
        "episode_seed": int(seed),
        "condition": condition,
    } for index, (block, seed, condition) in enumerate(schedule)]
    _write_csv(output / "condition_schedule.csv", schedule_rows)

    episodes = []
    for schedule_row, (block, seed, condition) in zip(schedule_rows, schedule):
        config = _condition_config(base, design, block, condition, seed)
        if args.num_samples is not None:
            if args.num_samples <= 0:
                raise ValueError("num_samples must be positive")
            config["planner"]["num_samples"] = int(args.num_samples)
        if args.max_steps is not None:
            if args.max_steps <= 0:
                raise ValueError("max_steps must be positive")
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output / "runs" / ("block_%d" % block["block"])
            / condition / ("seed_%d" % seed)
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and (run_dir / "trajectory.csv").exists():
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary.pop("metadata", None)
            summary.pop("provenance", None)
        else:
            result = ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True
            ).run()
            summary = dict(result.summary)
        row = dict(schedule_row)
        row.update(summary)
        row["task_cost"] = _task_cost(summary, design)
        row["rl_checkpoint"] = (
            str(block["rl_checkpoint"])
            if CONDITIONS[condition]["rl"] else "none"
        )
        row["icode_checkpoint"] = str(block["icode_checkpoint"])
        episodes.append(row)
    _write_csv(output / "episodes.csv", episodes)

    grouped = []
    for condition in conditions:
        rows = [row for row in episodes if row["condition"] == condition]
        grouped.append({
            "condition": condition,
            "episodes": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "collision_rate": sum(bool(row["collision"]) for row in rows) / len(rows),
            "task_cost_mean": sum(float(row["task_cost"]) for row in rows) / len(rows),
            "final_goal_distance_mean": sum(
                float(row["final_goal_distance"]) for row in rows
            ) / len(rows),
            "planner_compute_ms_mean": sum(
                float(row["planner_compute_ms_mean"]) for row in rows
            ) / len(rows),
            "rl_elite_fraction_mean": sum(
                float(row.get("rl_elite_fraction_mean", 0.0)) for row in rows
            ) / len(rows),
            "terminal_q_disagreement_mean": sum(
                float(row.get("terminal_q_disagreement_mean", 0.0))
                for row in rows
            ) / len(rows),
        })
    _write_csv(output / "condition_summary.csv", grouped)
    metadata = {
        "schema_version": 1,
        "preregistration": str(
            ROOT / "docs/rl/157_gate1_rl_driven_icode_mppi_prereg_2026-07-18.md"
        ),
        "source_config": str(_resolved(args.config)),
        "scene_config": str(_resolved(design["scene_config"])),
        "run_git_sha": git_sha(ROOT),
        "model_blocks": [{
            "block": item["block"],
            "rl_seed": item["rl_seed"],
            "icode_seed": item["icode_seed"],
            "rl_checkpoint": str(item["rl_checkpoint"]),
            "rl_checkpoint_sha256": _sha256(item["rl_checkpoint"]),
            "icode_checkpoint": str(item["icode_checkpoint"]),
            "icode_checkpoint_sha256": _sha256(item["icode_checkpoint"]),
        } for item in blocks],
        "episode_seeds": seeds,
        "sealed_confirmation_seeds_used": sorted(sealed.intersection(seeds)),
        "conditions": conditions,
        "num_samples_override": args.num_samples,
        "task_cost": dict(design["task_cost"]),
        "episodes": len(episodes),
        "interpretation_guard": (
            "episode seeds are repeated measures within independent model "
            "blocks; controller steps are technical measurements"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(grouped, indent=2, sort_keys=True))
    print("artifacts:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
