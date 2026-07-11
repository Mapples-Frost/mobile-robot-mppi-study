#!/usr/bin/env python3
"""Run the unified A-E residual-MPPI clean-dynamics benchmark.

Smoke mode uses five seeds and reduced sampling.  Formal sample-efficiency
sweeps are available through ``--sample-sweep`` and the configured
50/100/200/400 sample counts.  This script never bypasses an obstacle pipeline:
it is explicitly an obstacle-free dynamics-isolation benchmark.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.icode import run_oracle_residual_ablation as oracle_runner
from src.dynamics import DisturbanceConfig, DisturbedUnicycle
from src.planners.mppi_dynamics_adapter import build_prediction_dynamics


DEFAULT_CONFIG = ROOT / "configs" / "icode" / "mppi_residual_benchmark.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark YAML must contain a mapping")
    return copy.deepcopy(dict(payload))


def _path(value: Any) -> Path:
    candidate = Path(str(value)).expanduser()
    return (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _sha256(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _neutral_plant_config() -> Dict[str, Any]:
    return DisturbanceConfig().to_dict()


def _extended_metrics(
    summary: Mapping[str, Any], detail: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    states = [tuple(float(v) for v in state) for state in detail["states"]]
    controls = [tuple(float(v) for v in control) for control in detail["executed_controls"]]
    dt = float(config["benchmark"]["dt"])
    thresholds = config["event_metrics"]
    displacements = [
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(states[:-1], states[1:])
    ]
    stuck = sum(
        distance < float(thresholds["stuck_displacement_threshold"])
        for distance in displacements
    )
    spin = sum(
        distance < float(thresholds["spin_displacement_threshold"])
        and abs(control[1]) >= float(thresholds["spin_yaw_rate_threshold"])
        for distance, control in zip(displacements, controls)
    )
    jerks = [
        math.hypot((second[0] - first[0]) / dt, (second[1] - first[1]) / dt)
        for first, second in zip(controls[:-1], controls[1:])
    ]
    result = dict(summary)
    result.update(
        {
            "collision": int(bool(summary["bounds_violation"])),
            "minimum_clearance": None,
            "mean_absolute_yaw_rate": statistics.mean(abs(control[1]) for control in controls)
            if controls
            else 0.0,
            "control_jerk": statistics.mean(jerks) if jerks else 0.0,
            "stuck_steps": int(stuck),
            "spin_steps": int(spin),
            "maximum_planner_compute_ms": float(summary["max_planner_compute_ms"]),
            "minimum_clearance_note": "not_applicable_obstacle_free",
        }
    )
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    metrics = (
        "final_goal_distance",
        "path_length",
        "mean_absolute_yaw_rate",
        "control_jerk",
        "stuck_steps",
        "spin_steps",
        "mean_planner_compute_ms",
        "maximum_planner_compute_ms",
    )
    for key in sorted({(str(row["method"]), int(row["num_samples"])) for row in rows}):
        method, samples = key
        selected = [
            row for row in rows if row["method"] == method and int(row["num_samples"]) == samples
        ]
        entry: Dict[str, Any] = {
            "method": method,
            "num_samples": samples,
            "episodes": len(selected),
            "success_rate": statistics.mean(float(row["success"]) for row in selected),
            "collision_rate": statistics.mean(float(row["collision"]) for row in selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            entry[metric + "_mean"] = statistics.mean(values)
            entry[metric + "_population_std"] = statistics.pstdev(values)
        result["{}@{}".format(method, samples)] = entry
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, episodes: Sequence[Mapping[str, Any]], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    for episode in episodes:
        states = episode["states"]
        axis.plot(
            [state[0] for state in states],
            [state[1] for state in states],
            alpha=0.55,
            label="{} seed={}".format(episode["method"], episode["seed"]),
        )
    axis.scatter([0.0, 3.0], [0.0, 3.0], c=["black", "red"], marker="x")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if len(episodes) <= 12:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument("--icode-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sample-sweep", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    specification = _load_yaml(args.config.resolve())
    if int(specification.get("schema_version", -1)) != 1:
        raise ValueError("unsupported benchmark schema")
    base_path = _path(specification["base_config"])
    base = oracle_runner.load_config(base_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _path(specification["run"]["output_dir"])
    )
    resolved = oracle_runner.resolve_config(
        base,
        config_path=base_path,
        output_dir=output_dir,
        smoke=args.smoke,
        headless=args.headless,
    )
    resolved["event_metrics"] = copy.deepcopy(specification["event_metrics"])
    resolved["run"]["effective_seeds"] = list(
        specification["run"]["smoke_seeds" if args.smoke else "seeds"]
    )
    resolved["run"]["run_type"] = "smoke" if args.smoke else "research"
    if args.smoke:
        resolved["planner"]["horizon"] = int(specification["smoke"]["horizon"])
        resolved["benchmark"]["max_steps"] = int(specification["smoke"]["max_steps"])

    available = list(specification["methods"])
    selected = available if args.methods == "all" else [item.strip() for item in args.methods.split(",")]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError("unknown methods: {}".format(", ".join(unknown)))
    if "mlp_residual" in selected and args.mlp_checkpoint is None:
        raise ValueError("mlp_residual requires --mlp-checkpoint")
    if "icode_residual" in selected and args.icode_checkpoint is None:
        raise ValueError("icode_residual requires --icode-checkpoint")

    configured_samples = specification["sample_efficiency"][
        "smoke_num_samples" if args.smoke else "formal_num_samples"
    ]
    if args.num_samples is not None:
        sample_counts = [args.num_samples]
    elif args.sample_sweep or args.smoke:
        sample_counts = [int(value) for value in configured_samples]
    else:
        sample_counts = [int(resolved["planner"]["num_samples"])]
    if any(value <= 0 for value in sample_counts):
        raise ValueError("num_samples values must be positive")

    adapters: Dict[str, Any] = {}
    nominal_adapter = build_prediction_dynamics(
        "nominal", integration_method=resolved["benchmark"]["integration_method"], clone_per_rollout=False
    )
    adapters["nominal_no_mismatch"] = nominal_adapter
    adapters["nominal_mismatch"] = nominal_adapter
    if "oracle_residual" in selected:
        oracle_plant = DisturbedUnicycle(oracle_runner.disturbance_config(resolved))
        adapters["oracle_residual"] = build_prediction_dynamics(
            "oracle_residual",
            true_dynamics=oracle_plant,
            integration_method=resolved["benchmark"]["integration_method"],
            clone_per_rollout=False,
        )
    for method, checkpoint in (
        ("mlp_residual", args.mlp_checkpoint),
        ("icode_residual", args.icode_checkpoint),
    ):
        if method in selected and checkpoint is not None:
            adapters[method] = build_prediction_dynamics(
                method,
                checkpoint_path=checkpoint.expanduser().resolve(),
                integration_method=resolved["benchmark"]["integration_method"],
                clone_per_rollout=False,
                device=args.device,
            )

    git_sha = oracle_runner.get_git_sha(ROOT)
    summaries: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    for samples in sample_counts:
        for method in selected:
            episode_config = copy.deepcopy(resolved)
            episode_config["planner"]["num_samples"] = samples
            if method == "nominal_no_mismatch":
                episode_config["plant"] = _neutral_plant_config()
            for seed in resolved["run"]["effective_seeds"]:
                summary, rows, detail = oracle_runner.run_episode(
                    method, adapters[method], int(seed), episode_config, git_sha
                )
                extended = _extended_metrics(summary, detail, episode_config)
                extended["method"] = method
                extended["num_samples"] = samples
                summaries.append(extended)
                for row in rows:
                    row = dict(row)
                    row["method"] = method
                    row["num_samples"] = samples
                    trajectories.append(row)
                detail = dict(detail)
                detail["method"] = method
                detail["num_samples"] = samples
                episodes.append(detail)
                print(
                    "method={} samples={} seed={} success={} distance={:.4f} plan_ms={:.2f}".format(
                        method,
                        samples,
                        seed,
                        bool(extended["success"]),
                        extended["final_goal_distance"],
                        extended["mean_planner_compute_ms"],
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(summaries)
    manifest = {
        "schema_version": 1,
        "run_type": resolved["run"]["run_type"],
        "formal_results": False if args.smoke else None,
        "smoke_disclaimer": "Framework validation only; not formal results." if args.smoke else None,
        "git_sha": git_sha,
        "goal": [3.0, 3.0],
        "scene": "clean_dynamics",
        "obstacle_free": True,
        "memory_enabled": False,
        "retained_mujoco_scenarios": specification["scenarios"]["retained_mujoco_scenarios"],
        "methods": selected,
        "seeds": resolved["run"]["effective_seeds"],
        "num_samples": sample_counts,
        "checkpoints": {
            "mlp": None if args.mlp_checkpoint is None else str(args.mlp_checkpoint.resolve()),
            "icode": None if args.icode_checkpoint is None else str(args.icode_checkpoint.resolve()),
        },
        "checkpoint_sha256": {"mlp": _sha256(args.mlp_checkpoint), "icode": _sha256(args.icode_checkpoint)},
    }
    metrics = {"manifest": manifest, "aggregate": aggregate, "episodes": episodes}
    _atomic_text(output_dir / "metrics.json", json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_text(output_dir / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_text(output_dir / "config_snapshot.yaml", yaml.safe_dump(resolved, sort_keys=False))
    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(output_dir / "trajectory.csv", trajectories)
    _write_csv(output_dir / "aggregate.csv", list(aggregate.values()))
    _plot(
        output_dir / "trajectories.png",
        episodes,
        "Residual MPPI {} | seeds={} | samples={}".format(
            manifest["run_type"], len(manifest["seeds"]), sample_counts
        ),
    )
    print("metrics={}".format(output_dir / "metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
