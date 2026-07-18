#!/usr/bin/env python3
"""Audit and summarize the preregistered L43 clean path-tracking experiment."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobile_robot_mppi.core.config import load_yaml


CONDITIONS = ("traditional_nominal", "traditional_icode")


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty L43 table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _base_key(row):
    return (
        int(row["model_block"]),
        str(row["scene"]),
        str(row["physics_domain"]),
        int(row["episode_seed"]),
    )


def project_polyline(points, xy, theta):
    """Return per-sample distance, tangent error and monotonic completion."""

    points = np.asarray(points, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("polyline points must have shape [N,2+] with N >= 2")
    if xy.ndim != 2 or xy.shape[1] != 2 or theta.shape != (xy.shape[0],):
        raise ValueError("trajectory must have xy [T,2] and theta [T]")
    points = points[:, :2]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("polyline contains duplicate consecutive points")
    relative = xy[:, None, :] - points[None, :-1, :]
    fractions = np.sum(relative * segments[None, :, :], axis=2)
    fractions /= lengths[None, :] ** 2
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = points[None, :-1, :] + fractions[..., None] * segments[None, :, :]
    distances = np.linalg.norm(xy[:, None, :] - projections, axis=2)
    nearest = np.argmin(distances, axis=1)
    rows = np.arange(xy.shape[0])
    cross_track = distances[rows, nearest]
    tangent = np.arctan2(segments[nearest, 1], segments[nearest, 0])
    heading_error = np.arctan2(np.sin(theta - tangent), np.cos(theta - tangent))
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    progress = cumulative[nearest] + fractions[rows, nearest] * lengths[nearest]
    completion = np.maximum.accumulate(progress) / float(cumulative[-1])
    return cross_track, heading_error, np.clip(completion, 0.0, 1.0)


def _path_specs(design):
    result = {}
    for item in design["scenes"]:
        config = load_yaml(_resolved_path(item["path"]))
        name = str(config["scene"]["name"])
        if str(config["task"].get("type")) != "polyline":
            raise ValueError("L43 scenes must use polyline references")
        result[name] = np.asarray(config["task"]["points"], dtype=np.float64)
    return result


def _tracking_metrics(step_rows, path_specs):
    grouped = defaultdict(list)
    for row in step_rows:
        grouped[_base_key(row) + (str(row["condition"]),)].append(row)
    results = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["step"]))
        xy = np.asarray([[float(row["x"]), float(row["y"])] for row in rows])
        theta = np.asarray([float(row["theta"]) for row in rows])
        cross_track, heading_error, completion = project_polyline(
            path_specs[key[1]], xy, theta
        )
        if not (
            np.isfinite(cross_track).all()
            and np.isfinite(heading_error).all()
            and np.isfinite(completion).all()
        ):
            raise FloatingPointError("nonfinite L43 path metric")
        results.append({
            "model_block": key[0],
            "scene": key[1],
            "physics_domain": key[2],
            "episode_seed": key[3],
            "condition": key[4],
            "steps": len(rows),
            "cross_track_rmse_m": float(np.sqrt(np.mean(cross_track ** 2))),
            "cross_track_p95_m": float(np.percentile(cross_track, 95)),
            "cross_track_max_m": float(np.max(cross_track)),
            "tangent_heading_rmse_rad": float(np.sqrt(np.mean(heading_error ** 2))),
            "completion_ratio": float(completion[-1]),
        })
    return results


def _hierarchical_bootstrap(rows, field, seed, replicates):
    """Resample model blocks, scene/domain cells, then episode seeds."""

    hierarchy = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cell = (str(row["scene"]), str(row["physics_domain"]))
        hierarchy[int(row["model_block"])][cell].append(float(row[field]))
    blocks = sorted(hierarchy)
    if not blocks:
        raise ValueError("bootstrap requires paired effects")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        block_means = []
        for block in sampled_blocks:
            cells = sorted(hierarchy[int(block)])
            sampled_cells = rng.choice(len(cells), size=len(cells), replace=True)
            cell_means = []
            for cell_index in sampled_cells:
                values = np.asarray(hierarchy[int(block)][cells[int(cell_index)]])
                cell_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
            block_means.append(float(np.mean(cell_means)))
        samples[index] = float(np.mean(block_means))
    estimate = float(np.mean([
        np.mean([value for values in hierarchy[block].values() for value in values])
        for block in blocks
    ]))
    return {
        "estimate": estimate,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
    }


def summarize(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    episode_rows = []
    step_rows = []
    protected = set()
    metadata_sealed = set()
    for block in range(len(design["model_blocks"])):
        block_dir = Path(input_dir) / ("block_%d" % block)
        episode_rows.extend(_read_csv(block_dir / "episodes.csv"))
        step_rows.extend(_read_csv(block_dir / "factorial_steps.csv"))
        metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
        protected.update(int(value) for value in metadata["previous_protected_seeds_used"])
        metadata_sealed.update(int(value) for value in metadata["sealed_confirmation_seeds_used"])
    path_specs = _path_specs(design)
    tracking_rows = _tracking_metrics(step_rows, path_specs)
    tracking_lookup = {
        _base_key(row) + (str(row["condition"]),): row for row in tracking_rows
    }
    episode_lookup = {
        _base_key(row) + (str(row["condition"]),): row for row in episode_rows
    }
    expected = {
        (block, scene, str(domain["name"]), int(seed), condition)
        for block in range(len(design["model_blocks"]))
        for scene in path_specs
        for domain in design["physics_domains"]
        for seed in design["development_episode_seeds"]
        for condition in CONDITIONS
    }
    observed = set(episode_lookup)
    effects = []
    if observed == expected and set(tracking_lookup) == expected:
        for base in sorted(key[:-1] for key in expected if key[-1] == CONDITIONS[0]):
            nominal = tracking_lookup[base + (CONDITIONS[0],)]
            icode = tracking_lookup[base + (CONDITIONS[1],)]
            nominal_episode = episode_lookup[base + (CONDITIONS[0],)]
            icode_episode = episode_lookup[base + (CONDITIONS[1],)]
            nominal_rmse = float(nominal["cross_track_rmse_m"])
            icode_rmse = float(icode["cross_track_rmse_m"])
            effects.append({
                "model_block": base[0],
                "scene": base[1],
                "physics_domain": base[2],
                "episode_seed": base[3],
                "cross_track_improvement_m": nominal_rmse - icode_rmse,
                "relative_cross_track_reduction": (nominal_rmse - icode_rmse) / max(nominal_rmse, 1e-12),
                "heading_rmse_improvement_rad": float(nominal["tangent_heading_rmse_rad"]) - float(icode["tangent_heading_rmse_rad"]),
                "completion_ratio_difference": float(icode["completion_ratio"]) - float(nominal["completion_ratio"]),
                "success_difference": int(_bool(icode_episode["success"])) - int(_bool(nominal_episode["success"])),
                "collision_difference": int(_bool(icode_episode["collision"])) - int(_bool(nominal_episode["collision"])),
            })
    icode_episodes = [row for row in episode_rows if row["condition"] == CONDITIONS[1]]
    per_block = []
    for block in range(len(design["model_blocks"])):
        selected = [row for row in effects if row["model_block"] == block]
        per_block.append({
            "model_block": block,
            "pairs": len(selected),
            "mean_cross_track_improvement_m": float(np.mean([row["cross_track_improvement_m"] for row in selected])),
            "mean_relative_cross_track_reduction": float(np.mean([row["relative_cross_track_reduction"] for row in selected])),
            "mean_heading_rmse_improvement_rad": float(np.mean([row["heading_rmse_improvement_rad"] for row in selected])),
            "net_success_gain": int(sum(row["success_difference"] for row in selected)),
            "net_collision_increase": int(sum(row["collision_difference"] for row in selected)),
        })
    bootstrap = _hierarchical_bootstrap(
        effects,
        "cross_track_improvement_m",
        int(design["bootstrap_seed"]),
        int(design["bootstrap_replicates"]),
    )
    pooled = {
        "pairs": len(effects),
        "mean_cross_track_improvement_m": float(np.mean([row["cross_track_improvement_m"] for row in effects])),
        "mean_relative_cross_track_reduction": float(np.mean([row["relative_cross_track_reduction"] for row in effects])),
        "mean_heading_rmse_improvement_rad": float(np.mean([row["heading_rmse_improvement_rad"] for row in effects])),
        "mean_completion_ratio_difference": float(np.mean([row["completion_ratio_difference"] for row in effects])),
        "net_success_gain": int(sum(row["success_difference"] for row in effects)),
        "net_collision_increase": int(sum(row["collision_difference"] for row in effects)),
        "icode_mean_planner_compute_ms": float(np.mean([float(row["planner_compute_ms_mean"]) for row in icode_episodes])),
    }
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    used_sealed = sorted({int(row["episode_seed"]) for row in episode_rows} & sealed)
    confirmation_mode = bool(design.get("confirmation_mode", False))
    expected_confirmation_seeds = sorted(
        int(value) for value in design["development_episode_seeds"]
    )
    seed_integrity = bool(
        (
            used_sealed == expected_confirmation_seeds
            and sorted(metadata_sealed) == expected_confirmation_seeds
        )
        if confirmation_mode
        else (not used_sealed and not metadata_sealed)
    )
    nonfinite = any(
        not math.isfinite(float(row[field]))
        for row in tracking_rows
        for field in (
            "cross_track_rmse_m",
            "cross_track_p95_m",
            "cross_track_max_m",
            "tangent_heading_rmse_rad",
            "completion_ratio",
        )
    )
    integrity = bool(
        observed == expected
        and set(tracking_lookup) == expected
        and not protected
        and seed_integrity
        and not nonfinite
    )
    gate_cfg = design["eligibility_gate"]
    gate = {
        "artifact_integrity": integrity,
        "positive_tracking_model_blocks": int(sum(
            row["mean_cross_track_improvement_m"] > 0.0 for row in per_block
        )),
        "relative_cross_track_rmse_reduction": pooled["mean_relative_cross_track_reduction"],
        "cross_track_improvement_ci95_lower_m": bootstrap["ci95_lower"],
        "net_success_gain": pooled["net_success_gain"],
        "net_collision_increase": pooled["net_collision_increase"],
        "completion_ratio_difference": pooled["mean_completion_ratio_difference"],
        "icode_mean_planner_compute_ms": pooled["icode_mean_planner_compute_ms"],
    }
    gate["passed"] = bool(
        gate["artifact_integrity"]
        and gate["positive_tracking_model_blocks"] >= int(gate_cfg["minimum_positive_tracking_model_blocks"])
        and gate["relative_cross_track_rmse_reduction"] >= float(gate_cfg["minimum_relative_cross_track_rmse_reduction"])
        and gate["cross_track_improvement_ci95_lower_m"] > float(gate_cfg["minimum_cross_track_improvement_ci95_lower_m"])
        and gate["net_success_gain"] >= int(gate_cfg["minimum_net_success_gain"])
        and gate["net_collision_increase"] <= int(gate_cfg["maximum_net_collision_increase"])
        and gate["completion_ratio_difference"] >= float(gate_cfg["minimum_completion_ratio_difference"])
        and gate["icode_mean_planner_compute_ms"] <= float(gate_cfg["maximum_icode_mean_planner_compute_ms"])
    )
    summary = {
        "design_id": str(design["design_id"]),
        "audit": {
            "expected_episodes": len(expected),
            "observed_episodes": len(episode_rows),
            "missing_keys": len(expected - observed),
            "unexpected_keys": len(observed - expected),
            "protected_seeds_used": sorted(protected),
            "sealed_seeds_used": used_sealed,
            "metadata_sealed_seeds_used": sorted(metadata_sealed),
            "confirmation_mode": confirmation_mode,
            "seed_integrity": seed_integrity,
            "nonfinite": nonfinite,
        },
        "per_model_block": per_block,
        "pooled": pooled,
        "hierarchical_bootstrap": {"cross_track_improvement_m": bootstrap},
        "eligibility_gate": gate,
        "interpretation_guard": (
            "Independent sealed-seed confirmation."
            if confirmation_mode else
            "Residual path tracking on development seeds; not confirmation evidence."
        ),
    }
    _write_csv(Path(input_dir) / "path_tracking_episode_metrics.csv", tracking_rows)
    _write_csv(Path(input_dir) / "path_tracking_paired_effects.csv", effects)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    summary = summarize(config, input_dir)
    (input_dir / "path_tracking_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
