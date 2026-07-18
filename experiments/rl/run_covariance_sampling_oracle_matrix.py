#!/usr/bin/env python3
"""Run and aggregate the preregistered L83 covariance oracle pilot."""

import argparse
import csv
import gc
import json
import sys
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.rl.run_covariance_sampling_oracle_diagnostic import (
    SCALES,
    run as run_diagnostic,
)
from mobile_robot_mppi.core.config import git_sha


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("L83 matrix config must be a mapping")
    return value


def _rows(path, block, scene_key, unseen):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    text = {
        "scene", "target_phase", "scale_name", "candidate_sha256",
        "weighted_sequence_sha256",
    }
    integers = {"seed", "anchor_step", "candidate_count", "horizon"}
    output = []
    for row in raw:
        converted = {
            key: value if key in text else int(value) if key in integers else float(value)
            for key, value in row.items()
        }
        converted.update({"block": int(block), "scene_key": scene_key, "unseen": bool(unseen)})
        output.append(converted)
    return output


def _aggregate(rows):
    scale_names = tuple(name for name, _ in SCALES)
    anchor_keys = sorted(set(
        (row["block"], row["scene_key"], row["anchor_step"]) for row in rows
    ))
    lookup = {
        (row["block"], row["scene_key"], row["anchor_step"], row["scale_name"]): row
        for row in rows
    }
    mean_cost = {
        scale: float(np.mean([
            lookup[(*key, scale)]["true_weighted_cost"] for key in anchor_keys
        ]))
        for scale in scale_names
    }
    best_fixed = min(mean_cost, key=mean_cost.get)
    best_counts = Counter()
    per_anchor = []
    for key in anchor_keys:
        candidates = [lookup[(*key, scale)] for scale in scale_names]
        best = min(candidates, key=lambda row: row["true_weighted_cost"])
        best_counts[best["scale_name"]] += 1
        fixed_cost = lookup[(*key, best_fixed)]["true_weighted_cost"]
        baseline_cost = lookup[(*key, "baseline")]["true_weighted_cost"]
        per_anchor.append({
            "block": key[0],
            "scene_key": key[1],
            "anchor_step": key[2],
            "unseen": bool(best["unseen"]),
            "oracle_scale": best["scale_name"],
            "oracle_cost": best["true_weighted_cost"],
            "best_fixed_cost": fixed_cost,
            "baseline_cost": baseline_cost,
            "context_improvement": fixed_cost - best["true_weighted_cost"],
            "baseline_regret": baseline_cost - best["true_weighted_cost"],
        })
    block_improvements = {}
    for block in sorted(set(item["block"] for item in per_anchor)):
        subset = [item for item in per_anchor if item["block"] == block]
        block_improvements[str(block)] = float(np.mean([
            item["context_improvement"] for item in subset
        ]))
    unseen = [item for item in per_anchor if item["unseen"]]
    proportions = {
        scale: float(best_counts[scale] / len(per_anchor)) for scale in scale_names
    }
    active = np.asarray([value for value in proportions.values() if value > 0.0])
    context = float(np.mean([item["context_improvement"] for item in per_anchor]))
    unseen_context = float(np.mean([item["context_improvement"] for item in unseen]))
    summary = {
        "anchor_count": len(per_anchor),
        "mean_cost_by_scale": mean_cost,
        "best_fixed_scale": best_fixed,
        "best_scale_counts": {scale: int(best_counts[scale]) for scale in scale_names},
        "best_scale_proportions": proportions,
        "best_scale_entropy_nats": float(-np.sum(active * np.log(active))),
        "context_oracle_improvement": context,
        "baseline_oracle_regret": float(np.mean([
            item["baseline_regret"] for item in per_anchor
        ])),
        "baseline_optimal_fraction": proportions["baseline"],
        "unseen_context_oracle_improvement": unseen_context,
        "block_context_improvements": block_improvements,
        "positive_context_blocks": int(sum(value > 0.0 for value in block_improvements.values())),
    }
    qualifying_scales = sum(value >= 0.10 for value in proportions.values())
    gate = {
        "context_oracle_positive": context > 0.0,
        "two_of_three_blocks_positive": summary["positive_context_blocks"] >= 2,
        "at_least_two_material_scales": qualifying_scales >= 2,
        "unseen_context_nonnegative": unseen_context >= 0.0,
        "baseline_not_dominant": proportions["baseline"] <= 0.90,
    }
    gate["passed"] = bool(all(gate.values()))
    return per_anchor, summary, gate


def run_matrix(args):
    config_path = Path(args.config).resolve()
    config = _load(config_path)
    phase = dict(config["pilot"])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    manifests = []
    for block in config["model_blocks"]:
        for scene_offset, scene_key in enumerate(phase["scenes"]):
            scene = config["scenes"][scene_key]
            run_dir = output_dir / ("block_%d" % int(block["block"])) / scene_key
            result = run_diagnostic(Namespace(
                scene_config=str(ROOT / scene["config"]),
                icode_checkpoint=str(ROOT / block["icode_checkpoint"]),
                output_dir=str(run_dir),
                seed=int(block["seed"]) + scene_offset,
                anchor_steps=",".join(str(v) for v in phase["anchor_steps"]),
                candidate_count=int(phase["candidate_count"]),
            ))
            result.update({"block": int(block["block"]), "scene_key": scene_key})
            manifests.append(result)
            all_rows.extend(_rows(
                run_dir / "scale_metrics.csv",
                block["block"],
                scene_key,
                scene["unseen"],
            ))
            gc.collect()

    fields = list(all_rows[0].keys())
    with (output_dir / "aggregate_scale_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    per_anchor, summary, gate = _aggregate(all_rows)
    with (output_dir / "per_anchor_oracle.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_anchor[0].keys()))
        writer.writeheader()
        writer.writerows(per_anchor)
    manifest = {
        "design_id": config["design_id"],
        "phase": "pilot",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "config_path": str(config_path),
        "run_count": len(manifests),
        "record_count": len(all_rows),
        "summary": summary,
        "training_upgrade_gate": gate,
        "run_manifests": manifests,
    }
    with (output_dir / "matrix_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"summary": summary, "training_upgrade_gate": gate}, indent=2, sort_keys=True))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/rl/covariance_sampling_oracle_l83.yaml"),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_matrix(args)


if __name__ == "__main__":
    main()
