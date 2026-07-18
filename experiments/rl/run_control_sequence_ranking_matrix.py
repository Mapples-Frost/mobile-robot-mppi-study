#!/usr/bin/env python3
"""Run and aggregate the preregistered L82 model-block/scene matrix."""

import argparse
import csv
import gc
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.rl.run_control_sequence_ranking_diagnostic import run as run_diagnostic
from mobile_robot_mppi.core.config import git_sha


def _read_config(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("matrix config must be a mapping")
    return config


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_float_rows(rows):
    integer_fields = {
        "block", "seed", "anchor_step", "candidate_count", "horizon",
        "top1_hit", "top5_hit", "predicted_best_index", "true_best_index",
        "elite_count",
    }
    text_fields = {"scene", "model", "candidate_sha256"}
    output = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if key in text_fields:
                converted[key] = value
            elif key in integer_fields:
                converted[key] = int(value)
            else:
                converted[key] = float(value)
        output.append(converted)
    return output


def _paired_summary(rows, unseen_scenes):
    lookup = {
        (row["block"], row["scene"], row["anchor_step"], row["model"]): row
        for row in rows
    }
    pairs = []
    keys = sorted(
        set((row["block"], row["scene"], row["anchor_step"]) for row in rows)
    )
    for block, scene, anchor in keys:
        mlp = lookup[(block, scene, anchor, "mlp")]
        icode = lookup[(block, scene, anchor, "icode")]
        pairs.append({
            "block": block,
            "scene": scene,
            "anchor_step": anchor,
            "unseen": scene in unseen_scenes,
            "spearman_difference": icode["spearman"] - mlp["spearman"],
            "elite_recall_difference": icode["elite_recall"] - mlp["elite_recall"],
            "normalized_regret_difference": (
                icode["normalized_regret"] - mlp["normalized_regret"]
            ),
            "weight_js_difference": (
                icode["weight_js_divergence"] - mlp["weight_js_divergence"]
            ),
        })

    def mean(field, subset):
        return float(np.mean([item[field] for item in subset]))

    block_summaries = []
    for block in sorted(set(item["block"] for item in pairs)):
        subset = [item for item in pairs if item["block"] == block]
        block_summaries.append({
            "block": block,
            "spearman_difference": mean("spearman_difference", subset),
            "elite_recall_difference": mean("elite_recall_difference", subset),
            "normalized_regret_difference": mean(
                "normalized_regret_difference", subset
            ),
            "weight_js_difference": mean("weight_js_difference", subset),
        })
    unseen = [item for item in pairs if item["unseen"]]
    overall = {
        "spearman_difference": mean("spearman_difference", pairs),
        "elite_recall_difference": mean("elite_recall_difference", pairs),
        "normalized_regret_difference": mean(
            "normalized_regret_difference", pairs
        ),
        "weight_js_difference": mean("weight_js_difference", pairs),
        "positive_spearman_blocks": int(
            sum(item["spearman_difference"] > 0.0 for item in block_summaries)
        ),
        "unseen_spearman_difference": mean("spearman_difference", unseen),
        "unseen_elite_recall_difference": mean(
            "elite_recall_difference", unseen
        ),
        "unseen_normalized_regret_difference": mean(
            "normalized_regret_difference", unseen
        ),
    }
    return pairs, block_summaries, overall


def run_matrix(args):
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    phase = dict(config["phases"][args.phase])
    scene_names = tuple(str(name) for name in phase["scenes"])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    run_manifests = []

    for block in config["model_blocks"]:
        block_index = int(block["block"])
        for scene_offset, scene_name in enumerate(scene_names):
            scene = config["scenes"][scene_name]
            run_dir = output_dir / ("block_%d" % block_index) / scene_name
            diagnostic_args = Namespace(
                scene_config=str(ROOT / scene["config"]),
                icode_checkpoint=str(ROOT / block["icode_checkpoint"]),
                mlp_checkpoint=str(ROOT / block["mlp_checkpoint"]),
                output_dir=str(run_dir),
                seed=int(block["seed"]) + int(scene_offset),
                anchor_steps=",".join(str(value) for value in phase["anchor_steps"]),
                candidate_count=int(phase["candidate_count"]),
                candidate_noise_scale=float(phase["candidate_noise_scale"]),
            )
            manifest = run_diagnostic(diagnostic_args)
            manifest["block"] = block_index
            manifest["scene_key"] = scene_name
            run_manifests.append(manifest)
            rows = _as_float_rows(_read_rows(run_dir / "ranking_metrics.csv"))
            for row in rows:
                row["block"] = block_index
                row["scene"] = scene_name
            all_rows.extend(rows)
            gc.collect()

    aggregate_fields = list(all_rows[0].keys())
    with (output_dir / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    unseen_scenes = {
        name for name in scene_names if bool(config["scenes"][name]["unseen"])
    }
    pairs, block_summaries, overall = _paired_summary(all_rows, unseen_scenes)
    with (output_dir / "paired_icode_minus_mlp.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)

    direction_gate = {
        "mean_spearman_positive": overall["spearman_difference"] > 0.0,
        "at_least_two_positive_blocks": overall["positive_spearman_blocks"] >= 2,
        "elite_recall_noninferior": overall["elite_recall_difference"] >= 0.0,
        "normalized_regret_noninferior": (
            overall["normalized_regret_difference"] <= 0.0
        ),
        "unseen_direction_consistent": (
            overall["unseen_spearman_difference"] > 0.0
            and overall["unseen_elite_recall_difference"] >= 0.0
            and overall["unseen_normalized_regret_difference"] <= 0.0
        ),
    }
    direction_gate["passed"] = bool(all(direction_gate.values()))
    manifest = {
        "design_id": str(config["design_id"]),
        "phase": str(args.phase),
        "formal_gate_evaluated": args.phase == "formal",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "config_path": str(config_path),
        "model_block_count": len(config["model_blocks"]),
        "scene_count": len(scene_names),
        "anchor_count_per_scene": len(phase["anchor_steps"]),
        "candidate_count": int(phase["candidate_count"]),
        "record_count": len(all_rows),
        "unseen_scenes": sorted(unseen_scenes),
        "block_summaries": block_summaries,
        "overall_icode_minus_mlp": overall,
        "direction_gate": direction_gate,
        "run_manifests": run_manifests,
    }
    with (output_dir / "matrix_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "phase": args.phase,
        "overall_icode_minus_mlp": overall,
        "direction_gate": direction_gate,
    }, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/rl/control_sequence_ranking_l82.yaml"),
    )
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_matrix(args)


if __name__ == "__main__":
    main()
