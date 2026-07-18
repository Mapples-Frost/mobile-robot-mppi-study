#!/usr/bin/env python3
"""Evaluate one model block under the exact deployment gate chain."""

import argparse
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

from experiments.rl.run_cross_layer_factorial import (  # noqa: E402
    _condition_config,
    _domain_entries,
    _make_prior,
    _protected_previous_seeds,
    _scene_entries,
    _write_csv,
)
from mobile_robot_mppi.core.config import git_sha, load_yaml  # noqa: E402
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner  # noqa: E402


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seeds(text, default):
    values = default if text is None else [
        int(value) for value in text.split(",") if value.strip()
    ]
    values = [int(value) for value in values]
    if not values or len(values) != len(set(values)):
        raise ValueError("deployment episode seeds must be nonempty and unique")
    return values


def _candidate_entries(design, block):
    selection = design["deployment_checkpoint_selection"]
    run_dir = _resolved(str(selection["run_directory_template"]).format(
        rl_seed=int(block["rl_seed"])
    ))
    entries = [{
        "candidate_id": "base",
        "candidate_step": 0,
        "condition": "complexity_bc_icode",
        "checkpoint": run_dir / "checkpoints" / "initial.pt",
    }]
    for step in selection["candidate_steps"]:
        step = int(step)
        entries.append({
            "candidate_id": "step_%09d" % step,
            "candidate_step": step,
            "condition": "gated_lcb_icode",
            "checkpoint": run_dir / "checkpoints" / ("step_%09d.pt" % step),
        })
    ids = [entry["candidate_id"] for entry in entries]
    if len(ids) != len(set(ids)) or any(entry["candidate_step"] < 0 for entry in entries):
        raise ValueError(
            "deployment checkpoint candidates must be unique and non-negative"
        )
    for entry in entries:
        if not entry["checkpoint"].exists():
            raise FileNotFoundError(str(entry["checkpoint"]))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-block", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-seeds")
    parser.add_argument("--allow-sealed-confirmation", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    config_path = _resolved(args.config)
    base = load_yaml(config_path)
    design = base["rl"]["cross_layer_factorial"]
    study_label = str(design.get("study_label", "L71"))
    artifact_prefix = str(design.get("artifact_prefix", "l71"))
    blocks = list(design["model_blocks"])
    if not 0 <= args.model_block < len(blocks):
        raise ValueError("deployment model block index is out of range")
    block = blocks[args.model_block]
    seeds = _parse_seeds(
        args.episode_seeds, design["development_episode_seeds"]
    )
    sealed = {int(value) for value in design["sealed_confirmation_episode_seeds"]}
    if sealed.intersection(seeds) and not args.allow_sealed_confirmation:
        raise ValueError(
            "sealed %s seeds require explicit confirmation mode" % study_label
        )
    protected = _protected_previous_seeds(design.get("protected_config_paths", ()))
    overlap = protected.intersection(seeds)
    if overlap:
        raise ValueError(
            "%s seeds overlap protected earlier data: %s"
            % (study_label, sorted(overlap))
        )

    scenes = _scene_entries(design)
    domains = _domain_entries(design)
    if len(domains) != 1:
        raise ValueError(
            "deployment selection requires exactly one fixed domain"
        )
    domain = domains[0]
    candidates = _candidate_entries(design, block)
    icode_checkpoint = _resolved(block["icode_checkpoint"])
    if not icode_checkpoint.exists():
        raise FileNotFoundError(str(icode_checkpoint))
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    schedule = [
        {
            "scene": scene,
            "episode_seed": int(seed),
            "candidate": candidate,
        }
        for scene in scenes
        for seed in seeds
        for candidate in candidates
    ]
    schedule_seed = int(design["schedule_seed"]) + int(args.model_block)
    random.Random(schedule_seed).shuffle(schedule)
    schedule_rows = []
    for index, item in enumerate(schedule):
        candidate = item["candidate"]
        schedule_rows.append({
            "run_order": index,
            "model_block": int(args.model_block),
            "rl_training_seed": int(block["rl_seed"]),
            "icode_training_seed": int(block["icode_seed"]),
            "scene": item["scene"]["name"],
            "scene_role": item["scene"]["role"],
            "physics_domain": domain["name"],
            "episode_seed": int(item["episode_seed"]),
            "candidate_id": candidate["candidate_id"],
            "candidate_step": int(candidate["candidate_step"]),
            "condition": candidate["condition"],
            "rl_checkpoint": str(candidate["checkpoint"]),
        })
    _write_csv(output / "candidate_schedule.csv", schedule_rows)

    priors = {}
    for candidate in candidates:
        resolved = _condition_config(
            base,
            scenes[0]["path"],
            domain,
            candidate["condition"],
            candidate["checkpoint"],
            icode_checkpoint,
            seeds[0],
        )
        priors[candidate["candidate_id"]] = _make_prior(
            resolved, candidate["checkpoint"]
        )

    episodes = []
    for index, item in enumerate(schedule):
        candidate = item["candidate"]
        scene = item["scene"]
        seed = int(item["episode_seed"])
        resolved = _condition_config(
            base,
            scene["path"],
            domain,
            candidate["condition"],
            candidate["checkpoint"],
            icode_checkpoint,
            seed,
        )
        resolved["experiment"]["name"] = "%s_%s_%s_seed_%d" % (
            artifact_prefix, candidate["candidate_id"], scene["name"], seed
        )
        if args.max_steps is not None:
            if args.max_steps <= 0:
                raise ValueError("--max-steps must be positive")
            resolved["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output / "runs" / scene["name"] / candidate["candidate_id"]
            / ("seed_%d" % seed)
        )
        metrics_path = run_dir / "metrics.json"
        trajectory_path = run_dir / "trajectory.csv"
        if metrics_path.exists() and trajectory_path.exists():
            episode = json.loads(metrics_path.read_text(encoding="utf-8"))
            episode.pop("metadata", None)
            episode.pop("provenance", None)
        else:
            result = ExperimentRunner(
                resolved,
                ROOT,
                output_dir=run_dir,
                headless=True,
                rl_policy=priors[candidate["candidate_id"]],
            ).run()
            episode = dict(result.summary)
        episode.update(schedule_rows[index])
        episode["icode_checkpoint"] = str(icode_checkpoint)
        episodes.append(episode)
    _write_csv(output / "episodes.csv", episodes)

    metadata = {
        "study_label": study_label,
        "design_id": str(design["design_id"]),
        "model_block": int(args.model_block),
        "rl_training_seed": int(block["rl_seed"]),
        "icode_training_seed": int(block["icode_seed"]),
        "episode_seeds": seeds,
        "sealed_confirmation_seeds_used": sorted(sealed.intersection(seeds)),
        "protected_previous_seeds_used": sorted(protected.intersection(seeds)),
        "schedule_seed": schedule_seed,
        "scenes": [
            {"name": row["name"], "role": row["role"], "path": str(row["path"])}
            for row in scenes
        ],
        "physics_domain": domain,
        "candidates": [{
            "candidate_id": row["candidate_id"],
            "candidate_step": row["candidate_step"],
            "condition": row["condition"],
            "checkpoint": str(row["checkpoint"]),
            "checkpoint_sha256": _sha256(row["checkpoint"]),
        } for row in candidates],
        "icode_checkpoint": str(icode_checkpoint),
        "icode_checkpoint_sha256": _sha256(icode_checkpoint),
        "run_git_sha": git_sha(ROOT),
        "episodes": len(episodes),
        "interpretation_guard": (
            "Candidate checkpoints are repeated evaluations within a fixed "
            "training block. Independent evidence is summarized at the model-block level."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
