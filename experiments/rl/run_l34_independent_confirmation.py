#!/usr/bin/env python3
"""Run the preregistered L35 independent confirmation of L34 checkpoints."""

import argparse
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

from experiments.rl.evaluate_rl_sampling_prior import main as evaluate_main
from mobile_robot_mppi.core.config import git_sha, load_yaml


CHECKPOINT_ROLES = ("initial", "best")


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty confirmation table")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_step(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("training_state", {}).get("global_step", 0))


def _confirmation_design(config_path):
    config = load_yaml(_resolved_path(config_path))
    design = dict(config["rl"]["confirmation"])
    seeds = [int(value) for value in design["episode_seeds"]]
    forbidden = {int(value) for value in design.get("forbidden_episode_seeds", ())}
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("L35 episode seeds must be nonempty and unique")
    overlap = forbidden.intersection(seeds)
    if overlap:
        raise ValueError("L35 confirmation seeds overlap development seeds: %s" % sorted(overlap))
    if float(design["fixed_alpha"]) != 0.25:
        raise ValueError("L35 fixed alpha must remain preregistered at 0.25")
    if str(design["prediction_mode"]) != "nominal":
        raise ValueError("L35 confirmation must isolate RL with nominal rollout dynamics")
    runs = list(design["training_runs"])
    training_seeds = [int(item["training_seed"]) for item in runs]
    if len(runs) != 3 or len(training_seeds) != len(set(training_seeds)):
        raise ValueError("L35 requires exactly three unique training seeds")
    scenes = [_resolved_path(value) for value in design["scenes"]]
    if len(scenes) != 2 or len(scenes) != len(set(scenes)):
        raise ValueError("L35 requires exactly two unique held-out motion paths")
    design["episode_seeds"] = seeds
    design["training_runs"] = runs
    design["scenes"] = scenes
    return config, design


def _schedule(design):
    jobs = []
    for raw_run in design["training_runs"]:
        run_dir = _resolved_path(raw_run["path"])
        for role in CHECKPOINT_ROLES:
            checkpoint = run_dir / "checkpoints" / (role + ".pt")
            if not checkpoint.exists():
                raise FileNotFoundError(str(checkpoint))
            for scene in design["scenes"]:
                if not scene.exists():
                    raise FileNotFoundError(str(scene))
                scene_config = load_yaml(scene)
                jobs.append({
                    "training_seed": int(raw_run["training_seed"]),
                    "expected_best_step": int(raw_run["best_step"]),
                    "checkpoint_role": role,
                    "checkpoint": checkpoint,
                    "scene": scene,
                    "scene_name": str(scene_config["scene"]["name"]),
                })
    random.Random(int(design["schedule_seed"])).shuffle(jobs)
    return jobs


def _complete_evaluation(output_dir, seeds, checkpoint, fixed_alpha):
    required = (
        output_dir / "episodes.csv",
        output_dir / "summary.json",
        output_dir / "evaluation_config_snapshot.json",
    )
    if not all(path.exists() for path in required):
        return False
    snapshot = json.loads(required[2].read_text(encoding="utf-8"))
    if [int(value) for value in snapshot.get("episode_seeds", ())] != list(seeds):
        return False
    if Path(snapshot.get("checkpoint", "")).resolve() != checkpoint.resolve():
        return False
    resolved = snapshot.get("resolved_config", {})
    gate = resolved.get("rl", {}).get("gate", {})
    if gate.get("mode") != "fixed" or float(gate.get("fixed_alpha", -1.0)) != float(fixed_alpha):
        return False
    if resolved.get("planner", {}).get("prediction_mode") != "nominal":
        return False
    rows = _read_csv(required[0])
    return sorted(int(row["seed"]) for row in rows) == sorted(seeds)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    config_path = _resolved_path(args.config)
    _, design = _confirmation_design(config_path)
    output = _resolved_path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("L35 output exists; pass --resume to audit and continue")
    output.mkdir(parents=True, exist_ok=True)
    jobs = _schedule(design)
    seeds = list(design["episode_seeds"])
    fixed_alpha = float(design["fixed_alpha"])

    schedule_rows = []
    checkpoint_manifest = {}
    for index, job in enumerate(jobs):
        step = _checkpoint_step(job["checkpoint"])
        if job["checkpoint_role"] == "best" and step != job["expected_best_step"]:
            raise ValueError(
                "best checkpoint step mismatch for training seed %d: %d != %d"
                % (job["training_seed"], step, job["expected_best_step"])
            )
        checkpoint_key = "%d_%s" % (job["training_seed"], job["checkpoint_role"])
        checkpoint_manifest[checkpoint_key] = {
            "path": str(job["checkpoint"]),
            "sha256": _sha256(job["checkpoint"]),
            "global_step": step,
        }
        schedule_rows.append({
            "run_order": index,
            "training_seed": job["training_seed"],
            "checkpoint_role": job["checkpoint_role"],
            "checkpoint_global_step": step,
            "scene": job["scene_name"],
            "episode_seed_count": len(seeds),
            "episode_seeds": ",".join(str(value) for value in seeds),
        })

    _write_csv(output / "condition_schedule.csv", schedule_rows)
    manifest = {
        "schema_version": 1,
        "design_id": str(design["design_id"]),
        "source_config": str(config_path),
        "run_git_sha": git_sha(ROOT),
        "fixed_alpha": fixed_alpha,
        "prediction_mode": str(design["prediction_mode"]),
        "episode_seeds": seeds,
        "forbidden_episode_seeds": [
            int(value) for value in design.get("forbidden_episode_seeds", ())
        ],
        "physics_domain_config": str(_resolved_path(design["physics_domain_config"])),
        "physics_domain": str(design["physics_domain"]),
        "scenes": [str(value) for value in design["scenes"]],
        "checkpoint_manifest": checkpoint_manifest,
        "expected_jobs": len(jobs),
        "expected_episodes": len(jobs) * len(seeds),
        "interpretation_guard": (
            "training seeds are independent model replicates; scene-episode cells "
            "are paired repeated measures; control steps are not replicates"
        ),
    }
    (output / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_rows = []
    for job in jobs:
        evaluation_dir = (
            output / "runs" / ("training_seed_%d" % job["training_seed"])
            / job["checkpoint_role"] / job["scene_name"]
        )
        if not _complete_evaluation(
            evaluation_dir, seeds, job["checkpoint"], fixed_alpha
        ):
            evaluate_main([
                "--config", str(config_path),
                "--scene-config", str(job["scene"]),
                "--physics-domain-config", str(_resolved_path(design["physics_domain_config"])),
                "--physics-domain", str(design["physics_domain"]),
                "--checkpoint", str(job["checkpoint"]),
                "--output-dir", str(evaluation_dir),
                "--seeds", ",".join(str(value) for value in seeds),
                "--gate-mode", "fixed",
                "--fixed-alpha", str(fixed_alpha),
                "--correction-advantage-gate-mode", "none",
                "--prediction-mode", "nominal",
            ])
        if not _complete_evaluation(
            evaluation_dir, seeds, job["checkpoint"], fixed_alpha
        ):
            raise RuntimeError("incomplete L35 evaluation at %s" % evaluation_dir)
        for raw in _read_csv(evaluation_dir / "episodes.csv"):
            row = dict(raw)
            row.update({
                "training_seed": job["training_seed"],
                "checkpoint_role": job["checkpoint_role"],
                "checkpoint_global_step": _checkpoint_step(job["checkpoint"]),
                "checkpoint_sha256": checkpoint_manifest[
                    "%d_%s" % (job["training_seed"], job["checkpoint_role"])
                ]["sha256"],
                "scene": job["scene_name"],
                "physics_domain": str(design["physics_domain"]),
                "fixed_alpha": fixed_alpha,
                "prediction_mode": "nominal",
            })
            all_rows.append(row)

    _write_csv(output / "confirmation_episodes.csv", all_rows)
    manifest["observed_episodes"] = len(all_rows)
    manifest["completed"] = bool(len(all_rows) == manifest["expected_episodes"])
    (output / "run_metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
