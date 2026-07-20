#!/usr/bin/env python3
"""Audit and analyse the paired L218 expanded-scene development screen.

The independent experimental unit is the simulation seed.  The six scenes
are repeated strata within a seed and therefore remain clustered during the
bootstrap.  This script intentionally accepts qualification data only; it
must never relabel a development screen as sealed confirmation.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_effects,
)


ARMS = ("simple_combination", "full_proposed")
SCENES = (
    "l218_serpentine",
    "l218_giant_u",
    "l218_opposed_u",
    "l218_nested_u",
    "l218_cylinder_forest",
    "l218_cylinder_spiral",
)
METRICS = {
    "success": True,
    "collision": False,
    "path_completion_ratio": True,
    "path_cross_track_rmse_recomputed": False,
    "final_goal_distance": False,
    "steps": False,
    "minimum_clearance": True,
    "control_jerk": False,
    "safety_interventions": False,
    "planner_compute_ms_mean": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty L218 table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(value):
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError("L218 analysis needs at least two unique seeds")
    return result


def _git_output(*args):
    return subprocess.check_output(
        ("git",) + args, cwd=str(ROOT), text=False
    )


def _worktree_patch_sha256():
    digest = hashlib.sha256()
    digest.update(_git_output("diff", "--binary", "HEAD"))
    return digest.hexdigest()


def _as_float(row, name):
    value = row[name]
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("non-finite %s" % name)
    return result


def _descriptives(rows):
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark_arm"], "all")].append(row)
        groups[(row["benchmark_arm"], row["scene"])].append(row)
    for (arm, scene), group in sorted(groups.items()):
        item = {
            "benchmark_arm": arm,
            "scene": scene,
            "episodes": len(group),
            "independent_seeds": len({row["seed"] for row in group}),
        }
        for metric in METRICS:
            values = np.asarray(
                [_as_float(row, metric) for row in group], dtype=np.float64
            )
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_sd_episode_descriptive"] = (
                float(np.std(values, ddof=1)) if len(values) >= 2 else ""
            )
        for metric in (
            "reliability_hss_enabled_fraction",
            "residual_policy_context_enabled_fraction",
            "reliability_proposal_authority_mean",
            "reliability_proposal_fallback_fraction_mean",
        ):
            item[metric + "_mean"] = float(np.mean([
                _as_float(row, metric) for row in group
            ]))
        result.append(item)
    return result


def _validate(run_dirs, seeds):
    rows = []
    artifact_rows = []
    git_shas = set()
    for run_dir in run_dirs:
        progress = run_dir / "progress.csv"
        current = _read_csv(progress)
        if len(current) != len(ARMS):
            raise ValueError("%s does not contain two paired rows" % run_dir)
        rows.extend(current)
        for arm in ARMS:
            arm_dirs = list((run_dir / "runs" / arm).glob("*"))
            if len(arm_dirs) != 1:
                raise ValueError("%s has invalid %s run count" % (run_dir, arm))
            episode = arm_dirs[0]
            metrics_path = episode / "metrics.json"
            provenance_path = episode / "provenance.json"
            trajectory_path = episode / "trajectory.csv"
            config_path = episode / "config_resolved.yaml"
            for path in (
                metrics_path, provenance_path, trajectory_path, config_path
            ):
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError("missing or empty artifact: %s" % path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metadata = metrics.get("metadata", {})
            if metadata.get("plant_backend") != "mujoco_diff_drive":
                raise ValueError("non-MuJoCo L218 episode: %s" % episode)
            if str(metadata.get("mujoco_version")) != "3.2.3":
                raise ValueError("unexpected MuJoCo version: %s" % episode)
            if metadata.get("prediction_mode") != "icode_residual":
                raise ValueError("L218 paired arm bypassed ICODE: %s" % episode)
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            git_shas.add(str(provenance["git_sha"]))
            artifact_rows.append({
                "run_dir": str(episode),
                "arm": arm,
                "metrics_sha256": _sha256(metrics_path),
                "trajectory_sha256": _sha256(trajectory_path),
                "config_sha256": _sha256(config_path),
                "provenance_sha256": _sha256(provenance_path),
                "plant_backend": metadata["plant_backend"],
                "mujoco_version": metadata["mujoco_version"],
                "prediction_mode": metadata["prediction_mode"],
            })
    expected = {
        (scene, "nominal_seen", str(seed), arm)
        for scene in SCENES for seed in seeds for arm in ARMS
    }
    observed = set()
    for row in rows:
        key = (
            row["scene"], row["physics_domain"], row["seed"],
            row["benchmark_arm"],
        )
        if key in observed:
            raise ValueError("duplicated L218 paired cell: %s" % (key,))
        observed.add(key)
        if int(row["qualification"]) != 1:
            raise ValueError("L218 development analysis rejects formal rows")
        if row["metric_profile"] != "path_tracking":
            raise ValueError("L218 metric profile changed")
    if observed != expected:
        raise ValueError(
            "L218 cells differ: missing=%s unexpected=%s"
            % (sorted(expected - observed), sorted(observed - expected))
        )
    if len(git_shas) != 1:
        raise ValueError("L218 run-level Git SHAs differ: %s" % sorted(git_shas))
    return rows, artifact_rows, next(iter(git_shas))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-glob",
        default=(
            "results/research_platform/rl/"
            "l218_dev_screen_coupled_gate_v2_s*_seed*"
        ),
    )
    parser.add_argument("--seeds", default="91001,91002,91003")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    run_dirs = sorted(
        path for path in ROOT.glob(args.run_glob) if path.is_dir()
    )
    if len(run_dirs) != len(SCENES) * len(seeds):
        raise ValueError(
            "expected %d L218 result directories, found %d"
            % (len(SCENES) * len(seeds), len(run_dirs))
        )
    rows, artifact_rows, run_git_sha = _validate(run_dirs, seeds)
    control = [
        dict(row, method="l218_proposal_gate") for row in rows
        if row["benchmark_arm"] == "simple_combination"
    ]
    proposed = [
        dict(row, method="l218_proposal_gate") for row in rows
        if row["benchmark_arm"] == "full_proposed"
    ]
    paired = paired_checkpoint_effects(
        control,
        proposed,
        method="l218_proposal_gate",
        metrics=METRICS,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    full_rows = [
        row for row in rows if row["benchmark_arm"] == "full_proposed"
    ]
    simple_rows = [
        row for row in rows if row["benchmark_arm"] == "simple_combination"
    ]
    per_seed_success = {
        str(seed): {
            arm: int(sum(
                _as_float(row, "success")
                for row in rows
                if int(row["seed"]) == seed
                and row["benchmark_arm"] == arm
            ))
            for arm in ARMS
        }
        for seed in seeds
    }
    success_effect = paired["metrics"]["success"]["favorable_effect"]
    completion_effect = paired["metrics"][
        "path_completion_ratio"
    ]["favorable_effect"]
    collision_effect = paired["metrics"]["collision"]["favorable_effect"]
    mechanism = {
        "full_hss_enabled_min": float(min(
            _as_float(row, "reliability_hss_enabled_fraction")
            for row in full_rows
        )),
        "full_residual_context_enabled_min": float(min(
            _as_float(row, "residual_policy_context_enabled_fraction")
            for row in full_rows
        )),
        "full_proposal_authority_max": float(max(
            _as_float(row, "reliability_proposal_authority_mean")
            for row in full_rows
        )),
        "simple_proposal_authority_min": float(min(
            _as_float(row, "reliability_proposal_authority_mean")
            for row in simple_rows
        )),
    }
    gate_pass = bool(
        success_effect > 0.0
        and completion_effect > 0.10
        and collision_effect >= 0.0
        and all(
            values["full_proposed"] >= values["simple_combination"]
            for values in per_seed_success.values()
        )
        and mechanism["full_hss_enabled_min"] == 1.0
        and mechanism["full_residual_context_enabled_min"] == 1.0
        and mechanism["full_proposal_authority_max"] <= 0.05
        and mechanism["simple_proposal_authority_min"] >= 0.95
    )
    decision = {
        "status": (
            "development_gate_passed" if gate_pass
            else "development_gate_failed"
        ),
        "interpretation": (
            "Extreme-OOD safe fallback qualification; this does not show "
            "that the RL Actor improves over ICODE-MPPI."
        ),
        "criteria": {
            "positive_success_effect": success_effect > 0.0,
            "completion_gain_gt_0_10": completion_effect > 0.10,
            "no_collision_worsening": collision_effect >= 0.0,
            "no_seed_success_regression": all(
                values["full_proposed"] >= values["simple_combination"]
                for values in per_seed_success.values()
            ),
            "mechanism_activation": mechanism,
        },
        "per_seed_successes_out_of_six": per_seed_success,
    }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "combined_progress.csv", rows)
    _write_csv(output / "descriptive_summary.csv", _descriptives(rows))
    _write_csv(output / "artifact_checksums.csv", artifact_rows)
    (output / "paired_seed_cluster_effects.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "development_gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "status": "complete_development_qualification",
        "episodes": len(rows),
        "paired_cells": len(rows) // 2,
        "independent_unit": "seed",
        "independent_seeds": list(seeds),
        "repeated_strata": ["scene"],
        "scenes": list(SCENES),
        "arms": list(ARMS),
        "physics_domains": ["nominal_seen"],
        "qualification_rows": len(rows),
        "plant_backend": "mujoco_diff_drive",
        "mujoco_version": "3.2.3",
        "run_git_sha": run_git_sha,
        "analysis_git_head": _git_output("rev-parse", "HEAD").decode().strip(),
        "analysis_worktree_dirty": bool(
            _git_output("status", "--porcelain").strip()
        ),
        "analysis_worktree_tracked_patch_sha256": _worktree_patch_sha256(),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "formal_claim_allowed": False,
        "reason_formal_claim_forbidden": (
            "qualification seeds and an uncommitted development implementation"
        ),
        "gate_status": decision["status"],
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**audit, "decision": decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
