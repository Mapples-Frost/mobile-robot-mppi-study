#!/usr/bin/env python3
"""Audit and summarize the three completed L218 full-budget probes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCENES = ("serpentine", "nested_u", "cylinder_spiral")
SEED = 91001


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty probe table")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries = []
    checksums = []
    for scene in SCENES:
        run_root = root / (
            "l218_dev_screen_full_budget_probe_%s_seed%d" % (scene, SEED)
        )
        progress_rows = _read_csv(run_root / "progress.csv")
        if len(progress_rows) != 1:
            raise ValueError("probe must contain exactly one episode: %s" % run_root)
        row = progress_rows[0]
        expected = {
            "benchmark_arm": "full_proposed",
            "seed": str(SEED),
            "qualification": "1",
            "physics_domain": "nominal_seen",
            "rollout_budget_per_decision": "100",
            "paper_iterations": "2",
        }
        for name, value in expected.items():
            if str(row.get(name)) != value:
                raise ValueError("%s mismatch in %s" % (name, run_root))
        matches = list((run_root / "runs" / "full_proposed").glob("*"))
        if len(matches) != 1 or not matches[0].is_dir():
            raise ValueError("probe run artifact directory is incomplete")
        artifact = matches[0]
        provenance = json.loads(
            (run_root / "provenance.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (artifact / "metrics.json").read_text(encoding="utf-8")
        )
        metadata = dict(metrics.get("metadata", {}))
        if (
            provenance.get("status") != "pipeline_qualification"
            or metadata.get("plant_backend") != "mujoco_diff_drive"
            or metadata.get("mujoco_version") != "3.2.3"
            or metadata.get("prediction_mode") != "icode_residual"
        ):
            raise ValueError("probe is not an ICODE/MuJoCo run: %s" % artifact)
        summaries.append({
            "scene": row["scene"],
            "seed": int(row["seed"]),
            "success": row["success"],
            "termination_reason": row["termination_reason"],
            "steps": int(row["steps"]),
            "collision": row["collision"],
            "path_completion_ratio": row["path_completion_ratio"],
            "final_goal_distance": row["final_goal_distance"],
            "path_cross_track_rmse": row["path_cross_track_rmse_recomputed"],
            "minimum_clearance": row["minimum_clearance"],
            "safety_interventions": int(row["safety_interventions"]),
            "planner_compute_ms_mean": row["planner_compute_ms_mean"],
            "proposal_authority_mean": row[
                "reliability_proposal_authority_mean"
            ],
            "proposal_fallback_fraction_mean": row[
                "reliability_proposal_fallback_fraction_mean"
            ],
            "rollouts_per_decision": int(row["rollout_budget_per_decision"]),
            "iterations": int(row["paper_iterations"]),
            "plant_backend": metadata["plant_backend"],
            "mujoco_version": metadata["mujoco_version"],
            "prediction_mode": metadata["prediction_mode"],
            "benchmark_git_sha": provenance["git_sha"],
            "run_dir": str(artifact),
        })
        for filename in (
            "config_resolved.yaml", "trajectory.csv", "metrics.json", "provenance.json"
        ):
            path = artifact / filename
            if not path.is_file():
                raise ValueError("missing probe artifact: %s" % path)
            checksums.append({
                "scene": row["scene"],
                "artifact": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "path": str(path),
            })

    _write_csv(output / "full_budget_probe_summary.csv", summaries)
    _write_csv(output / "full_budget_probe_artifact_checksums.csv", checksums)
    decision = {
        "status": "development_gate_not_yet_passed",
        "episode_count": len(summaries),
        "all_mujoco": True,
        "all_collision_free": all(row["collision"] == "False" for row in summaries),
        "successes": sum(row["success"] == "True" for row in summaries),
        "interpretation": (
            "Increasing K, MPPI iterations, and max steps did not rescue the "
            "three hard maps. The residual-conditioned Actor remained fully "
            "rejected as OOD, so Actor cross-map adaptation is required before "
            "a sealed benchmark."
        ),
        "formal_claim_allowed": False,
        "reasons": [
            "qualification seed 91001",
            "three-map development probe",
            "Actor proposal authority remained zero",
        ],
    }
    (output / "full_budget_probe_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
