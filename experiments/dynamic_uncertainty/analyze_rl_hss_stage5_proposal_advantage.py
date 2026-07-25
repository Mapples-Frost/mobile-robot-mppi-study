"""Analyze the stopped Stage 5 proposal-advantage mechanism probe."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "research_artifacts"
    / "dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development"
)
ARMS = ("rl_hss_off", "rl_hss_shadow", "rl_hss_advantage_veto")
REQUIRED_RUN_FILES = ("config_resolved.yaml", "metrics.json", "trajectory.csv")


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be a mapping: %s" % path)
    return value


def _yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dir(input_dir, job):
    return Path(input_dir) / "runs" / str(job["arm"]) / (
        "seed_%d" % int(job["episode_seed"])
    )


def _complete(path):
    return all((Path(path) / name).is_file() for name in REQUIRED_RUN_FILES)


def _median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _active_invariant(run_dir):
    with (Path(run_dir) / "trajectory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    latch_indices = [
        index
        for index, row in enumerate(rows)
        if float(row["reliability_proposal_advantage_latched"]) > 0.5
    ]
    if not latch_indices:
        return {"pass": False, "reason": "active veto never latched"}
    latch = int(latch_indices[0])
    post = rows[latch + 1 :]
    checks = {
        "next_cycle_authority_zero": bool(
            post
            and float(post[0][
                "reliability_proposal_advantage_authority_applied"
            ]) == 0.0
        ),
        "post_latch_guided_zero": bool(
            post
            and max(float(row["paper_guided_unique_sequences"]) for row in post)
            == 0.0
        ),
        "post_latch_proposal_authority_zero": bool(
            post
            and max(float(row["reliability_proposal_authority"]) for row in post)
            == 0.0
        ),
        "post_latch_fallback_one": bool(
            post
            and min(
                float(row["reliability_proposal_fallback_fraction"])
                for row in post
            )
            == 1.0
        ),
        "post_latch_budget_600": bool(
            post
            and min(float(row["paper_total_rollouts"]) for row in post) == 600.0
            and max(float(row["paper_total_rollouts"]) for row in post) == 600.0
        ),
    }
    return {
        "pass": all(checks.values()),
        "latch_step_zero_based": latch,
        "latch_applied_authority": float(rows[latch][
            "reliability_proposal_advantage_authority_applied"
        ]),
        "post_latch_decisions": len(post),
        **checks,
    }


def _shadow_invariant(run_dir):
    with (Path(run_dir) / "trajectory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    applied = [
        float(row["reliability_proposal_advantage_authority_applied"])
        for row in rows
    ]
    would = [
        float(row["reliability_proposal_advantage_would_authority_next"])
        for row in rows
    ]
    latched = [
        float(row["reliability_proposal_advantage_latched"])
        for row in rows
    ]
    checks = {
        "applied_authority_always_one": min(applied) == max(applied) == 1.0,
        "counterfactual_veto_observed": min(would) == 0.0,
        "counterfactual_latch_observed": max(latched) == 1.0,
    }
    return {"pass": all(checks.values()), **checks}


def analyze(input_dir=DEFAULT_INPUT):
    input_dir = Path(input_dir).resolve()
    schedule = _json(input_dir / "schedule.json")["jobs"]
    progress = _json(input_dir / "progress.json")
    completed = []
    active_invariants = []
    shadow_invariants = []
    for job in schedule:
        run_dir = _run_dir(input_dir, job)
        if not _complete(run_dir):
            continue
        metrics = _json(run_dir / "metrics.json")
        config = _yaml(run_dir / "config_resolved.yaml")
        row = {
            **job,
            "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            "success": bool(metrics["success"]),
            "collision": bool(metrics["collision"]),
            "termination_reason": str(metrics["termination_reason"]),
            "steps": int(metrics["steps"]),
            "final_goal_distance_m": float(metrics["final_goal_distance"]),
            "planner_p95_ms": float(metrics["planner_compute_ms_p95"]),
            "paper_total_rollouts_mean": float(
                metrics.get("paper_total_rollouts_mean", 0.0)
            ),
            "proposal_advantage_latch_step": int(
                metrics.get("reliability_proposal_advantage_latch_step", -1)
            ),
        }
        completed.append(row)
        if job["arm"] == "rl_hss_advantage_veto":
            active_invariants.append({
                "episode_seed": int(job["episode_seed"]),
                **_active_invariant(run_dir),
            })
        elif job["arm"] == "rl_hss_shadow":
            shadow_invariants.append({
                "episode_seed": int(job["episode_seed"]),
                **_shadow_invariant(run_dir),
            })
        expected_arm = str(config["experiment"][
            "stage5_proposal_advantage_arm"
        ])
        if expected_arm != job["arm"]:
            raise ValueError("resolved Stage 5 arm does not match schedule")
    completed_keys = {row["experimental_key"] for row in completed}
    missing = [
        job for job in schedule if job["experimental_key"] not in completed_keys
    ]
    complete_seeds = [
        int(seed)
        for seed in sorted({row["episode_seed"] for row in completed})
        if all(
            any(
                row["episode_seed"] == seed and row["arm"] == arm
                for row in completed
            )
            for arm in ARMS
        )
    ]
    by_key = {
        (int(row["episode_seed"]), str(row["arm"])): row
        for row in completed
    }

    def paired(left, right):
        rows = []
        for seed in complete_seeds:
            a = by_key[(seed, left)]
            b = by_key[(seed, right)]
            rows.append({
                "episode_seed": seed,
                "success_delta": int(a["success"]) - int(b["success"]),
                "collision_delta": int(a["collision"]) - int(b["collision"]),
                "steps_delta": int(a["steps"]) - int(b["steps"]),
                "final_goal_distance_delta_m": (
                    a["final_goal_distance_m"] - b["final_goal_distance_m"]
                ),
                "planner_p95_delta_ms": (
                    a["planner_p95_ms"] - b["planner_p95_ms"]
                ),
            })
        return {
            "left": left,
            "right": right,
            "pair_count": len(rows),
            "pairs": rows,
            "median": {
                name: _median([row[name] for row in rows]) if rows else 0.0
                for name in (
                    "success_delta",
                    "collision_delta",
                    "steps_delta",
                    "final_goal_distance_delta_m",
                    "planner_p95_delta_ms",
                )
            },
        }

    arm_summary = {}
    for arm in ARMS:
        rows = [row for row in completed if row["arm"] == arm]
        arm_summary[arm] = {
            "completed": len(rows),
            "successes": sum(row["success"] for row in rows),
            "collisions": sum(row["collision"] for row in rows),
            "median_steps": _median([row["steps"] for row in rows]),
            "median_final_goal_distance_m": _median([
                row["final_goal_distance_m"] for row in rows
            ]),
            "median_planner_p95_ms": _median([
                row["planner_p95_ms"] for row in rows
            ]),
        }
    collision_rows = [row for row in completed if row["collision"]]
    stop_matches = bool(
        len(collision_rows) == 1
        and progress.get("status") == "stopped_by_preregistered_rule"
        and progress.get("early_stop", {}).get("experimental_key")
        == collision_rows[0]["experimental_key"]
        and len(completed) == int(collision_rows[0]["run_order"]) + 1
    )
    integrity = {
        "completed_jobs": len(completed),
        "expected_jobs": len(schedule),
        "remaining_jobs": len(missing),
        "first_collision_stop_matches": stop_matches,
        "missing_jobs": missing,
        "sealed_seeds_opened": bool(progress.get("sealed_seeds_opened", True)),
        "active_invariants": active_invariants,
        "shadow_invariants": shadow_invariants,
    }
    active_shadow = paired("rl_hss_advantage_veto", "rl_hss_shadow")
    active_off = paired("rl_hss_advantage_veto", "rl_hss_off")
    integrity_pass = bool(
        stop_matches
        and not integrity["sealed_seeds_opened"]
        and all(row["pass"] for row in active_invariants)
        and all(row["pass"] for row in shadow_invariants)
    )
    mechanism_pass = bool(
        len(complete_seeds) == 2
        and all(
            by_key[(seed, "rl_hss_advantage_veto")]["success"]
            == by_key[(seed, "rl_hss_off")]["success"]
            for seed in complete_seeds
        )
        and all(
            by_key[(seed, "rl_hss_advantage_veto")][
                "final_goal_distance_m"
            ]
            < by_key[(seed, "rl_hss_shadow")]["final_goal_distance_m"]
            for seed in complete_seeds
        )
        and all(
            by_key[(seed, "rl_hss_advantage_veto")]["planner_p95_ms"]
            < 100.0
            for seed in complete_seeds
        )
    )
    gate = {
        "result": "fail",
        "integrity_pass": integrity_pass,
        "zero_collision_gate": False,
        "full_matrix_complete": False,
        "bounded_mechanism_signal_supported": mechanism_pass,
        "reason": (
            "Preregistered first-collision stop in shadow seed 730100064; "
            "two complete blocks directionally support active veto."
        ),
        "inference_limit": (
            "Two complete paired blocks; descriptive mechanism evidence only."
        ),
    }
    return {
        "schema_version": 1,
        "analysis": "stage5_proposal_advantage_blocked_descriptive_analysis",
        "input_dir": str(input_dir.relative_to(ROOT)).replace("\\", "/"),
        "completed": completed,
        "arm_summary": arm_summary,
        "complete_block_seeds": complete_seeds,
        "active_minus_shadow": active_shadow,
        "active_minus_rl_off": active_off,
        "integrity": integrity,
        "gate": gate,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(args.input_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write:
        input_dir = Path(args.input_dir).resolve()
        (input_dir / "paired_analysis.json").write_text(
            json.dumps({
                "active_minus_shadow": result["active_minus_shadow"],
                "active_minus_rl_off": result["active_minus_rl_off"],
                "complete_block_seeds": result["complete_block_seeds"],
                "inference_limit": result["gate"]["inference_limit"],
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (input_dir / "integrity_audit.json").write_text(
            json.dumps(result["integrity"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (input_dir / "gate.json").write_text(
            json.dumps(result["gate"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (input_dir / "analysis.json").write_text(payload, encoding="utf-8")
        external_paths = (
            ROOT
            / "configs/research/"
            "dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development.yaml",
            ROOT
            / "docs/experiments/dynamic_uncertainty/"
            "RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_PREREGISTRATION.md",
            ROOT
            / "docs/experiments/dynamic_uncertainty/"
            "RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_EDA.md",
            ROOT
            / "docs/experiments/dynamic_uncertainty/"
            "RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_RESULT.md",
            Path(__file__).resolve(),
            ROOT / "stage5_proposal_advantage.stdout.log",
            ROOT / "stage5_proposal_advantage.stderr.log",
            ROOT / "stage5_targeted_pytest.stdout.log",
            ROOT / "stage5_targeted_pytest.stderr.log",
            ROOT / "stage5_full_pytest.stdout.log",
            ROOT / "stage5_full_pytest.stderr.log",
        )
        missing_external = [
            str(path) for path in external_paths if not path.is_file()
        ]
        if missing_external:
            raise FileNotFoundError(
                "Stage 5 result binding is incomplete: %s"
                % missing_external
            )
        artifact_paths = sorted(
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.name != "result_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact_root": str(input_dir.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "artifact_file_count": len(artifact_paths),
            "artifact_files": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256(path),
                }
                for path in artifact_paths
            ],
            "external_bindings": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256(path),
                }
                for path in external_paths
            ],
            "completed_jobs": result["integrity"]["completed_jobs"],
            "remaining_jobs": result["integrity"]["remaining_jobs"],
            "first_collision_stop_matches": result["integrity"][
                "first_collision_stop_matches"
            ],
            "sealed_seeds_opened": False,
        }
        (input_dir / "result_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
