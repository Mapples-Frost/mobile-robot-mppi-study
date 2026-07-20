#!/usr/bin/env python3
"""Audit and analyse the L219 six-map Actor development gate.

Seeds are the independent units; the six maps are repeated strata.  L219 is
development qualification, never sealed confirmation.  The old L218 Full arm
is loaded only as a sourced historical comparator, not as a randomized arm in
the new three-arm blocks.
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

from mobile_robot_mppi.evaluation.paired_checkpoint import paired_checkpoint_effects


ARMS = ("icode_mppi", "simple_combination", "full_proposed")
SCENES = (
    "l218_serpentine",
    "l218_giant_u",
    "l218_opposed_u",
    "l218_nested_u",
    "l218_cylinder_forest",
    "l218_cylinder_spiral",
)
PREVIOUSLY_FAILED = {
    "l218_serpentine", "l218_nested_u", "l218_cylinder_spiral"
}
PREVIOUSLY_SUCCESSFUL = {
    "l218_giant_u", "l218_opposed_u", "l218_cylinder_forest"
}
EXPECTED_GIT_SHA = "28378a60548a35768b4222723914921ab941c2a2"
EXPECTED_ACTOR_SHA = (
    "bf26a67ebac313930d63760db931e5d50704cdd9181923afd7f66d16159356b4"
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
        raise ValueError("refusing to write an empty L219 table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row, name):
    value = row[name]
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("non-finite %s" % name)
    return result


def _git_output(*args):
    return subprocess.check_output(("git",) + args, cwd=str(ROOT), text=True).strip()


def _validate_new(run_dirs, seeds, expected_git_sha=EXPECTED_GIT_SHA,
                  expected_actor_sha=EXPECTED_ACTOR_SHA):
    rows, artifacts, observed = [], [], set()
    for run_dir in run_dirs:
        provenance = json.loads((run_dir / "provenance.json").read_text())
        if provenance["git_sha"] != expected_git_sha:
            raise ValueError("unexpected top-level Git SHA: %s" % run_dir)
        if provenance["coupled_actor_checkpoint"]["sha256"] != expected_actor_sha:
            raise ValueError("unexpected L219 Actor: %s" % run_dir)
        schedule = json.loads((run_dir / "schedule.json").read_text())
        if len(schedule) != len(ARMS):
            raise ValueError("invalid randomized block: %s" % run_dir)
        current = _read_csv(run_dir / "progress.csv")
        if len(current) != len(ARMS):
            raise ValueError("expected three arms: %s" % run_dir)
        for row in current:
            key = (row["scene"], row["physics_domain"], row["seed"], row["benchmark_arm"])
            if key in observed:
                raise ValueError("duplicate L219 cell: %s" % (key,))
            observed.add(key)
            if int(row["qualification"]) != 1:
                raise ValueError("formal row found in development analysis")
            if int(float(row["rollout_budget_per_decision"])) != 30:
                raise ValueError("rollout budget changed: %s" % (key,))
            if int(float(row["paper_iterations"])) != 1:
                raise ValueError("iteration budget changed: %s" % (key,))
            episode = (
                run_dir / "runs" / row["benchmark_arm"] /
                ("%s__%s__%s__seed%s" % (
                    row["benchmark_arm"], row["scene"],
                    row["physics_domain"], row["seed"],
                ))
            )
            paths = {
                name: episode / name for name in (
                    "metrics.json", "trajectory.csv", "config_resolved.yaml",
                    "provenance.json",
                )
            }
            for path in paths.values():
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError("missing artifact: %s" % path)
            metrics = json.loads(paths["metrics.json"].read_text())
            metadata = metrics["metadata"]
            if metadata.get("plant_backend") != "mujoco_diff_drive":
                raise ValueError("non-MuJoCo episode: %s" % episode)
            if str(metadata.get("mujoco_version")) != "3.2.3":
                raise ValueError("unexpected MuJoCo version: %s" % episode)
            if metadata.get("prediction_mode") != "icode_residual":
                raise ValueError("L219 arm bypassed ICODE: %s" % episode)
            run_provenance = json.loads(paths["provenance.json"].read_text())
            if run_provenance["git_sha"] != expected_git_sha:
                raise ValueError("run-level Git SHA changed: %s" % episode)
            if row["benchmark_arm"] != "icode_mppi":
                if not metadata.get("rl_checkpoint", "").endswith("step_000010000.pt"):
                    raise ValueError("wrong Actor checkpoint: %s" % episode)
            artifacts.append({
                "run_dir": str(episode),
                "arm": row["benchmark_arm"],
                "scene": row["scene"],
                "seed": row["seed"],
                **{name.replace(".", "_") + "_sha256": _sha256(path)
                   for name, path in paths.items()},
            })
        rows.extend(current)
    expected = {
        (scene, "nominal_seen", str(seed), arm)
        for scene in SCENES for seed in seeds for arm in ARMS
    }
    if observed != expected:
        raise ValueError("L219 cells differ: missing=%s unexpected=%s" % (
            sorted(expected - observed), sorted(observed - expected)
        ))
    return rows, artifacts


def _load_old_full(run_dirs, seeds):
    rows = []
    observed = set()
    for run_dir in run_dirs:
        candidates = [row for row in _read_csv(run_dir / "progress.csv")
                      if row["benchmark_arm"] == "full_proposed"]
        if len(candidates) != 1:
            raise ValueError("old L218 directory lacks one Full row: %s" % run_dir)
        row = dict(candidates[0])
        key = (row["scene"], row["physics_domain"], row["seed"])
        if key in observed:
            raise ValueError("duplicate old Full cell: %s" % (key,))
        observed.add(key)
        row["benchmark_arm"] = "old_full_proposed"
        rows.append(row)
    expected = {(scene, "nominal_seen", str(seed))
                for scene in SCENES for seed in seeds}
    if observed != expected:
        raise ValueError("old Full cells do not match L219 repeated strata")
    return rows


def _descriptives(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark_arm"], "all")].append(row)
        groups[(row["benchmark_arm"], row["scene"])].append(row)
    output = []
    for (arm, scene), group in sorted(groups.items()):
        item = {
            "benchmark_arm": arm,
            "scene": scene,
            "episodes": len(group),
            "independent_seeds": len({row["seed"] for row in group}),
        }
        for metric in METRICS:
            values = [_as_float(row, metric) for row in group]
            item[metric + "_mean"] = float(np.mean(values))
        for metric in (
            "residual_policy_context_enabled_fraction",
            "reliability_hss_enabled_fraction",
            "reliability_actor_support_confidence_mean",
            "reliability_proposal_authority_mean",
            "reliability_proposal_fallback_fraction_mean",
        ):
            values = [_as_float(row, metric) for row in group]
            item[metric + "_mean"] = float(np.mean(values))
        output.append(item)
    return output


def _paired(rows, control_arm, proposed_arm, label, samples, seed):
    control = [dict(row, method=label) for row in rows
               if row["benchmark_arm"] == control_arm]
    proposed = [dict(row, method=label) for row in rows
                if row["benchmark_arm"] == proposed_arm]
    return paired_checkpoint_effects(
        control, proposed, method=label, metrics=METRICS,
        bootstrap_samples=samples, seed=seed,
    )


def _scene_summary(rows):
    output = []
    for scene in SCENES:
        for arm in ("icode_mppi", "simple_combination", "full_proposed", "old_full_proposed"):
            group = [row for row in rows if row["scene"] == scene and row["benchmark_arm"] == arm]
            output.append({
                "scene": scene,
                "arm": arm,
                "successes_out_of_3": int(sum(_as_float(row, "success") for row in group)),
                "collisions_out_of_3": int(sum(_as_float(row, "collision") for row in group)),
                "completion_mean": float(np.mean([_as_float(row, "path_completion_ratio") for row in group])),
                "proposal_authority_mean": float(np.mean([_as_float(row, "reliability_proposal_authority_mean") for row in group])),
                "actor_support_mean": float(np.mean([_as_float(row, "reliability_actor_support_confidence_mean") for row in group])),
            })
    return output


def _gate(rows, comparisons, scene_rows):
    lookup = {(row["scene"], row["arm"]): row for row in scene_rows}
    new_full = [row for row in rows if row["benchmark_arm"] == "full_proposed"]
    collision_ok = all(
        lookup[(scene, "full_proposed")]["collisions_out_of_3"]
        <= lookup[(scene, baseline)]["collisions_out_of_3"]
        for scene in SCENES for baseline in ("icode_mppi", "old_full_proposed")
    )
    mechanism_active = bool(
        max(_as_float(row, "reliability_proposal_authority_mean") for row in new_full) > 0.05
        and max(_as_float(row, "reliability_actor_support_confidence_mean") for row in new_full) > 0.05
    )
    hard_improvements = []
    for scene in sorted(PREVIOUSLY_FAILED):
        current = lookup[(scene, "full_proposed")]
        old = lookup[(scene, "old_full_proposed")]
        if (current["successes_out_of_3"] > old["successes_out_of_3"] or
                current["completion_mean"] - old["completion_mean"] > 0.10):
            hard_improvements.append(scene)
    successful_old = sum(lookup[(scene, "old_full_proposed")]["successes_out_of_3"]
                         for scene in PREVIOUSLY_SUCCESSFUL)
    successful_new = sum(lookup[(scene, "full_proposed")]["successes_out_of_3"]
                         for scene in PREVIOUSLY_SUCCESSFUL)
    regressed_success_maps = sum(
        lookup[(scene, "full_proposed")]["successes_out_of_3"]
        < lookup[(scene, "old_full_proposed")]["successes_out_of_3"]
        for scene in PREVIOUSLY_SUCCESSFUL
    )
    no_systematic_old_map_regression = (
        successful_new >= successful_old and regressed_success_maps <= 1
    )
    vs_simple = comparisons["full_vs_simple"]["metrics"]
    vs_icode = comparisons["full_vs_icode"]["metrics"]
    full_beats_simple = bool(
        vs_simple["collision"]["favorable_effect"] >= 0.0
        and vs_simple["success"]["favorable_effect"] >= 0.0
        and (vs_simple["success"]["favorable_effect"] > 0.0
             or vs_simple["path_completion_ratio"]["favorable_effect"] > 0.0)
    )
    honest_vs_icode = bool(
        vs_icode["collision"]["favorable_effect"] >= 0.0
        and vs_icode["success"]["favorable_effect"] >= 0.0
        and vs_icode["path_completion_ratio"]["favorable_effect"] >= -0.02
    )
    criteria = {
        "no_scene_collision_worsening_vs_icode_and_old_full": collision_ok,
        "actor_support_and_proposal_authority_nonzero": mechanism_active,
        "previously_failed_map_improved": bool(hard_improvements),
        "improved_previously_failed_maps": hard_improvements,
        "no_systematic_regression_on_previously_successful_maps": no_systematic_old_map_regression,
        "previously_successful_map_successes_old": successful_old,
        "previously_successful_map_successes_new": successful_new,
        "regressed_previously_successful_map_count": regressed_success_maps,
        "full_better_than_simple_under_equal_budget": full_beats_simple,
        "full_noninferior_to_icode_on_safety_success_completion": honest_vs_icode,
    }
    booleans = [value for value in criteria.values() if isinstance(value, bool)]
    return {
        "status": "development_gate_passed" if all(booleans) else "development_gate_failed",
        "criteria": criteria,
        "formal_claim_allowed": False,
        "reason": "Three reused development seed clusters; sealed confirmation is required.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument(
        "--new-run-glob",
        default=(
            "results/research_platform/rl/"
            "l218_dev_screen_l219_actor_s9100*_*_seed9100*"
        ),
        help="Repository-relative glob for the 18 new development run directories.",
    )
    parser.add_argument("--expected-git-sha", default=EXPECTED_GIT_SHA)
    parser.add_argument("--expected-actor-sha", default=EXPECTED_ACTOR_SHA)
    args = parser.parse_args(argv)
    seeds = (91001, 91002, 91003)
    new_dirs = sorted(path for path in ROOT.glob(args.new_run_glob) if path.is_dir())
    old_dirs = sorted(path for path in ROOT.glob(
        "results/research_platform/rl/l218_dev_screen_coupled_gate_v2_s*_seed*"
    ) if path.is_dir())
    if len(new_dirs) != 18 or len(old_dirs) != 18:
        raise ValueError("expected 18 new and 18 old map directories")
    new_rows, artifacts = _validate_new(
        new_dirs,
        seeds,
        expected_git_sha=args.expected_git_sha,
        expected_actor_sha=args.expected_actor_sha,
    )
    old_rows = _load_old_full(old_dirs, seeds)
    all_rows = new_rows + old_rows
    comparisons = {
        "full_vs_icode": _paired(new_rows, "icode_mppi", "full_proposed", "full_vs_icode", args.bootstrap_samples, args.bootstrap_seed),
        "full_vs_simple": _paired(new_rows, "simple_combination", "full_proposed", "full_vs_simple", args.bootstrap_samples, args.bootstrap_seed + 1),
        "simple_vs_icode": _paired(new_rows, "icode_mppi", "simple_combination", "simple_vs_icode", args.bootstrap_samples, args.bootstrap_seed + 2),
        "new_full_vs_old_full_auxiliary": _paired(all_rows, "old_full_proposed", "full_proposed", "new_full_vs_old_full_auxiliary", args.bootstrap_samples, args.bootstrap_seed + 3),
    }
    scene_rows = _scene_summary(all_rows)
    decision = _gate(all_rows, comparisons, scene_rows)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "combined_progress.csv", all_rows)
    _write_csv(output / "descriptive_summary.csv", _descriptives(all_rows))
    _write_csv(output / "scene_summary.csv", scene_rows)
    _write_csv(output / "artifact_checksums.csv", artifacts)
    (output / "paired_seed_cluster_effects.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n"
    )
    (output / "development_gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "status": "complete_development_qualification",
        "new_episodes": len(new_rows),
        "old_full_auxiliary_episodes": len(old_rows),
        "independent_unit": "seed",
        "independent_seeds": list(seeds),
        "repeated_strata": ["scene"],
        "scenes": list(SCENES),
        "new_arms": list(ARMS),
        "qualification": True,
        "plant_backend": "mujoco_diff_drive",
        "mujoco_version": "3.2.3",
        "run_git_sha": args.expected_git_sha,
        "actor_checkpoint_sha256": args.expected_actor_sha,
        "analysis_git_head": _git_output("rev-parse", "HEAD"),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "gate_status": decision["status"],
        "formal_claim_allowed": False,
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"audit": audit, "decision": decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
