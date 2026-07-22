#!/usr/bin/env python3
"""Generate leakage-safe complete-recovery chains for L267."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import _prepare_reset
from experiments.rl.train_rl_sampling_prior import _scene_configs
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.environment import DirectControlEnv
from mobile_robot_mppi.rl.scripted_direct_control import (
    ScriptedDirectControlConfig,
    ScriptedPolylineDirectControl,
)


DEFAULT_CONFIG = ROOT / "configs/rl/l267_recovery_balanced_intervention.yaml"
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l267_recovery_balanced_intervention/"
    "recovery_dataset"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(payload["l267"])
    checkpoint = ROOT / config["source_checkpoint"]
    if _sha256(checkpoint) != config["source_checkpoint_sha256"]:
        raise ValueError("L267 source checkpoint SHA256 mismatch")
    references = [config["source_checkpoint"], *config["scene_configs"]]
    lowered = "\n".join(references).lower()
    forbidden = [token for token in config["forbidden_tokens"] if token.lower() in lowered]
    if forbidden:
        raise ValueError("forbidden L267 input references: %s" % forbidden)
    return payload, config


def _schedule_for_scene(scene_index, config, smoke=False):
    collection = config["collection"]
    rng = np.random.RandomState(int(collection["plan_seed"]) + scene_index)
    severities = tuple(collection["severity_levels"])
    headings = tuple(collection["heading_error_magnitudes_rad"])
    cells = [
        (side, severity, heading)
        for side in (-1, 1)
        for severity in severities
        for heading in headings
    ]
    rng.shuffle(cells)
    if smoke:
        cells = [(1, "mild", "small")]
        splits = ("smoke",)
    else:
        train = list(cells)
        validation = []
        test = []
        for index, severity in enumerate(severities):
            side = -1 if (scene_index + index) % 2 else 1
            heading = headings[(scene_index + index) % len(headings)]
            validation.append((side, severity, heading))
            test.append((-side, severity, headings[1 - headings.index(heading)]))
        cells = train + validation + test
        splits = (
            ("train",) * len(train)
            + ("validation",) * len(validation)
            + ("test",) * len(test)
        )
    profiles = list(collection["perturbation_profiles"])
    schedule = []
    for index, (split, cell) in enumerate(zip(splits, cells)):
        side, severity, heading = cell
        heading_sign = -1 if (scene_index + index) % 2 else 1
        profile = profiles[(scene_index + index) % len(profiles)]
        schedule.append({
            "schedule_index": index,
            "split": split,
            "side": int(side),
            "severity": severity,
            "heading_class": heading,
            "heading_error_rad": float(
                heading_sign * collection["heading_error_magnitudes_rad"][heading]
            ),
            "anchor_kind": ("low_abs_curvature", "left", "right", "reversal")[index % 4],
            "perturbation_name": str(profile["name"]),
            "delta_v": float(profile["delta_v"]),
            "delta_omega": float(profile["delta_omega"]),
        })
    return schedule


def _geometry_lattice(reference, resolved, collection):
    """Enumerate feasible reset geometry using global nearest-path CTE.

    Local normal offsets are invalid for self-near or self-intersecting paths:
    a point displaced from one branch can be close to another branch.  L267
    therefore stratifies on the exact same global projection used by the
    environment observation and reward.
    """

    width, height = (float(value) for value in resolved["scene"]["field_size"])
    radius = float(resolved["plant"]["robot"]["collision_radius"])
    grid = int(collection["geometry_grid_points_per_axis"])
    xs = np.linspace(radius, width - radius, grid)
    ys = np.linspace(radius, height - radius, grid)
    minimum_progress = (
        float(collection["anchor_progress_min_fraction"]) * reference.total_length
    )
    maximum_progress = (
        float(collection["anchor_progress_max_fraction"]) * reference.total_length
    )
    rows = []
    for x in xs:
        for y in ys:
            projection = reference.project(np.asarray((x, y)), minimum_progress=None)
            if not minimum_progress <= projection.progress <= maximum_progress:
                continue
            side = 1 if projection.signed_cross_track_error >= 0.0 else -1
            rows.append({
                "position": np.asarray((x, y), dtype=np.float64),
                "progress": float(projection.progress),
                "tangent_heading": float(projection.tangent_heading),
                "curvature": float(projection.curvature),
                "cte_m": float(projection.cross_track_error),
                "side": int(side),
            })
    minimum_cte = float(collection["minimum_realized_cte_m"])
    margin = float(collection["maximum_cte_margin_m"])
    fractions = collection["severity_target_fractions"]
    audit = {"grid_points_per_axis": grid, "sides": {}}
    targets = {}
    for side in (-1, 1):
        values = [row["cte_m"] for row in rows if row["side"] == side]
        if not values:
            raise RuntimeError("L267 geometry lattice has no candidates for side %d" % side)
        maximum = float(max(values))
        upper = maximum - margin
        if upper <= minimum_cte:
            raise RuntimeError(
                "L267 geometry cannot realize off-path states for side %d: max CTE %.6f"
                % (side, maximum)
            )
        targets[side] = {
            severity: float(minimum_cte + float(fraction) * (upper - minimum_cte))
            for severity, fraction in fractions.items()
        }
        audit["sides"][str(side)] = {
            "maximum_realized_cte_m": maximum,
            "severity_targets_m": targets[side],
            "candidate_count": len(values),
        }
    return rows, targets, audit


def _candidate_anchors(lattice, side, target_cte, anchor_kind):
    candidates = [row for row in lattice if row["side"] == int(side)]
    if anchor_kind == "low_abs_curvature":
        anchor_score = lambda row: abs(row["curvature"])
    elif anchor_kind == "left":
        anchor_score = lambda row: -row["curvature"]
    elif anchor_kind == "right":
        anchor_score = lambda row: row["curvature"]
    else:
        # A discrete polyline has piecewise-constant curvature.  Near-zero
        # curvature adjacent to a sign-changing segment is a deterministic
        # proxy for a reversal without introducing privileged student fields.
        anchor_score = lambda row: abs(row["curvature"])
    candidates.sort(key=lambda row: (
        abs(row["cte_m"] - float(target_cte)),
        anchor_score(row),
        row["progress"],
        row["position"][0],
        row["position"][1],
    ))
    return candidates


def _inside_field(config, state):
    width, height = (float(value) for value in config["scene"]["field_size"])
    radius = float(config["plant"]["robot"]["collision_radius"])
    x, y = float(state[0]), float(state[1])
    return bool(radius <= x <= width - radius and radius <= y <= height - radius)


def _severity_matches(target_cte_m: float, cte_m: float, tolerance_m: float) -> bool:
    """Require MuJoCo reset CTE to match the preregistered geometry target."""

    return bool(abs(float(cte_m) - float(target_cte_m)) <= float(tolerance_m))


def _student_npz(path: Path, rows, group, chain_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    np.savez_compressed(
        path,
        observations=np.asarray([row["observation"] for row in rows], dtype=np.float32),
        actions=np.asarray([row["action"] for row in rows], dtype=np.float32),
        rewards=np.asarray([[row["reward"]] for row in rows], dtype=np.float32),
        constraint_costs=np.asarray([[row["cte"] ** 2] for row in rows], dtype=np.float32),
        next_observations=np.asarray([row["next_observation"] for row in rows], dtype=np.float32),
        dones=np.asarray([[row["terminated"]] for row in rows], dtype=np.float32),
        groups=np.full((count,), int(group), dtype=np.int32),
        chain_ids=np.full((count,), int(chain_id), dtype=np.int64),
        steps=np.arange(count, dtype=np.int32),
    )


def _collect_attempt(environment, route, schedule, progress, initial_state, seed, collection):
    observation, _, path_state = _prepare_reset(environment, initial_state, seed)
    initial_cte = float(path_state["cross_track_error"])
    if environment.truth.collision or initial_cte <= float(collection["start_cte_m"]):
        return None, [], {
            "accepted": False,
            "reason": "invalid_reset",
            "initial_cte_m": initial_cte,
        }
    if not _severity_matches(
        schedule["target_cte_m"], initial_cte,
        collection["realized_cte_tolerance_m"],
    ):
        return None, [], {
            "accepted": False,
            "reason": "severity_mismatch",
            "initial_cte_m": initial_cte,
        }
    teacher = ScriptedPolylineDirectControl(
        route,
        environment.action_spec,
        ScriptedDirectControlConfig(
            lookahead_distance=float(collection["teacher"]["lookahead_m"]),
            cruise_speed=float(collection["teacher"]["cruise_speed_mps"]),
            yaw_gain=float(collection["teacher"]["yaw_gain"]),
        ),
    )
    teacher.reset()
    teacher.tracker.progress = float(progress)
    rows = []
    audit = []
    stable = 0
    reentry_progress = None
    accepted = False
    failure_reason = "maximum_steps"
    for step in range(int(collection["maximum_steps"])):
        truth_before = environment.truth.pose
        action, teacher_info = teacher.action(truth_before)
        action = np.clip(
            np.asarray(action, dtype=np.float64)
            + np.asarray((schedule["delta_v"], schedule["delta_omega"])),
            -1.0,
            1.0,
        ).astype(np.float32)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        cte = float(info["cross_track_error"])
        progress_now = float(info["path_progress"])
        boundary = not _inside_field(environment.config, environment.truth.pose.as_array())
        finite = bool(np.isfinite(np.concatenate((
            np.asarray(observation).reshape(-1), action.reshape(-1),
            next_observation.reshape(-1), np.asarray((reward, cte, progress_now)),
        ))).all())
        rows.append({
            "observation": np.asarray(observation, dtype=np.float32).copy(),
            "action": action.copy(),
            "reward": float(reward),
            "next_observation": next_observation.copy(),
            "terminated": bool(terminated),
            "cte": cte,
        })
        audit.append({
            "step": step,
            "truth_x": float(environment.truth.pose.x),
            "truth_y": float(environment.truth.pose.y),
            "truth_theta": float(environment.truth.pose.theta),
            "cte_m": cte,
            "path_progress_m": progress_now,
            "teacher_route_progress_m": float(teacher_info["route_progress"]),
            "action_v": float(action[0]),
            "action_omega": float(action[1]),
            "reward": float(reward),
            "collision": bool(info["collision"]),
            "boundary_violation": boundary,
            "safety_override": bool(info["safety_override"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "finite": finite,
        })
        if not finite:
            failure_reason = "non_finite"
            break
        if info["collision"]:
            failure_reason = "collision"
            break
        if boundary:
            failure_reason = "boundary_violation"
            break
        if cte < float(collection["stable_cte_m"]):
            if reentry_progress is None:
                reentry_progress = progress_now
            stable += 1
        else:
            stable = 0
            reentry_progress = None
        if (
            stable >= int(collection["stable_steps"])
            and reentry_progress is not None
            and progress_now - reentry_progress
            >= float(collection["minimum_post_reentry_progress_m"])
        ):
            accepted = True
            failure_reason = "accepted"
            break
        if terminated or truncated:
            failure_reason = "terminated" if terminated else "truncated"
            break
        observation = next_observation
    summary = {
        "accepted": accepted,
        "reason": failure_reason,
        "initial_cte_m": initial_cte,
        "steps": len(rows),
        "minimum_cte_m": min((row["cte"] for row in rows), default=float("inf")),
        "final_cte_m": rows[-1]["cte"] if rows else initial_cte,
        "stable_steps": stable,
        "reentry_progress_m": reentry_progress,
        "final_progress_m": audit[-1]["path_progress_m"] if audit else float(path_state["progress"]),
        "safety_override_steps": sum(int(row["safety_override"]) for row in audit),
    }
    return rows, audit, summary


def generate(config_path: Path, output: Path, smoke=False):
    payload, config = _load_protocol(config_path)
    collection = config["collection"]
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    output.mkdir(parents=True, exist_ok=True)
    all_attempts = []
    accepted_records = []
    geometry_audits = {}
    global_chain_id = 0
    next_seed = int(collection["episode_seed_start"])
    for scene_index, scene_relative in enumerate(config["scene_configs"]):
        scene_path = ROOT / scene_relative
        resolved = _scene_configs(base, [scene_relative])[0]
        resolved["experiment"]["max_steps"] = int(collection["maximum_steps"]) + 1
        resolved["experiment"]["initial_state_noise"] = [0.0] * len(
            resolved["experiment"]["initial_state"]
        )
        resolved["rl"]["training"]["initial_state_curriculum"] = {"enabled": False}
        environment = DirectControlEnv(resolved, ROOT, seed=next_seed)
        try:
            route = np.asarray(resolved["task"]["points"], dtype=np.float64)
            lattice, targets, geometry_audit = _geometry_lattice(
                environment.components["reference"], resolved, collection,
            )
            geometry_audits[scene_relative] = geometry_audit
            schedule = _schedule_for_scene(scene_index, config, smoke=smoke)
            for specification in schedule:
                specification["target_cte_m"] = float(
                    targets[specification["side"]][specification["severity"]]
                )
                candidates = _candidate_anchors(
                    lattice, specification["side"], specification["target_cte_m"],
                    specification["anchor_kind"],
                )
                accepted = False
                for attempt_index, candidate in enumerate(
                    candidates[: int(collection["maximum_attempts_per_stratum"])]
                ):
                    progress = float(candidate["progress"])
                    curvature = float(candidate["curvature"])
                    seed = next_seed
                    next_seed += 1
                    initial = np.asarray((
                        candidate["position"][0], candidate["position"][1],
                        candidate["tangent_heading"] + specification["heading_error_rad"],
                        0.0, 0.0,
                    ), dtype=np.float64)
                    if not _inside_field(resolved, initial):
                        all_attempts.append({
                            **specification,
                            "scene_index": scene_index,
                            "scene": Path(scene_relative).stem,
                            "attempt_index": attempt_index,
                            "seed": seed,
                            "progress_m": progress,
                            "curvature": curvature,
                            "accepted": False,
                            "reason": "outside_field",
                            "steps": 0,
                        })
                        continue
                    rows, audit, summary = _collect_attempt(
                        environment, route, specification, progress, initial,
                        seed, collection,
                    )
                    attempt_id = "%02d_%02d_%02d" % (
                        scene_index, specification["schedule_index"], attempt_index
                    )
                    student_path = output / "raw" / "attempts" / (attempt_id + ".npz")
                    if rows:
                        _student_npz(student_path, rows, scene_index, global_chain_id)
                    audit_rows = []
                    for row in audit:
                        audit_rows.append({
                            "attempt_id": attempt_id,
                            "scene": Path(scene_relative).stem,
                            "seed": seed,
                            **row,
                        })
                    if audit_rows:
                        _write_csv(
                            output / "raw" / "audit" / (attempt_id + ".csv"),
                            audit_rows,
                        )
                    record = {
                        **specification,
                        "attempt_id": attempt_id,
                        "scene_index": scene_index,
                        "scene": Path(scene_relative).stem,
                        "scene_config": scene_relative,
                        "attempt_index": attempt_index,
                        "seed": seed,
                        "progress_m": progress,
                        "curvature": curvature,
                        "initial_state": initial.tolist(),
                        "student_npz": str(student_path.relative_to(output)) if rows else "",
                        **summary,
                    }
                    all_attempts.append(record)
                    if summary["accepted"]:
                        record["chain_id"] = global_chain_id
                        accepted_records.append(record.copy())
                        global_chain_id += 1
                        accepted = True
                        break
                if not accepted:
                    _write_csv(output / "attempts.csv", all_attempts)
                    _json_dump(output / "partial_manifest.json", {
                        "status": "quota_failure",
                        "failed_specification": specification,
                        "accepted_chains": accepted_records,
                        "attempts": all_attempts,
                    })
                    raise RuntimeError(
                        "L267 recovery quota failed for scene %s specification %s"
                        % (scene_relative, specification)
                    )
        finally:
            environment.close()
    expected_per_scene = 1 if smoke else int(collection["accepted_chains_per_scene"])
    counts = {
        Path(scene).stem: sum(record["scene_config"] == scene for record in accepted_records)
        for scene in config["scene_configs"]
    }
    if any(value != expected_per_scene for value in counts.values()):
        raise RuntimeError("accepted recovery-chain counts violate the frozen quota")
    split_counts = {
        split: sum(record["split"] == split for record in accepted_records)
        for split in sorted({record["split"] for record in accepted_records})
    }
    manifest = {
        "protocol": "L267",
        "status": "smoke_pass" if smoke else "complete",
        "git_sha": git_sha(ROOT),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "student_fields": [
            "observations", "actions", "rewards", "constraint_costs",
            "next_observations", "dones", "groups", "chain_ids", "steps",
        ],
        "privileged_fields_in_student_shards": False,
        "accepted_chain_count": len(accepted_records),
        "accepted_by_scene": counts,
        "accepted_by_split": split_counts,
        "failed_attempt_count": sum(not bool(row.get("accepted")) for row in all_attempts),
        "scene_geometry_sha256": {
            scene: _sha256(ROOT / scene) for scene in config["scene_configs"]
        },
        "geometry_feasibility_audit": geometry_audits,
        "chains": accepted_records,
    }
    _write_csv(output / "attempts.csv", all_attempts)
    _write_csv(output / "accepted_chains.csv", accepted_records)
    _json_dump(output / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "accepted_chain_count": len(accepted_records),
        "failed_attempt_count": manifest["failed_attempt_count"],
        "output": str(output),
    }, sort_keys=True))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.smoke and args.output_dir == DEFAULT_OUTPUT:
        output = output.parent / "recovery_dataset_smoke"
    generate(args.config.resolve(), output, smoke=args.smoke)


if __name__ == "__main__":
    main()
