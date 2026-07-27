"""Generate causal observations and privileged multimodal maneuver labels.

The offline oracle may use exact MuJoCo future truth to select labels.  The
student arrays contain only the frozen paper Actor's causal observation vector.
All ICODE, Collision Risk, and Safety checks remain identical to Gate v4.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.audit_complex_oracle_feasibility_timing_gate import (
    _action_bounds,
    _audit_risk_and_safety,
    _behavior_family,
    _cem_refine,
    _expand_segments,
    _offset_indices,
    _oracle_score,
    _physical_mask,
    _structured_parameters,
    _true_branch_metrics,
)
from experiments.dynamic_uncertainty.audit_complex_privileged_teacher_gate import (
    _anchor_index,
    _read_rows,
    _replay_to_anchor,
)
from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.rl.paper_policy import PaperDirectControlPolicy
from mobile_robot_mppi.runtime.factories import make_components


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_maneuver_actor_dataset_v1.yaml"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("protocol") != "complex_supervised_maneuver_actor_dataset_v1":
        raise ValueError("supervised maneuver dataset protocol mismatch")
    frozen = value["frozen_contract"]
    forbidden_changes = (
        "maps_changed",
        "obstacle_process_changed",
        "robot_dynamics_changed",
        "action_bounds_changed",
        "collision_risk_changed",
        "icode_changed",
        "mppi_changed",
        "safety_changed",
        "teacher_search_changed",
    )
    if any(bool(frozen[name]) for name in forbidden_changes):
        raise ValueError("dataset generation changed a frozen contract")
    if bool(frozen["deployment_future_truth_allowed"]):
        raise ValueError("deployment future truth must remain forbidden")
    if bool(frozen["formal_server_launch_authorized"]):
        raise ValueError("dataset generation cannot authorize formal compute")
    student = value["student_input"]
    privileged = (
        "include_future_truth",
        "include_future_obstacle_trajectory",
        "include_privileged_change_label",
        "include_absolute_simulator_pose",
    )
    if not bool(student["causal_only"]) or any(
        bool(student[name]) for name in privileged
    ):
        raise ValueError("student input contract contains privileged fields")
    export = value["oracle_export"]
    if int(export["segments"]) != 4:
        raise ValueError("teacher export requires four-stage controls")
    if int(export["deployment_prefix_steps"]) != 36:
        raise ValueError("deployment prefix must remain 36 steps")
    return value


def _verify_source_gate(protocol):
    gate_path = (ROOT / protocol["source_gate"]["result"]).resolve()
    if _sha256(gate_path) != protocol["source_gate"]["result_sha256"]:
        raise ValueError("Gate v4 result hash mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != protocol["source_gate"]["required_status"]:
        raise ValueError("Gate v4 did not pass")
    if bool(gate["actor_training_authorized"]) != bool(
        protocol["source_gate"]["required_actor_training_authorized"]
    ):
        raise ValueError("Gate v4 Actor authorization mismatch")
    return gate_path


def _source_items(protocol):
    sources = protocol["sources"]
    if isinstance(sources, dict):
        return [
            (str(map_name), item)
            for map_name, item in sources.items()
        ]
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset sources must be a non-empty mapping or list")
    items = []
    identities = set()
    for item in sources:
        map_name = str(item["map"])
        identity = (map_name, str(item["artifact"]))
        if identity in identities:
            raise ValueError("dataset source is duplicated: %s" % (identity,))
        identities.add(identity)
        items.append((map_name, item))
    return items


def _previous_control(rows, anchor_index):
    if int(anchor_index) <= 0:
        return np.zeros(2, dtype=np.float64)
    row = rows[int(anchor_index) - 1]
    return np.asarray(
        (float(row["applied_v"]), float(row["applied_omega"])),
        dtype=np.float64,
    )


def _causal_observation(config, components, rows, anchor_index, state, perceived):
    checkpoint = (ROOT / config["_dataset_actor_checkpoint"]).resolve()
    policy = PaperDirectControlPolicy.from_checkpoint(
        checkpoint,
        components["action_spec"],
        device="cpu",
        allow_controller_action_superset=True,
    )
    previous = _previous_control(rows, anchor_index)
    policy.set_previous(previous.reshape(1, 2))
    raw, _ = policy._encoded_batch(
        np.asarray(state, dtype=np.float64).reshape(1, -1),
        previous.reshape(1, 2),
        perceived.observation,
        components["reference"],
        components["state_spec"],
        0.0,
    )
    return raw[0].astype(np.float32), {
        "dimension": int(raw.shape[1]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "absolute_pose_enabled": bool(
            policy.encoder.config.include_absolute_pose
        ),
    }


def _export_state_job(protocol_path, map_name, item, anchor_index, offset_s):
    protocol = _load_protocol(protocol_path)
    artifact = (ROOT / item["artifact"]).resolve()
    config = load_yaml(artifact / "config_resolved.yaml")
    config["_dataset_actor_checkpoint"] = protocol["student_input"][
        "actor_checkpoint"
    ]
    rows = _read_rows(artifact / "trajectory.csv")
    components = make_components(config, ROOT)
    plant = components["plant"]
    try:
        _, perceived, state, target = _replay_to_anchor(
            config, components, rows, anchor_index
        )
        observation, observation_contract = _causal_observation(
            config, components, rows, anchor_index, state, perceived
        )
        snapshot = plant.snapshot()
        obstacles = tuple(config["scene"]["obstacles"])
        static = tuple(
            value for value in obstacles
            if not isinstance(value.get("motion"), dict)
        )
        dynamic = tuple(
            value for value in obstacles
            if isinstance(value.get("motion"), dict)
        )
        dynamic_radii = tuple(float(value["radius"]) for value in dynamic)
        target_xy = np.asarray((target.pose.x, target.pose.y), dtype=np.float64)
        export = protocol["oracle_export"]
        criteria = protocol["teacher_criteria"]
        dt = float(config["experiment"]["control_dt"])
        horizon_steps = int(round(float(export["horizon_s"]) / dt))
        tail_steps = int(round(float(export["stopping_tail_s"]) / dt))
        search_seed = (
            int(export["search_seed"])
            + 100000 * int(config["experiment"]["seed"])
            + 1000 * int(round(float(offset_s) * 10.0))
            + horizon_steps
        )
        labels, parameters = _structured_parameters(
            components["action_spec"],
            int(export["coarse_candidate_count"]),
            search_seed,
        )

        def evaluate(values):
            return _true_branch_metrics(
                plant,
                snapshot,
                values,
                dt,
                static,
                dynamic_radii,
                float(components["controller"].config.robot_radius),
                target_xy,
                tail_steps,
            )

        candidates = _expand_segments(parameters, horizon_steps)
        metrics = evaluate(candidates)
        parameters, labels, metrics = _cem_refine(
            plant,
            snapshot,
            parameters,
            labels,
            metrics,
            horizon_steps,
            components["action_spec"],
            export,
            criteria,
            evaluate,
            search_seed + 17,
        )
        candidates = _expand_segments(parameters, horizon_steps)
        physical = _physical_mask(metrics, criteria)
        online = _audit_risk_and_safety(
            candidates,
            labels,
            physical,
            state,
            perceived,
            components,
            static,
            criteria,
        )
        scores = _oracle_score(metrics, criteria)
        accepted = online["safety_nonstop_indices"]
        per_family = int(export["teachers_per_behavior_family_per_state"])
        selected = []
        for family in protocol["label"]["multimodal_behavior_families"]:
            matching = [
                int(index) for index in accepted
                if _behavior_family(labels[int(index)]) == family
            ]
            matching.sort(key=lambda index: float(scores[index]), reverse=True)
            selected.extend(matching[:per_family])
        lower, upper = _action_bounds(components["action_spec"])
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        prefix = int(export["deployment_prefix_steps"])
        rows_out = []
        for index in selected:
            controls = candidates[index, :prefix].astype(np.float32)
            normalized = np.clip(
                (controls - center[None, :]) / half[None, :], -1.0, 1.0
            ).astype(np.float32)
            rows_out.append({
                "observation": observation,
                "controls": controls,
                "normalized_controls": normalized,
                "family": _behavior_family(labels[index]),
                "teacher_label": str(labels[index]),
                "score": float(scores[index]),
                "progress_m": float(
                    metrics["local_target_progress_m"][index]
                ),
                "minimum_static_clearance_m": float(
                    metrics["minimum_static_clearance_m"][index]
                ),
                "minimum_dynamic_clearance_m": float(
                    metrics["minimum_dynamic_clearance_m"][index]
                ),
                "stopping_tail_dynamic_clearance_m": float(
                    metrics[
                        "stopping_tail_minimum_dynamic_clearance_m"
                    ][index]
                ),
            })
        return {
            "map": str(map_name),
            "split": str(item["split"]),
            "seed": int(config["experiment"]["seed"]),
            "scenario_id": "%s_seed%d" % (
                map_name, int(config["experiment"]["seed"])
            ),
            "anchor_index": int(anchor_index),
            "anchor_time_s": float(rows[int(anchor_index)]["time"]),
            "offset_s": float(offset_s),
            "accepted_count": int(len(accepted)),
            "selected_count": int(len(selected)),
            "accepted_behavior_families": sorted({
                _behavior_family(labels[int(index)]) for index in accepted
            }),
            "observation_contract": observation_contract,
            "rows": rows_out,
        }
    finally:
        plant.close()


def _dataset_gate(protocol, jobs):
    gate = protocol["bootstrap_dataset_gate"]
    splits = {}
    for split in ("train", "validation"):
        subset = [job for job in jobs if job["split"] == split]
        scenarios = sorted({job["scenario_id"] for job in subset})
        states = {
            (job["scenario_id"], int(job["anchor_index"]))
            for job in subset if job["rows"]
        }
        teacher_rows = [row for job in subset for row in job["rows"]]
        families = sorted({row["family"] for row in teacher_rows})
        splits[split] = {
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
            "causal_state_count": len(states),
            "teacher_row_count": len(teacher_rows),
            "behavior_families": families,
        }
    disjoint = not (
        set(splits["train"]["scenarios"])
        & set(splits["validation"]["scenarios"])
    )
    checks = {
        "minimum_train_scenarios": (
            splits["train"]["scenario_count"]
            >= int(gate["minimum_train_scenarios"])
        ),
        "minimum_validation_scenarios": (
            splits["validation"]["scenario_count"]
            >= int(gate["minimum_validation_scenarios"])
        ),
        "minimum_train_causal_states": (
            splits["train"]["causal_state_count"]
            >= int(gate["minimum_train_causal_states"])
        ),
        "minimum_validation_causal_states": (
            splits["validation"]["causal_state_count"]
            >= int(gate["minimum_validation_causal_states"])
        ),
        "minimum_train_teacher_rows": (
            splits["train"]["teacher_row_count"]
            >= int(gate["minimum_train_teacher_rows"])
        ),
        "minimum_validation_teacher_rows": (
            splits["validation"]["teacher_row_count"]
            >= int(gate["minimum_validation_teacher_rows"])
        ),
        "minimum_train_behavior_families": (
            len(splits["train"]["behavior_families"])
            >= int(gate["minimum_train_behavior_families"])
        ),
        "minimum_validation_behavior_families": (
            len(splits["validation"]["behavior_families"])
            >= int(gate["minimum_validation_behavior_families"])
        ),
        "scenario_split_disjoint": bool(disjoint),
        "no_privileged_student_fields": True,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "engineering_training_authorized": bool(
            status == "pass"
            and gate["engineering_training_authorized_only_on_pass"]
        ),
        "splits": splits,
        "checks": checks,
        "formal_claim_authorized": False,
        "closed_loop_gate_d_authorized": False,
    }


def generate(protocol_path, output):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path)
    gate_path = _verify_source_gate(protocol)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("dataset output already contains evidence")
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "protocol_resolved.yaml"
    resolved.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    jobs = []
    offsets = protocol["oracle_export"]["offsets_s"]
    for map_name, item in _source_items(protocol):
        artifact = (ROOT / item["artifact"]).resolve()
        rows = _read_rows(artifact / "trajectory.csv")
        conflict = _anchor_index(rows, item["conflict_rule"])
        for offset_s, anchor_index, _ in _offset_indices(
            rows, conflict, offsets
        ):
            jobs.append((map_name, item, anchor_index, offset_s))

    completed = []
    progress_path = output / "progress.json"
    workers = int(protocol["oracle_export"]["parallel_workers"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _export_state_job,
                str(protocol_path),
                map_name,
                item,
                anchor_index,
                offset_s,
            ): (map_name, anchor_index, offset_s)
            for map_name, item, anchor_index, offset_s in jobs
        }
        for future in as_completed(futures):
            completed.append(future.result())
            _write_json(progress_path, {
                "status": "running",
                "completed_states": len(completed),
                "total_states": len(jobs),
                "last_completed": futures[future],
                "parallel_workers": workers,
            })

    completed.sort(key=lambda item: (
        item["split"], item["map"], item["seed"], item["anchor_index"]
    ))
    flat = [
        (job, row)
        for job in completed
        for row in job["rows"]
    ]
    if flat:
        observations = np.stack([row["observation"] for _, row in flat])
        controls = np.stack([row["controls"] for _, row in flat])
        normalized = np.stack([
            row["normalized_controls"] for _, row in flat
        ])
    else:
        observations = np.empty((0, 0), dtype=np.float32)
        controls = np.empty((0, 36, 2), dtype=np.float32)
        normalized = np.empty((0, 36, 2), dtype=np.float32)
    np.savez_compressed(
        output / "student_dataset.npz",
        observations=observations,
        teacher_controls=controls,
        normalized_teacher_controls=normalized,
        split=np.asarray([job["split"] for job, _ in flat]),
        scenario_id=np.asarray([job["scenario_id"] for job, _ in flat]),
        map_name=np.asarray([job["map"] for job, _ in flat]),
        seed=np.asarray([job["seed"] for job, _ in flat], dtype=np.int64),
        anchor_index=np.asarray([
            job["anchor_index"] for job, _ in flat
        ], dtype=np.int64),
        offset_s=np.asarray([job["offset_s"] for job, _ in flat]),
        behavior_family=np.asarray([row["family"] for _, row in flat]),
        teacher_score=np.asarray([row["score"] for _, row in flat]),
        true_progress_m=np.asarray([row["progress_m"] for _, row in flat]),
        true_minimum_static_clearance_m=np.asarray([
            row["minimum_static_clearance_m"] for _, row in flat
        ]),
        true_minimum_dynamic_clearance_m=np.asarray([
            row["minimum_dynamic_clearance_m"] for _, row in flat
        ]),
    )
    audit_rows = []
    for job in completed:
        copy = {key: value for key, value in job.items() if key != "rows"}
        copy["teachers"] = [
            {key: value for key, value in row.items()
             if key not in {"observation", "controls", "normalized_controls"}}
            for row in job["rows"]
        ]
        audit_rows.append(copy)
    _write_json(output / "privileged_teacher_audit.json", audit_rows)
    gate = _dataset_gate(protocol, completed)
    _write_json(output / "dataset_gate_result.json", gate)
    _write_json(progress_path, {
        "status": "complete",
        "completed_states": len(completed),
        "total_states": len(jobs),
        "dataset_gate_status": gate["status"],
    })
    manifest = {
        "git_sha": git_sha(ROOT),
        "protocol_config_hash": config_hash(protocol),
        "protocol_sha256": _sha256(resolved),
        "source_gate": str(gate_path),
        "source_gate_sha256": _sha256(gate_path),
        "dataset_sha256": _sha256(output / "student_dataset.npz"),
        "audit_sha256": _sha256(output / "privileged_teacher_audit.json"),
        "gate_sha256": _sha256(output / "dataset_gate_result.json"),
        "student_observation_contains_future_truth": False,
        "teacher_label_selection_uses_future_truth": True,
        "scenario_level_split": True,
        "formal_server_launch_authorized": False,
    }
    _write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    return 0 if gate["status"] == "pass" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol.resolve())
    _verify_source_gate(protocol)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_pass",
            "sources": protocol["sources"],
            "scenario_level_split": True,
            "student_input_causal_only": True,
            "deployment_prefix_steps": int(
                protocol["oracle_export"]["deployment_prefix_steps"]
            ),
            "formal_server_launch_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")
    return generate(args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
