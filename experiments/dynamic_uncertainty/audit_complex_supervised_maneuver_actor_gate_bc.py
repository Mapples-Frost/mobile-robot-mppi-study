"""Audit learned maneuver proposals through physics, ICODE, Risk, and Safety."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.audit_complex_oracle_feasibility_timing_gate import (
    _minimum_dynamic_clearance,
    _true_branch_metrics,
)
from experiments.dynamic_uncertainty.audit_complex_privileged_teacher_gate import (
    _icode_controller,
    _read_rows,
    _replay_to_anchor,
    _risk_config,
)
from experiments.dynamic_uncertainty.generate_complex_supervised_maneuver_dataset import (
    _previous_control,
)
from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.evaluation.scene_feasibility import point_clearance
from mobile_robot_mppi.obstacles.collision_risk import evaluate_collision_risk
from mobile_robot_mppi.rl.maneuver_actor import (
    ManeuverProposalActor,
    normalized_to_physical,
)
from mobile_robot_mppi.runtime.factories import make_components


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_maneuver_actor_gate_bc_v1.yaml"
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
    if value.get("protocol") != (
        "complex_supervised_maneuver_actor_gate_bc_v1"
    ):
        raise ValueError("maneuver Actor Gate B-C protocol mismatch")
    frozen = value["frozen_contract"]
    for name in (
        "maps_changed",
        "actor_checkpoint_changed",
        "icode_changed",
        "collision_risk_changed",
        "mppi_changed",
        "safety_changed",
        "future_truth_used_by_actor",
        "formal_server_launch_authorized",
    ):
        if bool(frozen[name]):
            raise ValueError("Gate B-C changed frozen contract: %s" % name)
    return value


def _load_actor(protocol):
    path = (ROOT / protocol["actor"]["checkpoint"]).resolve()
    if _sha256(path) != protocol["actor"]["sha256"]:
        raise ValueError("maneuver Actor checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["model_config"]
    model = ManeuverProposalActor(
        payload["observation_dim"],
        heads=cfg["heads"],
        horizon=cfg["horizon"],
        action_dim=cfg["action_dim"],
        hidden_sizes=cfg["hidden_sizes"],
    )
    model.load_state_dict(payload["model"])
    model.eval()
    if model.heads != int(protocol["actor"]["expected_heads"]):
        raise ValueError("maneuver Actor head count mismatch")
    if model.horizon != int(protocol["actor"]["expected_horizon"]):
        raise ValueError("maneuver Actor horizon mismatch")
    return model, payload


def _clip_sequences(action_spec, sequences, previous, dt):
    values = np.asarray(sequences, dtype=np.float64).copy()
    preceding = np.repeat(
        np.asarray(previous, dtype=np.float64).reshape(1, -1),
        values.shape[0],
        axis=0,
    )
    for step in range(values.shape[1]):
        values[:, step] = action_spec.clip(
            values[:, step], previous=preceding, dt=float(dt)
        )
        preceding = values[:, step]
    return values


def _risk_and_safety(
    candidates,
    true_safe,
    state,
    perceived,
    components,
    static,
    protocol,
):
    icode = _icode_controller(components["controller"])
    forecast_key = icode.config.probabilistic_obstacle_forecast_key
    forecasts = tuple(
        perceived.observation.auxiliary.get(forecast_key, ())
    )
    if not forecasts:
        return {
            "forecast_count": 0,
            "risk_accepted": np.zeros(candidates.shape[0], dtype=bool),
            "safety_nonstop": np.zeros(candidates.shape[0], dtype=bool),
            "risk_maximum_probability": np.full(
                candidates.shape[0], np.nan
            ),
        }
    predicted = icode.rollout(state, candidates)
    risk = evaluate_collision_risk(
        predicted[:, 1:, :2], forecasts, _risk_config(icode)
    )
    predicted_static = np.asarray([
        min(
            point_clearance(
                float(position[0]),
                float(position[1]),
                static,
                float(components["controller"].config.robot_radius),
            )
            for position in path[1:, :2]
        )
        for path in predicted
    ])
    accepted = (
        ~risk.hard_violation
        & (
            predicted_static
            >= float(
                protocol["online_audit"][
                    "minimum_predicted_static_clearance_m"
                ]
            )
        )
    )
    nonstop = np.zeros(candidates.shape[0], dtype=bool)
    for index in np.flatnonzero(accepted):
        reset = getattr(components["safety"], "reset", None)
        if callable(reset):
            reset()
        decision = components["safety"].arbitrate(
            ControlCommand(
                candidates[index, 0],
                float(perceived.observation.timestamp),
                "supervised_maneuver_gate_bc",
            ),
            perceived.guard,
        )
        executed = np.asarray(
            decision.executed_control.values, dtype=np.float64
        )
        nonstop[index] = bool(
            abs(float(executed[0]))
            >= float(
                protocol["online_audit"][
                    "safety_nonstop_minimum_v_mps"
                ]
            )
            or abs(float(executed[1]))
            >= float(
                protocol["online_audit"][
                    "safety_nonstop_minimum_omega_radps"
                ]
            )
        )
    return {
        "forecast_count": len(forecasts),
        "risk_accepted": accepted,
        "safety_nonstop": nonstop,
        "risk_maximum_probability": np.asarray(
            risk.maximum_step_probability, dtype=np.float64
        ),
        "risk_accepted_true_safe": accepted & true_safe,
    }


def _audit_state(protocol, dataset, row_index, model, payload, source):
    artifact = (ROOT / source["artifact"]).resolve()
    config = load_yaml(artifact / "config_resolved.yaml")
    rows = _read_rows(artifact / "trajectory.csv")
    anchor = int(dataset["anchor_index"][row_index])
    components = make_components(config, ROOT)
    plant = components["plant"]
    try:
        _, perceived, state, target = _replay_to_anchor(
            config, components, rows, anchor
        )
        observation = dataset["observations"][row_index].astype(np.float32)
        normalized_observation = (
            observation - np.asarray(payload["observation_mean"])
        ) / np.asarray(payload["observation_std"])
        with torch.no_grad():
            normalized = model(torch.as_tensor(
                normalized_observation[None], dtype=torch.float32
            ))[0].cpu().numpy()
        candidates = normalized_to_physical(
            normalized,
            components["action_spec"].lower,
            components["action_spec"].upper,
        )
        dt = float(config["experiment"]["control_dt"])
        candidates = _clip_sequences(
            components["action_spec"],
            candidates,
            _previous_control(rows, anchor),
            dt,
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
        metrics = _true_branch_metrics(
            plant,
            snapshot,
            candidates,
            dt,
            static,
            tuple(float(value["radius"]) for value in dynamic),
            float(components["controller"].config.robot_radius),
            np.asarray((target.pose.x, target.pose.y), dtype=np.float64),
            int(protocol["physical_audit"]["stopping_tail_steps"]),
        )
        physical = protocol["physical_audit"]
        true_safe = (
            ~metrics["collision"]
            & (
                metrics["minimum_static_clearance_m"]
                >= float(physical["minimum_static_clearance_m"])
            )
            & (
                metrics["minimum_dynamic_clearance_m"]
                >= float(physical["minimum_dynamic_clearance_m"])
            )
            & (
                metrics["terminal_dynamic_clearance_m"]
                >= float(physical["minimum_terminal_dynamic_clearance_m"])
            )
        )
        progressing = (
            true_safe
            & (
                metrics["local_target_progress_m"]
                >= float(physical["minimum_progress_m"])
            )
        )
        online = _risk_and_safety(
            candidates,
            true_safe,
            state,
            perceived,
            components,
            static,
            protocol,
        )
        full_chain = (
            true_safe
            & progressing
            & online["risk_accepted"]
            & online["safety_nonstop"]
        )
        return {
            "map": str(dataset["map_name"][row_index]),
            "seed": int(dataset["seed"][row_index]),
            "scenario_id": str(dataset["scenario_id"][row_index]),
            "anchor_index": anchor,
            "offset_s": float(dataset["offset_s"][row_index]),
            "forecast_count": int(online["forecast_count"]),
            "candidate_count": int(candidates.shape[0]),
            "true_collision": metrics["collision"].astype(bool).tolist(),
            "true_safe": true_safe.astype(bool).tolist(),
            "true_progressing": progressing.astype(bool).tolist(),
            "risk_accepted": online["risk_accepted"].astype(bool).tolist(),
            "safety_nonstop": online["safety_nonstop"].astype(bool).tolist(),
            "full_chain": full_chain.astype(bool).tolist(),
            "minimum_static_clearance_m": metrics[
                "minimum_static_clearance_m"
            ].tolist(),
            "minimum_dynamic_clearance_m": metrics[
                "minimum_dynamic_clearance_m"
            ].tolist(),
            "progress_m": metrics["local_target_progress_m"].tolist(),
            "risk_maximum_probability": online[
                "risk_maximum_probability"
            ].tolist(),
            "first_segment_mean_yaw": np.mean(
                candidates[:, :12, 1], axis=1
            ).tolist(),
        }
    finally:
        plant.close()


def _summarize(protocol, states):
    candidates = sum(state["candidate_count"] for state in states)
    safe = sum(sum(state["true_safe"]) for state in states)
    risk = sum(sum(state["risk_accepted"]) for state in states)
    nonstop = sum(
        sum(
            accepted and moving
            for accepted, moving in zip(
                state["risk_accepted"], state["safety_nonstop"]
            )
        )
        for state in states
    )
    risk_true_safe = sum(
        sum(
            accepted and safe_value
            for accepted, safe_value in zip(
                state["risk_accepted"], state["true_safe"]
            )
        )
        for state in states
    )
    risk_collisions = sum(
        sum(
            accepted and collision
            for accepted, collision in zip(
                state["risk_accepted"], state["true_collision"]
            )
        )
        for state in states
    )
    covered = sum(any(state["full_chain"]) for state in states)
    progressing_covered = sum(
        any(
            full and progress
            for full, progress in zip(
                state["full_chain"], state["true_progressing"]
            )
        )
        for state in states
    )
    state_count = len(states)
    metrics = {
        "state_count": state_count,
        "candidate_count": candidates,
        "physical_safe_candidate_fraction": safe / max(candidates, 1),
        "risk_accepted_candidate_fraction": risk / max(candidates, 1),
        "state_full_chain_coverage_fraction": covered / max(state_count, 1),
        "progressing_full_chain_state_fraction": (
            progressing_covered / max(state_count, 1)
        ),
        "risk_accepted_true_collisions": risk_collisions,
        "risk_accepted_physical_safety_precision": (
            risk_true_safe / max(risk, 1)
        ),
        "safety_nonstop_fraction_of_risk_accepted": (
            nonstop / max(risk, 1)
        ),
    }
    b = protocol["gate_b"]
    c = protocol["gate_c"]
    checks = {
        "state_full_chain_coverage": (
            metrics["state_full_chain_coverage_fraction"]
            >= float(b["minimum_state_full_chain_coverage_fraction"])
        ),
        "physical_safe_candidate_fraction": (
            metrics["physical_safe_candidate_fraction"]
            >= float(b["minimum_physical_safe_candidate_fraction"])
        ),
        "risk_accepted_candidate_fraction": (
            metrics["risk_accepted_candidate_fraction"]
            >= float(b["minimum_risk_accepted_candidate_fraction"])
        ),
        "progressing_full_chain_state_fraction": (
            metrics["progressing_full_chain_state_fraction"]
            >= float(b["minimum_progressing_full_chain_state_fraction"])
        ),
        "risk_accepted_true_collisions": (
            metrics["risk_accepted_true_collisions"]
            <= int(c["maximum_risk_accepted_true_collisions"])
        ),
        "risk_accepted_physical_safety_precision": (
            metrics["risk_accepted_physical_safety_precision"]
            >= float(c["minimum_risk_accepted_physical_safety_precision"])
        ),
        "safety_nonstop_fraction": (
            metrics["safety_nonstop_fraction_of_risk_accepted"]
            >= float(c["minimum_safety_nonstop_fraction_of_risk_accepted"])
        ),
    }
    return metrics, checks


def audit(protocol_path, output):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path)
    dataset_path = (ROOT / protocol["dataset"]["file"]).resolve()
    if _sha256(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("Gate B-C dataset hash mismatch")
    dataset = np.load(dataset_path)
    model, payload = _load_actor(protocol)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Gate B-C output already contains evidence")
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "protocol_resolved.yaml"
    resolved.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    split = str(protocol["dataset"]["audit_split"])
    mask = dataset["split"] == split
    unique = {}
    for index in np.flatnonzero(mask):
        key = (
            str(dataset["scenario_id"][index]),
            int(dataset["anchor_index"][index]),
        )
        unique.setdefault(key, int(index))
    dataset_protocol = yaml.safe_load(
        (ROOT / protocol["dataset"]["protocol"]).read_text(encoding="utf-8")
    )
    states = []
    for _, index in sorted(unique.items()):
        map_name = str(dataset["map_name"][index])
        states.append(_audit_state(
            protocol,
            dataset,
            index,
            model,
            payload,
            dataset_protocol["sources"][map_name],
        ))
    metrics, checks = _summarize(protocol, states)
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "status": status,
        "gate_b_candidate_coverage_pass": bool(all(
            checks[name] for name in (
                "state_full_chain_coverage",
                "physical_safe_candidate_fraction",
                "risk_accepted_candidate_fraction",
                "progressing_full_chain_state_fraction",
            )
        )),
        "gate_c_independent_safety_pass": bool(all(
            checks[name] for name in (
                "risk_accepted_true_collisions",
                "risk_accepted_physical_safety_precision",
                "safety_nonstop_fraction",
            )
        )),
        "fresh_seed_expansion_authorized": status == "pass",
        "closed_loop_gate_d_authorized": False,
        "formal_claim_authorized": False,
        "metrics": metrics,
        "checks": checks,
        "states": states,
    }
    result_path = output / "gate_bc_result.json"
    _write_json(result_path, result)
    _write_json(output / "artifact_manifest.json", {
        "git_sha": git_sha(ROOT),
        "protocol_config_hash": config_hash(protocol),
        "protocol_sha256": _sha256(resolved),
        "dataset_sha256": _sha256(dataset_path),
        "actor_checkpoint_sha256": protocol["actor"]["sha256"],
        "result_sha256": _sha256(result_path),
        "actor_future_truth_input": False,
        "risk_icode_mppi_safety_changed": False,
        "formal_server_launch_authorized": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if status == "pass" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol.resolve())
    _load_actor(protocol)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_pass",
            "audit_split": protocol["dataset"]["audit_split"],
            "actor_future_truth_input": False,
            "risk_icode_mppi_safety_frozen": True,
            "formal_server_launch_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")
    return audit(args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
