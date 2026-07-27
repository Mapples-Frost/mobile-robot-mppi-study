"""Run one frozen paired closed-loop Gate D development episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.complex_full_method import (
    build_complex_full_config,
)
from mobile_robot_mppi.core.config import config_hash
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_maneuver_actor_gate_d_v1.yaml"
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_protocol(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        value.get("protocol")
        != "complex_supervised_maneuver_actor_closed_loop_gate_d_v1"
    ):
        raise ValueError("Gate D protocol mismatch")
    if bool(value["prerequisites"]["expansion_data_may_enter_training"]):
        raise ValueError("failed expansion data cannot enter Gate D training")
    if bool(value["actor"]["retraining_authorized"]):
        raise ValueError("Gate D Actor must remain frozen")
    if bool(value["actor"]["privileged_future_inputs"]):
        raise ValueError("Gate D Actor input must remain causal")
    if bool(value["online_interface"]["rollout_budget_may_increase"]):
        raise ValueError("Gate D cannot increase the rollout budget")
    if bool(value["frozen_contract"]["paid_server_launch_authorized"]):
        raise ValueError("Gate D cannot launch the paid server")
    for key in (
        "maps_changed",
        "actor_checkpoint_changed",
        "icode_changed",
        "collision_risk_changed",
        "mppi_changed",
        "safety_changed",
        "rollout_budget_changed",
        "map_specific_logic_allowed",
    ):
        if bool(value["frozen_contract"][key]):
            raise ValueError("Gate D changed frozen contract: %s" % key)
    for item in (
        value["prerequisites"]["gate_bc_result"],
        value["prerequisites"]["failed_expansion_result"],
        value["actor"]["checkpoint"],
    ):
        if not (ROOT / item).is_file():
            raise FileNotFoundError("Gate D prerequisite missing: %s" % item)
    hashes = (
        ("gate_bc_result", "gate_bc_result_sha256"),
        ("failed_expansion_result", "failed_expansion_result_sha256"),
    )
    for path_key, hash_key in hashes:
        actual = _sha256(ROOT / value["prerequisites"][path_key])
        if actual != str(value["prerequisites"][hash_key]).lower():
            raise ValueError("Gate D prerequisite hash mismatch: %s" % path_key)
    if _sha256(ROOT / value["actor"]["checkpoint"]) != str(
        value["actor"]["sha256"]
    ).lower():
        raise ValueError("Gate D Actor checkpoint hash mismatch")
    gate_bc = json.loads(
        (ROOT / value["prerequisites"]["gate_bc_result"]).read_text(
            encoding="utf-8"
        )
    )
    if gate_bc.get("status") != value["prerequisites"][
        "required_gate_bc_status"
    ]:
        raise ValueError("Gate B-C prerequisite did not pass")
    return value


def _scheduled_block(protocol, map_name, seed):
    matches = [
        item for item in protocol["paired_blocks"]
        if item["map"] == str(map_name) and int(item["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError("map/seed is not a frozen Gate D paired block")
    return matches[0]


def build_gate_d_config(protocol_path, map_name, seed, arm):
    protocol = _load_protocol(protocol_path)
    block = _scheduled_block(protocol, map_name, seed)
    arm = str(arm)
    if arm not in block["order"]:
        raise ValueError("arm is not scheduled in the paired block")
    config = build_complex_full_config(map_name, int(seed))
    actor = protocol["actor"]
    enabled = bool(
        protocol["arms"][arm]["supervised_maneuver_actor_enabled"]
    )
    config["planner"]["paper_rl_driven"][
        "supervised_maneuver_actor"
    ] = {
        "enabled": enabled,
        "checkpoint": str(actor["checkpoint"]),
        "expected_sha256": str(actor["sha256"]),
        "device": "cpu",
        "execution_authority": "proposal_only",
        "maximum_sequences_per_decision": int(
            protocol["online_interface"][
                "maximum_supervised_sequences_per_decision"
            ]
        ),
    }
    config["experiment"]["name"] = (
        "complex_maneuver_gate_d__%s__seed%d__%s"
        % (map_name, int(seed), arm)
    )
    config["gate_d_contract"] = {
        "protocol": str(Path(protocol_path).resolve()),
        "map": str(map_name),
        "seed": int(seed),
        "arm": arm,
        "paired_order": list(block["order"]),
        "actor_checkpoint_sha256": str(actor["sha256"]),
        "total_rollouts_per_decision": int(
            protocol["online_interface"]["total_rollouts_per_decision"]
        ),
        "cold_start_diagnostics_revision": int(
            protocol["preexecution_amendment"]["revision"]
        ),
        "development_only": True,
        "formal_claim_authorized": False,
    }
    iterations = int(
        config["planner"]["paper_rl_driven"]["iterations"]
    )
    rollouts = int(config["planner"]["num_samples"]) * iterations
    if rollouts != int(
        protocol["online_interface"]["total_rollouts_per_decision"]
    ):
        raise ValueError("Gate D changed total rollout budget")
    return config


def _write_yaml(path, value):
    Path(path).write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--map", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--arm",
        choices=("frozen_full", "full_plus_bootstrap_actor"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Gate D output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(args.protocol)
    config = build_gate_d_config(
        args.protocol, args.map, args.seed, args.arm
    )
    _write_yaml(output / "protocol_resolved.yaml", protocol)
    _write_yaml(output / "config_resolved.yaml", config)
    (output / "config_sha256.txt").write_text(
        config_hash(config) + "\n", encoding="ascii"
    )
    result = ExperimentRunner(
        config,
        ROOT,
        output_dir=output,
        headless=not bool(args.viewer),
    ).run()
    print(json.dumps(result.summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
