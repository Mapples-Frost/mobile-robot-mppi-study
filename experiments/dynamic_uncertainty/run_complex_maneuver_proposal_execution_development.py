"""Run one frozen proposal-to-execution consistency development episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_complex_supervised_maneuver_actor_gate_d import (
    DEFAULT_PROTOCOL as DEFAULT_GATE_D_PROTOCOL,
    _write_yaml,
    build_gate_d_config,
)
from mobile_robot_mppi.core.config import config_hash
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/complex_maneuver_proposal_execution_consistency_v1.yaml"
)
ARMS = {
    "margin0_control": 0.0,
    "margin002_treatment": 0.02,
}


def _load_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("mechanism_family") != "proposal_to_execution_consistency":
        raise ValueError("proposal-execution protocol mismatch")
    if value.get("status") != "frozen_before_implementation":
        raise ValueError("proposal-execution protocol is not frozen")
    if not bool(value.get("development_only")):
        raise ValueError("proposal-execution execution must remain development-only")
    rounds = {
        int(item["round"]): float(item["value"])
        for item in value["intervention"]["rounds"]
    }
    if rounds != {1: 0.02, 2: 0.05, 3: 0.10}:
        raise ValueError("proposal-execution round schedule changed")
    if float(value["intervention"]["baseline"]) != 0.0:
        raise ValueError("proposal-execution baseline changed")
    blocks = {
        (str(item["map"]), int(item["seed"]))
        for item in value["opened_development_blocks"]
    }
    expected = {
        ("chapter1", 791101019),
        ("chapter2", 790202019),
        ("chapter3", 791103019),
    }
    if blocks != expected:
        raise ValueError("opened proposal-execution blocks changed")
    frozen = value["frozen_components"]
    if bool(frozen["map_specific_logic_allowed"]):
        raise ValueError("map-specific proposal-execution logic is forbidden")
    if bool(frozen["seed_replacement_allowed"]):
        raise ValueError("seed replacement is forbidden")
    if int(frozen["total_rollouts_per_forecast_valid_cycle"]) != 600:
        raise ValueError("proposal-execution rollout budget changed")
    return value


def _scheduled_block(protocol, map_name, seed):
    block = (str(map_name), int(seed))
    allowed = {
        (str(item["map"]), int(item["seed"]))
        for item in protocol["opened_development_blocks"]
    }
    if block not in allowed:
        raise ValueError("map/seed is not an opened development block")
    return block


def build_development_config(
    protocol_path,
    gate_d_protocol_path,
    map_name,
    seed,
    arm,
):
    protocol = _load_protocol(protocol_path)
    _scheduled_block(protocol, map_name, seed)
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError("unsupported proposal-execution development arm")
    config = build_gate_d_config(
        gate_d_protocol_path,
        map_name,
        int(seed),
        "full_plus_bootstrap_actor",
    )
    paper = config["planner"]["paper_rl_driven"]
    if not bool(paper["supervised_maneuver_actor"]["enabled"]):
        raise ValueError("proposal-execution development requires the Actor")
    margin = float(ARMS[arm])
    paper["same_cycle_guided_relative_margin"] = margin
    config["experiment"]["name"] = (
        "complex_proposal_execution__%s__seed%d__%s"
        % (map_name, int(seed), arm)
    )
    config["proposal_execution_contract"] = {
        "protocol": str(Path(protocol_path).resolve()),
        "gate_d_protocol": str(Path(gate_d_protocol_path).resolve()),
        "map": str(map_name),
        "seed": int(seed),
        "arm": arm,
        "round": 0 if arm == "margin0_control" else 1,
        "same_cycle_guided_relative_margin": margin,
        "development_only": True,
        "formal_claim_authorized": False,
        "total_rollouts_per_forecast_valid_cycle": 600,
    }
    rollouts = int(config["planner"]["num_samples"]) * int(
        paper["iterations"]
    )
    if rollouts != 600:
        raise ValueError("proposal-execution execution changed rollout budget")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--gate-d-protocol", type=Path, default=DEFAULT_GATE_D_PROTOCOL
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "proposal-execution output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(args.protocol)
    config = build_development_config(
        args.protocol,
        args.gate_d_protocol,
        args.map,
        args.seed,
        args.arm,
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
