"""Run one frozen noise-basis A/B development episode on a complex map.

Control arm reproduces the historical sampler bit-exactly (noise_basis "iid").
Treatment arm swaps only the temporal correlation of the Gaussian perturbation
(noise_basis "ar1:2.0"); the per-step marginal variance, rollout budget,
horizon, action bounds, cost terms, actor, risk, ICODE and safety contracts are
all unchanged.

Usage:
    python -m experiments.dynamic_uncertainty.run_noise_basis_ab_development \
        --map chapter1 --seed 791101201 --arm control \
        --output research_artifacts/noise_basis_ab_v1/seed791101201/control
"""

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

from experiments.dynamic_uncertainty.complex_full_method import (  # noqa: E402
    build_complex_full_config,
)
from mobile_robot_mppi.core.config import config_hash  # noqa: E402
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner  # noqa: E402
from mobile_robot_mppi.sampling.bases import build_basis  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "configs/research/noise_basis_ab_development_v1.yaml"


def _write_yaml(path: Path, value) -> None:
    Path(path).write_text(
        yaml.safe_dump(value, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def _load_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("protocol") != "noise_basis_ab_development_v1":
        raise ValueError("noise-basis protocol mismatch")
    if value.get("status") != "frozen_pre_execution_after_wiring_fix":
        raise ValueError("noise-basis protocol is not frozen")
    if not bool(value.get("development_only")):
        raise ValueError("noise-basis execution must remain development-only")
    if bool(value.get("formal_claim_authorized")):
        raise ValueError("noise-basis protocol must not authorise formal claims")
    intervention = value["intervention"]
    if intervention["parameter"] != "planner.noise_basis":
        raise ValueError("noise-basis intervention parameter changed")
    if intervention["control"] != "iid":
        raise ValueError("noise-basis control arm must be the historical sampler")
    if intervention["treatment"] != "ar1:2.0":
        raise ValueError("noise-basis treatment arm must be ar1:2.0")
    seeds = [int(seed) for seed in value["design"]["seeds"]]
    if seeds != list(range(791101201, 791101213)):
        raise ValueError("noise-basis paired seed list changed")
    if int(value["design"]["episodes"]) != 24:
        raise ValueError("noise-basis episode count changed")
    thresholds = value["decision_rule"]["thresholds"]
    if (
        int(thresholds["treatment_only_collision_pairs"]) != 2
        or int(thresholds["net_success_pairs"]) != 2
        or float(thresholds["median_goal_distance_reduction_m"]) != 0.25
    ):
        raise ValueError("noise-basis decision thresholds changed")
    frozen = value["frozen_components"]
    if int(frozen["total_rollouts_per_forecast_valid_cycle"]) != 600:
        raise ValueError("noise-basis rollout budget changed")
    if int(frozen["horizon_steps"]) != 36:
        raise ValueError("noise-basis horizon changed")
    if bool(value["design"]["seed_replacement_allowed"]):
        raise ValueError("seed replacement is forbidden")
    if bool(value["design"]["map_specific_logic_allowed"]):
        raise ValueError("map-specific logic is forbidden")
    return value


def _arm_basis(protocol, arm):
    intervention = protocol["intervention"]
    mapping = {"control": intervention["control"], "treatment": intervention["treatment"]}
    if arm not in mapping:
        raise ValueError("arm must be 'control' or 'treatment'")
    return str(mapping[arm])


def build_ab_config(protocol_path, map_name, seed, arm):
    protocol = _load_protocol(protocol_path)

    if str(map_name) != str(protocol["map"]["name"]):
        raise ValueError(
            "map %r is not the scheduled map %r"
            % (map_name, protocol["map"]["name"])
        )
    scheduled = {int(v) for v in protocol["design"]["seeds"]}
    if int(seed) not in scheduled:
        raise ValueError("seed %d is not a scheduled paired seed" % int(seed))

    basis = _arm_basis(protocol, arm)
    build_basis(basis, float(protocol["frozen_components"]["control_dt"]))

    config = build_complex_full_config(map_name, int(seed))
    planner = config["planner"]

    if int(planner["horizon"]) != int(protocol["frozen_components"]["horizon_steps"]):
        raise ValueError("resolved horizon does not match the frozen protocol")
    planner["noise_basis"] = basis

    config["experiment"]["name"] = (
        "noise_basis_ab__%s__seed%d__%s" % (map_name, int(seed), arm)
    )
    config["noise_basis_ab_contract"] = {
        "protocol": str(Path(protocol_path).resolve()),
        "map": str(map_name),
        "seed": int(seed),
        "arm": str(arm),
        "noise_basis": basis,
        "control_is_bit_exact_historical_sampler": basis == "iid",
        "development_only": True,
        "formal_claim_authorized": False,
        "total_rollouts_per_forecast_valid_cycle": 600,
    }

    rollouts = int(planner["num_samples"]) * int(
        planner["paper_rl_driven"]["iterations"]
    )
    if rollouts != 600:
        raise ValueError(
            "noise-basis execution changed the rollout budget (%d)" % rollouts
        )
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--map", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="development inspection only; never use for measured episodes",
    )
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "noise-basis output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(args.protocol)
    config = build_ab_config(args.protocol, args.map, args.seed, args.arm)

    _write_yaml(output / "protocol_resolved.yaml", protocol)
    protocol_bytes = Path(args.protocol).read_bytes()
    (output / "protocol_sha256.txt").write_text(
        hashlib.sha256(protocol_bytes).hexdigest() + "\n",
        encoding="ascii",
    )
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
