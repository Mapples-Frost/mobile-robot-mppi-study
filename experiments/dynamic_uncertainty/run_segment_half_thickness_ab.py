"""Run one frozen segment-half-thickness A/B development episode.

Control  : planner.known_static_map_segment_half_thickness = false (legacy)
Treatment: planner.known_static_map_segment_half_thickness = true  (corrected)

``noise_basis`` is held at "ar1:2.0" in BOTH arms, because the deadlock under
test only appears reliably under AR(1). Everything else is frozen.

Usage:
    python -m experiments.dynamic_uncertainty.run_segment_half_thickness_ab \
        --map chapter1 --seed 791101301 --arm control \
        --output research_artifacts/segment_half_thickness_ab_v1/seed791101301/control
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

DEFAULT_PROTOCOL = (
    ROOT / "configs/research/segment_half_thickness_ab_development_v1.yaml"
)


def _write_yaml(path: Path, value) -> None:
    Path(path).write_text(
        yaml.safe_dump(value, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )


def _load_protocol(path):
    v = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if v.get("protocol") != "segment_half_thickness_ab_development_v1":
        raise ValueError("segment-half-thickness protocol mismatch")
    if v.get("status") != "frozen_before_execution":
        raise ValueError("protocol is not frozen")
    if not bool(v.get("development_only")):
        raise ValueError("execution must remain development-only")
    if bool(v.get("formal_claim_authorized")):
        raise ValueError("protocol must not authorise formal claims")
    i = v["intervention"]
    if i["parameter"] != "planner.known_static_map_segment_half_thickness":
        raise ValueError("intervention parameter changed")
    if bool(i["control"]) is not False or bool(i["treatment"]) is not True:
        raise ValueError("arm assignment changed")
    if str(v["held_constant"]["noise_basis"]) != "ar1:2.0":
        raise ValueError("held-constant noise basis changed")
    seeds = [int(s) for s in v["design"]["seeds"]]
    if seeds != list(range(791101301, 791101313)):
        raise ValueError("paired seed list changed")
    if int(v["design"]["episodes"]) != 24:
        raise ValueError("episode count changed")
    t = v["decision_rule"]["thresholds"]
    if (
        int(t["treatment_only_collision_pairs"]) != 2
        or int(t["net_success_pairs"]) != 2
        or float(t["median_goal_distance_reduction_m"]) != 0.25
        or float(t["minimum_clearance_regression_m"]) != 0.05
    ):
        raise ValueError("decision thresholds changed")
    f = v["frozen_components"]
    if int(f["total_rollouts_per_forecast_valid_cycle"]) != 600:
        raise ValueError("rollout budget changed")
    if int(f["horizon_steps"]) != 36:
        raise ValueError("horizon changed")
    if bool(v["design"]["seed_replacement_allowed"]):
        raise ValueError("seed replacement is forbidden")
    if bool(v["design"]["map_specific_logic_allowed"]):
        raise ValueError("map-specific logic is forbidden")
    return v


def build_ab_config(protocol_path, map_name, seed, arm):
    protocol = _load_protocol(protocol_path)
    if str(map_name) != str(protocol["map"]["name"]):
        raise ValueError("map is not the scheduled map")
    if int(seed) not in {int(s) for s in protocol["design"]["seeds"]}:
        raise ValueError("seed %d is not a scheduled paired seed" % int(seed))
    if arm not in ("control", "treatment"):
        raise ValueError("arm must be 'control' or 'treatment'")

    half = bool(protocol["intervention"]["treatment" if arm == "treatment" else "control"])
    basis = str(protocol["held_constant"]["noise_basis"])
    build_basis(basis, float(protocol["frozen_components"]["control_dt"]))

    config = build_complex_full_config(map_name, int(seed))
    planner = config["planner"]
    if int(planner["horizon"]) != int(protocol["frozen_components"]["horizon_steps"]):
        raise ValueError("resolved horizon does not match the frozen protocol")
    if not bool(planner.get("known_static_map_cost_enabled")):
        raise ValueError("known-static-map cost must be enabled for this protocol")

    planner["noise_basis"] = basis
    planner["known_static_map_segment_half_thickness"] = half

    config["experiment"]["name"] = (
        "segment_half_thickness_ab__%s__seed%d__%s" % (map_name, int(seed), arm)
    )
    config["segment_half_thickness_ab_contract"] = {
        "protocol": str(Path(protocol_path).resolve()),
        "map": str(map_name),
        "seed": int(seed),
        "arm": str(arm),
        "known_static_map_segment_half_thickness": half,
        "noise_basis": basis,
        "control_is_legacy_full_thickness": half is False,
        "development_only": True,
        "formal_claim_authorized": False,
        "total_rollouts_per_forecast_valid_cycle": 600,
    }

    rollouts = int(planner["num_samples"]) * int(planner["paper_rl_driven"]["iterations"])
    if rollouts != 600:
        raise ValueError("execution changed the rollout budget (%d)" % rollouts)
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--map", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", choices=("control", "treatment"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--viewer", action="store_true",
                   help="inspection only; never for measured episodes")
    args = p.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output already contains evidence: %s" % output)
    output.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(args.protocol)
    config = build_ab_config(args.protocol, args.map, args.seed, args.arm)

    _write_yaml(output / "protocol_resolved.yaml", protocol)
    (output / "protocol_sha256.txt").write_text(
        hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    _write_yaml(output / "config_resolved.yaml", config)
    (output / "config_sha256.txt").write_text(
        config_hash(config) + "\n", encoding="ascii"
    )

    result = ExperimentRunner(
        config, ROOT, output_dir=output, headless=not bool(args.viewer)
    ).run()
    print(json.dumps(result.summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
