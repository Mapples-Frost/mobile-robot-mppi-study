"""Run one chapter-1 budget-extension continuation episode.

Identical to the corresponding segment_half_thickness_ab_v1 treatment episode
except that experiment.max_steps is raised from 1200 to 2050. Steps 0-1199 must
therefore reproduce the reference artifact exactly; the analyzer verifies it.

Usage:
    python -m experiments.dynamic_uncertainty.run_budget_extension_screen \
        --seed 791101302 \
        --output research_artifacts/budget_extension_screen_v1/seed791101302
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

DEFAULT_PROTOCOL = ROOT / "configs/research/budget_extension_screen_v1.yaml"


def _write_yaml(path: Path, value) -> None:
    Path(path).write_text(
        yaml.safe_dump(value, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )


def _load_protocol(path):
    v = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if v.get("protocol") != "budget_extension_screen_v1":
        raise ValueError("budget-extension protocol mismatch")
    if v.get("status") != "frozen_before_execution":
        raise ValueError("protocol is not frozen")
    if not bool(v.get("development_only")):
        raise ValueError("execution must remain development-only")
    if bool(v.get("formal_claim_authorized")):
        raise ValueError("protocol must not authorise formal claims")
    i = v["intervention"]
    if i["parameter"] != "experiment.max_steps":
        raise ValueError("intervention parameter changed")
    if int(i["reference"]) != 1200 or int(i["treatment"]) != 2050:
        raise ValueError("budget values changed")
    hc = v["held_constant"]
    if hc["known_static_map_segment_half_thickness"] is not True:
        raise ValueError("thickness flag must be enabled")
    if str(hc["noise_basis"]) != "ar1:2.0":
        raise ValueError("held-constant noise basis changed")
    seeds = [int(s) for s in v["design"]["seeds"]]
    if seeds != [791101302, 791101305, 791101310]:
        raise ValueError("seed list changed")
    # The arrested seeds must never enter this screen.
    if set(seeds) & {791101301, 791101307, 791101311}:
        raise ValueError("arrested near_body_hard_stop seeds are excluded")
    if int(v["design"]["episodes"]) != 3:
        raise ValueError("episode count changed")
    f = v["frozen_components"]
    if int(f["total_rollouts_per_forecast_valid_cycle"]) != 600:
        raise ValueError("rollout budget changed")
    if int(f["horizon_steps"]) != 36:
        raise ValueError("horizon changed")
    if bool(v["design"]["seed_replacement_allowed"]):
        raise ValueError("seed replacement is forbidden")
    return v


def build_screen_config(protocol_path, seed):
    protocol = _load_protocol(protocol_path)
    if int(seed) not in {int(s) for s in protocol["design"]["seeds"]}:
        raise ValueError("seed %d is not scheduled" % int(seed))

    map_name = str(protocol["map"]["name"])
    hc = protocol["held_constant"]
    build_basis(str(hc["noise_basis"]), float(protocol["frozen_components"]["control_dt"]))

    config = build_complex_full_config(map_name, int(seed))
    planner = config["planner"]

    reference_steps = int(protocol["intervention"]["reference"])
    if int(config["experiment"]["max_steps"]) != reference_steps:
        raise ValueError(
            "map max_steps is %d, expected the reference %d; the continuation "
            "would not reproduce the reference prefix"
            % (int(config["experiment"]["max_steps"]), reference_steps)
        )

    planner["noise_basis"] = str(hc["noise_basis"])
    planner["known_static_map_segment_half_thickness"] = bool(
        hc["known_static_map_segment_half_thickness"]
    )
    config["experiment"]["max_steps"] = int(protocol["intervention"]["treatment"])
    config["experiment"]["name"] = (
        "budget_extension__%s__seed%d" % (map_name, int(seed))
    )
    config["budget_extension_contract"] = {
        "protocol": str(Path(protocol_path).resolve()),
        "map": map_name,
        "seed": int(seed),
        "reference_max_steps": reference_steps,
        "extended_max_steps": int(protocol["intervention"]["treatment"]),
        "noise_basis": str(hc["noise_basis"]),
        "known_static_map_segment_half_thickness": True,
        "reference_artifact": (
            "research_artifacts/segment_half_thickness_ab_v1/seed%d/treatment" % int(seed)
        ),
        "development_only": True,
        "formal_claim_authorized": False,
    }

    rollouts = int(planner["num_samples"]) * int(planner["paper_rl_driven"]["iterations"])
    if rollouts != 600:
        raise ValueError("rollout budget changed (%d)" % rollouts)
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--viewer", action="store_true",
                   help="inspection only; never for measured episodes")
    args = p.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output already contains evidence: %s" % output)
    output.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(args.protocol)
    config = build_screen_config(args.protocol, args.seed)

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
