"""Run one non-overwriting complex-map Actor-teacher upper-bound episode."""

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

from experiments.dynamic_uncertainty.complex_actor_teacher import (
    DEFAULT_PROTOCOL,
    build_complex_actor_teacher_config,
    load_teacher_protocol,
)
from experiments.dynamic_uncertainty.complex_full_method import MAPS
from mobile_robot_mppi.core.config import config_hash
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _write_yaml(path, value):
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", choices=sorted(MAPS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "teacher-probe output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    protocol = load_teacher_protocol(args.protocol)
    expected = {
        int(value)
        for value in protocol["development_seeds"][args.map]
    }
    if int(args.seed) not in expected:
        raise ValueError("teacher-probe seed is not frozen for this map")
    config = build_complex_actor_teacher_config(
        args.map,
        args.seed,
        protocol_path=args.protocol,
        maximum_steps=args.max_steps,
    )
    _write_yaml(output / "resolved_config.yaml", config)
    (output / "resolved_config_sha256.txt").write_text(
        config_hash(config) + "\n", encoding="ascii"
    )
    result = ExperimentRunner(
        config,
        ROOT,
        output_dir=output,
        headless=True,
    ).run()
    print(json.dumps(result.summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
