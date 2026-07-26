"""Run one non-overwriting B11 Full episode on a frozen complex map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
for _value in (ROOT, ROOT / "src"):
    if str(_value) not in sys.path:
        sys.path.insert(0, str(_value))

from experiments.dynamic_uncertainty.complex_full_method import (
    MAPS,
    ROOT,
    build_complex_full_config,
)
from mobile_robot_mppi.core.config import config_hash
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _write_resolved(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            value,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", choices=sorted(MAPS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "development output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    config = build_complex_full_config(args.map, args.seed)
    if args.max_steps is not None:
        if int(args.max_steps) <= 0:
            raise ValueError("max steps must be positive")
        config["experiment"]["max_steps"] = int(args.max_steps)
    _write_resolved(output / "resolved_config.yaml", config)
    (output / "resolved_config_sha256.txt").write_text(
        config_hash(config) + "\n", encoding="ascii"
    )

    result = ExperimentRunner(
        config,
        ROOT,
        output_dir=output,
        headless=not bool(args.viewer),
    ).run()
    print(
        json.dumps(result.summary, indent=2, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
