#!/usr/bin/env python3
"""Audit nominal-only state-observation domain calibration."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_cross_plant_nominal_calibration import (  # noqa: E402
    summarize,
)
from mobile_robot_mppi.core.config import load_yaml  # noqa: E402


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    input_dir = _resolved(args.input_dir)
    result = summarize(
        config,
        input_dir,
        calibration_key="observation_domain_calibration",
        interpretation_guard=(
            "Selection used traditional nominal MPPI only on the fixed anchor "
            "plant; no MLP, ICODE or RL outcome was observed. Raw wheel "
            "odometry is a reported stress domain and is not selected by the "
            "primary noise/latency calibration gate."
        ),
    )
    (input_dir / "observation_domain_calibration_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

