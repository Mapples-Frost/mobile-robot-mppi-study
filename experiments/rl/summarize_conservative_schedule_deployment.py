#!/usr/bin/env python3
"""Audit an L79 deployment validation of L78-selected checkpoints."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml  # noqa: E402
from experiments.rl.summarize_checkpoint_rule_validation import (  # noqa: E402
    summarize,
)
from experiments.rl.summarize_correction_support_gate import (  # noqa: E402
    _write_csv,
)


def _resolved(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    design = config["rl"]["cross_layer_factorial"]
    if str(design.get("study_label")) != "L79":
        raise ValueError("expected an L79 deployment design")
    result, episodes, effects = summarize(config, args.input_dir)
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "l79_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "l79_episodes.csv", episodes)
    _write_csv(output / "l79_paired_effects.csv", effects)
    _write_csv(output / "l79_by_block.csv", result["by_model_block"])
    _write_csv(
        output / "l79_by_scene.csv", result["by_scene"].values()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

