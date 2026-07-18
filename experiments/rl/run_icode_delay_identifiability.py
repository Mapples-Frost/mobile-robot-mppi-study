#!/usr/bin/env python3
"""Run every model block of the preregistered L38 delay diagnostic."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.run_cross_layer_factorial import main as run_block_main
from mobile_robot_mppi.core.config import load_yaml


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config_path = _resolved_path(args.config)
    config = load_yaml(config_path)
    design = config["rl"]["cross_layer_factorial"]
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for block_index, block in enumerate(design["model_blocks"]):
        block_dir = output / ("block_%d" % block_index)
        block_args = [
            "--config", str(config_path),
            "--rl-checkpoint", str(_resolved_path(block["rl_checkpoint"])),
            "--icode-checkpoint", str(_resolved_path(block["icode_checkpoint"])),
            "--model-block", str(block_index),
            "--output-dir", str(block_dir),
        ]
        if args.max_steps is not None:
            block_args.extend(["--max-steps", str(int(args.max_steps))])
        run_block_main(block_args)
        metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
        completed.append({
            "model_block": block_index,
            "icode_training_seed": int(metadata["icode_training_seed"]),
            "episodes": int(metadata["episodes"]),
            "metadata": str(block_dir / "metadata.json"),
        })
    payload = {
        "design_id": "l38_icode_delay_identifiability_v1",
        "source_config": str(config_path),
        "model_blocks": completed,
        "observed_episodes": int(sum(row["episodes"] for row in completed)),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
