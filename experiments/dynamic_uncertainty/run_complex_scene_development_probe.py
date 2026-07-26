"""Run one non-overwriting, headless complex-scene development episode."""

import argparse
import json
from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"development output already contains evidence: {output}"
        )

    config = load_yaml(args.config.resolve())
    config["experiment"]["seed"] = int(args.seed)
    config["experiment"]["name"] = (
        f"{config['experiment']['name']}_development_seed_{args.seed}"
    )
    if args.max_steps is not None:
        config["experiment"]["max_steps"] = int(args.max_steps)

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
