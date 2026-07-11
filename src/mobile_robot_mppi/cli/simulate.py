import argparse
import copy
import json
from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def build_parser():
    parser = argparse.ArgumentParser(description="Run the unified MPPI research simulation")
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("--output-dir")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--prediction-mode", choices=("nominal", "oracle_residual", "mlp_residual", "icode_residual"))
    parser.add_argument("--checkpoint")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = copy.deepcopy(load_yaml(args.config))
    if args.goal is not None:
        config["task"]["type"] = "point_goal"
        config["task"]["position"] = list(args.goal)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.max_steps is not None:
        config["experiment"]["max_steps"] = args.max_steps
    if args.num_samples is not None:
        config["planner"]["num_samples"] = args.num_samples
    if args.prediction_mode is not None:
        config["planner"]["prediction_mode"] = args.prediction_mode
    if args.checkpoint is not None:
        config["planner"]["checkpoint"] = args.checkpoint
    project_root = Path(__file__).resolve().parents[3]
    runner = ExperimentRunner(
        config,
        project_root=project_root,
        output_dir=args.output_dir,
        headless=not bool(args.viewer),
    )
    result = runner.run()
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print("artifacts:", result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
