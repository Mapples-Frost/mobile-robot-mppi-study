#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.learning.trainer import train_residual


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/research/icode_dynamic5.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    result = train_residual(
        load_yaml(args.config), args.dataset_dir, args.output_dir,
        device=args.device, epochs=args.epochs,
    )
    print(json.dumps({
        "best_validation_multistep_rmse": result["best_validation_multistep_rmse"],
        "epochs": len(result["history"]),
        "checkpoint": str(Path(args.output_dir).resolve() / "best.pt"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
