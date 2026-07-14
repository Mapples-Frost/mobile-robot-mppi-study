#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import audit_static_scene


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    reports = []
    failed = False
    for path in args.configs:
        config = load_yaml(path)
        report = audit_static_scene(
            config.get("scene", {}),
            config["experiment"].get("initial_state", (0.0, 0.0))[:2],
            config["task"]["position"],
            config["plant"]["robot"].get("collision_radius", 0.25),
            margin=args.margin,
            resolution=args.resolution,
        )
        report["config"] = str(Path(path).resolve())
        reports.append(report)
        failed = failed or not (
            report["start_free"] and report["goal_free"] and report["path_exists"]
        )
    payload = {"reports": reports, "passed": not failed}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
