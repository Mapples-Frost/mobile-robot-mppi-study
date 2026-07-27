"""Continue a saved complex configuration online for a long viewer preview.

This is visualization-only development work. It reloads the saved resolved
configuration, restarts the episode from its original initial state, and lets
the controller plan until success, collision, viewer closure, or a deliberately
large runaway guard. Its output must not be used as confirmatory evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runaway-guard-steps", type=int, default=5000)
    args = parser.parse_args(argv)

    if args.runaway_guard_steps < 1000:
        parser.error("runaway guard must be at least 1000 steps")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "extended preview output already contains files: %s" % output
        )
    config = load_yaml(args.config.resolve())
    config["experiment"]["max_steps"] = int(args.runaway_guard_steps)
    config["experiment"]["name"] += "__extended_viewer_preview"
    config["experiment"]["development_visualization_only"] = True
    config.setdefault("scope_guards", {})[
        "formal_experiment_started"
    ] = False
    ExperimentRunner(
        config,
        ROOT,
        output_dir=output,
        headless=False,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
