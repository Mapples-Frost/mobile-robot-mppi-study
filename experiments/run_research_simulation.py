#!/usr/bin/env python3
"""Thin source-tree compatibility entry for the unified experiment runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mobile_robot_mppi.cli.simulate import main


if __name__ == "__main__":
    raise SystemExit(main())
