#!/usr/bin/env python3
"""Versioned one-variable repair for vehicle V1 controlled-continuation diagnostic.

The frozen scientific protocol remains
research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.*.

Why this wrapper exists
-----------------------
The first v0 runner attempt (20260928T120152_908d20a6) failed before any
continuation rollout with:
    AttributeError: vehicle_fixed_h_opportunity_probe_v1_runner has no attribute
    load_terminal_grid
The terminal-grid helper is intentionally provided by the V0 helper module that
V1 imports as ``vehicle_fixed_h_opportunity_probe_v1_runner.base``.  This wrapper
keeps the frozen protocol and all rollout logic unchanged, patches that helper
lookup only, and redirects outputs to a fresh v0b directory so the failed v0
evidence is preserved.

No validation64 bank or sealed test access is introduced by this wrapper.  A
verified external backup proof after this repair source is still required before
running it.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_v1_controlled_continuation_diagnostic_v0_runner as v0  # noqa:E402

# Preserve the failed v0 output directory; write repaired outputs separately.
v0.OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928"
v0.STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928.md"
v0.MARKER = "vehicle-v1-controlled-continuation-diagnostic-v0b-one-variable-terminal-helper-repair-20260928"

# One-variable functional repair: expose the terminal-grid loader from the V0
# helper module through the V1 runner namespace expected by the v0 continuation
# runner.  Do not alter protocol targets, budgets, branch horizons, selection
# rules, environment/controller mechanics, or analysis thresholds.
if not hasattr(v0.v1probe, "load_terminal_grid"):
    v0.v1probe.load_terminal_grid = v0.v1probe.base.load_terminal_grid

_ORIGINAL_V0_SOURCE = Path(v0.__file__).resolve()
_THIS_SOURCE = Path(__file__).resolve()
_ORIGINAL_SOURCE_HASHES = v0.source_hashes


def source_hashes_with_repair_lineage():
    hashes = _ORIGINAL_SOURCE_HASHES()
    hashes[v0.rel(_ORIGINAL_V0_SOURCE)] = v0.sha256(_ORIGINAL_V0_SOURCE)
    hashes[v0.rel(_THIS_SOURCE)] = v0.sha256(_THIS_SOURCE)
    return hashes


v0.source_hashes = source_hashes_with_repair_lineage
# Make Path(__file__) references inside imported v0 helpers point to this
# versioned repair runner for raw/completed artifact hashes and backup requests.
v0.__file__ = str(_THIS_SOURCE)


def main(argv: Sequence[str] | None = None) -> int:
    return v0.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        v0.OUT_DIR.mkdir(parents=True, exist_ok=True)
        v0.write_json(v0.OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "repair_lineage": {
                "failed_v0_runner": v0.rel(_ORIGINAL_V0_SOURCE),
                "v0b_runner": v0.rel(_THIS_SOURCE),
                "one_variable_change": "alias v0.v1probe.load_terminal_grid to v0.v1probe.base.load_terminal_grid before invoking unchanged v0 main; redirect outputs to v0b directory",
                "failed_experiment_id": "20260928T120152_908d20a6"
            },
            "next_recovery_hint": "Preserve partial directory, audit failure, and do not broaden changes without a new hypothesis. Run only after a verified external backup covering this v0b source and the v0 failure artifacts."
        })
        raise
