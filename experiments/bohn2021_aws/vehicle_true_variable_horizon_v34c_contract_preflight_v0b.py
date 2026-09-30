#!/usr/bin/env python3
"""v34c contract preflight wrapper with current Astra report gate.

This is a versioned operational repair for
``vehicle_true_variable_horizon_v34c_contract_preflight_v0.py``.  The underlying
zero-solve preflight implementation was written against the earlier Astra report
constants inherited from the v34 solver-probe module.  The current authoritative
Astra handoff is docs/bohn2021_takeover/astra_reviews/20260930T084456Z.md for
request execution-result:20260930T084409_7dd79362.  This wrapper patches only
those gate constants and output path identifiers, then delegates to the v34c
implementation.  It performs no scientific-method changes and does not alter the
24-call solver-probe design.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34c_contract_preflight_v0 as c  # noqa:E402

CURRENT_REQUEST = "execution-result:20260930T084409_7dd79362"
CURRENT_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T084456Z.md"
CURRENT_SHA = "9888c40d0cb52640ee18cd548e001698236aab2f1414e615f361c04abfa40ba0"


def patch_runtime() -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "vehicle_true_variable_horizon_v34c_contract_preflight_v0b"
    c.NAME = name
    c.STAMP = stamp
    c.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{name}_{stamp}"
    c.ARRAY_DIR = c.RUN_DIR / "arrays"
    c.BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_V0B_{stamp}.json"
    c.STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_v34c_contract_preflight_v0b.md"
    c.MARKER = f"vehicle-v34c-contract-preflight-v0b-{stamp}"

    # Patch the imported v34 gate module to the current authoritative Astra
    # analysis.  No data/split/test gate is weakened: Task1 and v33 completion
    # checks, backup args, validation64=false and sealed-test=false remain intact.
    c.m.ASTRA_REPORT = CURRENT_REPORT
    c.m.CURRENT_ASTRA_REQUEST = CURRENT_REQUEST
    c.m.CURRENT_ASTRA_SHA = CURRENT_SHA


def main() -> int:
    patch_runtime()
    return int(c.run())


if __name__ == "__main__":
    raise SystemExit(main())
