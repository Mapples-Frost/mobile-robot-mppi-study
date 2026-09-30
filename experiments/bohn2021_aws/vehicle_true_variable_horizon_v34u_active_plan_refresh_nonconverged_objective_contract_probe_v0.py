#!/usr/bin/env python3
"""v34u operational retry wrapper for v34t/A13c-3 after active-plan refresh.

v34t failed before any solver call because PLAN_READY was atomically advanced
from the 20260930T120929 report to the 20260930T121504 report for the same
v34s/T-B4 evidence packet. This thin extension preserves the v34t repaired
A13c-3 logic and only binds its plan constants to the current PLAN_READY record
at startup. It does not mutate v34n/v34o/v34p/v34q/v34s/v34t.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0 as v34t  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
EXPECTED_REQUEST = "execution-result:20260930T120837_e51ace56"


def _load_current_plan() -> tuple[Path, str, str]:
    plan = json.loads((ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json").read_text(encoding="utf-8-sig"))
    request = str(plan.get("request_id"))
    if request != EXPECTED_REQUEST:
        raise v34t.ContractError(f"unexpected current PLAN_READY request_id={request!r}; expected the v34s/T-B4 request {EXPECTED_REQUEST!r}")
    report = ROOT / str(plan.get("report"))
    sha = str(plan.get("report_sha256"))
    if not report.exists():
        raise v34t.ContractError(f"current PLAN_READY report is missing: {report}")
    return report, sha, request


def run(argv: Optional[Sequence[str]] = None) -> int:
    report, sha, request = _load_current_plan()
    # Patch v34t module globals before it patches v34n. This is an operational
    # plan-identity repair only; all solver-contract logic remains v34t.
    v34t.__file__ = __file__
    v34t.NAME = NAME
    v34t.STAMP = STAMP
    v34t.OPUS_REPORT = report
    v34t.OPUS_REPORT_SHA = sha
    v34t.OPUS_REQUEST = request
    return v34t.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
