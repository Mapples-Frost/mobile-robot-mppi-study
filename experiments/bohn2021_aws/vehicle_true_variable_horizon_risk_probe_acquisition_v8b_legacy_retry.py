#!/usr/bin/env python3
"""Legacy-interpreter retry wrapper for v8 targeted risk-probe acquisition.

This is a versioned amendment after v8a failed before any simulation because it
was accidentally invoked under the modern Python environment without TF1.  The
scientific protocol is unchanged: development-only, source-independent,
true-H10/H15 paired risk-probe acquisition, validation64 closed and sealed test
closed.  Only the output NAME/STAMP/paths and provenance note are changed so the
failed v8a evidence is preserved rather than overwritten.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry"
STAMP = "20260929T1312Z"

v8.NAME = NAME
v8.STAMP = STAMP
v8.SOURCE = Path(__file__).resolve()
v8.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
v8.STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
v8.CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1312Z_after_risk_probe_acquisition_v8b_legacy_retry.md"
v8.PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
v8.MARKER = f"vehicle-true-variable-H-risk-probe-acquisition-v8b-legacy-retry-{STAMP}"

_ORIGINAL_BUILD_PROTOCOL = v8.build_protocol


def _build_protocol_with_amendment(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str]) -> Mapping[str, Any]:
    protocol = _ORIGINAL_BUILD_PROTOCOL(created, selected_cases, case_diag, input_hashes)
    protocol["amendment"] = {
        "amendment_id": f"{NAME}_legacy_retry_after_v8a_preflight_failure",
        "created_utc": created.isoformat(),
        "reason": "v8a was invoked with the modern interpreter and failed legacy runtime preflight before any simulation/training/refit/validation/test; preserve v8a evidence and retry unchanged scientific protocol under legacy interpreter in a fresh output directory.",
        "failed_prior_attempt": {
            "script": "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py",
            "run_registry": "research_artifacts/aws_runs/20260929T130819_cc2e807a/registry.json",
            "failed_artifact": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8_20260929T1258Z/failed.json",
            "scientific_effect": "none; no simulation episodes/control steps/training/refit/validation/test",
        },
        "unchanged_scientific_protocol_from_v8a": True,
        "changed_variables": ["NAME", "STAMP", "RUN_DIR", "STATE", "CONTINUE_STATE", "PROTOCOL", "MARKER", "SOURCE for provenance"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    v8.write_json(v8.PROTOCOL, protocol)
    return protocol


v8.build_protocol = _build_protocol_with_amendment


def _write_failure_continue(rc: int) -> None:
    created = dt.datetime.now(dt.timezone.utc)
    v8.CONTINUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    failure_artifact = v8.RUN_DIR / "failed.json"
    payload = {
        "utc": created.isoformat(),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "return_code": rc,
        "failure_artifact": failure_artifact.relative_to(ROOT).as_posix() if failure_artifact.exists() else None,
        "scientific_effect": "unknown until failed.json inspected; wrapper writes this only on nonzero return",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_action": "Inspect failed.json; if no simulations occurred, fix the bounded execution issue after backup; if partial simulations occurred, preserve and audit before any rerun.",
    }
    v8.CONTINUE_STATE.write_text("# Continue state after v8b legacy retry failure\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    rc = v8.main()
    if rc != 0:
        _write_failure_continue(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
