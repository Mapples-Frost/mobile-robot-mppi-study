#!/usr/bin/env python3
"""S-TC2H3 task-id/evidence-key wrapper for source242 microcontinuation.

The S-TC2H2 v0c wrapper preserved the intended nested 2x1 plant action
payload but failed before any solver/plant resource use because the delegated
v0 runner still required the old structured task id
``S-TC2H-source242-fixedH-microcontinuation-v0``.  This wrapper is a strictly
operational launch-contract repair: it keeps the v0c nested payload adapter and
pre-resource backup checks, binds the delegated runner to the new structured
task id, adds receipt-evidence aliases so scheduler gate names match the new
plan, and requires a verified external backup after both the v0c zero-resource
failure and this wrapper source before solver/plant resources are used.  It does
not change the source242 context, goal source, horizon grid, terminal mode,
objective, initialization, training/refit, validation64 access, sealed/final-test
access, or resource caps.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Mapping, Optional

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0c_nested_payload_repair as v0c

base = v0c.base
SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0d_taskid_evidence_repair"
TASK_ID = "S-TC2H3-source242-taskid-evidence-key-repair-v0"
V0C_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0c_nested_payload_repair_20260930T172125Z/failed.json"

_original_verify = base.verify_pre_resource_backup_and_priors
_original_record_outcome = base.execution_contract.record_outcome


def _latest_time(*values: dt.datetime) -> dt.datetime:
    out = values[0]
    for value in values[1:]:
        if value > out:
            out = value
    return out


def verify_pre_resource_backup_and_priors_v0d() -> dict:
    out = _original_verify()
    if not V0C_FAILED.exists():
        raise base.MicroError("previous v0c task-id failure marker missing before v0d repair")
    prev = base.read_json(V0C_FAILED)
    if prev.get("status") != "failed" or (prev.get("budget_actual") or {}).get("solver_calls") != 0:
        raise base.MicroError("previous v0c evidence marker is not the expected preserved zero-solver failed-result receipt")
    if "unexpected task_id in structured snapshot" not in str(prev.get("error")):
        raise base.MicroError("previous v0c failure was not the expected task-id binding mismatch")
    prev_time = base.parse_time(prev.get("created_utc")) or dt.datetime.fromtimestamp(V0C_FAILED.stat().st_mtime, dt.timezone.utc)
    source_time = dt.datetime.fromtimestamp(SOURCE.stat().st_mtime, dt.timezone.utc)
    min_time = _latest_time(prev_time, source_time)
    candidates = base.verified_backup_candidates(min_time)
    if not candidates:
        raise base.MicroError("no verified external backup after v0c failure and v0d source before v0d solver/plant resources")
    out["post_v0c_failure_and_v0d_source_backup_verified_before_solver_or_plant_steps"] = True
    out["post_v0c_failure_and_v0d_source_backup_candidates"] = candidates
    out["previous_v0c_failed"] = {
        "path": base.rel(V0C_FAILED),
        "sha256": base.sha256(V0C_FAILED),
        "created_utc": prev.get("created_utc"),
        "budget_actual": prev.get("budget_actual"),
        "error": prev.get("error"),
    }
    out["v0d_source"] = {
        "path": base.rel(SOURCE),
        "sha256": base.sha256(SOURCE),
        "mtime_utc": source_time.isoformat(),
    }
    return out


def _augment_evidence(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    if "backup_verified_before_solver_or_plant_steps" in out:
        backup_ok = bool(out.get("backup_verified_before_solver_or_plant_steps"))
        out.setdefault("backup_verified_after_v0b_before_solver_or_plant_steps", backup_ok)
        out.setdefault("backup_verified_after_v0c_failure_and_v0d_source_before_solver_or_plant_steps", backup_ok)
    if "prior_T_C2_and_S_TC2G_failures_preserved" in out:
        out.setdefault(
            "prior_T_C2_and_S_TC2H_failures_preserved",
            bool(out.get("prior_T_C2_and_S_TC2G_failures_preserved")),
        )
    out.setdefault("task_id_binding_repaired", True)
    out.setdefault("receipt_evidence_key_aliases_added", True)
    return out


def record_outcome_v0d(root: Any, outcome: str, used: Mapping[str, int], evidence: Any, engineering_error: Optional[str] = None) -> Any:
    # The historical base helper used the non-contract value "backup_dependency".
    # If that zero-use dependency path occurs again, report it under the
    # contract's allowed engineering-error vocabulary without altering counters.
    if engineering_error == "backup_dependency":
        engineering_error = "dependency"
    return _original_record_outcome(root, outcome, used, _augment_evidence(evidence), engineering_error=engineering_error)


# Apply only launch/receipt compatibility patches.  The nested action-payload
# repair and backup-after-v0b pre-resource checks are inherited from v0c.
base.NAME = NAME
base.TASK_ID = TASK_ID
base.__dict__["__file__"] = str(SOURCE)
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0d
base.execution_contract.record_outcome = record_outcome_v0d

if __name__ == "__main__":
    raise SystemExit(base.main())
