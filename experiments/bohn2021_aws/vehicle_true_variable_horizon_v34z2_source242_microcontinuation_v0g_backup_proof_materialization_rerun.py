#!/usr/bin/env python3
"""S-TC2H6 backup-proof materialization wrapper for source242 microcontinuation.

S-TC2H5/v0f was a valid zero-resource engineering/dependency failure: it
accepted the actual S-TC2G completed-marker schema but stopped before any solver
or plant resources because no local verified-backup proof newer than the active
v0f repair package was visible. The next supervisor/user context supplied a
verified backup at 2026-09-30T17:44:59Z, and this cycle materialized that fact in
`BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20260930T174459Z.json`.

This v0g wrapper is still an operational repair only. It adds the S-TC2H5
zero-resource failure and the locally materialized backup-proof record to the
pre-resource preservation/backup gate, requires a fresh verified external backup
after this v0g source and active solo plan before solver or plant resources, and
then delegates to the same source242 microcontinuation. It does not change the
source242 context, goal source, horizon grid, terminal mode, objective, costs,
training/refit state, validation64 access, sealed/final-test access, or resource
caps.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0f_prereq_schema_repair as v0f

base = v0f.base
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0g_backup_proof_materialization_rerun"
TASK_ID = "S-TC2H6-source242-backup-proof-materialization-rerun-v0"

V0F_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0f_prereq_schema_repair_20260930T174529Z/failed.json"
V0F_BACKUP_DEPENDENCY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0f_prereq_schema_repair_20260930T174529Z/backup_dependency_failure.json"
MATERIALIZED_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20260930T174459Z.json"

_previous_record_outcome = base.execution_contract.record_outcome


def _mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def _parse_or_mtime(path: Path, obj: Optional[Mapping[str, Any]] = None) -> dt.datetime:
    parsed = None
    if obj is not None:
        parsed = base.parse_time(obj.get("created_utc") or obj.get("time") or obj.get("timestamp"))
    return parsed if parsed is not None else _mtime_utc(path)


def _latest_time(values: Sequence[dt.datetime]) -> dt.datetime:
    if not values:
        raise base.MicroError("no timestamps supplied to v0g backup recency gate")
    out = values[0]
    for value in values[1:]:
        if value > out:
            out = value
    return out


def _require_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise base.MicroError("required prior artifact missing for v0g gate: %s -> %s" % (label, base.rel(path)))
    obj = base.read_json(path)
    if not isinstance(obj, dict):
        raise base.MicroError("required prior artifact is not a JSON object: %s" % base.rel(path))
    return obj


def _ready_paths_from_snapshot() -> List[Path]:
    # Reuse the v0f implementation so the launched immutable snapshot determines
    # the active report/execution-plan/user-authorization files.
    return list(v0f._ready_paths_from_snapshot())


def _commit_gate(candidates: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    return v0f._commit_gate(candidates)


def _validate_s_tc2h5_failure() -> Dict[str, Any]:
    failed = _require_json(V0F_FAILED, "S-TC2H5/v0f zero-resource failure")
    dependency = _require_json(V0F_BACKUP_DEPENDENCY, "S-TC2H5/v0f backup dependency details")
    used = failed.get("budget_actual") or {}
    if failed.get("status") != "failed" or int(used.get("solver_calls", -1)) != 0 or int(used.get("plant_steps", -1)) != 0:
        raise base.MicroError("v0f prior is not the expected preserved zero-solver/zero-plant backup-gate failure")
    if "external backup proof is not verified" not in str(failed.get("error")):
        raise base.MicroError("v0f prior failure reason changed from expected backup-proof dependency failure")
    if dependency.get("S_TC2G_completed_hard_pass") is not True:
        raise base.MicroError("v0f dependency details do not preserve repaired S-TC2G hard_pass schema evidence")
    return {
        "path": base.rel(V0F_FAILED),
        "created_utc": failed.get("created_utc"),
        "budget_actual": used,
        "error": failed.get("error"),
        "dependency_path": base.rel(V0F_BACKUP_DEPENDENCY),
        "dependency_required_min_time": dependency.get("required_min_time"),
        "dependency_schema_used": dependency.get("S_TC2G_hard_pass_schema_used"),
    }


def _validate_materialized_backup_proof() -> Dict[str, Any]:
    proof = _require_json(MATERIALIZED_BACKUP_PROOF, "materialized supervisor-context backup proof")
    if proof.get("status") != "verified" or int(proof.get("remaining_changed_files", -1)) != 0:
        raise base.MicroError("materialized supervisor-context backup proof is not verified/clean")
    if proof.get("commit") != "af16b54f2778671c633a6f02f8b6c6186fed2a59":
        raise base.MicroError("materialized supervisor-context backup proof commit differs from supplied context")
    if proof.get("package_sha256") != "461c91199770a51fbffd7167d9df1070db4f99fc5c0466d2b329d15d7b14518c":
        raise base.MicroError("materialized supervisor-context backup proof package SHA differs from supplied context")
    return {
        "path": base.rel(MATERIALIZED_BACKUP_PROOF),
        "json_time": proof.get("time"),
        "mtime_utc": _mtime_utc(MATERIALIZED_BACKUP_PROOF).isoformat(),
        "commit": proof.get("commit"),
        "package_sha256": proof.get("package_sha256"),
        "source": proof.get("source"),
    }


def verify_pre_resource_backup_and_priors_v0g() -> Dict[str, Any]:
    """Extend v0f's pre-resource gate to include the v0f dependency failure.

    This intentionally returns a false verification record instead of consuming
    resources when the next external-backup proof is absent. Solver and plant
    use remain gated only by ``verified_before_solver_or_plant_steps``.
    """
    prior = v0f.verify_pre_resource_backup_and_priors_v0f()
    v0f_failed = _validate_s_tc2h5_failure()
    materialized = _validate_materialized_backup_proof()

    required_paths = [
        V0F_FAILED,
        V0F_BACKUP_DEPENDENCY,
        MATERIALIZED_BACKUP_PROOF,
        SOURCE,
    ] + _ready_paths_from_snapshot()
    for path in required_paths:
        if not path.exists():
            raise base.MicroError("required v0g path missing before backup gate: " + base.rel(path))

    # Use filesystem mtimes for newly materialized local records and source/plan
    # files so a JSON field copied from supervisor context cannot back-date this
    # local preservation requirement.
    prior_min = base.parse_time(prior.get("required_min_time")) if isinstance(prior, Mapping) else None
    times: List[dt.datetime] = [prior_min] if prior_min is not None else []
    times.extend(_mtime_utc(p) for p in required_paths)
    min_time = _latest_time(times)

    candidates = base.verified_backup_candidates(min_time)
    commit_ok, head, enriched = _commit_gate(candidates)
    status = base.run_git(["status", "--porcelain", "--"] + [base.rel(p) for p in required_paths])
    clean_ok = bool(status.get("ok") and status.get("stdout") == "")
    ok = bool(prior.get("verified_before_solver_or_plant_steps") and enriched and clean_ok and commit_ok)

    out = dict(prior)
    out.update({
        "verified_before_solver_or_plant_steps": ok,
        "backup_verified_after_v0f_failure_and_v0g_source_and_plan_before_solver_or_plant_steps": ok,
        "backup_verified_after_S_TC2H5_zero_resource_failure_and_materialized_proof_before_solver_or_plant_steps": ok,
        "required_min_time_v0g": min_time.isoformat(),
        "backup_candidates_v0g": enriched,
        "current_git_head_v0g": head,
        "required_paths_git_status_porcelain_v0g": status,
        "required_path_hashes_v0g": {base.rel(p): base.sha256(p) for p in required_paths if p.exists()},
        "required_paths_checked_v0g": [base.rel(p) for p in required_paths],
        "S_TC2H5_zero_resource_failure_preserved": True,
        "materialized_supervisor_context_backup_proof_preserved": True,
        "v0f_failed": v0f_failed,
        "materialized_backup_proof": materialized,
        "v0g_source": {"path": base.rel(SOURCE), "sha256": base.sha256(SOURCE), "mtime_utc": _mtime_utc(SOURCE).isoformat()},
    })
    return out


def _augment_evidence_v0g(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    backup_ok = bool(out.get("backup_verified_before_solver_or_plant_steps"))
    out.setdefault("backup_verified_after_v0f_failure_and_v0g_source_and_plan_before_solver_or_plant_steps", backup_ok)
    out.setdefault("backup_verified_after_S_TC2H5_zero_resource_failure_and_materialized_proof_before_solver_or_plant_steps", backup_ok)
    out.setdefault("S_TC2H5_zero_resource_failure_preserved", V0F_FAILED.exists())
    out.setdefault("materialized_supervisor_context_backup_proof_preserved", MATERIALIZED_BACKUP_PROOF.exists())
    return out


def record_outcome_v0g(root: Any, outcome: str, used: Mapping[str, int], evidence: Any, engineering_error: Optional[str] = None) -> Any:
    if engineering_error == "backup_dependency":
        engineering_error = "dependency"
    return _previous_record_outcome(root, outcome, used, _augment_evidence_v0g(evidence), engineering_error=engineering_error)


# Apply only launch/prerequisite/receipt compatibility patches. The nested action
# payload repair and S-TC2G schema repair are inherited through v0f.
v0f.NAME = NAME
base.NAME = NAME
base.TASK_ID = TASK_ID
base.__dict__["__file__"] = str(SOURCE)
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0g
base.execution_contract.record_outcome = record_outcome_v0g

if __name__ == "__main__":
    raise SystemExit(base.main())
