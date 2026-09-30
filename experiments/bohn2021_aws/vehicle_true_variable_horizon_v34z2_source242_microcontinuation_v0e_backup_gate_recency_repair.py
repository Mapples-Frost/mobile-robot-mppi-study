#!/usr/bin/env python3
"""S-TC2H4 backup-gate recency wrapper for source242 microcontinuation.

This wrapper is a strict operational repair after the S-TC2H3 v0d run failed
before any solver/plant resource because the imported v0c pre-resource backup
check could not see a local verified-backup proof after the v0b action-payload
failure.  It keeps the scientific intervention unchanged: opened-development
source242 branch state, goal endpoint[61], fixed H=[12, 15, 35], V15_shared
terminal values, ordinary warm-start, no training/refit, no validation64, and no
sealed/final test.

The only changes relative to v0d are launch/receipt safeguards:
  * bind the delegated base runner to this task id;
  * inherit the v0c nested 2x1 numeric action payload;
  * replace the brittle inherited backup check with an explicit recency gate that
    requires a locally materialized verified external-backup proof after the v0b
    completed artifact, the v0c zero-resource failure, the v0d zero-resource
    failure, this v0e source, and the active solo plan/report files;
  * add stricter receipt evidence requiring actual plant transitions and absence
    of env/controller exceptions, so a three-arm solver-only failure cannot pass
    as a completed microcontinuation.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0d_taskid_evidence_repair as v0d

base = v0d.base
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0e_backup_gate_recency_repair"
TASK_ID = "S-TC2H4-source242-backup-gate-recency-repair-v0"

V0B_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0b_action_payload_repair_20260930T171033Z/completed.json"
V0C_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0c_nested_payload_repair_20260930T172125Z/failed.json"
V0D_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0d_taskid_evidence_repair_20260930T172745Z/failed.json"

_previous_record_outcome = base.execution_contract.record_outcome


def _latest_time(values: Sequence[dt.datetime]) -> dt.datetime:
    if not values:
        raise base.MicroError("no timestamps supplied to backup recency gate")
    out = values[0]
    for value in values[1:]:
        if value > out:
            out = value
    return out


def _artifact_time(path: Path, obj: Optional[Mapping[str, Any]] = None) -> dt.datetime:
    parsed = None
    if obj is not None:
        parsed = base.parse_time(obj.get("created_utc") or obj.get("time") or obj.get("timestamp"))
    if parsed is not None:
        return parsed
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def _require_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise base.MicroError("required prior artifact missing for v0e backup gate: %s -> %s" % (label, base.rel(path)))
    obj = base.read_json(path)
    if not isinstance(obj, dict):
        raise base.MicroError("required prior artifact is not a JSON object: %s" % base.rel(path))
    return obj


def _ready_paths_from_snapshot() -> List[Path]:
    paths: List[Path] = []
    try:
        snapshot = base.execution_contract.runtime_snapshot(ROOT)
    except Exception:
        snapshot = None
    if isinstance(snapshot, Mapping):
        ready = snapshot.get("ready") or {}
        if isinstance(ready, Mapping):
            for key in ("report", "execution_plan", "user_authorization"):
                value = ready.get(key)
                if isinstance(value, str) and value:
                    paths.append(ROOT / value)
    default_ready = ROOT / "docs/bohn2021_takeover/solo_gpt55/PLAN_READY.json"
    if default_ready.exists():
        paths.append(default_ready)
    latest = ROOT / "docs/bohn2021_takeover/solo_gpt55/LATEST.md"
    if latest.exists():
        paths.append(latest)
    # Preserve order while removing duplicates.
    out: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if str(resolved) not in seen:
            out.append(path)
            seen.add(str(resolved))
    return out


def _commit_gate(candidates: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    head = base.run_git(["rev-parse", "HEAD"])
    ok = False
    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        c = dict(item)
        commit = c.get("commit")
        if commit and head.get("ok"):
            anc = base.run_git(["merge-base", "--is-ancestor", str(commit), "HEAD"])
            current_or_ancestor = bool(anc.get("ok") or head.get("stdout") == commit)
            c["commit_is_current_or_ancestor"] = current_or_ancestor
            ok = ok or current_or_ancestor
        else:
            c["commit_is_current_or_ancestor"] = False
        enriched.append(c)
    return ok, head, enriched


def verify_pre_resource_backup_and_priors_v0e() -> Dict[str, Any]:
    """Return explicit pre-resource backup/prior evidence.

    The returned top-level key `verified_before_solver_or_plant_steps` is what
    the delegated base runner uses as the go/no-go switch before any solver call
    or plant step.
    """
    tc2_done = _require_json(base.TC2_DONE, "T-C2 completed")
    stc2g_done = _require_json(base.STC2G_DONE, "S-TC2G completed")
    v0b_completed = _require_json(V0B_COMPLETED, "S-TC2H v0b completed")
    v0c_failed = _require_json(V0C_FAILED, "S-TC2H2 v0c failed")
    v0d_failed = _require_json(V0D_FAILED, "S-TC2H3 v0d failed")

    for path in [base.TC2_CSV, base.STC2G_RAW, base.STC2G_CSV, base.STC2G_SOURCE, base.GATE_SOURCE, base.BACKUP_REQUEST, SOURCE]:
        if not path.exists():
            raise base.MicroError("required prior/source path missing for v0e backup gate: " + base.rel(path))
    if base.sha256(base.GATE_SOURCE) != base.EXPECTED_GATE_SHA256:
        raise base.MicroError("wrapped v34z2 gate source hash changed before v0e microcontinuation")
    if stc2g_done.get("hard_pass") is not True:
        raise base.MicroError("S-TC2G prerequisite did not record hard_pass true")
    if v0b_completed.get("status") != "complete" or (v0b_completed.get("budget_actual") or {}).get("solver_calls") != 3:
        raise base.MicroError("v0b prior is not the expected preserved three-solver completed artifact")
    if v0c_failed.get("status") != "failed" or (v0c_failed.get("budget_actual") or {}).get("solver_calls") != 0:
        raise base.MicroError("v0c prior is not the expected preserved zero-solver task-id failure")
    if "unexpected task_id in structured snapshot" not in str(v0c_failed.get("error")):
        raise base.MicroError("v0c prior failure reason changed from expected task-id mismatch")
    if v0d_failed.get("status") != "failed" or (v0d_failed.get("budget_actual") or {}).get("solver_calls") != 0:
        raise base.MicroError("v0d prior is not the expected preserved zero-solver backup-gate failure")
    if "no verified external backup after v0b" not in str(v0d_failed.get("error")):
        raise base.MicroError("v0d prior failure reason changed from expected local backup proof absence")

    ready_paths = _ready_paths_from_snapshot()
    required_paths = [
        base.TC2_DONE,
        base.TC2_CSV,
        base.STC2G_DONE,
        base.STC2G_RAW,
        base.STC2G_CSV,
        base.STC2G_SOURCE,
        base.GATE_SOURCE,
        base.BACKUP_REQUEST,
        V0B_COMPLETED,
        V0C_FAILED,
        V0D_FAILED,
        SOURCE,
    ] + ready_paths
    times = [_artifact_time(p) for p in required_paths if p.exists()]
    min_time = _latest_time(times)
    candidates = base.verified_backup_candidates(min_time)
    commit_ok, head, enriched = _commit_gate(candidates)
    status = base.run_git(["status", "--porcelain", "--"] + [base.rel(p) for p in required_paths])
    clean_ok = bool(status.get("ok") and status.get("stdout") == "")
    ok = bool(enriched and clean_ok and commit_ok)
    return {
        "verified_before_solver_or_plant_steps": ok,
        "backup_verified_after_v0b_before_solver_or_plant_steps": ok,
        "backup_verified_after_v0c_failure_and_v0d_source_before_solver_or_plant_steps": ok,
        "backup_verified_after_v0d_failure_and_v0e_source_and_plan_before_solver_or_plant_steps": ok,
        "required_min_time": min_time.isoformat(),
        "backup_candidates": enriched,
        "current_git_head": head,
        "required_paths_git_status_porcelain": status,
        "required_path_hashes": {base.rel(p): base.sha256(p) for p in required_paths if p.exists()},
        "required_paths_checked": [base.rel(p) for p in required_paths],
        "T_C2_status": tc2_done.get("status"),
        "S_TC2G_completed_hard_pass": stc2g_done.get("hard_pass"),
        "S_TC2G_budget_actual": stc2g_done.get("budget_actual"),
        "v0b_completed": {"path": base.rel(V0B_COMPLETED), "created_utc": v0b_completed.get("created_utc"), "budget_actual": v0b_completed.get("budget_actual")},
        "v0c_failed": {"path": base.rel(V0C_FAILED), "created_utc": v0c_failed.get("created_utc"), "budget_actual": v0c_failed.get("budget_actual"), "error": v0c_failed.get("error")},
        "v0d_failed": {"path": base.rel(V0D_FAILED), "created_utc": v0d_failed.get("created_utc"), "budget_actual": v0d_failed.get("budget_actual"), "error": v0d_failed.get("error")},
        "v0e_source": {"path": base.rel(SOURCE), "sha256": base.sha256(SOURCE), "mtime_utc": _artifact_time(SOURCE).isoformat()},
    }


def _latest_raw_for_current_name() -> Optional[Path]:
    diag_root = ROOT / "research_artifacts/aws_diagnostics"
    if not diag_root.exists():
        return None
    candidates = []
    for path in diag_root.glob(NAME + "_*/raw.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            pass
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _strict_microcontinuation_evidence() -> Dict[str, Any]:
    raw_path = _latest_raw_for_current_name()
    if raw_path is None or not raw_path.exists():
        return {
            "source242_microcontinuation_produced_plant_transitions": False,
            "all_three_horizon_arms_advanced_at_least_one_plant_step": False,
            "no_env_or_controller_exceptions_in_attempted_arms": False,
            "strict_microcontinuation_evidence_source": None,
        }
    try:
        raw = base.read_json(raw_path)
    except Exception as exc:
        return {
            "source242_microcontinuation_produced_plant_transitions": False,
            "all_three_horizon_arms_advanced_at_least_one_plant_step": False,
            "no_env_or_controller_exceptions_in_attempted_arms": False,
            "strict_microcontinuation_evidence_source": base.rel(raw_path),
            "strict_microcontinuation_evidence_error": "%s: %s" % (type(exc).__name__, exc),
        }
    arms = raw.get("arms") or []
    if not isinstance(arms, list):
        arms = []
    plant_total = int((raw.get("budget_actual") or {}).get("plant_steps", 0) or 0)
    stop_reasons = [str(a.get("stopped_reason")) for a in arms if isinstance(a, Mapping)]
    plant_by_horizon = {}
    for a in arms:
        if isinstance(a, Mapping):
            plant_by_horizon[str(a.get("horizon"))] = int(a.get("plant_steps_observed", 0) or 0)
    exception_free = bool(arms) and all(reason not in ("env_step_exception", "controller_get_action_exception") for reason in stop_reasons)
    all_three_advanced = bool(len(arms) == 3 and all(int(a.get("plant_steps_observed", 0) or 0) > 0 for a in arms if isinstance(a, Mapping)))
    if len(arms) != 3:
        all_three_advanced = False
    return {
        "source242_microcontinuation_produced_plant_transitions": plant_total > 0,
        "all_three_horizon_arms_advanced_at_least_one_plant_step": all_three_advanced,
        "no_env_or_controller_exceptions_in_attempted_arms": exception_free,
        "strict_microcontinuation_evidence_source": base.rel(raw_path),
        "observed_total_plant_steps": plant_total,
        "observed_arm_stop_reasons": stop_reasons,
        "observed_plant_steps_by_horizon": plant_by_horizon,
    }


def _augment_evidence_v0e(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    backup_ok = bool(out.get("backup_verified_before_solver_or_plant_steps"))
    out.setdefault("backup_verified_after_v0b_before_solver_or_plant_steps", backup_ok)
    out.setdefault("backup_verified_after_v0c_failure_and_v0d_source_before_solver_or_plant_steps", backup_ok)
    out.setdefault("backup_verified_after_v0d_failure_and_v0e_source_and_plan_before_solver_or_plant_steps", backup_ok)
    if "prior_T_C2_and_S_TC2G_failures_preserved" in out:
        out.setdefault("prior_T_C2_and_S_TC2H_failures_preserved", bool(out.get("prior_T_C2_and_S_TC2G_failures_preserved")))
    out.setdefault("task_id_binding_repaired", True)
    out.setdefault("receipt_evidence_key_aliases_added", True)
    out.setdefault("backup_gate_recency_repaired", backup_ok)
    out.setdefault("nested_2x1_action_payload_repair_inherited", True)
    out.update(_strict_microcontinuation_evidence())
    return out


def record_outcome_v0e(root: Any, outcome: str, used: Mapping[str, int], evidence: Any, engineering_error: Optional[str] = None) -> Any:
    if engineering_error == "backup_dependency":
        engineering_error = "dependency"
    return _previous_record_outcome(root, outcome, used, _augment_evidence_v0e(evidence), engineering_error=engineering_error)


# Apply only launch/receipt compatibility and stricter gate patches.  The nested
# action-payload repair is inherited through the v0d -> v0c import chain.
base.NAME = NAME
base.TASK_ID = TASK_ID
base.__dict__["__file__"] = str(SOURCE)
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0e
base.execution_contract.record_outcome = record_outcome_v0e

if __name__ == "__main__":
    raise SystemExit(base.main())
