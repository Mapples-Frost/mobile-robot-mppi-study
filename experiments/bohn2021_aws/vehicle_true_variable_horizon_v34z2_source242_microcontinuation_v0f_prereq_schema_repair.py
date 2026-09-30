#!/usr/bin/env python3
"""S-TC2H5 prerequisite-schema repair wrapper for source242 microcontinuation.

S-TC2H4/v0e failed before solver or plant resources because its prerequisite
check looked for ``hard_pass`` at the top level of the preserved S-TC2G completed
marker. The actual S-TC2G marker stores that Boolean at
``headline.hard_pass`` and also records all task pass evidence as true. This
wrapper changes only that prerequisite schema check and extends the pre-resource
backup gate to cover the v0e zero-resource failure plus this v0f source and the
active solo plan/report before any solver or plant resources are consumed.

The scientific intervention is unchanged: opened-development source242 branch
state, goal endpoint[61], fixed H=[12, 15, 35], V15_shared terminal values,
ordinary warm-start, no training/refit, no validation64, and no sealed/final
test. It preserves the inherited task-id, evidence-key, backup-recency and
nested 2x1 action-payload repairs, and still requires actual plant transitions
and no environment/controller exceptions in the execution receipt.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0e_backup_gate_recency_repair as v0e

base = v0e.base
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0f_prereq_schema_repair"
TASK_ID = "S-TC2H5-source242-prereq-schema-repair-v0"

V0B_COMPLETED = v0e.V0B_COMPLETED
V0C_FAILED = v0e.V0C_FAILED
V0D_FAILED = v0e.V0D_FAILED
V0E_SOURCE = v0e.SOURCE
V0E_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0e_backup_gate_recency_repair_20260930T174001Z/failed.json"

# Bypass v0e's strict raw lookup, which is tied to the v0e module-level NAME, but
# keep the older v0d outcome aliasing behavior.
_previous_record_outcome = getattr(v0e, "_previous_record_outcome", base.execution_contract.record_outcome)


def _latest_time(values: Sequence[dt.datetime]) -> dt.datetime:
    if not values:
        raise base.MicroError("no timestamps supplied to v0f backup recency gate")
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
        raise base.MicroError("required prior artifact missing for v0f prerequisite/backup gate: %s -> %s" % (label, base.rel(path)))
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


def _stc2g_hard_pass(done: Mapping[str, Any]) -> Tuple[bool, str]:
    """Accept the actual S-TC2G completed-marker schema without weakening gates."""
    if done.get("hard_pass") is True:
        return True, "top_level_hard_pass"
    headline = done.get("headline") or {}
    if isinstance(headline, Mapping) and headline.get("hard_pass") is True:
        return True, "headline.hard_pass"
    pass_evidence = done.get("pass_evidence") or {}
    required = [
        "h15_initialization_basin_triage_completed",
        "direct_nlp_and_objective_reconstruction_reported",
        "initialization_hashes_and_alpha_grid_reported",
        "no_plant_training_validation_or_test_usage",
        "prior_T_C2_failure_preserved",
        "solver_calls_at_most_three",
        "backup_verified_before_solver_calls_or_zero_usage_dependency_failure",
    ]
    if isinstance(pass_evidence, Mapping) and all(pass_evidence.get(k) is True for k in required):
        return True, "pass_evidence_all_required_true"
    return False, "no_supported_hard_pass_field_or_pass_evidence"


def verify_pre_resource_backup_and_priors_v0f() -> Dict[str, Any]:
    """Verify all prior artifacts and a backup newer than this repair package.

    The returned key ``verified_before_solver_or_plant_steps`` remains the only
    go/no-go switch used by the delegated base runner before any solver call or
    plant step.
    """
    tc2_done = _require_json(base.TC2_DONE, "T-C2 completed")
    stc2g_done = _require_json(base.STC2G_DONE, "S-TC2G completed")
    v0b_completed = _require_json(V0B_COMPLETED, "S-TC2H v0b completed")
    v0c_failed = _require_json(V0C_FAILED, "S-TC2H2 v0c failed")
    v0d_failed = _require_json(V0D_FAILED, "S-TC2H3 v0d failed")
    v0e_failed = _require_json(V0E_FAILED, "S-TC2H4 v0e failed")

    required_existing = [
        base.TC2_CSV,
        base.STC2G_RAW,
        base.STC2G_CSV,
        base.STC2G_SOURCE,
        base.GATE_SOURCE,
        base.BACKUP_REQUEST,
        V0E_SOURCE,
        SOURCE,
    ]
    for path in required_existing:
        if not path.exists():
            raise base.MicroError("required prior/source path missing for v0f backup gate: " + base.rel(path))
    if base.sha256(base.GATE_SOURCE) != base.EXPECTED_GATE_SHA256:
        raise base.MicroError("wrapped v34z2 gate source hash changed before v0f microcontinuation")

    stc2g_ok, stc2g_schema = _stc2g_hard_pass(stc2g_done)
    if not stc2g_ok:
        raise base.MicroError("S-TC2G prerequisite did not record a supported hard_pass/pass_evidence schema")
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
    if v0e_failed.get("status") != "failed" or (v0e_failed.get("budget_actual") or {}).get("solver_calls") != 0:
        raise base.MicroError("v0e prior is not the expected preserved zero-solver prerequisite-schema failure")
    if "S-TC2G prerequisite did not record hard_pass true" not in str(v0e_failed.get("error")):
        raise base.MicroError("v0e prior failure reason changed from expected S-TC2G hard_pass schema mismatch")

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
        V0E_SOURCE,
        V0E_FAILED,
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
        "backup_verified_after_v0e_failure_and_v0f_source_and_plan_before_solver_or_plant_steps": ok,
        "required_min_time": min_time.isoformat(),
        "backup_candidates": enriched,
        "current_git_head": head,
        "required_paths_git_status_porcelain": status,
        "required_path_hashes": {base.rel(p): base.sha256(p) for p in required_paths if p.exists()},
        "required_paths_checked": [base.rel(p) for p in required_paths],
        "T_C2_status": tc2_done.get("status"),
        "S_TC2G_completed_hard_pass": stc2g_ok,
        "S_TC2G_hard_pass_schema_used": stc2g_schema,
        "S_TC2G_budget_actual": stc2g_done.get("budget_actual"),
        "v0b_completed": {"path": base.rel(V0B_COMPLETED), "created_utc": v0b_completed.get("created_utc"), "budget_actual": v0b_completed.get("budget_actual")},
        "v0c_failed": {"path": base.rel(V0C_FAILED), "created_utc": v0c_failed.get("created_utc"), "budget_actual": v0c_failed.get("budget_actual"), "error": v0c_failed.get("error")},
        "v0d_failed": {"path": base.rel(V0D_FAILED), "created_utc": v0d_failed.get("created_utc"), "budget_actual": v0d_failed.get("budget_actual"), "error": v0d_failed.get("error")},
        "v0e_failed": {"path": base.rel(V0E_FAILED), "created_utc": v0e_failed.get("created_utc"), "budget_actual": v0e_failed.get("budget_actual"), "error": v0e_failed.get("error")},
        "v0f_source": {"path": base.rel(SOURCE), "sha256": base.sha256(SOURCE), "mtime_utc": _artifact_time(SOURCE).isoformat()},
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
    plant_by_horizon: Dict[str, int] = {}
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


def _augment_evidence_v0f(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    backup_ok = bool(out.get("backup_verified_before_solver_or_plant_steps"))
    out.setdefault("backup_verified_after_v0b_before_solver_or_plant_steps", backup_ok)
    out.setdefault("backup_verified_after_v0c_failure_and_v0d_source_before_solver_or_plant_steps", backup_ok)
    out.setdefault("backup_verified_after_v0d_failure_and_v0e_source_and_plan_before_solver_or_plant_steps", backup_ok)
    out.setdefault("backup_verified_after_v0e_failure_and_v0f_source_and_plan_before_solver_or_plant_steps", backup_ok)
    if "prior_T_C2_and_S_TC2G_failures_preserved" in out:
        out.setdefault("prior_T_C2_and_S_TC2H_failures_preserved", bool(out.get("prior_T_C2_and_S_TC2G_failures_preserved")))
    out.setdefault("task_id_binding_repaired", True)
    out.setdefault("receipt_evidence_key_aliases_added", True)
    out.setdefault("backup_gate_recency_repaired", backup_ok)
    out.setdefault("nested_2x1_action_payload_repair_inherited", True)
    out.setdefault("stc2g_hard_pass_schema_repaired", True)
    out.update(_strict_microcontinuation_evidence())
    return out


def record_outcome_v0f(root: Any, outcome: str, used: Mapping[str, int], evidence: Any, engineering_error: Optional[str] = None) -> Any:
    if engineering_error == "backup_dependency":
        engineering_error = "dependency"
    return _previous_record_outcome(root, outcome, used, _augment_evidence_v0f(evidence), engineering_error=engineering_error)


# Apply only launch/prerequisite/receipt compatibility patches. The nested action
# payload repair is inherited through the v0e -> v0d -> v0c import chain.
base.NAME = NAME
base.TASK_ID = TASK_ID
base.__dict__["__file__"] = str(SOURCE)
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0f
base.execution_contract.record_outcome = record_outcome_v0f

if __name__ == "__main__":
    raise SystemExit(base.main())
