#!/usr/bin/env python3
"""Receipt-key repair for the backup recheck after the current gated audit.

The approved backup recheck ran as a zero-resource metadata-only task and
correctly found that external backup recoverability was still blocked by a
current TimeoutError.  Its structured receipt was valid, but the evidence keys
used by the script did not match the lead plan's exact pass-condition names, so
structured task-gate acceptance failed even though the intended backup-gate
measurement was complete.

This repair reads only the already-created backup-recheck artifacts and emits a
new structured receipt with exact pass-condition keys.  It does not run backup,
open validation banks, execute controllers/solvers/plants, train/refit, create
validation episodes, or access sealed/final tests.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import execution_contract  # type: ignore

TASK_ID = "S-BACKUP-RECHECK-AFTER-CURRENT-GATED-RECEIPT-REPAIR-v0b-receipt-key-repair"
NAME = "backup_recheck_receipt_key_repair_solo_v0"
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
SCRIPT = ROOT / "experiments/bohn2021_aws/backup_recheck_receipt_key_repair_solo_v0.py"
PRIOR_DIAG = ROOT / "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_current_status_v0c_20261001T003237Z"
PRIOR_COMPLETED = PRIOR_DIAG / "completed.json"
PRIOR_SUMMARY = PRIOR_DIAG / "summary.md"
PRIOR_RAW = PRIOR_DIAG / "raw.json"
PRIOR_STATE = ROOT / "research_artifacts/aws_state/continue_state_20261001T003237Z_after_s_tc2h8_backup_current_status_v0c.md"
PRIOR_RUN = ROOT / "research_artifacts/aws_runs/20261001T003237_16fae53f"
PRIOR_RECEIPT = PRIOR_RUN / "outcome_receipt.json"
PRIOR_REGISTRY = PRIOR_RUN / "registry.json"
LATEST_GATED_AUDIT_BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_CURRENT_GATED_AUDIT_RECEIPT_REPAIR_SOLO_V0_20261001T002636Z.json"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
DOC_MARKER = "backup-recheck-receipt-key-repair-solo-v0-20261001"

EXPECTED = {
    "zero_solver_plant_training_validation_test_resources": True,
    "validation_bank_content_opened": False,
    "sealed_or_final_test_accessed": False,
    "backup_status_artifacts_inspected": True,
    "latest_receipt_repair_backup_request_seen": True,
    "backup_recoverability_status_recorded": True,
    "nonzero_work_gate_decision_recorded": True,
    "outputs_persisted": True,
    "prior_receipt_key_mismatch_recorded": True,
    "backup_still_blocks_nonzero_work": True,
}


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    token = "<!-- " + marker + " -->"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if token in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + block.strip() + "\n", encoding="utf-8")


def expected_mismatches(evidence: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, expected in EXPECTED.items():
        actual = evidence.get(key, "<missing>")
        if actual != expected or type(actual) is not type(expected):
            out[key] = {"expected": expected, "actual": actual}
    return out


def build_evidence(prior_completed: Mapping[str, Any], prior_raw: Mapping[str, Any], prior_receipt: Mapping[str, Any]) -> Dict[str, Any]:
    resources = prior_receipt.get("resources") or prior_completed.get("resources") or {}
    decision = prior_raw.get("backup_gate_decision") or prior_completed.get("backup_gate_decision") or {}
    access = prior_raw.get("access_flags") or {}
    prior_evidence = prior_receipt.get("evidence") or {}
    expected_prior_keys = {
        "zero_solver_plant_training_validation_test_resources",
        "validation_bank_content_opened",
        "sealed_or_final_test_accessed",
        "backup_status_artifacts_inspected",
        "latest_receipt_repair_backup_request_seen",
        "backup_recoverability_status_recorded",
        "nonzero_work_gate_decision_recorded",
        "outputs_persisted",
    }
    prior_key_mismatch = not expected_prior_keys.issubset(set(prior_evidence.keys()))
    summary_text = read_text(PRIOR_SUMMARY) if PRIOR_SUMMARY.exists() else ""
    latest_request = read_json(LATEST_GATED_AUDIT_BACKUP_REQUEST) if LATEST_GATED_AUDIT_BACKUP_REQUEST.exists() else None
    backup_status_recorded = isinstance(decision.get("adequate_backup_for_s_tc2h8"), bool) and isinstance(decision.get("status"), str)
    blocked = decision.get("adequate_backup_for_s_tc2h8") is False and decision.get("status") == "failed"
    evidence = {
        "zero_solver_plant_training_validation_test_resources": resources == ZERO,
        "validation_bank_content_opened": bool(access.get("validation64_bank_opened") or prior_completed.get("validation64_bank_opened")),
        "sealed_or_final_test_accessed": bool(access.get("sealed_test_accessed") or prior_completed.get("sealed_test_accessed")),
        "backup_status_artifacts_inspected": all(p.exists() for p in (PRIOR_COMPLETED, PRIOR_SUMMARY, PRIOR_RAW, PRIOR_RECEIPT, PRIOR_REGISTRY, PRIOR_STATE)),
        "latest_receipt_repair_backup_request_seen": latest_request is not None and LATEST_GATED_AUDIT_BACKUP_REQUEST.exists(),
        "backup_recoverability_status_recorded": backup_status_recorded,
        "nonzero_work_gate_decision_recorded": ("Nonzero S-TC2H8 resources remain blocked" in summary_text) or blocked,
        "outputs_persisted": all(p.exists() for p in (PRIOR_COMPLETED, PRIOR_SUMMARY, PRIOR_RAW, PRIOR_RECEIPT, PRIOR_REGISTRY, PRIOR_STATE)),
        "prior_receipt_key_mismatch_recorded": prior_key_mismatch,
        "backup_still_blocks_nonzero_work": blocked,
    }
    return evidence


def make_summary(raw: Mapping[str, Any]) -> str:
    decision = raw["backup_gate_decision"]
    return "\n".join([
        "# Backup recheck receipt-key repair (solo v0)",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "## Operational result",
        "",
        "The prior backup recheck was a valid zero-resource metadata-only measurement, but its structured receipt used non-matching evidence key names. This repair emits exact pass-condition keys for the successor receipt. It does not change the backup finding.",
        "",
        "## Backup gate status preserved",
        "",
        f"- Adequate verified external backup: `{decision.get('adequate_backup_for_s_tc2h8')}`.",
        f"- Status: `{decision.get('status')}`.",
        f"- Error type: `{decision.get('error_type')}`.",
        f"- Failure class: `{decision.get('current_failure_class')}`.",
        f"- Message: `{decision.get('message')}`.",
        f"- Reasons: `{decision.get('why_not')}`.",
        "",
        "Nonzero controller/solver/plant/training work remains blocked until a verified backup proof postdating this repair exists. The next useful bounded work is to repair/retry the backup path itself or continue only zero-resource preparation.",
        "",
        "## Evidence mismatches after repair",
        "",
        json.dumps(raw["evidence_mismatches"], indent=2, sort_keys=True),
        "",
        "## Resource and split safety",
        "",
        "Resources are zero for solver calls, plant steps, training steps, validation episodes, and test episodes. No validation64 bank/generator content, sealed test, or final test was accessed.",
    ]) + "\n"


def main() -> int:
    snapshot = execution_contract.runtime_snapshot(ROOT)
    if snapshot is None:
        raise RuntimeError("No structured execution snapshot is available")
    observed_task = (snapshot.get("task") or {}).get("task_id")
    if observed_task != TASK_ID:
        raise RuntimeError("Unexpected structured task_id: %r" % (observed_task,))

    created = now()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / (NAME + "_" + stamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_BACKUP_RECHECK_RECEIPT_KEY_REPAIR_SOLO_V0_" + stamp + ".json")

    required = [PRIOR_COMPLETED, PRIOR_SUMMARY, PRIOR_RAW, PRIOR_RECEIPT, PRIOR_REGISTRY, PRIOR_STATE, LATEST_GATED_AUDIT_BACKUP_REQUEST]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        failure = {
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "missing_inputs": missing,
            "resources": dict(ZERO),
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "no_scientific_outcome": True,
        }
        write_json(raw_path, failure)
        execution_contract.record_outcome(
            ROOT,
            "engineering_failure",
            dict(ZERO),
            {
                "no_scientific_outcome": True,
                "missing_inputs": missing,
                "zero_solver_plant_training_validation_test_resources": True,
                "validation_bank_content_opened": False,
                "sealed_or_final_test_accessed": False,
            },
            engineering_error="missing_prior_backup_recheck_artifact",
        )
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    prior_completed = read_json(PRIOR_COMPLETED)
    prior_raw = read_json(PRIOR_RAW)
    prior_receipt = read_json(PRIOR_RECEIPT)
    prior_registry = read_json(PRIOR_REGISTRY)
    decision = prior_raw.get("backup_gate_decision") or prior_completed.get("backup_gate_decision") or {}
    evidence = build_evidence(prior_completed, prior_raw, prior_receipt)
    bad = expected_mismatches(evidence)
    passed = not bad

    raw = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "authority_mode": "temporary_user_authorized_solo_self_review",
        "analysis_type": "zero_resource_receipt_key_repair_for_backup_recheck",
        "resources": dict(ZERO),
        "script": rel(SCRIPT),
        "script_sha256": sha256(SCRIPT),
        "input_artifacts": {rel(path): sha256(path) for path in required},
        "prior_run": {
            "experiment_id": prior_registry.get("experiment_id"),
            "exit_status": prior_registry.get("exit_status"),
            "runtime_seconds": prior_registry.get("runtime_seconds"),
            "coordination": prior_registry.get("coordination"),
            "prior_receipt_task_id": prior_receipt.get("task_id"),
            "prior_receipt_evidence_keys": sorted((prior_receipt.get("evidence") or {}).keys()),
        },
        "backup_gate_decision": decision,
        "prior_completed_hard_pass": prior_completed.get("hard_pass"),
        "evidence": evidence,
        "expected_evidence": EXPECTED,
        "evidence_mismatches": bad,
        "corrected_pass": passed,
        "split_safety": {
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "new_validation_episodes": 0,
            "controller_solver_plant_executed": False,
            "training_or_refit_executed": False,
        },
    }
    write_json(raw_path, raw)
    write_text(summary_path, make_summary(raw))
    backup_request = {
        "created_utc": created.isoformat(),
        "request": "external_backup_after_backup_recheck_receipt_key_repair_solo_v0",
        "reason": "Preserve the structured receipt-key repair and backup-still-blocked finding before any future nonzero controller/solver/training work.",
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "backup_gate_decision": decision,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(completed_path), rel(PRIOR_RAW), rel(PRIOR_SUMMARY), rel(PRIOR_COMPLETED), rel(PRIOR_RECEIPT)],
    }
    write_json(backup_request_path, backup_request)

    completed = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "passed": passed,
        "raw": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "completed": rel(completed_path),
        "backup_request": rel(backup_request_path),
        "backup_request_sha256": sha256(backup_request_path),
        "resources": dict(ZERO),
        "evidence": evidence,
        "evidence_mismatches": bad,
        "backup_gate_decision": decision,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
    }
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256(completed_path)
    write_json(completed_path, completed)

    log_block = f"""
### Backup recheck receipt-key repair solo v0 ({created.isoformat()})

- Repaired the structured evidence-key mismatch for the prior backup recheck run `20261001T003237_16fae53f`; the prior receipt was valid but did not use the lead plan's exact pass-condition keys.
- The backup finding is unchanged: adequate verified external backup is `{decision.get('adequate_backup_for_s_tc2h8')}`, status is `{decision.get('status')}`, failure class is `{decision.get('current_failure_class')}`, and error type is `{decision.get('error_type')}`.
- Nonzero controller/solver/plant/training work remains blocked until a verified backup proof postdating this repair exists.
- No validation bank/generator, controller, plant, solver, training/refit, new validation episode, sealed test, or final test was accessed. Resources are zero.
- Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(completed_path)}`.
"""
    for log_name in ("RESEARCH_LOG.md", "STATUS.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        append_once(ROOT / log_name, DOC_MARKER + "-" + log_name.lower().replace(".", "-"), log_block)
    append_once(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", DOC_MARKER + "-response-log", log_block)

    if passed:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    else:
        fail_evidence = dict(evidence)
        fail_evidence["no_scientific_outcome"] = True
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), fail_evidence, engineering_error="backup_recheck_receipt_key_mismatch_unresolved")
    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            execution_contract.record_outcome(
                ROOT,
                "engineering_failure",
                dict(ZERO),
                {
                    "no_scientific_outcome": True,
                    "zero_solver_plant_training_validation_test_resources": True,
                    "validation_bank_content_opened": False,
                    "sealed_or_final_test_accessed": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                },
                engineering_error="startup",
            )
        except Exception:
            pass
        raise
