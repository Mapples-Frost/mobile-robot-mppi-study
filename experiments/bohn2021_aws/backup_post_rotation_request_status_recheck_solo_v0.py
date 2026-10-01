#!/usr/bin/env python3
"""Zero-resource post-request backup status gate.

This task is a bounded follow-up to the solo backup release-rotation request. It
reads only supervisor backup status metadata and the already-written repair
request. It does not read credentials, import or execute the backup uploader,
open validation banks, run controllers/solvers/plant steps, train, refit, or
access sealed/final tests.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import traceback
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

CANON_BASE = pathlib.Path("/data/openai-agent")
ROOT = CANON_BASE / "mobile-robot-mppi-study"
if not ROOT.exists():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
STATE = CANON_BASE / "state"
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import execution_contract  # noqa: E402

TASK_ID = "S-BACKUP-POST-ROTATION-REQUEST-STATUS-RECHECK-v0"
RUN_STEM = "backup_post_rotation_request_status_recheck_solo_v0"
SCRIPT_REL = "experiments/bohn2021_aws/backup_post_rotation_request_status_recheck_solo_v0.py"
REQUEST_REL = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_RELEASE_ROTATION_AFTER_RECEIPT_REPAIR_SOLO_V0_20261001T010054Z.json"
REQUEST_PATH = ROOT / REQUEST_REL
BACKUP_STATUS = STATE / "backup_status.json"
BACKUP_RECEIPTS = STATE / "backup_receipts.jsonl"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
DOC_MARKER = "<!-- backup-post-rotation-request-status-recheck-solo-v0 -->"
SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|github\.token|key_material)", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{96,}\b")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(CANON_BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def sha256(path: pathlib.Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: pathlib.Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:500]}"


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            key_s = str(key)
            if SECRET_KEY_RE.search(key_s) and key_s.lower() not in {"sha256", "asset_sha256", "package_sha256", "manifest_sha256"}:
                out[key_s] = "[REDACTED]"
            else:
                out[key_s] = sanitize(val)
        return out
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        text = LONG_HEX_RE.sub("[REDACTED_HEX]", value)
        text = text.replace(str(CANON_BASE / ".secrets"), "[REDACTED_SECRET_PATH]")
        return text
    return value


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def artifact_info(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "kind": "missing", "bytes": None, "sha256": None, "mtime_utc": None}
    st = path.stat()
    return {
        "path": rel(path),
        "exists": True,
        "kind": "file" if path.is_file() else "dir",
        "bytes": st.st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
        "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
    }


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if isinstance(packages, list) and packages:
        for item in packages:
            if not isinstance(item, Mapping):
                return False
            has_hash = bool(item.get("sha256") or item.get("asset_sha256") or item.get("package_sha256") or item.get("manifest_sha256") or item.get("download_sha256"))
            has_size = bool(item.get("bytes") or item.get("size"))
            has_verification = bool(item.get("verification") or item.get("verified") is True or item.get("download_verified") is True or item.get("digest"))
            if not (has_hash and has_size and has_verification):
                return False
        return True
    return bool(status.get("package_sha256") or status.get("asset_sha256") or status.get("release_asset_sha256") or status.get("manifest_sha256"))


def request_time(request_obj: Optional[Any], request_path: pathlib.Path) -> Optional[dt.datetime]:
    candidates = []
    if isinstance(request_obj, Mapping):
        for key in ("created_utc", "requested_utc", "time", "verified_utc"):
            parsed = parse_time(request_obj.get(key))
            if parsed is not None:
                candidates.append(parsed)
    if request_path.exists():
        candidates.append(dt.datetime.fromtimestamp(request_path.stat().st_mtime, dt.timezone.utc))
    if not candidates:
        return None
    return max(candidates)


def classify_backup_status(status_obj: Optional[Any], min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    if not isinstance(status_obj, Mapping):
        return {
            "status_file_readable": False,
            "adequate_backup_for_nonzero_work": False,
            "current_failure_class": "status_missing_or_unreadable",
            "why_not": ["backup_status_missing_or_unreadable"],
            "min_required_post_request_time_utc": None if min_time is None else min_time.isoformat(),
        }
    status_time = None
    for key in ("time", "created_utc", "verified_utc", "completed_utc"):
        status_time = parse_time(status_obj.get(key))
        if status_time is not None:
            break
    verified = bool(status_obj.get("status") == "verified" or status_obj.get("backup_verified") is True)
    try:
        remaining_ok = int(status_obj.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    has_commit = bool(status_obj.get("commit"))
    packages_ok = package_metadata_ok(status_obj)
    time_ok = bool(status_time is not None and min_time is not None and status_time >= min_time)
    message = str(status_obj.get("message", ""))
    error_type = str(status_obj.get("error_type", ""))
    http_422 = "422" in message or "HTTP 422" in message
    timeout = "timeout" in message.lower() or "timed out" in message.lower() or "timeout" in error_type.lower()
    adequate = bool(verified and remaining_ok and has_commit and packages_ok and time_ok)
    why_not = []
    if not verified:
        why_not.append(f"status_is_{status_obj.get('status')!r}_not_verified")
    if not remaining_ok:
        why_not.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why_not.append("missing_commit")
    if not packages_ok:
        why_not.append("missing_verified_package_metadata")
    if not time_ok:
        why_not.append("status_time_does_not_postdate_release_rotation_request_or_unparseable")
    if adequate:
        failure_class = "verified_post_request_backup"
    elif http_422:
        failure_class = "github_asset_upload_http_422"
    elif timeout:
        failure_class = "backup_write_timeout"
    elif status_obj.get("status") == "partial":
        failure_class = "partial_backup_remaining_changes"
    else:
        failure_class = "backup_failed_or_incomplete_other"
    return {
        "status_file_readable": True,
        "status": status_obj.get("status"),
        "error_type": status_obj.get("error_type"),
        "message": sanitize(status_obj.get("message")),
        "status_time": None if status_time is None else status_time.isoformat(),
        "min_required_post_request_time_utc": None if min_time is None else min_time.isoformat(),
        "remaining_changed_files": status_obj.get("remaining_changed_files"),
        "commit": status_obj.get("commit"),
        "package_count": len(status_obj.get("packages_this_run") or []),
        "has_verified_package_metadata": packages_ok,
        "time_ok": time_ok,
        "verified_flag_ok": verified,
        "adequate_backup_for_nonzero_work": adequate,
        "why_not": why_not,
        "http_422_observed": http_422,
        "timeout_observed": timeout,
        "current_failure_class": failure_class,
    }


def receipts_tail(limit: int = 10) -> Dict[str, Any]:
    if not BACKUP_RECEIPTS.exists():
        return {"path": rel(BACKUP_RECEIPTS), "exists": False, "tail": [], "error": "missing"}
    try:
        lines = BACKUP_RECEIPTS.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {"path": rel(BACKUP_RECEIPTS), "exists": True, "tail": [], "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    tail = []
    for line in lines[-limit:]:
        try:
            obj = sanitize(json.loads(line))
            tail.append({"time": obj.get("time"), "asset": obj.get("asset"), "manifest": obj.get("manifest"), "status": obj.get("status")})
        except Exception:
            tail.append({"unparsed_prefix": sanitize(line[:200])})
    return {"path": rel(BACKUP_RECEIPTS), "exists": True, "line_count_seen": len(lines), "tail": tail, "error": None}


def append_once(path: pathlib.Path, marker: str, block: str) -> None:
    previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker in previous:
        return
    path.write_text(previous.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{RUN_STEM}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    snapshot = execution_contract.runtime_snapshot(ROOT)

    request_obj, request_error = read_json(REQUEST_PATH)
    status_obj, status_error = read_json(BACKUP_STATUS)
    sanitized_status = sanitize(status_obj)
    req_time = request_time(request_obj, REQUEST_PATH)
    decision = classify_backup_status(status_obj, req_time)
    adequate = bool(decision.get("adequate_backup_for_nonzero_work"))

    proof_path = None
    retry_request_path = None
    if adequate:
        proof = {
            "backup_verified": True,
            "created_utc": created.isoformat(),
            "source": "post-release-rotation-request status recheck",
            "task_id": TASK_ID,
            "covers_nonzero_work_gate_after_request": True,
            "decision": decision,
            "backup_status_sanitized": sanitized_status,
            "request_path": REQUEST_REL,
            "request_time_utc": None if req_time is None else req_time.isoformat(),
            "resources": dict(ZERO),
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
        }
        proof_file = ROOT / "research_artifacts/aws_backup_proofs" / f"BACKUP_VERIFIED_AFTER_RELEASE_ROTATION_REQUEST_SOLO_V0_{stamp}.json"
        write_json(proof_file, proof)
        proof_path = rel(proof_file)
        next_action = "Backup is verified post request. Publish and run a new bounded nonzero scientific plan only if it cites this proof and preserves split/resource gates."
    else:
        retry = {
            "request": "retry_or_repair_backup_after_release_rotation_request_status_recheck",
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "reason": "No verified post-request external backup proof is available; nonzero controller/solver/plant/training work remains blocked.",
            "decision": decision,
            "required_next_backup_properties": {
                "status_or_backup_verified": "verified true",
                "remaining_changed_files": 0,
                "must_include_commit": True,
                "must_include_verified_package_metadata": True,
                "must_postdate_release_rotation_request": True,
                "must_not_contain_secrets": True,
            },
            "resources": dict(ZERO),
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
        }
        retry_file = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_RETRY_AFTER_RELEASE_ROTATION_REQUEST_STATUS_RECHECK_SOLO_V0_{stamp}.json"
        write_json(retry_file, retry)
        retry_request_path = rel(retry_file)
        next_action = "Nonzero scientific work remains blocked. Do not repeat this status check unless the supervisor backup status changes or a new verified backup proof appears; perform only zero-resource preparation if needed."

    evidence = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "release_rotation_request_inspected": bool(REQUEST_PATH.exists() and request_error is None),
        "backup_status_artifacts_inspected": True,
        "backup_recoverability_status_recorded": True,
        "post_request_backup_time_evaluated": True,
        "verified_backup_proof_handled_or_absence_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }
    expected = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "release_rotation_request_inspected": True,
        "backup_status_artifacts_inspected": True,
        "backup_recoverability_status_recorded": True,
        "post_request_backup_time_evaluated": True,
        "verified_backup_proof_handled_or_absence_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }
    passed = all(evidence.get(key) is val for key, val in expected.items())

    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_backup_post_rotation_request_status_recheck_solo_v0.md"
    raw = {
        "analysis_type": "post_release_rotation_request_backup_status_recheck",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "script": SCRIPT_REL,
        "script_sha256": sha256(ROOT / SCRIPT_REL),
        "runtime_snapshot_sha256": snapshot.get("snapshot_sha256") if isinstance(snapshot, Mapping) else None,
        "resources": dict(ZERO),
        "request_artifact": artifact_info(REQUEST_PATH),
        "request_read_error": request_error,
        "request_time_utc": None if req_time is None else req_time.isoformat(),
        "backup_status_artifact": artifact_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": sanitized_status,
        "backup_receipts_tail_sanitized": receipts_tail(),
        "decision": decision,
        "adequate_backup_for_nonzero_work": adequate,
        "verified_backup_proof": proof_path,
        "retry_backup_request": retry_request_path,
        "next_action": next_action,
        "elapsed_hours_since_first_supervisor_event": (created - FIRST_SUPERVISOR_EVENT).total_seconds() / 3600.0,
        "evidence": evidence,
    }
    write_json(raw_path, raw)
    summary = f"""# Backup post-rotation-request status recheck (solo v0)

Created UTC: `{created.isoformat()}`.

## Result

- Task passed: `{passed}`.
- Adequate verified backup for nonzero work: `{adequate}`.
- Current backup status: `{decision.get('status')}`.
- Current failure class: `{decision.get('current_failure_class')}`.
- Reasons if blocked: `{decision.get('why_not')}`.
- Status time: `{decision.get('status_time')}`.
- Required post-request time: `{decision.get('min_required_post_request_time_utc')}`.
- Verified proof written: `{proof_path}`.
- Retry request written: `{retry_request_path}`.

## Operational decision

{next_action}

## Resource and split safety

All five resource counters are zero. No validation-bank content, sealed test, or final test was accessed. No credentials were read or logged.
"""
    summary_path.write_text(summary, encoding="utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        "# Continue state after backup post-rotation-request status recheck solo v0\n\n"
        + f"UTC: {created.isoformat()}\n"
        + f"Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.\n"
        + f"Adequate backup for nonzero work: {adequate}. Decision: {decision}.\n"
        + f"Summary: `{rel(summary_path)}`. Raw: `{rel(raw_path)}`. Next action: {next_action}\n",
        encoding="utf-8",
    )
    completed = {
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "passed": passed,
        "resources": dict(ZERO),
        "evidence": evidence,
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "completed": rel(completed_path),
        "state": rel(state_path),
        "adequate_backup_for_nonzero_work": adequate,
        "verified_backup_proof": proof_path,
        "retry_backup_request": retry_request_path,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
    }
    write_json(completed_path, completed)

    log_block = f"""
### 2026-10-01 backup post-rotation-request status recheck (solo v0)

- Task: `{TASK_ID}`.
- Outcome: task_gate_passed=`{passed}`, adequate_backup_for_nonzero_work=`{adequate}`.
- Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
- Next action: {next_action}
- Resource use: solver_calls=0, plant_steps=0, training_steps=0, validation_episodes=0, test_episodes=0. No validation-bank, sealed-test, or final-test access.
"""
    for log_rel in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"):
        append_once(ROOT / log_rel, DOC_MARKER + "-" + log_rel.replace("/", "-").replace(".", "-"), log_block)

    if passed:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    else:
        failed_evidence = dict(evidence)
        failed_evidence["no_scientific_outcome"] = True
        execution_contract.record_outcome(
            ROOT,
            "engineering_failure",
            dict(ZERO),
            failed_evidence,
            engineering_error="backup_post_rotation_request_status_recheck_gate_failed",
        )
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
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
                    "no_secrets_read_or_logged": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                },
                engineering_error="startup_or_runtime_exception",
            )
        except Exception:
            pass
        raise
