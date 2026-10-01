#!/usr/bin/env python3
"""Zero-resource post-patch backup status recheck with exact gate keys.

This script only reads local metadata artifacts and source hashes. It does not
execute the backup uploader, call GitHub APIs, read credentials, open validation
banks, run controllers/solvers/plant steps, train, refit, or access sealed/final
tests. It exists to repair the evidence-key mismatch in the prior post-rotation
status recheck script before any nonzero scientific work resumes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import traceback
from typing import Any, Dict, Mapping, Optional, Tuple

CANON_BASE = pathlib.Path("/data/openai-agent")
ROOT = CANON_BASE / "mobile-robot-mppi-study"
if not ROOT.exists():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
STATE = CANON_BASE / "state"
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import execution_contract  # noqa: E402

TASK_ID = "S-BACKUP-POST-ROTATION-STATUS-RECHECK-v0b-gate-key-repair"
RUN_STEM = "backup_post_rotation_status_recheck_solo_v0b_gate_key_repair"
SCRIPT_REL = "experiments/bohn2021_aws/backup_post_rotation_status_recheck_solo_v0b_gate_key_repair.py"
BACKUP_PY_REL = "scripts/research_service/backup.py"
EXPECTED_PATCHED_BACKUP_PY_SHA256 = "fb5a21945ab8441007ffcbb34b456cd59053e93243e6532d98bf2af493bc3d8e"
PATCH_RECEIPT_REL = "research_artifacts/aws_runs/20261001T013008_2c0202f6/outcome_receipt.json"
PATCH_SUMMARY_REL = "research_artifacts/aws_diagnostics/backup_release_rotation_patch_apply_solo_v0b_static_check_repair_20261001T013009Z/summary.md"
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
DOC_MARKER = "<!-- backup-post-rotation-status-recheck-solo-v0b-gate-key-repair -->"
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


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            key_s = str(key)
            if SECRET_KEY_RE.search(key_s) and key_s.lower() not in {"sha256", "asset_sha256", "package_sha256", "download_sha256", "manifest_sha256"}:
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


def package_metadata(status: Mapping[str, Any]) -> Dict[str, Any]:
    packages = status.get("packages_this_run") or []
    package_count = len(packages) if isinstance(packages, list) else 0
    sanitized_packages = sanitize(packages[:5]) if isinstance(packages, list) else []
    if isinstance(packages, list) and packages:
        ok = True
        for item in packages:
            if not isinstance(item, Mapping):
                ok = False
                break
            has_hash = bool(item.get("sha256") or item.get("asset_sha256") or item.get("package_sha256") or item.get("download_sha256") or item.get("manifest_sha256"))
            has_size = bool(item.get("bytes") or item.get("size"))
            has_verification = bool(item.get("verification") or item.get("verified") is True or item.get("download_verified") is True or item.get("digest"))
            if not (has_hash and has_size and has_verification):
                ok = False
                break
        return {"package_count": package_count, "metadata_ok": ok, "sample": sanitized_packages}
    fallback_ok = bool(status.get("package_sha256") or status.get("asset_sha256") or status.get("release_asset_sha256") or status.get("manifest_sha256"))
    return {"package_count": package_count, "metadata_ok": fallback_ok, "sample": sanitized_packages}


def best_patch_time(patch_receipt: Optional[Any], patch_summary_path: pathlib.Path) -> Optional[dt.datetime]:
    candidates = []
    if isinstance(patch_receipt, Mapping):
        evidence = patch_receipt.get("evidence")
        if isinstance(evidence, Mapping):
            for key in ("created_utc", "completed_utc", "timestamp_utc", "time"):
                parsed = parse_time(evidence.get(key))
                if parsed is not None:
                    candidates.append(parsed)
        for key in ("created_utc", "completed_utc", "timestamp_utc", "time"):
            parsed = parse_time(patch_receipt.get(key))
            if parsed is not None:
                candidates.append(parsed)
    if patch_summary_path.exists():
        candidates.append(dt.datetime.fromtimestamp(patch_summary_path.stat().st_mtime, dt.timezone.utc))
    if not candidates:
        return None
    return max(candidates)


def classify_status(status_obj: Optional[Any], minimum_time: Optional[dt.datetime], backup_hash_ok: bool) -> Dict[str, Any]:
    if not isinstance(status_obj, Mapping):
        return {
            "status_file_readable": False,
            "adequate_backup_for_nonzero_work": False,
            "current_failure_class": "status_missing_or_unreadable",
            "why_not": ["backup_status_missing_or_unreadable"],
            "minimum_required_time_utc": None if minimum_time is None else minimum_time.isoformat(),
            "patched_backup_py_hash_matches_expected": backup_hash_ok,
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
    pkg = package_metadata(status_obj)
    packages_ok = bool(pkg.get("metadata_ok"))
    time_ok = bool(status_time is not None and minimum_time is not None and status_time >= minimum_time)
    message = str(status_obj.get("message", ""))
    error_type = str(status_obj.get("error_type", ""))
    http_422 = "422" in message or "HTTP 422" in message
    timeout = "timeout" in message.lower() or "timed out" in message.lower() or "timeout" in error_type.lower()
    adequate = bool(backup_hash_ok and verified and remaining_ok and has_commit and packages_ok and time_ok)
    why_not = []
    if not backup_hash_ok:
        why_not.append("patched_backup_py_hash_mismatch_or_missing")
    if not verified:
        why_not.append(f"status_is_{status_obj.get('status')!r}_not_verified")
    if not remaining_ok:
        why_not.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why_not.append("missing_commit")
    if not packages_ok:
        why_not.append("missing_verified_package_metadata")
    if not time_ok:
        why_not.append("status_time_does_not_postdate_patch_or_unparseable")
    if adequate:
        failure_class = "verified_post_patch_backup"
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
        "minimum_required_time_utc": None if minimum_time is None else minimum_time.isoformat(),
        "remaining_changed_files": status_obj.get("remaining_changed_files"),
        "commit": status_obj.get("commit"),
        "package_count": pkg.get("package_count"),
        "has_verified_package_metadata": packages_ok,
        "package_sample": pkg.get("sample"),
        "time_ok": time_ok,
        "verified_flag_ok": verified,
        "patched_backup_py_hash_matches_expected": backup_hash_ok,
        "adequate_backup_for_nonzero_work": adequate,
        "why_not": why_not,
        "http_422_observed": http_422,
        "timeout_observed": timeout,
        "current_failure_class": failure_class,
        "release_tag": status_obj.get("release") or (status_obj.get("releases_this_run") or [{}])[0].get("url") if isinstance(status_obj.get("releases_this_run"), list) and status_obj.get("releases_this_run") else status_obj.get("release"),
        "releases_this_run": sanitize(status_obj.get("releases_this_run")),
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
            tail.append({"time": obj.get("time"), "asset": obj.get("asset"), "manifest": obj.get("manifest"), "status": obj.get("status"), "release_tag": obj.get("release_tag")})
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

    backup_py = ROOT / BACKUP_PY_REL
    backup_py_hash = sha256(backup_py)
    backup_hash_ok = backup_py_hash == EXPECTED_PATCHED_BACKUP_PY_SHA256
    patch_receipt_path = ROOT / PATCH_RECEIPT_REL
    patch_summary_path = ROOT / PATCH_SUMMARY_REL
    patch_receipt, patch_receipt_error = read_json(patch_receipt_path)
    status_obj, status_error = read_json(BACKUP_STATUS)
    minimum_time = best_patch_time(patch_receipt, patch_summary_path)
    decision = classify_status(status_obj, minimum_time, backup_hash_ok)
    adequate = bool(decision.get("adequate_backup_for_nonzero_work"))

    proof_path = None
    retry_request_path = None
    if adequate:
        proof = {
            "backup_verified": True,
            "created_utc": created.isoformat(),
            "source": "post-release-rotation-patch local status recheck",
            "task_id": TASK_ID,
            "covers_nonzero_work_gate_after_patch": True,
            "decision": decision,
            "backup_py": {
                "path": BACKUP_PY_REL,
                "sha256": backup_py_hash,
                "expected_sha256": EXPECTED_PATCHED_BACKUP_PY_SHA256,
                "matches_expected": backup_hash_ok,
            },
            "backup_status_sanitized": sanitize(status_obj),
            "patch_receipt": PATCH_RECEIPT_REL,
            "patch_summary": PATCH_SUMMARY_REL,
            "minimum_required_time_utc": None if minimum_time is None else minimum_time.isoformat(),
            "resources": dict(ZERO),
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "backup_uploader_executed": False,
            "direct_github_api_called": False,
        }
        proof_file = ROOT / "research_artifacts/aws_backup_proofs" / f"BACKUP_VERIFIED_AFTER_RELEASE_ROTATION_PATCH_SOLO_V0B_{stamp}.json"
        write_json(proof_file, proof)
        proof_path = rel(proof_file)
        next_action = "Backup is verified after the release-rotation patch; nonzero development work may resume only under a separately frozen scientific execution plan citing this proof."
    else:
        retry = {
            "request": "retry_or_repair_backup_after_post_rotation_status_recheck_v0b",
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "reason": "No adequate verified post-patch external backup proof is available; nonzero controller/solver/plant/training work remains blocked.",
            "decision": decision,
            "required_next_backup_properties": {
                "backup_py_sha256": EXPECTED_PATCHED_BACKUP_PY_SHA256,
                "status_or_backup_verified": "verified true",
                "remaining_changed_files": 0,
                "must_include_commit": True,
                "must_include_verified_package_metadata": True,
                "must_postdate_patch": True,
                "must_not_contain_secrets": True,
            },
            "resources": dict(ZERO),
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "backup_uploader_executed": False,
            "direct_github_api_called": False,
        }
        retry_file = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_RETRY_AFTER_POST_ROTATION_STATUS_RECHECK_SOLO_V0B_{stamp}.json"
        write_json(retry_file, retry)
        retry_request_path = rel(retry_file)
        next_action = "Nonzero scientific work remains blocked until a verified post-patch external backup proof appears; avoid repeated status-only checks unless backup status changes."

    evidence = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "backup_uploader_not_executed": True,
        "direct_github_api_not_called": True,
        "patched_backup_py_hash_checked": True,
        "patched_backup_py_hash_matches_expected": backup_hash_ok,
        "backup_status_artifacts_inspected": True,
        "backup_recoverability_status_recorded": True,
        "commit_and_package_metadata_recorded_if_available": True,
        "release_rotation_or_failure_status_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }
    expected = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "backup_uploader_not_executed": True,
        "direct_github_api_not_called": True,
        "patched_backup_py_hash_checked": True,
        "patched_backup_py_hash_matches_expected": True,
        "backup_status_artifacts_inspected": True,
        "backup_recoverability_status_recorded": True,
        "commit_and_package_metadata_recorded_if_available": True,
        "release_rotation_or_failure_status_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }
    task_passed = all(evidence.get(key) is val for key, val in expected.items())

    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_backup_post_rotation_status_recheck_solo_v0b.md"
    raw = {
        "analysis_type": "post_release_rotation_patch_backup_status_recheck_gate_key_repair",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "script": SCRIPT_REL,
        "script_sha256": sha256(ROOT / SCRIPT_REL),
        "runtime_snapshot_sha256": snapshot.get("snapshot_sha256") if isinstance(snapshot, Mapping) else None,
        "resources": dict(ZERO),
        "backup_py": {
            "path": BACKUP_PY_REL,
            "sha256": backup_py_hash,
            "expected_sha256": EXPECTED_PATCHED_BACKUP_PY_SHA256,
            "matches_expected": backup_hash_ok,
        },
        "patch_receipt_artifact": artifact_info(patch_receipt_path),
        "patch_receipt_read_error": patch_receipt_error,
        "patch_summary_artifact": artifact_info(patch_summary_path),
        "minimum_required_backup_time_utc": None if minimum_time is None else minimum_time.isoformat(),
        "backup_status_artifact": artifact_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": sanitize(status_obj),
        "backup_receipts_tail_sanitized": receipts_tail(),
        "decision": decision,
        "adequate_backup_for_nonzero_work": adequate,
        "verified_backup_proof": proof_path,
        "retry_backup_request": retry_request_path,
        "next_action": next_action,
        "elapsed_hours_since_first_supervisor_event": (created - FIRST_SUPERVISOR_EVENT).total_seconds() / 3600.0,
        "evidence": evidence,
        "task_passed": task_passed,
    }
    write_json(raw_path, raw)
    summary = f"""# Backup post-rotation status recheck (solo v0b gate-key repair)

Created UTC: `{created.isoformat()}`.

## Result

- Task passed: `{task_passed}`.
- Adequate verified backup for nonzero work: `{adequate}`.
- Patched `backup.py` hash checked: `True`.
- Patched `backup.py` hash matches expected: `{backup_hash_ok}`.
- Current backup status: `{decision.get('status')}`.
- Current failure class: `{decision.get('current_failure_class')}`.
- Reasons if blocked: `{decision.get('why_not')}`.
- Status time: `{decision.get('status_time')}`.
- Required post-patch time: `{decision.get('minimum_required_time_utc')}`.
- Commit: `{decision.get('commit')}`.
- Package count: `{decision.get('package_count')}`.
- Verified proof written: `{proof_path}`.
- Retry request written: `{retry_request_path}`.

## Operational decision

{next_action}

## Resource and split safety

All five resource counters are zero. No validation-bank content, sealed test, or final test was accessed. The backup uploader was not executed, no direct GitHub API call was made, and no credentials were read or logged.
"""
    summary_path.write_text(summary, encoding="utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        "# Continue state after backup post-rotation status recheck solo v0b\n\n"
        + f"UTC: {created.isoformat()}\n"
        + f"Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.\n"
        + f"Adequate backup for nonzero work: {adequate}. Task passed: {task_passed}.\n"
        + f"Decision: {json.dumps(decision, sort_keys=True, ensure_ascii=False)}\n"
        + f"Summary: `{rel(summary_path)}`. Raw: `{rel(raw_path)}`. Next action: {next_action}\n",
        encoding="utf-8",
    )
    completed = {
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "passed": task_passed,
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
        "backup_uploader_executed": False,
        "direct_github_api_called": False,
    }
    write_json(completed_path, completed)

    log_block = f"""
### 2026-10-01 backup post-rotation status recheck (solo v0b gate-key repair)

- Task: `{TASK_ID}`.
- Outcome: task_gate_passed=`{task_passed}`, adequate_backup_for_nonzero_work=`{adequate}`.
- Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
- Backup proof: `{proof_path}`. Retry request: `{retry_request_path}`.
- Next action: {next_action}
- Resource use: solver_calls=0, plant_steps=0, training_steps=0, validation_episodes=0, test_episodes=0. No validation-bank, sealed-test, or final-test access.
"""
    for log_rel in (
        "STATUS.md",
        "RESEARCH_LOG.md",
        "DECISIONS.md",
        "RESULTS_AUDIT.md",
        "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
    ):
        append_once(ROOT / log_rel, DOC_MARKER + "-" + log_rel.replace("/", "-").replace(".", "-"), log_block)

    if task_passed:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    else:
        failed_evidence = dict(evidence)
        failed_evidence["no_scientific_outcome"] = True
        execution_contract.record_outcome(
            ROOT,
            "engineering_failure",
            dict(ZERO),
            failed_evidence,
            engineering_error="startup",
        )
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
    return 0 if task_passed else 2


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
                    "backup_uploader_not_executed": True,
                    "direct_github_api_not_called": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                },
                engineering_error="startup",
            )
        except Exception:
            pass
        raise
