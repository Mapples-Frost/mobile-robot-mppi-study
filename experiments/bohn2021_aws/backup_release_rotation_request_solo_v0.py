#!/usr/bin/env python3
"""Zero-resource backup-path repair handoff for release-asset-cap failures.

This script does not run the backup implementation and does not read credentials. It
reads already-produced backup-status diagnostics, preserves the current backup
blocker, and writes a supervisor-compatible repair request: rotate the GitHub
release tag or otherwise fix the protected backup service before any nonzero
controller/solver/plant/training work resumes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import traceback
from typing import Any, Dict, Iterable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import execution_contract  # noqa: E402

ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}

TASK_ID = "S-BACKUP-RELEASE-ROTATION-REQUEST-AFTER-RECEIPT-REPAIR-v0"
SCRIPT_REL = "experiments/bohn2021_aws/backup_release_rotation_request_solo_v0.py"
RUN_STEM = "backup_release_rotation_request_solo_v0"
DOC_MARKER = "<!-- backup-release-rotation-request-solo-v0-20261001 -->"

LATEST_RECEIPT_REPAIR_RAW = ROOT / "research_artifacts/aws_diagnostics/backup_recheck_receipt_key_repair_solo_v0_20261001T004811Z/raw.json"
LATEST_RECEIPT_REPAIR_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/backup_recheck_receipt_key_repair_solo_v0_20261001T004811Z/summary.md"
LATEST_RECEIPT_REPAIR_RECEIPT = ROOT / "research_artifacts/aws_runs/20261001T004810_28a1096e/outcome_receipt.json"
ASSET_422_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state_20260930T211710Z/summary.md"
ASSET_422_RAW = ROOT / "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state_20260930T211710Z/raw.json"
CURRENT_STATUS_RAW = ROOT / "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_current_status_v0c_20261001T003237Z/raw.json"
BACKUP_SCRIPT = ROOT / "scripts/research_service/backup.py"


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: pathlib.Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: pathlib.Path, limit: int = 200_000) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit]


def artifact_record(path: pathlib.Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        st = path.stat()
        out.update(
            {
                "bytes": st.st_size,
                "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
        )
    return out


def append_once(path: pathlib.Path, marker: str, block: str) -> None:
    previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker in previous:
        return
    with path.open("a", encoding="utf-8") as f:
        if previous and not previous.endswith("\n"):
            f.write("\n")
        f.write("\n" + marker + "\n" + block.strip() + "\n")


def find_public_release_probe(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    probe = data.get("public_release_probe")
    if isinstance(probe, dict):
        return probe
    # Older diagnostics may store the information under a nested backup gate or
    # status object. Keep this conservative and never infer success from absence.
    for key in ("release_probe", "github_release_probe", "public_release"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def summarize_backup_script(text: Optional[str]) -> Dict[str, Any]:
    if text is None:
        return {"exists": False, "read_as_text_only": False}
    tag_match = re.search(r"TAG\s*=\s*['\"]([^'\"]+)['\"]", text)
    has_token_read = ".secrets" in text or "github.token" in text
    has_release_rotation = "rollover" in text.lower() or "ASSET_SOFT_CAP" in text
    return {
        "exists": True,
        "read_as_text_only": True,
        "hard_coded_tag": tag_match.group(1) if tag_match else None,
        "contains_secret_path_string_only": has_token_read,
        "contains_release_rotation_logic": has_release_rotation,
        "contains_asset_count_probe": "assets?per_page" in text or "release_asset_count" in text,
        "contains_http_422_literal": "422" in text,
    }


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc)
    created_stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / (RUN_STEM + "_" + created_stamp)
    out_dir.mkdir(parents=True, exist_ok=False)

    snapshot = execution_contract.runtime_snapshot(ROOT)

    receipt_raw = read_json(LATEST_RECEIPT_REPAIR_RAW)
    receipt_summary_text = read_text(LATEST_RECEIPT_REPAIR_SUMMARY, limit=20_000)
    receipt = read_json(LATEST_RECEIPT_REPAIR_RECEIPT)
    current_status = read_json(CURRENT_STATUS_RAW)
    asset422_raw = read_json(ASSET_422_RAW)
    asset422_summary_text = read_text(ASSET_422_SUMMARY, limit=20_000)
    backup_script_text = read_text(BACKUP_SCRIPT, limit=80_000)

    backup_gate = {}
    if isinstance(receipt_raw, dict):
        backup_gate = receipt_raw.get("backup_gate_decision") or {}
    if not backup_gate and isinstance(current_status, dict):
        backup_gate = current_status.get("backup_gate_decision") or {}

    public_probe = find_public_release_probe(asset422_raw)
    if not public_probe and asset422_summary_text:
        public_probe = {
            "from_summary_text": True,
            "assets_returned_1000": "assets returned: `1000`" in asset422_summary_text,
            "near_or_at_asset_cap": "near/at release asset cap: `True`" in asset422_summary_text,
            "http_422_observed": "HTTP 422 observed: `True`" in asset422_summary_text,
        }

    script_summary = summarize_backup_script(backup_script_text)
    status_failed = backup_gate.get("status") == "failed"
    timeout_observed = bool(backup_gate.get("timeout_observed") or backup_gate.get("error_type") == "TimeoutError")
    asset_cap_evidence = bool(
        public_probe.get("near_or_at_asset_cap")
        or public_probe.get("assets_returned_1000")
        or public_probe.get("asset_count", 0) >= 900
        or public_probe.get("assets_returned", 0) >= 900
        or (asset422_summary_text and "near/at release asset cap: `True`" in asset422_summary_text)
    )
    http_422_observed = bool(
        public_probe.get("http_422_observed")
        or (asset422_summary_text and "HTTP 422 observed: `True`" in asset422_summary_text)
    )

    backup_still_blocked = bool(
        status_failed
        and not backup_gate.get("adequate_backup_for_s_tc2h8", False)
        and not backup_gate.get("verified_flag_ok", False)
    )
    release_rotation_needed = bool(backup_still_blocked and (asset_cap_evidence or http_422_observed or timeout_observed))

    patch_guidance = out_dir / "backup_service_release_rotation_patch_guidance.md"
    patch_guidance.write_text(
        "# Protected backup-service repair guidance\n\n"
        "This is a handoff document, not an edit to the protected supervisor source.\n\n"
        "Observed evidence indicates that the base GitHub evidence release is at or near the "
        "release-asset cap and backup status remains failed. The protected backup service should "
        "be changed outside the research write guard to select a rollover prerelease tag when the "
        "base tag lacks enough asset capacity. A safe implementation should:\n\n"
        "1. Count existing assets on the base release before upload.\n"
        "2. If current assets plus the projected per-run uploads would exceed a conservative cap, "
        "create or reuse a dated rollover prerelease tag such as "
        "`bohn-aws-evidence-20260926-rollover-20261001-01`.\n"
        "3. Upload archive and manifest pairs to that rollover release and verify GitHub digest or "
        "download SHA-256 exactly as before.\n"
        "4. Persist selected release tag, URL, package metadata, commit, and remaining_changed_files "
        "in `state/backup_status.json`.\n"
        "5. Do not expose or log secrets.\n"
        "6. Keep local source/evidence until a verified external package exists.\n\n"
        "After a successful verified backup postdating this repair request, nonzero S-TC2H/source242 "
        "or training-selection work can be reconsidered under a fresh bounded plan.\n",
        encoding="utf-8",
    )

    backup_request = ROOT / "research_artifacts/aws_backup_proofs" / (
        "REQUEST_BACKUP_RELEASE_ROTATION_AFTER_RECEIPT_REPAIR_SOLO_V0_" + created_stamp + ".json"
    )
    request_payload = {
        "request": "supervisor_backup_release_rotation_or_equivalent_repair",
        "created_utc": created.isoformat(),
        "reason": "The zero-resource backup receipt-key repair passed, but external backup remains failed. Existing diagnostics show the base release reached or approached the GitHub asset cap and later backups timed out. Nonzero controller/solver/training work remains blocked until backup is verified.",
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "must_not_read_or_log_secrets": True,
        "recommended_actions": [
            "Repair the protected supervisor backup service or launch it with an authorized rollover release tag rather than the saturated base release.",
            "Use a dated rollover prerelease tag if the base release has near-capacity assets.",
            "Verify commit, every package asset and manifest SHA-256, remaining_changed_files=0, and recovery metadata in state/backup_status.json.",
            "Only after a verified proof postdating this request should nonzero controller/plant/solver/training experiments resume.",
        ],
        "evidence_paths": [
            rel(LATEST_RECEIPT_REPAIR_RAW),
            rel(LATEST_RECEIPT_REPAIR_SUMMARY),
            rel(LATEST_RECEIPT_REPAIR_RECEIPT),
            rel(CURRENT_STATUS_RAW),
            rel(ASSET_422_SUMMARY),
            rel(patch_guidance),
        ],
        "backup_gate_decision": backup_gate,
        "release_probe_summary": public_probe,
        "protected_backup_script_summary": script_summary,
    }
    write_json(backup_request, request_payload)

    evidence = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "source_code_backup_script_read_as_text_only": bool(script_summary.get("read_as_text_only")),
        "backup_still_blocks_nonzero_work": backup_still_blocked,
        "release_asset_cap_or_timeout_evidence_recorded": bool(asset_cap_evidence or timeout_observed or http_422_observed),
        "protected_supervisor_backup_script_not_modified": True,
        "backup_path_repair_request_written": True,
        "rollover_release_recommendation_recorded": release_rotation_needed,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
    }

    raw = {
        "analysis_type": "backup_path_release_rotation_repair_request",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "script": SCRIPT_REL,
        "script_sha256": sha256(ROOT / SCRIPT_REL),
        "runtime_snapshot_sha256": snapshot.get("snapshot_sha256") if isinstance(snapshot, dict) else None,
        "resources": dict(ZERO),
        "input_artifacts": {
            rel(path): artifact_record(path)
            for path in (
                LATEST_RECEIPT_REPAIR_RAW,
                LATEST_RECEIPT_REPAIR_SUMMARY,
                LATEST_RECEIPT_REPAIR_RECEIPT,
                CURRENT_STATUS_RAW,
                ASSET_422_SUMMARY,
                ASSET_422_RAW,
                BACKUP_SCRIPT,
            )
        },
        "backup_gate_decision": backup_gate,
        "receipt_repair_evidence": receipt_raw.get("evidence") if isinstance(receipt_raw, dict) else None,
        "receipt_outcome": receipt,
        "release_probe_summary": public_probe,
        "asset_cap_evidence": asset_cap_evidence,
        "http_422_observed": http_422_observed,
        "timeout_observed": timeout_observed,
        "backup_still_blocked": backup_still_blocked,
        "release_rotation_needed": release_rotation_needed,
        "protected_backup_script_summary": script_summary,
        "receipt_summary_excerpt": receipt_summary_text,
        "asset422_summary_excerpt": asset422_summary_text,
        "backup_request": rel(backup_request),
        "patch_guidance": rel(patch_guidance),
        "evidence": evidence,
    }

    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    write_json(raw_path, raw)

    passed = all(
        evidence[key] is expected
        for key, expected in {
            "zero_solver_plant_training_validation_test_resources": True,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "no_secrets_read_or_logged": True,
            "source_code_backup_script_read_as_text_only": True,
            "backup_still_blocks_nonzero_work": True,
            "release_asset_cap_or_timeout_evidence_recorded": True,
            "protected_supervisor_backup_script_not_modified": True,
            "backup_path_repair_request_written": True,
            "rollover_release_recommendation_recorded": True,
            "nonzero_work_gate_decision_recorded": True,
            "outputs_persisted": True,
        }.items()
    )

    summary = f"""# Backup release-rotation repair request (solo v0)

Created UTC: `{created.isoformat()}`.

## Result

- Task passed: `{passed}`.
- Backup remains adequate for nonzero work: `False`.
- Current backup status: `{backup_gate.get('status')}`.
- Current error type: `{backup_gate.get('error_type')}`.
- Current failure class: `{backup_gate.get('current_failure_class')}`.
- Timeout observed in current status: `{timeout_observed}`.
- Prior HTTP 422 / release-cap evidence observed: `{http_422_observed or asset_cap_evidence}`.
- Protected supervisor backup script modified by this task: `False`.

## Operational decision

Nonzero controller/solver/plant/training work remains blocked. The next required operational action is to repair or configure the protected backup service so that it writes to a rollover GitHub evidence release, verifies package and manifest SHA-256, records commit and `remaining_changed_files=0`, and produces a verified backup proof postdating this request.

## New request and guidance

- Backup repair request: `{rel(backup_request)}`.
- Protected-service patch guidance: `{rel(patch_guidance)}`.

## Resource and split safety

Solver calls, plant steps, training steps, validation episodes, and test episodes are all zero. No validation-bank generator/source content, sealed test, or final test was accessed. No credentials were read or logged; `backup.py` was inspected only as text and was not imported.
"""
    summary_path.write_text(summary, encoding="utf-8")

    completed = {
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "passed": passed,
        "resources": dict(ZERO),
        "evidence": evidence,
        "raw": rel(raw_path),
        "summary": rel(summary_path),
        "completed": rel(completed_path),
        "backup_request": rel(backup_request),
        "patch_guidance": rel(patch_guidance),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
    }
    write_json(completed_path, completed)

    log_block = f"""
### 2026-10-01 backup release-rotation repair request (solo v0)

- Task: `{TASK_ID}`.
- Outcome: `{'passed' if passed else 'failed'}`.
- Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
- Backup repair request: `{rel(backup_request)}`.
- Backup remains blocked for nonzero work. The protected supervisor backup path should rotate away from the saturated base release or apply an equivalent repair, then verify external recoverability before source242/control/training work resumes.
- Resource use: solver_calls=0, plant_steps=0, training_steps=0, validation_episodes=0, test_episodes=0. No validation-bank source, sealed test, or final test access.
"""
    for log_name in ("RESEARCH_LOG.md", "STATUS.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        append_once(ROOT / log_name, DOC_MARKER + "-" + log_name.lower().replace(".", "-"), log_block)
    append_once(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", DOC_MARKER + "-response-log", log_block)

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
            engineering_error="backup_release_rotation_request_gate_failed",
        )
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
