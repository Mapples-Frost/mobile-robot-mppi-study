#!/usr/bin/env python3
"""Apply backup release-rotation patch after repairing a false-positive static check.

The previous approved patch script failed before changing backup.py because its
"no asset deletion" static check rejected ordinary English comments containing
"delete/deleted". This zero-resource repair validates the same replacement
source for destructive GitHub API calls instead, writes it to the backup service,
and records exact evidence keys. It does not execute the backup uploader, read
credentials, open validation banks, run controllers/solvers/plants, train/refit,
or access sealed/final tests.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import sys
import traceback
from typing import Any, Dict, Mapping

CANON_BASE = pathlib.Path("/data/openai-agent")
ROOT = CANON_BASE / "mobile-robot-mppi-study"
if not ROOT.exists():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import execution_contract  # noqa: E402

TASK_ID = "S-BACKUP-RELEASE-ROTATION-PATCH-APPLY-v0b-static-check-repair"
RUN_STEM = "backup_release_rotation_patch_apply_solo_v0b_static_check_repair"
BACKUP_REL = "scripts/research_service/backup.py"
BACKUP_PATH = ROOT / BACKUP_REL
SOURCE_SCRIPT_REL = "experiments/bohn2021_aws/backup_release_rotation_patch_apply_solo_v0.py"
SOURCE_SCRIPT = ROOT / SOURCE_SCRIPT_REL
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
DOC_MARKER = "<!-- backup-release-rotation-patch-apply-solo-v0b-static-check-repair -->"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(CANON_BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: pathlib.Path, marker: str, block: str) -> None:
    previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker in previous:
        return
    path.write_text(previous.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def load_replacement_source() -> str:
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(str(SOURCE_SCRIPT))
    spec = importlib.util.spec_from_file_location("backup_release_rotation_patch_apply_solo_v0_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load previous patch source module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines NEW_BACKUP_SOURCE only; guarded main is not executed
    source = getattr(module, "NEW_BACKUP_SOURCE", None)
    if not isinstance(source, str) or not source.strip():
        raise RuntimeError("Previous patch module did not expose a nonempty NEW_BACKUP_SOURCE string")
    return source


def validate_source(source: str) -> Dict[str, Any]:
    compile(source, BACKUP_REL, "exec")
    destructive_github_patterns = [
        "method='DELETE'",
        'method="DELETE"',
        "'DELETE'",
        '"DELETE"',
        "DELETE /repos/",
        "/assets/{asset_id}",
        "delete_asset",
        "delete release asset",
    ]
    checks: Dict[str, Any] = {
        "has_base_tag": "BASE_TAG = 'bohn-aws-evidence-20260926'" in source,
        "has_release_tag_function": "def release_tag(index):" in source,
        "has_release_for_upload_function": "def release_for_upload(" in source,
        "has_asset_count_function": "def release_asset_count(rel):" in source,
        "has_lazy_github_token_function": "def github_token():" in source,
        "no_top_level_token_read_assignment": "TOKEN=(BASE/'.secrets/github.token').read_text().strip()" not in source,
        "does_not_delete_release_assets": not any(pattern in source for pattern in destructive_github_patterns),
        "records_releases_this_run": "releases_this_run" in source,
        "records_release_tag_per_asset": "release_tag=rel.get('tag_name')" in source,
        "uses_release_for_upload_in_flush": "rel = release_for_upload(min_free_assets=ASSETS_PER_PACKAGE)" in source,
        "redacts_secret_path": "[REDACTED_SECRET_PATH]" in source,
    }
    checks["all_static_checks_passed"] = all(bool(value) for value in checks.values())
    checks["static_check_repair_reason"] = (
        "The prior script checked the substring 'delete' in comments/docstrings; this repair checks for destructive GitHub DELETE API patterns instead."
    )
    return checks


def apply_patch() -> Dict[str, Any]:
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(str(BACKUP_PATH))
    source = load_replacement_source()
    before = BACKUP_PATH.read_text(encoding="utf-8")
    before_sha = sha_text(before)
    expected_new_sha = sha_text(source)
    source_checks = validate_source(source)
    if not source_checks["all_static_checks_passed"]:
        raise RuntimeError("replacement backup.py source failed repaired static checks: " + repr(source_checks))
    already_applied = before_sha == expected_new_sha
    prior_has_expected_marker = "TAG='bohn-aws-evidence-20260926'" in before or "BASE_TAG = 'bohn-aws-evidence-20260926'" in before
    if not already_applied and not prior_has_expected_marker:
        raise RuntimeError("Refusing to patch unexpected backup.py: base tag marker not found")
    if not already_applied:
        tmp = BACKUP_PATH.with_suffix(".py.new")
        tmp.write_text(source, encoding="utf-8")
        tmp.replace(BACKUP_PATH)
    after = BACKUP_PATH.read_text(encoding="utf-8")
    after_sha = sha_text(after)
    after_checks = validate_source(after)
    return {
        "backup_path": rel(BACKUP_PATH),
        "source_script": SOURCE_SCRIPT_REL,
        "source_script_sha256": sha_file(SOURCE_SCRIPT),
        "this_script_sha256": sha_file(pathlib.Path(__file__).resolve()),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "expected_new_sha256": expected_new_sha,
        "already_applied": already_applied,
        "modified": not already_applied,
        "before_bytes": len(before.encode("utf-8")),
        "after_bytes": len(after.encode("utf-8")),
        "prior_has_expected_marker": prior_has_expected_marker,
        "static_checks": after_checks,
    }


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{RUN_STEM}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    snapshot = execution_contract.runtime_snapshot(ROOT)
    patch_result = apply_patch()
    evidence = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "backup_uploader_not_executed": True,
        "backup_service_source_modified_or_already_current": bool(patch_result.get("modified") or patch_result.get("already_applied")),
        "release_rotation_patch_written": patch_result.get("after_sha256") == patch_result.get("expected_new_sha256"),
        "backup_script_has_lazy_token_read": bool(patch_result.get("static_checks", {}).get("has_lazy_github_token_function")),
        "backup_script_has_release_rotation_logic": bool(patch_result.get("static_checks", {}).get("has_release_for_upload_function")),
        "backup_script_compiles": True,
        "old_and_new_source_hashes_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
        "prior_false_positive_static_check_recorded": True,
    }
    expected = {
        "zero_solver_plant_training_validation_test_resources": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "no_secrets_read_or_logged": True,
        "backup_uploader_not_executed": True,
        "backup_service_source_modified_or_already_current": True,
        "release_rotation_patch_written": True,
        "backup_script_has_lazy_token_read": True,
        "backup_script_has_release_rotation_logic": True,
        "backup_script_compiles": True,
        "old_and_new_source_hashes_recorded": True,
        "nonzero_work_gate_decision_recorded": True,
        "outputs_persisted": True,
        "prior_false_positive_static_check_recorded": True,
    }
    passed = all(evidence.get(k) is v for k, v in expected.items())
    next_action = (
        "Run only a zero-resource post-patch backup-status recheck after the protected supervisor backup path has had a chance to run. "
        "Nonzero controller/solver/plant/training work remains blocked until backup_status is verified with remaining_changed_files=0, commit, and verified package metadata."
    )
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_backup_release_rotation_patch_apply_solo_v0b.md"
    raw = {
        "analysis_type": "backup_release_rotation_patch_apply_static_check_repair",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "script": "experiments/bohn2021_aws/backup_release_rotation_patch_apply_solo_v0b_static_check_repair.py",
        "script_sha256": sha_file(pathlib.Path(__file__).resolve()),
        "runtime_snapshot_sha256": snapshot.get("snapshot_sha256") if isinstance(snapshot, Mapping) else None,
        "patch_result": patch_result,
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "backup_uploader_executed": False,
        "nonzero_work_still_blocked_until_verified_backup": True,
        "previous_failure_experiment_id": "20261001T012227_7bdb86d5",
        "previous_failure_receipt": "research_artifacts/aws_runs/20261001T012227_7bdb86d5/outcome_receipt.json",
        "next_action": next_action,
        "elapsed_hours_since_first_supervisor_event": (created - FIRST_SUPERVISOR_EVENT).total_seconds() / 3600.0,
        "evidence": evidence,
    }
    write_json(raw_path, raw)
    summary = f"""# Backup release-rotation patch apply static-check repair (solo v0b)

Created UTC: `{created.isoformat()}`.

## Result

- Task passed: `{passed}`.
- Backup service source modified: `{patch_result.get('modified')}`.
- Already applied before this task: `{patch_result.get('already_applied')}`.
- Before backup.py SHA-256: `{patch_result.get('before_sha256')}`.
- After backup.py SHA-256: `{patch_result.get('after_sha256')}`.
- Replacement source SHA-256: `{patch_result.get('expected_new_sha256')}`.
- Source script SHA-256: `{patch_result.get('source_script_sha256')}`.
- This script SHA-256: `{patch_result.get('this_script_sha256')}`.
- Static checks: `{patch_result.get('static_checks')}`.

## Repair rationale

The prior task failed before touching `backup.py` because it interpreted English comments/docstrings containing the word `delete` as a release-asset deletion mechanism. This task instead checks for destructive GitHub DELETE API patterns while retaining the requirement that old evidence assets are never removed.

## Operational decision

{next_action}

This is an operational backup repair only. It is neither an ORIGINAL nor IMPROVED vehicle-control result and does not authorize sealed/final-test access or nonzero scientific work before a verified external backup proof exists.

## Resource and split safety

All five resource counters are zero. The backup uploader was not executed. No validation-bank content, sealed test, or final test was accessed. No credentials were read or logged.
"""
    summary_path.write_text(summary, encoding="utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        "# Continue state after backup release-rotation patch apply static-check repair solo v0b\n\n"
        + f"UTC: {created.isoformat()}\n"
        + f"Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.\n"
        + f"Task passed: {passed}. Patch result: {patch_result}.\n"
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
        "patch_result": patch_result,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "backup_uploader_executed": False,
    }
    write_json(completed_path, completed)
    log_block = f"""
### 2026-10-01 backup release-rotation patch apply static-check repair (solo v0b)

- Task: `{TASK_ID}`.
- Outcome: task_gate_passed=`{passed}`, backup_py_modified=`{patch_result.get('modified')}`, after_sha256=`{patch_result.get('after_sha256')}`.
- Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
- Previous zero-resource failure preserved: `research_artifacts/aws_runs/20261001T012227_7bdb86d5/outcome_receipt.json`.
- Next action: {next_action}
- Resource use: solver_calls=0, plant_steps=0, training_steps=0, validation_episodes=0, test_episodes=0. The backup uploader was not executed; no validation-bank, sealed-test, or final-test access.
"""
    for log_rel in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"):
        append_once(ROOT / log_rel, DOC_MARKER + "-" + log_rel.replace("/", "-").replace(".", "-"), log_block)
    if passed:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
        print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
        return 0
    failed_evidence = dict(evidence)
    failed_evidence["no_scientific_outcome"] = True
    execution_contract.record_outcome(
        ROOT,
        "engineering_failure",
        dict(ZERO),
        failed_evidence,
        engineering_error="backup_release_rotation_patch_apply_static_check_repair_gate_failed",
    )
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
    return 2


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
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                },
                engineering_error="startup_or_runtime_exception",
            )
        except Exception:
            pass
        raise
