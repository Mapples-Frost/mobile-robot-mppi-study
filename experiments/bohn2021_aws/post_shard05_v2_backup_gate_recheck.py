#!/usr/bin/env python3
"""Metadata-only backup-gate recheck after vehicle validation64 shard05 v2 audit.

This script deliberately performs no simulations/training, does not open the
vehicle validation bank, and does not open/hash sealed test content.  It only
inspects repository-local backup-proof/request metadata and persists the blocker
state required before shard06 can be run.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = REPO_ROOT / "research_artifacts" / "aws_backup_proofs"
DIAG_ROOT = REPO_ROOT / "research_artifacts" / "aws_diagnostics"
REQUIRED_AFTER_UTC = "2026-09-26T22:05:39.518925+00:00"
RUNNER_PATH = "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
RUNNER_SHA256 = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
GATE_PATH = "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
GATE_SHA256 = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
SHARD05_V2_COMPLETED = "research_artifacts/aws_diagnostics/vehicle_validation64_shard05_audit_v2_20260926/completed.json"
SHARD05_V2_COMPLETED_SHA256 = "d9d2211a5f6107919ed328f2d2a52b974b4ba808c7647dade24e554c7dc22c5e"
BLOCKER_STATE_REQUEST = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_BLOCKER_STATE_AFTER_RUN_FINALIZED_20260926T220539.json"
BLOCKER_STATE_REQUEST_SHA256 = "34172bf6e0ecb446652efd262e97d3b112d26aa2b5eed21bd59fc8b3c84c6898"
PREVIOUS_GATE_STATE = "research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_state_20260926T220539.md"
PREVIOUS_GATE_STATE_SHA256 = "e4c0e80ee7b91c692aa583b76fe40527a1355ebf287a9c41dbda6d69ec7b295e"
LATEST_INADEQUATE_SUPERVISOR_BACKUP = {
    "time": "2026-09-26T22:04:47.633765+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "d3143085da50deba7950c2354ba3a92a878e0c5a",
    "asset": "20260926T220446_9eee5a47.tar.gz",
    "asset_sha256": "02b026bf42b2a88f028b0e41016215f9647a89c11f079ee26eeb376ba5980e81",
    "why_inadequate": "Predates shard05 v2 audit run at 22:05:23Z, v2 outputs, finalized run registry/stdout/stderr, blocker-state request, and the later backup-gate diagnostic.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_label(ts: str) -> str:
    # 2026-09-26T22:10:11+00:00 -> 20260926T221011Z
    return ts.replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_maybe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def proof_time(data: Dict[str, Any]) -> Optional[str]:
    for key in ("proof_created_utc", "proof_recorded_utc", "time", "created_utc"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def proof_mentions_required_coverage(data: Dict[str, Any]) -> Dict[str, bool]:
    text = json.dumps(data, sort_keys=True)
    return {
        "mentions_shard05_formal": "shard05" in text and "formal" in text,
        "mentions_failed_v1_audit": "vehicle_validation64_shard05_audit_20260926" in text or "failed v1" in text,
        "mentions_passed_v2_audit": "vehicle_validation64_shard05_audit_v2_20260926" in text or "shard05 v2" in text,
        "mentions_v2_run_registry": "20260926T220523_31f1aba6" in text and "registry" in text,
        "mentions_blocker_state_request": "REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_BLOCKER_STATE_AFTER_RUN_FINALIZED_20260926T220539" in text,
        "runner_sha_match": RUNNER_SHA256 in text,
        "gate_sha_match": GATE_SHA256 in text,
        "github_sha_verification": "github_server_sha256" in text or "download_sha256" in text,
    }


def inspect_proofs() -> Dict[str, Any]:
    required_dt = parse_time(REQUIRED_AFTER_UTC)
    proof_records: List[Dict[str, Any]] = []
    adequate: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json_maybe(path)
        record: Dict[str, Any] = {"path": str(path.relative_to(REPO_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "json_ok": data is not None}
        if data is not None:
            t_str = proof_time(data)
            t_dt = parse_time(t_str)
            coverage = proof_mentions_required_coverage(data)
            record.update({
                "backup_verified": data.get("backup_verified"),
                "remaining_changed_files": data.get("remaining_changed_files"),
                "proof_time": t_str,
                "after_required_time": bool(required_dt and t_dt and t_dt > required_dt),
                "coverage_flags": coverage,
            })
            is_adequate = bool(
                data.get("backup_verified") is True
                and data.get("remaining_changed_files") == 0
                and required_dt
                and t_dt
                and t_dt > required_dt
                and coverage.get("runner_sha_match")
                and coverage.get("gate_sha_match")
                and coverage.get("github_sha_verification")
                and coverage.get("mentions_passed_v2_audit")
                and coverage.get("mentions_blocker_state_request")
            )
            record["adequate_for_pre_recheck_shard06"] = is_adequate
            if is_adequate:
                adequate.append(record)
        proof_records.append(record)
    return {
        "required_after_utc_before_this_recheck": REQUIRED_AFTER_UTC,
        "proof_count": len(proof_records),
        "proof_records": proof_records,
        "adequate_backup_proofs_found": len(adequate),
        "adequate_backup_proofs": adequate,
        "backup_proof_20260926T22_glob_count": len(list(BACKUP_DIR.glob("backup_proof_20260926T22*.json"))),
        "latest_local_backup_proof": proof_records[-1] if proof_records else None,
    }


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    created_utc = utc_now()
    label = safe_label(created_utc)
    short_label = label.replace("Z", "")
    diag_dir = DIAG_ROOT / f"post_shard05_v2_backup_gate_recheck_{short_label}"
    diag_dir.mkdir(parents=True, exist_ok=False)

    proof_inspection = inspect_proofs()
    blocked = proof_inspection["adequate_backup_proofs_found"] == 0

    # Verify key metadata files by hash without reading validation/test banks.
    key_hash_checks = []
    for rel, expected in [
        (SHARD05_V2_COMPLETED, SHARD05_V2_COMPLETED_SHA256),
        (BLOCKER_STATE_REQUEST, BLOCKER_STATE_REQUEST_SHA256),
        (PREVIOUS_GATE_STATE, PREVIOUS_GATE_STATE_SHA256),
    ]:
        p = REPO_ROOT / rel
        actual = sha256_file(p) if p.exists() else None
        key_hash_checks.append({"path": rel, "exists": p.exists(), "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected})

    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_SHARD05_V2_GATE_RECHECK_{short_label}.json"
    request = {
        "status": "external_backup_requested_after_shard05_v2_backup_gate_recheck",
        "created_utc": created_utc,
        "backup_verified": False,
        "required_before_shard6": True,
        "validation_accessed_by_this_recheck": False,
        "validation_bank_reopened_by_this_recheck": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations_by_this_recheck": 0,
        "new_control_steps_by_this_recheck": 0,
        "new_gradient_steps_by_this_recheck": 0,
        "reason": "Repository-local proof inventory still contains no adequate verified external backup after shard05 v2 audit finalization/blocker state. This request asks the supervisor backup to cover the new recheck artifacts and the run_experiment registry/stdout/stderr once finalized before shard06 is allowed.",
        "required_next_backup_proof_fields_before_shard6": {
            "backup_verified": True,
            "remaining_changed_files": 0,
            "proof_time_after_utc": created_utc,
            "github_release_asset_download_sha256_verified": True,
            "runner_sha256": RUNNER_SHA256,
            "gate_sha256": GATE_SHA256,
            "must_cover_validation64_shard05_formal_outputs": True,
            "must_cover_failed_shard05_v1_audit_outputs": True,
            "must_cover_passed_shard05_v2_audit_outputs": True,
            "must_cover_shard05_v2_run_registry_stdout_stderr": True,
            "must_cover_docs_and_experiment_registry_updates": True,
            "must_cover_all_shard05_v2_backup_requests_addenda_blocker_notes": True,
            "must_cover_prior_blocker_state_request": BLOCKER_STATE_REQUEST,
            "must_cover_this_gate_recheck_artifacts": True,
            "must_cover_this_request": str(request_path.relative_to(REPO_ROOT)),
            "must_cover_this_run_registry_stdout_stderr_after_run_experiment_finalizes": True,
        },
        "next_action_after_verified_backup": {
            "script": RUNNER_PATH,
            "args": ["--shard", "6", "--backup-proof", "<post-shard05-v2-gate-recheck-proof.json>", "--i-accept-validation-access"],
            "interpreter": "legacy",
            "method": "IMPROVED_latency_tree_vehicle_validation64_formal_shard06_not_original_SAC",
            "split": "validation64_shard6_224_episodes_no_test",
            "training_gradient_steps": 0,
            "validation_episodes": 224,
            "validation_control_step_upper_bound": 33600,
            "test_accessed": False,
        },
    }
    write_json(request_path, request)

    raw = {
        "created_utc": created_utc,
        "purpose": "Metadata-only backup proof inventory recheck after shard05 v2 audit; persist blocker before shard06.",
        "method_classification": "IMPROVED latency-tree vehicle validation, not ORIGINAL SAC",
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "required_after_utc_before_this_recheck": REQUIRED_AFTER_UTC,
        "blocked_before_shard06": blocked,
        "proof_inspection": proof_inspection,
        "latest_inadequate_supervisor_backup_from_context": LATEST_INADEQUATE_SUPERVISOR_BACKUP,
        "key_hash_checks": key_hash_checks,
        "request_path": str(request_path.relative_to(REPO_ROOT)),
        "formal_validation64_progress_before_recheck": {
            "formal_shards_completed": [0, 1, 2, 3, 4, 5],
            "audited_shards_passed": [0, 1, 2, 3, 4, 5],
            "formal_episodes_completed": 1344,
            "formal_control_steps_completed": 119058,
            "failed_attempts_to_report": [
                "20260926T161356_fb71c8d7 shard02 modern-interpreter TensorFlow-missing attempt: validation bank opened, 0 episodes/control steps/gradient steps",
                "20260926T144037_523e1646 shard00 audit-v1 registry-schema false negative, repaired by v2",
                "20260926T215916_c2ecb9ed shard05 audit-v1 schema false positive, repaired by v2",
            ],
        },
    }
    raw_path = diag_dir / "raw.json"
    write_json(raw_path, raw)

    summary_lines = [
        "# Post-shard05 v2 backup gate recheck",
        "",
        f"UTC: {created_utc}.",
        "",
        "Purpose: metadata-only recheck of repository-local backup proofs before any shard06 validation access. No simulation, training, validation-bank reopen, sealed-test open, or sealed-test hash occurred.",
        "",
        "## Result",
        "",
        f"- Adequate verified post-shard05-v2 backup proofs found: `{proof_inspection['adequate_backup_proofs_found']}`.",
        f"- `backup_proof_20260926T22*.json` count: `{proof_inspection['backup_proof_20260926T22_glob_count']}`.",
        f"- Latest local backup proof: `{proof_inspection['latest_local_backup_proof']['path'] if proof_inspection['latest_local_backup_proof'] else 'none'}`.",
        f"- Latest supervisor backup in context remains inadequate: `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}` predates shard05 v2 audit/blocker finalization.",
        "- Key shard05-v2 gate files still match their expected SHA256 values: " + str(all(x["matches"] for x in key_hash_checks)) + ".",
        "",
        "Shard06 remains **blocked**. The sealed final test remains closed and unauthorized. Validation64 evidence remains partial development/model-selection evidence only for an IMPROVED latency-tree method, not ORIGINAL SAC.",
        "",
        "## New backup request",
        "",
        f"- `{request_path.relative_to(REPO_ROOT)}`",
        "",
        "The next backup proof must be after this recheck and after the run_experiment registry/stdout/stderr finalize, with `backup_verified=true`, `remaining_changed_files=0`, GitHub release asset/download SHA256 verification, runner SHA `" + RUNNER_SHA256 + "`, gate SHA `" + GATE_SHA256 + "`, and coverage of shard05 formal outputs, failed v1 audit, passed v2 audit, v2 run logs, docs/registry, all shard05-v2 backup requests/addenda/blocker notes, this recheck, this request, and this metadata run's registry/stdout/stderr.",
        "",
        "## Next action if and only if the backup gate is satisfied",
        "",
        "Run exactly one formal experiment with the legacy interpreter:",
        "",
        "```text",
        "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 6 --backup-proof <post-shard05-v2-gate-recheck-proof.json> --i-accept-validation-access",
        "```",
        "",
        "Budget: 224 validation episodes, control-step upper bound 33,600, 0 training gradient steps, sealed final test closed.",
    ]
    summary_path = diag_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    completed = {
        "created_utc": created_utc,
        "passed": True,
        "blocked_before_shard06": blocked,
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "raw_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_sha256": sha256_file(raw_path),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "summary_sha256": sha256_file(summary_path),
        "backup_request": str(request_path.relative_to(REPO_ROOT)),
        "backup_request_sha256": sha256_file(request_path),
        "next_required_backup_after_utc": created_utc,
    }
    completed_path = diag_dir / "completed.json"
    write_json(completed_path, completed)
    completed["completed_path"] = str(completed_path.relative_to(REPO_ROOT))
    completed["completed_sha256"] = sha256_file(completed_path)
    # Rewrite so the completed record contains its own path/hash provenance.  The
    # final digest is recomputed below and reported in stdout.
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256_file(completed_path)

    marker = f"<!-- post-shard05-v2-backup-gate-recheck-{short_label} -->"
    common_section = f"""
{marker}
## 2026-09-26 post-shard05 v2 backup gate recheck

UTC: {created_utc}. Metadata-only backup inventory recheck before shard06: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. No adequate repository-local verified external backup proof after shard05 v2 audit finalization was found (`backup_proof_20260926T22*.json` count {proof_inspection['backup_proof_20260926T22_glob_count']}; adequate proofs {proof_inspection['adequate_backup_proofs_found']}). Latest supervisor backup in context remains `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}`, which predates shard05 v2 audit/blocker finalization and is insufficient. Shard06 remains backup-gated. New recheck artifacts: `{raw_path.relative_to(REPO_ROOT)}`, `{summary_path.relative_to(REPO_ROOT)}`, `{completed_path.relative_to(REPO_ROOT)}`. New backup request: `{request_path.relative_to(REPO_ROOT)}`. Next proof must be after this recheck and after the metadata run registry/stdout/stderr finalize, and must cover shard05 formal outputs, failed v1 audit, passed v2 audit, docs/registry, all shard05-v2 backup requests/addenda/blocker notes, this recheck/request, runner SHA `{RUNNER_SHA256}`, and gate SHA `{GATE_SHA256}`. No reproduction or final-test conclusion is permitted; method remains IMPROVED, not ORIGINAL SAC.
"""
    for rel in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(REPO_ROOT / rel, marker, common_section)

    print(json.dumps({
        "completed": str(completed_path.relative_to(REPO_ROOT)),
        "completed_sha256": completed["completed_sha256"],
        "raw": str(raw_path.relative_to(REPO_ROOT)),
        "raw_sha256": completed["raw_sha256"],
        "summary": str(summary_path.relative_to(REPO_ROOT)),
        "summary_sha256": completed["summary_sha256"],
        "backup_request": str(request_path.relative_to(REPO_ROOT)),
        "backup_request_sha256": completed["backup_request_sha256"],
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "blocked_before_shard06": blocked,
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
