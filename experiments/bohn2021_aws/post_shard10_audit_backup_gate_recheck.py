#!/usr/bin/env python3
"""Metadata-only backup gate recheck after vehicle validation64 shard10 audit.

This script is a storage/recoverability gate, not a scientific evaluation.  It
performs no simulation, no training, no validation-bank reopen, and no sealed
final-test open/hash.  It inspects repository-local backup proof metadata to
verify whether shard11 may start after the shard10 formal run and shard10 audit
have both finalized.  If no adequate proof is present, it writes a durable
backup request and documentation entries that keep shard11 blocked.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = REPO_ROOT / "research_artifacts" / "aws_backup_proofs"
DIAG_ROOT = REPO_ROOT / "research_artifacts" / "aws_diagnostics"

# The audit experiment ended at 2026-09-27T05:06:06.655706Z and the
# run-finalized addendum existed after the run registry/stdout/stderr/cloudwatch
# snapshot were available.  Require a proof strictly after this conservative
# post-audit-finalization time before shard11 can create new formal validation
# evidence.
REQUIRED_AFTER_UTC = "2026-09-27T05:06:10+00:00"
RUNNER_PATH = "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
RUNNER_SHA256 = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
GATE_PATH = "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
GATE_SHA256 = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"

SHARD10_FORMAL_EXPERIMENT_ID = "20260927T034937_4f27027e"
SHARD10_AUDIT_EXPERIMENT_ID = "20260927T050550_d3f6c4be"
SHARD10_FORMAL_COMPLETED = "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/completed.json"
SHARD10_FORMAL_COMPLETED_SHA256 = "393764461f67e6ccd58929c0cf1a2df53ac1a2feb331542ee94bd199799b71e6"
SHARD10_AUDIT_COMPLETED = "research_artifacts/aws_diagnostics/vehicle_validation64_shard10_audit_20260927/completed.json"
SHARD10_AUDIT_COMPLETED_SHA256 = "1e231a2eb0f439d5c9b3447967f79542e8edf472687ea8eb0baf2106467e62c5"
SHARD10_AUDIT_RAW = "research_artifacts/aws_diagnostics/vehicle_validation64_shard10_audit_20260927/raw.json"
SHARD10_AUDIT_RAW_SHA256 = "d6bb5f19b5627e6c74a9f84e08cdd7d81a9fa0e43d98218ac0e171147f9e6e5a"
SHARD10_AUDIT_SUMMARY = "research_artifacts/aws_diagnostics/vehicle_validation64_shard10_audit_20260927/summary.md"
SHARD10_AUDIT_SUMMARY_SHA256 = "e4ee77e84e2de52f4864722bf1e1c1aad2f5fe5a9c3dd5b21b8fd52e5d134a38"
SHARD10_BACKUP_REQUEST = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_20260927T050550.json"
SHARD10_BACKUP_REQUEST_SHA256 = "e6179c2e52d2af844036582ae8d51088ebad08928bade0baa6b5a1a1269f7f71"
SHARD10_FINAL_ADDENDUM = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_FINAL_ADDENDUM_20260927T050550.json"
SHARD10_FINAL_ADDENDUM_SHA256 = "9adbfd149d788a2fbec5a22c12b2ac962f45126825b74b0c9ca6eed261d1a381"
SHARD10_RUN_FINALIZED_ADDENDUM = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_RUN_FINALIZED_20260927T050550.json"
SHARD10_RUN_FINALIZED_ADDENDUM_SHA256 = "42d01c856d61601eac066bda0c04e1e9443e4616395e1c55474cc3e56ff39d8b"
SHARD10_BLOCKER_NOTE = "research_artifacts/aws_diagnostics/post_shard10_audit_backup_blocker_check_20260927.md"
SHARD10_BLOCKER_NOTE_SHA256 = "9772d2845c42dfff67e6d4b692bbed624145bc383d1171f0b9719e8cd72ecd56"

LATEST_INADEQUATE_SUPERVISOR_BACKUP = {
    "time": "2026-09-27T05:05:16.430103+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "c0cae3a1b572236b3ee57f1b6490318c2d47422c",
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 85044,
    "packages_this_run": [
        {
            "name": "20260927T050448_885ffebb.tar.gz",
            "sha256": "3fbc2b30ecce6ba706df072f18e9ef23ed6e380c9b9be539498fd9d44c4e613a",
            "bytes": 93008248,
            "verification": "github_server_sha256",
        },
        {
            "name": "20260927T050502_89c9853a.tar.gz",
            "sha256": "19333a305de8018df70232611f8b0ac91b060b2e328c43ae2a7c17dc2b3f4f90",
            "bytes": 79248350,
            "verification": "github_server_sha256",
        },
    ],
    "why_inadequate_for_shard11": "This verified backup predates shard10 audit experiment 20260927T050550_d3f6c4be completion, its finalized logs, and the run-finalized addendum, so it cannot cover the completed shard10 audit state required before shard11.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_label(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "").replace("Z", "")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_maybe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def proof_time(data: Dict[str, Any]) -> Optional[str]:
    for key in ("proof_created_utc", "proof_recorded_utc", "time", "created_utc", "reported_time_utc"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def filename_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"backup_proof_(\d{8}T\d{6})", path.name)
    return m.group(1) if m else None


def coverage_flags(data: Dict[str, Any]) -> Dict[str, bool]:
    text = json.dumps(data, sort_keys=True)
    return {
        "runner_sha_match": RUNNER_SHA256 in text,
        "gate_sha_match": GATE_SHA256 in text,
        "github_sha_verification": "github_server_sha256" in text or "download_sha256" in text or "verified" in text,
        "mentions_shard10_formal": "shard10" in text and "formal" in text,
        "mentions_shard10_formal_completed": SHARD10_FORMAL_COMPLETED in text or SHARD10_FORMAL_COMPLETED_SHA256 in text,
        "mentions_shard10_audit": "shard10" in text and "audit" in text,
        "mentions_shard10_audit_completed": SHARD10_AUDIT_COMPLETED in text or SHARD10_AUDIT_COMPLETED_SHA256 in text,
        "mentions_shard10_audit_run_registry": SHARD10_AUDIT_EXPERIMENT_ID in text and "registry" in text,
        "mentions_shard10_formal_run_registry": SHARD10_FORMAL_EXPERIMENT_ID in text and "registry" in text,
        "mentions_run_finalized_addendum": SHARD10_RUN_FINALIZED_ADDENDUM in text or SHARD10_RUN_FINALIZED_ADDENDUM_SHA256 in text or "run-finalized" in text or "run_finalized" in text,
        "mentions_docs_registry": "EXPERIMENT_REGISTRY.csv" in text and "STATUS.md" in text,
    }


def inspect_proofs() -> Dict[str, Any]:
    required_dt = parse_time(REQUIRED_AFTER_UTC)
    all_records: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []
    adequate_records: List[Dict[str, Any]] = []

    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json_maybe(path)
        record: Dict[str, Any] = {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "json_ok": data is not None,
            "filename_timestamp": filename_timestamp(path),
        }
        if data is not None:
            t_str = proof_time(data)
            t_dt = parse_time(t_str)
            flags = coverage_flags(data)
            after_required = bool(required_dt and t_dt and t_dt > required_dt)
            filename_late = bool(record["filename_timestamp"] and record["filename_timestamp"] >= "20260927T050610")
            is_candidate = after_required or filename_late
            is_adequate = bool(
                data.get("backup_verified") is True
                and data.get("remaining_changed_files") == 0
                and after_required
                and flags["runner_sha_match"]
                and flags["gate_sha_match"]
                and flags["github_sha_verification"]
                and flags["mentions_shard10_formal"]
                and flags["mentions_shard10_audit"]
                and flags["mentions_run_finalized_addendum"]
                and flags["mentions_shard10_audit_run_registry"]
            )
            record.update({
                "backup_verified": data.get("backup_verified"),
                "remaining_changed_files": data.get("remaining_changed_files"),
                "proof_time": t_str,
                "after_required_time": after_required,
                "candidate_post_shard10_time_or_filename": is_candidate,
                "coverage_flags": flags,
                "adequate_for_pre_shard11": is_adequate,
            })
            if is_candidate:
                candidate_records.append(record)
            if is_adequate:
                adequate_records.append(record)
        all_records.append(record)

    glob_05 = sorted(BACKUP_DIR.glob("backup_proof_20260927T05*.json"))
    glob_20260927 = sorted(BACKUP_DIR.glob("backup_proof_20260927T*.json"))
    return {
        "required_after_utc": REQUIRED_AFTER_UTC,
        "all_backup_proof_count": len(all_records),
        "all_backup_proofs": all_records,
        "backup_proof_20260927T05_glob": [rel(p) for p in glob_05],
        "backup_proof_20260927T05_count": len(glob_05),
        "backup_proof_20260927T_any_glob": [rel(p) for p in glob_20260927],
        "post_required_candidate_count": len(candidate_records),
        "post_required_candidates": candidate_records,
        "adequate_backup_proofs_found": len(adequate_records),
        "adequate_backup_proofs": adequate_records,
        "latest_local_backup_proof": all_records[-1] if all_records else None,
    }


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def key_hash_checks() -> List[Dict[str, Any]]:
    checks = []
    for name, expected in [
        (RUNNER_PATH, RUNNER_SHA256),
        (GATE_PATH, GATE_SHA256),
        (SHARD10_FORMAL_COMPLETED, SHARD10_FORMAL_COMPLETED_SHA256),
        (SHARD10_AUDIT_COMPLETED, SHARD10_AUDIT_COMPLETED_SHA256),
        (SHARD10_AUDIT_RAW, SHARD10_AUDIT_RAW_SHA256),
        (SHARD10_AUDIT_SUMMARY, SHARD10_AUDIT_SUMMARY_SHA256),
        (SHARD10_BACKUP_REQUEST, SHARD10_BACKUP_REQUEST_SHA256),
        (SHARD10_FINAL_ADDENDUM, SHARD10_FINAL_ADDENDUM_SHA256),
        (SHARD10_RUN_FINALIZED_ADDENDUM, SHARD10_RUN_FINALIZED_ADDENDUM_SHA256),
        (SHARD10_BLOCKER_NOTE, SHARD10_BLOCKER_NOTE_SHA256),
    ]:
        path = REPO_ROOT / name
        actual = sha256_file(path)
        checks.append({
            "path": name,
            "exists": path.exists(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        })
    for name in [
        f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/registry.json",
        f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/stdout.log",
        f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/stderr.log",
        f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/cloudwatch_snapshot.json",
        f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/registry.json",
        f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/stdout.log",
        f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/stderr.log",
        f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/cloudwatch_snapshot.json",
    ]:
        path = REPO_ROOT / name
        checks.append({"path": name, "exists": path.exists(), "sha256": sha256_file(path)})
    return checks


def main() -> int:
    created_utc = utc_now()
    label = safe_label(created_utc)
    diag_dir = DIAG_ROOT / f"post_shard10_audit_backup_gate_recheck_{label}"
    diag_dir.mkdir(parents=True, exist_ok=False)

    proof_inspection = inspect_proofs()
    checks = key_hash_checks()
    missing_or_mismatch = [c for c in checks if (c.get("exists") is False) or ("matches" in c and c.get("matches") is not True)]
    shard11_blocked = proof_inspection["adequate_backup_proofs_found"] == 0

    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_POST_SHARD10_AUDIT_GATE_RECHECK_{label}.json"
    request = {
        "created_utc": created_utc,
        "status": "external_backup_requested_after_post_shard10_audit_gate_recheck_no_adequate_proof" if shard11_blocked else "adequate_backup_found_by_gate_recheck",
        "backup_verified": False,
        "required_before_shard11": shard11_blocked,
        "validation_accessed_by_this_recheck": False,
        "validation_bank_reopened_by_this_recheck": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations_by_this_recheck": 0,
        "new_control_steps_by_this_recheck": 0,
        "new_gradient_steps_by_this_recheck": 0,
        "reason": "No adequate repository-local verified external backup proof after shard10 audit run-finalized state was found. Shard11 remains blocked until backup covers shard10 formal/audit artifacts, finalized formal/audit run logs, docs/registry updates, backup requests/addenda/blocker, this gate recheck, this request, and this metadata run's registry/stdout/stderr after finalization.",
        "required_after_utc_prior_gate": REQUIRED_AFTER_UTC,
        "latest_inadequate_supervisor_backup_from_context": LATEST_INADEQUATE_SUPERVISOR_BACKUP,
        "runner_sha256": RUNNER_SHA256,
        "gate_sha256": GATE_SHA256,
        "must_cover_before_shard11": [
            SHARD10_FORMAL_COMPLETED,
            "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/raw.json",
            "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/summary.md",
            "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/episodes/",
            f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/registry.json",
            f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/stdout.log",
            f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/stderr.log",
            f"research_artifacts/aws_runs/{SHARD10_FORMAL_EXPERIMENT_ID}/cloudwatch_snapshot.json",
            SHARD10_AUDIT_COMPLETED,
            SHARD10_AUDIT_RAW,
            SHARD10_AUDIT_SUMMARY,
            f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/registry.json",
            f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/stdout.log",
            f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/stderr.log",
            f"research_artifacts/aws_runs/{SHARD10_AUDIT_EXPERIMENT_ID}/cloudwatch_snapshot.json",
            SHARD10_BACKUP_REQUEST,
            SHARD10_FINAL_ADDENDUM,
            SHARD10_RUN_FINALIZED_ADDENDUM,
            SHARD10_BLOCKER_NOTE,
            str(request_path.relative_to(REPO_ROOT)),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
            "this metadata gate recheck's raw/summary/completed artifacts",
            "this metadata gate recheck run registry/stdout/stderr/cloudwatch snapshot after run_experiment finalizes",
        ],
        "next_action_after_verified_backup": {
            "script": RUNNER_PATH,
            "args": ["--shard", "11", "--backup-proof", "<post-shard10-audit-run-finalized-proof.json>", "--i-accept-validation-access"],
            "interpreter": "legacy",
            "method": "IMPROVED_latency_tree_vehicle_validation64_formal_shard11_not_original_SAC",
            "split": "validation64_shard11_224_episodes_no_test",
            "training_gradient_steps": 0,
            "validation_episodes": 224,
            "validation_control_step_upper_bound": 33600,
            "test_accessed": False,
            "sealed_test_bank_content_opened": False,
            "sealed_test_bank_hashed": False,
        },
    }
    write_json(request_path, request)

    raw = {
        "created_utc": created_utc,
        "purpose": "Metadata-only verification of post-shard10-audit backup gate before shard11.",
        "method_classification": "IMPROVED latency-tree vehicle validation, not ORIGINAL SAC",
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "proof_inspection": proof_inspection,
        "latest_inadequate_supervisor_backup_from_context": LATEST_INADEQUATE_SUPERVISOR_BACKUP,
        "key_hash_checks": checks,
        "key_hash_missing_or_mismatch": missing_or_mismatch,
        "shard11_blocked": shard11_blocked,
        "request_path": rel(request_path),
        "cumulative_vehicle_validation64_status_unchanged": {
            "formal_shards_completed": list(range(11)),
            "accepted_audited_shards": list(range(11)),
            "formal_episodes_completed": 2464,
            "formal_control_steps_completed": 217137,
            "planned_total_shards": 12,
            "remaining_formal_shards": [11],
            "failed_formal_attempts": [
                "20260926T161356_fb71c8d7 shard02 modern-interpreter TensorFlow-missing attempt: validation bank opened, 0 episodes/control steps/gradient steps; archived/preserved"
            ],
            "failed_audit_attempts_preserved": [
                "20260926T144037_523e1646 shard00 audit-v1 registry-schema false negative, repaired by v2",
                "20260926T215916_c2ecb9ed shard05 audit-v1 schema false positive, repaired by v2",
                "20260927T005319_5d1112ca shard07 audit-v1 validation_budget.episodes_exact schema false positive, repaired by v2",
            ],
        },
    }
    raw_path = diag_dir / "raw.json"
    write_json(raw_path, raw)

    summary_lines = [
        "# Post-shard10 audit backup gate recheck",
        "",
        f"UTC: `{created_utc}`.",
        "",
        "This was a metadata-only backup proof inventory check before vehicle validation64 shard11. It ran no simulations, opened no validation64 bank content, and did not open or hash the sealed final test.",
        "",
        "## Result",
        "",
        f"- Required proof time: after `{REQUIRED_AFTER_UTC}`.",
        f"- Repository-local `backup_proof_20260927T05*.json` count: `{proof_inspection['backup_proof_20260927T05_count']}`.",
        f"- Repository-local `backup_proof_20260927T*.json` files: `{proof_inspection['backup_proof_20260927T_any_glob']}`.",
        f"- Adequate post-shard10 backup proofs found: `{proof_inspection['adequate_backup_proofs_found']}`.",
        f"- Latest known supervisor backup from context: `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}`; inadequate because it predates shard10 audit completion/finalization.",
        f"- Shard11 blocked: `{shard11_blocked}`.",
        "",
        "No validation or formal scientific evidence was created by this gate recheck. Vehicle validation64 remains at 11/12 completed and audited shards (2464/2688 planned validation episodes; 217137 formal control steps). This is validation/model-selection evidence only for the IMPROVED latency-tree vehicle protocol, not ORIGINAL SAC and not final-test evidence.",
        "",
        "## Key integrity checks",
        "",
    ]
    for check in checks:
        if "matches" in check:
            summary_lines.append(f"- `{check['path']}` exists={check['exists']} matches_expected_sha256={check['matches']}")
        else:
            summary_lines.append(f"- `{check['path']}` exists={check['exists']} sha256={check.get('sha256')}")
    summary_lines.extend([
        "",
        "## Backup request",
        "",
        f"- `{rel(request_path)}`",
        "",
        "The next adequate backup proof must be after this recheck and after this run_experiment registry/stdout/stderr/cloudwatch snapshot finalizes, with `backup_verified=true`, `remaining_changed_files=0`, GitHub release asset/download SHA256 verification, runner SHA and gate SHA, and coverage of shard10 formal/audit artifacts, docs/registry updates, prior shard10 backup requests/addenda/blocker, this recheck, this request, and this metadata run's logs.",
        "",
        "## Next action if and only if the backup gate is satisfied",
        "",
        "Run exactly one formal validation shard with the legacy interpreter:",
        "",
        "```text",
        "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 11 --backup-proof <post-shard10-audit-run-finalized-proof.json> --i-accept-validation-access",
        "```",
        "",
        "Budget: 224 validation episodes, control-step upper bound 33,600, 0 gradient steps; sealed final test closed and not hashed.",
    ])
    summary_path = diag_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    completed = {
        "created_utc": created_utc,
        "passed": not missing_or_mismatch,
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "shard11_blocked": shard11_blocked,
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "raw_path": rel(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "summary_path": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "backup_request": rel(request_path),
        "backup_request_sha256": sha256_file(request_path),
        "next_required_backup_after_utc": created_utc,
        "key_hash_missing_or_mismatch_count": len(missing_or_mismatch),
    }
    completed_path = diag_dir / "completed.json"
    write_json(completed_path, completed)
    completed["completed_path"] = rel(completed_path)
    completed["completed_sha256"] = sha256_file(completed_path)
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256_file(completed_path)

    marker = f"<!-- post-shard10-audit-backup-gate-recheck-{label} -->"
    doc_section = f"""
{marker}
## 2026-09-27 post-shard10 audit backup gate recheck

UTC: {created_utc}. Metadata-only backup proof inventory check before shard11: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. Required proof was after `{REQUIRED_AFTER_UTC}`. Repository-local `backup_proof_20260927T05*.json` count was {proof_inspection['backup_proof_20260927T05_count']}; all `backup_proof_20260927T*.json` files were {proof_inspection['backup_proof_20260927T_any_glob']}; adequate post-shard10 proofs found: {proof_inspection['adequate_backup_proofs_found']}. The supervisor-context backup at `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}` is insufficient because it predates shard10 audit completion/finalization. Shard11 remains backup-gated. No formal validation evidence was created; vehicle validation64 remains 11/12 shards, 2464/2688 episodes, 217137 control steps, final test closed. New artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(completed_path)}`. New backup request: `{rel(request_path)}`. Next proof must be after this recheck and this metadata run's finalized registry/stdout/stderr/cloudwatch snapshot before shard11 can run. Method remains IMPROVED latency-tree, not ORIGINAL SAC, and no reproduction-success claim is supported.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(REPO_ROOT / name, marker, doc_section)

    print(json.dumps({
        "completed": rel(completed_path),
        "completed_sha256": completed["completed_sha256"],
        "raw": rel(raw_path),
        "raw_sha256": completed["raw_sha256"],
        "summary": rel(summary_path),
        "summary_sha256": completed["summary_sha256"],
        "backup_request": rel(request_path),
        "backup_request_sha256": completed["backup_request_sha256"],
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "shard11_blocked": shard11_blocked,
        "key_hash_missing_or_mismatch_count": len(missing_or_mismatch),
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
