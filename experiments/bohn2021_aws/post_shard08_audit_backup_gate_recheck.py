#!/usr/bin/env python3
"""Metadata-only backup gate recheck after vehicle validation64 shard08 audit.

This script is intentionally conservative: it does not import or execute the
vehicle simulator, does not open the validation64 scenario bank, and does not
open/hash sealed final-test content.  It only inspects repository-local backup
proof metadata to decide whether shard09 may start, and persists the backup gate
state when the required proof is absent.
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

REQUIRED_AFTER_UTC = "2026-09-27T02:15:35+00:00"
RUNNER_PATH = "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
RUNNER_SHA256 = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
GATE_PATH = "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
GATE_SHA256 = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"

SHARD08_FORMAL_COMPLETED = "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/completed.json"
SHARD08_FORMAL_COMPLETED_SHA256 = "c77e51254a0bd3e0a265e25928733b228340bf8f8815c3ba39a19387ec058672"
SHARD08_AUDIT_COMPLETED = "research_artifacts/aws_diagnostics/vehicle_validation64_shard08_audit_20260927/completed.json"
SHARD08_AUDIT_COMPLETED_SHA256 = "4c8cfbe5a6f68c1e4540c66dfd7e328e6222be0bde71162eec6e82c7aa4eb5a9"
SHARD08_RUN_FINALIZED_ADDENDUM = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_RUN_FINALIZED_20260927T021520.json"
SHARD08_RUN_FINALIZED_ADDENDUM_SHA256 = "286e5821b8e55e69295a8f2660f2ebc35f3d55703a4465afc184e6ce1d553ec6"
MANUAL_NO_PROOF_NOTE = "research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T0219_no_proof.md"
MANUAL_NO_PROOF_NOTE_SHA256 = "59ddea6e8a2a67360073c2f18b98675774c0a178c861c0efaf6b2159bdb30ec4"

LATEST_INADEQUATE_SUPERVISOR_BACKUP = {
    "time": "2026-09-27T02:14:50.580286+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "39f2fcef46464acefed66f0aa98029ea045638f6",
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "packages_this_run": [
        {
            "name": "20260927T021426_649e2ada.tar.gz",
            "sha256": "68dbbf7a684b1e0e16607b16bd157b54323e9b837fea823c25bbca496a9f23f2",
            "verification": "github_server_sha256",
        },
        {
            "name": "20260927T021439_b3ffdac0.tar.gz",
            "sha256": "3a8a6c7f3dfd1e492a32b4be58cfd68fdf17a0418bd643a0de43d03819c9dd74",
            "verification": "github_server_sha256",
        },
    ],
    "why_inadequate": "Predates required post-shard08-audit run-finalized threshold 2026-09-27T02:15:35Z and no corresponding repository-local proof exists.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_label(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "Z")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
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
        "mentions_shard08_formal": "shard08" in text and "formal" in text,
        "mentions_shard08_audit": "shard08" in text and "audit" in text,
        "mentions_shard08_audit_completed": SHARD08_AUDIT_COMPLETED in text,
        "mentions_run_finalized_addendum": SHARD08_RUN_FINALIZED_ADDENDUM in text or "run-finalized" in text or "run_finalized" in text,
        "mentions_audit_run_registry": "20260927T021520_6b245ecf" in text and "registry" in text,
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
            "path": str(path.relative_to(REPO_ROOT)),
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
            filename_late = bool(record["filename_timestamp"] and record["filename_timestamp"] >= "20260927T021535")
            is_candidate = after_required or filename_late
            is_adequate = bool(
                data.get("backup_verified") is True
                and data.get("remaining_changed_files") == 0
                and after_required
                and flags["runner_sha_match"]
                and flags["gate_sha_match"]
                and flags["github_sha_verification"]
                and flags["mentions_shard08_formal"]
                and flags["mentions_shard08_audit"]
                and flags["mentions_run_finalized_addendum"]
                and flags["mentions_audit_run_registry"]
            )
            record.update({
                "backup_verified": data.get("backup_verified"),
                "remaining_changed_files": data.get("remaining_changed_files"),
                "proof_time": t_str,
                "after_required_time": after_required,
                "candidate_post_shard08_time_or_filename": is_candidate,
                "coverage_flags": flags,
                "adequate_for_pre_shard09": is_adequate,
            })
            if is_candidate:
                candidate_records.append(record)
            if is_adequate:
                adequate_records.append(record)
        all_records.append(record)

    glob_02 = sorted(BACKUP_DIR.glob("backup_proof_20260927T02*.json"))
    glob_20260927 = sorted(BACKUP_DIR.glob("backup_proof_20260927T*.json"))
    return {
        "required_after_utc": REQUIRED_AFTER_UTC,
        "all_backup_proof_count": len(all_records),
        "all_backup_proofs": all_records,
        "backup_proof_20260927T02_glob": [str(p.relative_to(REPO_ROOT)) for p in glob_02],
        "backup_proof_20260927T02_count": len(glob_02),
        "backup_proof_20260927T_any_glob": [str(p.relative_to(REPO_ROOT)) for p in glob_20260927],
        "post_required_candidate_count": len(candidate_records),
        "post_required_candidates": candidate_records,
        "adequate_backup_proofs_found": len(adequate_records),
        "adequate_backup_proofs": adequate_records,
        "latest_local_backup_proof": all_records[-1] if all_records else None,
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def main() -> int:
    created_utc = utc_now()
    label = safe_label(created_utc).replace("Z", "")
    diag_dir = DIAG_ROOT / f"post_shard08_audit_backup_gate_recheck_{label}"
    diag_dir.mkdir(parents=True, exist_ok=False)

    proof_inspection = inspect_proofs()
    shard09_blocked = proof_inspection["adequate_backup_proofs_found"] == 0

    key_hash_checks = []
    for rel, expected in [
        (SHARD08_FORMAL_COMPLETED, SHARD08_FORMAL_COMPLETED_SHA256),
        (SHARD08_AUDIT_COMPLETED, SHARD08_AUDIT_COMPLETED_SHA256),
        (SHARD08_RUN_FINALIZED_ADDENDUM, SHARD08_RUN_FINALIZED_ADDENDUM_SHA256),
        (MANUAL_NO_PROOF_NOTE, MANUAL_NO_PROOF_NOTE_SHA256),
    ]:
        path = REPO_ROOT / rel
        actual = sha256_file(path)
        key_hash_checks.append({
            "path": rel,
            "exists": path.exists(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        })

    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_POST_SHARD08_AUDIT_GATE_RECHECK_{label}.json"
    request = {
        "created_utc": created_utc,
        "status": "external_backup_requested_after_post_shard08_audit_gate_recheck_no_adequate_proof",
        "backup_verified": False,
        "required_before_shard9": True,
        "validation_accessed_by_this_recheck": False,
        "validation_bank_reopened_by_this_recheck": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations_by_this_recheck": 0,
        "new_control_steps_by_this_recheck": 0,
        "new_gradient_steps_by_this_recheck": 0,
        "reason": "No adequate repository-local verified external backup proof after shard08 audit run-finalized state was found. Shard09 remains blocked until backup covers shard08 formal/audit artifacts, finalized run logs, docs/registry updates, prior run-finalized addendum, this gate recheck, this request, and this metadata run's registry/stdout/stderr after finalization.",
        "required_after_utc_prior_gate": REQUIRED_AFTER_UTC,
        "latest_inadequate_supervisor_backup_from_context": LATEST_INADEQUATE_SUPERVISOR_BACKUP,
        "runner_sha256": RUNNER_SHA256,
        "gate_sha256": GATE_SHA256,
        "must_cover_before_shard09": [
            SHARD08_FORMAL_COMPLETED,
            "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/raw.json",
            "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/summary.md",
            "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/episodes/",
            "research_artifacts/aws_runs/20260927T010021_f39456a9/registry.json",
            "research_artifacts/aws_runs/20260927T010021_f39456a9/stdout.log",
            "research_artifacts/aws_runs/20260927T010021_f39456a9/stderr.log",
            SHARD08_AUDIT_COMPLETED,
            "research_artifacts/aws_diagnostics/vehicle_validation64_shard08_audit_20260927/raw.json",
            "research_artifacts/aws_diagnostics/vehicle_validation64_shard08_audit_20260927/summary.md",
            "research_artifacts/aws_runs/20260927T021520_6b245ecf/registry.json",
            "research_artifacts/aws_runs/20260927T021520_6b245ecf/stdout.log",
            "research_artifacts/aws_runs/20260927T021520_6b245ecf/stderr.log",
            "research_artifacts/aws_runs/20260927T021520_6b245ecf/cloudwatch_snapshot.json",
            SHARD08_RUN_FINALIZED_ADDENDUM,
            MANUAL_NO_PROOF_NOTE,
            str(request_path.relative_to(REPO_ROOT)),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
            "this metadata run's registry/stdout/stderr/cloudwatch snapshot after run_experiment finalizes",
        ],
        "next_action_after_verified_backup": {
            "script": RUNNER_PATH,
            "args": ["--shard", "9", "--backup-proof", "<post-shard08-audit-run-finalized-proof.json>", "--i-accept-validation-access"],
            "interpreter": "legacy",
            "method": "IMPROVED_latency_tree_vehicle_validation64_formal_shard09_not_original_SAC",
            "split": "validation64_shard9_224_episodes_no_test",
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
        "purpose": "Metadata-only verification of post-shard08-audit backup gate before shard09.",
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
        "key_hash_checks": key_hash_checks,
        "shard09_blocked": shard09_blocked,
        "request_path": str(request_path.relative_to(REPO_ROOT)),
        "cumulative_vehicle_validation64_status_unchanged": {
            "formal_shards_completed": list(range(9)),
            "accepted_audited_shards": list(range(9)),
            "formal_episodes_completed": 2016,
            "formal_control_steps_completed": 177430,
            "planned_total_shards": 12,
            "remaining_formal_shards": [9, 10, 11],
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
        "# Post-shard08 audit backup gate recheck",
        "",
        f"UTC: `{created_utc}`.",
        "",
        "This was a metadata-only backup proof inventory check before vehicle validation64 shard09. It ran no simulations, opened no validation64 bank content, and did not open or hash the sealed final test.",
        "",
        "## Result",
        "",
        f"- Required proof time: after `{REQUIRED_AFTER_UTC}`.",
        f"- Repository-local `backup_proof_20260927T02*.json` count: `{proof_inspection['backup_proof_20260927T02_count']}`.",
        f"- Repository-local `backup_proof_20260927T*.json` files: `{proof_inspection['backup_proof_20260927T_any_glob']}`.",
        f"- Adequate post-shard08 backup proofs found: `{proof_inspection['adequate_backup_proofs_found']}`.",
        f"- Latest known supervisor backup from context: `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}`; inadequate because it predates `{REQUIRED_AFTER_UTC}` and no local proof file exists.",
        f"- Shard09 blocked: `{shard09_blocked}`.",
        "",
        "No formal validation evidence was created. Vehicle validation64 remains at 9/12 completed and audited shards (2016/2688 planned validation episodes; 177430 formal control steps). This is validation/model-selection evidence only for the IMPROVED latency-tree vehicle protocol, not ORIGINAL SAC and not final-test evidence.",
        "",
        "## Key integrity checks",
        "",
    ]
    for check in key_hash_checks:
        summary_lines.append(f"- `{check['path']}` exists={check['exists']} matches_expected_sha256={check['matches']}")
    summary_lines.extend([
        "",
        "## Backup request",
        "",
        f"- `{request_path.relative_to(REPO_ROOT)}`",
        "",
        "The next adequate backup proof must be after this recheck and after the run_experiment registry/stdout/stderr/cloudwatch snapshot finalize, with `backup_verified=true`, `remaining_changed_files=0`, GitHub release asset/download SHA256 verification, runner SHA and gate SHA, and coverage of shard08 formal/audit artifacts, docs/registry updates, prior run-finalized addendum, this recheck, this request, and this metadata run's logs.",
        "",
        "## Next action if and only if the backup gate is satisfied",
        "",
        "Run exactly one formal validation shard with the legacy interpreter:",
        "",
        "```text",
        "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 9 --backup-proof <post-shard08-audit-run-finalized-proof.json> --i-accept-validation-access",
        "```",
        "",
        "Budget: 224 validation episodes, control-step upper bound 33,600, 0 gradient steps; sealed final test closed and not hashed.",
    ])
    summary_path = diag_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    completed = {
        "created_utc": created_utc,
        "passed": True,
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "shard09_blocked": shard09_blocked,
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
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256_file(completed_path)

    marker = f"<!-- post-shard08-audit-backup-gate-recheck-{label} -->"
    doc_section = f"""
{marker}
## 2026-09-27 post-shard08 audit backup gate recheck

UTC: {created_utc}. Metadata-only backup proof inventory check before shard09: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. Required proof was after `{REQUIRED_AFTER_UTC}`. Repository-local `backup_proof_20260927T02*.json` count was {proof_inspection['backup_proof_20260927T02_count']}; all `backup_proof_20260927T*.json` files were {proof_inspection['backup_proof_20260927T_any_glob']}; adequate post-shard08 proofs found: {proof_inspection['adequate_backup_proofs_found']}. The supervisor-context backup at `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}` is insufficient because it predates the shard08 audit run-finalized threshold and no local proof file exists. Shard09 remains backup-gated. No formal validation evidence was created; vehicle validation64 remains 9/12 shards, 2016/2688 episodes, 177430 control steps, final test closed. New artifacts: `{raw_path.relative_to(REPO_ROOT)}`, `{summary_path.relative_to(REPO_ROOT)}`, `{completed_path.relative_to(REPO_ROOT)}`. New backup request: `{request_path.relative_to(REPO_ROOT)}`. Next proof must be after this recheck and this metadata run's finalized registry/stdout/stderr/cloudwatch snapshot before shard09 can run. Method remains IMPROVED latency-tree, not ORIGINAL SAC, and no reproduction-success claim is supported.
"""
    for rel in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(REPO_ROOT / rel, marker, doc_section)

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
        "shard09_blocked": shard09_blocked,
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
