#!/usr/bin/env python3
"""Persist post-shard03-audit backup-gate state without simulations.

This is a metadata/state-preservation helper for the AWS migration run. It does
not open validation/test banks and does not run any controller/environment code.
It records that shard04 formal validation remains blocked unless a verified
external backup proof after the shard03 audit/addendum exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_AFTER = datetime(2026, 9, 26, 19, 5, 30, tzinfo=timezone.utc)
RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
MARKER = "<!-- post-shard03-audit-backup-gate-state-20260926 -->"
DOCS = [
    "STATUS.md",
    "RESEARCH_LOG.md",
    "RESULTS_AUDIT.md",
    "DECISIONS.md",
    "REPRODUCTION_PROTOCOL.md",
]
BACKUP_DIR = ROOT / "research_artifacts" / "aws_backup_proofs"
DIAG_DIR = ROOT / "research_artifacts" / "aws_diagnostics" / "post_shard03_audit_backup_gate_state_20260926"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def inspect_backup_proofs() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    adequate: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json(path) or {}
        proof_time = parse_dt(data.get("proof_created_utc")) or parse_dt(data.get("proof_recorded_utc"))
        rec = {
            "path": rel(path),
            "sha256": sha256_file(path),
            "backup_verified": data.get("backup_verified"),
            "remaining_changed_files": data.get("remaining_changed_files"),
            "commit": data.get("commit") or data.get("commit_sha"),
            "proof_time_utc": proof_time.isoformat() if proof_time else None,
            "runner_sha256": data.get("runner_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "not_sufficient_for_shard04": data.get("not_sufficient_for_shard04"),
            "packages_this_run": data.get("packages_this_run"),
        }
        rec["adequate_for_shard04"] = bool(
            rec["backup_verified"] is True
            and rec["remaining_changed_files"] == 0
            and proof_time is not None
            and proof_time > REQUIRED_AFTER
            and rec["runner_sha256"] == RUNNER_SHA
            and rec["gate_sha256"] == GATE_SHA
            and rec["not_sufficient_for_shard04"] is not True
            and data.get("packages_this_run")
        )
        records.append(rec)
        if rec["adequate_for_shard04"]:
            adequate.append(rec)
    return records, adequate


def file_hashes(paths: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for p in paths:
        out[p] = sha256_file(ROOT / p)
    return out


def count_completed_shards() -> Dict[str, Any]:
    base = ROOT / "research_artifacts" / "aws_formal_validation" / "vehicle_validation64_20260926"
    shards: List[int] = []
    episodes = 0
    steps = 0
    for p in sorted(base.glob("shard*/completed.json")):
        try:
            idx = int(p.parent.name.replace("shard", ""))
        except ValueError:
            continue
        data = read_json(p) or {}
        shards.append(idx)
        episodes += int(data.get("episodes", 0) or 0)
        steps += int(data.get("control_steps", 0) or 0)
    return {"completed_shards": shards, "episodes": episodes, "control_steps": steps}


def append_docs(block: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for doc in DOCS:
        path = ROOT / doc
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER in old:
            changed = False
            new = old
        else:
            sep = "" if old.endswith("\n") or not old else "\n"
            new = old + sep + "\n" + block + "\n"
            path.write_text(new, encoding="utf-8")
            changed = True
        results[doc] = {
            "changed": changed,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return results


def main() -> int:
    created = datetime.now(timezone.utc).replace(microsecond=0)
    created_iso = created.isoformat()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    proof_records, adequate = inspect_backup_proofs()
    latest_proof = proof_records[-1] if proof_records else None
    progress = count_completed_shards()

    key_paths = [
        "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py",
        "experiments/bohn2021_aws/vehicle_validation64_shard03_audit.py",
        "experiments/bohn2021_aws/post_shard03_audit_backup_gate_state.py",
        "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json",
        "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_20260926T190503.json",
        "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_FINAL_ADDENDUM_20260926T190530.json",
        "research_artifacts/aws_backup_proofs/backup_proof_20260926T190416_after_shard03_formal_before_audit.json",
        "research_artifacts/aws_diagnostics/post_shard03_audit_backup_blocker_check_20260926.md",
        "research_artifacts/aws_diagnostics/vehicle_validation64_shard03_audit_20260926/raw.json",
        "research_artifacts/aws_diagnostics/vehicle_validation64_shard03_audit_20260926/summary.md",
        "research_artifacts/aws_diagnostics/vehicle_validation64_shard03_audit_20260926/completed.json",
        "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/raw.json",
        "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/summary.md",
        "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/completed.json",
        "research_artifacts/aws_runs/20260926T174726_c4dd26e4/registry.json",
        "research_artifacts/aws_runs/20260926T174726_c4dd26e4/stdout.log",
        "research_artifacts/aws_runs/20260926T174726_c4dd26e4/stderr.log",
        "research_artifacts/aws_runs/20260926T190449_67731c82/registry.json",
        "research_artifacts/aws_runs/20260926T190449_67731c82/stdout.log",
        "research_artifacts/aws_runs/20260926T190449_67731c82/stderr.log",
        "STATUS.md",
        "RESEARCH_LOG.md",
        "RESULTS_AUDIT.md",
        "DECISIONS.md",
        "REPRODUCTION_PROTOCOL.md",
        "EXPERIMENT_REGISTRY.csv",
    ]
    pre_hashes = file_hashes(key_paths)

    blocked = not bool(adequate)
    block = f"""{MARKER}
## 2026-09-26 post-shard03 audit backup gate state

UTC: {created_iso}. Metadata-only state preservation after shard03 audit and user continuation. No simulations, no control steps, no gradient steps, no validation-bank reopen, and no sealed-test access/open/hash occurred in this action. Existing shard03 formal/audit metadata may be hashed for provenance only; this is not model selection, not final-test evidence, and not an ORIGINAL SAC result.

Backup gate result: shard04 formal validation is {'BLOCKED' if blocked else 'UNBLOCKED BY AN ADEQUATE LOCAL PROOF'} at this state check. Required proof time remains after `2026-09-26T19:05:30Z` and must cover shard03 formal outputs, shard03 audit outputs, audit run registry/stdout/stderr, docs/registry updates, backup requests, final addendum, and the blocker note. Latest local proof observed: `{latest_proof['path'] if latest_proof else 'none'}` at `{latest_proof['proof_time_utc'] if latest_proof else 'none'}`; adequate proofs found: {len(adequate)}.

Vehicle validation64 progress remains {len(progress['completed_shards'])}/12 completed shards, {progress['episodes']} formal validation episodes, {progress['control_steps']} formal validation control steps, plus the archived modern-interpreter shard02 failed attempt with 0 episodes/control steps. Sealed final test remains closed and unauthorized.

Next action if and only if a verified external backup proof satisfying the gate appears: run exactly one formal experiment, `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 4 --backup-proof <post-shard03-audit-proof> --i-accept-validation-access`, with the legacy interpreter; then audit shard04 before any further shard. Do not use the modern interpreter for the TF1 runner.
"""
    doc_updates = append_docs(block)
    post_doc_hashes = {doc: sha256_file(ROOT / doc) for doc in DOCS}

    raw = {
        "created_utc": created_iso,
        "method": "IMPROVED_latency_tree_vehicle_post_shard03_audit_backup_gate_state_not_original_SAC",
        "formal_scientific_evidence_created": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation_accessed_existing_outputs_for_hashing": True,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "required_after_utc": REQUIRED_AFTER.isoformat(),
        "runner_sha256_required": RUNNER_SHA,
        "gate_sha256_required": GATE_SHA,
        "backup_proofs_observed": proof_records,
        "adequate_backup_proofs_for_shard04": adequate,
        "blocked_before_shard04": blocked,
        "latest_local_proof": latest_proof,
        "vehicle_validation64_progress": progress,
        "pre_hashes": pre_hashes,
        "doc_updates": doc_updates,
        "post_doc_hashes": post_doc_hashes,
        "next_action_if_unblocked": "vehicle_validation64_shard_runner.py --shard 4 with legacy interpreter, then shard04 audit",
        "do_not_repeat": [
            "do not rerun completed formal shards00-03 unless an audit/hash failure is discovered",
            "do not rerun shard03 audit; it already passed",
            "do not run shard04 before an adequate post-shard03-audit external backup proof",
            "do not open or hash sealed final-test bank",
            "do not use modern interpreter for TF1 runner",
        ],
    }

    raw_path = DIAG_DIR / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    request_path: Optional[Path] = None
    request_payload: Optional[Dict[str, Any]] = None
    if blocked:
        stamp = created.strftime("%Y%m%dT%H%M%S")
        request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_BLOCKER_STATE_{stamp}.json"
        request_payload = {
            "status": "external_backup_requested_after_validation64_shard03_audit_blocker_state",
            "created_utc": created_iso,
            "backup_verified": False,
            "reason": "No adequate repository-local verified external backup proof after shard03 audit/final addendum/blocker note exists; shard04 formal validation remains blocked.",
            "required_proof_after_utc": created_iso,
            "minimum_scientific_gate_time_utc": REQUIRED_AFTER.isoformat(),
            "validation_accessed_existing_outputs_for_hashing": True,
            "validation_bank_reopened_by_this_action": False,
            "test_accessed": False,
            "sealed_test_bank_content_opened": False,
            "sealed_test_bank_hashed_by_this_action": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_gradient_steps": 0,
            "runner_sha256": RUNNER_SHA,
            "gate_sha256": GATE_SHA,
            "must_cover": [
                "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/",
                "research_artifacts/aws_diagnostics/vehicle_validation64_shard03_audit_20260926/",
                "research_artifacts/aws_runs/20260926T174726_c4dd26e4/",
                "research_artifacts/aws_runs/20260926T190449_67731c82/",
                "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_20260926T190503.json",
                "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_FINAL_ADDENDUM_20260926T190530.json",
                "research_artifacts/aws_diagnostics/post_shard03_audit_backup_blocker_check_20260926.md",
                rel(raw_path),
                "research_artifacts/aws_diagnostics/post_shard03_audit_backup_gate_state_20260926/summary.md",
                "research_artifacts/aws_diagnostics/post_shard03_audit_backup_gate_state_20260926/completed.json",
                "experiments/bohn2021_aws/post_shard03_audit_backup_gate_state.py",
                "STATUS.md",
                "RESEARCH_LOG.md",
                "RESULTS_AUDIT.md",
                "DECISIONS.md",
                "REPRODUCTION_PROTOCOL.md",
                "EXPERIMENT_REGISTRY.csv",
                rel(request_path),
            ],
            "observed_latest_local_proof": latest_proof,
            "adequate_backup_proofs_found": adequate,
            "pre_hashes": pre_hashes,
            "post_doc_hashes": post_doc_hashes,
            "raw_json": rel(raw_path),
            "self_hash_note": "This request file is not self-hashed inside itself; completed.json records its SHA256 after write.",
            "not_sufficient_for_final_test": True,
            "next_action_after_verified_backup": "Run shard04 formal validation with legacy interpreter and sealed final test closed, then audit shard04.",
        }
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = DIAG_DIR / "summary.md"
    summary = [
        "# Post-shard03 audit backup gate state",
        "",
        f"Created UTC: `{created_iso}`",
        "",
        "No simulations, no control steps, no gradient steps, no validation-bank reopen, and no sealed-test access/open/hash occurred.",
        "",
        f"Required post-shard03-audit proof time: after `{REQUIRED_AFTER.isoformat()}`.",
        f"Adequate proofs found: `{len(adequate)}`.",
        f"Shard04 blocked: `{blocked}`.",
        f"Latest local proof: `{latest_proof['path'] if latest_proof else 'none'}` at `{latest_proof['proof_time_utc'] if latest_proof else 'none'}`.",
        "",
        f"Vehicle validation64 progress: {len(progress['completed_shards'])}/12 shards, {progress['episodes']} episodes, {progress['control_steps']} control steps; final test remains closed.",
        "",
        f"Backup request: `{rel(request_path) if request_path else 'not written because adequate proof was found'}`.",
        "",
        "Next action after adequate backup proof: run shard04 with the frozen runner under the legacy interpreter, then audit shard04 before any further shard.",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    completed = {
        "created_utc": created_iso,
        "passed": True,
        "blocked_before_shard04": blocked,
        "adequate_backup_proofs_found": len(adequate),
        "formal_scientific_evidence_created": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation_accessed_existing_outputs_for_hashing": True,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "raw": rel(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "backup_request": rel(request_path) if request_path else None,
        "backup_request_sha256": sha256_file(request_path) if request_path else None,
        "doc_updates": doc_updates,
        "next_action_after_verified_backup": "run shard04 with legacy interpreter and sealed final test closed; then audit shard04",
    }
    completed_path = DIAG_DIR / "completed.json"
    completed_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(completed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
