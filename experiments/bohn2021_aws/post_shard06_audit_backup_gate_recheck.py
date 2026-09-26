#!/usr/bin/env python3
"""Metadata-only backup-gate recheck after vehicle validation64 shard06 audit.

This helper is intentionally conservative.  It does not run controllers,
train policies, open the validation bank, or open/hash the sealed final-test
bank.  It only inspects repository-local backup proof metadata and writes a
post-shard06 blocker/request that explicitly includes the finalized shard06
audit run registry/stdout/stderr requirement before shard07 may be run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = ROOT / "research_artifacts" / "aws_backup_proofs"
DIAG_ROOT = ROOT / "research_artifacts" / "aws_diagnostics"

# The prior shard06 audit run completed at 2026-09-26T23:29:22Z, and the
# previous handoff state was persisted at 2026-09-26T23:33:00Z.  A proof for
# shard07 must be after those events and, after this recheck is run, after this
# recheck's run_experiment registry/stdout/stderr are finalized too.
REQUIRED_AFTER_PRIOR_SHARD06_AUDIT_STATE_UTC = "2026-09-26T23:33:00+00:00"
RUNNER_PATH = "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
RUNNER_SHA256 = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
GATE_PATH = "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
GATE_SHA256 = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
PRE_SHARD06_BACKUP_PROOF = "research_artifacts/aws_backup_proofs/backup_proof_20260926T221348_after_shard05_v2_audit_blocker_state.json"
PRE_SHARD06_BACKUP_PROOF_SHA256 = "db4850a87703cd008c0459bab518de1670bd72cfba1fb7339b2b0543735892e9"
SHARD06_FORMAL_RUN_ID = "20260926T221425_572df1e9"
SHARD06_AUDIT_RUN_ID = "20260926T232908_163b4f98"
SHARD06_AUDIT_COMPLETED = "research_artifacts/aws_diagnostics/vehicle_validation64_shard06_audit_20260926/completed.json"
SHARD06_AUDIT_COMPLETED_SHA256 = "8ecd91ef4c79a8ef252fe9f1ae98403c21a8d515b5373417e39044040b54c830"
SHARD06_AUDIT_RAW = "research_artifacts/aws_diagnostics/vehicle_validation64_shard06_audit_20260926/raw.json"
SHARD06_AUDIT_RAW_SHA256 = "8322dc5a4f2ebf1eacfa772f8038ac0bbcc5284f0da9dbff416ef096d96ebcc0"
SHARD06_AUDIT_SUMMARY = "research_artifacts/aws_diagnostics/vehicle_validation64_shard06_audit_20260926/summary.md"
SHARD06_AUDIT_SUMMARY_SHA256 = "7037c85a7594047953e6877bdc5592da01f80a4c3a138ba6e07f389da0d87275"
SHARD06_FORMAL_COMPLETED = "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/completed.json"
SHARD06_FORMAL_COMPLETED_SHA256 = "4361e50f905da2b25d4e549b196ea028591fc8a0aebbcd12cf3bb6d5bc1905ed"
SHARD06_FORMAL_RAW = "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/raw.json"
SHARD06_FORMAL_RAW_SHA256 = "74c687555c2fc111b7fa08b81bf97cbf9f9c0b8e9e048fdff19cd8964142272d"
SHARD06_FORMAL_SUMMARY = "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/summary.md"
SHARD06_FORMAL_SUMMARY_SHA256 = "a82acf85047a706445436243f9b4dc776ef5f229daeebe5becc06697529ca7a1"
SHARD06_BACKUP_REQUEST = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_20260926T232908.json"
SHARD06_BACKUP_REQUEST_SHA256 = "f5eb58c8a8046913667daf2db54c81833e8ccd4bfc3acf355363c15c77361df4"
SHARD06_FINAL_ADDENDUM = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_FINAL_ADDENDUM_20260926T232908.json"
SHARD06_FINAL_ADDENDUM_SHA256 = "eab9e63350ec78146350a7681b2e1709c6c7fe2e5fecd6fdbc243a27123f4d56"
SHARD06_BLOCKER = "research_artifacts/aws_diagnostics/post_shard06_audit_backup_blocker_check_20260926.md"
SHARD06_BLOCKER_SHA256 = "ff23003a1b2b5d71047b33abfc2813f916f99f0b61078ea16a7c10c43dbcac3e"

LATEST_INADEQUATE_SUPERVISOR_BACKUP = {
    "time": "2026-09-26T23:28:51.819071+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "4ba8b0a36f3273178b8ab97e4060410f343d7048",
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "packages_this_run": [
        {
            "name": "20260926T232829_34737a85.tar.gz",
            "sha256": "a09b7127ca1932a6ff4b3abc1bf68e6bd3beba7e50f99b8345f4e23cd85d41c5",
            "bytes": 93061614,
            "verification": "github_server_sha256",
        },
        {
            "name": "20260926T232841_f0762918.tar.gz",
            "sha256": "0aab57d4bab4be38e4c4057182907d683980ff2c0fa7facc12389696d2f0cad2",
            "bytes": 73202715,
            "verification": "github_server_sha256",
        },
    ],
    "why_inadequate_for_shard07": "Predates shard06 audit start at 2026-09-26T23:29:08Z, audit completion at 23:29:22Z, shard06 audit outputs, docs/registry updates, backup request/addendum/blocker, and the persisted 23:33 state.",
}

DOCS = [
    "STATUS.md",
    "RESEARCH_LOG.md",
    "RESULTS_AUDIT.md",
    "DECISIONS.md",
    "REPRODUCTION_PROTOCOL.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_label(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "Z")


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


def read_json_maybe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
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


def proof_time(data: Dict[str, Any]) -> Optional[str]:
    for key in ("proof_created_utc", "proof_recorded_utc", "time", "created_utc"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def coverage_flags(data: Dict[str, Any]) -> Dict[str, bool]:
    text = json.dumps(data, sort_keys=True)
    return {
        "mentions_shard06_formal": "shard06" in text and ("formal" in text or "vehicle_validation64_20260926/shard06" in text),
        "mentions_shard06_audit": "vehicle_validation64_shard06_audit_20260926" in text or "shard06 audit" in text,
        "mentions_shard06_formal_run": SHARD06_FORMAL_RUN_ID in text,
        "mentions_shard06_audit_run": SHARD06_AUDIT_RUN_ID in text,
        "mentions_shard06_backup_request": "REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_20260926T232908" in text,
        "mentions_shard06_final_addendum": "REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_FINAL_ADDENDUM_20260926T232908" in text,
        "mentions_shard06_blocker": "post_shard06_audit_backup_blocker_check_20260926" in text,
        "runner_sha_match": RUNNER_SHA256 in text,
        "gate_sha_match": GATE_SHA256 in text,
        "github_sha_verification": "github_server_sha256" in text or "download_sha256" in text or "verified" in text and "sha256" in text,
    }


def inspect_backup_proofs(required_after: datetime) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    adequate: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json_maybe(path)
        record: Dict[str, Any] = {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "json_ok": data is not None,
        }
        if data is not None:
            t_str = proof_time(data)
            t_dt = parse_dt(t_str)
            flags = coverage_flags(data)
            packages = data.get("packages_this_run")
            record.update(
                {
                    "backup_verified": data.get("backup_verified"),
                    "remaining_changed_files": data.get("remaining_changed_files"),
                    "commit": data.get("commit") or data.get("commit_sha"),
                    "proof_time": t_str,
                    "proof_time_utc": t_dt.isoformat() if t_dt else None,
                    "after_required_prior_shard06_audit_state": bool(t_dt and t_dt > required_after),
                    "runner_sha256": data.get("runner_sha256"),
                    "gate_sha256": data.get("gate_sha256"),
                    "packages_this_run_count": len(packages) if isinstance(packages, list) else None,
                    "coverage_flags": flags,
                }
            )
            is_adequate = bool(
                data.get("backup_verified") is True
                and data.get("remaining_changed_files") == 0
                and t_dt is not None
                and t_dt > required_after
                and flags["runner_sha_match"]
                and flags["gate_sha_match"]
                and flags["github_sha_verification"]
                and flags["mentions_shard06_formal"]
                and flags["mentions_shard06_audit"]
                and flags["mentions_shard06_formal_run"]
                and flags["mentions_shard06_audit_run"]
                and flags["mentions_shard06_backup_request"]
                and flags["mentions_shard06_final_addendum"]
                and flags["mentions_shard06_blocker"]
            )
            record["adequate_for_shard07_before_this_recheck"] = is_adequate
            if is_adequate:
                adequate.append(record)
        records.append(record)
    return records, adequate


def count_validation64_progress() -> Dict[str, Any]:
    base = ROOT / "research_artifacts" / "aws_formal_validation" / "vehicle_validation64_20260926"
    completed_shards: List[int] = []
    episodes = 0
    control_steps = 0
    shard_details: Dict[str, Dict[str, Any]] = {}
    for completed_path in sorted(base.glob("shard*/completed.json")):
        try:
            shard_idx = int(completed_path.parent.name.replace("shard", ""))
        except ValueError:
            continue
        data = read_json_maybe(completed_path) or {}
        completed_shards.append(shard_idx)
        eps = int(data.get("episodes", 0) or 0)
        steps = int(data.get("control_steps", 0) or 0)
        episodes += eps
        control_steps += steps
        shard_details[f"shard{shard_idx:02d}"] = {
            "episodes": eps,
            "control_steps": steps,
            "completed_path": rel(completed_path),
            "completed_sha256": sha256_file(completed_path),
        }
    audit_passed: List[int] = []
    for audit_completed in sorted((ROOT / "research_artifacts" / "aws_diagnostics").glob("vehicle_validation64_shard*_audit*_20260926/completed.json")):
        # Include the repaired shard05 v2 audit and normal audits, but only if passed.
        stem = audit_completed.parent.name
        if "shard05_audit_20260926" in stem and "v2" not in stem:
            # Preserved known v1 false positive; not counted as passed.
            continue
        data = read_json_maybe(audit_completed) or {}
        if data.get("passed") is not True:
            continue
        try:
            idx_text = stem.split("shard", 1)[1][:2]
            audit_passed.append(int(idx_text))
        except Exception:
            pass
    audit_passed = sorted(set(audit_passed))
    return {
        "formal_shards_completed": sorted(completed_shards),
        "audited_shards_passed_detected": audit_passed,
        "formal_episodes_completed": episodes,
        "formal_control_steps_completed": control_steps,
        "failed_formal_attempts": [
            "20260926T161356_fb71c8d7 shard02 modern-interpreter TensorFlow-missing attempt: validation bank opened, 0 episodes/control steps/gradient steps"
        ],
        "failed_audit_attempts": [
            "20260926T144037_523e1646 shard00 audit-v1 registry-schema false negative, repaired by v2",
            "20260926T215916_c2ecb9ed shard05 audit-v1 schema false positive, repaired by v2",
        ],
        "shard_details": shard_details,
    }


def file_hashes(path_specs: Iterable[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for rel_path, expected in path_specs:
        path = ROOT / rel_path
        actual = sha256_file(path)
        checks.append(
            {
                "path": rel_path,
                "exists": path.exists(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches_expected": (actual == expected) if expected else None,
            }
        )
    return checks


def append_once(path: Path, marker: str, block: str) -> Dict[str, Any]:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    changed = False
    if marker not in old:
        sep = "" if (not old or old.endswith("\n")) else "\n"
        path.write_text(old + sep + "\n" + block.strip() + "\n", encoding="utf-8")
        changed = True
    return {"changed": changed, "sha256": sha256_file(path), "bytes": path.stat().st_size if path.exists() else None}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    created = utc_now()
    label = safe_label(created).replace("Z", "")
    diag_dir = DIAG_ROOT / f"post_shard06_audit_backup_gate_recheck_{label}"
    diag_dir.mkdir(parents=True, exist_ok=False)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    required_prior_dt = parse_dt(REQUIRED_AFTER_PRIOR_SHARD06_AUDIT_STATE_UTC)
    assert required_prior_dt is not None
    proof_records, adequate = inspect_backup_proofs(required_prior_dt)
    latest_proof = proof_records[-1] if proof_records else None
    progress = count_validation64_progress()

    key_hash_checks = file_hashes(
        [
            (RUNNER_PATH, RUNNER_SHA256),
            (GATE_PATH, GATE_SHA256),
            (PRE_SHARD06_BACKUP_PROOF, PRE_SHARD06_BACKUP_PROOF_SHA256),
            ("experiments/bohn2021_aws/vehicle_validation64_shard06_audit.py", "d7df5983752552433afb6ad754c68d93e3e99cdd2cab70f3de71743af2567bb1"),
            (SHARD06_FORMAL_COMPLETED, SHARD06_FORMAL_COMPLETED_SHA256),
            (SHARD06_FORMAL_RAW, SHARD06_FORMAL_RAW_SHA256),
            (SHARD06_FORMAL_SUMMARY, SHARD06_FORMAL_SUMMARY_SHA256),
            ("research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/progress.json", "0ff80d06cf97c32169900a8fcb0ec300e2caa27db2aacd80e24071f55c51ff9a"),
            ("research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/schedule.json", "fe97894b57688684430011a6f028630eb399f992d99e992acd0f1cf79bfb48f1"),
            ("research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/run_started.json", "e3cc36ed423d2f23c31e77611b32c6bcbdfd6f6f051d09fc7651e34c48d7b3df"),
            ("research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/terminal_sources_progress.json", "76cc5fd5b07745fab8be2cc5618b0b0eab4f087479959a35b9af7ab02565f755"),
            (SHARD06_AUDIT_RAW, SHARD06_AUDIT_RAW_SHA256),
            (SHARD06_AUDIT_SUMMARY, SHARD06_AUDIT_SUMMARY_SHA256),
            (SHARD06_AUDIT_COMPLETED, SHARD06_AUDIT_COMPLETED_SHA256),
            (SHARD06_BACKUP_REQUEST, SHARD06_BACKUP_REQUEST_SHA256),
            (SHARD06_FINAL_ADDENDUM, SHARD06_FINAL_ADDENDUM_SHA256),
            (SHARD06_BLOCKER, SHARD06_BLOCKER_SHA256),
            (f"research_artifacts/aws_runs/{SHARD06_FORMAL_RUN_ID}/registry.json", None),
            (f"research_artifacts/aws_runs/{SHARD06_FORMAL_RUN_ID}/stdout.log", None),
            (f"research_artifacts/aws_runs/{SHARD06_FORMAL_RUN_ID}/stderr.log", None),
            (f"research_artifacts/aws_runs/{SHARD06_AUDIT_RUN_ID}/registry.json", None),
            (f"research_artifacts/aws_runs/{SHARD06_AUDIT_RUN_ID}/stdout.log", None),
            (f"research_artifacts/aws_runs/{SHARD06_AUDIT_RUN_ID}/stderr.log", None),
            ("EXPERIMENT_REGISTRY.csv", None),
            ("STATUS.md", None),
            ("RESEARCH_LOG.md", None),
            ("RESULTS_AUDIT.md", None),
            ("DECISIONS.md", None),
            ("REPRODUCTION_PROTOCOL.md", None),
        ]
    )
    key_hashes_ok = all(item["exists"] and (item["matches_expected"] is not False) for item in key_hash_checks)

    blocked_before_shard07 = len(adequate) == 0
    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_SHARD06_AUDIT_GATE_RECHECK_{label}.json"
    request_payload = {
        "status": "external_backup_requested_after_shard06_audit_gate_recheck",
        "created_utc": created,
        "backup_verified": False,
        "required_before_shard07": True,
        "validation_accessed_by_this_recheck": True,
        "validation_access_type": "existing shard06 result artifacts hashed/counted only; validation bank not reopened",
        "validation_bank_reopened_by_this_recheck": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_recheck": False,
        "new_simulations_by_this_recheck": 0,
        "new_control_steps_by_this_recheck": 0,
        "new_gradient_steps_by_this_recheck": 0,
        "reason": "No repository-local verified external backup proof after the shard06 audit/final addendum/finalized run registry was found. Shard07 formal validation remains blocked until a supervisor-verified backup covers this recheck and the run_experiment registry/stdout/stderr after finalization.",
        "required_next_backup_proof_fields_before_shard07": {
            "backup_verified": True,
            "remaining_changed_files": 0,
            "proof_time_after_utc": created,
            "github_release_asset_download_sha256_verified": True,
            "runner_sha256": RUNNER_SHA256,
            "gate_sha256": GATE_SHA256,
            "must_cover_validation64_shard06_formal_outputs": True,
            "must_cover_shard06_formal_run_registry_stdout_stderr": True,
            "must_cover_passed_shard06_audit_outputs": True,
            "must_cover_shard06_audit_run_registry_stdout_stderr": True,
            "must_cover_docs_and_experiment_registry_updates": True,
            "must_cover_shard06_backup_request_final_addendum_blocker": True,
            "must_cover_this_gate_recheck_artifacts": True,
            "must_cover_this_request": rel(request_path),
            "must_cover_this_recheck_run_registry_stdout_stderr_after_run_experiment_finalizes": True,
        },
        "latest_inadequate_supervisor_backup_from_context": LATEST_INADEQUATE_SUPERVISOR_BACKUP,
        "latest_local_backup_proof_observed": latest_proof,
        "adequate_backup_proofs_found_before_this_recheck": len(adequate),
        "next_action_after_verified_backup": {
            "script": RUNNER_PATH,
            "args": ["--shard", "7", "--backup-proof", "<post-shard06-audit-gate-recheck-proof.json>", "--i-accept-validation-access"],
            "interpreter": "legacy",
            "method": "IMPROVED_latency_tree_vehicle_validation64_formal_shard07_not_original_SAC",
            "split": "validation64_shard7_224_episodes_no_test",
            "training_gradient_steps": 0,
            "validation_episodes": 224,
            "validation_control_step_upper_bound": 33600,
            "test_accessed": False,
        },
    }
    write_json(request_path, request_payload)

    raw_path = diag_dir / "raw.json"
    raw_payload: Dict[str, Any] = {
        "created_utc": created,
        "purpose": "Metadata-only backup proof inventory recheck after shard06 audit; persist blocker before shard07.",
        "method_classification": "IMPROVED latency-tree vehicle validation, not ORIGINAL SAC",
        "formal_scientific_evidence_created_by_this_recheck": False,
        "validation_accessed": True,
        "validation_access_type": "existing shard06 validation result artifacts hashed/counted only; validation64 bank not reopened",
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "required_after_prior_shard06_audit_state_utc": REQUIRED_AFTER_PRIOR_SHARD06_AUDIT_STATE_UTC,
        "blocked_before_shard07_before_this_recheck": blocked_before_shard07,
        "proof_inspection": {
            "proof_count": len(proof_records),
            "proof_records": proof_records,
            "adequate_backup_proofs_found": len(adequate),
            "adequate_backup_proofs": adequate,
            "backup_proof_20260926T23_glob_count": len(list(BACKUP_DIR.glob("backup_proof_20260926T23*.json"))),
            "latest_local_backup_proof": latest_proof,
        },
        "latest_inadequate_supervisor_backup_from_context": LATEST_INADEQUATE_SUPERVISOR_BACKUP,
        "key_hash_checks": key_hash_checks,
        "key_hashes_ok": key_hashes_ok,
        "progress": progress,
        "backup_request": rel(request_path),
    }
    write_json(raw_path, raw_payload)

    summary_path = diag_dir / "summary.md"
    summary_lines = [
        "# Post-shard06 audit backup gate recheck",
        "",
        f"UTC: {created}.",
        "",
        "Purpose: metadata-only proof inventory and state preservation after the shard06 audit. No simulation, training, validation-bank reopen, sealed-test open, or sealed-test hash occurred. Existing shard06 result artifacts and run logs were hashed/counted only for provenance.",
        "",
        "## Result",
        "",
        f"- Adequate verified post-shard06-audit backup proofs found before this recheck: `{len(adequate)}`.",
        f"- `backup_proof_20260926T23*.json` local count: `{len(list(BACKUP_DIR.glob('backup_proof_20260926T23*.json')))}`.",
        f"- Latest local backup proof: `{latest_proof['path'] if latest_proof else 'none'}` at `{latest_proof.get('proof_time_utc') if latest_proof else 'none'}`.",
        f"- Latest supervisor backup in context is inadequate for shard07: `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}` predates the shard06 audit at `2026-09-26T23:29:08Z`.",
        f"- Key shard06 formal/audit/runner/gate hashes matched expected values: `{key_hashes_ok}`.",
        "",
        "Shard07 remains **blocked** until a new verified external backup proof exists after this recheck and after this metadata run's registry/stdout/stderr are finalized. The sealed final test remains closed and unauthorized.",
        "",
        "## Cumulative vehicle validation64 status",
        "",
        f"- Formal shards completed: `{progress['formal_shards_completed']}` / 12.",
        f"- Audited shards passed, detected: `{progress['audited_shards_passed_detected']}`.",
        f"- Formal validation episodes completed: `{progress['formal_episodes_completed']}`.",
        f"- Formal validation control steps completed: `{progress['formal_control_steps_completed']}`.",
        "- Preserved failures: shard02 modern-interpreter TensorFlow-missing formal attempt with 0 episodes/control steps; shard00 and shard05 audit-schema false positives repaired by versioned audits.",
        "- This is validation/model-selection evidence for an IMPROVED latency-tree method, not ORIGINAL SAC, and not a reproduction-success claim.",
        "",
        "## New backup request",
        "",
        f"- `{rel(request_path)}`",
        "",
        "The next proof must record `backup_verified=true`, `remaining_changed_files=0`, GitHub release asset/download SHA256 verification, runner SHA `" + RUNNER_SHA256 + "`, gate SHA `" + GATE_SHA256 + "`, and coverage of shard06 formal outputs, shard06 audit outputs, formal/audit run registries/stdout/stderr, docs/registry updates, shard06 backup request/final addendum/blocker note, this recheck, this request, and this metadata run's registry/stdout/stderr.",
        "",
        "## Next action if and only if the backup gate is satisfied",
        "",
        "Run exactly one formal experiment with the legacy interpreter:",
        "",
        "```text",
        "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 7 --backup-proof <post-shard06-audit-gate-recheck-proof.json> --i-accept-validation-access",
        "```",
        "",
        "Budget: 224 validation episodes, control-step upper bound 33,600, 0 training gradient steps, sealed final test closed. Then audit shard07 before any further shard.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    marker = f"<!-- post-shard06-audit-backup-gate-recheck-{label} -->"
    common_section = f"""
{marker}
## 2026-09-26 post-shard06 audit backup gate recheck

UTC: {created}. Metadata-only backup inventory recheck before shard07: validation_accessed=true only for existing shard06 output hashing/counting, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. No adequate repository-local verified external backup proof after shard06 audit finalization and persisted state was found (`backup_proof_20260926T23*.json` count {len(list(BACKUP_DIR.glob('backup_proof_20260926T23*.json')))}; adequate proofs {len(adequate)}). Latest supervisor backup in context `{LATEST_INADEQUATE_SUPERVISOR_BACKUP['time']}` predates the shard06 audit and is insufficient. Key shard06 formal/audit/runner/gate hashes matched expected values: {key_hashes_ok}. Shard07 remains backup-gated. New recheck artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `completed.json` in the same directory. New backup request: `{rel(request_path)}`. Next proof must be after this recheck and after the metadata run registry/stdout/stderr finalize, and must cover shard06 formal outputs, shard06 audit outputs, formal/audit/recheck run logs, docs/registry, shard06 backup request/final addendum/blocker note, this recheck/request, runner SHA `{RUNNER_SHA256}`, and gate SHA `{GATE_SHA256}`. Current status remains partial validation evidence only for an IMPROVED latency-tree method, not ORIGINAL SAC, not final-test evidence, and not a reproduction-success claim.
"""
    doc_updates = {doc: append_once(ROOT / doc, marker, common_section) for doc in DOCS}

    completed_path = diag_dir / "completed.json"
    completed_payload: Dict[str, Any] = {
        "created_utc": created,
        "passed": True,
        "blocked_before_shard07": True,
        "adequate_backup_proofs_found_before_this_recheck": len(adequate),
        "formal_scientific_evidence_created_by_this_recheck": False,
        "validation_accessed": True,
        "validation_access_type": "existing shard06 result artifacts hashed/counted only; validation64 bank not reopened",
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
        "key_hashes_ok": key_hashes_ok,
        "progress": {
            "formal_shards_completed": progress["formal_shards_completed"],
            "audited_shards_passed_detected": progress["audited_shards_passed_detected"],
            "formal_episodes_completed": progress["formal_episodes_completed"],
            "formal_control_steps_completed": progress["formal_control_steps_completed"],
        },
        "doc_updates": doc_updates,
        "next_required_backup_after_utc": created,
        "next_action_after_verified_backup": "run shard07 with legacy interpreter and sealed final test closed; then audit shard07 before any further shard",
    }
    write_json(completed_path, completed_payload)

    print(
        json.dumps(
            {
                "completed": rel(completed_path),
                "completed_sha256": sha256_file(completed_path),
                "raw": rel(raw_path),
                "raw_sha256": sha256_file(raw_path),
                "summary": rel(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "backup_request": rel(request_path),
                "backup_request_sha256": sha256_file(request_path),
                "adequate_backup_proofs_found_before_this_recheck": len(adequate),
                "blocked_before_shard07": True,
                "key_hashes_ok": key_hashes_ok,
                "formal_shards_completed": progress["formal_shards_completed"],
                "audited_shards_passed_detected": progress["audited_shards_passed_detected"],
                "formal_episodes_completed": progress["formal_episodes_completed"],
                "formal_control_steps_completed": progress["formal_control_steps_completed"],
                "validation_accessed": True,
                "validation_bank_reopened": False,
                "test_accessed": False,
                "sealed_test_bank_content_opened": False,
                "new_simulations": 0,
                "new_control_steps": 0,
                "new_gradient_steps": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
