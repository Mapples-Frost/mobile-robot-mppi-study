#!/usr/bin/env python3
"""Metadata-only blocker audit for the post-dry-run backup gate.

This script does not open validation64 or sealed-test bank content and does not
run simulations.  It verifies the already-created dry-run/gate artifacts and
writes an explicit repository-local request for the supervisor to create and
verify an external backup before any formal validation shard is run.
"""

from __future__ import print_function

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/post_dryrun_backup_blocker_audit_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
FORMAL_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
DRY_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
GATE_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/completed.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
BACKUP_NEEDED = BACKUP_DIR / "BACKUP_NEEDED_AFTER_DRYRUN_20260926T131600.json"
PRE_DRYRUN_PROOF = BACKUP_DIR / "backup_proof_20260926T131429_after_gate_runner_before_dryrun.json"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
DRYRUN_HASHES = {
    "research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/dry_run.json": "3b495a6c9e11ad7967ccf17d0c1b271ddecff59e6c21043f745017c321d04991",
    "research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/run_started.json": "ff679bc4405aa5223d1fa910ea76a4e2d422018673fab2914bfe23eceec74b83",
    "research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/summary.md": "f6581125fdd38645c0d55bfdfec73b04e2d09051279e25741c6244ad0f06de61",
}

DOC_MARKER = "<!-- post-dryrun-backup-blocker-audit-20260926 -->"


def rel(path):
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path, marker, text):
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + marker + "\n" + text.strip() + "\n", encoding="utf-8")
    return True


def collect_local_proofs():
    proofs = []
    if not BACKUP_DIR.exists():
        return proofs
    for path in sorted(BACKUP_DIR.glob("*.json")):
        try:
            data = read_json(path)
        except Exception as exc:
            proofs.append({"path": rel(path), "read_error": type(exc).__name__})
            continue
        item = {
            "path": rel(path),
            "sha256": sha256(path),
            "backup_verified": data.get("backup_verified"),
            "remaining_changed_files": data.get("remaining_changed_files"),
            "commit": data.get("commit"),
            "runner_sha256": data.get("runner_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "asset_sha256": data.get("asset_sha256"),
            "packages_this_run_count": len(data.get("packages_this_run") or []),
            "not_sufficient_for_formal_validation_after_dryrun": data.get("not_sufficient_for_formal_validation_after_dryrun"),
            "status": data.get("status"),
        }
        dry_hashes = data.get("dry_run_artifact_hashes") or data.get("dryrun_artifact_hashes") or {}
        item["records_required_dryrun_hashes"] = all(dry_hashes.get(k) == v for k, v in DRYRUN_HASHES.items())
        proofs.append(item)
    return proofs


def adequate_post_dryrun_proofs(proofs):
    adequate = []
    for item in proofs:
        if item.get("backup_verified") is not True:
            continue
        if item.get("remaining_changed_files") != 0:
            continue
        if item.get("runner_sha256") != EXPECTED_RUNNER_SHA:
            continue
        if item.get("gate_sha256") != EXPECTED_GATE_SHA:
            continue
        if not (item.get("asset_sha256") or item.get("packages_this_run_count")):
            continue
        if item.get("not_sufficient_for_formal_validation_after_dryrun"):
            continue
        if not item.get("records_required_dryrun_hashes"):
            continue
        adequate.append(item)
    return adequate


def main():
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required_existing = [
        RUNNER,
        GATE_JSON,
        GATE_COMPLETED,
        BACKUP_NEEDED,
        PRE_DRYRUN_PROOF,
        DRY_DIR / "dry_run.json",
        DRY_DIR / "summary.md",
        DRY_DIR / "completed.json",
        DRY_DIR / "run_started.json",
        ROOT / "research_artifacts/aws_runs/20260926T131600_af64aaa9/registry.json",
        ROOT / "research_artifacts/aws_runs/20260926T131600_af64aaa9/stdout.log",
        ROOT / "research_artifacts/aws_runs/20260926T131600_af64aaa9/stderr.log",
        ROOT / "STATUS.md",
        ROOT / "RESEARCH_LOG.md",
        ROOT / "RESULTS_AUDIT.md",
        ROOT / "DECISIONS.md",
        ROOT / "REPRODUCTION_PROTOCOL.md",
        ROOT / "EXPERIMENT_REGISTRY.csv",
    ]
    missing = [rel(path) for path in required_existing if not path.exists()]
    hashes = {}
    for path in required_existing:
        if path.exists() and path.is_file():
            hashes[rel(path)] = sha256(path)

    gate_sha = hashes.get(rel(GATE_JSON))
    runner_sha = hashes.get(rel(RUNNER))
    dry_completed = read_json(DRY_DIR / "completed.json") if (DRY_DIR / "completed.json").exists() else {}
    dry_hash_mismatches = []
    for name, expected in DRYRUN_HASHES.items():
        path = ROOT / name
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            dry_hash_mismatches.append({"path": name, "expected": expected, "actual": actual})

    proofs = collect_local_proofs()
    adequate = adequate_post_dryrun_proofs(proofs)
    formal_files = []
    if FORMAL_DIR.exists():
        for path in sorted(FORMAL_DIR.rglob("*")):
            if path.is_file():
                formal_files.append(rel(path))

    request_name = "REQUEST_POST_DRYRUN_BACKUP_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"
    request_path = BACKUP_DIR / request_name
    audit_paths = [
        rel(OUT_DIR / "raw.json"),
        rel(OUT_DIR / "summary.md"),
        rel(OUT_DIR / "completed.json"),
        rel(request_path),
    ]
    minimum_paths = sorted(set([rel(path) for path in required_existing] + audit_paths))
    request = {
        "status": "external_backup_requested_before_formal_validation64",
        "backup_verified": False,
        "created_utc": now,
        "reason": "Dry-run artifacts/docs/registry were created after the last verified external backup; opening validation64 remains blocked until these local changes are externally recoverable.",
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "runner_sha256": runner_sha,
        "gate_sha256": gate_sha,
        "dry_run_artifact_hashes": DRYRUN_HASHES,
        "required_backup_condition_for_next_proof": {
            "backup_verified": True,
            "remaining_changed_files": 0,
            "commit": "required: new commit or verified state after dry-run outputs/docs/registry/proofs",
            "release_or_asset_sha256": "required: GitHub release asset/server or download SHA256 verification",
            "runner_sha256": EXPECTED_RUNNER_SHA,
            "gate_sha256": EXPECTED_GATE_SHA,
            "must_cover_dryrun_artifacts_and_this_request": True,
        },
        "minimum_paths_to_include_in_next_external_backup": minimum_paths,
        "after_verified_backup_next_action": "Create a repository-local post-dryrun proof JSON with backup_verified=true and run vehicle_validation64_shard_runner.py --shard 0 --backup-proof <proof> --i-accept-validation-access; do not open sealed test.",
    }
    write_json(request_path, request)

    raw = {
        "created_utc": now,
        "purpose": "metadata-only backup blocker audit/request; no validation/test content and no simulation",
        "passed": True,
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "runner_sha256": runner_sha,
        "expected_runner_sha256": EXPECTED_RUNNER_SHA,
        "gate_sha256": gate_sha,
        "expected_gate_sha256": EXPECTED_GATE_SHA,
        "missing_required_paths": missing,
        "dry_run_completed_flags": {
            "passed": dry_completed.get("passed"),
            "validation_accessed": dry_completed.get("validation_accessed"),
            "validation64_bank_content_opened": dry_completed.get("validation64_bank_content_opened"),
            "test_accessed": dry_completed.get("test_accessed"),
            "sealed_test_bank_content_opened": dry_completed.get("sealed_test_bank_content_opened"),
            "new_simulations": dry_completed.get("new_simulations"),
            "new_gradient_steps": dry_completed.get("new_gradient_steps"),
        },
        "dry_run_hash_mismatches": dry_hash_mismatches,
        "repo_local_backup_proofs_seen": proofs,
        "adequate_post_dryrun_backup_proofs_seen": adequate,
        "formal_validation_files_present": formal_files,
        "formal_validation_blocked": len(adequate) == 0,
        "backup_request_path": rel(request_path),
        "backup_request_sha256": sha256(request_path),
        "required_path_hashes_current": hashes,
    }
    if missing or dry_hash_mismatches or gate_sha != EXPECTED_GATE_SHA or runner_sha != EXPECTED_RUNNER_SHA:
        raw["passed"] = False
    if formal_files:
        raw["partial_formal_validation_outputs_require_audit"] = True
    write_json(OUT_DIR / "raw.json", raw)

    summary = []
    summary.append("# Post-dry-run backup blocker audit")
    summary.append("")
    summary.append("Created UTC: `%s`." % now)
    summary.append("")
    summary.append("Result: backup gate remains **blocked**; adequate post-dry-run external backup proofs seen: `%d`." % len(adequate))
    summary.append("")
    summary.append("Validation/test access: validation_accessed=false, validation64_bank_content_opened=false, test_accessed=false, sealed_test_bank_content_opened=false. New simulations=0.")
    summary.append("")
    summary.append("Runner SHA256: `%s`; gate SHA256: `%s`." % (runner_sha, gate_sha))
    summary.append("")
    summary.append("Formal validation output files currently present: `%d`." % len(formal_files))
    summary.append("")
    summary.append("Backup request written: `%s` (sha256 `%s`)." % (rel(request_path), raw["backup_request_sha256"]))
    summary.append("")
    summary.append("Next permitted action: supervisor/external backup must verify remaining_changed_files=0 and a GitHub release asset SHA256 covering dry-run artifacts/docs/registry/proofs; then create a post-dryrun proof and run only shard0. Do not open sealed test.")
    (OUT_DIR / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    completed_hashes = {
        rel(OUT_DIR / "raw.json"): sha256(OUT_DIR / "raw.json"),
        rel(OUT_DIR / "summary.md"): sha256(OUT_DIR / "summary.md"),
        rel(request_path): sha256(request_path),
    }
    completed = {
        "passed": raw["passed"],
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "formal_validation_blocked": raw["formal_validation_blocked"],
        "adequate_post_dryrun_backup_proofs_seen": len(adequate),
        "backup_request_path": rel(request_path),
        "hashes": completed_hashes,
    }
    write_json(OUT_DIR / "completed.json", completed)
    completed["hashes"][rel(OUT_DIR / "completed.json")] = sha256(OUT_DIR / "completed.json")
    write_json(OUT_DIR / "completed.json", completed)

    doc_text = """
## 2026-09-26 post-dry-run backup blocker audit

UTC: {now}. Metadata-only blocker audit completed with no validation/test bank content opened and no simulations. The dry-run/gate/runner hashes remain consistent, but no adequate post-dry-run external backup proof is present in the repository. A new backup request was written at `{request}`. Formal vehicle validation64 shard0 remains blocked until the supervisor provides a verified external backup proof with `backup_verified=true`, `remaining_changed_files=0`, commit, GitHub release asset/download SHA256, runner/gate hashes, and dry-run artifact hashes. Sealed test remains closed.
""".format(now=now, request=rel(request_path)).strip()
    append_once(ROOT / "STATUS.md", DOC_MARKER, doc_text)
    append_once(ROOT / "RESEARCH_LOG.md", DOC_MARKER, doc_text)
    append_once(ROOT / "RESULTS_AUDIT.md", DOC_MARKER, doc_text)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER, doc_text)
    append_once(ROOT / "REPRODUCTION_PROTOCOL.md", DOC_MARKER, doc_text)

    print(json.dumps({
        "passed": raw["passed"],
        "formal_validation_blocked": raw["formal_validation_blocked"],
        "adequate_post_dryrun_backup_proofs_seen": len(adequate),
        "backup_request_path": rel(request_path),
        "backup_request_sha256": raw["backup_request_sha256"],
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
    }, sort_keys=True))
    if not raw["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
