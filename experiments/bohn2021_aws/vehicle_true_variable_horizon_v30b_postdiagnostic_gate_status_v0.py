#!/usr/bin/env python3
"""Metadata-only backup/Astra gate status after v30b postdiagnostic.

This script performs no simulation, no MPC rollout, no training/refit, and no
validation/test access. It records whether the v30b feature-stability
postdiagnostic is externally backed up and whether Astra has returned an
ANALYSIS_READY response for the current NEXT_REVIEW_REQUEST.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
STATE = BASE / "state"
BACKUP_STATUS = STATE / "backup_status.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v30b_postdiagnostic_gate_status_v0"
STAMP = "20260930T0400Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
CONTINUE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v30b_postdiagnostic_gate_status.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
PROOF_IF_OK = BACKUP_DIR / f"backup_proof_{STAMP}_from_supervisor_context_after_v30b_postdiagnostic.json"
REQUEST = BACKUP_DIR / "REQUEST_BACKUP_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_20260930T0355Z.json"
RETRY_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_GATE_STATUS_{STAMP}.json"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
DOC_MARKER = f"<!-- {NAME}-{STAMP} -->"

REQUIRED = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/completed.json",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0355Z_after_v30b_feature_stability_postdiagnostic.md",
    REQUEST,
    NEXT_REVIEW,
    RESPONSE_LOG,
]
EXPECTED_NEXT_REQUEST_ID = "v30b-feature-stability-postdiagnostic-20260930T0355Z"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_json(path: Path) -> Optional[Any]:
    try:
        return read_json(path)
    except Exception:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str):
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def path_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": None, "sha256": None, "mtime_utc": None}
    st = path.stat()
    return {"path": rel(path), "exists": True, "bytes": st.st_size, "sha256": sha256(path) if path.is_file() else None, "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat()}


def min_required_time(paths: list[Path]) -> dt.datetime:
    times = [dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) for p in paths if p.exists()]
    return max(times) if times else now()


def git(args: list[str]) -> Dict[str, Any]:
    try:
        p = subprocess.run(["git", *args], cwd=str(ROOT), text=True, capture_output=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:4000], "stderr": p.stderr.strip()[:1000]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def package_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list) or not packages:
        return False
    return all(isinstance(p, Mapping) and p.get("sha256") and p.get("bytes") and p.get("verification") for p in packages)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-metadata-only-gate-status", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_metadata_only_gate_status:
        raise SystemExit("missing explicit metadata-only gate-status acknowledgement")

    created = now()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created.isoformat(), "method": NAME, "new_simulations": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})

    required = REQUIRED + [Path(__file__).resolve()]
    min_time = min_required_time(required)
    missing = [rel(p) for p in required if not p.exists()]
    backup_status = safe_json(BACKUP_STATUS)
    status_time = parse_time(backup_status.get("time") if isinstance(backup_status, Mapping) else None)
    verified = bool(isinstance(backup_status, Mapping) and (backup_status.get("status") == "verified" or backup_status.get("backup_verified") is True))
    remaining_ok = False
    if isinstance(backup_status, Mapping):
        try:
            remaining_ok = int(backup_status.get("remaining_changed_files", -1)) == 0
        except Exception:
            remaining_ok = False
    time_ok = bool(status_time and status_time >= min_time)
    packages_ok = bool(isinstance(backup_status, Mapping) and package_ok(backup_status))
    adequate_backup = bool((not missing) and verified and remaining_ok and packages_ok and time_ok)
    why_not = []
    if missing:
        why_not.append("missing_required_artifacts")
    if not verified:
        why_not.append("backup_status_not_verified_or_missing")
    if not remaining_ok:
        why_not.append("remaining_changed_files_not_zero_or_missing")
    if not packages_ok:
        why_not.append("missing_verified_package_metadata")
    if not time_ok:
        why_not.append("backup_time_predates_postdiagnostic_or_unparseable")

    next_review = safe_json(NEXT_REVIEW) or {}
    analysis_ready = safe_json(ANALYSIS_READY)
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    ready_request_id = analysis_ready.get("request_id") if isinstance(analysis_ready, Mapping) else None
    astra_ready_for_current = bool(analysis_ready and ready_request_id == current_request_id == EXPECTED_NEXT_REQUEST_ID)

    if adequate_backup and isinstance(backup_status, Mapping):
        proof = dict(backup_status)
        proof.update({
            "backup_verified": True,
            "created_utc": created.isoformat(),
            "source": "sanitized supervisor backup_status captured by v30b postdiagnostic gate status",
            "covers_v30b_feature_stability_postdiagnostic": True,
            "min_required_backup_time_utc": min_time.isoformat(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
        })
        write_json(PROOF_IF_OK, proof)
        backup_action = "backup gate passed; use proof before any next unique science"
        backup_proof_path = rel(PROOF_IF_OK)
        retry_request_path = None
    else:
        write_json(RETRY_REQUEST, {
            "requested_utc": created.isoformat(),
            "reason": "post-v30b feature-stability postdiagnostic not yet covered by adequate verified external backup",
            "backup_required_before_more_unique_science": True,
            "backup_gate_why_not": why_not,
            "min_required_backup_time_utc": min_time.isoformat(),
            "latest_backup_status_time_utc": None if status_time is None else status_time.isoformat(),
            "must_cover": [rel(p) for p in required] + [rel(OUT), rel(CONTINUE), rel(RETRY_REQUEST)],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
        })
        backup_action = "obtain external backup; do not run unique simulations/refits/validation"
        backup_proof_path = None
        retry_request_path = rel(RETRY_REQUEST)

    if not adequate_backup:
        next_action = "Wait for/obtain verified backup covering this gate status and v30b postdiagnostic, then re-check ANALYSIS_READY.json."
    elif astra_ready_for_current:
        next_action = "Read Astra linked report and implement its selected next execution task."
    else:
        next_action = "Backup is adequate but Astra analysis is still pending; continue only reversible implementation/data-integrity preparation within the already approved plan."

    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "metadata_only_backup_and_astra_gate_no_sim_no_validation_no_test_no_training",
        "required_artifacts": [path_info(p) for p in required],
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_subset": None if not isinstance(backup_status, Mapping) else {k: backup_status.get(k) for k in ("status", "time", "remaining_changed_files", "commit", "changed_files", "packages_this_run", "release", "tracked_files")},
        "backup_gate": {"adequate_backup": adequate_backup, "why_not": why_not, "min_required_backup_time_utc": min_time.isoformat(), "status_time_utc": None if status_time is None else status_time.isoformat(), "copied_verified_proof": backup_proof_path, "retry_backup_request": retry_request_path},
        "astra_gate": {"next_review_request_exists": NEXT_REVIEW.exists(), "next_review_request_id": current_request_id, "expected_request_id": EXPECTED_NEXT_REQUEST_ID, "analysis_ready_exists": ANALYSIS_READY.exists(), "analysis_ready_request_id": ready_request_id, "analysis_ready_for_current_request": astra_ready_for_current, "analysis_ready_report": None if not isinstance(analysis_ready, Mapping) else analysis_ready.get("report")},
        "git": {"head": git(["rev-parse", "HEAD"]), "status_critical": git(["status", "--short", "--", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic.py", "docs/bohn2021_takeover/astra_reviews", "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z", "research_artifacts/aws_state/continue_state_20260930T0355Z_after_v30b_feature_stability_postdiagnostic.md", "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_20260930T0355Z.json"])},
        "backup_action": backup_action,
        "next_action": next_action,
        "budgets_actual": {"new_simulations": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)

    summary = "\n".join([
        "# v30b postdiagnostic backup/Astra gate status",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only: no simulations, no control steps, no training/refit, no validation64, no sealed test.",
        "",
        "## Backup gate",
        f"- Adequate backup: `{adequate_backup}`.",
        f"- Reasons if blocked: `{why_not}`.",
        f"- Backup status time: `{raw['backup_gate']['status_time_utc']}`; minimum required: `{raw['backup_gate']['min_required_backup_time_utc']}`.",
        f"- Proof copied: `{backup_proof_path}`; retry request: `{retry_request_path}`.",
        "",
        "## Astra gate",
        f"- NEXT_REVIEW_REQUEST id: `{current_request_id}`.",
        f"- ANALYSIS_READY present: `{ANALYSIS_READY.exists()}`; request id: `{ready_request_id}`; matches current: `{astra_ready_for_current}`.",
        "",
        "## Next executor action",
        next_action,
    ]) + "\n"
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    CONTINUE.parent.mkdir(parents=True, exist_ok=True)
    CONTINUE.write_text(summary, encoding="utf-8")

    doc_block = f"""## 2026-09-30 v30b postdiagnostic backup/Astra gate status

UTC: {created.isoformat()}. Metadata-only gate status completed; no simulation/control/training/refit/validation64/sealed-test access. Adequate backup=`{adequate_backup}` with reasons=`{why_not}`. Astra ready for current request `{EXPECTED_NEXT_REQUEST_ID}`=`{astra_ready_for_current}`. Next action: {next_action}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, DOC_MARKER, doc_block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created.isoformat()},{NAME},metadata_backup_astra_gate,no_validation_no_test,0,0,0,0,0,{adequate_backup},{rel(OUT / 'completed.json')}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [Path(__file__).resolve(), OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", CONTINUE, REQUEST, RETRY_REQUEST, PROOF_IF_OK] + required
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw["classification"],
        "backup_gate": raw["backup_gate"],
        "astra_gate": raw["astra_gate"],
        "next_action": next_action,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda x: rel(x)) if p.exists() and p.is_file()},
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_backup": adequate_backup, "backup_why_not": why_not, "astra_ready_for_current_request": astra_ready_for_current, "next_action": next_action, "new_simulations": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
