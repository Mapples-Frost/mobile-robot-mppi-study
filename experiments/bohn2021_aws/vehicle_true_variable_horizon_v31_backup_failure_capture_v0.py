#!/usr/bin/env python3
"""Metadata-only capture of the post-v31 backup failure/status.

This script is deliberately limited to operational integrity.  It does not run
MPC, does not import controller/environment code, does not train or refit a
selector, and does not open validation64 or sealed-test banks.  It records the
current supervisor backup status/receipts (sanitized) after the reported 04:12
HTTP-502 backup failure, re-checks the Astra handoff gate, and writes a durable
state/backup request so the next iteration can resume safely.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
BACKUP_RECEIPTS = SUP_STATE / "backup_receipts.jsonl"
BACKUP_INDEX = SUP_STATE / "backup_index.sqlite"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

NAME = "vehicle_true_variable_horizon_v31_backup_failure_capture_v0"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST_REVIEW = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"

# Artifacts that must already be externally recoverable before new simulations,
# training/refit, validation64 access, or final-test work.  They intentionally
# cover v31 plus the first v31 gate-status run; this script creates an additional
# request for its own outputs at the end.
PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE: List[Path] = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/run_started.json",
    ROOT / "research_artifacts/aws_runs/20260930T040615_4f9729e6/registry.json",
    ROOT / "research_artifacts/aws_runs/20260930T041028_4dccdd03/registry.json",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0405Z_after_v31_cluster_stability_diagnostic.md",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0407_after_v31_executor_state.md",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0415Z_after_v31_backup_astra_gate_status.md",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_20260930T0405Z.json",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_BACKUP_ASTRA_GATE_STATUS_20260930T0415Z.json",
    NEXT_REVIEW,
    RESPONSE_LOG,
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    ROOT / "EXPERIMENT_REGISTRY.csv",
]

SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|github\.token|key_material|private_key)", re.I)
SECRET_PATH_RE = re.compile(r"(/[^\s\"']*)?\.secrets[^\s\"']*", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{80,}\b")


def now_utc() -> dt.datetime:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            k = str(key)
            if SECRET_KEY_RE.search(k) and k.lower() not in {"sha256", "asset_sha256", "package_sha256"}:
                out[k] = "[REDACTED]"
            else:
                out[k] = sanitize(val)
        return out
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if isinstance(value, str):
        out = SECRET_PATH_RE.sub("[REDACTED_SECRET_PATH]", value)
        out = LONG_HEX_RE.sub("[REDACTED_LONG_HEX]", out)
        return out
    return value


def safe_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return sanitize(read_json(path)), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:500]}"


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def path_info(path: Path, include_hash: bool = True) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": None, "sha256": None, "mtime_utc": None}
    st = path.stat()
    return {
        "path": rel(path),
        "exists": True,
        "bytes": st.st_size if path.is_file() else None,
        "sha256": sha256(path) if include_hash and path.is_file() else None,
        "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
        "is_dir": path.is_dir(),
    }


def max_existing_mtime(paths: List[Path], fallback: dt.datetime) -> dt.datetime:
    mtimes: List[dt.datetime] = []
    for p in paths:
        if p.exists():
            mtimes.append(dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc))
    return max(mtimes) if mtimes else fallback


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list) or not packages:
        return False
    for p in packages:
        if not isinstance(p, Mapping):
            return False
        if not p.get("sha256") or not p.get("bytes") or not p.get("verification"):
            return False
    return True


def compact_backup_status(status: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(status, Mapping):
        return None
    keys = [
        "time",
        "status",
        "backup_verified",
        "error_type",
        "message",
        "remaining_changed_files",
        "commit",
        "changed_files",
        "packages_this_run",
        "release",
        "tracked_files",
    ]
    return {k: status.get(k) for k in keys if k in status}


def classify_backup(status: Any, min_required: dt.datetime, missing_required: List[str]) -> Dict[str, Any]:
    status_mapping = isinstance(status, Mapping)
    status_time = parse_time(status.get("time") if status_mapping else None)  # type: ignore[union-attr]
    verified = bool(status_mapping and (status.get("status") == "verified" or status.get("backup_verified") is True))  # type: ignore[union-attr]
    try:
        remaining_ok = bool(status_mapping and int(status.get("remaining_changed_files", -1)) == 0)  # type: ignore[union-attr]
    except Exception:
        remaining_ok = False
    has_commit = bool(status_mapping and status.get("commit"))  # type: ignore[union-attr]
    packages_ok = bool(status_mapping and package_metadata_ok(status))  # type: ignore[arg-type]
    time_ok = bool(status_time and status_time >= min_required)
    why_not: List[str] = []
    if missing_required:
        why_not.append("missing_pre_existing_required_artifacts")
    if not status_mapping:
        why_not.append("backup_status_missing_or_unreadable")
    if not verified:
        why_not.append("backup_status_not_verified")
    if not remaining_ok:
        why_not.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why_not.append("missing_backup_commit")
    if not packages_ok:
        why_not.append("missing_verified_package_metadata")
    if not time_ok:
        why_not.append("backup_time_predates_required_artifacts_or_unparseable")
    adequate = bool((not missing_required) and verified and remaining_ok and has_commit and packages_ok and time_ok)
    return {
        "adequate_backup_for_pre_existing_v31_and_gate_outputs": adequate,
        "why_not": why_not,
        "status_time_utc": None if status_time is None else status_time.isoformat(),
        "min_required_backup_time_utc": min_required.isoformat(),
        "status": status.get("status") if status_mapping else None,  # type: ignore[union-attr]
        "error_type": status.get("error_type") if status_mapping else None,  # type: ignore[union-attr]
        "message": status.get("message") if status_mapping else None,  # type: ignore[union-attr]
        "remaining_changed_files": status.get("remaining_changed_files") if status_mapping else None,  # type: ignore[union-attr]
        "commit": status.get("commit") if status_mapping else None,  # type: ignore[union-attr]
        "package_count": len(status.get("packages_this_run") or []) if status_mapping else 0,  # type: ignore[union-attr]
        "packages_metadata_ok": packages_ok,
        "time_ok": time_ok,
        "missing_required_artifacts": missing_required,
    }


def receipts_tail(n: int = 8) -> Tuple[List[Any], Optional[str]]:
    if not BACKUP_RECEIPTS.exists():
        return [], "missing"
    try:
        lines = BACKUP_RECEIPTS.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:500]}"
    out: List[Any] = []
    for line in lines[-n:]:
        try:
            obj = sanitize(json.loads(line))
            if isinstance(obj, Mapping):
                out.append({k: obj.get(k) for k in ["time", "asset", "asset_sha256", "manifest", "status", "bytes", "verification"] if k in obj})
            else:
                out.append(obj)
        except Exception:
            out.append({"unparsed_line_prefix": sanitize(line[:200])})
    return out, None


def ready_matches_current(ready: Any, current_request_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not current_request_id:
        return False
    if ready.get("request_id") == current_request_id:
        return True
    if ready.get("supersedes_request_id") == current_request_id:
        return True
    covered = ready.get("covers_request_ids") or ready.get("covered_request_ids")
    return isinstance(covered, list) and current_request_id in covered


def git(args: List[str]) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0")
    try:
        p = subprocess.run(["git", *args], cwd=str(ROOT), text=True, capture_output=True, timeout=30, env=env)
        return {
            "returncode": p.returncode,
            "stdout": sanitize(p.stdout.strip()[:5000]),
            "stderr": sanitize(p.stderr.strip()[:1000]),
        }
    except Exception as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-accept-metadata-only-backup-failure-capture", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_accept_metadata_only_backup_failure_capture:
        raise SystemExit("missing explicit metadata-only backup-failure-capture acknowledgement")

    created_dt = now_utc()
    stamp = created_dt.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    continue_md = ROOT / f"research_artifacts/aws_state/continue_state_{stamp}_after_v31_backup_failure_capture.md"
    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_BACKUP_FAILURE_CAPTURE_{stamp}.json"
    proof_path = BACKUP_DIR / f"backup_proof_{stamp}_from_supervisor_context_after_v31_gate_outputs.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_started.json", {
        "started_utc": created_dt.isoformat(),
        "method": NAME,
        "classification": "metadata_only_backup_failure_status_no_science",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    missing_required = [rel(p) for p in PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE if not p.exists()]
    min_required = max_existing_mtime(PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE, created_dt)
    backup_status, backup_status_error = safe_json(BACKUP_STATUS)
    backup_decision = classify_backup(backup_status, min_required, missing_required)
    receipts, receipts_error = receipts_tail()

    next_review, next_review_error = safe_json(NEXT_REVIEW)
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    analysis_ready, analysis_ready_error = safe_json(ANALYSIS_READY)
    astra_ready = ready_matches_current(analysis_ready, current_request_id)
    report_path = None
    if isinstance(analysis_ready, Mapping):
        report_path = analysis_ready.get("report") or analysis_ready.get("report_path")

    adequate_pre_existing_backup = bool(backup_decision["adequate_backup_for_pre_existing_v31_and_gate_outputs"])
    copied_proof: Optional[str] = None
    if adequate_pre_existing_backup and isinstance(backup_status, Mapping):
        proof = dict(compact_backup_status(backup_status) or {})
        proof.update({
            "backup_verified": True,
            "source": "sanitized supervisor backup_status captured by v31 backup-failure/status script",
            "created_utc": created_dt.isoformat(),
            "covers_pre_existing_v31_and_gate_outputs": True,
            "min_required_backup_time_utc": min_required.isoformat(),
            "required_artifacts": [rel(p) for p in PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "note": "This proof covers artifacts that existed before this metadata-only capture. The capture outputs themselves still need routine backup.",
        })
        write_json(proof_path, proof)
        copied_proof = rel(proof_path)

    if not adequate_pre_existing_backup:
        next_action = "Retry/obtain verified external backup covering v31 and gate-status outputs before any new simulation, selector refit, training, validation64, or final-test work. Continue only reversible integrity/Astra-readiness checks."
    elif astra_ready:
        next_action = "After routine backup of this metadata-only capture, read the matching/superseding Astra report and implement its selected plan."
    else:
        next_action = "Pre-existing v31 outputs are backed up, but Astra analysis is still pending; after backing up this capture, continue only reversible preparation/integrity checks, not a new scientific branch."

    request_payload = {
        "requested_utc": created_dt.isoformat(),
        "reason": "post-v31 backup failure/status capture after reported HTTP-502 external backup failure; protect v31/gate outputs and this status capture before more unique science",
        "backup_gate_decision": backup_decision,
        "latest_backup_status_subset": compact_backup_status(backup_status),
        "backup_status_read_error": backup_status_error,
        "must_cover": [rel(p) for p in PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE] + [rel(Path(__file__).resolve()), rel(out_dir), rel(continue_md), rel(request_path), rel(proof_path)],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    }
    write_json(request_path, request_payload)

    elapsed = (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds()
    raw = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "method": NAME,
        "classification": "metadata_only_backup_failure_status_no_sim_no_validation_no_test_no_training_no_refit",
        "pre_existing_required_artifacts": [path_info(p) for p in PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE],
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": backup_status_error,
        "backup_status_subset": compact_backup_status(backup_status),
        "backup_receipts_file": path_info(BACKUP_RECEIPTS, include_hash=False),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts,
        "backup_index_file": path_info(BACKUP_INDEX, include_hash=False),
        "backup_gate": {
            **backup_decision,
            "copied_verified_proof": copied_proof,
            "backup_request_after_this_capture": rel(request_path),
        },
        "astra_gate": {
            "latest_md_exists": LATEST_REVIEW.exists(),
            "next_review_request_exists": NEXT_REVIEW.exists(),
            "next_review_request_read_error": next_review_error,
            "next_review_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "next_request_matches_expected": current_request_id == EXPECTED_REQUEST_ID,
            "analysis_ready_exists": ANALYSIS_READY.exists(),
            "analysis_ready_read_error": analysis_ready_error,
            "analysis_ready_request_id": analysis_ready.get("request_id") if isinstance(analysis_ready, Mapping) else None,
            "analysis_ready_matches_or_supersedes_current": astra_ready,
            "analysis_ready_report": report_path,
        },
        "git": {
            "head": git(["rev-parse", "HEAD"]),
            "critical_status": git(["status", "--short", "--", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0.py", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_failure_capture_v0.py", "docs/bohn2021_takeover/astra_reviews", "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z", "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z", "research_artifacts/aws_state/continue_state_20260930T0415Z_after_v31_backup_astra_gate_status.md", "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_BACKUP_ASTRA_GATE_STATUS_20260930T0415Z.json"]),
        },
        "budgets_actual": {
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
        },
        "next_action": next_action,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(out_dir / "raw.json", raw)

    summary = "\n".join([
        "# v31 backup failure/status capture",
        "",
        f"UTC: `{created_dt.isoformat()}`. Metadata-only infrastructure/Astra-readiness capture; no simulations, no control steps, no selector refits, no training, no validation64, no sealed test.",
        "",
        "## Backup gate",
        f"- Adequate backup for pre-existing v31 + gate-status outputs: `{adequate_pre_existing_backup}`.",
        f"- Supervisor backup status: `{backup_decision.get('status')}`; error_type: `{backup_decision.get('error_type')}`; message: `{backup_decision.get('message')}`.",
        f"- Reasons if blocked: `{backup_decision.get('why_not')}`.",
        f"- Status time: `{backup_decision.get('status_time_utc')}`; minimum required artifact mtime: `{backup_decision.get('min_required_backup_time_utc')}`.",
        f"- Copied verified proof, if any: `{copied_proof}`.",
        f"- Backup request after this capture: `{rel(request_path)}`.",
        "",
        "## Astra gate",
        f"- Current NEXT_REVIEW_REQUEST id: `{current_request_id}` (expected `{EXPECTED_REQUEST_ID}`).",
        f"- ANALYSIS_READY present: `{ANALYSIS_READY.exists()}`; matching/superseding current request: `{astra_ready}`; report: `{report_path}`.",
        "",
        "## Next executor action",
        next_action,
    ]) + "\n"
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_md.parent.mkdir(parents=True, exist_ok=True)
    continue_md.write_text(summary, encoding="utf-8")

    doc_marker = f"<!-- {NAME}-{stamp} -->"
    doc_block = f"""## 2026-09-30 v31 backup failure/status capture

UTC: {created_dt.isoformat()}. Metadata-only operational capture after the post-v31 backup failure/status check; no simulation/control/training/refit/validation64/sealed-test access. Adequate backup for pre-existing v31 plus gate-status outputs=`{adequate_pre_existing_backup}`; reasons=`{backup_decision.get('why_not')}`; supervisor backup status=`{backup_decision.get('status')}` error_type=`{backup_decision.get('error_type')}` message=`{backup_decision.get('message')}`. Astra ready for current request `{current_request_id}`=`{astra_ready}`. Next action: {next_action}. Artifacts: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, doc_marker, doc_block)

    response_marker = f"<!-- response-log-{NAME}-{stamp} -->"
    response_block = f"""## Operational follow-up: v31 backup failure/status capture

Updated by GPT-5.5 executor at `{created_dt.isoformat()}`. This entry addresses `A12_registry_backup_schema_contract` only; it is not a new scientific result. No simulations, control steps, selector refits, training, validation64 access, or sealed-test access occurred.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; blocker remains if backup inadequate | Backup gate adequate=`{adequate_pre_existing_backup}`; supervisor status=`{backup_decision.get('status')}`, error_type=`{backup_decision.get('error_type')}`, message=`{backup_decision.get('message')}`; reasons=`{backup_decision.get('why_not')}`. Evidence: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`. | `{next_action}` |
| Astra v31 handoff | pending unless ANALYSIS_READY matches current request | Current request id=`{current_request_id}`; ANALYSIS_READY present=`{ANALYSIS_READY.exists()}`; matches/supersedes=`{astra_ready}`. | If a matching report appears, read it at next safe boundary and implement after backup gate permits. |
"""
    append_once(RESPONSE_LOG, response_marker, response_block)

    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created_dt.isoformat()},{NAME},metadata_backup_failure_status,no_validation_no_test,0,0,0,0,0,{adequate_pre_existing_backup},{rel(out_dir / 'completed.json')}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [
        Path(__file__).resolve(),
        out_dir / "run_started.json",
        out_dir / "raw.json",
        out_dir / "summary.md",
        continue_md,
        request_path,
        proof_path,
        RESPONSE_LOG,
        ROOT / "STATUS.md",
        ROOT / "RESEARCH_LOG.md",
        ROOT / "DECISIONS.md",
        ROOT / "RESULTS_AUDIT.md",
        ROOT / "REPRODUCTION_PROTOCOL.md",
        ROOT / "EXPERIMENT_REGISTRY.csv",
    ] + PRE_EXISTING_REQUIRED_FOR_NEXT_SCIENCE
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
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
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request_after_this_capture": rel(request_path),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda x: rel(x)) if p.exists() and p.is_file()},
    }
    write_json(out_dir / "completed.json", completed)

    print(json.dumps({
        "completed": rel(out_dir / "completed.json"),
        "summary": rel(out_dir / "summary.md"),
        "adequate_backup_for_pre_existing_v31_and_gate_outputs": adequate_pre_existing_backup,
        "backup_status": backup_decision.get("status"),
        "backup_error_type": backup_decision.get("error_type"),
        "backup_message": backup_decision.get("message"),
        "backup_why_not": backup_decision.get("why_not"),
        "backup_request_after_this_capture": rel(request_path),
        "astra_ready_for_current_request": astra_ready,
        "analysis_ready_report": report_path,
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "next_action": next_action,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
