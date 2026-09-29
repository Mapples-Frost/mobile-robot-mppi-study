#!/usr/bin/env python3
"""Sanitized metadata-only capture after backup failure before v8c.

Purpose: backup.py failed with empty stdout/stderr while trying to protect
v8b partial evidence and the v8c repair source.  This script reads only
supervisor backup status/receipts and repository metadata, sanitizes public
fields, and writes a durable backup-gate decision.  It must not run backup.py,
read credentials, import vehicle/TF/controller code, generate scenarios, run
MPC rollouts, train/refit selectors, or access validation64/sealed-test banks.
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
STAMP = "20260929T1324Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/backup_failure_status_capture_after_v8c_prebackup_v0_{STAMP}"
STATE_MD = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_backup_failure_before_v8c.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_V8C_PREBACKUP_FAILURE_{STAMP}.json"
PROOF = BACKUP_DIR / f"backup_proof_{STAMP}_from_state_before_v8c.json"
FAILED_BACKUP_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T132120_54f69c3b/registry.json"
FAILED_V8B_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T131628_ec8e3d21/registry.json"
FAILED_V8B_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry_20260929T1312Z"
V8C_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count.py"
V8B_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T1320_after_v8b_failure_v8c_prebackup.md"
V8B_REQUEST = BACKUP_DIR / "REQUEST_BACKUP_AFTER_V8B_PARTIAL_FAILURE_BEFORE_V8C_20260929T1320Z.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|github\.token|key_material)", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{80,}\b")


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": None, "sha256": None, "mtime_utc": None}
    st = path.stat()
    return {"path": rel(path), "exists": True, "bytes": st.st_size, "sha256": sha256(path), "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat()}


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            ks = str(k)
            if SECRET_KEY_RE.search(ks) and ks.lower() not in {"sha256", "asset_sha256", "package_sha256"}:
                out[ks] = "[REDACTED]"
            else:
                out[ks] = sanitize(v)
        return out
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if isinstance(value, str):
        return LONG_HEX_RE.sub("[REDACTED_HEX]", value).replace(str(BASE / ".secrets"), "[REDACTED_SECRET_PATH]")
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
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def package_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list) or not packages:
        return False
    return all(isinstance(p, Mapping) and p.get("sha256") and p.get("bytes") and p.get("verification") for p in packages)


def classify(status: Optional[Mapping[str, Any]], min_required: dt.datetime) -> Dict[str, Any]:
    if not isinstance(status, Mapping):
        return {"status_file_readable": False, "adequate_backup_before_v8c": False, "why_not": ["status_file_missing_or_unreadable"], "min_required_backup_time_utc": min_required.isoformat()}
    t = parse_time(status.get("time"))
    verified = status.get("status") == "verified" or status.get("backup_verified") is True
    try:
        remaining_ok = int(status.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    has_commit = bool(status.get("commit"))
    packages = package_ok(status)
    time_ok = t is not None and t >= min_required
    why: List[str] = []
    if not verified:
        why.append(f"status_is_{status.get('status')!r}_not_verified")
    if not remaining_ok:
        why.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why.append("missing_commit")
    if not packages:
        why.append("missing_verified_package_metadata")
    if not time_ok:
        why.append("status_time_predates_required_v8c_artifacts_or_unparseable")
    return {
        "status_file_readable": True,
        "status_time": None if t is None else t.isoformat(),
        "status": status.get("status"),
        "error_type": status.get("error_type"),
        "message": status.get("message"),
        "remaining_changed_files": status.get("remaining_changed_files"),
        "commit": status.get("commit"),
        "package_count": len(status.get("packages_this_run") or []),
        "has_verified_package_metadata": packages,
        "time_ok": bool(time_ok),
        "min_required_backup_time_utc": min_required.isoformat(),
        "adequate_backup_before_v8c": bool(verified and remaining_ok and has_commit and packages and time_ok),
        "why_not": why,
    }


def receipts_tail(n: int = 6) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not BACKUP_RECEIPTS.exists():
        return [], "missing"
    try:
        lines = BACKUP_RECEIPTS.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:500]}"
    out: List[Dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            obj = sanitize(json.loads(line))
            out.append({"time": obj.get("time"), "asset": obj.get("asset"), "manifest": obj.get("manifest")})
        except Exception:
            out.append({"unparsed_line_prefix": line[:200]})
    return out, None


def git_meta() -> Dict[str, Any]:
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT="0")
    out: Dict[str, Any] = {}
    for name, args in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_porcelain": ["git", "status", "--porcelain"],
        "diff_cached_stat": ["git", "diff", "--cached", "--stat"],
    }.items():
        try:
            p = subprocess.run(args, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=30)
            out[name] = {"returncode": p.returncode, "stdout_prefix": sanitize(p.stdout[:6000]), "stderr_empty": not bool(p.stderr)}
            if p.stderr:
                out[name]["stderr_sanitized_prefix"] = sanitize(p.stderr[:500])
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return out


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-accept-no-simulation-backup-status-capture", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_accept_no_simulation_backup_status_capture:
        raise SystemExit("missing explicit no-simulation backup-status-capture acknowledgement")

    created_dt = now(); created = created_dt.isoformat(); OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created, "method": "backup_failure_status_capture_after_v8c_prebackup_v0", "classification": "metadata_only", "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
    required = [FAILED_BACKUP_REGISTRY, FAILED_V8B_REGISTRY, FAILED_V8B_DIR, V8C_SOURCE, V8B_STATE, V8B_REQUEST]
    missing = [rel(p) for p in required if not p.exists()]
    status_obj, status_error = safe_json(BACKUP_STATUS)
    receipts, receipts_error = receipts_tail()
    times = [dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) for p in required + [Path(__file__)] if p.exists()]
    min_required = max(times) if times else created_dt
    decision = classify(status_obj if isinstance(status_obj, Mapping) else None, min_required)
    failed_registry, failed_registry_error = safe_json(FAILED_BACKUP_REGISTRY)
    if decision["adequate_backup_before_v8c"]:
        proof = dict(status_obj)  # type: ignore[arg-type]
        proof.update({"backup_verified": True, "source": "sanitized copy from supervisor backup_status.json after v8c prebackup failure", "created_utc": created, "min_required_backup_time_utc": min_required.isoformat(), "covers_v8b_partial_and_v8c_source": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0})
        write_json(PROOF, sanitize(proof))
        next_action = "run v8c under legacy with --run --backup-verified-commit <commit> --i-accept-development-risk-probe-v8"
        decision_text = "Supervisor backup status is already verified and recent enough for v8c; science can proceed after using the recorded commit."
        retry_path = None
    else:
        write_json(REQUEST, {"requested_utc": created, "reason": "retry/fix external backup after backup.py failed before v8c", "failed_backup_registry": rel(FAILED_BACKUP_REGISTRY), "backup_status_classification": decision, "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "min_required_backup_time_utc": min_required.isoformat(), "scientific_effect": "none; infrastructure backup failure only", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "artifacts_requiring_backup": [rel(p) for p in required] + [rel(OUT), rel(STATE_MD), rel(REQUEST)]})
        next_action = "diagnose/fix backup infrastructure or obtain verified external backup; do not run v8c simulations until backup is verified"
        decision_text = "Backup remains inadequate for further formal evidence accumulation.  v8c risk-probe simulations are blocked until external recovery is verified."
        retry_path = rel(REQUEST)
    elapsed = (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds()
    raw: Dict[str, Any] = {
        "created_utc": created,
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "method": "backup_failure_status_capture_after_v8c_prebackup_v0",
        "classification": "metadata_only_infrastructure_no_simulation",
        "missing_required_artifacts": missing,
        "failed_backup_registry": rel(FAILED_BACKUP_REGISTRY),
        "failed_backup_registry_read_error": failed_registry_error,
        "failed_backup_registry_sanitized_tail_fields": failed_registry,
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": status_obj,
        "backup_receipts_file": path_info(BACKUP_RECEIPTS),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts,
        "backup_index_file": path_info(BACKUP_INDEX),
        "required_artifacts": [path_info(p) for p in required],
        "backup_gate_decision": decision,
        "git_metadata_sanitized": git_meta(),
        "copied_verified_proof": rel(PROOF) if PROOF.exists() else None,
        "retry_backup_request": retry_path,
        "decision_text": decision_text,
        "next_action": next_action,
        "budgets_actual": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "mobile_robot_mppi_resumed": False},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    summary = ["# Backup failure status capture after v8c prebackup", "", f"UTC: `{created}`. Metadata-only; no simulation/control/training/refit/validation64/sealed test.", "", f"Failed backup registry: `{rel(FAILED_BACKUP_REGISTRY)}`.", f"Supervisor status: `{decision.get('status')}` / `{decision.get('error_type')}`; message: `{decision.get('message')}`.", f"Adequate backup before v8c: `{decision['adequate_backup_before_v8c']}`; reasons: `{decision['why_not']}`.", "", f"Decision: {decision_text}", f"Next action: `{next_action}`."]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text("# Continue state after backup failure before v8c\n\n" + f"UTC: {created}. No simulation/training/refit/validation/test. Adequate backup before v8c={decision['adequate_backup_before_v8c']}. Reasons={decision['why_not']}. Error={decision.get('error_type')} message={decision.get('message')}. Next action: {next_action}.\n", encoding="utf-8")
    marker = "backup-failure-status-capture-after-v8c-prebackup-v0-20260929T1324Z"
    block = f"<!-- {marker} -->\n## 2026-09-29 backup failure capture before v8c\n\nUTC: {created}. Metadata-only after backup.py failed before v8c; no simulations/training/refit/validation64/sealed-test access. Adequate backup before v8c=`{decision['adequate_backup_before_v8c']}`; reasons=`{decision['why_not']}`; error=`{decision.get('error_type')}` message=`{decision.get('message')}`. Next action: `{next_action}`.\n"
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / name, marker, block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created},backup_failure_status_capture_after_v8c_prebackup_v0,metadata_no_simulation,infrastructure_backup_gate,0,0,0,0,0,{decision['adequate_backup_before_v8c']},{rel(OUT / 'completed.json')}\n"
    if reg.exists() and "backup_failure_status_capture_after_v8c_prebackup_v0" not in reg.read_text(encoding="utf-8", errors="replace")[-40000:]:
        reg.write_text(reg.read_text(encoding="utf-8", errors="replace").rstrip() + "\n" + row, encoding="utf-8")
    completed_paths = [OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_MD, REQUEST, PROOF] + required
    completed = {"passed": True, "hard_pass": True, "created_utc": created, "adequate_backup_before_v8c": decision["adequate_backup_before_v8c"], "backup_gate_decision": decision, "copied_verified_proof": rel(PROOF) if PROOF.exists() else None, "retry_backup_request": rel(REQUEST) if REQUEST.exists() else None, "next_action": next_action, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "hashes": {rel(p): sha256(p) for p in sorted(set(completed_paths)) if p.exists()}}
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_backup_before_v8c": decision["adequate_backup_before_v8c"], "why_not": decision["why_not"], "error_type": decision.get("error_type"), "message": decision.get("message"), "commit": decision.get("commit"), "copied_verified_proof": completed["copied_verified_proof"], "retry_backup_request": completed["retry_backup_request"], "next_action": next_action, "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
