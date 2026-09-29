#!/usr/bin/env python3
"""Backup-gate status capture for fresh-source confirmation runner.

Metadata-only diagnostic. It does not run simulations, open validation/test banks,
train, refit, or access credentials. It copies sanitized supervisor backup status
into repository artifacts and decides whether the prepared fresh-source
confirmation runner is externally recoverable enough to execute the frozen
136-episode development run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
BACKUP_RECEIPTS = SUP_STATE / "backup_receipts.jsonl"
BACKUP_INDEX = SUP_STATE / "backup_index.sqlite"

STAMP = "20260929T0955Z"
NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_backup_gate_status_v0"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_MD = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T0955_after_fresh_source_backup_gate_status.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
PROOF_IF_VERIFIED = BACKUP_DIR / f"backup_proof_{STAMP}_after_fresh_source_confirmation_runner.json"
RETRY_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_RUNNER_V0_{STAMP}.json"
DOC_MARKER = f"<!-- {NAME}-{STAMP} -->"

RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v0_runner.py"
EXPECTED_RUNNER_SHA = "ff3a6d315c5a91ba3369651b88ae5e9f1ad9b7f4f4472b1f96c857cbf87c34c8"
FREEZE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json"
FREEZE_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_20260929T1015Z/completed.json"
FREEZE_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_20260929T1015Z/summary.md"
FREEZE_REQUEST = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_FREEZE_V0_20260929T1015Z.json"
DEFAULT_BACKUP_ATTEMPT = ROOT / "research_artifacts/aws_runs/20260929T094849_9117e01a/registry.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|key_material)", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{65,}\b")


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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if SECRET_KEY_RE.search(key) and key.lower() not in {"sha256", "asset_sha256", "package_sha256"}:
                out[key] = "[REDACTED]"
            else:
                out[key] = sanitize(v)
        return out
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return LONG_HEX_RE.sub("[REDACTED_HEX]", value)
    return value


def safe_read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return sanitize(read_json(path)), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:500]}"


def path_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": None, "sha256": None, "mtime_utc": None}
    stat = path.stat()
    return {"path": rel(path), "exists": True, "bytes": stat.st_size, "sha256": sha256(path), "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat()}


def required_time(paths: Iterable[Path]) -> dt.datetime:
    times: List[dt.datetime] = []
    for path in paths:
        if path.exists():
            times.append(dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc))
    return max(times) if times else now_utc()


def git_cmd(args: List[str]) -> Dict[str, Any]:
    try:
        r = subprocess.run(["git"] + args, cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        return {"returncode": r.returncode, "stdout": r.stdout.strip()[:2000], "stderr": r.stderr.strip()[:1000]}
    except Exception as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def receipts_tail(n: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not BACKUP_RECEIPTS.exists():
        return [], "missing"
    try:
        lines = BACKUP_RECEIPTS.read_text(encoding="utf-8", errors="replace").splitlines()
        out: List[Dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                obj = sanitize(json.loads(line))
                out.append({"time": obj.get("time"), "asset": obj.get("asset"), "manifest": obj.get("manifest")})
            except Exception:
                out.append({"unparsed_prefix": line[:200]})
        return out, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:500]}"


def backup_index_probe() -> Dict[str, Any]:
    if not BACKUP_INDEX.exists():
        return {"exists": False}
    try:
        con = sqlite3.connect(str(BACKUP_INDEX))
        row_count = con.execute("select count(*) from files").fetchone()[0]
        interesting = []
        for key in [
            "mobile-robot-mppi-study/research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json",
            "mobile-robot-mppi-study/research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_20260929T1015Z/completed.json",
            "mobile-robot-mppi-study/research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_FREEZE_V0_20260929T1015Z.json",
        ]:
            hit = con.execute("select size, sha, asset from files where path=?", (key,)).fetchone()
            interesting.append({"path": key, "indexed": hit is not None, "size": None if hit is None else hit[0], "sha256": None if hit is None else hit[1], "asset": None if hit is None else hit[2]})
        con.close()
        return {"exists": True, "row_count": row_count, "interesting_artifacts": interesting}
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list):
        return False
    # A successful backup usually has at least the state snapshot package; require
    # public verification metadata when declaring the repo/artifact state recovered.
    if len(packages) == 0:
        return False
    for p in packages:
        if not isinstance(p, Mapping):
            return False
        if not p.get("sha256") or not p.get("bytes") or not p.get("verification"):
            return False
    return True


def classify(status: Optional[Mapping[str, Any]], min_required: dt.datetime, runner_sha: str, git_info: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(status, Mapping):
        return {"adequate_fresh_source_runner_backup": False, "why_not": ["backup_status_missing_or_unreadable"], "min_required_backup_time_utc": min_required.isoformat()}
    t = parse_time(status.get("time"))
    verified = status.get("status") == "verified" or status.get("backup_verified") is True
    try:
        remaining_ok = int(status.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    has_commit = bool(status.get("commit"))
    time_ok = bool(t is not None and t >= min_required)
    runner_hash_ok = runner_sha == EXPECTED_RUNNER_SHA
    packages_ok = package_metadata_ok(status)
    runner_git_clean = (git_info.get("runner_status") or {}).get("stdout", "") == ""
    why: List[str] = []
    if not verified:
        why.append(f"status_is_{status.get('status')!r}_not_verified")
    if not remaining_ok:
        why.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why.append("missing_commit")
    if not packages_ok:
        why.append("missing_verified_package_metadata")
    if not time_ok:
        why.append("status_time_predates_runner_or_unparseable")
    if not runner_hash_ok:
        why.append("runner_sha_mismatch")
    if not runner_git_clean:
        why.append("runner_has_uncommitted_or_untracked_git_status")
    return {
        "adequate_fresh_source_runner_backup": bool(verified and remaining_ok and has_commit and packages_ok and time_ok and runner_hash_ok and runner_git_clean),
        "why_not": why,
        "status": status.get("status"),
        "status_time": None if t is None else t.isoformat(),
        "remaining_changed_files": status.get("remaining_changed_files"),
        "commit": status.get("commit"),
        "package_count": len(status.get("packages_this_run") or []),
        "has_verified_package_metadata": packages_ok,
        "time_ok": time_ok,
        "runner_sha256": runner_sha,
        "expected_runner_sha256": EXPECTED_RUNNER_SHA,
        "runner_hash_ok": runner_hash_ok,
        "runner_git_clean": runner_git_clean,
        "min_required_backup_time_utc": min_required.isoformat(),
    }


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + text.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    d = raw["backup_gate_decision"]
    status = raw.get("backup_status_sanitized") or {}
    lines = [
        "# Fresh-source confirmation runner backup-gate status v0",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only: no simulations, no control steps, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Inputs checked",
        "",
        f"- Runner: `{raw['runner']['path']}` sha256 `{raw['runner']['sha256']}`; hash ok `{d['runner_hash_ok']}`; git-clean `{d['runner_git_clean']}`.",
        f"- Latest backup attempt registry: `{raw['backup_attempt_registry']['path']}` exit `{raw['backup_attempt_registry'].get('exit_status')}` failure `{raw['backup_attempt_registry'].get('failure_reason')}`.",
        f"- Supervisor backup status: `{status.get('status')}` at `{status.get('time')}`; remaining changed `{status.get('remaining_changed_files')}`; packages `{len(status.get('packages_this_run') or [])}`.",
        "",
        "## Backup gate decision",
        "",
        f"Adequate fresh-source-runner backup: `{d['adequate_fresh_source_runner_backup']}`.",
        f"Reasons if blocked: `{d['why_not']}`.",
        f"Minimum required backup time: `{d['min_required_backup_time_utc']}`.",
        "",
        "## Next action",
        "",
        str(raw["next_action"]),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-attempt-registry", default=str(DEFAULT_BACKUP_ATTEMPT))
    ap.add_argument("--i-accept-no-simulation-backup-status-capture", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_no_simulation_backup_status_capture:
        raise SystemExit("missing explicit no-simulation backup-status-capture acknowledgement")
    created_dt = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_dt.isoformat(), "pid": os.getpid(), "method": NAME, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})

    required = [RUNNER, FREEZE_PROTOCOL, FREEZE_COMPLETED, FREEZE_SUMMARY, FREEZE_REQUEST]
    missing = [rel(p) for p in required if not p.exists()]
    attempt_path = Path(args.backup_attempt_registry)
    if not attempt_path.is_absolute():
        attempt_path = ROOT / attempt_path
    attempt = read_json(attempt_path) if attempt_path.exists() else {}
    required_with_attempt = required + ([attempt_path] if attempt_path.exists() else [])
    min_required = required_time(required_with_attempt)
    status_obj, status_error = safe_read_json(BACKUP_STATUS)
    receipts, receipts_error = receipts_tail()
    runner_sha = sha256(RUNNER) if RUNNER.exists() else "missing"
    git_info = {
        "head": git_cmd(["rev-parse", "HEAD"]),
        "runner_status": git_cmd(["status", "--short", "--", rel(RUNNER)]),
        "runner_tracked": git_cmd(["ls-files", "--error-unmatch", rel(RUNNER)]),
    }
    decision = classify(status_obj if isinstance(status_obj, Mapping) else None, min_required, runner_sha, git_info)

    if decision["adequate_fresh_source_runner_backup"]:
        proof = dict(status_obj)  # type: ignore[arg-type]
        proof.update({
            "backup_verified": True,
            "created_utc": created_dt.isoformat(),
            "source": "sanitized copy from supervisor backup_status after fresh-source runner backup-gate status diagnostic",
            "covers_fresh_source_confirmation_runner": True,
            "runner_path": rel(RUNNER),
            "runner_sha256": runner_sha,
            "min_required_backup_time_utc": min_required.isoformat(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
        })
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        retry_request_path = None
        copied_proof = rel(PROOF_IF_VERIFIED)
        next_action = "Run the frozen 136-episode fresh-source confirmation runner with --backup-verified-commit set to the verified commit; do not open validation64 or sealed test."
    else:
        write_json(RETRY_REQUEST, {
            "requested_utc": created_dt.isoformat(),
            "reason": "fresh-source confirmation runner source/status not yet covered by an adequate verified external backup after scripts/research_service/backup.py failed",
            "backup_required_before_more_simulations": True,
            "backup_gate_decision": decision,
            "backup_attempt_registry": rel(attempt_path),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "artifacts_to_cover_before_runner": [rel(RUNNER), rel(FREEZE_PROTOCOL), rel(FREEZE_COMPLETED.parent), rel(FREEZE_REQUEST), rel(OUT), rel(STATE_MD), rel(CONTINUE_STATE), rel(RETRY_REQUEST)],
        })
        retry_request_path = rel(RETRY_REQUEST)
        copied_proof = None
        next_action = "Retry external backup once (or diagnose backup infrastructure if it repeats). Fresh-source confirmation simulations remain blocked until verified backup proof exists."

    raw: Dict[str, Any] = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "metadata_only_backup_gate_diagnostic_no_simulation_no_validation_no_test",
        "runner": path_info(RUNNER),
        "required_inputs": [path_info(p) for p in required],
        "missing_required_inputs": missing,
        "backup_attempt_registry": {"path": rel(attempt_path), "exists": attempt_path.exists(), "exit_status": attempt.get("exit_status"), "failure_reason": attempt.get("failure_reason"), "runtime_seconds": attempt.get("runtime_seconds"), "sha256": sha256(attempt_path) if attempt_path.exists() else None},
        "supervisor_backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": status_obj,
        "backup_receipts_file": path_info(BACKUP_RECEIPTS),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts,
        "backup_index_probe": backup_index_probe(),
        "git_info": git_info,
        "backup_gate_decision": decision,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request_path,
        "next_action": next_action,
        "budgets_actual": {"new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(
        f"# Continue state after fresh-source backup-gate status\n\nUTC: {created_dt.isoformat()}\nElapsed since first supervisor event: {(created_dt - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.\n\nNo simulation/training/validation/test was run. Adequate backup={decision['adequate_fresh_source_runner_backup']}; reasons={decision['why_not']}.\nSummary: `{rel(OUT / 'summary.md')}`. Completed: `{rel(OUT / 'completed.json')}`.\nNext action: {next_action}\n",
        encoding="utf-8",
    )
    doc_block = f"""## 2026-09-29 fresh-source confirmation runner backup-gate status

UTC: {created_dt.isoformat()}. Metadata-only backup-gate diagnostic completed; no simulations, no training/refit, no validation64 and no sealed test. Adequate fresh-source-runner backup=`{decision['adequate_fresh_source_runner_backup']}`; reasons=`{decision['why_not']}`. Backup attempt registry `{rel(attempt_path)}` exit `{attempt.get('exit_status')}`. Next action: {next_action}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, DOC_MARKER, doc_block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created_dt.isoformat()},{NAME},metadata_backup_gate,no_validation_no_test,0,0,0,0,0,{decision['adequate_fresh_source_runner_backup']},{rel(OUT / 'completed.json')}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_MD, CONTINUE_STATE, RUNNER, FREEZE_PROTOCOL, FREEZE_COMPLETED, FREEZE_SUMMARY, FREEZE_REQUEST]
    if attempt_path.exists():
        files.append(attempt_path)
    if PROOF_IF_VERIFIED.exists():
        files.append(PROOF_IF_VERIFIED)
    if RETRY_REQUEST.exists():
        files.append(RETRY_REQUEST)
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw["classification"],
        "adequate_fresh_source_runner_backup": decision["adequate_fresh_source_runner_backup"],
        "backup_gate_decision": decision,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request_path,
        "next_action": next_action,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_fresh_source_runner_backup": decision["adequate_fresh_source_runner_backup"], "why_not": decision["why_not"], "copied_verified_proof": copied_proof, "retry_backup_request": retry_request_path, "next_action": next_action, "new_simulations": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
