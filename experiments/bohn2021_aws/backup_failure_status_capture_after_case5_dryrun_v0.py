#!/usr/bin/env python3
"""Sanitized no-simulation capture of backup status after case5 dry-run backup failure.

The external-backup program writes its operational state under
`/data/openai-agent/state`, outside the repository tree.  The experiment registry
therefore cannot directly archive the failure reason.  This script copies only
non-secret, sanitized status/receipt metadata into repository artifacts and
re-evaluates whether a verified post-dryrun backup proof exists before any more
simulation is allowed.

It must not run backups, touch credentials, import TF/legacy controllers, open
validation64 or sealed-test banks, generate candidates, reset environments, or
execute rollouts.  It only reads already-created backup status files and repo
metadata, writes a diagnostic summary, and if needed writes a new backup retry
request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
BACKUP_RECEIPTS = SUP_STATE / "backup_receipts.jsonl"
BACKUP_INDEX = SUP_STATE / "backup_index.sqlite"

STAMP = "20260929T0600Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/backup_failure_status_capture_after_case5_dryrun_v0_{STAMP}"
STATE_MD = ROOT / f"research_artifacts/aws_state/backup_failure_status_capture_after_case5_dryrun_v0_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
RETRY_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_CASE5_DRYRUN_BACKUP_FAILURE_{STAMP}.json"
PROOF_IF_VERIFIED = BACKUP_DIR / f"backup_proof_{STAMP}_from_state_after_case5_dryrun.json"

CASE5_DRYRUN_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_dryrun_20260929T0605Z/completed.json"
CASE5_DRYRUN_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_dryrun_20260929T0605Z/summary.md"
CASE5_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_case5_positive_stability_timing_v0_frozen_20260929T0605Z.json"
CASE5_RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_stress_v1e_case5_positive_stability_timing_v0_runner.py"
CASE5_REQUEST = BACKUP_DIR / "REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_CASE5_POSITIVE_STABILITY_TIMING_V0_RUN_20260929T0605Z.json"
FAILED_BACKUP_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T055736_7b5b40b7/registry.json"
CONTINUE_STATE_AFTER_FAILURE = ROOT / "research_artifacts/aws_state/continue_state_20260929T0558_after_case5_backup_failure.md"

FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|key_material)", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{40,}\b")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": None, "sha256": None, "mtime_utc": None}
    stat = path.stat()
    return {
        "path": rel(path),
        "exists": True,
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
    }


def sanitize(value: Any) -> Any:
    """Remove possible secrets without discarding useful backup metadata."""
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if SECRET_KEY_RE.search(str(k)):
                # Keep public package SHA fields; redact only likely credential keys.
                if str(k).lower() in {"sha256", "asset_sha256", "release_asset_sha256", "package_sha256"}:
                    out[str(k)] = sanitize(v)
                else:
                    out[str(k)] = "[REDACTED]"
            else:
                out[str(k)] = sanitize(v)
        return out
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        # Keep ordinary SHA256/package digests; redact only very long hex blobs that
        # are not exactly a SHA256-like 64-character digest or git SHA-sized 40 chars.
        def repl(match: re.Match[str]) -> str:
            text = match.group(0)
            if len(text) in (40, 64):
                return text
            return "[REDACTED_HEX]"
        return LONG_HEX_RE.sub(repl, value)
    return value


def safe_read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return sanitize(read_json(path)), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:500]}"


def read_receipts_tail(path: Path, n: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return [], "missing"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:500]}"
    receipts: List[Dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            obj = sanitize(json.loads(line))
            # Keep only public audit fields.
            receipts.append({
                "time": obj.get("time"),
                "asset": obj.get("asset"),
                "manifest": obj.get("manifest"),
            })
        except Exception:
            receipts.append({"unparsed_line_prefix": line[:200]})
    return receipts, None


def package_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list) or not packages:
        return False
    for pkg in packages:
        if not isinstance(pkg, Mapping):
            return False
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            return False
    return True


def max_required_time(paths: Iterable[Path], dryrun_completed: Mapping[str, Any]) -> dt.datetime:
    times: List[dt.datetime] = []
    t = parse_time(dryrun_completed.get("created_utc"))
    if t is not None:
        times.append(t)
    for path in paths:
        if not path.exists():
            continue
        times.append(dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc))
    return max(times) if times else now()


def classify_backup_status(status_obj: Optional[Mapping[str, Any]], min_required: dt.datetime) -> Dict[str, Any]:
    if not isinstance(status_obj, Mapping):
        return {
            "status_file_readable": False,
            "adequate_post_case5_dryrun_backup": False,
            "why_not": ["status_file_missing_or_unreadable"],
        }
    t = parse_time(status_obj.get("time"))
    verified = status_obj.get("status") == "verified" or status_obj.get("backup_verified") is True
    try:
        remaining_ok = int(status_obj.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    has_commit = bool(status_obj.get("commit"))
    packages = package_ok(status_obj)
    time_ok = bool(t is not None and t >= min_required)
    why: List[str] = []
    if not verified:
        why.append(f"status_is_{status_obj.get('status')!r}_not_verified")
    if not remaining_ok:
        why.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why.append("missing_commit")
    if not packages:
        why.append("missing_verified_package_metadata")
    if not time_ok:
        why.append("status_time_predates_required_case5_dryrun_or_unparseable")
    return {
        "status_file_readable": True,
        "status_time": None if t is None else t.isoformat(),
        "status": status_obj.get("status"),
        "remaining_changed_files": status_obj.get("remaining_changed_files"),
        "commit": status_obj.get("commit"),
        "package_count": len(status_obj.get("packages_this_run") or []),
        "has_verified_package_metadata": packages,
        "time_ok": time_ok,
        "min_required_backup_time_utc": min_required.isoformat(),
        "adequate_post_case5_dryrun_backup": bool(verified and remaining_ok and has_commit and packages and time_ok),
        "why_not": why,
    }


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    marker = "backup-failure-status-capture-after-case5-dryrun-v0-20260929T0600Z"
    decision = raw["backup_gate_decision"]
    lines = [
        f"<!-- {marker} -->",
        "## 2026-09-29 backup status capture after case5 dry-run backup failure",
        "",
        f"UTC: {raw['created_utc']}. Metadata-only capture of supervisor backup state completed with no simulations, no control steps, no training/refit, no validation64-bank access and no sealed-test access.",
        f"The preceding backup attempt registry is `{raw['failed_backup_attempt_registry']}` with exit status `{raw['failed_backup_attempt_exit_status']}` and runtime seconds `{raw['failed_backup_attempt_runtime_seconds']}`.",
        f"Backup status classification: adequate post-case5-dryrun backup=`{decision['adequate_post_case5_dryrun_backup']}`; reasons=`{decision['why_not']}`.",
        f"Minimum required backup time for the case5 run is `{decision.get('min_required_backup_time_utc')}`.",
        f"Next action: `{raw['next_action']}`. Artifacts: `{raw['summary_path']}`, `{raw['completed_path']}`.",
    ]
    block = "\n".join(lines) + "\n"
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, marker, block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = (
        f"{raw['created_utc']},backup_failure_status_capture_after_case5_dryrun_v0,metadata_no_simulation,"
        f"no_validation_no_test,0,0,0,0,0,{decision['adequate_post_case5_dryrun_backup']},"
        f"{raw['completed_path']}\n"
    )
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if "backup_failure_status_capture_after_case5_dryrun_v0" not in old[-20000:]:
            reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    decision = raw["backup_gate_decision"]
    status = raw.get("backup_status_sanitized") or {}
    lines = [
        "# Backup status capture after case5 dry-run backup failure",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no control steps, no training/refit, no validation64 bank and no sealed test.",
        "",
        "## Backup attempt inspected",
        "",
        f"- Failed registry: `{raw['failed_backup_attempt_registry']}`.",
        f"- Exit status: `{raw['failed_backup_attempt_exit_status']}`; runtime seconds: `{raw['failed_backup_attempt_runtime_seconds']}`.",
        f"- stdout/stderr were empty: `{raw['failed_backup_attempt_stdout_stderr_empty']}`.",
        "",
        "## Supervisor backup status",
        "",
        f"- Status file exists: `{raw['backup_status_file']['exists']}`.",
        f"- Sanitized status: `{status.get('status')}` at `{status.get('time')}`.",
        f"- Error type/message: `{status.get('error_type')}` / `{status.get('message')}`.",
        f"- Packages in this status: `{len(status.get('packages_this_run') or [])}`.",
        f"- Adequate post-case5-dryrun proof: `{decision['adequate_post_case5_dryrun_backup']}`; reasons: `{decision['why_not']}`.",
        "",
        "## Decision",
        "",
        f"{raw['decision_text']}",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-accept-no-simulation-backup-status-capture", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_accept_no_simulation_backup_status_capture:
        raise SystemExit("missing explicit no-simulation backup-status-capture acknowledgement")

    created_dt = now()
    created = created_dt.isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {
        "started_utc": created,
        "pid": os.getpid(),
        "method": "backup_failure_status_capture_after_case5_dryrun_v0",
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    required = [CASE5_DRYRUN_COMPLETED, CASE5_DRYRUN_SUMMARY, CASE5_PROTOCOL, CASE5_RUNNER, CASE5_REQUEST, FAILED_BACKUP_REGISTRY, CONTINUE_STATE_AFTER_FAILURE]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        write_json(OUT / "failure.json", {
            "failed_utc": now().isoformat(),
            "error": "required_case5_or_backup_failure_artifact_missing",
            "missing": missing,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        return 2

    dryrun_completed = read_json(CASE5_DRYRUN_COMPLETED)
    registry = read_json(FAILED_BACKUP_REGISTRY)
    min_required = max_required_time(required, dryrun_completed if isinstance(dryrun_completed, Mapping) else {})
    status_obj, status_error = safe_read_json(BACKUP_STATUS)
    receipts_tail, receipts_error = read_receipts_tail(BACKUP_RECEIPTS)
    decision = classify_backup_status(status_obj if isinstance(status_obj, Mapping) else None, min_required)

    if decision["adequate_post_case5_dryrun_backup"]:
        proof = dict(status_obj)  # type: ignore[arg-type]
        proof.update({
            "backup_verified": True,
            "source": "sanitized copy from /data/openai-agent/state/backup_status.json after backup_failure_status_capture_after_case5_dryrun_v0",
            "created_utc": created,
            "min_required_backup_time_utc": min_required.isoformat(),
            "covers_case5_dryrun": True,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
        })
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        next_action = "run frozen 24-episode case5 H10-vs-H15 repeated stability/timing diagnostic with legacy interpreter using the copied verified proof"
        decision_text = (
            "A verified post-case5-dryrun external backup is available in supervisor state and has been copied into the repository as a sanitized proof. "
            "The next bounded experiment may run the frozen 24-episode case5 stability/timing diagnostic under the legacy interpreter."
        )
        retry_request_path = None
    else:
        write_json(RETRY_REQUEST, {
            "requested_utc": created,
            "reason": "retry external backup after scripts/research_service/backup.py failed before case5 stability/timing simulation",
            "failed_backup_registry": rel(FAILED_BACKUP_REGISTRY),
            "backup_status_classification": decision,
            "backup_required_before_more_simulations": True,
            "min_required_backup_time_utc": min_required.isoformat(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "artifacts": [
                rel(CASE5_RUNNER),
                rel(CASE5_PROTOCOL),
                rel(CASE5_DRYRUN_COMPLETED.parent),
                rel(CASE5_REQUEST),
                rel(FAILED_BACKUP_REGISTRY),
                rel(CONTINUE_STATE_AFTER_FAILURE),
                rel(OUT),
                rel(STATE_MD),
                rel(RETRY_REQUEST),
            ],
        })
        next_action = "retry scripts/research_service/backup.py once; if it fails again, continue infrastructure diagnosis before simulations"
        decision_text = (
            "No adequate verified post-case5-dryrun backup proof exists. More simulations remain blocked. "
            "Retry the external backup once after preserving this sanitized failure status; if the retry fails, diagnose backup infrastructure rather than running control rollouts."
        )
        retry_request_path = rel(RETRY_REQUEST)

    elapsed_seconds = (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds()
    raw: Dict[str, Any] = {
        "created_utc": created,
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "method": "backup_failure_status_capture_after_case5_dryrun_v0",
        "classification": "metadata_only_no_simulation_no_validation_no_test",
        "case5_dryrun_completed": rel(CASE5_DRYRUN_COMPLETED),
        "case5_dryrun_completed_sha256": sha256(CASE5_DRYRUN_COMPLETED),
        "case5_protocol": rel(CASE5_PROTOCOL),
        "case5_protocol_sha256": sha256(CASE5_PROTOCOL),
        "case5_runner": rel(CASE5_RUNNER),
        "case5_runner_sha256": sha256(CASE5_RUNNER),
        "case5_backup_request": rel(CASE5_REQUEST),
        "case5_backup_request_sha256": sha256(CASE5_REQUEST),
        "failed_backup_attempt_registry": rel(FAILED_BACKUP_REGISTRY),
        "failed_backup_attempt_registry_sha256": sha256(FAILED_BACKUP_REGISTRY),
        "failed_backup_attempt_exit_status": registry.get("exit_status"),
        "failed_backup_attempt_runtime_seconds": registry.get("runtime_seconds"),
        "failed_backup_attempt_failure_reason": registry.get("failure_reason"),
        "failed_backup_attempt_stdout_stderr_empty": bool(registry.get("stdout") and registry.get("stderr") and (Path(registry["stdout"]).exists() and Path(registry["stdout"]).stat().st_size == 0) and (Path(registry["stderr"]).exists() and Path(registry["stderr"]).stat().st_size == 0)),
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": status_obj,
        "backup_receipts_file": path_info(BACKUP_RECEIPTS),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts_tail,
        "backup_index_file": path_info(BACKUP_INDEX),
        "backup_gate_decision": decision,
        "copied_verified_proof": rel(PROOF_IF_VERIFIED) if decision["adequate_post_case5_dryrun_backup"] else None,
        "retry_backup_request": retry_request_path,
        "decision_text": decision_text,
        "next_action": next_action,
        "summary_path": rel(OUT / "summary.md"),
        "completed_path": rel(OUT / "completed.json"),
        "budgets_actual": {
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text(
        "# Backup status capture after case5 dry-run backup failure\n\n"
        f"UTC: {created}. No simulation/training/validation/test. Backup adequate={decision['adequate_post_case5_dryrun_backup']}. "
        f"Reasons={decision['why_not']}. Next action: {next_action}.\n",
        encoding="utf-8",
    )
    append_docs(raw)

    files = [
        OUT / "run_started.json",
        OUT / "raw.json",
        OUT / "summary.md",
        STATE_MD,
        CASE5_DRYRUN_COMPLETED,
        CASE5_DRYRUN_SUMMARY,
        CASE5_PROTOCOL,
        CASE5_RUNNER,
        CASE5_REQUEST,
        FAILED_BACKUP_REGISTRY,
        CONTINUE_STATE_AFTER_FAILURE,
    ]
    if PROOF_IF_VERIFIED.exists():
        files.append(PROOF_IF_VERIFIED)
    if RETRY_REQUEST.exists():
        files.append(RETRY_REQUEST)
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "adequate_post_case5_dryrun_backup": decision["adequate_post_case5_dryrun_backup"],
        "backup_gate_decision": decision,
        "copied_verified_proof": rel(PROOF_IF_VERIFIED) if PROOF_IF_VERIFIED.exists() else None,
        "retry_backup_request": rel(RETRY_REQUEST) if RETRY_REQUEST.exists() else None,
        "next_action": next_action,
        "new_rollouts": 0,
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
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "adequate_post_case5_dryrun_backup": decision["adequate_post_case5_dryrun_backup"],
        "why_not": decision["why_not"],
        "copied_verified_proof": completed["copied_verified_proof"],
        "retry_backup_request": completed["retry_backup_request"],
        "next_action": next_action,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
