#!/usr/bin/env python3
"""Sanitized no-simulation capture of current backup failure before v8b.

This script is intentionally metadata-only.  It reads supervisor backup-status
files outside the repository, sanitizes public audit fields into repository
artifacts, and records whether another scientific/development rollout remains
blocked.  It must not run the backup program, touch credentials, import legacy
TF/controller code, open validation64 or sealed-test banks, generate scenarios,
execute MPC rollouts, train/refit selectors, or delete/modify failed evidence.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
BACKUP_RECEIPTS = SUP_STATE / "backup_receipts.jsonl"
BACKUP_INDEX = SUP_STATE / "backup_index.sqlite"

STAMP = "20260929T1314Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/backup_failure_status_capture_v8b_v0_{STAMP}"
STATE_MD = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_backup_failure_before_v8b.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
RETRY_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_V8B_BACKUP_FAILURE_{STAMP}.json"
PROOF_IF_VERIFIED = BACKUP_DIR / f"backup_proof_{STAMP}_from_state_before_v8b.json"

FAILED_BACKUP_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T131215_23fa83ef/registry.json"
FAILED_V8_LEGACY_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T130819_cc2e807a/registry.json"
FAILED_V8_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8_20260929T1258Z"
V8B_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry.py"
PRIOR_CONTINUE = ROOT / "research_artifacts/aws_state/continue_state_20260929T1310_after_v8_modern_prefight_failure.md"
PRIOR_REQUEST = BACKUP_DIR / "REQUEST_BACKUP_AFTER_V8_LEGACY_PREFLIGHT_FAILURE_20260929T1309Z.json"

FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|key_material|github\.token)", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{40,}\b")


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
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


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
    return {
        "path": rel(path),
        "exists": True,
        "bytes": st.st_size,
        "sha256": sha256(path),
        "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            key_s = str(key)
            if SECRET_KEY_RE.search(key_s):
                if key_s.lower() in {"sha256", "asset_sha256", "release_asset_sha256", "package_sha256"}:
                    out[key_s] = sanitize(val)
                else:
                    out[key_s] = "[REDACTED]"
            else:
                out[key_s] = sanitize(val)
        return out
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            text = match.group(0)
            if len(text) in (40, 64):
                return text
            return "[REDACTED_HEX]"
        # Keep public URLs and digests; redact only unusually long hex blobs.
        return LONG_HEX_RE.sub(repl, value).replace(str(BASE / ".secrets"), "[REDACTED_SECRET_PATH]")
    return value


def safe_read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return sanitize(read_json(path)), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:500]}"


def receipts_tail(path: Path, n: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return [], "missing"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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


def min_required_time(paths: Iterable[Path]) -> dt.datetime:
    times: List[dt.datetime] = []
    for path in paths:
        if path.exists():
            times.append(dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc))
    return max(times) if times else now()


def classify(status_obj: Optional[Mapping[str, Any]], min_required: dt.datetime) -> Dict[str, Any]:
    if not isinstance(status_obj, Mapping):
        return {"status_file_readable": False, "adequate_backup_before_v8b": False, "why_not": ["status_file_missing_or_unreadable"], "min_required_backup_time_utc": min_required.isoformat()}
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
        why.append("status_time_predates_required_v8b_artifacts_or_unparseable")
    return {
        "status_file_readable": True,
        "status_time": None if t is None else t.isoformat(),
        "status": status_obj.get("status"),
        "error_type": status_obj.get("error_type"),
        "message": status_obj.get("message"),
        "remaining_changed_files": status_obj.get("remaining_changed_files"),
        "commit": status_obj.get("commit"),
        "package_count": len(status_obj.get("packages_this_run") or []),
        "has_verified_package_metadata": packages,
        "time_ok": time_ok,
        "min_required_backup_time_utc": min_required.isoformat(),
        "adequate_backup_before_v8b": bool(verified and remaining_ok and has_commit and packages and time_ok),
        "why_not": why,
    }


def git_status() -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0")
    result: Dict[str, Any] = {}
    for name, args in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_porcelain": ["git", "status", "--porcelain"],
    }.items():
        try:
            p = subprocess.run(args, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=30)
            result[name] = {"returncode": p.returncode, "stdout": p.stdout[:4000], "stderr_suppressed_empty": not bool(p.stderr)}
            if p.stderr:
                result[name]["stderr_sanitized_prefix"] = sanitize(p.stderr[:500])
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return result


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    marker = "backup-failure-status-capture-v8b-v0-20260929T1314Z"
    decision = raw["backup_gate_decision"]
    block = "\n".join([
        f"<!-- {marker} -->",
        "## 2026-09-29 backup failure capture before v8b risk probe",
        "",
        f"UTC: {raw['created_utc']}. Metadata-only capture after backup.py failed before v8b; no simulation/control/training/refit/validation64/sealed-test access.",
        f"Failed backup registry: `{raw['failed_backup_registry']}`; exit status `{raw['failed_backup_exit_status']}`, runtime seconds `{raw['failed_backup_runtime_seconds']}`.",
        f"Sanitized supervisor status: `{decision.get('status')}` / `{decision.get('error_type')}`; adequate backup before v8b=`{decision['adequate_backup_before_v8b']}`; reasons=`{decision['why_not']}`.",
        f"Decision: `{raw['decision_text']}` Next: `{raw['next_action']}`.",
    ]) + "\n"
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / name, marker, block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = (
        f"{raw['created_utc']},backup_failure_status_capture_v8b_v0,metadata_no_simulation,"
        f"infrastructure_backup_gate,0,0,0,0,0,{decision['adequate_backup_before_v8b']},"
        f"{raw['completed_path']}\n"
    )
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if "backup_failure_status_capture_v8b_v0" not in old[-30000:]:
            reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    d = raw["backup_gate_decision"]
    status = raw.get("backup_status_sanitized") or {}
    lines = [
        "# Backup failure status capture before v8b",
        "",
        f"UTC: `{raw['created_utc']}`. This is a metadata-only infrastructure diagnostic: no simulations, no control steps, no selector refit/training, no validation64 and no sealed test.",
        "",
        "## Inspected backup attempt",
        f"- Registry: `{raw['failed_backup_registry']}`",
        f"- Exit status: `{raw['failed_backup_exit_status']}`; runtime seconds: `{raw['failed_backup_runtime_seconds']}`; stdout/stderr empty: `{raw['failed_backup_stdout_stderr_empty']}`.",
        "",
        "## Sanitized supervisor backup status",
        f"- Status: `{status.get('status')}` at `{status.get('time')}`; error type: `{status.get('error_type')}`.",
        f"- Message: `{status.get('message')}`.",
        f"- Adequate backup before v8b: `{d['adequate_backup_before_v8b']}`.",
        f"- Why not: `{d['why_not']}`.",
        f"- Minimum required backup time: `{d.get('min_required_backup_time_utc')}`.",
        "",
        "## Decision",
        raw["decision_text"],
        "",
        f"Next action: `{raw['next_action']}`.",
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
        "method": "backup_failure_status_capture_v8b_v0",
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    required = [FAILED_BACKUP_REGISTRY, FAILED_V8_LEGACY_REGISTRY, FAILED_V8_DIR, V8B_SOURCE, PRIOR_CONTINUE, PRIOR_REQUEST]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        write_json(OUT / "failed.json", {
            "failed_utc": now().isoformat(),
            "error": "required_pre_v8b_artifact_missing",
            "missing": missing,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        return 2

    registry = read_json(FAILED_BACKUP_REGISTRY)
    status_obj, status_error = safe_read_json(BACKUP_STATUS)
    receipts, receipts_error = receipts_tail(BACKUP_RECEIPTS)
    min_time = min_required_time(required + [Path(__file__)])
    decision = classify(status_obj if isinstance(status_obj, Mapping) else None, min_time)

    if decision["adequate_backup_before_v8b"]:
        proof = dict(status_obj)  # type: ignore[arg-type]
        proof.update({
            "backup_verified": True,
            "source": "sanitized copy from /data/openai-agent/state/backup_status.json by backup_failure_status_capture_v8b_v0",
            "created_utc": created,
            "min_required_backup_time_utc": min_time.isoformat(),
            "covers_failed_v8_and_v8b_source": True,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
        })
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        next_action = "run vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry.py under legacy with --run and the copied verified commit"
        decision_text = "A verified backup already covers the required pre-v8b state; v8b can proceed under the legacy interpreter."
        retry_path = None
    else:
        write_json(RETRY_REQUEST, {
            "requested_utc": created,
            "reason": "retry external backup after backup.py failed immediately before v8b risk-probe acquisition diagnostic",
            "failed_backup_registry": rel(FAILED_BACKUP_REGISTRY),
            "backup_status_classification": decision,
            "backup_required_before_more_simulations": True,
            "backup_required_before_training_or_refit": True,
            "min_required_backup_time_utc": min_time.isoformat(),
            "scientific_effect": "none; infrastructure failure only",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "artifacts_requiring_backup": [rel(p) for p in required] + [rel(OUT), rel(STATE_MD), rel(RETRY_REQUEST)],
        })
        next_action = "diagnose/fix backup infrastructure or obtain a verified backup; do not run v8b simulations until backup is verified"
        decision_text = (
            "Backup remains inadequate for further formal evidence accumulation.  v8b risk-probe simulations are blocked until external recovery is verified. "
            "The immediate scientific path remains v8b after backup; current action is infrastructure diagnosis, not method completion."
        )
        retry_path = rel(RETRY_REQUEST)

    elapsed_seconds = (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds()
    raw: Dict[str, Any] = {
        "created_utc": created,
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "method": "backup_failure_status_capture_v8b_v0",
        "classification": "metadata_only_infrastructure_no_simulation",
        "failed_backup_registry": rel(FAILED_BACKUP_REGISTRY),
        "failed_backup_registry_sha256": sha256(FAILED_BACKUP_REGISTRY),
        "failed_backup_exit_status": registry.get("exit_status"),
        "failed_backup_runtime_seconds": registry.get("runtime_seconds"),
        "failed_backup_failure_reason": registry.get("failure_reason"),
        "failed_backup_stdout_stderr_empty": bool(registry.get("stdout") and registry.get("stderr") and Path(registry["stdout"]).exists() and Path(registry["stdout"]).stat().st_size == 0 and Path(registry["stderr"]).exists() and Path(registry["stderr"]).stat().st_size == 0),
        "required_artifacts": [path_info(p) for p in required],
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": status_obj,
        "backup_receipts_file": path_info(BACKUP_RECEIPTS),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts,
        "backup_index_file": path_info(BACKUP_INDEX),
        "backup_gate_decision": decision,
        "git_status_sanitized": git_status(),
        "copied_verified_proof": rel(PROOF_IF_VERIFIED) if PROOF_IF_VERIFIED.exists() else None,
        "retry_backup_request": retry_path,
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
            "mobile_robot_mppi_resumed": False,
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text(
        "# Continue state after backup failure before v8b\n\n"
        f"UTC: {created}. No simulation/training/refit/validation/test. Backup adequate before v8b={decision['adequate_backup_before_v8b']}. "
        f"Reasons={decision['why_not']}. Error={decision.get('error_type')} message={decision.get('message')}. "
        f"Next action: {next_action}.\n",
        encoding="utf-8",
    )
    append_docs(raw)

    files = [OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", OUT / "completed.json", STATE_MD, RETRY_REQUEST, PROOF_IF_VERIFIED] + required
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "adequate_backup_before_v8b": decision["adequate_backup_before_v8b"],
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
        "adequate_backup_before_v8b": decision["adequate_backup_before_v8b"],
        "why_not": decision["why_not"],
        "error_type": decision.get("error_type"),
        "message": decision.get("message"),
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
