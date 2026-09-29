#!/usr/bin/env python3
"""Metadata-only backup-gate status capture for selector-refit diagnostics.

This script is intentionally not a simulation, not training, not validation, and
not a selector refit.  It records sanitized external-backup state after repeated
backup.py failures and decides whether the selector-refit sources/recent true-H
fresh-source evidence are externally recoverable enough to run the next bounded
no-simulation selector-refit diagnostic.

It never reads credential files.  It reads only supervisor backup status/receipts
and repository artifact metadata.
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

NAME = "vehicle_true_variable_horizon_selector_refit_backup_gate_status_v0"
STAMP = "20260929T1150Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_MD = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T1150_after_selector_refit_backup_gate_status.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
PROOF_IF_VERIFIED = BACKUP_DIR / f"backup_proof_{STAMP}_after_selector_refit_sources_and_fresh_v1.json"
RETRY_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_SELECTOR_REFIT_SOURCES_AND_FRESH_V1_{STAMP}.json"
DOC_MARKER = f"<!-- {NAME}-{STAMP} -->"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

REQUIRED_ARTIFACTS = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2_diagnostic.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2b_schema_repair.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_guard_veto_diagnostic_v0.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/completed.json",
    ROOT / "research_artifacts/aws_state/continue_state_20260929T1125_after_fresh_v1_weak_and_selector_refit_source.md",
    ROOT / "research_artifacts/aws_runs/20260929T105359_8b1ea4c5/registry.json",
    ROOT / "research_artifacts/aws_runs/20260929T105837_216ed181/registry.json",
]
EXPECTED_SOURCE_SHA = {
    "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2_diagnostic.py": "66bfe0221b7cd91713b52781769cabac1f6eed8679880e7f1b499c59bafeaf0b",
    "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2b_schema_repair.py": "614cac191abf48eeb3dda89adf7a9dbfc44a046f931ab5212cb7373b4acad3dc",
    "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema.py": "ff63389ca0ec45f25c747ea636b73006f9784cc9550ee2aea1e77e3f8d073e53",
}
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
        return {"returncode": r.returncode, "stdout": r.stdout.strip()[:4000], "stderr": r.stderr.strip()[:1000]}
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


def backup_index_probe(required: Iterable[Path]) -> Dict[str, Any]:
    if not BACKUP_INDEX.exists():
        return {"exists": False}
    try:
        con = sqlite3.connect(str(BACKUP_INDEX))
        row_count = con.execute("select count(*) from files").fetchone()[0]
        interesting = []
        for p in required:
            key = "mobile-robot-mppi-study/" + rel(p)
            hit = con.execute("select size, sha, asset from files where path=?", (key,)).fetchone()
            local_sha = sha256(p) if p.exists() else None
            interesting.append({
                "path": key,
                "exists_local": p.exists(),
                "indexed": hit is not None,
                "size": None if hit is None else hit[0],
                "sha256_index": None if hit is None else hit[1],
                "sha256_local": local_sha,
                "sha_matches": bool(hit is not None and local_sha is not None and hit[1] == local_sha),
                "asset": None if hit is None else hit[2],
            })
        con.close()
        return {"exists": True, "row_count": row_count, "interesting_artifacts": interesting}
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list) or len(packages) == 0:
        return False
    for p in packages:
        if not isinstance(p, Mapping):
            return False
        if not p.get("sha256") or not p.get("bytes") or not p.get("verification"):
            return False
    return True


def classify(status: Optional[Mapping[str, Any]], min_required: dt.datetime, required: Iterable[Path], git_info: Mapping[str, Any], index_probe: Mapping[str, Any]) -> Dict[str, Any]:
    source_hash_ok = all((ROOT / rel_path).exists() and sha256(ROOT / rel_path) == digest for rel_path, digest in EXPECTED_SOURCE_SHA.items())
    missing = [rel(p) for p in required if not p.exists()]
    git_sources_clean = (git_info.get("selector_source_status") or {}).get("stdout", "") == ""
    git_docs_clean = (git_info.get("critical_artifact_status") or {}).get("stdout", "") == ""
    index_items = index_probe.get("interesting_artifacts") if isinstance(index_probe, Mapping) else None
    index_covers = bool(isinstance(index_items, list) and all(x.get("sha_matches") for x in index_items if x.get("exists_local")))
    if not isinstance(status, Mapping):
        return {"adequate_selector_refit_backup": False, "why_not": ["backup_status_missing_or_unreadable"], "min_required_backup_time_utc": min_required.isoformat(), "source_hash_ok": source_hash_ok, "missing_required_artifacts": missing}
    t = parse_time(status.get("time"))
    verified = status.get("status") == "verified" or status.get("backup_verified") is True
    try:
        remaining_ok = int(status.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    has_commit = bool(status.get("commit"))
    time_ok = bool(t is not None and t >= min_required)
    packages_ok = package_metadata_ok(status)
    why: List[str] = []
    if missing:
        why.append("missing_required_artifacts")
    if not verified:
        why.append(f"status_is_{status.get('status')!r}_not_verified")
    if not remaining_ok:
        why.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why.append("missing_commit")
    if not packages_ok:
        why.append("missing_verified_package_metadata")
    if not time_ok:
        why.append("status_time_predates_required_artifacts_or_unparseable")
    if not source_hash_ok:
        why.append("selector_source_sha_mismatch")
    if not git_sources_clean:
        why.append("selector_sources_have_uncommitted_or_untracked_git_status")
    if not git_docs_clean:
        why.append("critical_artifacts_have_uncommitted_or_untracked_git_status")
    # Index coverage is informative; do not make it a hard block when backup_status
    # says remaining_changed_files==0 and uploaded packages were verified, because
    # source code may be covered by git commit rather than release asset index.
    return {
        "adequate_selector_refit_backup": bool((not missing) and verified and remaining_ok and has_commit and packages_ok and time_ok and source_hash_ok and git_sources_clean and git_docs_clean),
        "why_not": why,
        "status": status.get("status"),
        "status_time": None if t is None else t.isoformat(),
        "remaining_changed_files": status.get("remaining_changed_files"),
        "commit": status.get("commit"),
        "package_count": len(status.get("packages_this_run") or []),
        "has_verified_package_metadata": packages_ok,
        "time_ok": time_ok,
        "source_hash_ok": source_hash_ok,
        "selector_sources_git_clean": git_sources_clean,
        "critical_artifacts_git_clean": git_docs_clean,
        "backup_index_covers_local_required_hashes": index_covers,
        "missing_required_artifacts": missing,
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
        "# Selector-refit backup-gate status v0",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only: no simulations, no control steps, no selector refits, no training, no validation64 bank, no sealed test.",
        "",
        "## Inputs checked",
        "",
        f"- Required artifacts present/missing: `{d['missing_required_artifacts']}`.",
        f"- Selector source hashes ok: `{d['source_hash_ok']}`.",
        f"- Supervisor backup status: `{status.get('status')}` at `{status.get('time')}`; remaining changed `{status.get('remaining_changed_files')}`; packages `{len(status.get('packages_this_run') or [])}`.",
        f"- Minimum required backup time: `{d['min_required_backup_time_utc']}`.",
        "",
        "## Backup gate decision",
        "",
        f"Adequate selector-refit backup: `{d['adequate_selector_refit_backup']}`.",
        f"Reasons if blocked: `{d['why_not']}`.",
        "",
        "## Next action",
        "",
        str(raw["next_action"]),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-no-simulation-selector-refit-backup-status-capture", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_no_simulation_selector_refit_backup_status_capture:
        raise SystemExit("missing explicit no-simulation selector-refit backup-status-capture acknowledgement")
    created_dt = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_dt.isoformat(), "pid": os.getpid(), "method": NAME, "new_simulations": 0, "new_control_steps": 0, "new_selector_refits": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})

    required_with_self = REQUIRED_ARTIFACTS + [Path(__file__).resolve()]
    min_required = required_time(required_with_self)
    status_obj, status_error = safe_read_json(BACKUP_STATUS)
    receipts, receipts_error = receipts_tail()
    git_info = {
        "head": git_cmd(["rev-parse", "HEAD"]),
        "selector_source_status": git_cmd(["status", "--short", "--", "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2_diagnostic.py", "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2b_schema_repair.py", "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema.py", "experiments/bohn2021_aws/vehicle_true_variable_horizon_selector_refit_backup_gate_status_v0.py"]),
        "critical_artifact_status": git_cmd(["status", "--short", "--", "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z", "research_artifacts/aws_runs/20260929T105837_216ed181", "research_artifacts/aws_state/continue_state_20260929T1125_after_fresh_v1_weak_and_selector_refit_source.md"]),
    }
    index_probe = backup_index_probe(required_with_self)
    decision = classify(status_obj if isinstance(status_obj, Mapping) else None, min_required, required_with_self, git_info, index_probe)

    if decision["adequate_selector_refit_backup"]:
        proof = dict(status_obj)  # type: ignore[arg-type]
        proof.update({
            "backup_verified": True,
            "created_utc": created_dt.isoformat(),
            "source": "sanitized copy from supervisor backup_status after selector-refit backup-gate diagnostic",
            "covers_selector_refit_sources_and_fresh_v1": True,
            "min_required_backup_time_utc": min_required.isoformat(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_selector_refits": 0,
        })
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        retry_request_path = None
        copied_proof = rel(PROOF_IF_VERIFIED)
        next_action = "Run selector-refit v2c relaxed-schema diagnostic with --backup-verified-commit set to the verified commit; still do not open validation64 or sealed test."
    else:
        write_json(RETRY_REQUEST, {
            "requested_utc": created_dt.isoformat(),
            "reason": "selector-refit sources/recent true-H evidence not yet covered by an adequate verified external backup after backup.py failure",
            "backup_required_before_more_simulations": True,
            "backup_required_before_selector_refit": True,
            "backup_gate_decision": decision,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_selector_refits": 0,
            "artifacts_to_cover_before_selector_refit": [rel(p) for p in required_with_self] + [rel(OUT), rel(STATE_MD), rel(CONTINUE_STATE), rel(RETRY_REQUEST)],
        })
        retry_request_path = rel(RETRY_REQUEST)
        copied_proof = None
        next_action = "Retry external backup or resolve backup infrastructure; selector-refit/simulation/validation remain blocked until verified backup proof exists."

    raw: Dict[str, Any] = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "metadata_only_selector_refit_backup_gate_no_simulation_no_validation_no_test",
        "required_inputs": [path_info(p) for p in required_with_self],
        "expected_source_sha": EXPECTED_SOURCE_SHA,
        "supervisor_backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": status_obj,
        "backup_receipts_file": path_info(BACKUP_RECEIPTS),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts,
        "backup_index_probe": index_probe,
        "git_info": git_info,
        "backup_gate_decision": decision,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request_path,
        "next_action": next_action,
        "budgets_actual": {"new_simulations": 0, "new_control_steps": 0, "new_selector_refits": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(
        f"# Continue state after selector-refit backup-gate status\n\nUTC: {created_dt.isoformat()}\nElapsed since first supervisor event: {(created_dt - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.\n\nNo simulation/training/selector-refit/validation/test was run. Adequate backup={decision['adequate_selector_refit_backup']}; reasons={decision['why_not']}.\nSummary: `{rel(OUT / 'summary.md')}`. Completed: `{rel(OUT / 'completed.json')}`.\nNext action: {next_action}\n",
        encoding="utf-8",
    )
    doc_block = f"""## 2026-09-29 selector-refit backup-gate status

UTC: {created_dt.isoformat()}. Metadata-only backup-gate diagnostic completed; no simulations, no selector refits, no training, no validation64 and no sealed test. Adequate selector-refit backup=`{decision['adequate_selector_refit_backup']}`; reasons=`{decision['why_not']}`. Next action: {next_action}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, DOC_MARKER, doc_block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created_dt.isoformat()},{NAME},metadata_backup_gate,no_validation_no_test,0,0,0,0,0,{decision['adequate_selector_refit_backup']},{rel(OUT / 'completed.json')}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_MD, CONTINUE_STATE, Path(__file__).resolve()]
    files += REQUIRED_ARTIFACTS
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
        "adequate_selector_refit_backup": decision["adequate_selector_refit_backup"],
        "backup_gate_decision": decision,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request_path,
        "next_action": next_action,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_selector_refits": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda x: rel(x)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_selector_refit_backup": decision["adequate_selector_refit_backup"], "why_not": decision["why_not"], "copied_verified_proof": copied_proof, "retry_backup_request": retry_request_path, "next_action": next_action, "new_simulations": 0, "new_selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
