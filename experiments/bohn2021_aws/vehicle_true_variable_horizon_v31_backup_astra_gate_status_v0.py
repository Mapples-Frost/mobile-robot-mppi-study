#!/usr/bin/env python3
"""Metadata-only backup/Astra gate status after v31 cluster-stability diagnostic.

This script performs no simulation, no MPC rollout, no selector refit, no
training, and no validation/test access.  It reads only repository artifacts and
sanitized supervisor backup status to decide whether the latest v31 diagnostic
outputs are externally recoverable before any further unique science.  It also
checks whether Astra has returned ANALYSIS_READY for the current v31 review
request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

NAME = "vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0"
STAMP = "20260930T0415Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
CONTINUE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v31_backup_astra_gate_status.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
PROOF_IF_OK = BACKUP_DIR / f"backup_proof_{STAMP}_from_supervisor_context_after_v31_outputs.json"
RETRY_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_BACKUP_ASTRA_GATE_STATUS_{STAMP}.json"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST_REVIEW = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
DOC_MARKER = f"<!-- {NAME}-{STAMP} -->"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"

# Required artifacts that existed before this gate-status script and must be
# covered before further simulations/refits/validation.  Deliberately do not
# include this gate script/output in the adequacy test, otherwise a just-created
# no-simulation status check would make any existing post-v31 backup appear
# stale.  This script separately requests backup of its own outputs.
REQUIRED_V31 = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/run_started.json",
    ROOT / "research_artifacts/aws_runs/20260930T040615_4f9729e6/registry.json",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0405Z_after_v31_cluster_stability_diagnostic.md",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0407_after_v31_executor_state.md",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_20260930T0405Z.json",
    NEXT_REVIEW,
    RESPONSE_LOG,
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    ROOT / "EXPERIMENT_REGISTRY.csv",
]


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
    if not isinstance(value, str) or not value:
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
    return {
        "path": rel(path),
        "exists": True,
        "bytes": st.st_size,
        "sha256": sha256(path) if path.is_file() else None,
        "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
    }


def max_mtime(paths: list[Path]) -> dt.datetime:
    times = [dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) for p in paths if p.exists()]
    return max(times) if times else now()


def git(args: list[str]) -> Dict[str, Any]:
    try:
        p = subprocess.run(["git", *args], cwd=str(ROOT), text=True, capture_output=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:4000], "stderr": p.stderr.strip()[:1000]}
    except Exception as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


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


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def compact_backup_status(status: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(status, Mapping):
        return None
    keys = ["status", "backup_verified", "time", "remaining_changed_files", "commit", "changed_files", "packages_this_run", "release", "tracked_files"]
    return {k: status.get(k) for k in keys if k in status}


def ready_matches_current(ready: Any, current_request_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not current_request_id:
        return False
    if ready.get("request_id") == current_request_id:
        return True
    if ready.get("supersedes_request_id") == current_request_id:
        return True
    covered = ready.get("covers_request_ids") or ready.get("covered_request_ids")
    return isinstance(covered, list) and current_request_id in covered


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-metadata-only-gate-status", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_metadata_only_gate_status:
        raise SystemExit("missing explicit metadata-only gate-status acknowledgement")

    created = now()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {
        "started_utc": created.isoformat(),
        "method": NAME,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    missing = [rel(p) for p in REQUIRED_V31 if not p.exists()]
    min_required = max_mtime(REQUIRED_V31)
    backup_status = safe_json(BACKUP_STATUS)
    status_time = parse_time(backup_status.get("time") if isinstance(backup_status, Mapping) else None)
    verified = bool(isinstance(backup_status, Mapping) and (backup_status.get("status") == "verified" or backup_status.get("backup_verified") is True))
    try:
        remaining_ok = int(backup_status.get("remaining_changed_files", -1)) == 0 if isinstance(backup_status, Mapping) else False
    except Exception:
        remaining_ok = False
    has_commit = bool(isinstance(backup_status, Mapping) and backup_status.get("commit"))
    packages_ok = bool(isinstance(backup_status, Mapping) and package_metadata_ok(backup_status))
    time_ok = bool(status_time and status_time >= min_required)
    adequate_v31_backup = bool((not missing) and verified and remaining_ok and has_commit and packages_ok and time_ok)
    why_not: list[str] = []
    if missing:
        why_not.append("missing_required_v31_artifacts")
    if not verified:
        why_not.append("backup_status_not_verified_or_missing")
    if not remaining_ok:
        why_not.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why_not.append("missing_backup_commit")
    if not packages_ok:
        why_not.append("missing_verified_package_metadata")
    if not time_ok:
        why_not.append("backup_time_predates_v31_artifacts_or_unparseable")

    next_review = safe_json(NEXT_REVIEW)
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    analysis_ready = safe_json(ANALYSIS_READY)
    astra_ready = ready_matches_current(analysis_ready, current_request_id)
    report_path = analysis_ready.get("report") if isinstance(analysis_ready, Mapping) else None
    if not report_path and isinstance(analysis_ready, Mapping):
        report_path = analysis_ready.get("report_path")

    if adequate_v31_backup and isinstance(backup_status, Mapping):
        proof = dict(compact_backup_status(backup_status) or {})
        proof.update({
            "backup_verified": True,
            "created_utc": created.isoformat(),
            "source": "sanitized supervisor backup_status captured by v31 backup/Astra gate status",
            "covers_v31_cluster_stability_outputs": True,
            "min_required_backup_time_utc": min_required.isoformat(),
            "required_v31_artifacts": [rel(p) for p in REQUIRED_V31],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "note": "This proof covers artifacts that existed before this metadata-only gate run; this gate run's own outputs still require routine backup.",
        })
        write_json(PROOF_IF_OK, proof)
        copied_proof = rel(PROOF_IF_OK)
    else:
        copied_proof = None

    # Always request backup for the gate-status outputs themselves and, if needed,
    # for the still-uncovered v31 artifacts.
    write_json(RETRY_REQUEST, {
        "requested_utc": created.isoformat(),
        "reason": "backup request after v31 backup/Astra gate status; also covers v31 artifacts if adequate_v31_backup is false",
        "adequate_v31_backup_before_this_gate": adequate_v31_backup,
        "backup_gate_why_not": why_not,
        "min_required_v31_backup_time_utc": min_required.isoformat(),
        "latest_backup_status_time_utc": None if status_time is None else status_time.isoformat(),
        "must_cover": [rel(p) for p in REQUIRED_V31] + [rel(Path(__file__).resolve()), rel(OUT), rel(CONTINUE), rel(RETRY_REQUEST), rel(PROOF_IF_OK)],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    })

    if not adequate_v31_backup:
        next_action = "Obtain verified external backup covering v31 outputs/handoff/state/registry before any new simulation/refit/validation. Continue only reversible integrity checks if needed."
    elif astra_ready:
        next_action = "Read the matching/superseding Astra report and execute its selected plan, subject to backup of this gate-status output before any unique simulation/refit/validation."
    else:
        next_action = "v31 outputs are externally backed up, but Astra analysis is still pending; avoid scientific branch selection and continue only reversible preparation/integrity work."

    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "metadata_only_backup_and_astra_gate_no_sim_no_validation_no_test_no_training",
        "required_v31_artifacts": [path_info(p) for p in REQUIRED_V31],
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_subset": compact_backup_status(backup_status),
        "backup_gate": {
            "adequate_v31_backup": adequate_v31_backup,
            "why_not": why_not,
            "min_required_backup_time_utc": min_required.isoformat(),
            "status_time_utc": None if status_time is None else status_time.isoformat(),
            "copied_verified_proof": copied_proof,
            "backup_request_after_gate_status": rel(RETRY_REQUEST),
        },
        "astra_gate": {
            "latest_md_exists": LATEST_REVIEW.exists(),
            "next_review_request_exists": NEXT_REVIEW.exists(),
            "next_review_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "next_request_matches_expected": current_request_id == EXPECTED_REQUEST_ID,
            "analysis_ready_exists": ANALYSIS_READY.exists(),
            "analysis_ready_request_id": analysis_ready.get("request_id") if isinstance(analysis_ready, Mapping) else None,
            "analysis_ready_matches_or_supersedes_current": astra_ready,
            "analysis_ready_report": report_path,
        },
        "git": {
            "head": git(["rev-parse", "HEAD"]),
            "status_critical": git(["status", "--short", "--", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0.py", "docs/bohn2021_takeover/astra_reviews", "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z", "research_artifacts/aws_state/continue_state_20260930T0407_after_v31_executor_state.md", "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_20260930T0405Z.json"]),
        },
        "next_action": next_action,
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
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)

    summary = "\n".join([
        "# v31 backup/Astra gate status",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only: no simulations, no control steps, no selector refits, no training, no validation64, no sealed test.",
        "",
        "## Backup gate for v31 outputs",
        f"- Adequate v31 backup: `{adequate_v31_backup}`.",
        f"- Reasons if blocked: `{why_not}`.",
        f"- Backup status time: `{raw['backup_gate']['status_time_utc']}`; minimum required for v31 artifacts: `{raw['backup_gate']['min_required_backup_time_utc']}`.",
        f"- Copied verified proof, if any: `{copied_proof}`.",
        f"- Backup request after this gate status: `{rel(RETRY_REQUEST)}`.",
        "",
        "## Astra gate",
        f"- NEXT_REVIEW_REQUEST id: `{current_request_id}`; expected current id: `{EXPECTED_REQUEST_ID}`.",
        f"- ANALYSIS_READY present: `{ANALYSIS_READY.exists()}`; request id: `{raw['astra_gate']['analysis_ready_request_id']}`; matches/supersedes current: `{astra_ready}`.",
        f"- Report path, if supplied: `{report_path}`.",
        "",
        "## Next executor action",
        next_action,
    ]) + "\n"
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    CONTINUE.parent.mkdir(parents=True, exist_ok=True)
    CONTINUE.write_text(summary, encoding="utf-8")

    doc_block = f"""## 2026-09-30 v31 backup/Astra gate status

UTC: {created.isoformat()}. Metadata-only gate status completed; no simulation/control/training/refit/validation64/sealed-test access. Adequate v31 backup=`{adequate_v31_backup}` with reasons=`{why_not}`. Astra ready for current request `{current_request_id}`=`{astra_ready}`. Next action: {next_action}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, DOC_MARKER, doc_block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created.isoformat()},{NAME},metadata_backup_astra_gate,no_validation_no_test,0,0,0,0,0,{adequate_v31_backup},{rel(OUT / 'completed.json')}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [Path(__file__).resolve(), OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", CONTINUE, RETRY_REQUEST, PROOF_IF_OK] + REQUIRED_V31
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
        "backup_request_after_gate_status": rel(RETRY_REQUEST),
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "adequate_v31_backup": adequate_v31_backup,
        "backup_why_not": why_not,
        "astra_ready_for_current_request": astra_ready,
        "analysis_ready_report": report_path,
        "next_action": next_action,
        "new_simulations": 0,
        "new_control_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
