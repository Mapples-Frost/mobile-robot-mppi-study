#!/usr/bin/env python3
"""Post-v31 backup and Astra readiness recheck.

Metadata-only/reversible operational audit.  This script does not run MPC, does
not simulate, does not refit selectors, does not train, and does not open
validation64 or sealed-test banks.  It reads only repository artifacts,
Astra-handoff files, a sanitized supervisor backup_status.json, and (if
available) an aggregate server-token counter from research.sqlite.

Purpose after the v31 cluster-stability diagnostic: materialize a repository
proof that the supervisor backup at/after v31 covers the pre-existing v31
outputs, check whether ANALYSIS_READY has arrived for the current request, and
preserve a bounded next-state note without choosing a new scientific branch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
RESEARCH_SQLITE = SUP_STATE / "research.sqlite"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

NAME = "vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST_REVIEW = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"

# Pre-existing v31/gate artifacts that must have been backed up before any new
# unique simulation/refit/validation.  This list intentionally excludes this new
# script and its outputs, which are created after the latest backup and therefore
# will require a follow-up backup request.
PREEXISTING_REQUIRED = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/completed.json",
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


def max_existing_mtime(paths: Iterable[Path]) -> dt.datetime:
    times = [dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) for p in paths if p.exists()]
    return max(times) if times else now_utc()


def git(args: list[str]) -> Dict[str, Any]:
    try:
        p = subprocess.run(["git", *args], cwd=str(ROOT), text=True, capture_output=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:8000], "stderr": p.stderr.strip()[:2000]}
    except Exception as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def compact_backup_status(status: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(status, Mapping):
        return None
    keys = [
        "status", "backup_verified", "time", "remaining_changed_files", "commit",
        "changed_files", "packages_this_run", "release", "tracked_files",
    ]
    return {k: status.get(k) for k in keys if k in status}


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if not isinstance(packages, list) or not packages:
        return False
    for package in packages:
        if not isinstance(package, Mapping):
            return False
        if not package.get("sha256") or not package.get("bytes") or not package.get("verification"):
            return False
    return True


def ready_matches_current(ready: Any, current_request_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not current_request_id:
        return False
    if ready.get("request_id") == current_request_id:
        return True
    if ready.get("supersedes_request_id") == current_request_id:
        return True
    covered = ready.get("covers_request_ids") or ready.get("covered_request_ids")
    return isinstance(covered, list) and current_request_id in covered


def sqlite_token_total() -> Dict[str, Any]:
    """Return only aggregate token counts; do not read/store prompts."""
    result: Dict[str, Any] = {
        "available": False,
        "path_exists": RESEARCH_SQLITE.exists(),
        "table": None,
        "column": None,
        "call_count": None,
        "total_tokens": None,
        "error": None,
    }
    if not RESEARCH_SQLITE.exists():
        result["error"] = "research.sqlite not found"
        return result
    try:
        conn = sqlite3.connect(f"file:{RESEARCH_SQLITE}?mode=ro", uri=True, timeout=5.0)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [str(r[0]) for r in cur.fetchall()]
        preferred_tables = [t for t in tables if t == "calls"] + [t for t in tables if t != "calls"]
        for table in preferred_tables:
            cur.execute(f"PRAGMA table_info({table!r})")
            columns = [str(row[1]) for row in cur.fetchall()]
            candidate_cols = [c for c in columns if c == "total_tokens"] + [c for c in columns if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
            if not candidate_cols:
                continue
            col = candidate_cols[0]
            # Table/column names come only from sqlite metadata above. Quote using doubled quotes.
            q_table = '"' + table.replace('"', '""') + '"'
            q_col = '"' + col.replace('"', '""') + '"'
            cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {q_col} IS NULL THEN 0 ELSE {q_col} END) FROM {q_table}")
            count, total = cur.fetchone()
            result.update({
                "available": True,
                "table": table,
                "column": col,
                "call_count": int(count or 0),
                "total_tokens": int(total or 0),
                "error": None,
            })
            break
        if not result["available"]:
            result["error"] = "no total_tokens-like column found"
        conn.close()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return result


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-metadata-only-recheck", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_metadata_only_recheck:
        raise SystemExit("missing explicit metadata-only recheck acknowledgement")

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    continue_path = ROOT / f"research_artifacts/aws_state/continue_state_{stamp}_after_v31_postbackup_astra_gate_recheck.md"
    proof_path = BACKUP_DIR / f"backup_proof_{stamp}_from_supervisor_status_after_v31_gate_status.json"
    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_POSTBACKUP_ASTRA_GATE_RECHECK_{stamp}.json"
    marker = f"<!-- {NAME}-{stamp} -->"

    write_json(out / "run_started.json", {
        "started_utc": created.isoformat(),
        "method": NAME,
        "classification": "metadata_only_reversible_backup_astra_gate_recheck",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    # Backup adequacy is evaluated before this script modifies docs/registry.
    preexisting_missing = [rel(p) for p in PREEXISTING_REQUIRED if not p.exists()]
    min_required_time = max_existing_mtime(PREEXISTING_REQUIRED)
    backup_status = safe_json(BACKUP_STATUS)
    status_time = parse_time(backup_status.get("time") if isinstance(backup_status, Mapping) else None)
    verified = bool(isinstance(backup_status, Mapping) and (backup_status.get("status") == "verified" or backup_status.get("backup_verified") is True))
    try:
        remaining_ok = int(backup_status.get("remaining_changed_files", -1)) == 0 if isinstance(backup_status, Mapping) else False
    except Exception:
        remaining_ok = False
    has_commit = bool(isinstance(backup_status, Mapping) and backup_status.get("commit"))
    packages_ok = bool(isinstance(backup_status, Mapping) and package_metadata_ok(backup_status))
    time_ok = bool(status_time and status_time >= min_required_time)
    preexisting_v31_backup_ok = bool((not preexisting_missing) and verified and remaining_ok and has_commit and packages_ok and time_ok)
    why_not: list[str] = []
    if preexisting_missing:
        why_not.append("missing_preexisting_required_artifacts")
    if not verified:
        why_not.append("backup_status_not_verified_or_missing")
    if not remaining_ok:
        why_not.append("backup_status_remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why_not.append("backup_status_missing_commit")
    if not packages_ok:
        why_not.append("backup_status_missing_verified_package_metadata")
    if not time_ok:
        why_not.append("backup_status_time_predates_required_artifacts_or_unparseable")

    if preexisting_v31_backup_ok and isinstance(backup_status, Mapping):
        proof = dict(compact_backup_status(backup_status) or {})
        proof.update({
            "backup_verified": True,
            "created_utc": created.isoformat(),
            "source": "sanitized supervisor backup_status.json captured by post-v31 backup/Astra recheck",
            "covers_preexisting_v31_cluster_and_gate_status_outputs": True,
            "min_required_backup_time_utc": min_required_time.isoformat(),
            "required_preexisting_artifacts": [rel(p) for p in PREEXISTING_REQUIRED],
            "note": "This proof materializes the already-completed supervisor backup. The proof file itself, this script, run registry, and this recheck's outputs were created after that backup and require the new backup request.",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_simulations": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
        })
        write_json(proof_path, proof)
        proof_written = rel(proof_path)
    else:
        proof_written = None

    next_review = safe_json(NEXT_REVIEW)
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    analysis_ready = safe_json(ANALYSIS_READY)
    astra_ready = ready_matches_current(analysis_ready, current_request_id)
    report_path = None
    if isinstance(analysis_ready, Mapping):
        report_path = analysis_ready.get("report") or analysis_ready.get("report_path")

    token_info = sqlite_token_total()

    # Always request backup for this operational recheck's new source/output/proof.
    write_json(request_path, {
        "requested_utc": created.isoformat(),
        "reason": "Backup request after post-v31 backup/Astra gate recheck; covers new recheck source/output/proof/docs/registry before any subsequent unique simulation/refit/validation.",
        "preexisting_v31_backup_ok_before_this_recheck": preexisting_v31_backup_ok,
        "preexisting_backup_block_reasons_if_any": why_not,
        "must_cover": [
            rel(Path(__file__).resolve()),
            rel(out),
            rel(continue_path),
            rel(proof_path),
            rel(request_path),
            rel(ROOT / "STATUS.md"),
            rel(ROOT / "RESEARCH_LOG.md"),
            rel(ROOT / "DECISIONS.md"),
            rel(ROOT / "RESULTS_AUDIT.md"),
            rel(ROOT / "REPRODUCTION_PROTOCOL.md"),
            rel(ROOT / "EXPERIMENT_REGISTRY.csv"),
            rel(RESPONSE_LOG),
        ],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    })

    if astra_ready:
        next_action = "Read the matching/superseding Astra report and execute only its evidence-supported plan after backing up this recheck output if new unique science is required."
    elif preexisting_v31_backup_ok:
        next_action = "Pre-existing v31 outputs are externally backed up, but Astra analysis for v31 is still pending; do not choose a fresh scientific branch. Continue only reversible preparation/integrity checks or implement an already-frozen non-scientific repair."
    else:
        next_action = "Backup remains inadequate for pre-existing v31 outputs; do not run new simulation/refit/validation. Diagnose backup only."

    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "metadata_only_reversible_backup_astra_gate_recheck_no_sim_no_validation_no_test_no_training",
        "preexisting_required_artifacts": [path_info(p) for p in PREEXISTING_REQUIRED],
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_subset": compact_backup_status(backup_status),
        "preexisting_backup_gate": {
            "adequate_for_preexisting_v31_outputs": preexisting_v31_backup_ok,
            "why_not": why_not,
            "min_required_backup_time_utc": min_required_time.isoformat(),
            "backup_status_time_utc": None if status_time is None else status_time.isoformat(),
            "proof_written": proof_written,
            "new_outputs_require_followup_backup": True,
            "followup_backup_request": rel(request_path),
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
        "server_api_token_total_from_research_sqlite": token_info,
        "git": {
            "head": git(["rev-parse", "HEAD"]),
            "status_selected_after_writing_outputs": git(["status", "--short", "--", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0.py", "docs/bohn2021_takeover/astra_reviews", "research_artifacts/aws_diagnostics", "research_artifacts/aws_state", "research_artifacts/aws_backup_proofs", "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"]),
        },
        "budget_actual": {
            "new_development_simulation_episodes": 0,
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
    write_json(out / "raw.json", raw)

    token_sentence = "unknown"
    if token_info.get("available"):
        total = int(token_info.get("total_tokens") or 0)
        token_sentence = f"{total:,} ({total/1_000_000:.3f}M) from research.sqlite table `{token_info.get('table')}` column `{token_info.get('column')}` over {token_info.get('call_count')} rows"
    elif token_info.get("error"):
        token_sentence = f"unknown ({token_info.get('error')})"

    elapsed = raw["elapsed_since_first_supervisor_event_seconds"]
    days = int(elapsed // 86400)
    rem = elapsed - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    seconds = rem - minutes * 60
    elapsed_text = f"{days}d {hours}h {minutes}m {seconds:.3f}s"

    summary = "\n".join([
        "# v31 post-backup/Astra gate recheck",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only/reversible audit; no simulations, no control steps, no selector refits, no training, no validation64 bank open, no sealed-test access.",
        "",
        "## Required status-line values",
        f"- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.",
        f"- Server API total_tokens from research.sqlite: `{token_sentence}`.",
        "",
        "## Backup gate",
        f"- Pre-existing v31 outputs externally backed up: `{preexisting_v31_backup_ok}`.",
        f"- Backup status time: `{raw['preexisting_backup_gate']['backup_status_time_utc']}`; minimum required artifact mtime: `{raw['preexisting_backup_gate']['min_required_backup_time_utc']}`.",
        f"- Reasons if blocked: `{why_not}`.",
        f"- Materialized proof: `{proof_written}`.",
        f"- New recheck outputs/source require follow-up backup: `True`; request `{rel(request_path)}`.",
        "",
        "## Astra gate",
        f"- Current NEXT_REVIEW_REQUEST id: `{current_request_id}`; expected `{EXPECTED_REQUEST_ID}`.",
        f"- ANALYSIS_READY present: `{ANALYSIS_READY.exists()}`; matches/supersedes current: `{astra_ready}`; report `{report_path}`.",
        "",
        "## Next executor action",
        next_action,
    ]) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.parent.mkdir(parents=True, exist_ok=True)
    continue_path.write_text(summary, encoding="utf-8")

    response_block = f"""## Operational follow-up v31 post-backup/Astra gate recheck

Updated by GPT-5.5 executor at `{created.isoformat()}`. This is an operational metadata-only audit; it adds no scientific labels, no simulations, no training/refit, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; pre-existing v31 artifacts covered but new recheck artifacts pending backup | Supervisor backup status time `{raw['preexisting_backup_gate']['backup_status_time_utc']}` vs min required `{raw['preexisting_backup_gate']['min_required_backup_time_utc']}`; adequate=`{preexisting_v31_backup_ok}`; proof `{proof_written}`. | Do not run unique simulation/refit/validation until this recheck output/source/proof are backed up. |
| Astra role-correction handoff | accepted; still pending | NEXT_REVIEW_REQUEST id `{current_request_id}`; ANALYSIS_READY exists=`{ANALYSIS_READY.exists()}`, matches current=`{astra_ready}`. | If matching report appears, read and implement it; otherwise avoid new scientific branch selection. |
"""
    append_once(RESPONSE_LOG, marker, response_block)

    doc_block = f"""## 2026-09-30 v31 post-backup/Astra gate recheck

UTC: {created.isoformat()}. Metadata-only/reversible audit completed; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_sentence}`. Pre-existing v31 outputs backed up=`{preexisting_v31_backup_ok}` (proof `{proof_written}`), but this recheck's source/output/proof require follow-up backup request `{rel(request_path)}`. Astra ANALYSIS_READY for `{current_request_id}` present/matching=`{astra_ready}`. Next action: {next_action}. Artifacts: `{rel(out / 'summary.md')}`, `{rel(out / 'raw.json')}`, `{rel(out / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old_reg = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    row = f"{created.isoformat()},{NAME},metadata_postbackup_astra_gate_recheck,no_validation_no_test,0,0,0,0,0,{preexisting_v31_backup_ok},{rel(out / 'completed.json')}\n"
    if NAME not in old_reg[-50000:]:
        reg.write_text(old_reg.rstrip() + "\n" + row, encoding="utf-8")

    files_for_hash = [
        Path(__file__).resolve(), out / "run_started.json", out / "raw.json", out / "summary.md",
        continue_path, proof_path, request_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md",
        ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
    ] + PREEXISTING_REQUIRED
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "classification": raw["classification"],
        "preexisting_backup_gate": raw["preexisting_backup_gate"],
        "astra_gate": raw["astra_gate"],
        "server_api_token_total_from_research_sqlite": token_info,
        "new_development_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_action": next_action,
        "raw": rel(out / "raw.json"),
        "summary": rel(out / "summary.md"),
        "backup_request": rel(request_path),
        "proof_written": proof_written,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files_for_hash), key=lambda x: rel(x)) if p.exists() and p.is_file()},
    }
    write_json(out / "completed.json", completed)

    print(json.dumps({
        "completed": rel(out / "completed.json"),
        "summary": rel(out / "summary.md"),
        "preexisting_v31_backup_ok": preexisting_v31_backup_ok,
        "proof_written": proof_written,
        "analysis_ready_matches_current": astra_ready,
        "analysis_ready_report": report_path,
        "token_total_available": token_info.get("available"),
        "server_total_tokens": token_info.get("total_tokens"),
        "new_outputs_need_backup": True,
        "backup_request": rel(request_path),
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
