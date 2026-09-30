#!/usr/bin/env python3
"""Materialize the 2026-09-30T04:23 supervisor backup proof and recheck Astra.

Operational metadata-only script. It performs no MPC simulation, no rollout,
no selector refit, no training/value refit, no validation64 access and no
sealed-test access. It records the explicit user/supervisor context reporting a
verified external backup at 2026-09-30T04:23:35Z, checks that known v31/v30b
artifacts predate that backup, checks for ANALYSIS_READY for the current Astra
request, and writes durable state plus a follow-up backup request for this run's
new outputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0"
ASTRA = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA / "ANALYSIS_READY.json"
LATEST = ASTRA / "LATEST.md"
RESPONSE_LOG = ASTRA / "RESPONSE_LOG.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"
SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite"]

USER_CONTEXT_BACKUP: Dict[str, Any] = {
    "time": "2026-09-30T04:23:35.649634+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "122927b8b338861842766471537414af8020c480",
    "changed_files": 22,
    "packages_this_run": [
        {
            "name": "20260930T042332_284b2996.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T042332_284b2996.tar.gz",
            "id": 600023226,
            "sha256": "2c6b3cf0693e34d8229f38b9c801b7fcc279d4918af45876599e353306de9835",
            "bytes": 15638710,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 148898,
}

# Known scientific/operational artifacts that existed before the 04:23 backup.
# This list intentionally excludes this new script/output/proof because those are
# created after the backup and therefore need a follow-up backup.
PRE_0423_ARTIFACTS = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0.py",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_user_context_backup_materialize_v0.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/completed.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/completed.json",
    ROOT / "research_artifacts/aws_runs/20260930T040615_4f9729e6/registry.json",
    ROOT / "research_artifacts/aws_runs/20260930T041028_4dccdd03/registry.json",
    ROOT / "research_artifacts/aws_runs/20260930T042029_5b596afb/registry.json",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0405Z_after_v31_cluster_stability_diagnostic.md",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0407_after_v31_executor_state.md",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T0415Z_after_v31_backup_astra_gate_status.md",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T042029Z_after_v31_postbackup_astra_gate_recheck.md",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_20260930T0405Z.json",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_BACKUP_ASTRA_GATE_STATUS_20260930T0415Z.json",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_POSTBACKUP_ASTRA_GATE_RECHECK_20260930T042029Z.json",
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


def parse_time(v: Any) -> Optional[dt.datetime]:
    if not isinstance(v, str):
        return None
    try:
        t = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


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


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def ready_matches(ready: Any, request_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not request_id:
        return False
    if ready.get("request_id") == request_id or ready.get("supersedes_request_id") == request_id:
        return True
    covered = ready.get("covers_request_ids") or ready.get("covered_request_ids")
    return isinstance(covered, list) and request_id in covered


def sqlite_tokens() -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "path": None, "table": None, "column": None, "call_count": None, "total_tokens": None, "error": None}
    for p in SQLITE_CANDIDATES:
        if not p.exists():
            continue
        out["path"] = rel(p)
        try:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [str(r[0]) for r in cur.fetchall()]
            for table in ([t for t in tables if t == "calls"] + [t for t in tables if t != "calls"]):
                qt = '"' + table.replace('"', '""') + '"'
                cur.execute(f"PRAGMA table_info({qt})")
                cols = [str(r[1]) for r in cur.fetchall()]
                token_cols = [c for c in cols if c == "total_tokens"] + [c for c in cols if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
                if not token_cols:
                    continue
                col = token_cols[0]
                qc = '"' + col.replace('"', '""') + '"'
                cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {qc} IS NULL THEN 0 ELSE {qc} END) FROM {qt}")
                count, total = cur.fetchone()
                conn.close()
                out.update({"available": True, "table": table, "column": col, "call_count": int(count or 0), "total_tokens": int(total or 0), "error": None})
                return out
            conn.close()
            out["error"] = "sqlite present but no total_tokens-like column found"
            return out
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return out
    out["error"] = "research.sqlite not found in checked locations"
    return out


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - 86400 * days
    hours = int(rem // 3600)
    rem -= 3600 * hours
    mins = int(rem // 60)
    sec = rem - 60 * mins
    return f"{days}d {hours}h {mins}m {sec:.3f}s"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-0423-user-context-backup-proof", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_0423_user_context_backup_proof:
        raise SystemExit("missing explicit 04:23 user-context backup proof acknowledgement")

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    proof_path = BACKUP_DIR / "backup_proof_20260930T042335_from_user_context_after_v31_postbackup_recheck.json"
    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_0423_BACKUP_ASTRA_MATERIALIZE_{stamp}.json"
    continue_path = ROOT / f"research_artifacts/aws_state/continue_state_{stamp}_after_v31_0423_backup_astra_materialize.md"
    marker = f"<!-- {NAME}-{stamp} -->"

    pre_infos = [path_info(p) for p in PRE_0423_ARTIFACTS]
    missing = [i["path"] for i in pre_infos if not i["exists"]]
    backup_time = parse_time(USER_CONTEXT_BACKUP["time"])
    mtimes = [parse_time(i.get("mtime_utc")) for i in pre_infos if i.get("mtime_utc")]
    latest_mtime = max([t for t in mtimes if t is not None], default=None)
    packages = USER_CONTEXT_BACKUP.get("packages_this_run") or []
    backup_valid = bool(
        USER_CONTEXT_BACKUP.get("status") == "verified"
        and USER_CONTEXT_BACKUP.get("remaining_changed_files") == 0
        and USER_CONTEXT_BACKUP.get("commit")
        and isinstance(packages, list) and packages
        and all(isinstance(p, Mapping) and p.get("sha256") and p.get("bytes") and p.get("verification") for p in packages)
        and backup_time is not None
    )
    artifact_times_ok = bool(latest_mtime and backup_time and latest_mtime <= backup_time)
    covers_known_pre_0423 = bool(backup_valid and not missing and artifact_times_ok)

    write_json(out_dir / "run_started.json", {
        "started_utc": created.isoformat(),
        "method": NAME,
        "classification": "metadata_only_0423_backup_materialization_astra_gate_recheck",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    next_req = safe_json(NEXT_REVIEW)
    current_id = next_req.get("request_id") if isinstance(next_req, Mapping) else None
    ready = safe_json(ANALYSIS_READY)
    astra_ready = ready_matches(ready, current_id)
    report_path = None
    if isinstance(ready, Mapping):
        report_path = ready.get("report") or ready.get("report_path")
    tokens = sqlite_tokens()
    elapsed = (created - FIRST).total_seconds()
    elapsed_text = fmt_elapsed(elapsed)
    token_text = "unknown"
    if tokens.get("available"):
        total = int(tokens.get("total_tokens") or 0)
        token_text = f"{total:,} ({total/1_000_000:.3f}M) from {tokens.get('path')} table `{tokens.get('table')}` column `{tokens.get('column')}` over {tokens.get('call_count')} rows"
    elif tokens.get("error"):
        token_text = f"unknown ({tokens.get('error')})"

    proof = {
        "proof_file_created_utc": created.isoformat(),
        "source": "explicit user/supervisor context in current instruction, reporting verified backup after the prior iteration",
        "backup": USER_CONTEXT_BACKUP,
        "backup_valid": backup_valid,
        "covers_known_pre_0423_v31_and_postbackup_recheck_artifacts": covers_known_pre_0423,
        "latest_known_pre_0423_artifact_mtime_utc": None if latest_mtime is None else latest_mtime.isoformat(),
        "known_pre_0423_artifacts": pre_infos,
        "missing_known_pre_0423_artifacts": missing,
        "scope_limitations": [
            "This proof file, this script, this run's outputs, and docs/state/registry edits made by this run are after the 04:23 backup and require the next external backup before unique simulation/refit/validation.",
            "The proof is materialized from the explicit supervisor context; repository tools cannot independently download the GitHub release asset here.",
            "No final-test, validation64, simulation, control, training, or refit evidence is added by this metadata-only step.",
        ],
        "budgets_actual": {"new_simulations": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
    }
    write_json(proof_path, proof)

    if astra_ready:
        next_action = "Read the matching/superseding Astra report and execute its selected plan after backing up this metadata run if the plan requires unique simulation/refit/validation."
    else:
        next_action = "Astra analysis for v31 is still pending. Continue only reversible preparation/integrity work and do not choose a fresh scientific branch until a matching/superseding report is read."

    write_json(request_path, {
        "requested_utc": created.isoformat(),
        "reason": "backup request after 04:23 backup proof materialization and Astra gate recheck; required before any new unique simulation/refit/validation",
        "pre_0423_artifacts_covered_by_materialized_context_proof": covers_known_pre_0423,
        "must_cover": [
            rel(Path(__file__).resolve()), rel(out_dir), rel(proof_path), rel(request_path), rel(continue_path),
            rel(RESPONSE_LOG), rel(ROOT / "STATUS.md"), rel(ROOT / "RESEARCH_LOG.md"), rel(ROOT / "DECISIONS.md"),
            rel(ROOT / "RESULTS_AUDIT.md"), rel(ROOT / "REPRODUCTION_PROTOCOL.md"), rel(ROOT / "EXPERIMENT_REGISTRY.csv"),
        ],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    })

    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "method": NAME,
        "classification": "metadata_only_backup_materialization_astra_gate_no_sim_no_validation_no_test_no_training",
        "backup_valid": backup_valid,
        "covers_known_pre_0423_artifacts": covers_known_pre_0423,
        "backup_proof": rel(proof_path),
        "backup_request": rel(request_path),
        "missing_known_pre_0423_artifacts": missing,
        "known_pre_0423_artifacts": pre_infos,
        "astra_gate": {
            "latest_md_exists": LATEST.exists(),
            "next_review_request_id": current_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": ANALYSIS_READY.exists(),
            "analysis_ready_request_id": ready.get("request_id") if isinstance(ready, Mapping) else None,
            "analysis_ready_matches_or_supersedes_current": astra_ready,
            "analysis_ready_report": report_path,
        },
        "server_api_token_total_from_research_sqlite": tokens,
        "budget_actual": {"new_development_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "next_action": next_action,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(out_dir / "raw.json", raw)

    summary = "\n".join([
        "# v31 04:23 backup proof materialization and Astra gate recheck",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only/reversible audit; no simulations, no control steps, no selector refits, no training/value refit, no validation64 bank open, no sealed-test access.",
        "",
        "## Required status-line values",
        f"- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.",
        f"- Server API total_tokens from research.sqlite: `{token_text}`.",
        "",
        "## Backup gate",
        f"- User-context backup valid: `{backup_valid}`.",
        f"- Covers known pre-04:23 v31/postbackup-recheck artifacts: `{covers_known_pre_0423}`.",
        f"- Backup time: `{USER_CONTEXT_BACKUP['time']}`; commit `{USER_CONTEXT_BACKUP['commit']}`; package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}`.",
        f"- Latest known pre-04:23 artifact mtime: `{None if latest_mtime is None else latest_mtime.isoformat()}`.",
        f"- Materialized proof: `{rel(proof_path)}`.",
        f"- This run's outputs need next backup: `{rel(request_path)}`.",
        "",
        "## Astra gate",
        f"- Current NEXT_REVIEW_REQUEST id: `{current_id}`; expected `{EXPECTED_REQUEST_ID}`.",
        f"- ANALYSIS_READY present: `{ANALYSIS_READY.exists()}`; matches/supersedes current: `{astra_ready}`; report `{report_path}`.",
        "",
        "## Next executor action",
        next_action,
    ]) + "\n"
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.parent.mkdir(parents=True, exist_ok=True)
    continue_path.write_text(summary, encoding="utf-8")

    response_block = f"""## Operational follow-up v31 04:23 backup proof materialization

Updated by GPT-5.5 executor at `{created.isoformat()}`. Metadata-only; no simulations, no control steps, no training/refit, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; latest pre-run v31/v30b operational artifacts are now evidence-linked to the 04:23 verified user-context backup; this run remains pending backup | User/supervisor context: backup `{USER_CONTEXT_BACKUP['time']}`, remaining_changed_files=0, commit `{USER_CONTEXT_BACKUP['commit']}`, package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}`. Materialized proof `{rel(proof_path)}`; known pre-04:23 artifacts covered=`{covers_known_pre_0423}`. | Request backup `{rel(request_path)}` before unique simulation/refit/validation. |
| Astra role-correction handoff | accepted; still pending unless ANALYSIS_READY appears | NEXT_REVIEW_REQUEST id `{current_id}`; ANALYSIS_READY present=`{ANALYSIS_READY.exists()}`, matches current=`{astra_ready}`, report=`{report_path}`. | Do not select a fresh scientific branch; read matching Astra report when available and execute its plan. |
"""
    append_once(RESPONSE_LOG, marker, response_block)

    doc_block = f"""## 2026-09-30 v31 04:23 backup/Astra materialization

UTC: {created.isoformat()}. Metadata-only/reversible step; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_text}`. The explicit user-context backup `{USER_CONTEXT_BACKUP['time']}` / commit `{USER_CONTEXT_BACKUP['commit']}` / package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}` is materialized at `{rel(proof_path)}`; known pre-04:23 v31/postbackup-recheck artifacts covered=`{covers_known_pre_0423}`. This run's source/output/proof/doc edits require `{rel(request_path)}` before unique simulation/refit/validation. Astra ANALYSIS_READY for `{current_id}` matching=`{astra_ready}`. Next action: {next_action}. Artifacts: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    row = f"{created.isoformat()},{NAME},metadata_0423_backup_astra_materialization,no_validation_no_test,0,0,0,0,0,{covers_known_pre_0423},{rel(out_dir / 'completed.json')}\n"
    if NAME not in old[-80000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "raw.json", out_dir / "summary.md", continue_path, proof_path, request_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"] + PRE_0423_ARTIFACTS
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "classification": raw["classification"],
        "backup_valid": backup_valid,
        "covers_known_pre_0423_artifacts": covers_known_pre_0423,
        "backup_proof": rel(proof_path),
        "backup_request": rel(request_path),
        "astra_gate": raw["astra_gate"],
        "server_api_token_total_from_research_sqlite": tokens,
        "new_development_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_action": next_action,
        "raw": rel(out_dir / "raw.json"),
        "summary": rel(out_dir / "summary.md"),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda x: rel(x)) if p.exists() and p.is_file()},
    }
    write_json(out_dir / "completed.json", completed)

    print(json.dumps({
        "completed": rel(out_dir / "completed.json"),
        "summary": rel(out_dir / "summary.md"),
        "backup_valid": backup_valid,
        "covers_known_pre_0423_artifacts": covers_known_pre_0423,
        "backup_proof": rel(proof_path),
        "backup_request": rel(request_path),
        "analysis_ready_matches_current": astra_ready,
        "analysis_ready_report": report_path,
        "server_total_tokens": tokens.get("total_tokens"),
        "token_total_available": tokens.get("available"),
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
