#!/usr/bin/env python3
"""Materialize the post-v31 supervisor backup proof supplied in user context.

No simulation, no MPC rollout, no selector refit, no training, no validation64
bank access, and no sealed-test access.  The supervisor context reports a
verified backup at 2026-09-30T04:17:35Z covering changed files through the
previous iteration.  Repository tools cannot read the supervisor backup_status
file directly, so this script writes an evidence-scoped proof JSON from the
explicit user/supervisor context and records that subsequent local edits made by
this script and the prior 04:20 recheck still need another backup before any new
unique science.
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
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RESEARCH_SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite"]
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST_REVIEW = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
NAME = "vehicle_true_variable_horizon_v31_user_context_backup_materialize_v0"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"
USER_CONTEXT_BACKUP = {
    "time": "2026-09-30T04:17:35.864312+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "1736b76b39f6d7f71e0ad35a223b943e98ac2bc8",
    "changed_files": 41,
    "packages_this_run": [
        {
            "name": "20260930T041732_8d4dc5b8.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T041732_8d4dc5b8.tar.gz",
            "id": 600013683,
            "sha256": "0f703ca87a5b46dca0425ed7a2fb9f17352866c3ae3890350b5b0ae51da9e83e",
            "bytes": 15814014,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 148886,
}
# Artifacts known to be produced before the user-context backup and therefore
# covered by the backup if the supervisor context is accepted.  Mutable docs are
# not included here because this script and the prior 04:20 recheck changed them
after the backup; the backup commit covers their previous versions.
PRE_BACKUP_V31_EVIDENCE = [
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
]
POST_BACKUP_LOCAL_ARTIFACTS_KNOWN = [
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0.py",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/run_started.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/summary.md",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/completed.json",
    ROOT / "research_artifacts/aws_runs/20260930T042029_5b596afb/registry.json",
    ROOT / "research_artifacts/aws_state/continue_state_20260930T042029Z_after_v31_postbackup_astra_gate_recheck.md",
    BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_POSTBACKUP_ASTRA_GATE_RECHECK_20260930T042029Z.json",
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_json(path: Path) -> Optional[Any]:
    try:
        return read_json(path)
    except Exception:
        return None


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
    return {
        "path": rel(path),
        "exists": True,
        "bytes": st.st_size,
        "sha256": sha256(path) if path.is_file() else None,
        "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
    }


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


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
    result: Dict[str, Any] = {"available": False, "path": None, "table": None, "column": None, "call_count": None, "total_tokens": None, "error": None}
    for candidate in RESEARCH_SQLITE_CANDIDATES:
        if not candidate.exists():
            continue
        result["path"] = rel(candidate)
        try:
            conn = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [str(row[0]) for row in cur.fetchall()]
            for table in ([t for t in tables if t == "calls"] + [t for t in tables if t != "calls"]):
                q_table = '"' + table.replace('"', '""') + '"'
                cur.execute(f"PRAGMA table_info({q_table})")
                columns = [str(row[1]) for row in cur.fetchall()]
                token_cols = [c for c in columns if c == "total_tokens"] + [c for c in columns if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
                if not token_cols:
                    continue
                col = token_cols[0]
                q_col = '"' + col.replace('"', '""') + '"'
                cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {q_col} IS NULL THEN 0 ELSE {q_col} END) FROM {q_table}")
                count, total = cur.fetchone()
                conn.close()
                result.update({"available": True, "table": table, "column": col, "call_count": int(count or 0), "total_tokens": int(total or 0), "error": None})
                return result
            conn.close()
            result["error"] = "sqlite present but no total_tokens-like column found"
            return result
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return result
    result["error"] = "research.sqlite not found in checked locations"
    return result


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {sec:.3f}s"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-user-context-backup-proof", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_user_context_backup_proof:
        raise SystemExit("missing explicit user-context backup proof acknowledgement")

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    proof_path = BACKUP_DIR / "backup_proof_20260930T041735_from_user_context_after_v31_gate_status.json"
    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_USER_CONTEXT_BACKUP_MATERIALIZATION_{stamp}.json"
    continue_path = ROOT / f"research_artifacts/aws_state/continue_state_{stamp}_after_v31_user_context_backup_materialization.md"
    marker = f"<!-- {NAME}-{stamp} -->"

    write_json(out / "run_started.json", {
        "started_utc": created.isoformat(),
        "method": NAME,
        "classification": "metadata_only_user_context_backup_materialization",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    backup_time = parse_time(USER_CONTEXT_BACKUP["time"])
    evidence_infos = [path_info(p) for p in PRE_BACKUP_V31_EVIDENCE]
    missing = [info["path"] for info in evidence_infos if not info["exists"]]
    latest_mtime = max((parse_time(info["mtime_utc"]) for info in evidence_infos if info.get("mtime_utc")), default=None)
    evidence_files_pre_backup = bool(latest_mtime and backup_time and latest_mtime <= backup_time)
    user_context_backup_valid = bool(USER_CONTEXT_BACKUP["status"] == "verified" and USER_CONTEXT_BACKUP["remaining_changed_files"] == 0 and USER_CONTEXT_BACKUP["commit"] and USER_CONTEXT_BACKUP["packages_this_run"] and backup_time)
    covers_pre_backup_v31_evidence = bool(user_context_backup_valid and not missing and evidence_files_pre_backup)

    proof = {
        "proof_file_created_utc": created.isoformat(),
        "source": "explicit user/supervisor context in current instruction; local repository tools could not read /data/openai-agent/state/backup_status.json",
        "backup": USER_CONTEXT_BACKUP,
        "backup_verified": user_context_backup_valid,
        "covers_pre_backup_v31_scientific_and_gate_artifacts": covers_pre_backup_v31_evidence,
        "latest_pre_backup_v31_artifact_mtime_utc": None if latest_mtime is None else latest_mtime.isoformat(),
        "pre_backup_v31_evidence_artifacts": evidence_infos,
        "missing_pre_backup_v31_evidence_artifacts": missing,
        "scope_limitations": [
            "The proof file itself is created after the external backup and therefore needs the next backup.",
            "The prior 04:20 postbackup recheck script/output/docs and this materialization script/output/docs are after the 04:17 backup and need the next backup before unique simulation/refit/validation.",
            "Mutable docs such as STATUS.md/RESEARCH_LOG.md/RESPONSE_LOG.md have changed after the 04:17 backup; the backup commit covers their pre-04:17 versions only.",
        ],
        "post_backup_local_artifacts_known_not_covered": [path_info(p) for p in POST_BACKUP_LOCAL_ARTIFACTS_KNOWN],
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "budgets_actual": {"new_simulations": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0},
    }
    write_json(proof_path, proof)

    next_review = safe_json(NEXT_REVIEW)
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    analysis_ready = safe_json(ANALYSIS_READY)
    astra_ready = ready_matches_current(analysis_ready, current_request_id)
    report_path = None
    if isinstance(analysis_ready, Mapping):
        report_path = analysis_ready.get("report") or analysis_ready.get("report_path")

    token_info = sqlite_token_total()
    elapsed_seconds = (created - FIRST_SUPERVISOR_EVENT).total_seconds()
    elapsed_text = fmt_elapsed(elapsed_seconds)
    token_text = "unknown"
    if token_info.get("available"):
        total = int(token_info.get("total_tokens") or 0)
        token_text = f"{total:,} ({total / 1_000_000:.3f}M) from {token_info.get('path')} table `{token_info.get('table')}` column `{token_info.get('column')}` over {token_info.get('call_count')} rows"
    elif token_info.get("error"):
        token_text = f"unknown ({token_info.get('error')})"

    write_json(request_path, {
        "requested_utc": created.isoformat(),
        "reason": "Backup request after materializing user-context proof and writing operational docs/state; required before any new unique simulation/refit/validation.",
        "must_cover": [
            rel(Path(__file__).resolve()), rel(out), rel(proof_path), rel(request_path), rel(continue_path),
            rel(RESPONSE_LOG), rel(ROOT / "STATUS.md"), rel(ROOT / "RESEARCH_LOG.md"), rel(ROOT / "DECISIONS.md"),
            rel(ROOT / "RESULTS_AUDIT.md"), rel(ROOT / "REPRODUCTION_PROTOCOL.md"), rel(ROOT / "EXPERIMENT_REGISTRY.csv"),
        ],
        "pre_backup_v31_evidence_already_covered_by_user_context_proof": covers_pre_backup_v31_evidence,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    })

    if astra_ready:
        next_action = "Read the matching/superseding Astra report and implement its selected plan after obtaining backup for the post-04:17 operational outputs if the plan requires unique simulation/refit/validation."
    else:
        next_action = "Astra analysis for v31 is still pending. Pre-04:17 v31 evidence is covered by the user-context backup, but post-04:17 operational outputs require another backup; until then avoid new unique simulation/refit/validation and continue only reversible preparation."

    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "method": NAME,
        "classification": "metadata_only_user_context_backup_materialization_no_sim_no_validation_no_test_no_training",
        "user_context_backup_valid": user_context_backup_valid,
        "covers_pre_backup_v31_evidence": covers_pre_backup_v31_evidence,
        "proof_path": rel(proof_path),
        "backup_request": rel(request_path),
        "pre_backup_v31_evidence_artifacts": evidence_infos,
        "post_backup_local_artifacts_known_not_covered": [path_info(p) for p in POST_BACKUP_LOCAL_ARTIFACTS_KNOWN],
        "astra_gate": {
            "latest_md_exists": LATEST_REVIEW.exists(),
            "next_review_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": ANALYSIS_READY.exists(),
            "analysis_ready_request_id": analysis_ready.get("request_id") if isinstance(analysis_ready, Mapping) else None,
            "analysis_ready_matches_or_supersedes_current": astra_ready,
            "analysis_ready_report": report_path,
        },
        "server_api_token_total_from_research_sqlite": token_info,
        "budget_actual": {"new_development_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "next_action": next_action,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(out / "raw.json", raw)

    summary = "\n".join([
        "# v31 user-context backup proof materialization",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only/reversible proof materialization; no simulations, no control steps, no selector refits, no training, no validation64 bank open, no sealed-test access.",
        "",
        "## Required status-line values",
        f"- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.",
        f"- Server API total_tokens from research.sqlite: `{token_text}`.",
        "",
        "## Backup interpretation",
        f"- User-context backup valid: `{user_context_backup_valid}`.",
        f"- Covers pre-04:17 v31 evidence/gate artifacts: `{covers_pre_backup_v31_evidence}`.",
        f"- Backup time: `{USER_CONTEXT_BACKUP['time']}`; commit `{USER_CONTEXT_BACKUP['commit']}`; package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}`.",
        f"- Materialized proof: `{rel(proof_path)}`.",
        f"- New post-04:17 operational outputs require backup request: `{rel(request_path)}`.",
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

    response_block = f"""## Operational follow-up v31 user-context backup proof materialization

Updated by GPT-5.5 executor at `{created.isoformat()}`. This is an operational metadata-only step; it adds no scientific labels, no simulations, no training/refit, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; pre-04:17 v31 artifacts now have a materialized user-context proof, while post-04:17 operational outputs remain pending backup | User/supervisor context reports verified backup `{USER_CONTEXT_BACKUP['time']}` with remaining_changed_files=0, commit `{USER_CONTEXT_BACKUP['commit']}`, package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}`. Materialized proof `{rel(proof_path)}`; current run request `{rel(request_path)}`. | Do not run unique simulation/refit/validation until this materialization/recheck output is also externally backed up. |
| Astra role-correction handoff | accepted; still pending | NEXT_REVIEW_REQUEST id `{current_request_id}`; ANALYSIS_READY exists=`{ANALYSIS_READY.exists()}`, matches current=`{astra_ready}`. | Await/read matching Astra report before selecting a fresh scientific branch. |
"""
    append_once(RESPONSE_LOG, marker, response_block)

    doc_block = f"""## 2026-09-30 v31 user-context backup proof materialization

UTC: {created.isoformat()}. Metadata-only/reversible step; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_text}`. User-context backup `{USER_CONTEXT_BACKUP['time']}` / commit `{USER_CONTEXT_BACKUP['commit']}` / package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}` is materialized at `{rel(proof_path)}` and covers pre-04:17 v31 evidence/gate artifacts=`{covers_pre_backup_v31_evidence}`. This proof file, the prior 04:20 recheck, this materialization run, docs/state/registry updates remain unbacked and require `{rel(request_path)}` before unique simulation/refit/validation. Astra ANALYSIS_READY for `{current_request_id}` present/matching=`{astra_ready}`. Next action: {next_action}. Artifacts: `{rel(out / 'summary.md')}`, `{rel(out / 'raw.json')}`, `{rel(out / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    row = f"{created.isoformat()},{NAME},metadata_user_context_backup_materialization,no_validation_no_test,0,0,0,0,0,{covers_pre_backup_v31_evidence},{rel(out / 'completed.json')}\n"
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    files = [Path(__file__).resolve(), out / "run_started.json", out / "raw.json", out / "summary.md", continue_path, proof_path, request_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"] + PRE_BACKUP_V31_EVIDENCE + POST_BACKUP_LOCAL_ARTIFACTS_KNOWN
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "classification": raw["classification"],
        "user_context_backup_valid": user_context_backup_valid,
        "covers_pre_backup_v31_evidence": covers_pre_backup_v31_evidence,
        "proof_path": rel(proof_path),
        "backup_request": rel(request_path),
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
        "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda x: rel(x)) if p.exists() and p.is_file()},
    }
    write_json(out / "completed.json", completed)

    print(json.dumps({
        "completed": rel(out / "completed.json"),
        "summary": rel(out / "summary.md"),
        "user_context_backup_valid": user_context_backup_valid,
        "covers_pre_backup_v31_evidence": covers_pre_backup_v31_evidence,
        "proof_path": rel(proof_path),
        "backup_request": rel(request_path),
        "analysis_ready_matches_current": astra_ready,
        "analysis_ready_report": report_path,
        "server_total_tokens": token_info.get("total_tokens"),
        "token_total_available": token_info.get("available"),
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
