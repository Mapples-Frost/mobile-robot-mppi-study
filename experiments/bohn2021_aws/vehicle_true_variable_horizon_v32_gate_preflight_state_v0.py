#!/usr/bin/env python3
"""V32 backup/Astra gate preflight and evidence-link verification.

This is an operational/integrity action only. It verifies that the current
NEXT_REVIEW_REQUEST is evidence-linked, checks whether a backup proof exists
that can cover the v32 diagnostic outputs and refreshed Astra request, checks
ANALYSIS_READY status, and writes a durable continuation packet.

It must not run MPC, open validation64 bank content, access sealed tests,
train/refit a model/selector, or generate new scientific labels.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v32_gate_preflight_state_v0"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_DIR = ROOT / "research_artifacts/aws_state"
REQUEST_TO_SATISFY = BACKUP_DIR / "REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json"
CURRENT_REQUEST_ID = "v32-h12-supported-default-h35-diagnostic-20260930T051611Z"
LATEST_OLD_REPORT = "docs/bohn2021_takeover/astra_reviews/20260929T153837Z.md"
SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite", Path("/data/openai-agent/state/research.sqlite")]

# Latest backup summary explicitly supplied by the supervisor/user context for
# this iteration. Its timestamp predates v32 outputs and therefore cannot by
# itself clear the current v32 post-output backup gate, but recording it avoids
# losing the provenance boundary.
USER_CONTEXT_BACKUP = {
    "time": "2026-09-30T05:15:38.012838+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "f2227d24fada4c7e7ba5fb3e7cbe060b47457cc1",
    "changed_files": 36,
    "packages_this_run": [
        {
            "name": "20260930T051534_c9fcd67c.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T051534_c9fcd67c.tar.gz",
            "id": 600102144,
            "sha256": "081ce0831d187e40aa419262e36ff7a7fcd2a08208a2fc45e7b1196b36f9e876",
            "bytes": 16338056,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 149023,
    "source": "current_supervisor_context",
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {sec:.3f}s"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def append_registry_once(row_id: str, row: str) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if row_id not in old:
        path.write_text(old.rstrip() + "\n" + row.strip() + "\n", encoding="utf-8")


def file_info(rel_path: str) -> Dict[str, Any]:
    p = ROOT / rel_path
    out: Dict[str, Any] = {"path": rel_path, "exists": p.exists(), "is_file": p.is_file()}
    if p.exists():
        st = p.stat()
        out.update({
            "bytes": st.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
            "sha256": sha256(p) if p.is_file() else None,
        })
    return out


def token_total() -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "path": None, "table": None, "column": None, "call_count": None, "total_tokens": None, "error": None}
    seen: set[str] = set()
    for candidate in SQLITE_CANDIDATES:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        result["path"] = str(candidate)
        try:
            conn = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [str(row[0]) for row in cur.fetchall()]
            for table in ([t for t in tables if t == "calls"] + [t for t in tables if t != "calls"]):
                qt = '"' + table.replace('"', '""') + '"'
                cur.execute(f"PRAGMA table_info({qt})")
                cols = [str(row[1]) for row in cur.fetchall()]
                candidates = [c for c in cols if c == "total_tokens"] + [c for c in cols if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
                if not candidates:
                    continue
                col = candidates[0]
                qc = '"' + col.replace('"', '""') + '"'
                cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {qc} IS NULL THEN 0 ELSE {qc} END) FROM {qt}")
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


def ready_matches(ready: Any, request_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not request_id:
        return False
    if ready.get("request_id") == request_id or ready.get("supersedes_request_id") == request_id:
        return True
    for key in ("covers_request_ids", "covered_request_ids"):
        v = ready.get(key)
        if isinstance(v, list) and request_id in v:
            return True
    return False


def backup_candidate_from_obj(path: str, obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, Mapping):
        return None
    backup = obj.get("backup") if isinstance(obj.get("backup"), Mapping) else obj
    status = backup.get("status") if isinstance(backup, Mapping) else obj.get("status")
    remaining = backup.get("remaining_changed_files") if isinstance(backup, Mapping) else obj.get("remaining_changed_files")
    commit = backup.get("commit") if isinstance(backup, Mapping) else obj.get("commit")
    packages = backup.get("packages_this_run") if isinstance(backup, Mapping) else obj.get("packages_this_run")
    t = backup.get("time") if isinstance(backup, Mapping) else obj.get("time")
    if t is None:
        t = obj.get("proof_file_created_utc") or obj.get("created_utc") or obj.get("timestamp_utc")
    parsed = parse_time(t)
    if not parsed:
        return None
    package_ok = False
    package_sha = None
    if isinstance(packages, list) and packages:
        p0 = packages[0]
        if isinstance(p0, Mapping):
            package_sha = p0.get("sha256")
            package_ok = bool(package_sha and p0.get("verification") == "github_server_sha256" and (p0.get("bytes") or 0) > 0)
    if status == "verified" and remaining == 0 and commit and package_ok:
        return {
            "path": path,
            "time": parsed.isoformat(),
            "status": status,
            "remaining_changed_files": remaining,
            "commit": commit,
            "package_sha256": package_sha,
            "package_ok": package_ok,
        }
    return None


def collect_backup_candidates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cand = backup_candidate_from_obj("current_supervisor_context", USER_CONTEXT_BACKUP)
    if cand:
        out.append(cand)
    if BACKUP_DIR.exists():
        for p in BACKUP_DIR.glob("backup_proof_*.json"):
            obj = read_json(p)
            cand = backup_candidate_from_obj(rel(p), obj)
            if cand:
                out.append(cand)
    out.sort(key=lambda x: x["time"])
    return out


def read_text_head(path: Path, limit: int = 800) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:limit]


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    started = {
        "started_utc": created.isoformat(),
        "classification": "metadata_only_v32_gate_preflight_no_sim_no_validation_no_test_no_training_no_refit",
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(out_dir / "run_started.json", started)

    tokens = token_total()
    elapsed_text = fmt_elapsed((created - FIRST_SUPERVISOR_EVENT).total_seconds())
    token_text = (
        f"{int(tokens.get('total_tokens') or 0):,} ({int(tokens.get('total_tokens') or 0) / 1_000_000:.3f}M)"
        if tokens.get("available")
        else f"unknown ({tokens.get('error')})"
    )

    request = read_json(REQUEST_TO_SATISFY) or {}
    request_time = parse_time(request.get("requested_utc") if isinstance(request, Mapping) else None)
    must_cover = [str(x) for x in (request.get("must_cover", []) if isinstance(request, Mapping) else [])]
    infos = [file_info(p) for p in must_cover]
    missing = [x["path"] for x in infos if not x.get("exists")]

    next_review = read_json(NEXT_REVIEW) or {}
    evidence_paths = [str(x) for x in (next_review.get("evidence_paths", []) if isinstance(next_review, Mapping) else [])]
    evidence_infos = [file_info(p) for p in evidence_paths]
    missing_evidence = [x["path"] for x in evidence_infos if not x.get("exists")]
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None

    ready = read_json(ANALYSIS_READY)
    ready_exists = ANALYSIS_READY.exists()
    ready_match = ready_matches(ready, current_request_id)
    ready_report = None
    if isinstance(ready, Mapping):
        ready_report = ready.get("report") or ready.get("report_path")
    latest_md_head = read_text_head(LATEST)

    backup_candidates = collect_backup_candidates()
    latest_backup = backup_candidates[-1] if backup_candidates else None
    backup_time = parse_time(latest_backup.get("time")) if latest_backup else None
    after_backup: List[Dict[str, str]] = []
    if backup_time:
        for x in infos:
            mt = parse_time(x.get("mtime_utc"))
            if mt and mt > backup_time + dt.timedelta(seconds=2):
                after_backup.append({"path": x["path"], "mtime_utc": mt.isoformat()})
    request_present = REQUEST_TO_SATISFY.exists()
    files_present = request_present and not missing
    latest_backup_after_request = bool(backup_time and request_time and backup_time >= request_time)
    backup_gate_clear = bool(latest_backup and files_present and not after_backup and latest_backup_after_request)

    v32_raw_path = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0_20260930T051611Z/raw.json"
    v32_raw = read_json(v32_raw_path) or {}
    v32_digest: Dict[str, Any] = {"exists": v32_raw_path.exists(), "raw_sha256": sha256(v32_raw_path)}
    if isinstance(v32_raw, Mapping):
        v32_digest.update({
            "classification": v32_raw.get("classification"),
            "family_counts": v32_raw.get("family_counts"),
            "h12_default_h35_logo_bad_count": (((v32_raw.get("logo_h12_default_h35") or {}).get("aggregate_metric") or {}).get("bad_count")),
            "h12_default_h35_h_counts": (((v32_raw.get("logo_h12_default_h35") or {}).get("aggregate_metric") or {}).get("h_counts")),
            "h12_default_h35_saving_vs_fixed_h35": (((v32_raw.get("logo_h12_default_h35") or {}).get("aggregate_metric") or {}).get("decision_saving_vs_fixed_H35_on_same_rows")),
            "oracle_bad_count": (((v32_raw.get("comparators_on_same_opened_rows") or {}).get("oracle_h12_h15_h35") or {}).get("bad_count")),
            "fixed_h35_bad_count": ((((v32_raw.get("comparators_on_same_opened_rows") or {}).get("fixed") or {}).get("35") or {}).get("bad_count")),
            "not_validation_or_test_evidence": ((v32_raw.get("interpretation_for_astra") or {}).get("not_validation_or_test_evidence")),
            "diagnostic_not_branch_decision": ((v32_raw.get("interpretation_for_astra") or {}).get("diagnostic_not_branch_decision")),
        })

    branch_neutral_execution_matrix = {
        "fresh_source_independent_label_acquisition": {
            "executor_status": "prepared_to_implement_if_astra_selects; not started here",
            "blocked_by": ["current backup gate" if not backup_gate_clear else None, "Astra analysis" if not ready_match else None],
            "would_count_as_unique_science": True,
        },
        "terminal_risk_value_refit_or_training_ablation": {
            "executor_status": "prepared_to_implement_if_astra_selects; not started here",
            "blocked_by": ["current backup gate" if not backup_gate_clear else None, "Astra analysis" if not ready_match else None],
            "would_count_as_unique_science": True,
        },
        "scenario_or_comparison_protocol_redesign": {
            "executor_status": "can draft protocol after Astra report; no branch selected here",
            "blocked_by": ["Astra analysis" if not ready_match else None],
            "would_count_as_unique_science": "protocol drafting no; new rollouts yes",
        },
    }
    # JSON-clean the optional blocker entries.
    for item in branch_neutral_execution_matrix.values():
        item["blocked_by"] = [x for x in item["blocked_by"] if x]

    if ready_match:
        next_action = "Read the matching/superseding Astra report, verify cited evidence, update RESPONSE_LOG dispositions, then implement the selected plan after latest metadata outputs are backed up."
    else:
        next_action = "Astra analysis for v32 is not yet available; continue only reversible integrity/preparation and do not select a new acquisition/refit/training/scenario branch."
    if not backup_gate_clear:
        next_action = "Latest v32 backup gate is not clear; request/await verified external backup covering v32 outputs, refreshed Astra request and this preflight before unique science. " + next_action

    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V32_GATE_PREFLIGHT_STATE_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v32_gate_preflight.md"
    planned_cover = [
        rel(Path(__file__).resolve()),
        rel(out_dir / "run_started.json"),
        rel(out_dir / "summary.md"),
        rel(out_dir / "raw.json"),
        rel(out_dir / "completed.json"),
        rel(backup_request),
        rel(continue_path),
        "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
        "STATUS.md",
        "RESEARCH_LOG.md",
        "DECISIONS.md",
        "RESULTS_AUDIT.md",
        "REPRODUCTION_PROTOCOL.md",
        "EXPERIMENT_REGISTRY.csv",
    ]
    write_json(backup_request, {
        "requested_utc": created.isoformat(),
        "reason": "Back up v32 gate preflight/state-preservation metadata before unique simulation/refit/training/validation/final-test work.",
        "must_cover": planned_cover,
        "evaluated_prior_request": rel(REQUEST_TO_SATISFY),
        "prior_v32_backup_gate_clear_by_local_check": backup_gate_clear,
        "analysis_ready_matches_current_at_preflight": ready_match,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    raw = {
        "created_utc": created.isoformat(),
        "classification": started["classification"],
        "elapsed_since_first_supervisor_event_text": elapsed_text,
        "server_api_token_total_from_research_sqlite": tokens,
        "current_next_review_request": {
            "path": rel(NEXT_REVIEW),
            "request_id": current_request_id,
            "expected_request_id": CURRENT_REQUEST_ID,
            "status": next_review.get("status") if isinstance(next_review, Mapping) else None,
            "supersedes_request_id": next_review.get("supersedes_request_id") if isinstance(next_review, Mapping) else None,
            "evidence_path_count": len(evidence_paths),
            "missing_evidence_paths": missing_evidence,
            "evidence_infos": evidence_infos,
        },
        "astra_gate": {
            "analysis_ready_path": rel(ANALYSIS_READY),
            "analysis_ready_exists": ready_exists,
            "analysis_ready_matches_or_supersedes_current": ready_match,
            "analysis_ready_report": ready_report,
            "latest_md_head": latest_md_head,
            "latest_old_report_predates_v29_v32": LATEST_OLD_REPORT in latest_md_head,
        },
        "backup_gate": {
            "request_to_satisfy": rel(REQUEST_TO_SATISFY),
            "request_present": request_present,
            "request_time": request_time.isoformat() if request_time else None,
            "must_cover_count": len(must_cover),
            "files_present": files_present,
            "missing_files": missing,
            "latest_verified_backup_candidate": latest_backup,
            "latest_backup_after_request_time": latest_backup_after_request,
            "files_with_mtime_after_latest_backup": after_backup,
            "backup_gate_clear_by_local_check": backup_gate_clear,
            "new_backup_request": rel(backup_request),
        },
        "v32_digest": v32_digest,
        "branch_neutral_execution_matrix": branch_neutral_execution_matrix,
        "budgets_actual": {
            "new_simulation_episodes": 0,
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
        "python": sys.version,
    }
    write_json(out_dir / "raw.json", raw)

    missing_evidence_text = "none" if not missing_evidence else ", ".join(missing_evidence)
    latest_backup_text = latest_backup if latest_backup else "none"
    summary = f"""# v32 gate preflight / state preservation

UTC: `{created.isoformat()}`. Metadata-only integrity/preparation action; no simulations, no control steps, no selector refit, no training, no validation64 bank access and no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## Evidence-link check
- Current NEXT_REVIEW_REQUEST: `{current_request_id}` (expected `{CURRENT_REQUEST_ID}`), status `{next_review.get('status') if isinstance(next_review, Mapping) else None}`.
- Evidence paths listed: `{len(evidence_paths)}`; missing evidence paths: `{missing_evidence_text}`.
- V32 digest: family_counts `{v32_digest.get('family_counts')}`, H12/H35 diagnostic LOGO bad_count `{v32_digest.get('h12_default_h35_logo_bad_count')}`, h_counts `{v32_digest.get('h12_default_h35_h_counts')}`, saving vs fixed H35 `{v32_digest.get('h12_default_h35_saving_vs_fixed_h35')}`.

## Backup gate check
- Request evaluated: `{rel(REQUEST_TO_SATISFY)}`; present `{request_present}`, must_cover `{len(must_cover)}` files, missing `{len(missing)}`.
- Latest verified backup candidate found: `{latest_backup_text}`.
- Latest backup is after request time: `{latest_backup_after_request}`.
- Must-cover files with mtimes after latest backup: `{len(after_backup)}`.
- Current v32 backup gate clear by local check: `{backup_gate_clear}`.
- New metadata outputs from this preflight require follow-up backup: `{rel(backup_request)}`.

## Astra gate check
- ANALYSIS_READY present: `{ready_exists}`; matches/supersedes current request: `{ready_match}`; report `{ready_report}`.
- `LATEST.md` still references the old report when `latest_old_report_predates_v29_v32=True`: `{LATEST_OLD_REPORT in latest_md_head}`.

## Next action
{next_action}
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    log_block = f"""## v32 gate preflight / state preservation

Updated by GPT-5.5 executor at `{created.isoformat()}`. Metadata-only integrity/preparation; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| Astra v32 direction request | pending | NEXT_REVIEW_REQUEST `{current_request_id}` has `{len(evidence_paths)}` evidence paths; missing evidence paths=`{missing_evidence}`. ANALYSIS_READY exists=`{ready_exists}`, matches/supersedes current=`{ready_match}`. `LATEST.md` remains old report=`{LATEST_OLD_REPORT in latest_md_head}`. | Do not choose acquisition/refit/training/scenario branch until matching Astra report is read and verified. |
| `A12_registry_backup_schema_contract` | accepted; latest v32 output backup gate checked and still not clear by local evidence | Evaluated `{rel(REQUEST_TO_SATISFY)}` with must_cover `{len(must_cover)}`, missing `{len(missing)}`; latest verified backup candidate `{latest_backup}`; after-request backup=`{latest_backup_after_request}`; files after latest backup count `{len(after_backup)}`; gate_clear=`{backup_gate_clear}`. | New request `{rel(backup_request)}` covers this metadata run. Unique simulation/refit/training/validation/final-test work remains blocked until external backup covers v32 and this preflight. |
| v32 H12-supported/H35-default diagnostic | evidence verified; interpretation deferred to Astra | Raw digest: family_counts `{v32_digest.get('family_counts')}`, LOGO bad_count `{v32_digest.get('h12_default_h35_logo_bad_count')}`, h_counts `{v32_digest.get('h12_default_h35_h_counts')}`, saving vs fixed H35 `{v32_digest.get('h12_default_h35_saving_vs_fixed_h35')}`. | Preserve as opened-development analysis only; not validation/test/speed/reproduction evidence and not a deployable three-way selector. |
"""
    append_once(RESPONSE_LOG, marker, log_block)

    doc_block = f"""## 2026-09-30 v32 gate preflight/state preservation

UTC: {created.isoformat()}. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_text}`. Verified current Astra request `{current_request_id}` has `{len(evidence_paths)}` evidence paths with missing=`{missing_evidence}`. ANALYSIS_READY matching current=`{ready_match}`. Backup request `{rel(REQUEST_TO_SATISFY)}` present=`{request_present}` and files present=`{files_present}`, but latest verified backup candidate `{latest_backup}` is after request time=`{latest_backup_after_request}` and files-after-backup count=`{len(after_backup)}`, so v32 backup gate clear=`{backup_gate_clear}`. V32 digest remains opened-development only: counts `{v32_digest.get('family_counts')}`, H12/H35 LOGO bad `{v32_digest.get('h12_default_h35_logo_bad_count')}`, saving vs fixed H35 `{v32_digest.get('h12_default_h35_saving_vs_fixed_h35')}`. Next: await backup plus matching Astra report before unique science; if Astra appears, read/verify/report dispositions.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    registry_id = f"{stamp}_{NAME}"
    append_registry_once(
        registry_id,
        f"{registry_id},{created.isoformat()},metadata_only_v32_gate_preflight_state_no_science,none,no_validation_no_test,unknown,complete,0,0,0,{rel(out_dir / 'raw.json')}"
    )

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request": rel(backup_request),
        "continue_state": rel(continue_path),
        "backup_gate_clear_by_local_check": backup_gate_clear,
        "analysis_ready_matches_or_supersedes_current": ready_match,
        "missing_evidence_paths": missing_evidence,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)
    hash_paths = [
        Path(__file__).resolve(),
        out_dir / "run_started.json",
        out_dir / "summary.md",
        out_dir / "raw.json",
        out_dir / "completed.json",
        backup_request,
        continue_path,
        RESPONSE_LOG,
        ROOT / "STATUS.md",
        ROOT / "RESEARCH_LOG.md",
        ROOT / "DECISIONS.md",
        ROOT / "RESULTS_AUDIT.md",
        ROOT / "REPRODUCTION_PROTOCOL.md",
        ROOT / "EXPERIMENT_REGISTRY.csv",
    ]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths}
    write_json(out_dir / "completed.json", completed)
    raw["completed_hash"] = sha256(out_dir / "completed.json")
    write_json(out_dir / "raw.json", raw)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
