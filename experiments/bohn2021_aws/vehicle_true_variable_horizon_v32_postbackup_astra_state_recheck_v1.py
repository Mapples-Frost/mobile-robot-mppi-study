#!/usr/bin/env python3
"""Metadata-only post-backup/Astra state recheck for v32 handoff.

This script performs no simulations, no control rollouts, no training, no
selector refits, no validation64 bank reads, and no sealed-test reads.  It only
materializes the verified backup claim supplied by the current supervisor
context, checks outstanding backup requests against that proof, rechecks Astra
handoff readiness, summarizes already-opened v31/v32 diagnostics, audits
server-token totals from aggregate SQLite usage JSON, and writes durable state.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
CURRENT_REQUEST_ID = "v32-h12-supported-default-h35-diagnostic-20260930T051611Z"
NAME = "vehicle_true_variable_horizon_v32_postbackup_astra_state_recheck_v1"

# Copied from the latest explicit user/supervisor context.  This script does not
# perform a backup operation; it records/verifies the claim for local gate logic.
SUPERVISOR_BACKUP_053041 = {
    "time": "2026-09-30T05:30:41.005783+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "2ff567eeaced865b54be46711d4eb9853a832156",
    "changed_files": 35,
    "packages_this_run": [
        {
            "name": "20260930T053037_a47dbe04.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T053037_a47dbe04.tar.gz",
            "id": 600120752,
            "sha256": "1ef95bdbb2e9d77eed32706a471f06983fd4eda26f60305898b82d14709501af",
            "bytes": 16465943,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 149084,
}

PRIOR_REQUEST_PATTERNS = [
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT*.json",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32*.json",
]
V31_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/raw.json"
V32_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0_20260930T051611Z/raw.json"
V32_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0_20260930T051611Z/summary.md"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
STATUS = ROOT / "STATUS.md"
RESEARCH_LOG = ROOT / "RESEARCH_LOG.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_DIR = ROOT / "research_artifacts/aws_state"
DIAG_DIR = ROOT / "research_artifacts/aws_diagnostics"

ACCESS_ZERO = {
    "new_simulation_episodes": 0,
    "new_control_steps": 0,
    "new_training_or_gradient_steps": 0,
    "selector_refits": 0,
    "validation64_bank_opened": False,
    "validation64_episodes": 0,
    "sealed_test_accessed": False,
    "sealed_test_episodes": 0,
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(x: dt.datetime) -> str:
    return x.astimezone(dt.timezone.utc).isoformat()


def parse_iso(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        out = dt.datetime.fromisoformat(s)
        if out.tzinfo is None:
            out = out.replace(tzinfo=dt.timezone.utc)
        return out.astimezone(dt.timezone.utc)
    except Exception:
        return None


def fmt_elapsed(delta: dt.timedelta) -> str:
    total = delta.total_seconds()
    days = int(total // 86400)
    rem = total - days * 86400
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
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def file_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": rel(path), "exists": path.exists(), "is_file": path.is_file()}
    if path.exists():
        st = path.stat()
        info.update({
            "bytes": st.st_size,
            "mtime_utc": iso(dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)),
            "sha256": sha256(path) if path.is_file() else None,
        })
    return info


def materialize_backup_claim(created: dt.datetime) -> Path:
    path = BACKUP_DIR / "backup_proof_20260930T053041_from_user_context_after_v32_token_gate_recheck.json"
    obj = {
        "proof_file_created_by": "GPT-5.5 executor from explicit current user/supervisor context; no backup operation was performed by this script",
        "proof_file_created_utc": iso(created),
        "backup": SUPERVISOR_BACKUP_053041,
        "backup_claim_valid": True,
        "interpretation": (
            "This materializes the prompt-supplied verified backup at 2026-09-30T05:30:41Z. "
            "It is used to verify coverage of files written before that time, including the 05:24 API-token audit "
            "and 05:29 v32/token gate recheck. The proof file created now and this recheck output require a later backup."
        ),
        "access_flags": {
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
        },
        "budgets_actual": ACCESS_ZERO,
    }
    if not path.exists():
        write_json(path, obj)
    return path


def backup_candidate_from_obj(label: str, obj: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    backup = obj.get("backup") if isinstance(obj.get("backup"), Mapping) else obj
    if not isinstance(backup, Mapping):
        return None
    btime = parse_iso(backup.get("time") or backup.get("timestamp") or obj.get("time") or obj.get("timestamp"))
    if not btime:
        return None
    packages = backup.get("packages_this_run") or obj.get("packages_this_run") or []
    package_sha = None
    package_verification = None
    if isinstance(packages, list) and packages:
        first = packages[0]
        if isinstance(first, Mapping):
            package_sha = first.get("sha256")
            package_verification = first.get("verification")
    return {
        "path": label,
        "time": iso(btime),
        "status": backup.get("status") or obj.get("status"),
        "remaining_changed_files": backup.get("remaining_changed_files", obj.get("remaining_changed_files")),
        "commit": backup.get("commit") or obj.get("commit"),
        "package_sha256": package_sha,
        "package_verification": package_verification,
        "package_ok": bool(package_sha) and package_verification in {"github_server_sha256", "download_sha256", "sha256"},
    }


def load_backup_candidates(materialized: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pstr in glob.glob(str(BACKUP_DIR / "backup_proof_*.json")):
        p = Path(pstr)
        obj = read_json(p)
        if isinstance(obj, Mapping):
            cand = backup_candidate_from_obj(rel(p), obj)
            if cand:
                out.append(cand)
    # Add context claim independently; the newly materialized proof file has an
    # mtime after the backup, but the context claim itself is valid for older files.
    cand = backup_candidate_from_obj("current_supervisor_context_backup_20260930T053041", {"backup": SUPERVISOR_BACKUP_053041})
    if cand:
        if not any(x.get("time") == cand.get("time") and x.get("commit") == cand.get("commit") for x in out):
            out.append(cand)
    out.sort(key=lambda x: x.get("time", ""), reverse=True)
    return out


def request_files() -> List[Path]:
    found: List[Path] = []
    for pat in PRIOR_REQUEST_PATTERNS:
        found.extend(Path(x) for x in glob.glob(str(ROOT / pat)))
    unique = sorted({p.resolve(): p for p in found}.values(), key=lambda p: rel(p))
    return unique


def check_request(req_path: Path, cand: Dict[str, Any]) -> Dict[str, Any]:
    req = read_json(req_path)
    if not isinstance(req, Mapping):
        return {"request": rel(req_path), "request_exists": req_path.exists(), "clear": False, "error": "request_json_unreadable"}
    must_cover = req.get("must_cover") if isinstance(req.get("must_cover"), list) else []
    btime = parse_iso(cand.get("time"))
    missing: List[str] = []
    after: List[Dict[str, str]] = []
    infos: List[Dict[str, Any]] = []
    for item in must_cover:
        p = ROOT / str(item)
        info = file_info(p)
        infos.append(info)
        if not p.is_file():
            missing.append(str(item))
        elif btime is not None:
            mtime = parse_iso(info.get("mtime_utc"))
            if mtime and mtime > btime:
                after.append({"path": str(item), "mtime_utc": info["mtime_utc"]})
    clear = bool(
        btime
        and cand.get("status") == "verified"
        and cand.get("remaining_changed_files") == 0
        and cand.get("package_ok")
        and not missing
        and not after
    )
    return {
        "request": rel(req_path),
        "requested_utc": req.get("requested_utc"),
        "must_cover_count": len(must_cover),
        "candidate": cand,
        "missing_count": len(missing),
        "missing": missing,
        "files_after_backup_count": len(after),
        "files_after_backup_sample": after[:20],
        "clear": clear,
    }


def best_check(req_path: Path, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks = [check_request(req_path, c) for c in candidates]
    clear = [c for c in checks if c.get("clear")]
    return {"best": clear[0] if clear else (checks[0] if checks else {"request": rel(req_path), "clear": False, "error": "no_backup_candidates"}), "checked_candidates": len(checks)}


def token_usage() -> Dict[str, Any]:
    db = Path("/data/openai-agent/state/research.sqlite")
    if not db.exists():
        return {"available": False, "selected_text": "unknown (research.sqlite not found)", "path": str(db)}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(calls)").fetchall()]
            if "usage" not in cols:
                return {"available": False, "selected_text": "unknown (calls.usage absent)", "path": str(db), "columns": cols}
            total = 0
            rows = 0
            parsed = 0
            for (usage,) in conn.execute("SELECT usage FROM calls"):
                rows += 1
                if usage is None:
                    continue
                try:
                    obj = json.loads(usage) if isinstance(usage, str) else usage
                except Exception:
                    continue
                if isinstance(obj, Mapping) and isinstance(obj.get("total_tokens"), (int, float)):
                    total += int(obj["total_tokens"])
                    parsed += 1
            return {
                "available": True,
                "path": str(db),
                "source": "calls.usage.total_tokens",
                "basis": "json_field_sum",
                "row_count": rows,
                "json_rows_with_total_tokens": parsed,
                "total_tokens": total,
                "selected_text": f"{total:,} ({total / 1_000_000:.3f}M) from calls.usage.total_tokens",
            }
        finally:
            conn.close()
    except Exception as exc:
        return {"available": False, "selected_text": f"unknown ({type(exc).__name__}: {exc})", "path": str(db)}


def astra_status() -> Dict[str, Any]:
    req_path = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
    ready_path = ROOT / "docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json"
    latest_path = ROOT / "docs/bohn2021_takeover/astra_reviews/LATEST.md"
    req = read_json(req_path)
    current = req.get("request_id") if isinstance(req, Mapping) else None
    ready = read_json(ready_path) if ready_path.exists() else None
    ids: List[str] = []
    report_paths: List[str] = []
    if isinstance(ready, Mapping):
        for key in ["request_id", "for_request_id", "covers_request_id", "current_request_id", "supersedes_request_id"]:
            val = ready.get(key)
            if isinstance(val, str):
                ids.append(val)
            elif isinstance(val, list):
                ids.extend([x for x in val if isinstance(x, str)])
        for key in ["report", "report_path", "analysis_path", "path"]:
            val = ready.get(key)
            if isinstance(val, str):
                report_paths.append(val)
    latest = latest_path.read_text(encoding="utf-8", errors="replace") if latest_path.exists() else ""
    return {
        "current_request_id": current,
        "expected_request_id": CURRENT_REQUEST_ID,
        "current_is_expected": current == CURRENT_REQUEST_ID,
        "analysis_ready_exists": ready_path.exists(),
        "analysis_ready_ids": ids,
        "analysis_ready_matches_current": bool(current and current in ids),
        "analysis_ready_report_paths": report_paths,
        "latest_md_path": rel(latest_path),
        "latest_md_old_report_marker": "20260929T153837Z" in latest,
        "latest_md_excerpt": latest[:600],
    }


def extract_v31_v32() -> Dict[str, Any]:
    v31 = read_json(V31_RAW)
    v32 = read_json(V32_RAW)
    out: Dict[str, Any] = {
        "v31_raw": file_info(V31_RAW),
        "v32_raw": file_info(V32_RAW),
        "v32_summary": file_info(V32_SUMMARY),
    }
    if isinstance(v31, Mapping):
        out["v31_family_counts"] = v31.get("current_oracle_label_independent_source_family_counts")
        out["v31_basic_new_families_needed"] = v31.get("budget_bounds_if_astra_selects_fresh_source_label_acquisition", {}).get("basic_logo_class_presence", {}).get("additional_independent_families_by_label") if isinstance(v31.get("budget_bounds_if_astra_selects_fresh_source_label_acquisition"), Mapping) else None
        out["v31_three_source_new_families_needed"] = v31.get("budget_bounds_if_astra_selects_fresh_source_label_acquisition", {}).get("three_source_stability_target", {}).get("additional_independent_families_by_label") if isinstance(v31.get("budget_bounds_if_astra_selects_fresh_source_label_acquisition"), Mapping) else None
    if isinstance(v32, Mapping):
        metric = v32.get("logo_h12_default_h35", {}).get("aggregate_metric", {}) if isinstance(v32.get("logo_h12_default_h35"), Mapping) else {}
        out["v32_family_counts"] = v32.get("family_counts")
        out["v32_logo_h12_default_h35_bad_count"] = metric.get("bad_count") if isinstance(metric, Mapping) else None
        out["v32_logo_h12_default_h35_h_counts"] = metric.get("h_counts") if isinstance(metric, Mapping) else None
        out["v32_logo_h12_default_h35_decision_saving_vs_H35"] = metric.get("decision_saving_vs_fixed_H35_on_same_rows") if isinstance(metric, Mapping) else None
        out["v32_bad_rows"] = metric.get("bad_rows") if isinstance(metric, Mapping) else None
        interp = v32.get("interpretation_for_astra")
        out["v32_interpretation_for_astra"] = interp if isinstance(interp, Mapping) else None
    return out


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = DIAG_DIR / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    run_started = out_dir / "run_started.json"
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    state_path = STATE_DIR / f"continue_state_{stamp}_after_v32_postbackup_astra_state_recheck.md"
    backup_request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V32_POSTBACKUP_ASTRA_STATE_RECHECK_{stamp}.json"

    write_json(run_started, {"started_utc": iso(created), "classification": "metadata_only_no_sim_no_validation_no_test_no_training_no_refit", **ACCESS_ZERO})

    materialized = materialize_backup_claim(created)
    candidates = load_backup_candidates(materialized)
    reqs = request_files()
    checks = {rel(p): best_check(p, candidates) for p in reqs}
    prior_requests_clear = bool(reqs) and all(v.get("best", {}).get("clear") for v in checks.values())
    token = token_usage()
    elapsed = fmt_elapsed(created - FIRST_SUPERVISOR_EVENT)
    astra = astra_status()
    evidence = extract_v31_v32()
    astra_ready = bool(astra.get("analysis_ready_matches_current"))

    decision = (
        "Prior backup gate appears clear from the prompt-supplied 05:30:41 verified backup, "
        "but current unique scientific work remains blocked because Astra analysis for v32 is not ready."
        if prior_requests_clear and not astra_ready else
        "Prior backup gate and Astra gate both appear clear; next cycle must read and verify the Astra report before implementing its plan."
        if prior_requests_clear and astra_ready else
        "Prior backup gate is still not clear locally; continue reversible integrity/preparation only."
    )

    raw = {
        "created_utc": iso(created),
        "classification": "metadata_only_v32_postbackup_astra_state_recheck_no_sim_no_validation_no_test_no_training_no_refit",
        "elapsed_since_first_supervisor_event_text": elapsed,
        "server_api_total_tokens_status_line": token.get("selected_text"),
        "desktop_conversation_tokens_excluded": True,
        "access_and_budget": ACCESS_ZERO,
        "materialized_backup_proof": rel(materialized),
        "backup_candidates_count": len(candidates),
        "latest_backup_candidate": candidates[0] if candidates else None,
        "checked_backup_requests": list(checks.keys()),
        "backup_checks": checks,
        "prior_unique_science_backup_gate_clear": prior_requests_clear,
        "astra_status": astra,
        "astra_gate_ready_for_current_request": astra_ready,
        "evidence_digest": evidence,
        "unique_science_gate_for_next_cycle_clear_before_this_run_outputs": bool(prior_requests_clear and astra_ready),
        "this_run_outputs_require_later_backup": True,
        "decision": decision,
    }
    write_json(raw_path, raw)

    lines: List[str] = []
    lines.append("# v32 post-backup / Astra state recheck")
    lines.append("")
    lines.append(f"UTC: `{iso(created)}`. Metadata-only integrity/handoff check; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.")
    lines.append("")
    lines.append("## Required status-line values")
    lines.append(f"- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed}`.")
    lines.append(f"- Cumulative server API total_tokens from research.sqlite: `{token.get('selected_text')}`; desktop conversation tokens excluded.")
    lines.append("")
    lines.append("## Backup gate")
    lines.append(f"- Materialized prompt-supplied verified backup proof: `{rel(materialized)}`.")
    lines.append(f"- Latest backup candidate: `{candidates[0] if candidates else None}`.")
    lines.append(f"- Prior v32/API-token backup requests checked: `{len(checks)}`; all clear=`{prior_requests_clear}`.")
    for req, chk in checks.items():
        best = chk.get("best", {})
        lines.append(f"  - `{req}` clear=`{best.get('clear')}`; best_time=`{best.get('candidate', {}).get('time')}`; missing={best.get('missing_count')}; files_after_backup={best.get('files_after_backup_count')}")
    lines.append("")
    lines.append("## Astra gate")
    lines.append(f"- Current request: `{astra.get('current_request_id')}`; matches expected=`{astra.get('current_is_expected')}`.")
    lines.append(f"- ANALYSIS_READY exists=`{astra.get('analysis_ready_exists')}`; matches current=`{astra.get('analysis_ready_matches_current')}`.")
    lines.append(f"- LATEST.md still old 20260929 report marker=`{astra.get('latest_md_old_report_marker')}`.")
    lines.append("")
    lines.append("## Evidence digest rechecked")
    lines.append(f"- v31/v32 family counts: v31 `{evidence.get('v31_family_counts')}`, v32 `{evidence.get('v32_family_counts')}`.")
    lines.append(f"- v32 H12-if-low-abs_obs_07 else H35 LOGO bad_count=`{evidence.get('v32_logo_h12_default_h35_bad_count')}`, h_counts=`{evidence.get('v32_logo_h12_default_h35_h_counts')}`, decision saving vs fixed H35=`{evidence.get('v32_logo_h12_default_h35_decision_saving_vs_H35')}`.")
    lines.append(f"- v32 residual bad rows: `{evidence.get('v32_bad_rows')}`.")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- {decision}")
    lines.append("- v29/v30b/v31/v32 remain opened-development diagnostics only; do not claim validation/test/reproduction/deployed-speed/deployable-selector success.")
    lines.append("- This metadata recheck output itself requires a later verified external backup before unique scientific work.")
    lines.append("- Sealed final test remains unopened and unauthorized.")
    lines.append("")
    summary = "\n".join(lines)
    summary_path.write_text(summary, encoding="utf-8")
    state_path.write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    log_block = f"""## v32 post-backup / Astra state recheck

Updated by GPT-5.5 executor at `{iso(created)}`. Metadata-only; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; prior v32/API-token backup gate rechecked | Materialized supervisor-context proof `{rel(materialized)}` for backup time `2026-09-30T05:30:41.005783+00:00`, commit `2ff567eeaced865b54be46711d4eb9853a832156`, package SHA256 `1ef95bdbb2e9d77eed32706a471f06983fd4eda26f60305898b82d14709501af`. Prior v32/API-token requests checked `{len(checks)}`; all clear=`{prior_requests_clear}`. | New request `{rel(backup_request_path)}` covers this recheck/proof/log update and must be backed up before unique scientific work. |
| Astra v32 direction request | pending | Current request `{astra.get('current_request_id')}`; ANALYSIS_READY exists=`{astra.get('analysis_ready_exists')}`, matches current=`{astra.get('analysis_ready_matches_current')}`; LATEST old-report marker=`{astra.get('latest_md_old_report_marker')}`. | Do not choose acquisition/refit/training/scenario branch until matching/superseding Astra report is available and verified. |
| v32 opened-development signal | preserved; interpretation deferred to Astra | v32 LOGO H12-default-H35 bad_count `{evidence.get('v32_logo_h12_default_h35_bad_count')}`, h_counts `{evidence.get('v32_logo_h12_default_h35_h_counts')}`, decision saving `{evidence.get('v32_logo_h12_default_h35_decision_saving_vs_H35')}`, residual bad rows `{evidence.get('v32_bad_rows')}`. | Keep as development-only diagnostic; no deployable selector or validation claim. |
"""
    for doc in [RESPONSE_LOG, STATUS, RESEARCH_LOG]:
        append_once(doc, marker, log_block if doc == RESPONSE_LOG else summary)

    must_cover = [
        rel(Path(__file__).resolve()),
        rel(run_started),
        rel(raw_path),
        rel(summary_path),
        rel(completed_path),
        rel(state_path),
        rel(materialized),
        rel(backup_request_path),
        "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
        "STATUS.md",
        "RESEARCH_LOG.md",
        "EXPERIMENT_REGISTRY.csv",
    ]
    write_json(backup_request_path, {
        "requested_utc": iso(created),
        "reason": "Back up metadata-only v32 post-backup/Astra state recheck outputs and materialized 05:30 supervisor backup proof.",
        "must_cover": must_cover,
        **ACCESS_ZERO,
    })

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": iso(created),
        "classification": raw["classification"],
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "backup_request": rel(backup_request_path),
        "continue_state": rel(state_path),
        "prior_unique_science_backup_gate_clear": prior_requests_clear,
        "astra_gate_ready_for_current_request": astra_ready,
        "this_run_outputs_require_later_backup": True,
        "server_api_total_tokens_status_line": token.get("selected_text"),
        "access_and_budget": ACCESS_ZERO,
        "hashes": {p: sha256(ROOT / p) for p in must_cover if (ROOT / p).is_file()},
    }
    write_json(completed_path, completed)
    # Refresh raw with final completed hash.
    raw["completed_sha256"] = sha256(completed_path)
    write_json(raw_path, raw)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
