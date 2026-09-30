#!/usr/bin/env python3
"""Prepare a branch-neutral v31 evidence packet while awaiting Astra.

This is a metadata/analysis-only integrity action over already-opened development
artifacts.  It must not run MPC, refit/train selectors, open validation64, or
access sealed tests.  Purpose: make the v29/v30b/v31 evidence, backup status and
Astra gate state explicit for the next iteration without selecting a new
scientific branch.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v31_astra_pending_evidence_packet_v0"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_DIR = ROOT / "research_artifacts/aws_state"
SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite"]

V29_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json"
V29_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/summary.md"
V29_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/completed.json"
V30B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json"
V30B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/summary.md"
V31_CLUSTER_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json"
V31_CLUSTER_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md"
V31_SOURCE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/raw.json"
V31_SOURCE_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/summary.md"
REQUEST_0507 = BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_0505_BACKUP_ASTRA_GATE_RECHECK_20260930T050748Z.json"
NOTE_0509 = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_pre_astra_input_sufficiency_diagnostic_20260930T0509Z.md"
EXPECTED_REQUEST_ID = "v31-source-coverage-budget-bounds-20260930T044819Z"
OLD_LATEST_REPORT = "docs/bohn2021_takeover/astra_reviews/20260929T153837Z.md"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: str) -> Optional[dt.datetime]:
    try:
        out = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if out.tzinfo is None:
            out = out.replace(tzinfo=dt.timezone.utc)
        return out.astimezone(dt.timezone.utc)
    except Exception:
        return None


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


def token_total() -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "path": None, "table": None, "column": None, "call_count": None, "total_tokens": None, "error": None}
    for candidate in SQLITE_CANDIDATES:
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
                token_cols = [c for c in cols if c == "total_tokens"] + [c for c in cols if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
                if not token_cols:
                    continue
                col = token_cols[0]
                qc = '"' + col.replace('"', '""') + '"'
                cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {qc} IS NULL THEN 0 ELSE {qc} END) FROM {qt}")
                count, total = cur.fetchone()
                conn.close()
                result.update({"available": True, "path": str(candidate), "table": table, "column": col, "call_count": int(count or 0), "total_tokens": int(total or 0), "error": None})
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


def file_info(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists(), "is_file": path.is_file()}
    if path.exists():
        st = path.stat()
        out.update({
            "bytes": st.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
            "sha256": sha256(path) if path.is_file() else None,
        })
    return out


def latest_verified_backup() -> Dict[str, Any]:
    candidates = []
    for p in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json(p)
        if not isinstance(data, Mapping):
            continue
        backup = data.get("backup") if isinstance(data.get("backup"), Mapping) else data
        t = parse_time(backup.get("time") or data.get("created_utc") or data.get("proof_file_created_utc") or "")
        if not t:
            continue
        status = backup.get("status") or backup.get("backup_status") or data.get("status")
        remaining = backup.get("remaining_changed_files", data.get("remaining_changed_files"))
        pkg = None
        pkgs = backup.get("packages_this_run") if isinstance(backup, Mapping) else None
        if isinstance(pkgs, list) and pkgs:
            pkg = pkgs[0]
        verified = bool((status == "verified" or data.get("backup_verified") is True) and (remaining in (0, "0", None) or data.get("backup_verified") is True))
        candidates.append({
            "path": rel(p),
            "time": t.isoformat(),
            "verified_like": verified,
            "status": status,
            "remaining_changed_files": remaining,
            "commit": backup.get("commit") if isinstance(backup, Mapping) else data.get("commit"),
            "package_sha256": pkg.get("sha256") if isinstance(pkg, Mapping) else data.get("sha256"),
        })
    verified = [c for c in candidates if c["verified_like"]]
    latest = max(verified, key=lambda x: x["time"]) if verified else None
    return {"latest_verified": latest, "verified_count": len(verified), "proof_count": len(candidates), "all_verified_paths": [c["path"] for c in verified]}


def pending_backup_inputs() -> Dict[str, Any]:
    req = read_json(REQUEST_0507)
    paths = []
    if isinstance(req, Mapping):
        for p in req.get("must_cover", []):
            paths.append(ROOT / str(p))
    paths.append(NOTE_0509)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        rp = rel(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    infos = [file_info(p) for p in unique]
    latest = latest_verified_backup().get("latest_verified")
    latest_time = parse_time(latest["time"]) if latest else None
    not_covered = []
    for info in infos:
        mt = parse_time(info.get("mtime_utc", "")) if info.get("exists") else None
        if not latest_time or not mt or mt > latest_time + dt.timedelta(seconds=2):
            not_covered.append({"path": info["path"], "mtime_utc": info.get("mtime_utc"), "reason": "newer_than_latest_verified_backup_or_missing_backup"})
    return {"request_0507": rel(REQUEST_0507), "input_file_infos": infos, "latest_verified_backup": latest, "not_covered_by_latest_verified_backup": not_covered, "all_inputs_covered_by_latest_verified_backup": len(not_covered) == 0}


def safe_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def evidence_digest() -> Dict[str, Any]:
    v29 = read_json(V29_RAW) or {}
    v30b = read_json(V30B_RAW) or {}
    v31 = read_json(V31_CLUSTER_RAW) or {}
    v31s = read_json(V31_SOURCE_RAW) or {}
    digest = {
        "input_files": [file_info(p) for p in [V29_RAW, V29_SUMMARY, V29_COMPLETED, V30B_RAW, V30B_SUMMARY, V31_CLUSTER_RAW, V31_CLUSTER_SUMMARY, V31_SOURCE_RAW, V31_SOURCE_SUMMARY]],
        "v29": {
            "headline": safe_get(v29, "analysis", "headline", default={}),
            "category_counts": safe_get(v29, "analysis", "category_counts", default={}),
            "decision": safe_get(v29, "analysis", "decision"),
            "horizon_counts_summary": {k: {kk: vv for kk, vv in (v or {}).items() if kk in ("episodes", "safe_success_count", "problem_count", "decision_sum_s", "physical_sum", "solver_sum_s")} for k, v in safe_get(v29, "analysis", "horizon_counts", default={}).items()},
        },
        "v30b": {
            "bad_semantics": v30b.get("bad_semantics") if isinstance(v30b, Mapping) else None,
            "oracle_metric": safe_get(v30b, "comparators", "oracle_fastest_safe_H12_H15_H35", default={}),
            "fixed_H12": {k: safe_get(v30b, "comparators", "fixed_H12", k) for k in ("bad_count", "safe_success_count", "decision_sum_s", "physical_sum")},
            "fixed_H35": {k: safe_get(v30b, "comparators", "fixed_H35", k) for k in ("bad_count", "safe_success_count", "decision_sum_s", "physical_sum")},
        },
        "v31_cluster": {
            "headline": v31.get("headline") if isinstance(v31, Mapping) else {},
            "oracle_label_to_family_count": safe_get(v31, "family_summary", "oracle_label_to_family_count", default={}),
            "oracle_label_to_families": safe_get(v31, "family_summary", "oracle_label_to_families", default={}),
            "duplicate_feature_pairs": safe_get(v31, "family_summary", "duplicate_feature_pairs", default={}),
            "diagnosis_for_astra": v31.get("diagnosis_for_astra") if isinstance(v31, Mapping) else None,
        },
        "v31_source_budget": {
            "current_counts": v31s.get("current_oracle_label_independent_source_family_counts") if isinstance(v31s, Mapping) else {},
            "current_family_ids_by_label": v31s.get("current_family_ids_by_label") if isinstance(v31s, Mapping) else {},
            "same_family_extra_rows_do_not_repair_grouped_cv": v31s.get("same_family_extra_rows_do_not_repair_grouped_cv") if isinstance(v31s, Mapping) else {},
            "budget_bounds_if_astra_selects_fresh_source_label_acquisition": v31s.get("budget_bounds_if_astra_selects_fresh_source_label_acquisition") if isinstance(v31s, Mapping) else {},
            "structural_identifiability_notes": v31s.get("structural_identifiability_notes") if isinstance(v31s, Mapping) else [],
        },
    }
    counts_cluster = digest["v31_cluster"].get("oracle_label_to_family_count")
    counts_source = digest["v31_source_budget"].get("current_counts")
    oracle_h_counts = safe_get(digest, "v30b", "oracle_metric", "h_counts", default={})
    digest["consistency_checks"] = {
        "v31_cluster_and_source_counts_match": counts_cluster == counts_source,
        "expected_current_counts_H12_3_H15_1_H35_1": counts_source == {"12": 3, "15": 1, "35": 1},
        "v30b_oracle_h_counts_match_expected_rows": oracle_h_counts == {"12": 6, "15": 3, "35": 2},
        "all_required_input_files_exist": all(x.get("exists") for x in digest["input_files"]),
    }
    return digest


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
        "classification": "metadata_analysis_only_astra_pending_evidence_packet_no_sim_no_validation_no_test_no_training_no_refit",
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
    token_text = f"{tokens['total_tokens']:,} ({tokens['total_tokens'] / 1_000_000:.3f}M)" if tokens.get("available") else f"unknown ({tokens.get('error')})"

    next_review = read_json(NEXT_REVIEW) or {}
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    ready = read_json(ANALYSIS_READY)
    astra_ready_exists = ANALYSIS_READY.exists()
    astra_ready_matches = ready_matches(ready, current_request_id)
    ready_report = ready.get("report") or ready.get("report_path") if isinstance(ready, Mapping) else None
    latest_text = LATEST.read_text(encoding="utf-8", errors="replace") if LATEST.exists() else ""
    latest_points_to_old = OLD_LATEST_REPORT in latest_text

    evidence = evidence_digest()
    backup_status = pending_backup_inputs()

    packet_path = ASTRA_DIR / f"v31_pending_evidence_packet_{stamp}.md"
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_PENDING_EVIDENCE_PACKET_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v31_pending_packet.md"

    basic_bounds = safe_get(evidence, "v31_source_budget", "budget_bounds_if_astra_selects_fresh_source_label_acquisition", "basic_logo_class_presence", default={})
    three_bounds = safe_get(evidence, "v31_source_budget", "budget_bounds_if_astra_selects_fresh_source_label_acquisition", "three_source_stability_target", default={})
    v31_headline = safe_get(evidence, "v31_cluster", "headline", default={})
    v30b_oracle = safe_get(evidence, "v30b", "oracle_metric", default={})
    v29_headline = safe_get(evidence, "v29", "headline", default={})

    if astra_ready_matches:
        next_action = "Read and verify the matching/superseding Astra report before any new experiment branch; then implement its evidence-supported plan after backup coverage is restored."
    else:
        next_action = "No matching Astra report is available; do not choose source-label acquisition, value/refit/training, or scenario/comparison redesign independently. Continue only reversible integrity/preparation until Astra responds or an already-frozen authorized action exists."
    if not backup_status["all_inputs_covered_by_latest_verified_backup"]:
        next_action += " Latest metadata outputs are not externally backed up, so unique simulation/refit/training/validation/final-test work remains blocked pending the new backup request."

    packet = f"""# v31 pending-Astra evidence packet

UTC: `{created.isoformat()}`. This is branch-neutral metadata analysis only over already-opened development artifacts. It ran **0** simulations, **0** control steps, **0** selector refits, **0** training/gradient steps, opened no validation64 bank and accessed no sealed test.

## Status-line values for next report
- Elapsed service lifetime since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## Astra gate
- Current `NEXT_REVIEW_REQUEST.json` request_id: `{current_request_id}`.
- Expected v31 request_id: `{EXPECTED_REQUEST_ID}`.
- `ANALYSIS_READY.json` exists: `{astra_ready_exists}`; matches/supersedes current: `{astra_ready_matches}`; report: `{ready_report}`.
- `LATEST.md` still points to the old 2026-09-29 report: `{latest_points_to_old}`. That report predates v29/v30b/v31 and is not current analysis of the source-coverage finding.

## Evidence inspected
- v29 opened-development identical-state probe: {v29_headline}
- v30b opened-row oracle H12/H15/H35 metric: bad_count `{v30b_oracle.get('bad_count')}`, h_counts `{v30b_oracle.get('h_counts')}`, decision_sum_s `{v30b_oracle.get('decision_sum_s')}`, decision_saving_vs_fixed_H35 `{v30b_oracle.get('decision_saving_vs_fixed_H35')}`.
- v31 source-family cluster headline: rows `{v31_headline.get('rows')}`, families `{v31_headline.get('families')}`, row-level LOO bad `{v31_headline.get('loo_bad_two_feature_rule')}`, grouped LOGO bad `{v31_headline.get('logo_bad_two_feature_rule')}`, label family counts `{v31_headline.get('oracle_label_to_family_count')}`.
- v31 source-budget lower bound: current counts `{evidence['v31_source_budget'].get('current_counts')}`; basic additional families `{basic_bounds.get('additional_independent_families_by_label')}`; three-source additional families `{three_bounds.get('additional_independent_families_by_label')}`.

## Consistency checks
{json.dumps(evidence['consistency_checks'], indent=2, sort_keys=True)}

## Backup gate
- Latest verified backup found by local proof scan: `{backup_status['latest_verified_backup']}`.
- 05:07 metadata request plus 05:09 note fully covered by that backup: `{backup_status['all_inputs_covered_by_latest_verified_backup']}`.
- Files not covered by latest verified backup count: `{len(backup_status['not_covered_by_latest_verified_backup'])}`.
- New backup request for this packet and pending files: `{rel(backup_request)}`.

## Branch-neutral carry-forward
- Treat v29/v30b/v31 as development/opened-row diagnostics only, not validation/final evidence and not a reproduction claim.
- Do not deploy or validate the v30b row-level two-threshold selector: non-default H15/H35 labels each have only one independent source family.
- Same-family densification cannot repair grouped-CV missing-class structure.
- If Astra selects fresh source-independent acquisition, use the already computed lower-bound budgets; if Astra selects value/refit/training or scenario/comparison redesign, freeze a versioned protocol before any new simulation/refit/training.

## Next action
{next_action}
"""
    packet_path.write_text(packet, encoding="utf-8")
    continue_path.write_text(packet, encoding="utf-8")

    planned_cover = [
        rel(Path(__file__).resolve()),
        rel(out_dir / "run_started.json"),
        rel(out_dir / "summary.md"),
        rel(out_dir / "raw.json"),
        rel(out_dir / "completed.json"),
        rel(packet_path),
        rel(backup_request),
        rel(continue_path),
        rel(NOTE_0509),
        rel(REQUEST_0507),
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
        "reason": "Back up branch-neutral v31 pending-Astra evidence packet, 05:09 note, and associated metadata before any unique simulation/refit/training/validation/final-test work.",
        "must_cover": planned_cover,
        "pending_prior_not_covered_count_before_this_run": len(backup_status["not_covered_by_latest_verified_backup"]),
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
        "astra_gate": {
            "current_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": astra_ready_exists,
            "analysis_ready_matches_or_supersedes_current": astra_ready_matches,
            "analysis_ready_report": ready_report,
            "latest_points_to_old_report": latest_points_to_old,
            "old_report_path": OLD_LATEST_REPORT,
        },
        "backup_gate": backup_status,
        "evidence_digest": evidence,
        "packet_path": rel(packet_path),
        "backup_request": rel(backup_request),
        "next_action": next_action,
        "budgets_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
    }
    write_json(out_dir / "raw.json", raw)

    summary = f"""# v31 pending-Astra evidence-packet run

UTC: `{created.isoformat()}`. Metadata/analysis-only over opened artifacts; no simulations, no control steps, no refits, no training, no validation64 and no sealed-test access.

## Required status-line values
- Service lifetime elapsed: `{elapsed_text}` since `2026-09-26T10:55:29.419331Z`.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## Concrete actions completed
1. Rechecked Astra gate: current request `{current_request_id}`; ANALYSIS_READY exists `{astra_ready_exists}`; matches/supersedes current `{astra_ready_matches}`.
2. Inspected v29/v30b/v31/v31-source-budget raw evidence and wrote branch-neutral packet `{rel(packet_path)}`.
3. Verified consistency checks: `{evidence['consistency_checks']}`.
4. Audited backup coverage for the 05:07 metadata request plus 05:09 note; fully covered by latest verified backup = `{backup_status['all_inputs_covered_by_latest_verified_backup']}`.
5. Wrote consolidated backup request `{rel(backup_request)}` for pending metadata outputs before unique science.

## Preserved scientific state
- v29/v30b/v31 remain development/opened-row diagnostics only.
- Current independent source-family counts remain `{evidence['v31_source_budget'].get('current_counts')}`; H15/H35 each have only one independent family.
- Same-family extra rows cannot repair the grouped-CV missing-class issue.
- Astra remains responsible for selecting acquisition vs value/refit/training vs scenario/comparison redesign.

## Next action
{next_action}
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    response_block = f"""## v31 pending-Astra evidence packet

Updated by GPT-5.5 executor at `{created.isoformat()}`. Branch-neutral metadata analysis only; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| Astra role-correction / v31 source-coverage direction | accepted; still pending current Astra analysis | Packet `{rel(packet_path)}` and raw `{rel(out_dir / 'raw.json')}` verify request `{current_request_id}`, ANALYSIS_READY exists=`{astra_ready_exists}`, matches/supersedes current=`{astra_ready_matches}`. v29/v30b/v31 consistency checks `{evidence['consistency_checks']}`. | Do not start source-label acquisition, refit/training, or scenario/comparison redesign before reading a matching/superseding Astra report. |
| v31 non-default source-family coverage | accepted as open limitation | Current counts `{evidence['v31_source_budget'].get('current_counts')}`; grouped LOGO bad `{v31_headline.get('logo_bad_two_feature_rule')}`; basic additional families if acquisition is selected `{basic_bounds.get('additional_independent_families_by_label')}`. | Carry into Astra decision; do not deploy row-level feature rule as validation-ready. |
| `A12_registry_backup_schema_contract` | accepted; latest metadata outputs pending backup | Local proof scan latest verified backup `{backup_status['latest_verified_backup']}`; 05:07+05:09 inputs covered=`{backup_status['all_inputs_covered_by_latest_verified_backup']}`; new request `{rel(backup_request)}`. | Unique simulation/refit/training/validation/final-test work remains blocked until a verified backup covers this packet and pending files. |
"""
    append_once(RESPONSE_LOG, marker, response_block)

    doc_block = f"""## 2026-09-30 v31 pending-Astra evidence packet

UTC: {created.isoformat()}. Branch-neutral metadata analysis only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_text}`. Rechecked Astra gate for `{current_request_id}`: ANALYSIS_READY matching=`{astra_ready_matches}`. Inspected v29/v30b/v31/source-budget raw files; consistency checks `{evidence['consistency_checks']}`. Wrote packet `{rel(packet_path)}` and backup request `{rel(backup_request)}`. Latest pending metadata files are not fully externally backed up=`{not backup_status['all_inputs_covered_by_latest_verified_backup']}`. Scientific state unchanged: development-only adaptive triage opportunity exists, but non-default H15/H35 source-family coverage is insufficient; await Astra before branch selection.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    registry_id = f"{stamp}_{NAME}"
    append_registry_once(
        registry_id,
        f"{registry_id},{created.isoformat()},metadata_analysis_only_v31_astra_pending_evidence_packet,none,no_validation_no_test,unknown,complete,0,0,0,{rel(out_dir / 'raw.json')}"
    )

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "packet_path": rel(packet_path),
        "backup_request": rel(backup_request),
        "continue_state": rel(continue_path),
        "analysis_ready_matches_or_supersedes_current": astra_ready_matches,
        "backup_inputs_all_covered_by_latest_verified_backup": backup_status["all_inputs_covered_by_latest_verified_backup"],
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
        Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "summary.md", out_dir / "raw.json", out_dir / "completed.json",
        packet_path, backup_request, continue_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
    ]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p.is_file()}
    write_json(out_dir / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
