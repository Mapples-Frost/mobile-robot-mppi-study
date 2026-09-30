#!/usr/bin/env python3
"""Materialize the 2026-09-30T04:59 supervisor backup and recheck Astra gate.

Operational/integrity action only.  It verifies the latest supervisor-provided
external backup metadata against the prior metadata-run backup request, checks
current Astra handoff state, and preserves a continuation record.  It must not
run MPC, refit/train a selector/value model, open validation64 bank content, or
access sealed tests.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v31_0459_backup_astra_gate_recheck_v0"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_DIR = ROOT / "research_artifacts/aws_state"
REQUEST_TO_SATISFY = BACKUP_DIR / "REQUEST_BACKUP_AFTER_V31_0453_BACKUP_ASTRA_GATE_RECHECK_20260930T045656Z.json"
EXPECTED_REQUEST_ID = "v31-source-coverage-budget-bounds-20260930T044819Z"
SOURCE_BUDGET_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/raw.json"
SOURCE_BUDGET_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/summary.md"
BRANCH_NEUTRAL_NOTE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_pre_astra_branch_neutral_diagnostic_20260930T0458Z.md"
LATEST_OLD_REPORT = "docs/bohn2021_takeover/astra_reviews/20260929T153837Z.md"
SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite"]

# Explicit supervisor/user-context backup from the prompt that launched this cycle.
USER_CONTEXT_BACKUP = {
    "time": "2026-09-30T04:59:24.496081+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "928f24eed5ef783c119add827079555636850f6d",
    "changed_files": 24,
    "packages_this_run": [
        {
            "name": "20260930T045921_ec0476c2.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T045921_ec0476c2.tar.gz",
            "id": 600079261,
            "sha256": "5d012cdb94ee946e8321123406c224a2d58ca6d92a0020c2bf8973c2f21a2e9f",
            "bytes": 15995046,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 148984,
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: str) -> dt.datetime:
    out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
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


def source_budget_digest() -> Dict[str, Any]:
    raw = read_json(SOURCE_BUDGET_RAW) or {}
    if not isinstance(raw, Mapping):
        return {"exists": SOURCE_BUDGET_RAW.exists(), "error": "raw not JSON object"}
    bounds = raw.get("budget_bounds_if_astra_selects_fresh_source_label_acquisition", {})
    basic = bounds.get("basic_logo_class_presence", {}) if isinstance(bounds, Mapping) else {}
    three = bounds.get("three_source_stability_target", {}) if isinstance(bounds, Mapping) else {}
    return {
        "exists": True,
        "raw_path": rel(SOURCE_BUDGET_RAW),
        "summary_path": rel(SOURCE_BUDGET_SUMMARY),
        "raw_sha256": sha256(SOURCE_BUDGET_RAW),
        "summary_sha256": sha256(SOURCE_BUDGET_SUMMARY),
        "counts": raw.get("current_oracle_label_independent_source_family_counts"),
        "basic_additional": basic.get("additional_independent_families_by_label") if isinstance(basic, Mapping) else None,
        "basic_rollout_bounds": basic.get("rollout_budget_lower_bounds") if isinstance(basic, Mapping) else None,
        "three_source_additional": three.get("additional_independent_families_by_label") if isinstance(three, Mapping) else None,
        "three_source_rollout_bounds": three.get("rollout_budget_lower_bounds") if isinstance(three, Mapping) else None,
        "not_branch_decision": raw.get("not_a_branch_decision"),
        "not_validation_or_test_evidence": raw.get("not_validation_or_test_evidence"),
    }


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
        "classification": "metadata_only_backup_astra_gate_recheck_no_sim_no_validation_no_test_no_training_no_refit",
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(out_dir / "run_started.json", started)

    backup_time = parse_time(USER_CONTEXT_BACKUP["time"])
    request = read_json(REQUEST_TO_SATISFY) or {}
    must_cover = request.get("must_cover", []) if isinstance(request, Mapping) else []
    must_cover = [str(x) for x in must_cover]
    extra_known = [rel(BRANCH_NEUTRAL_NOTE)] if BRANCH_NEUTRAL_NOTE.exists() else []
    all_checked = list(dict.fromkeys(must_cover + extra_known))
    infos = [file_info(p) for p in all_checked]
    missing = [x["path"] for x in infos if not x.get("exists")]
    after_backup = []
    for x in infos:
        mt = x.get("mtime_utc")
        if not mt:
            continue
        try:
            if parse_time(mt) > backup_time + dt.timedelta(seconds=2):
                after_backup.append({"path": x["path"], "mtime_utc": mt})
        except Exception:
            pass

    package = USER_CONTEXT_BACKUP["packages_this_run"][0]
    backup_claim_valid = bool(
        USER_CONTEXT_BACKUP.get("status") == "verified"
        and USER_CONTEXT_BACKUP.get("remaining_changed_files") == 0
        and USER_CONTEXT_BACKUP.get("commit")
        and USER_CONTEXT_BACKUP.get("packages_this_run")
    )
    package_ok = bool(package.get("sha256") and package.get("verification") == "github_server_sha256" and package.get("bytes", 0) > 0)
    request_files_present = not missing
    temporal_consistent = not after_backup
    clears_prior_backup_gate = bool(backup_claim_valid and package_ok and request_files_present and temporal_consistent)

    next_review = read_json(NEXT_REVIEW) or {}
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    ready = read_json(ANALYSIS_READY)
    astra_ready_exists = ANALYSIS_READY.exists()
    astra_ready_matches = ready_matches(ready, current_request_id)
    report = None
    if isinstance(ready, Mapping):
        report = ready.get("report") or ready.get("report_path")

    digest = source_budget_digest()
    tokens = token_total()
    elapsed_text = fmt_elapsed((created - FIRST_SUPERVISOR_EVENT).total_seconds())
    token_text = (
        f"{int(tokens.get('total_tokens') or 0):,} ({int(tokens.get('total_tokens') or 0) / 1_000_000:.3f}M)"
        if tokens.get("available")
        else f"unknown ({tokens.get('error')})"
    )

    proof_path = BACKUP_DIR / "backup_proof_20260930T045924_from_user_context_after_v31_0453_gate_recheck.json"
    proof = {
        "proof_file_created_utc": created.isoformat(),
        "proof_file_created_by": "GPT-5.5 executor from explicit user/supervisor context",
        "backup": USER_CONTEXT_BACKUP,
        "request_evaluated": rel(REQUEST_TO_SATISFY),
        "request_must_cover_count": len(must_cover),
        "extra_known_prebackup_files_checked": extra_known,
        "checked_file_count": len(all_checked),
        "request_files_present": request_files_present,
        "missing_request_files": missing,
        "backup_claim_valid": backup_claim_valid,
        "package_verification_claim_ok": package_ok,
        "temporal_consistency_with_local_mtime": temporal_consistent,
        "files_with_mtime_after_backup_time": after_backup,
        "clears_prior_backup_gate_for_unique_science_by_this_local_check": clears_prior_backup_gate,
        "covered_file_infos": infos,
        "interpretation": "This materializes the user/supervisor verified GitHub release backup claim from the current prompt. It only clears the prior 04:56 metadata-run backup gate if all requested/known files are present and locally older than the claimed backup time.",
        "new_files_from_this_recheck_need_future_backup": True,
        "budgets_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
    }
    write_json(proof_path, proof)

    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_0459_BACKUP_ASTRA_GATE_RECHECK_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v31_0459_backup_astra_gate_recheck.md"
    planned_cover = [
        rel(Path(__file__).resolve()),
        rel(out_dir / "run_started.json"),
        rel(out_dir / "summary.md"),
        rel(out_dir / "raw.json"),
        rel(out_dir / "completed.json"),
        rel(proof_path),
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
        "reason": "Back up v31 04:59 backup-proof materialization/Astra gate recheck outputs before unique simulation/refit/training/validation/final-test work.",
        "must_cover": planned_cover,
        "prior_backup_gate_cleared_by_local_check": clears_prior_backup_gate,
        "prior_backup_temporal_mtime_mismatch_count": len(after_backup),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    if astra_ready_matches:
        next_action = "Read the matching/superseding Astra report, verify cited evidence, log dispositions, then implement its selected plan only after this recheck output is backed up."
    else:
        next_action = "Astra analysis remains pending for the current v31 source-budget request; do not choose acquisition/refit/training/scenario branch. Continue only reversible integrity/preparation while waiting."
    if not clears_prior_backup_gate:
        next_action += " The 04:59 backup materialization did not independently clear prior backup coverage; require a later explicit backup proof/confirmation before unique science."

    raw = {
        "created_utc": created.isoformat(),
        "classification": started["classification"],
        "elapsed_since_first_supervisor_event_text": elapsed_text,
        "server_api_token_total_from_research_sqlite": tokens,
        "backup_gate": {
            "user_context_backup": USER_CONTEXT_BACKUP,
            "materialized_proof": rel(proof_path),
            "request_evaluated": rel(REQUEST_TO_SATISFY),
            "extra_known_prebackup_files_checked": extra_known,
            "backup_claim_valid": backup_claim_valid,
            "package_verification_claim_ok": package_ok,
            "request_files_present": request_files_present,
            "missing_request_files": missing,
            "temporal_consistency_with_local_mtime": temporal_consistent,
            "files_with_mtime_after_backup_time": after_backup,
            "clears_prior_backup_gate_for_unique_science_by_this_local_check": clears_prior_backup_gate,
            "new_backup_request": rel(backup_request),
        },
        "astra_gate": {
            "next_review_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": astra_ready_exists,
            "analysis_ready_matches_or_supersedes_current": astra_ready_matches,
            "analysis_ready_report": report,
            "latest_known_old_review": LATEST_OLD_REPORT,
        },
        "source_budget_digest": digest,
        "budgets_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "next_action": next_action,
        "python": sys.version,
    }
    write_json(out_dir / "raw.json", raw)

    summary = f"""# v31 04:59 backup/Astra gate recheck

UTC: `{created.isoformat()}`. Metadata-only; no simulations, no control steps, no selector refit, no training, no validation64 bank access and no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## Backup gate check
- Materialized user-context backup proof: `{rel(proof_path)}`.
- Supervisor/user context backup claim: status `{USER_CONTEXT_BACKUP['status']}`, remaining_changed_files `{USER_CONTEXT_BACKUP['remaining_changed_files']}`, commit `{USER_CONTEXT_BACKUP['commit']}`, package SHA256 `{package['sha256']}`.
- Evaluated request: `{rel(REQUEST_TO_SATISFY)}`; extra known pre-backup file check: `{extra_known}`.
- Requested/known files present locally: `{request_files_present}` ({len(missing)} missing).
- Temporal consistency with local mtimes: `{temporal_consistent}` ({len(after_backup)} checked files had mtime after the claimed backup time).
- Prior 04:56 metadata backup gate cleared by this local check: `{clears_prior_backup_gate}`.
- New metadata outputs from this recheck require follow-up backup: `{rel(backup_request)}`.

## Astra gate check
- Current NEXT_REVIEW_REQUEST id: `{current_request_id}`.
- ANALYSIS_READY present: `{astra_ready_exists}`; matches/supersedes current: `{astra_ready_matches}`; report: `{report}`.

## Evidence state preserved
- Source-budget diagnostic counts remain `{digest.get('counts')}`.
- Basic lower-bound additional families: `{digest.get('basic_additional')}`.
- Three-source lower-bound additional families: `{digest.get('three_source_additional')}`.
- This remains development/opened-row analysis only, not branch selection, not selector validation, and not final-test evidence.

## Next action
{next_action}
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    log_block = f"""## v31 04:59 backup/Astra gate recheck

Updated by GPT-5.5 executor at `{created.isoformat()}`. Metadata-only; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; latest supervisor backup claim materialized and checked against the prior 04:56 metadata-run request | Proof `{rel(proof_path)}` records commit `{USER_CONTEXT_BACKUP['commit']}`, package SHA256 `{package['sha256']}`, request/known files present=`{request_files_present}`, temporal consistency=`{temporal_consistent}`, prior gate cleared by local check=`{clears_prior_backup_gate}`. | New request `{rel(backup_request)}` covers this metadata run. Do not run unique simulation/refit/training/validation/final-test work unless latest backup gate and Astra gate are explicitly clear. |
| Astra v31 direction request | pending | NEXT_REVIEW_REQUEST id `{current_request_id}`; ANALYSIS_READY exists=`{astra_ready_exists}`, matches/supersedes current=`{astra_ready_matches}`. | Await/read matching Astra report before choosing acquisition/refit/training/scenario branch. |
"""
    append_once(RESPONSE_LOG, marker, log_block)

    doc_block = f"""## 2026-09-30 v31 04:59 backup/Astra gate recheck

UTC: {created.isoformat()}. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_text}`. Materialized supervisor backup claim `{USER_CONTEXT_BACKUP['time']}` at `{rel(proof_path)}` for request `{rel(REQUEST_TO_SATISFY)}` plus branch-neutral note check; requested/known files present=`{request_files_present}`, temporal consistency with local mtimes=`{temporal_consistent}`, prior 04:56 metadata backup gate cleared by local check=`{clears_prior_backup_gate}`. Current Astra request `{current_request_id}`; ANALYSIS_READY matching=`{astra_ready_matches}`. New backup request: `{rel(backup_request)}`. Evidence status unchanged: v31 source-budget result is analysis-only and awaits Astra branch selection.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    registry_id = f"{stamp}_{NAME}"
    append_registry_once(
        registry_id,
        f"{registry_id},{created.isoformat()},metadata_only_v31_backup_astra_gate_recheck_no_science,none,no_validation_no_test,unknown,complete,0,0,0,{rel(out_dir / 'raw.json')}"
    )

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_proof": rel(proof_path),
        "backup_request": rel(backup_request),
        "continue_state": rel(continue_path),
        "prior_backup_gate_cleared_by_local_check": clears_prior_backup_gate,
        "temporal_mtime_mismatch_count": len(after_backup),
        "analysis_ready_matches_or_supersedes_current": astra_ready_matches,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)
    hash_paths = [Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "summary.md", out_dir / "raw.json", out_dir / "completed.json", proof_path, backup_request, continue_path, RESPONSE_LOG, ROOT / "EXPERIMENT_REGISTRY.csv"]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p.is_file()}
    write_json(out_dir / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
