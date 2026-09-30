#!/usr/bin/env python3
"""Materialize the 04:36 supervisor backup proof and recheck Astra gate.

This is a reversible metadata-only preparation step.  It must not open
validation/test banks, run MPC simulations, refit selectors, or perform training.
It records the explicit supervisor/user-context backup proof, verifies the
current v31 handoff state, summarizes already-opened development evidence, and
writes a continuation state plus a backup request for the newly created files.
"""
from __future__ import annotations

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
NAME = "vehicle_true_variable_horizon_v31_0436_backup_astra_recheck_v0"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
LATEST = ASTRA_DIR / "LATEST.md"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
RUNS_DIR = ROOT / "research_artifacts/aws_diagnostics"
STATE_DIR = ROOT / "research_artifacts/aws_state"
RESEARCH_SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite"]

USER_CONTEXT_BACKUP = {
    "time": "2026-09-30T04:36:20.681764+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "a035b38e4a01581045c745ee3956f4c8c4e2414f",
    "changed_files": 15,
    "packages_this_run": [
        {
            "name": "20260930T043617_4efa00f9.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T043617_4efa00f9.tar.gz",
            "id": 600043849,
            "sha256": "59ebb7023f5ce9a1b4a3bcef37ba47af5c21e22aa1055af83226635fef8477b5",
            "bytes": 15769843,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 148918,
}

KNOWN_PRE_0436_ARTIFACTS = [
    # Scientific/development evidence carried into Astra handoff.
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/summary.md",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/completed.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/summary.md",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/completed.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json",
    # Handoff and coordination.
    "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json",
    "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
    "docs/bohn2021_takeover/astra_reviews/LATEST.md",
    "docs/bohn2021_takeover/astra_reviews/COORDINATION.md",
    # Operational materialization/recheck outputs before 04:36.
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0_20260930T042732Z/summary.md",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0_20260930T042732Z/raw.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0_20260930T042732Z/completed.json",
    "research_artifacts/aws_backup_proofs/backup_proof_20260930T042914_from_user_context_after_v31_0427_materialization.json",
    "research_artifacts/aws_state/continue_state_20260930_after_0429_backup_astra_absent_recheck_v2.md",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_0429_BACKUP_ASTRA_ABSENT_RECHECK_V2_20260930.json",
]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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


def path_info(rel_path: str) -> Dict[str, Any]:
    path = ROOT / rel_path
    if not path.exists():
        return {"path": rel_path, "exists": False, "bytes": None, "sha256": None, "mtime_utc": None}
    st = path.stat()
    return {
        "path": rel_path,
        "exists": True,
        "bytes": st.st_size,
        "sha256": sha256(path),
        "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
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


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {sec:.3f}s"


def sqlite_token_total() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "path": None,
        "table": None,
        "column": None,
        "call_count": None,
        "total_tokens": None,
        "error": None,
    }
    for candidate in RESEARCH_SQLITE_CANDIDATES:
        if not candidate.exists():
            continue
        out["path"] = str(candidate)
        try:
            conn = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [str(r[0]) for r in cur.fetchall()]
            for table in ([t for t in tables if t == "calls"] + [t for t in tables if t != "calls"]):
                q_table = '"' + table.replace('"', '""') + '"'
                cur.execute(f"PRAGMA table_info({q_table})")
                cols = [str(r[1]) for r in cur.fetchall()]
                token_cols = [c for c in cols if c == "total_tokens"] + [c for c in cols if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
                if not token_cols:
                    continue
                col = token_cols[0]
                q_col = '"' + col.replace('"', '""') + '"'
                cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {q_col} IS NULL THEN 0 ELSE {q_col} END) FROM {q_table}")
                count, total = cur.fetchone()
                conn.close()
                out.update({"available": True, "table": table, "column": col, "call_count": int(count or 0), "total_tokens": int(total or 0), "error": None})
                return out
            conn.close()
            out["error"] = "sqlite present but no total_tokens-like column found"
            return out
        except Exception as exc:  # pragma: no cover - diagnostic only
            out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return out
    out["error"] = "research.sqlite not found in checked locations"
    return out


def ready_matches_current(ready: Any, current_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not current_id:
        return False
    if ready.get("request_id") == current_id or ready.get("supersedes_request_id") == current_id:
        return True
    covered = ready.get("covers_request_ids") or ready.get("covered_request_ids")
    return isinstance(covered, list) and current_id in covered


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def compact_v31() -> Dict[str, Any]:
    raw = safe_json(RUNS_DIR / "vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json") or {}
    headline = raw.get("headline", {}) if isinstance(raw, Mapping) else {}
    family = raw.get("family_summary", {}) if isinstance(raw, Mapping) else {}
    return {
        "rows": headline.get("rows"),
        "families": headline.get("families"),
        "oracle_label_to_family_count": headline.get("oracle_label_to_family_count"),
        "in_sample_bad_two_feature_rule": headline.get("in_sample_bad_two_feature_rule"),
        "loo_bad_two_feature_rule": headline.get("loo_bad_two_feature_rule"),
        "logo_bad_two_feature_rule": headline.get("logo_bad_two_feature_rule"),
        "logo_bad_folds": headline.get("logo_bad_folds"),
        "saving_vs_fixed_H35_if_bad_zero": {
            "in_sample": headline.get("in_sample_saving_vs_fixed_H35"),
            "loo": headline.get("loo_saving_vs_fixed_H35"),
            "logo_nominal_not_valid_when_bad_nonzero": headline.get("logo_saving_vs_fixed_H35"),
        },
        "families_detail_keys": sorted((family.get("families") or {}).keys()) if isinstance(family, Mapping) else [],
    }


def compact_v30b() -> Dict[str, Any]:
    raw = safe_json(RUNS_DIR / "vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json") or {}
    headline = raw.get("headline", {}) if isinstance(raw, Mapping) else {}
    oracle = raw.get("oracle_fastest_safe_H12_H15_H35") or raw.get("oracle_metric") or raw.get("oracle") or {}
    # The summary is the authoritative concise source when schema varies.
    return {
        "headline_keys": sorted(headline.keys()) if isinstance(headline, Mapping) else [],
        "oracle_bad_count": (oracle.get("bad_count") if isinstance(oracle, Mapping) else None),
        "oracle_decision_saving_vs_fixed_H35": (oracle.get("decision_saving_vs_fixed_H35") if isinstance(oracle, Mapping) else None),
        "note": "Summary reports oracle bad=0 and 50.5688% decision-time saving vs fixed H35; simple two-threshold rule failed initial LOO with 1 bad row.",
    }


def compact_v29() -> Dict[str, Any]:
    raw = safe_json(RUNS_DIR / "vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json") or {}
    return {
        "budget_actual": raw.get("budget_actual") if isinstance(raw, Mapping) else None,
        "headline": raw.get("headline") if isinstance(raw, Mapping) else None,
        "note": "Summary reports 44 development episodes / 2557 control steps; H35 rescues both source242 H12/H15 both-fail rows; H12 fastest safe on 6 controls; H15 fastest on 3 v19 H12-risk rows.",
    }


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": created.isoformat(),
        "classification": "metadata_only_no_sim_no_validation_no_test_no_training",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    })

    backup_time = parse_time(USER_CONTEXT_BACKUP["time"])
    backup_valid = bool(
        USER_CONTEXT_BACKUP.get("status") == "verified"
        and USER_CONTEXT_BACKUP.get("remaining_changed_files") == 0
        and USER_CONTEXT_BACKUP.get("commit")
        and USER_CONTEXT_BACKUP.get("packages_this_run")
        and backup_time
    )
    covered_infos = [path_info(p) for p in KNOWN_PRE_0436_ARTIFACTS]
    missing = [x["path"] for x in covered_infos if not x["exists"]]
    covers_known = bool(backup_valid and not missing)

    proof_path = BACKUP_DIR / "backup_proof_20260930T043620_from_user_context_after_0429_state.json"
    proof = {
        "proof_file_created_by": "GPT-5.5 executor via metadata audit, from explicit user/supervisor context",
        "proof_file_created_utc": created.isoformat(),
        "backup": USER_CONTEXT_BACKUP,
        "interpretation": {
            "backup_valid_from_user_context": backup_valid,
            "covers_known_pre_0436_artifacts": covers_known,
            "missing_known_pre_0436_artifacts": missing,
            "new_files_after_this_backup_need_future_backup": True,
            "no_new_simulations": True,
            "no_new_control_steps": True,
            "no_new_training_or_gradient_steps": True,
            "no_selector_refits": True,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "known_pre_0436_artifacts": covered_infos,
    }
    write_json(proof_path, proof)

    next_review = safe_json(NEXT_REVIEW)
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    analysis_ready = safe_json(ANALYSIS_READY)
    astra_ready = ready_matches_current(analysis_ready, current_request_id)
    report_path = None
    if isinstance(analysis_ready, Mapping):
        report_path = analysis_ready.get("report") or analysis_ready.get("report_path")

    tokens = sqlite_token_total()
    elapsed_seconds = (created - FIRST_SUPERVISOR_EVENT).total_seconds()
    elapsed_text = fmt_elapsed(elapsed_seconds)
    backup_elapsed_text = fmt_elapsed((backup_time - FIRST_SUPERVISOR_EVENT).total_seconds()) if backup_time else "unknown"
    token_text = "unknown"
    if tokens.get("available"):
        total = int(tokens.get("total_tokens") or 0)
        token_text = f"{total:,} ({total / 1_000_000:.3f}M)"
    elif tokens.get("error"):
        token_text = f"unknown ({tokens.get('error')})"

    v31 = compact_v31()
    v30b = compact_v30b()
    v29 = compact_v29()

    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_0436_BACKUP_ASTRA_RECHECK_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v31_0436_backup_astra_recheck.md"
    must_cover = [
        rel(Path(__file__).resolve()),
        rel(out_dir / "run_started.json"), rel(out_dir / "raw.json"), rel(out_dir / "summary.md"), rel(out_dir / "completed.json"),
        rel(proof_path), rel(request_path), rel(continue_path),
        "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
        "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv",
    ]
    write_json(request_path, {
        "requested_utc": created.isoformat(),
        "reason": "Back up metadata-only 04:36 proof materialization/Astra recheck outputs before any unique simulation/refit/training/validation/final-test work.",
        "prior_backup": USER_CONTEXT_BACKUP,
        "must_cover": must_cover,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    if astra_ready:
        next_action = "Read the matching/superseding Astra report, verify cited evidence against raw artifacts, log dispositions in RESPONSE_LOG.md, then execute its plan after the post-04:36 metadata outputs are externally backed up if the plan needs unique science."
    else:
        next_action = "Astra analysis remains pending for v31. Do not select a fresh scientific branch. Continue only reversible integrity/preparation work until a matching/superseding ANALYSIS_READY.json arrives and post-04:36 metadata outputs are backed up."

    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "elapsed_since_first_supervisor_event_text": elapsed_text,
        "backup_elapsed_since_first_supervisor_event_text": backup_elapsed_text,
        "classification": "metadata_only_backup_materialization_astra_gate_and_evidence_recheck_no_sim_no_validation_no_test_no_training",
        "backup_valid": backup_valid,
        "covers_known_pre_0436_artifacts": covers_known,
        "backup_proof": rel(proof_path),
        "backup_request": rel(request_path),
        "astra_gate": {
            "next_review_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": ANALYSIS_READY.exists(),
            "analysis_ready_matches_or_supersedes_current": astra_ready,
            "analysis_ready_report": report_path,
            "latest_md_exists": LATEST.exists(),
        },
        "evidence_recheck": {"v29": v29, "v30b": v30b, "v31": v31},
        "server_api_token_total_from_research_sqlite": tokens,
        "budgets_actual": {
            "new_simulation_episodes": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "next_action": next_action,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(out_dir / "raw.json", raw)

    summary = f"""# v31 04:36 backup/Astra gate recheck

UTC: `{created.isoformat()}`. Reversible metadata-only step; no simulations, no control steps, no selector refits, no training, no validation64 bank open, no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}` at this audit; `{backup_elapsed_text}` at the verified 04:36 backup.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## Backup status
- Materialized user-context proof: `{rel(proof_path)}`.
- User-context backup valid: `{backup_valid}`; covers known pre-04:36 artifacts: `{covers_known}`; missing expected pre-04:36 artifacts: `{len(missing)}`.
- Backup commit: `{USER_CONTEXT_BACKUP['commit']}`; package SHA256: `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}`.
- New metadata outputs from this run are not covered by that backup; request written: `{rel(request_path)}`.

## Astra status
- Current NEXT_REVIEW_REQUEST id: `{current_request_id}`.
- ANALYSIS_READY present: `{ANALYSIS_READY.exists()}`; matches/supersedes current: `{astra_ready}`; report: `{report_path}`.

## Evidence recheck carried forward
- v29: development-only longer-H feasibility over 11 opened states / 44 episodes / 2557 control steps; H35 rescues both H12/H15 both-fail source242 rows; H12 fastest safe on 6 safe-control rows; H15 fastest on 3 v19 H12-risk rows.
- v30b: analysis-only three-way audit; oracle H12/H15/H35 triage on opened rows has bad=0 and 50.5688% decision-time saving vs fixed H35, but the simple deployable feature rule was not valid for validation because initial LOO had a bad row.
- v31: analysis-only cluster diagnostic; 11 rows collapse to 5 source-family groups with oracle-label family counts `{v31.get('oracle_label_to_family_count')}`; row-level two-feature LOO bad=`{v31.get('loo_bad_two_feature_rule')}`, leave-one-source-family-out bad=`{v31.get('logo_bad_two_feature_rule')}` because H15/H35 regimes each have only one independent opened family.

## Next executor action
{next_action}
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.parent.mkdir(parents=True, exist_ok=True)
    continue_path.write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    response_block = f"""## Operational follow-up v31 04:36 backup/Astra gate recheck

Updated by GPT-5.5 executor at `{created.isoformat()}`. Metadata-only; no simulations, no control steps, no training/refit, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; pre-04:36 state is externally backed up, this run now needs follow-up backup | Supervisor context backup `{USER_CONTEXT_BACKUP['time']}`, remaining_changed_files=0, commit `{USER_CONTEXT_BACKUP['commit']}`, package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}`; materialized proof `{rel(proof_path)}`; covers known pre-04:36 artifacts=`{covers_known}`. | Request `{rel(request_path)}` must be satisfied before unique simulation/refit/training/validation/final-test work. |
| Astra role-correction handoff | accepted; still pending unless ANALYSIS_READY appears after this audit | NEXT_REVIEW_REQUEST id `{current_request_id}`; ANALYSIS_READY exists=`{ANALYSIS_READY.exists()}` and matches/supersedes current=`{astra_ready}`. | Read and implement a matching/superseding Astra report when available; otherwise keep only reversible preparation. |
"""
    append_once(RESPONSE_LOG, marker, response_block)

    doc_block = f"""## 2026-09-30 v31 04:36 backup/Astra gate recheck

UTC: {created.isoformat()}. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`; server API total_tokens `{token_text}`. User-context backup `{USER_CONTEXT_BACKUP['time']}` / commit `{USER_CONTEXT_BACKUP['commit']}` / package SHA256 `{USER_CONTEXT_BACKUP['packages_this_run'][0]['sha256']}` is materialized at `{rel(proof_path)}` and covers known pre-04:36 artifacts=`{covers_known}`. Current Astra request `{current_request_id}`; ANALYSIS_READY present/matching=`{astra_ready}`. New outputs from this audit require backup request `{rel(request_path)}` before unique science. Evidence status remains: v29/v30b/v31 are development/opened-row diagnostics only; v31 shows source-family coverage insufficiency, not a deployable selector or reproduction claim. Next action: {next_action}
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "backup_valid": backup_valid,
        "covers_known_pre_0436_artifacts": covers_known,
        "backup_proof": rel(proof_path),
        "backup_request": rel(request_path),
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "continue_state": rel(continue_path),
        "astra_gate": raw["astra_gate"],
        "server_api_token_total_from_research_sqlite": tokens,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_action": next_action,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)

    hash_paths = [Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "raw.json", out_dir / "summary.md", out_dir / "completed.json", proof_path, request_path, continue_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md"]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p.is_file()}
    write_json(out_dir / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
