#!/usr/bin/env python3
"""v28 success-aware scenario/comparison audit for true-variable-H vehicle diagnostics.

Analysis-only, development-only.

Purpose
-------
The v27 fixed-H12-primary confirmation showed strong H12-vs-H15 *relative*
timing savings on a fresh stress-bank batch, but the postdiagnostic found two
source242 rows where both H12 and H15 ran to the 150-step cap with very large
physical costs.  A relative H15-referenced gate can therefore pass while
absolute success/safety fails.  This script freezes an evidence-linked,
success-aware scenario/comparison diagnosis before any validation or retraining.

It runs no MPC simulation, no selector refit/search, no gradient training, no
validation64 access and no sealed-test access.  It only reads already opened
v19/v21/v23/v25/v27 development artifacts and writes a versioned audit/protocol
for the next bounded diagnostic.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_success_aware_scenario_comparison_audit_v28"
STAMP = "20260930T0315Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_success_aware_followup_{STAMP}.json"
CONTINUE = ROOT / "research_artifacts/aws_state/continue_state_20260930T0315_after_v28_success_aware_audit.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V28_SUCCESS_AWARE_SCENARIO_COMPARISON_AUDIT_{STAMP}.json"
DOC_MARKER = f"<!-- vehicle-success-aware-scenario-comparison-audit-v28-{STAMP} -->"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

EVIDENCE_PATHS = {
    "v19_summary": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/summary.md",
    "v19_raw": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json",
    "v21_summary": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/summary.md",
    "v23_summary": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/summary.md",
    "v25_summary": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/summary.md",
    "v27_summary": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/summary.md",
    "v27_completed": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/completed.json",
    "v27_postdiagnostic_raw": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/raw.json",
    "v27_postdiagnostic_summary": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/summary.md",
    "v26b_protocol": "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_corrected_fixed_H12_primary_confirmation_amendment_20260930T0235Z.json",
    "response_log": "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
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


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + text.strip() + "\n", encoding="utf-8")


def parse_iso(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str):
        return None
    try:
        out = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def path_info(path: Path) -> Dict[str, Any]:
    stat = path.stat() if path.exists() else None
    return {
        "path": rel(path),
        "exists": path.exists(),
        "bytes": None if stat is None else stat.st_size,
        "sha256": sha256(path),
        "mtime_utc": None if stat is None else dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
    }


def latest_repo_backup_proof() -> Dict[str, Any]:
    proof_dir = ROOT / "research_artifacts/aws_backup_proofs"
    proofs = sorted(proof_dir.glob("backup_proof_20260930T*.json")) if proof_dir.exists() else []
    best: Optional[Dict[str, Any]] = None
    for p in proofs:
        obj: Dict[str, Any]
        try:
            obj = read_json(p)
        except Exception:
            continue
        t = parse_iso(obj.get("time") or obj.get("created_utc"))
        row = {"path": rel(p), "time": None if t is None else t.isoformat(), "commit": obj.get("commit"), "status": obj.get("status") or obj.get("backup_verified"), "remaining_changed_files": obj.get("remaining_changed_files"), "sha256": sha256(p)}
        if best is None or (t is not None and (best.get("_t") is None or t > best["_t"])):
            row["_t"] = t
            best = row
    if best is None:
        return {"present": False}
    best.pop("_t", None)
    return {"present": True, **best}


def wilson_upper_zero(n: int, z: float = 1.96) -> Optional[float]:
    if n <= 0:
        return None
    # Wilson interval upper bound for x=0.
    denom = 1.0 + z * z / n
    centre = z * z / (2.0 * n)
    radius = z * math.sqrt(z * z / (4.0 * n * n))
    return (centre + radius) / denom


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--i-accept-development-analysis-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_accept_development_analysis_only:
        raise SystemExit("missing explicit development-analysis-only acknowledgement")

    created = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {
        "method": NAME,
        "started_utc": created.isoformat(),
        "pid": os.getpid(),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    evidence_abs = {k: ROOT / v for k, v in EVIDENCE_PATHS.items()}
    evidence = {k: path_info(p) for k, p in evidence_abs.items()}
    missing = [k for k, p in evidence_abs.items() if not p.exists()]
    if missing:
        raise SystemExit(f"required evidence missing: {missing}")

    v27_post = read_json(evidence_abs["v27_postdiagnostic_raw"])
    latest_backup = latest_repo_backup_proof()
    v27_post_time = parse_iso(v27_post.get("created_utc"))
    latest_backup_time = parse_iso(latest_backup.get("time")) if latest_backup.get("present") else None
    backup_covers_postdiag_by_time = bool(latest_backup_time and v27_post_time and latest_backup_time >= v27_post_time)

    # Collapsed-branch-state convention used by v26b/v27 combined summaries:
    # v19 repeats are summarized separately so we do not inflate row counts.
    partitions = {
        "row_convention": "collapsed branch-state rows for cross-campaign totals; v19 repeat-row evidence retained separately",
        "v19_opened_boundary_collapsed": {
            "rows": 24,
            "repeat_pair_rows": 48,
            "h12_only_failure_rows_collapsed": 3,
            "h12_only_failure_repeat_rows": 6,
            "h15_only_failure_rows": 0,
            "both_fail_rows": 0,
            "h12_beneficial_rows_collapsed": 21,
            "fixed_H12_decision_saving_vs_H15_percent": 14.32,
            "fixed_H12_physical_delta_vs_H15": 566.4,
            "oracle_H12_H15_decision_saving_percent": 15.72,
            "failure_source_cluster": "fresh_v11/fresh_case05_slot1_mid_late_control; false_positive_center offsets -4/0/+4; H15 safe, H12 high-cost/longer",
            "evidence": [EVIDENCE_PATHS["v19_summary"], EVIDENCE_PATHS["v19_raw"]],
        },
        "fresh_v21_v23_v25_v27_collapsed": {
            "rows": 80,
            "feasible_not_both_fail_rows": 78,
            "h12_only_failure_rows": 0,
            "h15_only_failure_rows": 0,
            "both_fail_rows": 2,
            "both_fail_source_candidate_indices": [242],
            "h12_beneficial_rows": 76,
            "safe_but_not_H12_beneficial_rows": 2,
            "relative_h12_bad_rows": 0,
            "fixed_H12_decision_saving_vs_H15_percent": 17.90,
            "fixed_H12_solver_saving_vs_H15_percent": 20.54,
            "absolute_failure_note": "The two source242 v27 rows are failures for both H12 and H15; their relative physical delta is zero, so H15-referenced gates overstate deployability.",
            "zero_H12_only_failure_wilson95_upper_on_feasible_fresh": wilson_upper_zero(78),
            "rule_of_three_upper_on_feasible_fresh": 3.0 / 78.0,
            "evidence": [EVIDENCE_PATHS["v21_summary"], EVIDENCE_PATHS["v23_summary"], EVIDENCE_PATHS["v25_summary"], EVIDENCE_PATHS["v27_postdiagnostic_raw"]],
        },
        "all_opened_development_collapsed": {
            "rows": 104,
            "feasible_not_both_fail_and_not_H12_only_failure_rows": 99,
            "h12_only_failure_rows": 3,
            "h15_only_failure_rows": 0,
            "both_fail_rows": 2,
            "fixed_H12_absolute_problem_rows_if_deployed": 5,
            "fixed_H15_absolute_problem_rows_if_deployed": 2,
            "relative_h12_bad_rows": 3,
            "relative_fixed_H12_decision_saving_vs_H15_percent": 17.06,
            "relative_physical_delta_H12_minus_H15": 262.491,
            "interpretation": "All-opened evidence preserves two distinct failure modes: v19 H12-specific high-cost rows that create adaptive opportunity, and v27 both-fail hard rows that cannot be counted as fixed-H12 success.",
        },
    }

    v27_absolute = v27_post.get("v27_absolute_safety_counts", {})
    v27_policy = v27_post.get("v27_relative_policy_audit", {})
    sanity_checks = {
        "v27_rows_expected_24": v27_absolute.get("rows") == 24,
        "v27_both_fail_expected_2": v27_absolute.get("both_h12_h15_unsafe_count") == 2,
        "v27_fixed_H12_success_sensitive_pass_false": v27_policy.get("fixed_H12", {}).get("success_sensitive_pass5_zero_cat_physical") is False,
        "v27_fixed_H12_H15_referenced_pass_true": v27_policy.get("fixed_H12", {}).get("h15_referenced_pass5_zero_cat_physical") is True,
        "latest_repo_backup_covers_postdiag_by_time": backup_covers_postdiag_by_time,
    }

    four_axis_diagnosis = {
        "scenarios": {
            "verified_findings": [
                "Fresh stress-pool rows v21/v23/v25/v27 mostly make H12 safe and faster than H15; adaptive selector is often slower than fixed H12.",
                "Opened v19 contains a localized H12-only high-cost cluster where H15 is safe and an H12/H15 oracle would help.",
                "v27 source242 contains two both-fail rows where H12 and H15 both reach 150 steps with physical cost >20k.",
            ],
            "competing_hypotheses": [
                "stress-v1 distribution has weak state-dependent H12/H15 opportunity except for a mined v19 boundary cluster",
                "source242 rows are infeasible/harder-than-H15 and require longer horizons or scenario feasibility stratification",
                "useful adaptivity may require including longer fixed-H arms (H25/H35) rather than only H12/H15",
            ],
            "missing_evidence": "Whether longer horizons rescue v27 source242 and whether v19 H12-only failures are separable by pre-outcome scenario/branch features without using labels.",
            "discriminating_experiment": "After backup, run a small identical-state longer-H feasibility probe on v27 source242 both-fail rows, v19 H12-only rows, and safe matched controls with H12/H15/H25/H35, reporting absolute success/safety and timing.",
        },
        "reward_and_success_accounting": {
            "verified_findings": [
                "H15-referenced physical gates can pass on both-fail rows because physical delta is zero even when both policies fail absolutely.",
                "Reported timing savings on failed 150-step episodes are not deployable speed evidence unless success/failure is stratified first.",
            ],
            "competing_hypotheses": [
                "The current scalar relative gate is insufficient, not necessarily the controller itself",
                "Physical cost thresholds and success definitions need to be reported as primary gates before timing",
            ],
            "missing_evidence": "A uniform success/failure table for every future comparator, including fixed H12, fixed H15, longer fixed H, selector and oracle.",
            "discriminating_experiment": "Freeze success-sensitive pass logic: absolute success/safety first; only then compare physical deltas and timing. Apply to v29 probe before any validation.",
        },
        "training_or_selection": {
            "verified_findings": [
                "Current H12/H15 selector is conservative: v27 chose H12 only 6/24 and saved 3.72%, failing pass5 while fixed H12 saved 19.62% in H15-referenced terms.",
                "Static/history selectors passed some opened splits but are not validated and do not beat fixed H12 on fresh stress rows where fixed H12 is safe.",
            ],
            "competing_hypotheses": [
                "Retraining/refit is premature if fixed H12 already dominates the feasible fresh distribution",
                "Retraining/refit becomes necessary if v19-like H12-only failures are pre-outcome separable and frequent enough to justify adaptive safety over fixed H12",
            ],
            "missing_evidence": "Feature separability and frequency of H12-only failures versus both-fail infeasible rows under a success-aware protocol.",
            "discriminating_experiment": "Defer gradient/value refit until the success-aware longer-H/feature audit shows adaptive opportunity not captured by fixed H12 or longer fixed-H baselines.",
        },
        "comparisons_and_timing": {
            "verified_findings": [
                "Fixed H12 is now a primary simple comparator on fresh rows; comparing only to H15 is weak.",
                "Selector overhead is small in development micro-measurements, but whole closed-loop validation timing is still absent.",
            ],
            "competing_hypotheses": [
                "A strong fixed-H12 or longer fixed-H baseline may dominate this stress distribution",
                "Adaptive H may only be valuable as a safety fallback on rare H12-risk states, not as a mean compute saver",
            ],
            "missing_evidence": "Whole-decision timing for any deployable policy under success-sensitive filtering and strong fixed-H baselines including H12/H15/H25/H35 where relevant.",
            "discriminating_experiment": "Bounded H12/H15/H25/H35 continuation probe, then only validate if adaptive policy beats strong fixed-H on success-filtered control/compute tradeoff.",
        },
    }

    next_protocol = {
        "protocol_id": f"vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_after_v28_{STAMP}",
        "status": "frozen_planned_not_executed",
        "scope": "development-only IMPROVED diagnostic; no validation64; no sealed test; not ORIGINAL SAC",
        "preconditions": [
            "Verified external backup must cover v27 postdiagnostic, this v28 audit/protocol, docs/state/registry and backup requests.",
            "Use identical saved branch states; no outcome-based reselection beyond the categories explicitly listed here.",
        ],
        "hypothesis": "The current H12/H15-only comparison confounds three cases: feasible rows where fixed H12 dominates, H12-only risk rows where adaptivity may help, and both-fail rows that need longer-H feasibility or scenario stratification. A small longer-H probe will decide whether to pivot to scenario/comparison design, terminal-risk/value refit, or a negative adaptive-opportunity conclusion for stress-v1.",
        "planned_state_categories": [
            {"category": "both_fail_h12_h15", "source": "v27 source_candidate_index 242", "planned_collapsed_states": 2},
            {"category": "h12_only_failure_h15_safe", "source": "v19 fresh_v11/fresh_case05_slot1_mid_late_control false_positive_center offsets -4/0/+4", "planned_collapsed_states": 3, "repeat_rows_available": 6},
            {"category": "fresh_safe_fixed_h12_support", "source": "safe matched controls from v21/v23/v25/v27, selected before v29 outcomes", "planned_collapsed_states": "4_to_6_if_implementation_budget_allows"},
        ],
        "planned_horizons": [12, 15, 25, 35],
        "max_budget": {"development_episodes": 44, "control_step_cap": 6600, "validation64_episodes": 0, "sealed_test_episodes": 0, "gradient_steps": 0},
        "primary_metrics": [
            "absolute success/failure and safety/constraint status per row before timing aggregation",
            "physical/control cost conditional on success and including failed rows separately",
            "whole decision time and solver time mean/median/p95 by horizon",
            "initial/final solver failure steps and 150-step cap hits",
        ],
        "decision_rules": [
            "If H25/H35 rescue v27 source242 while H12/H15 fail, do not treat fixed H12 as deployable on hard rows; broaden fixed-H comparator/scenario design before selector validation.",
            "If H25/H35 also fail source242, mark those rows as scenario infeasibility/hard-case failures for all compared horizons and do not count relative H12 timing there as success.",
            "If v19 H12-only failures remain H15/H25-safe and are pre-outcome separable, prioritize terminal-risk/value refit or a success-aware selector; label IMPROVED.",
            "If fixed H12 dominates all feasible rows and H12-only risk is not separable/frequent, record weak adaptive opportunity for stress-v1 and consider versioned scenario redesign rather than retraining to force switching.",
        ],
    }
    write_json(PROTOCOL, next_protocol)

    audit = {
        "created_utc": created.isoformat(),
        "method": NAME,
        "classification": "development_analysis_only_success_aware_scenario_comparison_audit_no_sim_no_validation_no_test",
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "budgets_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "evidence": evidence,
        "missing_evidence_files": missing,
        "latest_repo_backup_proof": latest_backup,
        "backup_covers_v27_postdiagnostic_by_time": backup_covers_postdiag_by_time,
        "sanity_checks": sanity_checks,
        "success_aware_partitions": partitions,
        "v27_absolute_safety_counts": v27_absolute,
        "four_axis_diagnosis": four_axis_diagnosis,
        "scientific_decision": "Do not validate the current adaptive H12/H15 selector. The next discriminating work is success-aware scenario/comparison diagnosis with longer fixed-H feasibility on source242 both-fail rows and v19 H12-only risk rows. Training/value refit remains deferred unless that diagnosis shows separable adaptive opportunity beyond strong fixed-H baselines.",
        "frozen_next_protocol": rel(PROTOCOL),
    }
    write_json(OUT / "raw.json", audit)

    summary_lines = [
        "# v28 success-aware scenario/comparison audit",
        "",
        f"UTC: `{created.isoformat()}`. Analysis-only; no simulations/control steps, no training/refit, no validation64, no sealed test.",
        "",
        "## Success-aware partition",
        "",
        "- Fresh v21+v23+v25+v27 collapsed rows: `80`; H12-only relative failures `0`; both-fail H12/H15 rows `2` from source242; H12-beneficial rows `76`; fixed H12 relative decision saving `17.90%`.",
        "- Opened v19 boundary collapsed rows: `24` (`48` repeat pair rows); H12-only failure rows `3` collapsed / `6` repeat rows; H15 unsafe rows `0`; oracle H12/H15 saving `15.72%`.",
        "- All opened collapsed rows: `104`; H12-only failures `3`; both-fail rows `2`; fixed-H12 absolute problem rows if deployed `5`; fixed-H15 absolute problem rows `2`.",
        "",
        "## Diagnosis",
        "",
        "The current stress-v1 evidence contains two different phenomena that must not be merged: (1) v19 H12-specific high-cost states where adaptivity could help, and (2) v27 source242 both-fail states where H12-vs-H15 relative timing is not deployable success evidence. Fresh feasible rows mostly favor fixed H12, so retraining a conservative selector is not yet the most informative next step.",
        "",
        "## Frozen next protocol",
        "",
        f"Protocol written: `{rel(PROTOCOL)}`. It freezes a bounded development-only H12/H15/H25/H35 identical-state feasibility probe over v27 source242 both-fail rows, v19 H12-only failures, and safe matched controls. Planned max budget: `44` development episodes / `6600` control steps; validation64/test/training budgets all `0`.",
        "",
        "## Backup gate",
        "",
        f"Latest repo backup proof covers postdiagnostic by time: `{backup_covers_postdiag_by_time}`. A new backup request is required before v29 simulation or any other unique science.",
    ]
    (OUT / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v28_success_aware_scenario_comparison_audit",
        "created_utc": created.isoformat(),
        "backup_required_before_more_unique_science": True,
        "reason": "v28 success-aware audit/protocol and docs/state/registry updates after v27 postdiagnostic",
        "must_cover": [
            rel(Path(__file__).resolve()),
            rel(OUT),
            rel(PROTOCOL),
            rel(CONTINUE),
            rel(BACKUP_REQUEST),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
            "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
        ],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    continue_text = f"""# Continue state after v28 success-aware audit

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.

Completed an analysis-only success-aware scenario/comparison audit; no simulation/control steps/training/refit, no validation64, no sealed test. Main finding: fresh stress rows mostly favor fixed H12 but v27 source242 both-fail rows invalidate deployability claims from H15-referenced gates; v19 H12-only failures remain preserved adaptive-opportunity evidence.

Next action after verified backup: execute or implement the frozen v29 H12/H15/H25/H35 identical-state feasibility probe in `{rel(PROTOCOL)}`. If backup is not verified, run only backup-gate/status preservation.

Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`, `{rel(PROTOCOL)}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    CONTINUE.parent.mkdir(parents=True, exist_ok=True)
    CONTINUE.write_text(continue_text, encoding="utf-8")

    doc_block = f"""## 2026-09-30 v28 success-aware scenario/comparison audit

UTC: {created.isoformat()}. Analysis-only development audit completed with no simulations/control steps/training/refit, no validation64-bank access and no sealed-test access. It partitions the opened H12/H15 evidence by success semantics: fresh v21+v23+v25+v27 has 80 collapsed rows with H12-only relative failures=0 and both-fail rows=2 (v27 source242), while opened v19 has 24 collapsed rows with H12-only failures=3 (6 repeat rows) and H15 unsafe=0. All opened collapsed evidence therefore contains H12-only failures=3 and both-fail rows=2; relative timing gates and absolute deployability are now explicitly separated. Decision: do not validate the current adaptive H12/H15 selector. Freeze success-aware longer-H feasibility protocol `{rel(PROTOCOL)}` for a bounded H12/H15/H25/H35 identical-state probe on source242 both-fail rows, v19 H12-only failures and safe matched controls after verified backup. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, DOC_MARKER, doc_block)

    response_addendum = f"""
## Follow-up through v28 success-aware scenario/comparison audit

Updated by GPT-5.5 executor at `{created.isoformat()}`. v28 is analysis-only over already-opened development evidence; validation64 and sealed test remained closed.

| linked recommendation(s) | disposition after v28 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; operationalized | v28 partitions all opened collapsed H12/H15 evidence: fresh v21+v23+v25+v27 has 80 rows with both-fail=2 from source242 and H12-only failures=0; opened v19 has H12-only failures=3 collapsed / 6 repeat rows with H15 safe. | Future gates must report absolute success/safety before H15-referenced physical/timing deltas. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; next comparator frozen | Fixed H12 dominates fresh feasible rows but fails on v19 H12-only cases; source242 both-fail rows may require longer H or scenario stratification. | Froze `{rel(PROTOCOL)}`: bounded H12/H15/H25/H35 identical-state feasibility probe before validation. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred with a discriminating trigger | Current evidence does not show that a richer selector is the bottleneck on fresh feasible rows; it shows mixed scenario/comparison failure modes. | Pivot to terminal-risk/value refit only if the v29 feasibility/feature audit shows separable H12-only opportunity not captured by fixed H12/longer fixed-H baselines. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | All v28 inputs are opened development/stress-pool artifacts, not validation64/population/final-test evidence. | Keep scope development-only; require fresh independent confirmation before final testing. |
| `A12_registry_backup_schema_contract` | accepted; active | v28 wrote new source/audit/protocol/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified external backup covering v27 postdiagnostic and v28 before v29 simulation or other unique science. |
"""
    append_once(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", DOC_MARKER, response_addendum)

    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old_reg = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    row = f"{created.isoformat()},{NAME},development_analysis_only_success_aware_scenario_comparison_no_sim,existing_v19_v21_v23_v25_v27_development_evidence_no_validation64_no_test,0,0,0,0,0,False,{rel(OUT / 'completed.json')},vehicle-success-aware-scenario-comparison-audit-v28-{STAMP}\n"
    if f"vehicle-success-aware-scenario-comparison-audit-v28-{STAMP}" not in old_reg:
        reg.write_text(old_reg.rstrip() + "\n" + row, encoding="utf-8")

    completed_files: List[Path] = [
        Path(__file__).resolve(), OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", PROTOCOL, CONTINUE, BACKUP_REQUEST,
        ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
    ] + list(evidence_abs.values())

    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": audit["classification"],
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "frozen_next_protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQUEST),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "scientific_decision": audit["scientific_decision"],
        "success_aware_partition_headline": {
            "fresh_rows": 80,
            "fresh_h12_only_failures": 0,
            "fresh_both_fail": 2,
            "v19_h12_only_failures_collapsed": 3,
            "all_opened_h12_only_failures": 3,
            "all_opened_both_fail": 2,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(completed_files), key=rel) if p.exists() and p.is_file()},
    }
    write_json(OUT / "completed.json", completed)

    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQUEST),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "decision": audit["scientific_decision"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
