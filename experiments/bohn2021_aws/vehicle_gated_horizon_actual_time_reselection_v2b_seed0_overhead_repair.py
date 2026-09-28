#!/usr/bin/env python3
"""Repair diagnostic for actual-time-aware gated-horizon re-selection V2.

The first V2 metadata run used ``overhead_by_seed.get(seed or -1)``.  That is
wrong for seed 0 because ``0 or -1`` selects -1, so seed0 adaptive candidates
were scored with zero selection-overhead fraction.  This script is a
metadata-only V2b repair: it reuses the frozen V2 inputs and rule, applies
selection overhead only to adaptive candidates with an explicit seed key, writes
new artifacts, and marks the previous V2 candidate timing fields for seed0 as
superseded.

No simulations, no training, no historical validation64 bank reopen, and no
sealed final-test access.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2.py"
PREVIOUS_V2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/raw.json"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z.md"
BACKUP = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_ACTUAL_TIME_RESELECTION_V2B_SEED0_OVERHEAD_REPAIR_20260928T0915Z.json"
DOCS = [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md"]


def load_base_module():
    spec = importlib.util.spec_from_file_location("vehicle_actual_time_reselection_v2_base", str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import base V2 script")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + body.strip() + "\n", encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def str_table(rows: List[List[Any]]) -> str:
    return "\n".join("|" + "|".join(str(x) for x in r) + "|" for r in rows)


def evaluate_corrected(base: Any, candidates: List[Dict[str, Any]], timing_by_h: Dict[int, Dict[str, Any]], overhead_by_seed: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    RULE = base.RULE
    evaluated: List[Dict[str, Any]] = []
    for c in candidates:
        e = dict(c)
        seed = base.inum(e.get("seed"))
        short_h = base.inum(e.get("short_h"))
        is_adaptive_candidate = e.get("policy_kind") != "fixed_H25_baseline" and short_h is not None and short_h < 25
        reasons: List[str] = []
        if not is_adaptive_candidate:
            reasons.append("not_adaptive_short_h_candidate")
        if not base.bval(e.get("fully_evaluated_all_24_cases")):
            reasons.append("not_fully_evaluated_all_24_training_cases")
        if base.bval(e.get("historically_rejected_or_pruned")):
            reasons.append("historically_rejected_or_pruned")
        if base.bval(e.get("worse_success_than_fixed")):
            reasons.append("worse_success_than_fixed")
        if base.bval(e.get("worse_constraint_than_fixed")):
            reasons.append("worse_constraint_than_fixed")
        if base.bval(e.get("worse_initial_solver_fail_than_fixed")):
            reasons.append("worse_initial_solver_fail_than_fixed")
        if base.bval(e.get("worse_final_solver_fail_than_fixed")):
            reasons.append("worse_final_solver_fail_than_fixed")

        tr = timing_by_h.get(short_h if short_h is not None else -1, {})
        n = int(tr.get("n") or 0)
        med = base.fnum(tr.get("median"))
        if n < RULE["min_timing_support_n"]:
            reasons.append("timing_support_n_lt_%d" % RULE["min_timing_support_n"])
        if med is None:
            reasons.append("missing_short_H_timing_ratio")

        short_frac = base.fnum(e.get("short_step_fraction")) or 0.0
        # Corrected bookkeeping: seed 0 is a valid key, and fixed/non-adaptive
        # fallback candidates do not pay adaptive selector overhead.
        if is_adaptive_candidate and seed is not None:
            overhead = (overhead_by_seed.get(seed) or {}).get("selection_overhead_fraction", 0.0) or 0.0
        else:
            overhead = 0.0
        gross = short_frac * max(0.0, 1.0 - (med if med is not None else 1.0)) if is_adaptive_candidate else 0.0
        net = gross - overhead if is_adaptive_candidate else 0.0

        e["timing_support_n"] = n
        e["timing_ratio_median_vs_H25"] = med
        e["selection_overhead_fraction"] = overhead
        e["estimated_gross_time_saving_fraction"] = gross
        e["estimated_net_time_saving_fraction"] = net
        if base.fnum(e.get("mean_physical_delta_pct_vs_fixed")) is None or base.fnum(e.get("mean_physical_delta_pct_vs_fixed")) > RULE["mean_physical_delta_pct_max"]:
            reasons.append("mean_physical_delta_pct_gt_0p05")
        if base.fnum(e.get("max_pair_physical_regression_pct")) is None or base.fnum(e.get("max_pair_physical_regression_pct")) > RULE["max_pair_physical_regression_pct_max"]:
            reasons.append("max_pair_physical_regression_pct_gt_1")
        if base.fnum(e.get("positive_tail_cvar80_pct")) is None or base.fnum(e.get("positive_tail_cvar80_pct")) > RULE["positive_tail_cvar80_pct_max"]:
            reasons.append("positive_tail_cvar80_pct_gt_0p5")
        if short_frac < RULE["short_step_fraction_min"]:
            reasons.append("short_step_fraction_lt_0p03")
        if (base.fnum(e.get("episodes_with_short_estimated")) or 0.0) < RULE["episodes_with_short_min"]:
            reasons.append("episodes_with_short_lt_8")
        top2 = base.fnum(e.get("top2_short_step_share"))
        if top2 is None or top2 > RULE["top2_short_step_share_max"]:
            reasons.append("top2_short_step_share_gt_0p50")
        switch_rate = base.fnum(e.get("switch_rate"))
        if switch_rate is None or switch_rate > RULE["switch_rate_max"]:
            reasons.append("switch_rate_gt_0p08")
        if net < RULE["net_time_saving_fraction_min"]:
            reasons.append("estimated_net_time_saving_fraction_lt_0p005")
        e["actual_time_primary_eligible"] = len(reasons) == 0
        e["actual_time_ineligibility_reasons"] = reasons
        evaluated.append(e)

    nominations: Dict[str, Any] = {}
    for seed in (0, 1, 2):
        rows = [e for e in evaluated if base.inum(e.get("seed")) == seed]
        eligible = [e for e in rows if e.get("actual_time_primary_eligible")]
        eligible.sort(key=lambda e: (
            max(0.0, base.fnum(e.get("max_pair_physical_regression_pct")) or 0.0),
            max(0.0, base.fnum(e.get("positive_tail_cvar80_pct")) or 0.0),
            max(0.0, base.fnum(e.get("mean_physical_delta_pct_vs_fixed")) or 0.0),
            -(base.fnum(e.get("estimated_net_time_saving_fraction")) or -999.0),
            base.fnum(e.get("top2_short_step_share")) if base.fnum(e.get("top2_short_step_share")) is not None else 999.0,
            base.fnum(e.get("switch_rate")) if base.fnum(e.get("switch_rate")) is not None else 999.0,
            str(e.get("candidate_id")),
        ))
        fixed = next((e for e in rows if e.get("candidate_id") == "fixed"), None)
        current = next((e for e in rows if e.get("stored_current_policy")), None)
        nearest = sorted([e for e in rows if e.get("policy_kind") != "fixed_H25_baseline"], key=lambda e: (
            len(e.get("actual_time_ineligibility_reasons") or []),
            max(0.0, RULE["net_time_saving_fraction_min"] - (base.fnum(e.get("estimated_net_time_saving_fraction")) or -999.0)),
            max(0.0, (base.fnum(e.get("max_pair_physical_regression_pct")) or 999.0) - RULE["max_pair_physical_regression_pct_max"]),
            str(e.get("candidate_id")),
        ))[:8]
        nominations[str(seed)] = {
            "eligible_count": len(eligible),
            "nominated": base.compact(eligible[0]) if eligible else base.compact(fixed) if fixed else None,
            "fallback_to_fixed": len(eligible) == 0,
            "stored_current": base.compact(current) if current else None,
            "nearest_nonfixed": [base.compact(x) for x in nearest],
        }
    return evaluated, nominations


def main() -> int:
    if OUT.exists():
        raise RuntimeError("Output directory already exists; refusing to overwrite: %s" % OUT)
    base = load_base_module()
    candidates = base.load_candidates()
    v2_diag = base.read_json(base.V2_RAW)
    timing_by_h, overhead_by_seed = base.build_timing_model(v2_diag)
    evaluated, nominations = evaluate_corrected(base, candidates, timing_by_h, overhead_by_seed)
    previous = base.read_json(PREVIOUS_V2_RAW) if PREVIOUS_V2_RAW.exists() else {}
    prev_noms = previous.get("nominations") or {}
    prev_ids = {str(s): ((prev_noms.get(str(s)) or {}).get("nominated") or {}).get("candidate_id") for s in (0, 1, 2)}
    new_ids = {str(s): ((nominations.get(str(s)) or {}).get("nominated") or {}).get("candidate_id") for s in (0, 1, 2)}
    prev_acceptance = bool(previous.get("acceptance_for_smoke_met")) if previous else None
    adaptive_nom_count = sum(1 for v in nominations.values() if not v.get("fallback_to_fixed"))
    acceptance_met = adaptive_nom_count >= 2
    next_action = "after_backup_freeze_and_run_small_actual_time_reselection_v2b_smoke" if acceptance_met else "after_backup_design_broader_training_representation_or_scenario_opportunity_diagnostic"
    changes = {
        "seed0_overhead_fraction_applied_now": (overhead_by_seed.get(0) or {}).get("selection_overhead_fraction"),
        "previous_v2_acceptance_for_smoke_met": prev_acceptance,
        "corrected_acceptance_for_smoke_met": acceptance_met,
        "acceptance_changed": (prev_acceptance is not None and bool(prev_acceptance) != bool(acceptance_met)),
        "previous_v2_nomination_ids": prev_ids,
        "corrected_nomination_ids": new_ids,
        "nomination_ids_changed": prev_ids != new_ids,
    }

    OUT.mkdir(parents=True, exist_ok=False)
    fields = ["seed", "candidate_id", "stored_current_policy", "short_h", "profile", "guard", "mean_physical_delta_pct_vs_fixed", "max_pair_physical_regression_pct", "positive_tail_cvar80_pct", "short_step_fraction", "episodes_with_short_estimated", "top2_short_step_share", "switch_rate", "timing_ratio_median_vs_H25", "timing_support_n", "estimated_gross_time_saving_fraction", "selection_overhead_fraction", "estimated_net_time_saving_fraction", "actual_time_primary_eligible", "actual_time_ineligibility_reasons"]
    with (OUT / "candidate_timing_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in sorted(evaluated, key=lambda x: (base.inum(x.get("seed")) or -1, str(x.get("candidate_id")))):
            row = {k: e.get(k) for k in fields}
            row["actual_time_ineligibility_reasons"] = ";".join(row.get("actual_time_ineligibility_reasons") or [])
            w.writerow(row)
    raw = {
        "created_utc": now(),
        "method": "IMPROVED_vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_metadata_only",
        "classification": "finite candidate search/reselection; zero gradient training; not ORIGINAL SAC",
        "repair_reason": "V2 scored seed0 adaptive candidates with zero overhead due to seed or -1 bookkeeping; V2b uses explicit seed key and no overhead for fixed fallback.",
        "access_flags": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_accessed": False, "historical_validation64_bank_opened": False},
        "frozen_rule": base.RULE,
        "timing_model_from_v2_matched_fixed_grid": timing_by_h,
        "selection_overhead_model_from_v2_risk_traces": overhead_by_seed,
        "nominations": nominations,
        "adaptive_nominated_seed_count": adaptive_nom_count,
        "acceptance_for_smoke_met": acceptance_met,
        "next_action": next_action,
        "changes_vs_previous_v2": changes,
        "evaluated_candidates": evaluated,
        "input_hashes": {rel(p): sha256(p) for p in [BASE_SCRIPT, PREVIOUS_V2_RAW, base.V1_RAW, base.V1_CSV, base.V2_RAW, base.AMENDMENT, Path(__file__).resolve()]},
    }
    write_json(OUT / "raw.json", raw)
    write_json(OUT / "nominated_policies.json", {"nominations": nominations, "acceptance_for_smoke_met": acceptance_met, "next_action": next_action, "changes_vs_previous_v2": changes})
    lines = [
        "# Vehicle gated-horizon actual-time-aware re-selection V2b seed0-overhead repair",
        "", "UTC: `%s`." % now(), "",
        "Metadata-only repair. No simulations, no training, no historical validation64 bank reopen, and no sealed-test access occurred.",
        "", "## Repair", "",
        "- Previous V2 seed0 adaptive candidate timing fields are superseded because seed0 selector overhead was accidentally looked up with key -1.",
        "- V2b applies seed0 overhead fraction `%s` to seed0 adaptive candidates and keeps fixed fallback overhead at zero." % changes["seed0_overhead_fraction_applied_now"],
        "", "## Corrected nominations", "",
        str_table([[" seed ", " nominated ", " adaptive? ", " eligible_count ", " net_time_saving ", " physical gates ", " current "]] + [[s, (v.get("nominated") or {}).get("candidate_id"), not v.get("fallback_to_fixed"), v.get("eligible_count"), (v.get("nominated") or {}).get("estimated_net_time_saving_fraction"), {k: (v.get("nominated") or {}).get(k) for k in ["mean_physical_delta_pct_vs_fixed", "max_pair_physical_regression_pct", "positive_tail_cvar80_pct"]}, (v.get("stored_current") or {}).get("candidate_id")] for s, v in sorted(nominations.items())]),
        "", "## Change relative to V2", "", "`%s`" % changes,
        "", "## Decision", "",
        "- Corrected adaptive nominated seed count: `%d` / 3." % adaptive_nom_count,
        "- Corrected acceptance for smoke met: `%s`." % acceptance_met,
        "- Next action after backup: `%s`." % next_action,
        "", "This remains metadata/model-selection evidence only, not validation or final-test evidence.", "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    completed = {"created_utc": now(), "passed": True, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "candidate_timing_metrics_csv": rel(OUT / "candidate_timing_metrics.csv"), "nominated_policies": rel(OUT / "nominated_policies.json"), "adaptive_nominated_seed_count": adaptive_nom_count, "acceptance_for_smoke_met": acceptance_met, "changes_vs_previous_v2": changes, "next_action": next_action, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollouts": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(BACKUP)}
    write_json(OUT / "completed.json", completed)
    STATE.write_text("# Actual-time-aware re-selection V2b seed0-overhead repair state\n\nUTC: %s\n\n- Completed metadata-only repair of V2 seed0 overhead bookkeeping if this file was produced by a run.\n- Corrected nominations: %s\n- acceptance_for_smoke_met=%s; next_action=%s\n- No simulations/training/test/validation64 bank reopen.\n" % (now(), new_ids, acceptance_met, next_action), encoding="utf-8")
    backup = {"created_utc": now(), "reason": "Backup required after V2b seed0-overhead repair before any further simulations.", "backup_required_before_more_simulations": True, "artifacts_requiring_backup": [rel(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "candidate_timing_metrics.csv", OUT / "nominated_policies.json", OUT / "completed.json", STATE, BACKUP, Path(__file__).resolve()]], "sha256": {rel(p): sha256(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "candidate_timing_metrics.csv", OUT / "nominated_policies.json", OUT / "completed.json", STATE, Path(__file__).resolve()]}, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollouts": 0}
    write_json(BACKUP, backup)
    marker = "<!-- vehicle-gated-horizon-actual-time-reselection-v2b-seed0-overhead-repair-20260928 -->"
    block = "%s\n## 2026-09-28 vehicle actual-time-aware re-selection V2b seed0-overhead repair\n\nUTC: %s. Metadata-only repair of V2 seed0 selection-overhead bookkeeping; no simulations, no training, no validation64 bank reopen, and no sealed-test access. Corrected nominations: %s; adaptive_nominated_seed_count=%d/3; acceptance_for_smoke_met=%s; changes_vs_V2=%s. Next action after backup: `%s`.\n" % (marker, now(), new_ids, adaptive_nom_count, acceptance_met, changes, next_action)
    for d in DOCS:
        append_once(d, marker, block)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
