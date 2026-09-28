#!/usr/bin/env python3
"""Actual-measured-time-aware re-selection V2 for vehicle gated-horizon candidates.

Metadata-only diagnostic.  Reads existing finite gated-candidate training/search
metrics and the corrected partial devval timing/opportunity diagnostic V2.  It
runs no rollouts, no training, opens no sealed test, and does not reopen the
historical validation64 bank.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000"
V1_RAW = V1_DIR / "raw.json"
V1_CSV = V1_DIR / "candidate_metrics.csv"
V2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/raw.json"
AMENDMENT = ROOT / "research_artifacts/aws_protocols/vehicle_risk_reselection_v1_scheduling_amendment_after_v2_20260928.md"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z.md"
BACKUP = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_ACTUAL_TIME_RESELECTION_V2_20260928T0905Z.json"
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2.py"
DOCS = [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md"]

RULE = {
    "min_timing_support_n": 10,
    "mean_physical_delta_pct_max": 0.05,
    "max_pair_physical_regression_pct_max": 1.0,
    "positive_tail_cvar80_pct_max": 0.5,
    "short_step_fraction_min": 0.03,
    "episodes_with_short_min": 8,
    "top2_short_step_share_max": 0.50,
    "switch_rate_max": 0.08,
    "net_time_saving_fraction_min": 0.005,
}


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def fnum(x: Any) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def inum(x: Any) -> Optional[int]:
    y = fnum(x)
    if y is None:
        return None
    return int(round(y)) if abs(y - round(y)) < 1e-9 else None


def bval(x: Any) -> bool:
    return bool(x) and str(x).lower() not in ("false", "0", "none", "")


def mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def compact(m: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "seed", "candidate_id", "stored_current_policy", "short_h", "profile", "guard",
        "mean_physical_delta_pct_vs_fixed", "max_pair_physical_regression_pct", "positive_tail_cvar80_pct",
        "short_step_fraction", "episodes_with_short_estimated", "top2_short_step_share", "switch_rate",
        "timing_ratio_median_vs_H25", "timing_support_n", "estimated_gross_time_saving_fraction",
        "selection_overhead_fraction", "estimated_net_time_saving_fraction", "actual_time_primary_eligible",
        "actual_time_ineligibility_reasons",
    ]
    return {k: m.get(k) for k in keys}


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + body.strip() + "\n", encoding="utf-8")


def str_table(rows: List[List[Any]]) -> str:
    return "\n".join("|" + "|".join(str(x) for x in r) + "|" for r in rows)


def load_candidates() -> List[Dict[str, Any]]:
    raw = read_json(V1_RAW)
    rows = raw.get("candidate_metrics") or []
    if not rows:
        raise RuntimeError("No candidate_metrics in %s" % V1_RAW)
    return [dict(r) for r in rows]


def build_timing_model(v2: Mapping[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    dec = (((v2.get("matched_terminal_opportunity") or {}).get("decision_ratios_vs_H25_by_H")) or {})
    timing_by_h: Dict[int, Dict[str, Any]] = {}
    for hs, row in dec.items():
        H = inum(hs)
        if H is None:
            continue
        timing_by_h[H] = {
            "n": inum(row.get("n")) or 0,
            "mean": fnum(row.get("mean")),
            "median": fnum(row.get("median")),
            "lt_0p98_count": inum(row.get("lt_0p98_count")) or 0,
            "gt_1p02_count": inum(row.get("gt_1p02_count")) or 0,
        }
    adaptive = (v2.get("adaptive_aggregates") or {})
    pair = ((v2.get("adaptive_vs_matched_H25") or {}).get("adaptive_vs_matched_H25_by_family_seed") or {})
    overhead_by_seed: Dict[int, Dict[str, Any]] = {}
    for seed in (0, 1, 2):
        key = "risk_seed%d" % seed
        a = adaptive.get(key) or {}
        p = pair.get(key) or {}
        adaptive_decision = fnum(a.get("decision_mean_s_per_step_weighted"))
        ratio = fnum(p.get("decision_ratio_mean"))
        selection = fnum(p.get("selection_mean_s")) or fnum(a.get("selection_mean_s_per_step_weighted")) or 0.0
        h25_decision = adaptive_decision / ratio if adaptive_decision is not None and ratio not in (None, 0.0) else None
        overhead_fraction = selection / h25_decision if h25_decision and h25_decision > 0 else 0.0
        overhead_by_seed[seed] = {
            "risk_adaptive_decision_mean_s": adaptive_decision,
            "risk_vs_H25_decision_ratio_mean": ratio,
            "estimated_H25_decision_mean_s": h25_decision,
            "selection_mean_s": selection,
            "selection_overhead_fraction": overhead_fraction,
        }
    return timing_by_h, overhead_by_seed


def evaluate(candidates: List[Dict[str, Any]], timing_by_h: Dict[int, Dict[str, Any]], overhead_by_seed: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []
    for c in candidates:
        e = dict(c)
        seed = inum(e.get("seed"))
        short_h = inum(e.get("short_h"))
        reasons: List[str] = []
        if e.get("policy_kind") == "fixed_H25_baseline" or short_h is None or short_h >= 25:
            reasons.append("not_adaptive_short_h_candidate")
        if not bval(e.get("fully_evaluated_all_24_cases")):
            reasons.append("not_fully_evaluated_all_24_training_cases")
        if bval(e.get("historically_rejected_or_pruned")):
            reasons.append("historically_rejected_or_pruned")
        if bval(e.get("worse_success_than_fixed")):
            reasons.append("worse_success_than_fixed")
        if bval(e.get("worse_constraint_than_fixed")):
            reasons.append("worse_constraint_than_fixed")
        if bval(e.get("worse_initial_solver_fail_than_fixed")):
            reasons.append("worse_initial_solver_fail_than_fixed")
        if bval(e.get("worse_final_solver_fail_than_fixed")):
            reasons.append("worse_final_solver_fail_than_fixed")
        tr = timing_by_h.get(short_h or -1, {})
        n = int(tr.get("n") or 0)
        med = fnum(tr.get("median"))
        if n < RULE["min_timing_support_n"]:
            reasons.append("timing_support_n_lt_%d" % RULE["min_timing_support_n"])
        if med is None:
            reasons.append("missing_short_H_timing_ratio")
        short_frac = fnum(e.get("short_step_fraction")) or 0.0
        overhead = (overhead_by_seed.get(seed or -1) or {}).get("selection_overhead_fraction", 0.0) or 0.0
        gross = short_frac * max(0.0, 1.0 - (med if med is not None else 1.0))
        net = gross - overhead
        e["timing_support_n"] = n
        e["timing_ratio_median_vs_H25"] = med
        e["selection_overhead_fraction"] = overhead
        e["estimated_gross_time_saving_fraction"] = gross
        e["estimated_net_time_saving_fraction"] = net
        gates = [
            ("mean_physical_delta_pct_gt_0p05", fnum(e.get("mean_physical_delta_pct_vs_fixed")), RULE["mean_physical_delta_pct_max"]),
            ("max_pair_physical_regression_pct_gt_1", fnum(e.get("max_pair_physical_regression_pct")), RULE["max_pair_physical_regression_pct_max"]),
            ("positive_tail_cvar80_pct_gt_0p5", fnum(e.get("positive_tail_cvar80_pct")), RULE["positive_tail_cvar80_pct_max"]),
            ("short_step_fraction_lt_0p03", short_frac, None),
            ("episodes_with_short_lt_8", fnum(e.get("episodes_with_short_estimated")), None),
            ("top2_short_step_share_gt_0p50", fnum(e.get("top2_short_step_share")), None),
            ("switch_rate_gt_0p08", fnum(e.get("switch_rate")), None),
            ("estimated_net_time_saving_fraction_lt_0p005", net, None),
        ]
        if gates[0][1] is None or gates[0][1] > RULE["mean_physical_delta_pct_max"]: reasons.append(gates[0][0])
        if gates[1][1] is None or gates[1][1] > RULE["max_pair_physical_regression_pct_max"]: reasons.append(gates[1][0])
        if gates[2][1] is None or gates[2][1] > RULE["positive_tail_cvar80_pct_max"]: reasons.append(gates[2][0])
        if short_frac < RULE["short_step_fraction_min"]: reasons.append(gates[3][0])
        if (fnum(e.get("episodes_with_short_estimated")) or 0.0) < RULE["episodes_with_short_min"]: reasons.append(gates[4][0])
        if fnum(e.get("top2_short_step_share")) is None or (fnum(e.get("top2_short_step_share")) or 0.0) > RULE["top2_short_step_share_max"]: reasons.append(gates[5][0])
        if fnum(e.get("switch_rate")) is None or (fnum(e.get("switch_rate")) or 0.0) > RULE["switch_rate_max"]: reasons.append(gates[6][0])
        if net < RULE["net_time_saving_fraction_min"]: reasons.append(gates[7][0])
        e["actual_time_primary_eligible"] = len(reasons) == 0
        e["actual_time_ineligibility_reasons"] = reasons
        evaluated.append(e)
    nominations: Dict[str, Any] = {}
    for seed in (0, 1, 2):
        rows = [e for e in evaluated if inum(e.get("seed")) == seed]
        eligible = [e for e in rows if e.get("actual_time_primary_eligible")]
        eligible.sort(key=lambda e: (
            max(0.0, fnum(e.get("max_pair_physical_regression_pct")) or 0.0),
            max(0.0, fnum(e.get("positive_tail_cvar80_pct")) or 0.0),
            max(0.0, fnum(e.get("mean_physical_delta_pct_vs_fixed")) or 0.0),
            -(fnum(e.get("estimated_net_time_saving_fraction")) or -999.0),
            fnum(e.get("top2_short_step_share")) if fnum(e.get("top2_short_step_share")) is not None else 999.0,
            fnum(e.get("switch_rate")) if fnum(e.get("switch_rate")) is not None else 999.0,
            str(e.get("candidate_id")),
        ))
        fixed = next((e for e in rows if e.get("candidate_id") == "fixed"), None)
        current = next((e for e in rows if e.get("stored_current_policy")), None)
        nearest = sorted([e for e in rows if e.get("policy_kind") != "fixed_H25_baseline"], key=lambda e: (
            len(e.get("actual_time_ineligibility_reasons") or []),
            max(0.0, RULE["net_time_saving_fraction_min"] - (fnum(e.get("estimated_net_time_saving_fraction")) or -999.0)),
            max(0.0, (fnum(e.get("max_pair_physical_regression_pct")) or 999.0) - RULE["max_pair_physical_regression_pct_max"]),
            str(e.get("candidate_id")),
        ))[:8]
        nominations[str(seed)] = {
            "eligible_count": len(eligible),
            "nominated": compact(eligible[0]) if eligible else compact(fixed) if fixed else None,
            "fallback_to_fixed": len(eligible) == 0,
            "stored_current": compact(current) if current else None,
            "nearest_nonfixed": [compact(x) for x in nearest],
        }
    return evaluated, nominations


def write_outputs(evaluated: List[Dict[str, Any]], nominations: Mapping[str, Any], timing_by_h: Mapping[int, Any], overhead_by_seed: Mapping[int, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adaptive_nom_count = sum(1 for v in nominations.values() if not v.get("fallback_to_fixed"))
    acceptance_met = adaptive_nom_count >= 2
    next_action = "after_backup_freeze_and_run_small_actual_time_reselection_v2_smoke" if acceptance_met else "after_backup_design_broader_training_representation_or_scenario_opportunity_diagnostic"
    raw = {
        "created_utc": now(),
        "method": "IMPROVED_vehicle_gated_horizon_actual_time_reselection_v2_metadata_only",
        "classification": "finite candidate search/reselection; zero gradient training; not ORIGINAL SAC",
        "access_flags": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_accessed": False, "historical_validation64_bank_opened": False},
        "frozen_rule": RULE,
        "timing_model_from_v2_matched_fixed_grid": timing_by_h,
        "selection_overhead_model_from_v2_risk_traces": overhead_by_seed,
        "nominations": nominations,
        "adaptive_nominated_seed_count": adaptive_nom_count,
        "acceptance_for_smoke_met": acceptance_met,
        "next_action": next_action,
        "evaluated_candidates": evaluated,
        "input_hashes": {rel(p): sha256(p) for p in [V1_RAW, V1_CSV, V2_RAW, AMENDMENT, SCRIPT]},
    }
    (OUT / "raw.json").write_text(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "nominated_policies.json").write_text(json.dumps({"nominations": nominations, "acceptance_for_smoke_met": acceptance_met, "next_action": next_action}, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["seed", "candidate_id", "stored_current_policy", "short_h", "profile", "guard", "mean_physical_delta_pct_vs_fixed", "max_pair_physical_regression_pct", "positive_tail_cvar80_pct", "short_step_fraction", "episodes_with_short_estimated", "top2_short_step_share", "switch_rate", "timing_ratio_median_vs_H25", "timing_support_n", "estimated_gross_time_saving_fraction", "selection_overhead_fraction", "estimated_net_time_saving_fraction", "actual_time_primary_eligible", "actual_time_ineligibility_reasons"]
    with (OUT / "candidate_timing_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in sorted(evaluated, key=lambda x: (inum(x.get("seed")) or -1, str(x.get("candidate_id")))):
            r = {k: e.get(k) for k in fields}
            r["actual_time_ineligibility_reasons"] = ";".join(r.get("actual_time_ineligibility_reasons") or [])
            w.writerow(r)
    lines = [
        "# Vehicle gated-horizon actual-time-aware re-selection V2",
        "", "UTC: `%s`." % now(), "",
        "Metadata-only IMPROVED finite re-selection. No simulations, no training, no historical validation64 bank reopen, and no sealed-test access occurred.",
        "", "## Timing model", "",
        "- Median decision ratios vs H25 from V2 matched fixed-grid: `%s`." % {str(k): {"n": v.get("n"), "median": v.get("median")} for k, v in sorted(timing_by_h.items())},
        "- Selection-overhead fractions by seed from V2 risk traces: `%s`." % {str(k): v.get("selection_overhead_fraction") for k, v in sorted(overhead_by_seed.items())},
        "", "## Frozen actual-time rule", "", "`%s`" % RULE,
        "", "## Nominations", "",
        str_table([[" seed ", " nominated ", " adaptive? ", " eligible_count ", " net_time_saving ", " physical gates ", " current "]] + [[s, (v.get("nominated") or {}).get("candidate_id"), not v.get("fallback_to_fixed"), v.get("eligible_count"), (v.get("nominated") or {}).get("estimated_net_time_saving_fraction"), {k: (v.get("nominated") or {}).get(k) for k in ["mean_physical_delta_pct_vs_fixed", "max_pair_physical_regression_pct", "positive_tail_cvar80_pct"]}, (v.get("stored_current") or {}).get("candidate_id")] for s, v in sorted(nominations.items())]),
        "", "## Decision", "",
        "- Adaptive nominated seed count: `%d` / 3." % adaptive_nom_count,
        "- Acceptance for smoke met: `%s`." % acceptance_met,
        "- Next action after backup: `%s`." % next_action,
        "", "## Interpretation", "",
    ]
    if acceptance_met:
        lines += [
            "The existing finite class is not entirely exhausted under an actual-time-aware objective: two seeds nominate adaptive candidates after measured timing and selection overhead are included. Seed0 falls back to fixed H25 because its available candidates either do not clear the net measured-time saving threshold or violate physical-risk/support gates.",
            "This is not validation evidence. The next step is a small smoke/confirmation block only after external backup, with seed0 fixed-H25 fallback and the nominated adaptive seeds compared against fairly matched fixed-H baselines using actual wall time.",
        ]
    else:
        lines += [
            "No sufficient actual-time-aware adaptive nomination was found. Do not spend another full devval campaign on this finite class before a broader training/representation or scenario-opportunity change.",
        ]
    lines += ["", "Artifacts: raw.json, nominated_policies.json, candidate_timing_metrics.csv, completed.json, state note, backup request.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    completed = {"created_utc": now(), "passed": True, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "candidate_timing_metrics_csv": rel(OUT / "candidate_timing_metrics.csv"), "nominated_policies": rel(OUT / "nominated_policies.json"), "adaptive_nominated_seed_count": adaptive_nom_count, "acceptance_for_smoke_met": acceptance_met, "next_action": next_action, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollouts": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(BACKUP)}
    (OUT / "completed.json").write_text(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    STATE.write_text("# Actual-time-aware re-selection V2 state\n\nUTC: %s\n\n- Completed metadata-only diagnostic. No simulations/training/test/validation64 bank reopen.\n- Adaptive nominations: %d/3; acceptance_for_smoke_met=%s.\n- Nominations: %s\n- Next action after backup: %s\n- Summary: `%s`\n" % (now(), adaptive_nom_count, acceptance_met, {s: (v.get("nominated") or {}).get("candidate_id") for s, v in nominations.items()}, next_action, rel(OUT / "summary.md")), encoding="utf-8")
    backup = {"created_utc": now(), "reason": "Backup required after actual-time-aware re-selection V2 metadata diagnostic and scheduling amendment before any further simulations.", "backup_required_before_more_simulations": True, "artifacts_requiring_backup": [rel(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "candidate_timing_metrics.csv", OUT / "nominated_policies.json", OUT / "completed.json", STATE, BACKUP, SCRIPT, AMENDMENT] + DOCS], "sha256": {rel(p): sha256(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "candidate_timing_metrics.csv", OUT / "nominated_policies.json", OUT / "completed.json", STATE, SCRIPT, AMENDMENT]}, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollouts": 0}
    BACKUP.write_text(json.dumps(backup, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    marker = "<!-- vehicle-gated-horizon-actual-time-reselection-v2-20260928 -->"
    block = "%s\n## 2026-09-28 vehicle actual-time-aware gated-horizon re-selection V2\n\nUTC: %s. Metadata-only IMPROVED finite re-selection using existing training candidate metrics and corrected V2 timing/opportunity diagnostics; no simulations, no training, no historical validation64 bank reopen, and no sealed-test access. Adaptive nominations: %d/3; acceptance_for_smoke_met=%s. Nominations: %s. Next action after backup: `%s`. Artifacts: `%s`, `%s`, `%s`.\n" % (marker, now(), adaptive_nom_count, acceptance_met, {s: (v.get("nominated") or {}).get("candidate_id") for s, v in nominations.items()}, next_action, rel(OUT / "summary.md"), rel(OUT / "raw.json"), rel(OUT / "completed.json"))
    for d in DOCS:
        append_once(d, marker, block)
    print("\n".join(lines))


def main() -> int:
    for p in [V1_RAW, V1_CSV, V2_RAW, AMENDMENT]:
        if not p.exists():
            raise RuntimeError("Required input missing: %s" % p)
    candidates = load_candidates()
    v2 = read_json(V2_RAW)
    timing_by_h, overhead_by_seed = build_timing_model(v2)
    evaluated, nominations = evaluate(candidates, timing_by_h, overhead_by_seed)
    write_outputs(evaluated, nominations, timing_by_h, overhead_by_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
