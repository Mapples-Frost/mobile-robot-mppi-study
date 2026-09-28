#!/usr/bin/env python3
"""Vehicle gated-horizon risk-first re-selection v1.

Metadata-only controlled re-selection over the CURRENT gated-horizon search
outputs.  This script does not run rollouts, train, read validation64, or access
sealed tests.  It is IMPROVED finite candidate re-selection/refit, not original
SAC and not neural retraining.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
GATED_ROOT = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25"
TRAIN_ROOT = GATED_ROOT / "train"
DIAG_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1.py"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_risk_reselection_v1_protocol_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_risk_reselection_v1_protocol_20260928.json"
CURRENT_AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_20260928T015606+0000/completed.json"
SEEDS = (0, 1, 2)
BASE_H = 25
EXPECTED_CASES = 24
METHOD = "IMPROVED_vehicle_gated_horizon_risk_reselection_v1_metadata_only"

PRIMARY = {
    "mean_physical_delta_pct_max": 0.25,
    "max_pair_physical_regression_pct_max": 2.0,
    "positive_tail_cvar80_pct_max": 1.0,
    "short_step_fraction_min": 0.03,
    "work_saving_fraction_min": 0.02,
    "episodes_with_short_min": 8,
    "top2_short_step_share_max": 0.60,
    "switch_rate_max": 0.08,
}


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def stamp_from_time(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "+0000")


def fnum(x: Any) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def inum(x: Any) -> Optional[int]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        if math.isfinite(y) and abs(y - round(y)) < 1e-9:
            return int(round(y))
    except Exception:
        return None
    return None


def summary(values: Iterable[Any]) -> Dict[str, Any]:
    xs = sorted(v for v in (fnum(x) for x in values) if v is not None)
    if not xs:
        return {"count": 0, "mean": None, "median": None, "min": None, "p80": None, "p95": None, "max": None}
    n = len(xs)
    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        pos = (n - 1) * p
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"count": n, "mean": float(math.fsum(xs) / n), "median": float(q(0.5)), "min": float(xs[0]), "p80": float(q(0.8)), "p95": float(q(0.95)), "max": float(xs[-1])}


def episode_case(ep: Mapping[str, Any], fallback: int) -> int:
    return int(ep.get("case", fallback))


def physical_cost(ep: Mapping[str, Any]) -> float:
    for key in ("physical_constraint_cost", "physical_cost"):
        val = fnum(ep.get(key))
        if val is not None:
            return val
    perf = fnum(ep.get("performance_cost")) or 0.0
    con = fnum(ep.get("constraint_cost")) or 0.0
    return float(perf + con)


def total_cost(ep: Mapping[str, Any]) -> float:
    val = fnum(ep.get("total_cost"))
    if val is not None:
        return val
    return physical_cost(ep) + (fnum(ep.get("h_penalty")) or 0.0)


def success_value(ep: Mapping[str, Any]) -> int:
    return 1 if bool(ep.get("success")) else 0


def constraint_value(ep: Mapping[str, Any]) -> int:
    return 1 if bool(ep.get("constraint")) else 0


def short_steps_estimate(policy: Mapping[str, Any], ep: Mapping[str, Any]) -> float:
    short_h = inum(policy.get("short_h"))
    steps = inum(ep.get("steps")) or 0
    mh = fnum(ep.get("mean_horizon"))
    if short_h is None or short_h >= BASE_H or mh is None or steps <= 0:
        return 0.0
    denom = float(BASE_H - short_h)
    est = (BASE_H - mh) * steps / denom
    if est < 0 and est > -1e-7:
        est = 0.0
    if est > steps and est < steps + 1e-6:
        est = float(steps)
    return float(max(0.0, min(float(steps), est)))


def paired_episodes(episodes: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for idx, ep in enumerate(episodes):
        out[episode_case(ep, idx)] = ep
    return out


def old_selected_id(fit: Mapping[str, Any]) -> str:
    return str((fit.get("selected") or {}).get("id"))


def candidate_metrics(seed: int, result: Mapping[str, Any], fixed: Mapping[str, Any], current_id: str) -> Dict[str, Any]:
    policy = result.get("policy") or {}
    cid = str(policy.get("id"))
    episodes = list(result.get("episodes") or [])
    fixed_eps = paired_episodes(fixed.get("episodes") or [])
    cand_eps = paired_episodes(episodes)
    fully = bool(result.get("fully_evaluated")) and len(episodes) >= EXPECTED_CASES and all(i in cand_eps for i in range(EXPECTED_CASES))
    rejected = bool(result.get("rejected"))
    short_h = inum(policy.get("short_h"))
    is_fixed = cid == "fixed"
    paired_case_metrics: List[Dict[str, Any]] = []
    regressions_pct: List[float] = []
    positive_regressions_pct: List[float] = []
    improvements_abs: List[float] = []
    physical_deltas: List[float] = []
    raw_deltas: List[float] = []
    short_by_case: List[float] = []
    worse_success = worse_constraint = worse_initial_fail = worse_final_fail = False
    for case in range(EXPECTED_CASES):
        fe = fixed_eps.get(case)
        ce = cand_eps.get(case)
        if fe is None or ce is None:
            continue
        fphys = physical_cost(fe)
        cphys = physical_cost(ce)
        fraw = total_cost(fe)
        craw = total_cost(ce)
        dphys = cphys - fphys
        draw = craw - fraw
        denom = max(abs(fphys), 1.0)
        rpct = 100.0 * dphys / denom
        regressions_pct.append(rpct)
        positive_regressions_pct.append(max(0.0, rpct))
        improvements_abs.append(max(0.0, fphys - cphys))
        physical_deltas.append(dphys)
        raw_deltas.append(draw)
        ss = short_steps_estimate(policy, ce)
        short_by_case.append(ss)
        if success_value(ce) < success_value(fe):
            worse_success = True
        if constraint_value(ce) > constraint_value(fe):
            worse_constraint = True
        if (inum(ce.get("initial_failed_steps")) or 0) > (inum(fe.get("initial_failed_steps")) or 0):
            worse_initial_fail = True
        if (inum(ce.get("solver_failure_steps")) or 0) > (inum(fe.get("solver_failure_steps")) or 0):
            worse_final_fail = True
        paired_case_metrics.append({
            "case": case,
            "fixed_physical": fphys,
            "candidate_physical": cphys,
            "physical_delta": dphys,
            "physical_regression_pct": rpct,
            "fixed_total": fraw,
            "candidate_total": craw,
            "raw_delta": draw,
            "candidate_short_steps_estimated": ss,
            "candidate_steps": inum(ce.get("steps")) or 0,
            "candidate_switches": inum(ce.get("switches")) or 0,
        })
    fixed_mean_phys = fnum(fixed.get("mean_physical_cost"))
    cand_mean_phys = fnum(result.get("mean_physical_cost"))
    fixed_mean_raw = fnum(fixed.get("mean_raw_cost"))
    cand_mean_raw = fnum(result.get("mean_raw_cost"))
    steps = int(sum((inum(ep.get("steps")) or 0) for ep in episodes))
    switches = int(sum((inum(ep.get("switches")) or 0) for ep in episodes))
    short_steps = float(math.fsum(short_by_case))
    episodes_with_short = int(sum(1 for x in short_by_case if x > 0.5))
    if short_steps > 0:
        top2_short_share = float(math.fsum(sorted(short_by_case, reverse=True)[:2]) / short_steps)
    else:
        top2_short_share = None
    total_improvement = float(math.fsum(improvements_abs))
    if total_improvement > 0:
        top2_improvement_share = float(math.fsum(sorted(improvements_abs, reverse=True)[:2]) / total_improvement)
    else:
        top2_improvement_share = None
    tail_k = max(1, int(math.ceil(0.2 * EXPECTED_CASES)))
    positive_tail_cvar80_pct = float(math.fsum(sorted(positive_regressions_pct, reverse=True)[:tail_k]) / tail_k) if positive_regressions_pct else None
    max_regression_pct = max(regressions_pct) if regressions_pct else None
    mean_phys_delta_pct = None
    if cand_mean_phys is not None and fixed_mean_phys not in (None, 0.0):
        mean_phys_delta_pct = 100.0 * (cand_mean_phys - fixed_mean_phys) / abs(fixed_mean_phys)
    mean_raw_delta_pct = None
    if cand_mean_raw is not None and fixed_mean_raw not in (None, 0.0):
        mean_raw_delta_pct = 100.0 * (cand_mean_raw - fixed_mean_raw) / abs(fixed_mean_raw)
    mean_h_penalty = summary(ep.get("h_penalty") for ep in episodes)["mean"]
    fixed_h_penalty = summary(ep.get("h_penalty") for ep in fixed.get("episodes") or [])["mean"]
    mean_h_penalty_delta = None if mean_h_penalty is None or fixed_h_penalty is None else float(mean_h_penalty - fixed_h_penalty)
    short_step_fraction = float(short_steps / steps) if steps else None
    if short_h is not None and short_h < BASE_H and short_step_fraction is not None:
        work_saving_fraction = float(short_step_fraction * (BASE_H - short_h) / BASE_H)
    else:
        work_saving_fraction = 0.0
    switch_rate = float(switches / steps) if steps else None

    reasons: List[str] = []
    if is_fixed:
        reasons.append("fixed_baseline_not_adaptive_candidate")
    if not fully:
        reasons.append("not_fully_evaluated_all_24_cases")
    if rejected:
        reasons.append("historically_rejected_or_pruned")
    if short_h is None or short_h >= BASE_H:
        reasons.append("no_short_horizon")
    if worse_success:
        reasons.append("per_case_success_worse_than_fixed")
    if worse_constraint:
        reasons.append("per_case_constraint_worse_than_fixed")
    if worse_initial_fail:
        reasons.append("per_case_initial_solver_failure_worse_than_fixed")
    if worse_final_fail:
        reasons.append("per_case_final_solver_failure_worse_than_fixed")
    if mean_phys_delta_pct is None or mean_phys_delta_pct > PRIMARY["mean_physical_delta_pct_max"]:
        reasons.append("mean_physical_delta_pct_gt_0p25")
    if max_regression_pct is None or max_regression_pct > PRIMARY["max_pair_physical_regression_pct_max"]:
        reasons.append("max_pair_physical_regression_pct_gt_2")
    if positive_tail_cvar80_pct is None or positive_tail_cvar80_pct > PRIMARY["positive_tail_cvar80_pct_max"]:
        reasons.append("positive_tail_cvar80_pct_gt_1")
    if short_step_fraction is None or short_step_fraction < PRIMARY["short_step_fraction_min"]:
        reasons.append("short_step_fraction_lt_0p03")
    if work_saving_fraction < PRIMARY["work_saving_fraction_min"]:
        reasons.append("work_saving_fraction_lt_0p02")
    if episodes_with_short < PRIMARY["episodes_with_short_min"]:
        reasons.append("episodes_with_short_lt_8")
    if top2_short_share is None or top2_short_share > PRIMARY["top2_short_step_share_max"]:
        reasons.append("top2_short_step_share_gt_0p60")
    if switch_rate is None or switch_rate > PRIMARY["switch_rate_max"]:
        reasons.append("switch_rate_gt_0p08")
    primary_eligible = len(reasons) == 0

    return {
        "seed": seed,
        "candidate_id": cid,
        "stored_current_policy": cid == current_id,
        "policy_kind": "fixed_H25_baseline" if is_fixed else "gated_shortening_candidate",
        "short_h": short_h,
        "profile": inum(policy.get("profile")),
        "guard": inum(policy.get("guard")),
        "fully_evaluated_all_24_cases": fully,
        "episodes_run": len(episodes),
        "scored_steps_run": steps,
        "historically_rejected_or_pruned": rejected,
        "unrun_case_count": len(result.get("unrun_cases") or []),
        "historical_reject_reasons": sorted({str(reason) for item in (result.get("rejected") or []) for reason in (item.get("reasons") or [])}),
        "mean_raw_cost": cand_mean_raw,
        "mean_physical_cost": cand_mean_phys,
        "mean_raw_delta_pct_vs_fixed": mean_raw_delta_pct,
        "mean_physical_delta_pct_vs_fixed": mean_phys_delta_pct,
        "mean_h_penalty_delta_vs_fixed": mean_h_penalty_delta,
        "mean_physical_delta_abs_vs_fixed": None if cand_mean_phys is None or fixed_mean_phys is None else cand_mean_phys - fixed_mean_phys,
        "mean_raw_delta_abs_vs_fixed": None if cand_mean_raw is None or fixed_mean_raw is None else cand_mean_raw - fixed_mean_raw,
        "max_pair_physical_regression_pct": max_regression_pct,
        "positive_tail_cvar80_pct": positive_tail_cvar80_pct,
        "pair_physical_delta_summary": summary(physical_deltas),
        "pair_raw_delta_summary": summary(raw_deltas),
        "short_steps_estimated": short_steps,
        "episodes_with_short_estimated": episodes_with_short,
        "short_step_fraction": short_step_fraction,
        "work_saving_fraction": work_saving_fraction,
        "switches_sum": switches,
        "switch_rate": switch_rate,
        "top2_short_step_share": top2_short_share,
        "top2_physical_improvement_share": top2_improvement_share,
        "worse_success_than_fixed": worse_success,
        "worse_constraint_than_fixed": worse_constraint,
        "worse_initial_solver_fail_than_fixed": worse_initial_fail,
        "worse_final_solver_fail_than_fixed": worse_final_fail,
        "primary_adaptive_eligible": primary_eligible,
        "primary_ineligibility_reasons": reasons,
        "paired_case_metrics": paired_case_metrics,
    }


def primary_sort_key(m: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        max(0.0, fnum(m.get("max_pair_physical_regression_pct")) or 0.0),
        max(0.0, fnum(m.get("positive_tail_cvar80_pct")) or 0.0),
        max(0.0, fnum(m.get("mean_physical_delta_pct_vs_fixed")) or 0.0),
        -(fnum(m.get("work_saving_fraction")) or 0.0),
        fnum(m.get("switch_rate")) if fnum(m.get("switch_rate")) is not None else float("inf"),
        str(m.get("candidate_id")),
    )


def nearest_sort_key(m: Mapping[str, Any]) -> Tuple[Any, ...]:
    reasons = [r for r in (m.get("primary_ineligibility_reasons") or []) if r != "fixed_baseline_not_adaptive_candidate"]
    return (
        len(reasons),
        max(0.0, (fnum(m.get("max_pair_physical_regression_pct")) or 0.0) - PRIMARY["max_pair_physical_regression_pct_max"]),
        max(0.0, (fnum(m.get("positive_tail_cvar80_pct")) or 0.0) - PRIMARY["positive_tail_cvar80_pct_max"]),
        max(0.0, (fnum(m.get("mean_physical_delta_pct_vs_fixed")) or 0.0) - PRIMARY["mean_physical_delta_pct_max"]),
        -(fnum(m.get("work_saving_fraction")) or 0.0),
        str(m.get("candidate_id")),
    )


def compact_metric(m: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if m is None:
        return None
    keys = [
        "candidate_id", "stored_current_policy", "short_h", "profile", "guard", "primary_adaptive_eligible",
        "primary_ineligibility_reasons", "mean_raw_delta_pct_vs_fixed", "mean_physical_delta_pct_vs_fixed",
        "max_pair_physical_regression_pct", "positive_tail_cvar80_pct", "short_step_fraction",
        "work_saving_fraction", "episodes_with_short_estimated", "top2_short_step_share",
        "top2_physical_improvement_share", "switch_rate",
    ]
    return {k: m.get(k) for k in keys}


def sensitivity_counts(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    base = [m for m in metrics if m.get("policy_kind") != "fixed_H25_baseline" and m.get("fully_evaluated_all_24_cases") and not m.get("historically_rejected_or_pruned")]
    for max_reg in (1.0, 2.0, 5.0):
        for min_short in (0.0, 0.01, 0.03, 0.05):
            key = "maxreg_%s_minshort_%s" % (str(max_reg).replace(".", "p"), str(min_short).replace(".", "p"))
            out[key] = sum(
                1 for m in base
                if (fnum(m.get("mean_physical_delta_pct_vs_fixed")) is not None and fnum(m.get("mean_physical_delta_pct_vs_fixed")) <= PRIMARY["mean_physical_delta_pct_max"])
                and (fnum(m.get("max_pair_physical_regression_pct")) is not None and fnum(m.get("max_pair_physical_regression_pct")) <= max_reg)
                and (fnum(m.get("positive_tail_cvar80_pct")) is not None and fnum(m.get("positive_tail_cvar80_pct")) <= PRIMARY["positive_tail_cvar80_pct_max"])
                and (fnum(m.get("short_step_fraction")) is not None and fnum(m.get("short_step_fraction")) >= min_short)
                and (fnum(m.get("switch_rate")) is not None and fnum(m.get("switch_rate")) <= PRIMARY["switch_rate_max"])
            )
    return out


def write_candidate_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "seed", "candidate_id", "stored_current_policy", "policy_kind", "short_h", "profile", "guard",
        "fully_evaluated_all_24_cases", "episodes_run", "scored_steps_run", "historically_rejected_or_pruned",
        "unrun_case_count", "historical_reject_reasons", "mean_raw_cost", "mean_physical_cost",
        "mean_raw_delta_pct_vs_fixed", "mean_physical_delta_pct_vs_fixed", "mean_h_penalty_delta_vs_fixed",
        "max_pair_physical_regression_pct", "positive_tail_cvar80_pct", "short_steps_estimated",
        "episodes_with_short_estimated", "short_step_fraction", "work_saving_fraction", "switches_sum",
        "switch_rate", "top2_short_step_share", "top2_physical_improvement_share", "primary_adaptive_eligible",
        "primary_ineligibility_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["seed"]), r["candidate_id"] != "fixed", str(r["candidate_id"]))):
            item = {k: row.get(k) for k in fields}
            item["historical_reject_reasons"] = ";".join(item.get("historical_reject_reasons") or [])
            item["primary_ineligibility_reasons"] = ";".join(item.get("primary_ineligibility_reasons") or [])
            writer.writerow(item)


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")


def append_registry(created: str, completed: Path) -> None:
    p = ROOT / "EXPERIMENT_REGISTRY.csv"
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    if rel(completed) in existing:
        return
    row = {
        "experiment_id": "",
        "timestamp": created,
        "method": METHOD,
        "seed": "metadata_existing_gated_horizon_vehicle_s0_s1_s2_no_rng",
        "split": "same_24_training_cases_existing_gated_search_no_validation_no_test",
        "commit_sha": "",
        "status": "completed_metadata_reselection",
        "exit_status": "",
        "runtime_seconds": "",
        "peak_process_rss_kb": "",
        "record": rel(completed),
    }
    with p.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not existing.strip():
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    created = utc_now()
    stamp = stamp_from_time(created)
    out_dir = DIAG_ROOT / ("vehicle_gated_horizon_risk_reselection_v1_%s" % stamp)
    out_dir.mkdir(parents=True, exist_ok=False)

    input_hashes = {
        rel(SCRIPT): sha256(SCRIPT),
        rel(PROTOCOL_MD): sha256(PROTOCOL_MD),
        rel(PROTOCOL_JSON): sha256(PROTOCOL_JSON),
        rel(CURRENT_AUDIT): sha256(CURRENT_AUDIT),
    }
    all_rows: List[Dict[str, Any]] = []
    seed_summaries: Dict[str, Any] = {}
    nominations: Dict[str, Any] = {}

    for seed in SEEDS:
        root = TRAIN_ROOT / ("vehicle_s%d" % seed)
        fit_path = root / "fit_completed.json"
        policy_path = root / "policy.json"
        fit = read_json(fit_path)
        stored_policy = read_json(policy_path)
        input_hashes[rel(fit_path)] = sha256(fit_path)
        input_hashes[rel(policy_path)] = sha256(policy_path)
        results = fit.get("results") or []
        if not results or (results[0].get("policy") or {}).get("id") != "fixed":
            raise RuntimeError("Expected fixed result first for vehicle_s%d" % seed)
        current_id = str(stored_policy.get("id") or old_selected_id(fit))
        fixed = results[0]
        metrics = [candidate_metrics(seed, r, fixed, current_id) for r in results]
        all_rows.extend(metrics)
        primary = sorted([m for m in metrics if m.get("primary_adaptive_eligible")], key=primary_sort_key)
        nominated = primary[0] if primary else next(m for m in metrics if m["candidate_id"] == "fixed")
        fallback_reason = None if primary else "no_primary_adaptive_candidate_by_frozen_risk_rule"
        current_metric = next((m for m in metrics if m["candidate_id"] == current_id), None)
        fixed_metric = next((m for m in metrics if m["candidate_id"] == "fixed"), None)
        near = sorted([m for m in metrics if m.get("policy_kind") != "fixed_H25_baseline"], key=nearest_sort_key)[:10]
        changed_from_current = nominated.get("candidate_id") != current_id
        nominations[str(seed)] = {
            "stored_current_policy": compact_metric(current_metric),
            "fixed_fallback": compact_metric(fixed_metric),
            "nominated_policy": compact_metric(nominated),
            "nominated_candidate_id": nominated.get("candidate_id"),
            "fallback_reason": fallback_reason,
            "changed_from_current": bool(changed_from_current),
            "primary_adaptive_eligible_count": len(primary),
        }
        reason_counts: Dict[str, int] = {}
        for m in metrics:
            for reason in m.get("primary_ineligibility_reasons") or []:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        seed_summaries[str(seed)] = {
            "fit_path": rel(fit_path),
            "policy_path": rel(policy_path),
            "stored_policy": stored_policy,
            "historical_selected_policy": fit.get("selected"),
            "candidate_count": len(metrics),
            "fully_evaluated_nonfixed_count": sum(1 for m in metrics if m.get("policy_kind") != "fixed_H25_baseline" and m.get("fully_evaluated_all_24_cases")),
            "primary_adaptive_eligible_count": len(primary),
            "nominated": nominations[str(seed)],
            "nearest_candidates": [compact_metric(m) for m in near],
            "sensitivity_counts": sensitivity_counts(metrics),
            "primary_ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        }

    adaptive_changed = sum(1 for s in nominations.values() if s["changed_from_current"] and s["fallback_reason"] is None)
    adaptive_nominated = sum(1 for s in nominations.values() if s["fallback_reason"] is None)
    next_action = (
        "freeze_and_run_small_smoke_after_backup" if adaptive_changed >= 2
        else "do_not_smoke_existing_reselection_design_expanded_policy_or_training_ablation"
    )
    interpretation: List[str] = []
    if adaptive_nominated == 0:
        interpretation.append("No seed produced a primary-adaptive-eligible candidate under the frozen risk-first rule; existing current gated-search traces do not support robust re-selection without expanding/refitting the method.")
    else:
        interpretation.append("At least one seed produced a primary-adaptive-eligible candidate, but this metadata result is not validation evidence; smoke/validation would be required before any controller use.")
    if adaptive_changed < 2:
        interpretation.append("The frozen next-action criterion for a smoke is not met (<2 seeds nominate a changed adaptive policy), so the next informative action is a versioned expanded policy/search or training/value ablation rather than more validation of unchanged v1/v2 local patches.")
    else:
        interpretation.append("The frozen next-action criterion is met (>=2 changed adaptive nominations); after backup verification, freeze a smoke protocol before any fresh validation.")

    candidate_csv = out_dir / "candidate_metrics.csv"
    write_candidate_csv(candidate_csv, all_rows)
    nominations_path = out_dir / "nominated_policies.json"
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"

    raw = {
        "created_utc": created,
        "method": METHOD,
        "scope": "metadata-only risk-first re-selection over existing CURRENT gated-horizon vehicle search outputs",
        "protocol": {"md": rel(PROTOCOL_MD), "json": rel(PROTOCOL_JSON), "primary_rule": PRIMARY},
        "input_hashes": input_hashes,
        "access_flags": {
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "historical_validation64_bank_opened": False,
            "fresh_devval_bank_generated": False,
            "sealed_test_accessed": False,
            "final_test_authorization_requested": False,
        },
        "seed_summaries": seed_summaries,
        "nominations": nominations,
        "candidate_metrics": all_rows,
        "decision": {
            "adaptive_nominated_seed_count": adaptive_nominated,
            "adaptive_changed_from_current_seed_count": adaptive_changed,
            "next_action": next_action,
            "interpretation": interpretation,
        },
        "artifacts": {
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "candidate_metrics_csv": rel(candidate_csv),
            "nominated_policies": rel(nominations_path),
            "completed": rel(completed_path),
        },
    }
    write_json(nominations_path, nominations)
    write_json(raw_path, raw)

    lines = [
        "# Vehicle gated-horizon risk-first re-selection v1",
        "",
        "UTC: `%s`." % created,
        "",
        "Metadata-only IMPROVED finite candidate re-selection over existing current gated-horizon search outputs. No rollout/control steps, no training/gradient steps, no validation64 reopen, no fresh devval bank, and no sealed-test access occurred.",
        "",
        "## Frozen hypothesis and primary rule",
        "",
        "Hypothesis: the old mean raw-cost objective with a synthetic horizon penalty may over-reward rare/fragile gate triggers and hide physical tail risk. The frozen primary rule requires fully evaluated, unrejected adaptive candidates with mean physical delta <= +0.25%, max paired physical regression <= +2%, positive-tail CVaR80 <= +1%, short-step fraction >= 0.03, horizon-work saving >= 0.02, >=8 cases with short-horizon use, top-two short-step share <=0.60, and switch rate <=0.08.",
        "",
        "## Nominations",
        "",
        "| seed | current | current primary? | nominated | primary eligible count | changed from current | fallback reason | current risk summary | nominated risk summary |",
        "|---:|---|---:|---|---:|---:|---|---|---|",
    ]
    for seed in map(str, SEEDS):
        nom = nominations[seed]
        cur = nom["stored_current_policy"] or {}
        got = nom["nominated_policy"] or {}
        def risk(m: Mapping[str, Any]) -> str:
            return "physΔ%={:.4g}, maxReg%={:.4g}, cvar%={:.4g}, short={:.4g}, work={:.4g}, top2short={}, switch={:.4g}".format(
                fnum(m.get("mean_physical_delta_pct_vs_fixed")) or 0.0,
                fnum(m.get("max_pair_physical_regression_pct")) or 0.0,
                fnum(m.get("positive_tail_cvar80_pct")) or 0.0,
                fnum(m.get("short_step_fraction")) or 0.0,
                fnum(m.get("work_saving_fraction")) or 0.0,
                "None" if m.get("top2_short_step_share") is None else ("%.4g" % (fnum(m.get("top2_short_step_share")) or 0.0)),
                fnum(m.get("switch_rate")) or 0.0,
            )
        lines.append("| %s | `%s` | %s | `%s` | %d | %s | %s | %s | %s |" % (
            seed,
            cur.get("candidate_id"),
            cur.get("primary_adaptive_eligible"),
            got.get("candidate_id"),
            nom["primary_adaptive_eligible_count"],
            nom["changed_from_current"],
            nom["fallback_reason"] or "",
            risk(cur),
            risk(got),
        ))
    lines += [
        "",
        "## Per-seed near misses",
        "",
    ]
    for seed in map(str, SEEDS):
        s = seed_summaries[seed]
        lines.append("### vehicle_s%s" % seed)
        lines.append("- Stored policy: `%s`; primary adaptive eligible count: `%s`; sensitivity counts: `%s`." % (s["stored_policy"], s["primary_adaptive_eligible_count"], s["sensitivity_counts"]))
        lines.append("- Top nearest candidates (not primary rule changes):")
        for m in s["nearest_candidates"][:5]:
            lines.append("  - `%s`: eligible=%s, reasons=%s, physΔ%%=%s, maxReg%%=%s, cvar%%=%s, short=%s, work=%s, top2short=%s, switch=%s" % (
                m.get("candidate_id"), m.get("primary_adaptive_eligible"), m.get("primary_ineligibility_reasons"),
                m.get("mean_physical_delta_pct_vs_fixed"), m.get("max_pair_physical_regression_pct"),
                m.get("positive_tail_cvar80_pct"), m.get("short_step_fraction"), m.get("work_saving_fraction"),
                m.get("top2_short_step_share"), m.get("switch_rate"),
            ))
        lines.append("")
    lines += [
        "## Decision",
        "",
        "- Adaptive nominated seed count: `%d`." % adaptive_nominated,
        "- Adaptive changed-from-current seed count: `%d`." % adaptive_changed,
        "- Frozen next action: `%s`." % next_action,
        "- Interpretation: %s" % " ".join(interpretation),
        "",
        "This result is not fresh validation evidence and does not alter frozen v1/v2 controllers or completed shard00 outcomes.",
        "",
        "Artifacts: `raw.json`, `candidate_metrics.csv`, `nominated_policies.json`, `completed.json`, and backup request.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_%s.json" % stamp)
    backup = {
        "request_utc": created,
        "reason": "after vehicle gated-horizon risk-first re-selection v1 metadata diagnostic",
        "artifacts": [rel(summary_path), rel(raw_path), rel(candidate_csv), rel(nominations_path), rel(completed_path), rel(PROTOCOL_MD), rel(PROTOCOL_JSON), rel(SCRIPT)],
        "sealed_test_accessed": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }
    write_json(backup_path, backup)
    completed = {
        "created_utc": created,
        "passed": True,
        "method": METHOD,
        "summary_path": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "raw_path": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "candidate_metrics_csv": rel(candidate_csv),
        "candidate_metrics_csv_sha256": sha256(candidate_csv),
        "nominated_policies": rel(nominations_path),
        "nominated_policies_sha256": sha256(nominations_path),
        "backup_request": rel(backup_path),
        "backup_request_sha256": sha256(backup_path),
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "fresh_devval_bank_generated": False,
        "sealed_test_accessed": False,
        "decision": raw["decision"],
    }
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256(completed_path)
    write_json(completed_path, completed)

    doc_body = (
        "### Vehicle gated-horizon risk-first re-selection v1 (%s)\n"
        "Metadata-only re-selection over existing current gated-search training traces completed: %s. "
        "No rollout/control/training/test access. Frozen decision `%s`; adaptive nominated seeds=%d, changed-from-current adaptive seeds=%d. Summary: %s. Backup request: %s."
        % (created, rel(completed_path), next_action, adaptive_nominated, adaptive_changed, rel(summary_path), rel(backup_path))
    )
    marker = "vehicle-gated-horizon-risk-reselection-v1-%s" % stamp
    for doc in (ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "RESULTS_AUDIT.md", ROOT / "DECISIONS.md"):
        append_once(doc, marker, doc_body)
    append_registry(created, completed_path)

    print(json.dumps({
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "candidate_metrics_csv": rel(candidate_csv),
        "nominated_policies": rel(nominations_path),
        "completed": rel(completed_path),
        "backup_request": rel(backup_path),
        "decision": raw["decision"],
        "sealed_test_accessed": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
