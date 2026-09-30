#!/usr/bin/env python3
"""v24 offline adaptivity-opportunity audit for true-variable H12/H15 evidence.

Development-only IMPROVED diagnostic.  It uses already-opened development outputs
(v19, v21, v22, v23) to answer the immediate post-v23 question:

  * Is the current stress/source-independent distribution mainly a fixed-H12
    compute-saving result rather than an adaptive-horizon result?
  * Where, if anywhere, does H12/H15 state-dependent adaptation remain necessary?
  * What bounded follow-up should be run before any validation/test planning?

No simulator is called.  No validation64 bank or sealed test artifact is read.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24"
STAMP = "20260930T0205Z"
MARKER = f"vehicle-true-variable-H-h12-adaptivity-opportunity-audit-v24-{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0205_after_h12_adaptivity_opportunity_audit_v24.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_ADAPTIVITY_OPPORTUNITY_AUDIT_V24_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V19_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/completed.json"
V20B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/completed.json"
V21_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json"
V21_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/completed.json"
V22_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/completed.json"
V22_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/raw.json"
V23_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/raw.json"
V23_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/completed.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

MIN_SAVE = 0.05


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def pct(x: Any) -> str:
    return "NA" if x is None else f"{100.0 * sf(x):.2f}%"


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return math.fsum(vals) / len(vals) if vals else 0.0


def finite_summary(xs: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def quant(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = q * (len(vals) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"n": len(vals), "min": vals[0], "median": quant(0.5), "mean": mean(vals), "p95": quant(0.95), "max": vals[-1], "sum": math.fsum(vals)}


def assert_completed_dev(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing {label}: {rel(path)}")
    obj = read_json(path)
    ok = obj.get("status") in ("complete", "completed") or obj.get("hard_pass") is True or obj.get("passed") is True
    if not ok:
        raise ContractError(f"prerequisite not complete/pass: {label} {rel(path)}")
    for key in ("sealed_test_accessed", "sealed_test_bank_opened", "test_accessed", "validation64_bank_opened"):
        if obj.get(key) is True:
            raise ContractError(f"forbidden {key}=true in prerequisite {label}")
    return obj


def row_tolerance(h15_phys: float, explicit: Optional[float] = None) -> float:
    if explicit is not None and math.isfinite(float(explicit)) and float(explicit) > 0:
        return float(explicit)
    return max(2.0, 0.05 * abs(h15_phys))


def canonical_row(dataset: str, row_id: str, role: str, source_key: str,
                  h12_phys: float, h15_phys: float, h12_dec: float, h15_dec: float,
                  h12_solver: float, h15_solver: float, tol: float,
                  h12_cat: bool, h12_ben: bool,
                  source_candidate_index: Optional[int] = None,
                  branch_step: Optional[int] = None,
                  extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "row_id": str(row_id),
        "role": str(role),
        "source_key": str(source_key),
        "source_candidate_index": source_candidate_index,
        "branch_step": branch_step,
        "h12_physical": float(h12_phys),
        "h15_physical": float(h15_phys),
        "phys_delta_h12_minus_h15": float(h12_phys - h15_phys),
        "h12_decision_sum_s": float(h12_dec),
        "h15_decision_sum_s": float(h15_dec),
        "decision_gain_h12_vs_h15_s": float(h15_dec - h12_dec),
        "h12_solver_sum_s": float(h12_solver),
        "h15_solver_sum_s": float(h15_solver),
        "solver_gain_h12_vs_h15_s": float(h15_solver - h12_solver),
        "row_physical_tolerance": float(tol),
        "h12_catastrophic_vs_h15": bool(h12_cat),
        "h12_beneficial_vs_h15": bool(h12_ben),
        "extra": dict(extra or {}),
    }


def load_v19_rows() -> List[Dict[str, Any]]:
    raw = read_json(V19_RAW)
    pairs = [r for r in raw.get("analysis", {}).get("pair_rows", []) if not r.get("missing_pair")]
    by_id: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in pairs:
        by_id[str(r.get("candidate_id"))].append(r)
    out: List[Dict[str, Any]] = []
    for cid, rr in sorted(by_id.items(), key=lambda kv: si(kv[1][0].get("candidate_index"), 9999)):
        first = rr[0]
        h12_phys = mean([sf(r.get("short_physical")) for r in rr])
        h15_phys = mean([sf(r.get("ref_physical")) for r in rr])
        h12_dec = mean([sf(r.get("short_decision_sum_s")) for r in rr])
        h15_dec = mean([sf(r.get("ref_decision_sum_s")) for r in rr])
        h12_solver = mean([sf(r.get("short_solver_sum_s")) for r in rr])
        h15_solver = mean([sf(r.get("ref_solver_sum_s")) for r in rr])
        tol = mean([sf(r.get("row_tolerance_vs_ref"), row_tolerance(h15_phys)) for r in rr])
        h12_cat = any(bool(r.get("short_catastrophic_vs_ref")) for r in rr)
        # For repeat-averaged v19 states, classify beneficial if all repeats were beneficial or, equivalently,
        # if the averaged row is safe/non-cat, within tolerance and faster.
        h12_ben = (not h12_cat) and (h12_phys - h15_phys <= tol) and (h12_dec < h15_dec)
        out.append(canonical_row(
            "v19_opened_boundary", cid, first.get("role"), first.get("source_key"),
            h12_phys, h15_phys, h12_dec, h15_dec, h12_solver, h15_solver, tol,
            h12_cat, h12_ben,
            branch_step=si(first.get("branch_step"), -1),
            extra={"candidate_index": si(first.get("candidate_index"), -1), "repeat_rows": len(rr), "bank_id": first.get("bank_id"), "base_state_id": first.get("base_state_id"), "offset_from_center": first.get("offset_from_center")},
        ))
    return out


def nested_to_row(dataset: str, r: Mapping[str, Any]) -> Dict[str, Any]:
    if "h12_physical" in r and "h15_physical" in r:
        h12_phys = sf(r.get("h12_physical")); h15_phys = sf(r.get("h15_physical"))
        h12_dec = sf(r.get("h12_decision_sum_s")); h15_dec = sf(r.get("h15_decision_sum_s"))
        h12_solver = sf(r.get("h12_solver_sum_s")); h15_solver = sf(r.get("h15_solver_sum_s"))
        tol = row_tolerance(h15_phys, sf(r.get("row_physical_tolerance"), float("nan")))
        row_id = str(r.get("candidate_id") or r.get("base_state_id"))
        role = str(r.get("role") or r.get("risk_probe_role") or "unknown")
        source_key = str(r.get("source_key") or row_id)
    else:
        h12 = r.get("h12") or {}
        h15 = r.get("h15") or {}
        h12_phys = sf(h12.get("physical")); h15_phys = sf(h15.get("physical"))
        h12_dec = sf(h12.get("decision_sum_s")); h15_dec = sf(h15.get("decision_sum_s"))
        h12_solver = sf(h12.get("solver_sum_s")); h15_solver = sf(h15.get("solver_sum_s"))
        tol = row_tolerance(h15_phys, sf(r.get("row_tolerance_vs_H15"), float("nan")))
        row_id = str(r.get("base_state_id") or r.get("candidate_id"))
        role = str(r.get("risk_probe_role") or r.get("role") or "unknown")
        source_key = f"{dataset}/source{si(r.get('source_candidate_index'), -1):03d}/{row_id}"
    cat = bool(r.get("h12_catastrophic_vs_h15"))
    if "h12_beneficial_vs_h15" in r:
        ben = bool(r.get("h12_beneficial_vs_h15"))
    else:
        ben = (not cat) and (h12_phys - h15_phys <= tol) and (h12_dec < h15_dec)
    return canonical_row(dataset, row_id, role, source_key, h12_phys, h15_phys, h12_dec, h15_dec, h12_solver, h15_solver, tol, cat, ben, source_candidate_index=si(r.get("source_candidate_index"), -1), branch_step=si(r.get("branch_step"), -1), extra={"base_state_id": r.get("base_state_id"), "branch_state_slot": r.get("branch_state_slot"), "fresh_case_index": r.get("fresh_case_index")})


def find_state_rows(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates = [
        raw.get("analysis", {}).get("evaluation_rows"),
        raw.get("analysis", {}).get("label_analysis", {}).get("state_rows"),
        raw.get("analysis", {}).get("state_rows"),
    ]
    for obj in candidates:
        if isinstance(obj, list) and obj:
            return obj
    raise ContractError("could not locate state/evaluation rows in raw artifact")


def load_rows() -> Dict[str, List[Dict[str, Any]]]:
    v21_raw = read_json(V21_RAW)
    v23_raw = read_json(V23_RAW)
    return {
        "v19_opened_boundary": load_v19_rows(),
        "v21_source_independent": [nested_to_row("v21_source_independent", r) for r in find_state_rows(v21_raw)],
        "v23_broader_source_independent": [nested_to_row("v23_broader_source_independent", r) for r in find_state_rows(v23_raw)],
    }


def evaluate_fixed_or_oracle(rows: Sequence[Mapping[str, Any]], policy: str) -> Dict[str, Any]:
    h15_phys = h15_dec = h15_solver = 0.0
    pol_phys = pol_dec = pol_solver = 0.0
    tol_sum = 0.0
    bad = 0
    counts: Counter[str] = Counter()
    chosen_bad_rows: List[str] = []
    for r in rows:
        if policy == "fixed_H15":
            h = 15
        elif policy == "fixed_H12":
            h = 12
        elif policy == "oracle_H12_H15":
            h = 12 if bool(r.get("h12_beneficial_vs_h15")) else 15
        else:
            raise ValueError(policy)
        h15_phys += sf(r.get("h15_physical")); h15_dec += sf(r.get("h15_decision_sum_s")); h15_solver += sf(r.get("h15_solver_sum_s")); tol_sum += sf(r.get("row_physical_tolerance"))
        counts[str(h)] += 1
        if h == 12:
            pol_phys += sf(r.get("h12_physical")); pol_dec += sf(r.get("h12_decision_sum_s")); pol_solver += sf(r.get("h12_solver_sum_s"))
            if bool(r.get("h12_catastrophic_vs_h15")):
                bad += 1; chosen_bad_rows.append(str(r.get("row_id")))
        else:
            pol_phys += sf(r.get("h15_physical")); pol_dec += sf(r.get("h15_decision_sum_s")); pol_solver += sf(r.get("h15_solver_sum_s"))
    phys_delta = pol_phys - h15_phys
    dec_save = (h15_dec - pol_dec) / h15_dec if h15_dec > 0 else 0.0
    sol_save = (h15_solver - pol_solver) / h15_solver if h15_solver > 0 else 0.0
    phys_gate = phys_delta <= tol_sum
    return {
        "policy": policy,
        "rows": len(rows),
        "chosen_counts": dict(counts),
        "physical_delta_vs_H15": phys_delta,
        "physical_tolerance_vs_H15": tol_sum,
        "physical_gate_vs_H15": phys_gate,
        "decision_sum_s": pol_dec,
        "h15_decision_sum_s": h15_dec,
        "decision_relative_saving_vs_H15": dec_save,
        "solver_relative_saving_vs_H15": sol_save,
        "catastrophic_false_positive_count": bad,
        "chosen_bad_rows": chosen_bad_rows,
        "pass5_zero_cat_physical": bool(bad == 0 and phys_gate and dec_save >= MIN_SAVE),
    }


def dataset_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cat_rows = [r for r in rows if r.get("h12_catastrophic_vs_h15")]
    ben_rows = [r for r in rows if r.get("h12_beneficial_vs_h15")]
    nonben_safe = [r for r in rows if (not r.get("h12_catastrophic_vs_h15")) and (not r.get("h12_beneficial_vs_h15"))]
    by_role: Dict[str, Dict[str, Any]] = {}
    for role in sorted({str(r.get("role")) for r in rows}):
        rr = [r for r in rows if str(r.get("role")) == role]
        by_role[role] = {
            "n": len(rr),
            "h12_beneficial": sum(1 for r in rr if r.get("h12_beneficial_vs_h15")),
            "h12_catastrophic": sum(1 for r in rr if r.get("h12_catastrophic_vs_h15")),
            "mean_decision_gain_s": mean([sf(r.get("decision_gain_h12_vs_h15_s")) for r in rr]),
            "mean_phys_delta": mean([sf(r.get("phys_delta_h12_minus_h15")) for r in rr]),
        }
    return {
        "rows": len(rows),
        "h12_beneficial_count": len(ben_rows),
        "h12_catastrophic_count": len(cat_rows),
        "safe_nonbeneficial_count": len(nonben_safe),
        "catastrophic_sources": dict(Counter(str(r.get("source_key")) for r in cat_rows)),
        "catastrophic_roles": dict(Counter(str(r.get("role")) for r in cat_rows)),
        "safe_nonbeneficial_rows": [{"dataset": r.get("dataset"), "row_id": r.get("row_id"), "role": r.get("role"), "decision_gain_s": r.get("decision_gain_h12_vs_h15_s"), "phys_delta": r.get("phys_delta_h12_minus_h15")} for r in nonben_safe],
        "decision_gain_summary_s": finite_summary([sf(r.get("decision_gain_h12_vs_h15_s")) for r in rows]),
        "physical_delta_summary": finite_summary([sf(r.get("phys_delta_h12_minus_h15")) for r in rows]),
        "by_role": by_role,
        "fixed_H12": evaluate_fixed_or_oracle(rows, "fixed_H12"),
        "oracle_H12_H15": evaluate_fixed_or_oracle(rows, "oracle_H12_H15"),
    }


def extract_completed_headlines() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, p in [("v19", V19_DONE), ("v20b", V20B_DONE), ("v21", V21_DONE), ("v22", V22_DONE), ("v23", V23_DONE)]:
        obj = read_json(p)
        out[name] = obj.get("headline") or {k: obj.get(k) for k in ("status", "hard_pass", "created_utc")}
    return out


def choose_next_decision(summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    fresh = summaries["v21_plus_v23_source_independent"]
    v19 = summaries["v19_opened_boundary"]
    allsum = summaries["all_opened_h12_h15_rows"]
    fixed_fresh = fresh["fixed_H12"]
    fixed_v19 = v19["fixed_H12"]
    oracle_all = allsum["oracle_H12_H15"]
    if fixed_fresh["pass5_zero_cat_physical"] and fixed_v19["catastrophic_false_positive_count"] > 0 and oracle_all["pass5_zero_cat_physical"]:
        decision = "Fixed H12 dominates the fresh source-independent stress-pool rows, while adaptivity is only required by the opened v19 lower-stress mid-late negative cluster. The next experiment should not be another broad fresh support sweep; it should specifically test whether v19-like H12-risk states reproduce in source-independent neighbors selected pre-outcome from H15 traces."
        next_action = "After backup, freeze/run v25 H12-risk-family acquisition: metadata/H15-trace-only selection of lower-stress source72-neighborhood candidates not previously used, branch states matched to the v19/v11 mid-late negative morphology, with fixed H12/H15 and the v20b/v22 selector measured. If no fresh negatives appear, treat fixed H12 as the stronger simple baseline for this stress distribution and pivot to scenario design/fixed-H confirmation; if negatives appear and selector fails, pivot to terminal-risk/value refit/training."
    elif fixed_fresh["pass5_zero_cat_physical"] and fixed_v19["catastrophic_false_positive_count"] == 0:
        decision = "No current evidence that adaptivity is needed over fixed H12 on these rows; plan fixed-H12-primary confirmation or scenario-opportunity redesign."
        next_action = "Freeze a fixed-H12-primary confirmation/scenario-opportunity protocol, not a selector validation."
    else:
        decision = "H12 is not robust even in fresh rows; selector/training/value refit should take priority."
        next_action = "Freeze a bounded terminal-risk/value-refit or training ablation before more scenario sweeps."
    return {"decision": decision, "next_action": next_action}


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    s = raw["summaries"]
    lines = [
        "# Vehicle true-variable-H H12/H15 adaptivity-opportunity audit v24",
        "",
        f"UTC: `{raw['created_utc']}`. Offline development diagnostic only; no simulations, no validation64, no sealed test, no training/refit.",
        "",
        "## Headline",
        "",
        f"- Fresh source-independent rows (v21+v23): fixed H12 pass5 `{s['v21_plus_v23_source_independent']['fixed_H12']['pass5_zero_cat_physical']}`, bad `{s['v21_plus_v23_source_independent']['fixed_H12']['catastrophic_false_positive_count']}`, decision saving `{pct(s['v21_plus_v23_source_independent']['fixed_H12']['decision_relative_saving_vs_H15'])}`; oracle saving `{pct(s['v21_plus_v23_source_independent']['oracle_H12_H15']['decision_relative_saving_vs_H15'])}`.",
        f"- v19 opened boundary rows: fixed H12 pass5 `{s['v19_opened_boundary']['fixed_H12']['pass5_zero_cat_physical']}`, bad `{s['v19_opened_boundary']['fixed_H12']['catastrophic_false_positive_count']}`, decision saving `{pct(s['v19_opened_boundary']['fixed_H12']['decision_relative_saving_vs_H15'])}`; oracle pass5 `{s['v19_opened_boundary']['oracle_H12_H15']['pass5_zero_cat_physical']}`.",
        f"- All opened H12/H15 rows combined: fixed H12 bad `{s['all_opened_h12_h15_rows']['fixed_H12']['catastrophic_false_positive_count']}`, pass5 `{s['all_opened_h12_h15_rows']['fixed_H12']['pass5_zero_cat_physical']}`; oracle pass5 `{s['all_opened_h12_h15_rows']['oracle_H12_H15']['pass5_zero_cat_physical']}`, oracle saving `{pct(s['all_opened_h12_h15_rows']['oracle_H12_H15']['decision_relative_saving_vs_H15'])}`.",
        f"- Decision: {raw['decision']['decision']}",
        "",
        "## Dataset table",
        "",
        "| dataset | rows | H12 beneficial | H12 catastrophic | safe non-beneficial | fixed H12 save | fixed H12 bad/pass | oracle save/pass |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for key in ["v19_opened_boundary", "v21_source_independent", "v23_broader_source_independent", "v21_plus_v23_source_independent", "all_opened_h12_h15_rows"]:
        d = s[key]; fx = d["fixed_H12"]; oracle = d["oracle_H12_H15"]
        lines.append(f"| `{key}` | {d['rows']} | {d['h12_beneficial_count']} | {d['h12_catastrophic_count']} | {d['safe_nonbeneficial_count']} | {pct(fx['decision_relative_saving_vs_H15'])} | {fx['catastrophic_false_positive_count']}/{fx['pass5_zero_cat_physical']} | {pct(oracle['decision_relative_saving_vs_H15'])}/{oracle['pass5_zero_cat_physical']} |")
    lines += [
        "",
        "## H12-negative concentration",
        "",
        "```json",
        json.dumps(clean({k: s[k]["catastrophic_sources"] for k in ["v19_opened_boundary", "v21_source_independent", "v23_broader_source_independent", "v21_plus_v23_source_independent"]}), indent=2, sort_keys=True),
        "```",
        "",
        "Interpretation: v21/v23 deliberately sampled fresh source-independent stress-bank neighbors and support regions, yet all H12 catastrophic rows remain concentrated in the already-opened v19 boundary cluster. This makes the present fresh stress-pool evidence a strong fixed-H12 result, not yet an adaptive-H result.",
        "",
        "## Evidence table for next causal decision",
        "",
        "| axis | verified finding | competing hypothesis / missing evidence | discriminating next experiment |",
        "|---|---|---|---|",
        f"| Scenarios/opportunity | v21+v23 have {s['v21_plus_v23_source_independent']['h12_catastrophic_count']} H12 catastrophes and {s['v21_plus_v23_source_independent']['h12_beneficial_count']}/{s['v21_plus_v23_source_independent']['rows']} beneficial H12 rows; fixed H12 beats the selector on v23. | H12-risk states may be rare/idiosyncratic to v19 source72 mid-late, or the current source-independent stress pool may under-sample the true adaptive-opportunity boundary. | v25 pre-outcome source72-neighborhood H12-risk-family acquisition from H15 traces only; include fixed H12/H15 and v20b/v22 selector. |",
        "| Comparisons | Fixed H12 is now the primary simple baseline on v21/v23-like states; comparing only to H15 overstates adaptive value. | Adaptivity may still matter only when v19-like H12-risk states are present. | Any confirmation must report fixed H12 first and only claim adaptive value if it handles fresh H12-risk states that fixed H12 fails. |",
        "| Training/value | Current selector is static/offline and often conservatively chooses H15 on safe fresh rows, losing about half of fixed-H12 savings in v23. | If fresh risk-family negatives exist, static features may not separate them; terminal-risk/value refit may be needed. | If v25 produces false positives/missed opportunities, freeze bounded terminal-risk/value refit or training ablation; otherwise do not train merely to force switching. |",
        "| Reward/terminal | All current H12/H15 rows use shared H15 terminal for branch fairness; earlier terminal audits showed severe terminal-profile transfer risk for H10. | H12 risk may still be terminal/profile-specific; not yet separated for fresh H12 risk-family states. | v25 should keep shared-H15 primary and, if failures recur, add a small matched-terminal counterfactual before attributing to policy learning. |",
        "",
        "## Next action after backup",
        "",
        raw["decision"]["next_action"],
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    fresh = raw["summaries"]["v21_plus_v23_source_independent"]
    v19 = raw["summaries"]["v19_opened_boundary"]
    allsum = raw["summaries"]["all_opened_h12_h15_rows"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle H12/H15 adaptivity-opportunity audit v24

UTC: {raw['created_utc']}. Offline/no-simulation diagnostic over already-opened v19/v21/v23 H12/H15 development rows; no validation64, no sealed test, no training/refit. Fresh source-independent v21+v23 rows: fixed H12 bad={fresh['fixed_H12']['catastrophic_false_positive_count']}, pass5={fresh['fixed_H12']['pass5_zero_cat_physical']}, decision saving={fresh['fixed_H12']['decision_relative_saving_vs_H15']}; v19 opened-boundary rows: fixed H12 bad={v19['fixed_H12']['catastrophic_false_positive_count']}, pass5={v19['fixed_H12']['pass5_zero_cat_physical']}, oracle pass5={v19['oracle_H12_H15']['pass5_zero_cat_physical']}; all opened rows: fixed H12 bad={allsum['fixed_H12']['catastrophic_false_positive_count']}, oracle pass5={allsum['oracle_H12_H15']['pass5_zero_cat_physical']}. Decision: {raw['decision']['decision']} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_if_missing(ROOT / name, MARKER, block)

    response = f"""
## Follow-up through v24 H12/H15 adaptivity-opportunity audit

Updated by GPT-5.5 executor at `{raw['created_utc']}`. Stable Astra IDs preserved. v24 is offline development analysis of v19/v21/v23 only; validation64 and sealed test remained closed.

| linked recommendation(s) | disposition after v24 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; fixed H12 is now the primary comparator on fresh stress-pool rows | v24 summary: v21+v23 fixed H12 bad={fresh['fixed_H12']['catastrophic_false_positive_count']}, pass5={fresh['fixed_H12']['pass5_zero_cat_physical']}, save={pct(fresh['fixed_H12']['decision_relative_saving_vs_H15'])}; v19 fixed H12 bad={v19['fixed_H12']['catastrophic_false_positive_count']} and fails; all-row oracle H12/H15 pass5={allsum['oracle_H12_H15']['pass5_zero_cat_physical']}. | No adaptive validation until a fresh protocol shows adaptive value beyond fixed H12. Next diagnostic should target v19-like H12-risk-family reproduction with fixed H12 as the first baseline. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v24 shows v21/v23 zero-cat evidence is source-independent but still stress-pool/development; all H12 catastrophes remain concentrated in opened v19 boundary rows. | Treat v21/v23 as evidence that broad stress-pool support favors fixed H12, not as a population guarantee. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred for one more discriminating scenario diagnostic | The current bottleneck is ambiguous: no fresh H12 negatives in v21/v23, but v19 has a reproducible negative cluster. Static selector conservatism loses fixed-H12 savings on v23. | After backup, freeze/run v25 H12-risk-family acquisition. If fresh negatives recur and static selector fails, pivot to bounded terminal-risk/value refit or training; if not, prioritize scenario/comparison design and fixed-H12-primary confirmation. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; unchanged by v24 | v24 performs no new timing; it relies on v22/v23 overhead evidence and row-level branch timings. | Continue to require full closed-loop whole-decision timing for any final speed claim. |
| `A12_registry_backup_schema_contract` | accepted; active | v24 wrote new source/results/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified external backup covering v24 before v25 or other unique science. |
"""
    append_if_missing(RESPONSE_LOG, MARKER, response)


def update_registry(raw: Mapping[str, Any]) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    if path.exists() and MARKER in path.read_text(encoding="utf-8", errors="replace"):
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([
            f"{STAMP}_v24_offline_audit",
            raw["created_utc"],
            "IMPROVED_offline_H12_H15_adaptivity_opportunity_audit_no_sim_no_validation_no_test",
            "deterministic_existing_rows",
            "development_existing_v19_v21_v23_only",
            raw.get("supervisor_backup", {}).get("commit", "unknown"),
            "complete",
            0,
            0,
            "NA",
            rel(RUN_DIR / "completed.json") + " # " + MARKER,
        ])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ack-v23-backup", action="store_true", help="acknowledge supervisor-verified post-v23 backup context")
    ap.add_argument("--backup-commit", default="8c2bbefa69d20852b24190bae50017e816aa8a00")
    ap.add_argument("--backup-package-sha256", default="371ed4c2348a7bc795dda05c775dd82ff6ac042e88fed9150b326174d9b16107")
    args = ap.parse_args(argv)
    if not args.run or not args.ack_v23_backup:
        raise ContractError("requires --run and --ack-v23-backup")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / "completed.json").exists():
        done = read_json(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "sealed_test_accessed": False, "validation64_bank_opened": False}, sort_keys=True))
        return 0

    for label, p in [("v19", V19_DONE), ("v20b", V20B_DONE), ("v21", V21_DONE), ("v22", V22_DONE), ("v23", V23_DONE)]:
        assert_completed_dev(p, label)
    input_paths = [SOURCE, V19_RAW, V19_DONE, V20B_DONE, V21_RAW, V21_DONE, V22_DONE, V22_RAW, V23_RAW, V23_DONE, RESPONSE_LOG]
    rows_by = load_rows()
    combined: Dict[str, List[Dict[str, Any]]] = {
        **rows_by,
        "v21_plus_v23_source_independent": rows_by["v21_source_independent"] + rows_by["v23_broader_source_independent"],
        "all_opened_h12_h15_rows": rows_by["v19_opened_boundary"] + rows_by["v21_source_independent"] + rows_by["v23_broader_source_independent"],
    }
    summaries = {k: dataset_summary(v) for k, v in combined.items()}
    decision = choose_next_decision(summaries)
    created = now_utc()
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_offline_H12_H15_adaptivity_opportunity_audit_no_sim_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_grid_evaluations": 0,
        "input_hashes": {rel(p): sha256(p) for p in input_paths if p.exists()},
        "supervisor_backup": {"acknowledged_verified_post_v23_backup": bool(args.ack_v23_backup), "commit": args.backup_commit, "package_sha256": args.backup_package_sha256, "status": "verified_per_supervisor_context_20260930T013344Z"},
        "completed_headlines": extract_completed_headlines(),
        "rows_by_dataset": rows_by,
        "summaries": summaries,
        "decision": decision,
        "interpretation_limits": ["offline analysis of opened development data only", "not a population estimate", "not validation64", "not sealed test", "not ORIGINAL SAC", "does not measure new closed-loop selector timing"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v24_h12_adaptivity_opportunity_audit",
        "created_utc": created.isoformat(),
        "backup_required_before_more_unique_science": True,
        "reason": "new offline diagnostic, docs/state/registry/response-log updates and next v25 decision",
        "must_cover": [rel(SOURCE), rel(RUN_DIR), rel(STATE_PATH), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
    })
    headline = {
        "fresh_fixed_H12_save": summaries["v21_plus_v23_source_independent"]["fixed_H12"]["decision_relative_saving_vs_H15"],
        "fresh_fixed_H12_bad": summaries["v21_plus_v23_source_independent"]["fixed_H12"]["catastrophic_false_positive_count"],
        "fresh_fixed_H12_pass5": summaries["v21_plus_v23_source_independent"]["fixed_H12"]["pass5_zero_cat_physical"],
        "v19_fixed_H12_bad": summaries["v19_opened_boundary"]["fixed_H12"]["catastrophic_false_positive_count"],
        "all_oracle_pass5": summaries["all_opened_h12_h15_rows"]["oracle_H12_H15"]["pass5_zero_cat_physical"],
        "decision": decision["decision"],
        "next_action": decision["next_action"],
    }
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "budget_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0},
        "headline": headline,
        "backup_request": rel(BACKUP_REQUEST),
        "hashes": {},
    }
    write_json(RUN_DIR / "completed.json", completed)
    raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(f"# Continue state after v24 H12/H15 adaptivity-opportunity audit\n\nUTC: {created.isoformat()}\n\nDecision: {decision['decision']}\n\nNext action after verified backup: {decision['next_action']}\n\nArtifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`. No validation64/sealed test.\n", encoding="utf-8")
    update_docs(raw)
    update_registry(raw)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE_PATH, BACKUP_REQUEST, RESPONSE_LOG]
    completed["hashes"] = {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": headline, "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
