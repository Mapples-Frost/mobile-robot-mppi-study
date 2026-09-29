#!/usr/bin/env python3
"""Offline H10/H15 risk-aware CV and terminal-profile residual diagnostic.

Development-only, no simulation, no training/refit and no validation/test access.

Purpose: after true variable-H oracle/risk-anchor evidence showed measured compute
savings but strong terminal-profile dependence and fixed-H15 absorption, test the
smallest more-informative alternative to the failed H10/H15/H25 selector:

  * Is there enough terminal-consistent H10-vs-H15 risk-aware opportunity to beat
    fixed true H15 using logged measured decision/solver time?
  * Do simple deployable features predict when H10 is safe enough, or is the
    opportunity still terminal/model dependent and/or absorbed by fixed H15?

Inputs are existing development branch results only:
  - true-variable-H oracle bank v0 raw.json
  - source-independent risk-anchor acquisition v0 raw.json
  - terminal-profile postdiagnostic v0 raw.json (for consistency checks)

This script deliberately performs no gradient update, no value refit, no rollout,
no validation64 read and no sealed-test read.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0"
STAMP = "20260929T0840Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ORACLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
RISK_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json"
RISK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json"
POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_profile_effect_postdiagnostic_v0_20260929T0835Z/raw.json"
POST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_profile_effect_postdiagnostic_v0_20260929T0835Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-h10-h15-residual-cv-diagnostic-v0-{STAMP}"
H10, H15 = "10", "15"
MIN_SAVE_VS_H15 = 0.05
MIN_SAVE_VS_H15_STRONG = 0.10


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def clean(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, Mapping):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    return v


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: " + rel(path))
    for k in ("sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(k) is True:
            raise ContractError("sealed test flag unexpectedly true in " + rel(path))
    return obj


def finite(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        i = (len(xs) - 1) * q
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return xs[lo] if lo == hi else xs[lo] * (hi - i) + xs[hi] * (i - lo)
    return {"n": len(xs), "min": xs[0], "median": pct(0.5), "mean": math.fsum(xs) / len(xs), "p95": pct(0.95), "max": xs[-1], "sum": math.fsum(xs)}


def parse_case(state_id: str, row: Mapping[str, Any]) -> int:
    if "case" in row and row.get("case") is not None:
        return si(row.get("case"), -1)
    m = re.search(r"case(\d+)", state_id)
    return int(m.group(1)) if m else -1


def role_family(row: Mapping[str, Any], source: str) -> str:
    text = str(row.get("target_role") or row.get("stratum") or row.get("selection_group") or source)
    if "risk_anchor" in text and "control" not in text:
        return "risk_anchor"
    if "control" in text:
        return "control"
    if "primary" in text:
        return "primary"
    if "previously_used" in text:
        return "previously_used_anchor"
    return text[:64]


def safe(m: Mapping[str, Any]) -> bool:
    if "safe_all" in m:
        return bool(m.get("safe_all"))
    if m.get("constraint_any") is True:
        return False
    if m.get("success_all") is False:
        return False
    return m.get("physical") is not None


def label_h10_h15(m10: Mapping[str, Any], m15: Mapping[str, Any]) -> int:
    h15_phys = sf(m15.get("physical"))
    tol = max(2.0, 0.05 * abs(h15_phys))
    h10_dec = sf(m10.get("decision_sum_s")); h15_dec = sf(m15.get("decision_sum_s"))
    save = (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else 0.0
    if safe(m10) and safe(m15) and sf(m10.get("physical")) <= h15_phys + tol and save >= MIN_SAVE_VS_H15:
        return 10
    return 15


def extract_obs(raw: Mapping[str, Any], source: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for e in raw.get("episodes") or []:
        sid = str(e.get("state_id"))
        obs = (((e.get("branch_reset") or {}).get("initial_observation_at_branch")) or [])
        if sid and sid not in out and isinstance(obs, list) and obs:
            out[f"{source}::{sid}"] = [sf(x) for x in obs]
    return out


def load_samples() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    for p in (ORACLE_RAW, ORACLE_DONE, RISK_RAW, RISK_DONE, POST_RAW, POST_DONE):
        if not p.exists():
            raise ContractError("missing input: " + rel(p))
    completed_ok(ORACLE_DONE); completed_ok(RISK_DONE); completed_ok(POST_DONE)
    raws = {"oracle_bank": read_json(ORACLE_RAW), "risk_anchor": read_json(RISK_RAW)}
    post = read_json(POST_RAW)
    samples: List[Dict[str, Any]] = []
    obs_by_key: Dict[str, List[float]] = {}
    for source, raw in raws.items():
        if raw.get("sealed_test_accessed") is True:
            raise ContractError("sealed test flag true in " + source)
        obs_by_key.update(extract_obs(raw, source))
    for source, raw in raws.items():
        rows = ((raw.get("analysis") or {}).get("state_profile_rows") or [])
        for row in rows:
            med = row.get("median_by_h") or {}
            if H10 not in med or H15 not in med:
                continue
            sid = str(row.get("state_id"))
            key = f"{source}::{sid}"
            m10, m15 = med[H10], med[H15]
            label = label_h10_h15(m10, m15)
            h15_phys = sf(m15.get("physical")); h15_dec = sf(m15.get("decision_sum_s")); h15_solver = sf(m15.get("solver_sum_s"))
            h10_phys = sf(m10.get("physical")); h10_dec = sf(m10.get("decision_sum_s")); h10_solver = sf(m10.get("solver_sum_s"))
            samples.append({
                "sample_id": f"{source}::{sid}::{row.get('terminal_profile')}",
                "state_key": key,
                "state_id": sid,
                "source": source,
                "case": parse_case(sid, row),
                "terminal_profile": str(row.get("terminal_profile")),
                "window": str(row.get("window") or "missing"),
                "selection_group": str(row.get("selection_group") or "missing"),
                "role_family": role_family(row, source),
                "label": label,
                "h10": {"physical": h10_phys, "decision": h10_dec, "solver": h10_solver, "safe": safe(m10)},
                "h15": {"physical": h15_phys, "decision": h15_dec, "solver": h15_solver, "safe": safe(m15)},
                "h10_minus_h15_physical": h10_phys - h15_phys,
                "h10_vs_h15_decision_saving": (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else None,
                "obs": obs_by_key.get(key),
            })
    return samples, {"postdiagnostic": post}


def group_label(rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 15
    h10_phys = math.fsum(sf(r["h10"]["physical"]) for r in rows)
    h15_phys = math.fsum(sf(r["h15"]["physical"]) for r in rows)
    h10_dec = math.fsum(sf(r["h10"]["decision"]) for r in rows)
    h15_dec = math.fsum(sf(r["h15"]["decision"]) for r in rows)
    tol = max(2.0 * len(rows), 0.05 * abs(h15_phys))
    save = (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else 0.0
    if all(bool(r["h10"]["safe"]) for r in rows) and h10_phys - h15_phys <= tol and save >= MIN_SAVE_VS_H15:
        return 10
    return 15


def eval_policy(rows: Sequence[Mapping[str, Any]], preds: Sequence[int]) -> Dict[str, Any]:
    n = len(rows)
    h15_phys = math.fsum(sf(r["h15"]["physical"]) for r in rows)
    h15_dec = math.fsum(sf(r["h15"]["decision"]) for r in rows)
    h15_solver = math.fsum(sf(r["h15"]["solver"]) for r in rows)
    phys = dec = solver = 0.0
    unsafe = 0
    counts: Dict[str, int] = {}
    regrets: List[float] = []
    for r, p in zip(rows, preds):
        p = int(p)
        counts[str(p)] = counts.get(str(p), 0) + 1
        chosen = r["h10"] if p == 10 else r["h15"]
        phys += sf(chosen["physical"]); dec += sf(chosen["decision"]); solver += sf(chosen["solver"])
        if not bool(chosen["safe"]):
            unsafe += 1
        oracle_h = int(r["label"])
        oracle = r["h10"] if oracle_h == 10 else r["h15"]
        regrets.append(sf(chosen["physical"]) - sf(oracle["physical"]))
    tol = max(2.0 * n, 0.05 * abs(h15_phys)) if n else 0.0
    saving = (h15_dec - dec) / h15_dec if h15_dec > 0 else None
    solver_saving = (h15_solver - solver) / h15_solver if h15_solver > 0 else None
    return {
        "n": n,
        "horizon_counts": counts,
        "physical_sum": phys,
        "decision_sum_s": dec,
        "solver_sum_s": solver,
        "fixed_H15_physical_sum": h15_phys,
        "fixed_H15_decision_sum_s": h15_dec,
        "physical_delta_vs_H15": phys - h15_phys,
        "physical_tolerance_vs_H15": tol,
        "decision_relative_saving_vs_H15": saving,
        "solver_relative_saving_vs_H15": solver_saving,
        "unsafe_chosen_groups": unsafe,
        "oracle_regret_physical_sum": math.fsum(regrets),
        "oracle_regret_physical_summary": finite(regrets),
        "nonconstant": len(counts) > 1,
        "core_pass_5pct": bool(unsafe == 0 and phys - h15_phys <= tol and saving is not None and saving >= MIN_SAVE_VS_H15 and len(counts) > 1),
        "core_pass_10pct": bool(unsafe == 0 and phys - h15_phys <= tol and saving is not None and saving >= MIN_SAVE_VS_H15_STRONG and len(counts) > 1),
    }


def fit_category(train: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    default = group_label(train)
    values = sorted({str(r.get(field, "missing")) for r in train})
    rules = {}
    for v in values:
        part = [r for r in train if str(r.get(field, "missing")) == v]
        rules[v] = group_label(part)
    return {"field": field, "default": default, "rules": rules}


def pred_category(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    field = str(model["field"]); default = int(model["default"]); rules = model["rules"]
    return [int(rules.get(str(r.get(field, "missing")), default)) for r in rows]


def fit_obs_stump(train: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    default = group_label(train)
    obs_rows = [r for r in train if isinstance(r.get("obs"), list) and r.get("obs")]
    if len(obs_rows) < 4:
        return {"kind": "default", "default": default}
    d = min(len(r["obs"]) for r in obs_rows)
    best = None
    for j in range(d):
        xs = sorted({sf(r["obs"][j]) for r in obs_rows})
        if len(xs) < 2:
            continue
        qs = [xs[int((len(xs) - 1) * q)] for q in (0.25, 0.5, 0.75)]
        for thr in sorted(set(qs)):
            le = [r for r in train if isinstance(r.get("obs"), list) and len(r["obs"]) > j and sf(r["obs"][j]) <= thr]
            gt = [r for r in train if not (isinstance(r.get("obs"), list) and len(r["obs"]) > j and sf(r["obs"][j]) <= thr)]
            model = {"kind": "stump", "feature": j, "threshold": thr, "le": group_label(le), "gt": group_label(gt), "default": default}
            preds = pred_obs_stump(model, train)
            ev = eval_policy(train, preds)
            key = (1 if ev["core_pass_5pct"] else 0, sf(ev["decision_relative_saving_vs_H15"), -sf(ev["physical_delta_vs_H15"]), -sf(ev["oracle_regret_physical_sum"]))
            if best is None or key > best[0]:
                best = (key, model, ev)
    return best[1] if best else {"kind": "default", "default": default}


def pred_obs_stump(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    if model.get("kind") != "stump":
        return [int(model.get("default", 15)) for _ in rows]
    j = int(model["feature"]); thr = sf(model["threshold"])
    out = []
    for r in rows:
        obs = r.get("obs")
        if isinstance(obs, list) and len(obs) > j and sf(obs[j]) <= thr:
            out.append(int(model["le"]))
        else:
            out.append(int(model["gt"]))
    return out


def cross_validate(rows: Sequence[Mapping[str, Any]], fold_key: str, policy: str) -> Dict[str, Any]:
    preds_by_id: Dict[str, int] = {}
    models = []
    folds = sorted({str(r.get(fold_key)) for r in rows})
    for f in folds:
        train = [r for r in rows if str(r.get(fold_key)) != f]
        test = [r for r in rows if str(r.get(fold_key)) == f]
        if not train or not test:
            continue
        if policy.startswith("cat:"):
            model = fit_category(train, policy.split(":", 1)[1])
            preds = pred_category(model, test)
        elif policy == "obs_stump":
            model = fit_obs_stump(train)
            preds = pred_obs_stump(model, test)
        else:
            raise ValueError(policy)
        models.append({"fold": f, "model": model, "test_n": len(test)})
        for r, p in zip(test, preds):
            preds_by_id[str(r["sample_id"])] = int(p)
    ordered = [r for r in rows if str(r["sample_id"]) in preds_by_id]
    preds = [preds_by_id[str(r["sample_id"])] for r in ordered]
    ev = eval_policy(ordered, preds)
    ev["fold_key"] = fold_key; ev["folds"] = len(folds); ev["models"] = models[:20]
    return ev


def dataset_variants(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_state: Dict[str, List[Dict[str, Any]]] = {}
    for r in samples:
        by_state.setdefault(str(r["state_key"]), []).append(r)
    agreement_keys = set()
    flips = []
    for k, rows in by_state.items():
        labs = {r["terminal_profile"]: int(r["label"]) for r in rows}
        if len(set(labs.values())) <= 1:
            agreement_keys.add(k)
        else:
            flips.append({"state_key": k, "labels": labs})
    return {
        "all_profiles": list(samples),
        "matched_terminal_only": [r for r in samples if r["terminal_profile"] == "matched_terminal"],
        "shared_h15_terminal_only": [r for r in samples if r["terminal_profile"] == "shared_h15_terminal"],
        "terminal_agreement_states_all_profiles": [r for r in samples if r["state_key"] in agreement_keys],
    }, flips


def analyze_dataset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows), "state_count": len({r["state_key"] for r in rows}), "case_count": len({r["case"] for r in rows})}
    out["label_counts"] = {"10": sum(1 for r in rows if r["label"] == 10), "15": sum(1 for r in rows if r["label"] == 15)}
    out["h10_minus_h15_physical_summary"] = finite([r["h10_minus_h15_physical"] for r in rows])
    out["h10_vs_h15_decision_saving_summary"] = finite([r["h10_vs_h15_decision_saving"] for r in rows if r["h10_vs_h15_decision_saving"] is not None])
    baselines = {
        "fixed_H15": eval_policy(rows, [15] * len(rows)),
        "fixed_H10": eval_policy(rows, [10] * len(rows)),
        "oracle_H10H15": eval_policy(rows, [int(r["label"]) for r in rows]),
    }
    # fixed H15 is allowed to be constant, so do not use nonconstant core gate for oracle opportunity.
    oracle = baselines["oracle_H10H15"]
    oracle["opportunity_vs_H15_5pct"] = bool(oracle["unsafe_chosen_groups"] == 0 and oracle["physical_delta_vs_H15"] <= oracle["physical_tolerance_vs_H15"] and sf(oracle["decision_relative_saving_vs_H15"]) >= MIN_SAVE_VS_H15)
    oracle["opportunity_vs_H15_10pct"] = bool(oracle["unsafe_chosen_groups"] == 0 and oracle["physical_delta_vs_H15"] <= oracle["physical_tolerance_vs_H15"] and sf(oracle["decision_relative_saving_vs_H15"]) >= MIN_SAVE_VS_H15_STRONG)
    out["baselines"] = baselines
    policies = ["cat:terminal_profile", "cat:window", "cat:selection_group", "cat:role_family", "obs_stump"]
    cv: Dict[str, Any] = {}
    for pol in policies:
        cv[pol] = {
            "leave_state_out": cross_validate(rows, "state_key", pol) if out["state_count"] >= 3 else {"skipped": "too_few_states"},
            "leave_case_out": cross_validate(rows, "case", pol) if out["case_count"] >= 3 else {"skipped": "too_few_cases"},
        }
    out["cv"] = cv
    passes = []
    for pol, res in cv.items():
        ls = res.get("leave_state_out", {}); lc = res.get("leave_case_out", {})
        if ls.get("core_pass_5pct") and lc.get("core_pass_5pct"):
            deployable = pol in ("cat:window", "cat:selection_group", "obs_stump")
            passes.append({"policy": pol, "deployable_proxy": deployable, "strong_10pct": bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct")), "leave_state": {k: ls.get(k) for k in ("horizon_counts", "physical_delta_vs_H15", "decision_relative_saving_vs_H15", "unsafe_chosen_groups")}, "leave_case": {k: lc.get(k) for k in ("horizon_counts", "physical_delta_vs_H15", "decision_relative_saving_vs_H15", "unsafe_chosen_groups")}})
    out["cv_core_passes"] = passes
    return out


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    done_path = OUT_DIR / "completed.json"
    if done_path.exists():
        done = read_json(done_path)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    samples, aux = load_samples()
    variants, flips = dataset_variants(samples)
    results = {name: analyze_dataset(rows) for name, rows in variants.items() if rows}
    deployable_passes = []
    for dname, res in results.items():
        for p in res.get("cv_core_passes", []):
            if p.get("deployable_proxy"):
                deployable_passes.append({"dataset": dname, **p})
    oracle_opportunities = {k: {"opp5": v["baselines"]["oracle_H10H15"].get("opportunity_vs_H15_5pct"), "opp10": v["baselines"]["oracle_H10H15"].get("opportunity_vs_H15_10pct"), "decision_saving": v["baselines"]["oracle_H10H15"].get("decision_relative_saving_vs_H15"), "physical_delta": v["baselines"]["oracle_H10H15"].get("physical_delta_vs_H15"), "label_counts": v.get("label_counts")} for k, v in results.items()}
    terminal_flip_rate = len(flips) / float(len({r["state_key"] for r in samples}) or 1)
    if deployable_passes:
        decision = "freeze a tiny development rollout protocol only after backup, because at least one deployable-proxy H10/H15 CV rule beats fixed H15; include selector overhead and fixed H10/H15/H25 baselines"
    elif any(v["opp5"] for v in oracle_opportunities.values()):
        decision = "do not rollout/refit yet: oracle H10/H15 opportunity exists against fixed H15, but simple deployable CV did not pass; prioritize terminal-value calibration or richer supervised/value representation before any closed-loop selector"
    else:
        decision = "fixed H15 absorbs the H10/H15 tradeoff in the available development bank; pivot to terminal/modeling or scenario-design intervention rather than selector scaling"
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_H10_H15_RESIDUAL_CV_DIAGNOSTIC_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_h10_h15_terminal_residual_cv_not_validation_not_final_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_pool_resets": 0,
        "inputs": {"oracle_raw": rel(ORACLE_RAW), "risk_raw": rel(RISK_RAW), "terminal_postdiagnostic_raw": rel(POST_RAW)},
        "sample_count": len(samples),
        "state_count": len({r["state_key"] for r in samples}),
        "terminal_h10_h15_label_flips": flips,
        "terminal_h10_h15_label_flip_rate": terminal_flip_rate,
        "datasets": results,
        "oracle_opportunities_vs_fixed_H15": oracle_opportunities,
        "deployable_cv_passes": deployable_passes,
        "decision": decision,
        "backup_request_after_diagnostic": rel(req),
        "interpretation_limits": ["development-only", "offline replay of logged branch outcomes", "no selector overhead measured", "no validation64 or sealed test", "not ORIGINAL reproduction"],
        "postdiagnostic_headline_reused": (aux["postdiagnostic"].get("label_flip_counts") or {}),
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle true-variable-H H10/H15 residual/CV diagnostic v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Samples: `{len(samples)}` profile-rows from oracle-bank + risk-anchor development branches; states: `{raw['state_count']}`.",
        f"- H10/H15 label flips across terminal profiles: `{len(flips)}` states; flip rate `{terminal_flip_rate:.3f}`.",
        f"- Deployable-proxy CV passes vs fixed H15: `{deployable_passes}`.",
        f"- Decision: {decision}.",
        "",
        "## Oracle H10/H15 opportunity against fixed H15",
        "",
        "| dataset | labels | physΔ vs H15 | decision saving vs H15 | opp>=5% | opp>=10% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d, o in oracle_opportunities.items():
        lines.append("| `%s` | `%s` | %.6g | %.6g | `%s` | `%s` |" % (d, o.get("label_counts"), sf(o.get("physical_delta")), sf(o.get("decision_saving")), o.get("opp5"), o.get("opp10")))
    lines += ["", "## CV core passes", ""]
    if deployable_passes:
        for p in deployable_passes:
            lines.append(f"- `{p['dataset']}` / `{p['policy']}`: {p}")
    else:
        lines.append("No deployable-proxy H10/H15 rule passed both leave-state-out and leave-case-out gates against fixed H15.")
    lines += [
        "",
        "## Interpretation",
        "",
        "This diagnostic uses measured logged decision/solver times, not horizon-length proxies. It tests a lower-ambition H10/H15 risk-aware selector because H25 labels were terminal-profile dependent and fixed H15 was the main comparator threat. A pass here is still not validation evidence and would only justify a separately frozen development rollout with selector overhead; a failure supports terminal-value calibration/refit or scenario/modeling work rather than scaling the current supervised selector.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text((OUT_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup offline H10/H15 residual/CV diagnostic before any rollout, simulation, training or refit", "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(req)]})
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H H10/H15 residual/CV diagnostic v0

UTC: {created.isoformat()}. Offline/no-simulation diagnostic parsed existing oracle-bank and risk-anchor branch outputs. Validation64 and sealed test stayed closed; no training/refit. Samples={len(samples)}, states={raw['state_count']}, terminal H10/H15 label flips={len(flips)}. Deployable-proxy CV passes vs fixed H15: {deployable_passes}. Decision: {decision}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE_PATH, req, ORACLE_RAW, RISK_RAW, POST_RAW]
    done = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0, "headline": {"sample_count": len(samples), "state_count": raw["state_count"], "terminal_h10_h15_label_flipped_states": len(flips), "deployable_cv_pass_count": len(deployable_passes), "decision": decision}, "backup_request": rel(req), "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(OUT_DIR / "summary.md"), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
