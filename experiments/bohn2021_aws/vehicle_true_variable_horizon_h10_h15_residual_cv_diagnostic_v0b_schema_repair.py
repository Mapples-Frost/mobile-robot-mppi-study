#!/usr/bin/env python3
"""Syntax/schema-repaired offline H10/H15 residual-CV diagnostic (v0b).

Development-only.  No simulation, no training/refit, no validation64 read, and no
sealed-test read.  This replaces the failed v0 script, whose only execution
failed at import time with a syntax error before reading data.

Question: after H10/H15/H25 labels proved terminal-profile dependent and fixed
true H15 absorbed the H25 tradeoff on risk-anchor states, is there still a
lower-ambition deployable H10-vs-H15 risk-aware rule that beats fixed true H15
using already logged branch outcomes and measured decision/solver times?
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair"
STAMP = "20260929T0845Z"
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
MARKER = f"vehicle-true-variable-H-h10-h15-residual-cv-diagnostic-v0b-{STAMP}"
H10, H15 = "10", "15"
MIN_SAVE = 0.05
STRONG_SAVE = 0.10


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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


def si(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def finite(vals: Iterable[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in vals:
        try:
            y = float(v)
        except Exception:
            continue
        if math.isfinite(y):
            xs.append(y)
    xs.sort()
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}

    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {"n": len(xs), "min": xs[0], "median": pct(0.5), "mean": math.fsum(xs) / len(xs), "p95": pct(0.95), "max": xs[-1], "sum": math.fsum(xs)}


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"input did not pass: {rel(path)}")
    for key in ("sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(key) is True:
            raise ContractError(f"sealed test flag unexpectedly true in {rel(path)}")
    return obj


def parse_case(state_id: str, row: Mapping[str, Any]) -> int:
    if row.get("case") is not None:
        return si(row.get("case"), -1)
    match = re.search(r"case(\d+)", state_id)
    return int(match.group(1)) if match else -1


def role_family(row: Mapping[str, Any], source: str) -> str:
    text = str(row.get("target_role") or row.get("source_stratum") or row.get("selection_group") or source)
    if "risk_anchor" in text and "control" not in text:
        return "risk_anchor"
    if "control" in text or "negative" in text:
        return "control_or_negative"
    if "primary" in text or "high_trace" in text:
        return "primary"
    if "previously_used" in text:
        return "previously_used_anchor"
    return text[:64]


def safe(metrics: Mapping[str, Any]) -> bool:
    if metrics.get("safe_all") is not None:
        return bool(metrics.get("safe_all"))
    if metrics.get("constraint_any") is True:
        return False
    if metrics.get("success_all") is False:
        return False
    return metrics.get("physical") is not None


def h10_h15_label(m10: Mapping[str, Any], m15: Mapping[str, Any]) -> int:
    h15_phys = sf(m15.get("physical"))
    tol = max(2.0, 0.05 * abs(h15_phys))
    h10_dec, h15_dec = sf(m10.get("decision_sum_s")), sf(m15.get("decision_sum_s"))
    saving = (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else 0.0
    if safe(m10) and safe(m15) and sf(m10.get("physical")) <= h15_phys + tol and saving >= MIN_SAVE:
        return 10
    return 15


def extract_obs(raw: Mapping[str, Any], source: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for ep in raw.get("episodes") or []:
        sid = str(ep.get("state_id") or "")
        obs = ((ep.get("branch_reset") or {}).get("initial_observation_at_branch") or [])
        if sid and isinstance(obs, list) and obs and f"{source}::{sid}" not in out:
            out[f"{source}::{sid}"] = [sf(x) for x in obs]
    return out


def load_samples() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    for path in (ORACLE_RAW, ORACLE_DONE, RISK_RAW, RISK_DONE, POST_RAW, POST_DONE):
        if not path.exists():
            raise ContractError(f"missing input: {rel(path)}")
    completed_ok(ORACLE_DONE)
    completed_ok(RISK_DONE)
    completed_ok(POST_DONE)
    raw_by_source = {"oracle_bank": read_json(ORACLE_RAW), "risk_anchor": read_json(RISK_RAW)}
    post = read_json(POST_RAW)
    obs_by_key: Dict[str, List[float]] = {}
    for source, raw in raw_by_source.items():
        if raw.get("sealed_test_accessed") is True:
            raise ContractError(f"sealed test flag unexpectedly true in {source}")
        obs_by_key.update(extract_obs(raw, source))
    samples: List[Dict[str, Any]] = []
    for source, raw in raw_by_source.items():
        rows = ((raw.get("analysis") or {}).get("state_profile_rows") or [])
        for row in rows:
            med = row.get("median_by_h") or {}
            if H10 not in med or H15 not in med:
                continue
            sid = str(row.get("state_id") or "")
            term = str(row.get("terminal_profile") or "missing")
            key = f"{source}::{sid}"
            m10, m15 = med[H10], med[H15]
            h10_dec, h15_dec = sf(m10.get("decision_sum_s")), sf(m15.get("decision_sum_s"))
            samples.append({
                "sample_id": f"{source}::{sid}::{term}",
                "state_key": key,
                "state_id": sid,
                "source": source,
                "case": parse_case(sid, row),
                "terminal_profile": term,
                "window": str(row.get("window") or "missing"),
                "selection_group": str(row.get("selection_group") or row.get("source_stratum") or "missing"),
                "role_family": role_family(row, source),
                "target_role": str(row.get("target_role") or "missing"),
                "label": h10_h15_label(m10, m15),
                "h10": {"physical": sf(m10.get("physical")), "decision": h10_dec, "solver": sf(m10.get("solver_sum_s")), "safe": safe(m10)},
                "h15": {"physical": sf(m15.get("physical")), "decision": h15_dec, "solver": sf(m15.get("solver_sum_s")), "safe": safe(m15)},
                "h10_minus_h15_physical": sf(m10.get("physical")) - sf(m15.get("physical")),
                "h10_vs_h15_decision_saving": (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else None,
                "obs": obs_by_key.get(key),
            })
    return samples, {"terminal_postdiagnostic_raw": post}


def group_label(rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 15
    h10_phys = math.fsum(sf(r["h10"]["physical"]) for r in rows)
    h15_phys = math.fsum(sf(r["h15"]["physical"]) for r in rows)
    h10_dec = math.fsum(sf(r["h10"]["decision"]) for r in rows)
    h15_dec = math.fsum(sf(r["h15"]["decision"]) for r in rows)
    tol = max(2.0 * len(rows), 0.05 * abs(h15_phys))
    saving = (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else 0.0
    if all(bool(r["h10"]["safe"]) for r in rows) and h10_phys - h15_phys <= tol and saving >= MIN_SAVE:
        return 10
    return 15


def eval_policy(rows: Sequence[Mapping[str, Any]], preds: Sequence[int], require_nonconstant: bool = True) -> Dict[str, Any]:
    n = len(rows)
    h15_phys = math.fsum(sf(r["h15"]["physical"]) for r in rows)
    h15_dec = math.fsum(sf(r["h15"]["decision"]) for r in rows)
    h15_solver = math.fsum(sf(r["h15"]["solver"]) for r in rows)
    phys = dec = solver = 0.0
    unsafe = 0
    counts: Dict[str, int] = {}
    regret: List[float] = []
    for row, pred in zip(rows, preds):
        h = int(pred)
        counts[str(h)] = counts.get(str(h), 0) + 1
        chosen = row["h10"] if h == 10 else row["h15"]
        oracle = row["h10"] if int(row["label"]) == 10 else row["h15"]
        phys += sf(chosen["physical"])
        dec += sf(chosen["decision"])
        solver += sf(chosen["solver"])
        if not bool(chosen["safe"]):
            unsafe += 1
        regret.append(sf(chosen["physical"]) - sf(oracle["physical"]))
    tol = max(2.0 * n, 0.05 * abs(h15_phys)) if n else 0.0
    saving = (h15_dec - dec) / h15_dec if h15_dec > 0 else None
    solver_saving = (h15_solver - solver) / h15_solver if h15_solver > 0 else None
    nonconstant = len(counts) > 1
    admissible = unsafe == 0 and phys - h15_phys <= tol and saving is not None and saving >= MIN_SAVE
    return {
        "n": n,
        "horizon_counts": counts,
        "physical_sum": phys,
        "decision_sum_s": dec,
        "solver_sum_s": solver,
        "fixed_H15_physical_sum": h15_phys,
        "fixed_H15_decision_sum_s": h15_dec,
        "fixed_H15_solver_sum_s": h15_solver,
        "physical_delta_vs_H15": phys - h15_phys,
        "physical_tolerance_vs_H15": tol,
        "decision_relative_saving_vs_H15": saving,
        "solver_relative_saving_vs_H15": solver_saving,
        "unsafe_chosen_groups": unsafe,
        "oracle_regret_physical_sum": math.fsum(regret),
        "oracle_regret_physical_summary": finite(regret),
        "nonconstant": nonconstant,
        "core_pass_5pct": bool(admissible and (nonconstant or not require_nonconstant)),
        "core_pass_10pct": bool(admissible and saving is not None and saving >= STRONG_SAVE and (nonconstant or not require_nonconstant)),
    }


def fit_category(train: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    default = group_label(train)
    rules = {}
    for value in sorted({str(r.get(field, "missing")) for r in train}):
        part = [r for r in train if str(r.get(field, "missing")) == value]
        rules[value] = group_label(part)
    return {"kind": "category", "field": field, "default": default, "rules": rules}


def pred_category(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    field = str(model["field"])
    default = int(model["default"])
    rules = model.get("rules") or {}
    return [int(rules.get(str(r.get(field, "missing")), default)) for r in rows]


def fit_obs_stump(train: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    default = group_label(train)
    with_obs = [r for r in train if isinstance(r.get("obs"), list) and r.get("obs")]
    if len(with_obs) < 4:
        return {"kind": "default", "default": default, "reason": "too_few_obs"}
    dim = min(len(r["obs"]) for r in with_obs)
    best: Tuple[Tuple[float, float, float, float], Dict[str, Any]] | None = None
    for j in range(dim):
        xs = sorted({sf(r["obs"][j]) for r in with_obs})
        if len(xs) < 2:
            continue
        thresholds = sorted({xs[int((len(xs) - 1) * q)] for q in (0.25, 0.5, 0.75)})
        for thr in thresholds:
            le = [r for r in train if isinstance(r.get("obs"), list) and len(r["obs"]) > j and sf(r["obs"][j]) <= thr]
            gt = [r for r in train if r not in le]
            model = {"kind": "stump", "feature": j, "threshold": thr, "le": group_label(le), "gt": group_label(gt), "default": default}
            ev = eval_policy(train, pred_obs_stump(model, train), require_nonconstant=True)
            key = (
                1.0 if ev.get("core_pass_5pct") else 0.0,
                sf(ev.get("decision_relative_saving_vs_H15")),
                -sf(ev.get("physical_delta_vs_H15")),
                -sf(ev.get("oracle_regret_physical_sum")),
            )
            if best is None or key > best[0]:
                best = (key, model)
    return best[1] if best is not None else {"kind": "default", "default": default, "reason": "no_split"}


def pred_obs_stump(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    if model.get("kind") != "stump":
        return [int(model.get("default", 15)) for _ in rows]
    j = int(model["feature"])
    thr = sf(model["threshold"])
    out: List[int] = []
    for r in rows:
        obs = r.get("obs")
        if isinstance(obs, list) and len(obs) > j and sf(obs[j]) <= thr:
            out.append(int(model["le"]))
        else:
            out.append(int(model["gt"]))
    return out


def cross_validate(rows: Sequence[Mapping[str, Any]], fold_key: str, policy: str) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    preds_by_id: Dict[str, int] = {}
    models = []
    for fold in folds:
        train = [r for r in rows if str(r.get(fold_key)) != fold]
        test = [r for r in rows if str(r.get(fold_key)) == fold]
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
        models.append({"fold": fold, "test_n": len(test), "model": model})
        for row, pred in zip(test, preds):
            preds_by_id[str(row["sample_id"])] = int(pred)
    ordered = [r for r in rows if str(r["sample_id"]) in preds_by_id]
    ev = eval_policy(ordered, [preds_by_id[str(r["sample_id"])] for r in ordered], require_nonconstant=True)
    ev.update({"fold_key": fold_key, "folds": len(folds), "models_preview": models[:24]})
    return ev


def variants(samples: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    by_state: Dict[str, List[Dict[str, Any]]] = {}
    for row in samples:
        by_state.setdefault(str(row["state_key"]), []).append(row)
    agreement = set()
    flips: List[Dict[str, Any]] = []
    for key, rows in by_state.items():
        labels = {str(r["terminal_profile"]): int(r["label"]) for r in rows}
        if len(set(labels.values())) <= 1:
            agreement.add(key)
        else:
            flips.append({"state_key": key, "state_id": rows[0].get("state_id"), "case": rows[0].get("case"), "source": rows[0].get("source"), "labels": labels})
    return {
        "all_profiles": list(samples),
        "matched_terminal_only": [r for r in samples if r["terminal_profile"] == "matched_terminal"],
        "shared_h15_terminal_only": [r for r in samples if r["terminal_profile"] == "shared_h15_terminal"],
        "terminal_agreement_states_all_profiles": [r for r in samples if r["state_key"] in agreement],
        "risk_anchor_source_all_profiles": [r for r in samples if r["source"] == "risk_anchor"],
        "risk_anchor_primary_rows": [r for r in samples if r["source"] == "risk_anchor" and r["role_family"] == "risk_anchor"],
    }, flips


def analyze_dataset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": len(rows),
        "state_count": len({r["state_key"] for r in rows}),
        "case_count": len({r["case"] for r in rows}),
        "label_counts": {"10": sum(1 for r in rows if int(r["label"]) == 10), "15": sum(1 for r in rows if int(r["label"]) == 15)},
        "h10_minus_h15_physical_summary": finite(r["h10_minus_h15_physical"] for r in rows),
        "h10_vs_h15_decision_saving_summary": finite(r["h10_vs_h15_decision_saving"] for r in rows),
    }
    baselines = {
        "fixed_H15": eval_policy(rows, [15] * len(rows), require_nonconstant=False),
        "fixed_H10": eval_policy(rows, [10] * len(rows), require_nonconstant=False),
        "oracle_H10H15": eval_policy(rows, [int(r["label"]) for r in rows], require_nonconstant=False),
    }
    baselines["oracle_H10H15"]["opportunity_vs_H15_5pct"] = bool(baselines["oracle_H10H15"].get("core_pass_5pct"))
    baselines["oracle_H10H15"]["opportunity_vs_H15_10pct"] = bool(baselines["oracle_H10H15"].get("core_pass_10pct"))
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
        ls = res.get("leave_state_out") or {}
        lc = res.get("leave_case_out") or {}
        if ls.get("core_pass_5pct") and lc.get("core_pass_5pct"):
            deployable = pol in ("cat:window", "cat:selection_group", "obs_stump")
            passes.append({
                "policy": pol,
                "deployable_proxy": deployable,
                "strong_10pct": bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct")),
                "leave_state": {k: ls.get(k) for k in ("horizon_counts", "physical_delta_vs_H15", "physical_tolerance_vs_H15", "decision_relative_saving_vs_H15", "solver_relative_saving_vs_H15", "unsafe_chosen_groups")},
                "leave_case": {k: lc.get(k) for k in ("horizon_counts", "physical_delta_vs_H15", "physical_tolerance_vs_H15", "decision_relative_saving_vs_H15", "solver_relative_saving_vs_H15", "unsafe_chosen_groups")},
            })
    out["cv_core_passes"] = passes
    return out


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    done_path = OUT_DIR / "completed.json"
    if done_path.exists():
        done = read_json(done_path)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    samples, aux = load_samples()
    data_variants, flips = variants(samples)
    results = {name: analyze_dataset(rows) for name, rows in data_variants.items() if rows}
    deployable_passes = []
    for dataset_name, res in results.items():
        for item in res.get("cv_core_passes", []):
            if item.get("deployable_proxy"):
                deployable_passes.append({"dataset": dataset_name, **item})
    oracle_opportunities = {
        name: {
            "label_counts": res.get("label_counts"),
            "opp5": res["baselines"]["oracle_H10H15"].get("opportunity_vs_H15_5pct"),
            "opp10": res["baselines"]["oracle_H10H15"].get("opportunity_vs_H15_10pct"),
            "physical_delta_vs_H15": res["baselines"]["oracle_H10H15"].get("physical_delta_vs_H15"),
            "decision_relative_saving_vs_H15": res["baselines"]["oracle_H10H15"].get("decision_relative_saving_vs_H15"),
            "solver_relative_saving_vs_H15": res["baselines"]["oracle_H10H15"].get("solver_relative_saving_vs_H15"),
        }
        for name, res in results.items()
    }
    state_count = len({r["state_key"] for r in samples})
    terminal_flip_rate = len(flips) / float(state_count or 1)
    if deployable_passes:
        decision = "after verified backup, freeze a tiny development rollout protocol with selector overhead and blocked timing against fixed true H10/H15/H25; do not use validation64/test"
    elif any(bool(v.get("opp5")) for v in oracle_opportunities.values()):
        decision = "do not rollout/refit current selector: H10/H15 oracle opportunity exists against fixed H15, but deployable leave-state/case CV did not pass; next intervention should be terminal-value/objective calibration or a richer representation with explicit fixed-H15 baseline"
    else:
        decision = "fixed H15 absorbs the available H10/H15 opportunity; pivot away from selector scaling to terminal/modeling/scenario-design intervention"
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_H10_H15_RESIDUAL_CV_DIAGNOSTIC_V0B_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_h10_h15_residual_cv_syntax_repair_not_validation_not_test",
        "replaces_failed_run": "research_artifacts/aws_runs/20260929T083604_51cb8db5/registry.json",
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
        "state_count": state_count,
        "terminal_h10_h15_label_flips": flips,
        "terminal_h10_h15_label_flip_rate": terminal_flip_rate,
        "datasets": results,
        "oracle_opportunities_vs_fixed_H15": oracle_opportunities,
        "deployable_cv_passes": deployable_passes,
        "decision": decision,
        "backup_request_after_diagnostic": rel(req),
        "postdiagnostic_headline_reused": (aux["terminal_postdiagnostic_raw"].get("headline") or aux["terminal_postdiagnostic_raw"].get("label_flip_counts") or {}),
        "interpretation_limits": ["development-only", "offline replay of logged branch outcomes", "no selector overhead measured", "no validation64 or sealed test", "IMPROVED only, not ORIGINAL"],
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle true-variable-H H10/H15 residual/CV diagnostic v0b",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation syntax-repair run; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Samples: `{len(samples)}` profile-rows from oracle-bank + risk-anchor development branches; states: `{state_count}`.",
        f"- H10/H15 label flips across terminal profiles: `{len(flips)}` states; flip rate `{terminal_flip_rate:.3f}`.",
        f"- Deployable-proxy CV passes vs fixed H15: `{deployable_passes}`.",
        f"- Decision: {decision}.",
        "",
        "## Oracle H10/H15 opportunity against fixed H15",
        "",
        "| dataset | labels | physΔ vs H15 | decision saving vs H15 | solver saving vs H15 | opp>=5% | opp>=10% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dname, opp in oracle_opportunities.items():
        lines.append("| `%s` | `%s` | %.6g | %.6g | %.6g | `%s` | `%s` |" % (
            dname,
            opp.get("label_counts"),
            sf(opp.get("physical_delta_vs_H15")),
            sf(opp.get("decision_relative_saving_vs_H15")),
            sf(opp.get("solver_relative_saving_vs_H15")),
            opp.get("opp5"),
            opp.get("opp10"),
        ))
    lines += ["", "## CV core passes", ""]
    if deployable_passes:
        for item in deployable_passes:
            lines.append(f"- `{item['dataset']}` / `{item['policy']}`: {item}")
    else:
        lines.append("No deployable-proxy H10/H15 rule passed both leave-state-out and leave-case-out gates against fixed H15.")
    lines += [
        "",
        "## Interpretation",
        "",
        "This diagnostic uses logged measured whole-decision and solver times, not horizon-length proxies. It tests a lower-ambition H10/H15 risk-aware selector because H25 labels were terminal-profile dependent and fixed H15 is the main comparator threat. A pass would only justify a separately frozen development rollout with selector overhead. A failure supports terminal-value/objective calibration or scenario/modeling work rather than scaling the current supervised selector.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup offline H10/H15 residual/CV diagnostic v0b before any rollout, simulation, training or refit",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(req)],
    })
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H H10/H15 residual/CV diagnostic v0b

UTC: {created.isoformat()}. Offline/no-simulation syntax-repaired diagnostic parsed existing oracle-bank and risk-anchor branch outputs. Validation64 and sealed test stayed closed; no training/refit. Samples={len(samples)}, states={state_count}, terminal H10/H15 label flips={len(flips)}. Deployable-proxy CV passes vs fixed H15: {deployable_passes}. Decision: {decision}. Artifacts: `{rel(summary_path)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE_PATH, req, ORACLE_RAW, RISK_RAW, POST_RAW]
    done = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_pool_resets": 0,
        "headline": {"sample_count": len(samples), "state_count": state_count, "terminal_h10_h15_label_flipped_states": len(flips), "deployable_cv_pass_count": len(deployable_passes), "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary_path), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
