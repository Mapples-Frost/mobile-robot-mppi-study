#!/usr/bin/env python3
"""Offline safety-gated robust-label selector CV diagnostic.

Development-only. No new simulation, no rollout, no validation64 read, no sealed
final-test read, no training/refit, and no gradient updates.

Hypothesis: the previous state-observable H10/H15 CV failures are not solely a
lack of variable-horizon opportunity. Robust terminal-independent labels retain
measured compute value, but the prior KNN/centroid selection objective tolerated
unsafe H10 false positives when inner folds had no fully passing model. A
safety-first high-precision H10 selector with abstention to fixed true H15 should
avoid catastrophic false positives if the currently recorded online state
representation contains enough information. Failure would point to inadequate
representation/state coverage or terminal/value calibration rather than another
unchanged label-density sweep.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_safety_gated_selector_cv_v0"
STAMP = "20260929T0940Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0B_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair.py"
ROBUST_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_robust_terminal_label_cv_v0.py"
ROBUST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_robust_terminal_label_cv_v0_20260929T0925Z/completed.json"
TERMINAL_CONSIST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_consistency_audit_v0_20260929T0915Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-safety-gated-selector-cv-v0-{STAMP}"
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


def clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return rel(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    return obj


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


def sf(value: Any, default: float = 0.0) -> float:
    try:
        y = float(value)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return default
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finite(values: Iterable[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        try:
            y = float(v)
        except Exception:
            continue
        if math.isfinite(y):
            xs.append(y)
    xs.sort()
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    return {"n": len(xs), "min": xs[0], "median": quantile(xs, 0.5), "mean": math.fsum(xs) / len(xs), "p95": quantile(xs, 0.95), "max": xs[-1], "sum": math.fsum(xs)}


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing prerequisite completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"prerequisite did not pass: {rel(path)}")
    if obj.get("validation64_bank_opened") is True or obj.get("sealed_test_accessed") is True:
        raise ContractError(f"unexpected validation/test access flag in {rel(path)}")
    return obj


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import {rel(path)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def obs(row: Mapping[str, Any]) -> Optional[List[float]]:
    value = row.get("obs")
    if not isinstance(value, list) or not value:
        return None
    out: List[float] = []
    for x in value:
        try:
            y = float(x)
        except Exception:
            return None
        if not math.isfinite(y):
            return None
        out.append(y)
    return out


def transform(xs: Sequence[float], mode: str, dim: int) -> List[float]:
    base = [float(xs[i]) if i < len(xs) else 0.0 for i in range(dim)]
    if mode == "raw":
        return base
    if mode == "raw_abs_l2":
        return base + [abs(x) for x in base] + [math.sqrt(math.fsum(x * x for x in base))]
    raise ValueError(mode)


def make_std(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    obs_rows = [o for o in (obs(r) for r in rows) if o]
    if not obs_rows:
        return {"mode": mode, "dim": 0, "center": [], "scale": []}
    dim = min(len(o) for o in obs_rows)
    feats = [transform(o, mode, dim) for o in obs_rows]
    fdim = min(len(f) for f in feats)
    center, scale = [], []
    for j in range(fdim):
        col = [f[j] for f in feats]
        med = quantile(col, 0.5)
        iqr = max(quantile(col, 0.75, med) - quantile(col, 0.25, med), 1e-6)
        center.append(med)
        scale.append(iqr)
    return {"mode": mode, "dim": dim, "center": center, "scale": scale}


def feat(row: Mapping[str, Any], std: Mapping[str, Any]) -> Optional[List[float]]:
    x = obs(row)
    if x is None:
        return None
    center = list(std.get("center") or [])
    scale = list(std.get("scale") or [])
    dim = int(std.get("dim") or 0)
    if not center or not scale or dim <= 0:
        return None
    raw = transform(x, str(std.get("mode")), dim)
    n = min(len(raw), len(center), len(scale))
    return [(raw[i] - float(center[i])) / max(float(scale[i]), 1e-6) for i in range(n)]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return float("inf")
    return math.sqrt(math.fsum((float(a[i]) - float(b[i])) ** 2 for i in range(n)) / n)


def candidate_configs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"family": "all_h15"}]
    # Predeclared compact high-precision H10 one-class/near-neighbour guard.
    # It is intentionally smaller than the previous broad 36-config sweep.
    for mode in ("raw", "raw_abs_l2"):
        for k in (1, 3):
            for pos_radius_q in (0.50, 0.75):
                for negative_margin in (1.00, 1.25):
                    out.append({
                        "family": "positive_support_knn_guard",
                        "mode": mode,
                        "k": k,
                        "min_positive_fraction": 1.0,
                        "min_positive_support": 2,
                        "positive_radius_quantile": pos_radius_q,
                        "negative_margin": negative_margin,
                    })
    return out


def fit_model(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if cfg.get("family") == "all_h15":
        return {"cfg": dict(cfg), "default": 15, "entries": [], "positive_count": 0, "negative_count": len(rows)}
    mode = str(cfg.get("mode", "raw"))
    std = make_std(rows, mode)
    entries = []
    for r in rows:
        z = feat(r, std)
        if z is not None:
            entries.append({"z": z, "label": int(r.get("label", 15)), "sample_id": str(r.get("sample_id"))})
    pos = [e for e in entries if int(e["label"]) == 10]
    neg = [e for e in entries if int(e["label"]) != 10]
    pos_nn: List[float] = []
    for i, e in enumerate(pos):
        others = [dist(e["z"], p["z"]) for j, p in enumerate(pos) if i != j]
        if others:
            pos_nn.append(min(others))
    radius = quantile(pos_nn, float(cfg.get("positive_radius_quantile", 0.75)), default=0.0)
    if not pos_nn:
        radius = -1.0  # disables H10 unless at least two positive supports exist
    return {
        "cfg": dict(cfg),
        "std": std,
        "entries": entries,
        "positive_count": len(pos),
        "negative_count": len(neg),
        "positive_radius": radius,
        "default": 15,
    }


def pred_one(model: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    cfg = model.get("cfg") or {}
    if cfg.get("family") == "all_h15":
        return 15
    if int(model.get("positive_count", 0)) < int(cfg.get("min_positive_support", 2)):
        return 15
    z = feat(row, model.get("std") or {})
    entries = list(model.get("entries") or [])
    if z is None or not entries:
        return 15
    pos_d = [dist(z, e["z"]) for e in entries if int(e["label"]) == 10]
    neg_d = [dist(z, e["z"]) for e in entries if int(e["label"]) != 10]
    if not pos_d:
        return 15
    dpos = min(pos_d)
    dneg = min(neg_d) if neg_d else float("inf")
    radius = sf(model.get("positive_radius"), default=-1.0)
    if radius < 0 or dpos > radius:
        return 15
    nearest = sorted((dist(z, e["z"]), int(e["label"])) for e in entries)
    k = max(1, min(int(cfg.get("k", 1)), len(nearest)))
    frac10 = sum(1 for _, lab in nearest[:k] if lab == 10) / float(k)
    if frac10 < float(cfg.get("min_positive_fraction", 1.0)):
        return 15
    margin = float(cfg.get("negative_margin", 1.0))
    if math.isfinite(dneg) and dneg < margin * dpos:
        return 15
    return 10


def predict(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    return [pred_one(model, r) for r in rows]


def eval_policy(v0b: Any, rows: Sequence[Mapping[str, Any]], preds: Sequence[int], require_nonconstant: bool = True) -> Dict[str, Any]:
    return v0b.eval_policy(list(rows), list(preds), require_nonconstant=require_nonconstant)


def selection_score(ev: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
    saving = sf(ev.get("decision_relative_saving_vs_H15"), -1.0)
    phys = sf(ev.get("physical_delta_vs_H15"), 1e9)
    tol = max(sf(ev.get("physical_tolerance_vs_H15"), 1.0), 1e-9)
    unsafe = sf(ev.get("unsafe_chosen_groups"), 999.0)
    counts = ev.get("horizon_counts") or {}
    h10_count = sf(counts.get("10", 0), 0.0)
    physically_ok = 1.0 if unsafe == 0 and phys <= tol else 0.0
    # Safety/physical admissibility dominates measured-time saving. This is the
    # explicit objective repair versus the previous selection diagnostic.
    return (
        1.0 if unsafe == 0 else 0.0,
        physically_ok,
        1.0 if ev.get("core_pass_5pct") else 0.0,
        1.0 if ev.get("core_pass_10pct") else 0.0,
        saving if physically_ok else -max(0.0, phys / tol),
        1.0 if ev.get("nonconstant") else 0.0,
        h10_count,
    )


def fixed_cv(v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    if len(folds) < 3:
        model = fit_model(rows, cfg)
        ev = eval_policy(v0b, rows, predict(model, rows), require_nonconstant=True)
        ev.update({"fold_key": fold_key, "folds": len(folds), "fallback": "resubstitution_too_few_folds"})
        return ev
    preds: Dict[str, int] = {}
    for fold in folds:
        train = [r for r in rows if str(r.get(fold_key)) != fold]
        test = [r for r in rows if str(r.get(fold_key)) == fold]
        if not train or not test:
            continue
        model = fit_model(train, cfg)
        for r, p in zip(test, predict(model, test)):
            preds[str(r["sample_id"])] = int(p)
    ordered = [r for r in rows if str(r["sample_id"]) in preds]
    ev = eval_policy(v0b, ordered, [preds[str(r["sample_id"])] for r in ordered], require_nonconstant=True)
    ev.update({"fold_key": fold_key, "folds": len(folds)})
    return ev


def select_cfg(v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, cand: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    best_cfg: Optional[Dict[str, Any]] = None
    best_ev: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[float, ...]] = None
    for cfg in cand:
        ev = fixed_cv(v0b, rows, fold_key, cfg)
        sc = selection_score(ev)
        if best_score is None or sc > best_score:
            best_score = sc
            best_cfg = dict(cfg)
            best_ev = dict(ev)
    if best_cfg is None or best_ev is None:
        raise ContractError("no model configuration selected")
    return best_cfg, best_ev, len(cand)


def nested_cv(v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, cand: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    if len(folds) < 3:
        return {"skipped": f"too_few_{fold_key}_folds", "fold_key": fold_key, "folds": len(folds), "n": len(rows)}
    preds: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []
    inner_evals = 0
    for fold in folds:
        train = [r for r in rows if str(r.get(fold_key)) != fold]
        test = [r for r in rows if str(r.get(fold_key)) == fold]
        if not train or not test:
            continue
        cfg, inner_ev, n_eval = select_cfg(v0b, train, fold_key, cand)
        inner_evals += n_eval
        model = fit_model(train, cfg)
        ps = predict(model, test)
        for r, p in zip(test, ps):
            preds[str(r["sample_id"])] = int(p)
        selected.append({
            "fold": fold,
            "test_n": len(test),
            "selected_config": cfg,
            "inner_core5": bool(inner_ev.get("core_pass_5pct")),
            "inner_physically_ok": bool(sf(inner_ev.get("unsafe_chosen_groups"), 999.0) == 0 and sf(inner_ev.get("physical_delta_vs_H15"), 1e9) <= sf(inner_ev.get("physical_tolerance_vs_H15"), 0.0)),
            "inner_decision_saving": inner_ev.get("decision_relative_saving_vs_H15"),
            "inner_physical_delta": inner_ev.get("physical_delta_vs_H15"),
            "inner_horizon_counts": inner_ev.get("horizon_counts"),
            "test_pred_counts": {"10": sum(1 for p in ps if int(p) == 10), "15": sum(1 for p in ps if int(p) == 15)},
        })
    ordered = [r for r in rows if str(r["sample_id"]) in preds]
    ev = eval_policy(v0b, ordered, [preds[str(r["sample_id"])] for r in ordered], require_nonconstant=True)
    fam_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for s in selected:
        cfg = s.get("selected_config") or {}
        fam_counts[str(cfg.get("family"))] = fam_counts.get(str(cfg.get("family")), 0) + 1
        mode_counts[str(cfg.get("mode", "none"))] = mode_counts.get(str(cfg.get("mode", "none")), 0) + 1
    false_positive_rows = []
    pred_seq = [preds.get(str(r["sample_id"]), 15) for r in ordered]
    for r, p in zip(ordered, pred_seq):
        if int(p) == 10 and int(r.get("label", 15)) != 10:
            false_positive_rows.append({
                "sample_id": r.get("sample_id"),
                "state_key": r.get("state_key"),
                "case": r.get("case"),
                "source": r.get("source"),
                "terminal_profile": r.get("terminal_profile"),
                "h10_minus_h15_physical": sf(r.get("h10_minus_h15_physical")),
                "h10_safe": bool((r.get("h10") or {}).get("safe")),
            })
    ev.update({
        "fold_key": fold_key,
        "folds": len(folds),
        "outer_folds_completed": len(selected),
        "candidate_configs_per_outer_fold": len(cand),
        "inner_config_evaluations": inner_evals,
        "selected_family_counts": fam_counts,
        "selected_mode_counts": mode_counts,
        "selected_preview": selected[:32],
        "h10_false_positive_count": len(false_positive_rows),
        "h10_false_positive_physical_summary": finite(x["h10_minus_h15_physical"] for x in false_positive_rows),
        "h10_false_positive_preview": false_positive_rows[:16],
    })
    return ev


def analyze_dataset(v0b: Any, name: str, rows: Sequence[Mapping[str, Any]], cand: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(r) for r in rows if obs(r) is not None]
    res: Dict[str, Any] = {
        "dataset": name,
        "n": len(rows),
        "state_count": len({r["state_key"] for r in rows}),
        "case_count": len({r["case"] for r in rows}),
        "label_counts": {"10": sum(1 for r in rows if int(r["label"]) == 10), "15": sum(1 for r in rows if int(r["label"]) == 15)},
    }
    res["baselines"] = {
        "fixed_H15": eval_policy(v0b, rows, [15] * len(rows), require_nonconstant=False),
        "fixed_H10": eval_policy(v0b, rows, [10] * len(rows), require_nonconstant=False),
        "oracle_H10H15": eval_policy(v0b, rows, [int(r["label"]) for r in rows], require_nonconstant=False),
    }
    res["safety_gated_nested_cv"] = {
        "leave_state_out": nested_cv(v0b, rows, "state_key", cand),
        "leave_case_out": nested_cv(v0b, rows, "case", cand),
    }
    ls = res["safety_gated_nested_cv"]["leave_state_out"]
    lc = res["safety_gated_nested_cv"]["leave_case_out"]
    res["passes_core5_both_folds"] = bool(ls.get("core_pass_5pct") and lc.get("core_pass_5pct"))
    res["passes_core10_both_folds"] = bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct"))
    return res


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
    robust_done = completed_ok(ROBUST_DONE)
    terminal_done = completed_ok(TERMINAL_CONSIST_DONE)
    v0b = import_module(V0B_SCRIPT, "h10h15_v0b_for_safety_gated")
    robust = import_module(ROBUST_SCRIPT, "robust_terminal_for_safety_gated")
    samples, _aux = v0b.load_samples()
    cand = candidate_configs()
    strategies = ["terminal_agreement_only", "aggregate_terminal_robust", "strict_per_profile_regret2"]
    analyses: Dict[str, Any] = {}
    for strategy in strategies:
        repaired, debug = robust.apply_strategy(samples, strategy)
        datasets = {
            "all_profiles": repaired,
            "risk_anchor_source_all_profiles": [r for r in repaired if r.get("source") == "risk_anchor"],
            "matched_terminal_only": [r for r in repaired if r.get("terminal_profile") == "matched_terminal"],
            "shared_h15_terminal_only": [r for r in repaired if r.get("terminal_profile") == "shared_h15_terminal"],
        }
        results = {name: analyze_dataset(v0b, name, rows, cand) for name, rows in datasets.items() if rows}
        pass_datasets = [name for name, res in results.items() if res.get("passes_core5_both_folds")]
        strong_datasets = [name for name, res in results.items() if res.get("passes_core10_both_folds")]
        analyses[strategy] = {"debug": debug, "results": results, "pass_datasets": pass_datasets, "strong_pass_datasets": strong_datasets}
    pass_summary = {s: analyses[s]["pass_datasets"] for s in strategies}
    strong_summary = {s: analyses[s]["strong_pass_datasets"] for s in strategies}
    all_profile_pass = [s for s in strategies if "all_profiles" in pass_summary[s]]
    terminal_fixed_pass = [s for s in strategies for d in pass_summary[s] if d in ("matched_terminal_only", "shared_h15_terminal_only")]
    risk_only_pass = [s for s in strategies if "risk_anchor_source_all_profiles" in pass_summary[s] and "all_profiles" not in pass_summary[s]]
    if all_profile_pass or terminal_fixed_pass:
        decision = "safety-gated state-observable selector passes a deployable terminal-independent gate; after backup freeze tiny selector-overhead smoke versus fixed true H15 before any validation"
    elif risk_only_pass:
        decision = "safety-gated selector still passes only the risk-anchor subset; collect source-independent confirmation/control-state coverage or value-calibration smoke before rollout"
    else:
        decision = "safety-gated selector cannot recover safe compute savings from current online observations; prioritize representation/value-calibration and additional controlled state coverage before selector rollout"
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_SAFETY_GATED_SELECTOR_CV_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_safety_gated_selector_cv_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_configs_per_outer_fold": len(cand),
        "inputs": {"robust_completed": rel(ROBUST_DONE), "terminal_consistency_completed": rel(TERMINAL_CONSIST_DONE), "v0b_script": rel(V0B_SCRIPT), "robust_script": rel(ROBUST_SCRIPT)},
        "prerequisite_headlines": {"robust_terminal_label_cv": robust_done.get("headline"), "terminal_consistency": terminal_done.get("headline")},
        "hypothesis": "safety-first selection/representation can avoid H10 false positives that made previous robust/state-observable CV fail despite robust oracle compute value",
        "pass_summary": pass_summary,
        "strong_pass_summary": strong_summary,
        "all_profile_pass_strategies": all_profile_pass,
        "terminal_fixed_pass_strategies": terminal_fixed_pass,
        "risk_only_pass_strategies": risk_only_pass,
        "decision": decision,
        "analyses": analyses,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle true-variable-H safety-gated selector CV v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Candidate safety-gated configs per outer fold: `{len(cand)}`.",
        f"- Pass summary: `{pass_summary}`.",
        f"- Strong pass summary: `{strong_summary}`.",
        f"- All-profile pass strategies: `{all_profile_pass}`.",
        f"- Terminal-fixed pass strategies: `{terminal_fixed_pass}`.",
        f"- Decision: {decision}",
        "",
        "## Per-strategy comparison against fixed true H15",
        "",
        "| strategy | dataset | labels | oracle dec save | oracle physΔ/tol | LS pass/save/physΔ/fp | LC pass/save/physΔ/fp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, a in analyses.items():
        for dname, res in (a.get("results") or {}).items():
            oracle = ((res.get("baselines") or {}).get("oracle_H10H15") or {})
            ls = ((res.get("safety_gated_nested_cv") or {}).get("leave_state_out") or {})
            lc = ((res.get("safety_gated_nested_cv") or {}).get("leave_case_out") or {})
            lines.append("| `%s` | `%s` | `%s` | %.6g | %.6g/%.6g | `%s`/%.6g/%.6g/%s | `%s`/%.6g/%.6g/%s |" % (
                strategy, dname, res.get("label_counts"), sf(oracle.get("decision_relative_saving_vs_H15")), sf(oracle.get("physical_delta_vs_H15")), sf(oracle.get("physical_tolerance_vs_H15")),
                bool(ls.get("core_pass_5pct")), sf(ls.get("decision_relative_saving_vs_H15")), sf(ls.get("physical_delta_vs_H15")), ls.get("h10_false_positive_count"),
                bool(lc.get("core_pass_5pct")), sf(lc.get("decision_relative_saving_vs_H15")), sf(lc.get("physical_delta_vs_H15")), lc.get("h10_false_positive_count"),
            ))
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A safety-gated selector can only move to closed-loop selector-overhead smoke if terminal-independent/all-profile or terminal-fixed leave-state and leave-case gates pass against fixed true H15 with no safety/physical regression. Risk-only wins remain development diagnostics and trigger source-independent confirmation or value calibration.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    summary = OUT_DIR / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup safety-gated selector CV outputs before any selector smoke, simulations, training or refit",
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
## 2026-09-29 vehicle true-variable-H safety-gated selector CV v0

UTC: {created.isoformat()}. Offline/no-simulation safety-gated robust-label selector diagnostic; validation64 and sealed test stayed closed. Pass summary: {pass_summary}; strong pass summary: {strong_summary}. Decision: {decision}. Artifacts: `{rel(summary)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), V0B_SCRIPT, ROBUST_SCRIPT, ROBUST_DONE, TERMINAL_CONSIST_DONE, STATE_PATH, req]
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
        "headline": {"pass_summary": pass_summary, "strong_pass_summary": strong_summary, "all_profile_pass_strategies": all_profile_pass, "terminal_fixed_pass_strategies": terminal_fixed_pass, "risk_only_pass_strategies": risk_only_pass, "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
