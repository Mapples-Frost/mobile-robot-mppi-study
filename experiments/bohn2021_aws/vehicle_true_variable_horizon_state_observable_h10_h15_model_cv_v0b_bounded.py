#!/usr/bin/env python3
"""Bounded offline state-observable H10/H15 model-CV diagnostic.

Development-only.  No new simulation, no rollout, no gradient training, no
validation64 read, and no sealed-test read.

This is the bounded successor to the feature-deployability audit: v0b's apparent
H10/H15 wins were driven by `selection_group`/protocol metadata.  Here we allow
only online branch observations (`initial_observation_at_branch`) and simple
predeclared transforms, compare to fixed true H15 with measured whole-decision
and solver timings, and require both leave-state and leave-case CV gates before
any later rollout can be justified.
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
NAME = "vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded"
STAMP = "20260929T0905Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0B_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair.py"
V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/completed.json"
FEATURE_AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0_20260929T0850Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-state-observable-h10-h15-model-cv-v0b-bounded-{STAMP}"
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
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"prerequisite did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("validation64_bank_opened") is True:
        raise ContractError(f"unexpected validation/test access flag in {rel(path)}")
    return obj


def load_v0b() -> Any:
    spec = importlib.util.spec_from_file_location("h10h15_v0b_schema_repair", str(V0B_SCRIPT))
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import {rel(V0B_SCRIPT)}")
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
    if mode == "abs":
        return [abs(x) for x in base]
    if mode == "raw_abs":
        return base + [abs(x) for x in base]
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


def group_label(rows: Sequence[Mapping[str, Any]]) -> int:
    # Conservative, cost-aware aggregate label matching v0b's fixed-H15 comparator.
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


def configs() -> List[Dict[str, Any]]:
    # Deliberately bounded: 36 candidates per inner selection, all using only
    # online branch observations.  No metadata, terminal-profile, role, source,
    # or hand-labelled window features are included.
    out: List[Dict[str, Any]] = []
    for mode in ("raw", "abs", "raw_abs", "raw_abs_l2"):
        for k in (1, 3, 5):
            for vote in (0.5, 0.67):
                out.append({"family": "knn", "mode": mode, "k": k, "vote_threshold": vote})
        for ratio in (0.9, 1.0, 1.25):
            out.append({"family": "centroid", "mode": mode, "h10_distance_ratio": ratio})
    return out


def fit_model(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    std = make_std(rows, str(cfg.get("mode", "raw")))
    entries = []
    for r in rows:
        z = feat(r, std)
        if z is not None:
            entries.append({"z": z, "label": int(r.get("label", 15)), "sample_id": str(r.get("sample_id"))})
    model: Dict[str, Any] = {"cfg": dict(cfg), "std": std, "entries": entries, "default": group_label(rows)}
    pos = [e for e in entries if int(e["label"]) == 10]
    neg = [e for e in entries if int(e["label"]) != 10]
    model["positive_count"] = len(pos)
    model["negative_count"] = len(neg)
    if str(cfg.get("family")) == "centroid" and pos:
        def centroid(parts: Sequence[Mapping[str, Any]]) -> Optional[List[float]]:
            if not parts:
                return None
            n = min(len(p["z"]) for p in parts)
            return [math.fsum(float(p["z"][j]) for p in parts) / len(parts) for j in range(n)]
        model["c10"] = centroid(pos)
        model["c15"] = centroid(neg)
    return model


def pred_one(model: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    default = int(model.get("default", 15))
    z = feat(row, model.get("std") or {})
    entries = list(model.get("entries") or [])
    if z is None or not entries or int(model.get("positive_count", 0)) <= 0:
        return default
    cfg = model.get("cfg") or {}
    if str(cfg.get("family")) == "knn":
        pairs = sorted((dist(z, e["z"]), int(e["label"])) for e in entries)
        k = max(1, min(int(cfg.get("k", 1)), len(pairs)))
        chosen = pairs[:k]
        frac10 = sum(1 for _, lab in chosen if lab == 10) / float(k)
        return 10 if frac10 >= float(cfg.get("vote_threshold", 0.67)) else 15
    if str(cfg.get("family")) == "centroid":
        c10, c15 = model.get("c10"), model.get("c15")
        if not isinstance(c10, list):
            return default
        d10 = dist(z, c10)
        d15 = dist(z, c15) if isinstance(c15, list) else float("inf")
        return 10 if d10 <= float(cfg.get("h10_distance_ratio", 1.0)) * d15 else 15
    return default


def predict(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    return [pred_one(model, r) for r in rows]


def eval_policy(v0b: Any, rows: Sequence[Mapping[str, Any]], preds: Sequence[int], require_nonconstant: bool = True) -> Dict[str, Any]:
    return v0b.eval_policy(list(rows), list(preds), require_nonconstant=require_nonconstant)


def score(ev: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float]:
    saving = sf(ev.get("decision_relative_saving_vs_H15"), -1.0)
    phys = sf(ev.get("physical_delta_vs_H15"), 1e9)
    tol = max(sf(ev.get("physical_tolerance_vs_H15"), 1.0), 1e-9)
    unsafe = sf(ev.get("unsafe_chosen_groups"), 999.0)
    regret = sf(ev.get("oracle_regret_physical_sum"), 1e9)
    return (
        1.0 if ev.get("core_pass_5pct") else 0.0,
        1.0 if ev.get("core_pass_10pct") else 0.0,
        1.0 if ev.get("nonconstant") else 0.0,
        -unsafe,
        saving,
        -max(0.0, phys / tol) - 0.001 * max(0.0, regret),
    )


def fixed_cv(v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    if len(folds) < 2:
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
    best_cfg, best_ev, best_score = None, None, None
    for cfg in cand:
        ev = fixed_cv(v0b, rows, fold_key, cfg)
        sc = score(ev)
        if best_score is None or sc > best_score:
            best_score, best_cfg, best_ev = sc, dict(cfg), dict(ev)
    assert best_cfg is not None and best_ev is not None
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
            "inner_decision_saving": inner_ev.get("decision_relative_saving_vs_H15"),
            "inner_physical_delta": inner_ev.get("physical_delta_vs_H15"),
            "positive_count": model.get("positive_count"),
            "negative_count": model.get("negative_count"),
            "test_pred_counts": {"10": sum(1 for p in ps if int(p) == 10), "15": sum(1 for p in ps if int(p) == 15)},
        })
    ordered = [r for r in rows if str(r["sample_id"]) in preds]
    ev = eval_policy(v0b, ordered, [preds[str(r["sample_id"])] for r in ordered], require_nonconstant=True)
    fam_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for s in selected:
        cfg = s.get("selected_config") or {}
        fam_counts[str(cfg.get("family"))] = fam_counts.get(str(cfg.get("family")), 0) + 1
        mode_counts[str(cfg.get("mode"))] = mode_counts.get(str(cfg.get("mode")), 0) + 1
    ev.update({
        "fold_key": fold_key,
        "folds": len(folds),
        "outer_folds_completed": len(selected),
        "candidate_configs_per_outer_fold": len(cand),
        "inner_config_evaluations": inner_evals,
        "selected_family_counts": fam_counts,
        "selected_mode_counts": mode_counts,
        "selected_preview": selected[:24],
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
        "obs_dim_summary": finite(len(obs(r) or []) for r in rows),
    }
    res["baselines"] = {
        "fixed_H15": eval_policy(v0b, rows, [15] * len(rows), require_nonconstant=False),
        "fixed_H10": eval_policy(v0b, rows, [10] * len(rows), require_nonconstant=False),
        "oracle_H10H15": eval_policy(v0b, rows, [int(r["label"]) for r in rows], require_nonconstant=False),
    }
    res["state_observable_nested_cv"] = {
        "leave_state_out": nested_cv(v0b, rows, "state_key", cand),
        "leave_case_out": nested_cv(v0b, rows, "case", cand),
    }
    ls = res["state_observable_nested_cv"]["leave_state_out"]
    lc = res["state_observable_nested_cv"]["leave_case_out"]
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
    v0b_done = completed_ok(V0B_DONE)
    feature_done = completed_ok(FEATURE_AUDIT_DONE)
    if not ((feature_done.get("headline") or {}).get("leakage_blocks_selector_rollout") is True):
        raise ContractError("feature audit did not establish leakage-block precondition")
    v0b = load_v0b()
    samples, _aux = v0b.load_samples()
    variants, flips = v0b.variants(samples)
    cand = configs()
    results = {name: analyze_dataset(v0b, name, rows, cand) for name, rows in variants.items() if rows}
    pass_datasets = [name for name, r in results.items() if r.get("passes_core5_both_folds")]
    strong_pass_datasets = [name for name, r in results.items() if r.get("passes_core10_both_folds")]
    terminal_fixed_passes = [name for name in pass_datasets if name in ("matched_terminal_only", "shared_h15_terminal_only")]
    risk_passes = [name for name in pass_datasets if name.startswith("risk_anchor")]
    oracle_opps = {
        name: {
            "labels": r.get("label_counts"),
            "physical_delta_vs_H15": r["baselines"]["oracle_H10H15"].get("physical_delta_vs_H15"),
            "decision_relative_saving_vs_H15": r["baselines"]["oracle_H10H15"].get("decision_relative_saving_vs_H15"),
            "solver_relative_saving_vs_H15": r["baselines"]["oracle_H10H15"].get("solver_relative_saving_vs_H15"),
            "core_pass_5pct": r["baselines"]["oracle_H10H15"].get("core_pass_5pct"),
            "core_pass_10pct": r["baselines"]["oracle_H10H15"].get("core_pass_10pct"),
        }
        for name, r in results.items()
    }
    if terminal_fixed_passes:
        decision = "state-observable CV found a terminal-fixed pass; after backup freeze a tiny development rollout with selector overhead and blocked/randomized timing versus fixed true H10/H15/H25"
        proceed = True
    elif pass_datasets:
        decision = "state-observable CV passes only in pooled/diagnostic subsets; do not rollout yet because terminal-profile dependence remains unresolved"
        proceed = False
    else:
        decision = "no state-observable H10/H15 model passed both leave-state and leave-case gates; selector rollout remains blocked, pivot to terminal-value/objective calibration or scenario/modeling intervention"
        proceed = False
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_STATE_OBSERVABLE_H10_H15_MODEL_CV_V0B_BOUNDED_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_state_observable_h10_h15_model_cv_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "offline_model_candidate_configs_per_outer_fold": len(cand),
        "inputs": {"v0b_completed": rel(V0B_DONE), "feature_audit_completed": rel(FEATURE_AUDIT_DONE), "v0b_script": rel(V0B_SCRIPT)},
        "prerequisite_headlines": {"v0b": v0b_done.get("headline"), "feature_audit": feature_done.get("headline")},
        "terminal_label_flipped_states": len(flips),
        "oracle_opportunities_vs_fixed_H15": oracle_opps,
        "pass_datasets": pass_datasets,
        "strong_pass_datasets": strong_pass_datasets,
        "terminal_fixed_passes": terminal_fixed_passes,
        "risk_passes": risk_passes,
        "proceed_to_selector_rollout": proceed,
        "decision": decision,
        "results": results,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)

    def m(obj: Mapping[str, Any], key: str) -> str:
        v = obj.get(key)
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    lines = [
        "# Vehicle true-variable-H state-observable H10/H15 model-CV v0b bounded",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Candidate state-observable configs per outer fold: `{len(cand)}` (KNN/centroid using only `initial_observation_at_branch`).",
        f"- Terminal-profile label-flipped states inherited from v0b: `{len(flips)}`.",
        f"- Datasets passing both leave-state and leave-case 5% gates: `{pass_datasets}`.",
        f"- Strong 10% passes: `{strong_pass_datasets}`.",
        f"- Terminal-fixed passes: `{terminal_fixed_passes}`.",
        f"- Proceed to selector rollout: `{proceed}`.",
        f"- Decision: {decision}",
        "",
        "## Per-dataset comparison against fixed true H15",
        "",
        "| dataset | n | labels | oracle dec save | oracle physΔ | LS pass | LS dec save | LS physΔ/tol | LS H counts | LC pass | LC dec save | LC physΔ/tol | LC H counts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        oracle = r["baselines"]["oracle_H10H15"]
        ls = r["state_observable_nested_cv"]["leave_state_out"]
        lc = r["state_observable_nested_cv"]["leave_case_out"]
        lines.append("| `%s` | %s | `%s` | %s | %s | `%s` | %s | %s/%s | `%s` | `%s` | %s | %s/%s | `%s` |" % (
            name, r.get("n"), r.get("label_counts"), m(oracle, "decision_relative_saving_vs_H15"), m(oracle, "physical_delta_vs_H15"),
            bool(ls.get("core_pass_5pct")), m(ls, "decision_relative_saving_vs_H15"), m(ls, "physical_delta_vs_H15"), m(ls, "physical_tolerance_vs_H15"), ls.get("horizon_counts"),
            bool(lc.get("core_pass_5pct")), m(lc, "decision_relative_saving_vs_H15"), m(lc, "physical_delta_vs_H15"), m(lc, "physical_tolerance_vs_H15"), lc.get("horizon_counts"),
        ))
    lines += [
        "",
        "## Interpretation rule frozen before execution",
        "",
        "A rollout is allowed only if an online/state-observable model passes both leave-state and leave-case gates with physical cost within fixed-H15 tolerance, zero unsafe chosen groups, nonconstant choices, and >=5% measured whole-decision saving. Terminal-fixed confirmation is required for immediate rollout; pooled-only success remains diagnostic because terminal-profile label dependence is already established.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    summary_path = OUT_DIR / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup state-observable H10/H15 offline model-CV diagnostic before any selector rollout, simulation, training or refit",
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
## 2026-09-29 vehicle true-variable-H state-observable H10/H15 model-CV v0b bounded

UTC: {created.isoformat()}. Offline/no-simulation diagnostic using only initial branch observations and online-derived transforms; validation64 and sealed test stayed closed. Candidate configs per outer fold={len(cand)}. Passing datasets={pass_datasets}; strong passes={strong_pass_datasets}; terminal-fixed passes={terminal_fixed_passes}. Proceed to selector rollout={proceed}. Decision: {decision}. Artifacts: `{rel(summary_path)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), V0B_SCRIPT, V0B_DONE, FEATURE_AUDIT_DONE, STATE_PATH, req]
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
        "offline_model_candidate_configs_per_outer_fold": len(cand),
        "headline": {"pass_datasets": pass_datasets, "strong_pass_datasets": strong_pass_datasets, "terminal_fixed_passes": terminal_fixed_passes, "proceed_to_selector_rollout": proceed, "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary_path), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
