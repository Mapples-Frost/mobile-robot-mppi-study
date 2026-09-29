#!/usr/bin/env python3
"""Offline state-observable H10/H15 model-CV/refit diagnostic v0.

Development-only. No new simulation, no controller rollout, no gradient training,
no validation64 read, and no sealed-test read.

Why this exists: v0b showed an apparent H10/H15 residual-CV win against fixed
true H15, but the follow-up feature-deployability audit found that the passing
rules used `selection_group` / protocol-source metadata rather than online state
features.  This script freezes the next discriminating intervention: can a small
state-observable model, trained only from existing branch observations and
measured continuation outcomes, recover the H10/H15 control-vs-compute tradeoff
under leave-state and leave-case CV?  If not, selector rollout remains blocked
and the next intervention should be terminal-value/objective calibration or
scenario/modeling work.
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
NAME = "vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0"
STAMP = "20260929T0900Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0B_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair.py"
V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/completed.json"
FEATURE_AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0_20260929T0850Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-state-observable-h10-h15-model-cv-v0-{STAMP}"
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


def median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not ys:
        return 0.0
    n = len(ys)
    mid = n // 2
    if n % 2:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def quantile(xs: Sequence[float], q: float, default: float = 0.0) -> float:
    ys = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not ys:
        return default
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


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
    return {
        "n": len(xs),
        "min": xs[0],
        "median": quantile(xs, 0.5),
        "mean": math.fsum(xs) / len(xs),
        "p95": quantile(xs, 0.95),
        "max": xs[-1],
        "sum": math.fsum(xs),
    }


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing prerequisite completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"prerequisite did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("validation64_bank_opened") is True:
        raise ContractError(f"unexpected validation/test access flag in prerequisite: {rel(path)}")
    return obj


def load_v0b_module() -> Any:
    if not V0B_SCRIPT.exists():
        raise ContractError(f"missing v0b script: {rel(V0B_SCRIPT)}")
    spec = importlib.util.spec_from_file_location("h10h15_v0b_schema_repair", str(V0B_SCRIPT))
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import v0b script: {rel(V0B_SCRIPT)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def obs_values(row: Mapping[str, Any]) -> Optional[List[float]]:
    obs = row.get("obs")
    if not isinstance(obs, list) or not obs:
        return None
    vals: List[float] = []
    for v in obs:
        try:
            y = float(v)
        except Exception:
            return None
        if not math.isfinite(y):
            return None
        vals.append(y)
    return vals


def transform_obs(obs: Sequence[float], mode: str, dim: int) -> List[float]:
    base = [float(obs[i]) if i < len(obs) else 0.0 for i in range(dim)]
    if mode == "raw":
        return base
    if mode == "abs":
        return [abs(x) for x in base]
    if mode == "raw_abs":
        return base + [abs(x) for x in base]
    if mode == "raw_abs_l2":
        return base + [abs(x) for x in base] + [math.sqrt(math.fsum(x * x for x in base))]
    raise ValueError(mode)


def make_standardizer(train_rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    obs_list = [obs_values(r) for r in train_rows]
    obs_list = [o for o in obs_list if o]
    if not obs_list:
        return {"mode": mode, "dim": 0, "center": [], "scale": []}
    dim = min(len(o) for o in obs_list)
    raw_feats = [transform_obs(o, mode, dim) for o in obs_list]
    fdim = min(len(f) for f in raw_feats) if raw_feats else 0
    center: List[float] = []
    scale: List[float] = []
    for j in range(fdim):
        col = [f[j] for f in raw_feats]
        med = median(col)
        q25 = quantile(col, 0.25, med)
        q75 = quantile(col, 0.75, med)
        sc = max(q75 - q25, 1e-6)
        center.append(med)
        scale.append(sc)
    return {"mode": mode, "dim": dim, "center": center, "scale": scale}


def featurize(row: Mapping[str, Any], standardizer: Mapping[str, Any]) -> Optional[List[float]]:
    obs = obs_values(row)
    if obs is None:
        return None
    mode = str(standardizer.get("mode"))
    dim = int(standardizer.get("dim") or 0)
    center = list(standardizer.get("center") or [])
    scale = list(standardizer.get("scale") or [])
    if dim <= 0 or not center or not scale:
        return None
    raw = transform_obs(obs, mode, dim)
    n = min(len(raw), len(center), len(scale))
    return [(raw[i] - float(center[i])) / max(float(scale[i]), 1e-6) for i in range(n)]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return math.sqrt(math.fsum((float(a[i]) - float(b[i])) ** 2 for i in range(n)) / n)


def candidate_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    modes = ["raw", "abs", "raw_abs", "raw_abs_l2"]
    for mode in modes:
        for k in (1, 3, 5, 7):
            for vote_threshold in (0.5, 0.67, 0.8, 1.0):
                for radius_q in (None, 0.5, 0.75, 0.9):
                    for h10_ratio in (0.9, 1.0, 1.25, 10.0):
                        configs.append({
                            "family": "knn_vote",
                            "mode": mode,
                            "k": k,
                            "vote_threshold": vote_threshold,
                            "radius_q": radius_q,
                            "h10_distance_ratio": h10_ratio,
                        })
        for ratio in (0.8, 1.0, 1.25, 1.5):
            configs.append({"family": "nearest_centroid", "mode": mode, "h10_distance_ratio": ratio})
        for radius_q in (0.25, 0.5, 0.75, 0.9):
            configs.append({"family": "positive_radius", "mode": mode, "radius_q": radius_q})
    return configs


def default_label(train_rows: Sequence[Mapping[str, Any]]) -> int:
    # Use the imported v0b group label if available through row metrics. A safe
    # fallback is H15, which is the comparator and conservative default.
    if not train_rows:
        return 15
    h10_phys = math.fsum(sf(r["h10"]["physical"]) for r in train_rows)
    h15_phys = math.fsum(sf(r["h15"]["physical"]) for r in train_rows)
    h10_dec = math.fsum(sf(r["h10"]["decision"]) for r in train_rows)
    h15_dec = math.fsum(sf(r["h15"]["decision"]) for r in train_rows)
    tol = max(2.0 * len(train_rows), 0.05 * abs(h15_phys))
    saving = (h15_dec - h10_dec) / h15_dec if h15_dec > 0 else 0.0
    h10_all_safe = all(bool(r["h10"]["safe"]) for r in train_rows)
    if h10_all_safe and h10_phys - h15_phys <= tol and saving >= MIN_SAVE:
        return 10
    return 15


def fit_model(train_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    std = make_standardizer(train_rows, str(config.get("mode", "raw")))
    entries: List[Dict[str, Any]] = []
    for r in train_rows:
        z = featurize(r, std)
        if z is not None:
            entries.append({"z": z, "label": int(r.get("label", 15)), "sample_id": str(r.get("sample_id"))})
    default = default_label(train_rows)
    labels = [int(e["label"]) for e in entries]
    positive = [e for e in entries if int(e["label"]) == 10]
    negative = [e for e in entries if int(e["label"]) != 10]
    model: Dict[str, Any] = {"config": dict(config), "standardizer": std, "entries": entries, "default": default, "positive_count": len(positive), "negative_count": len(negative)}
    if not entries or not positive:
        model["reason"] = "no_observable_entries_or_no_positive_train_labels"
        return model
    family = str(config.get("family"))
    if family == "knn_vote":
        # Training-manifold radius from leave-one nearest-neighbour distances.
        nn_d: List[float] = []
        for i, e in enumerate(entries):
            others = [dist(e["z"], f["z"]) for j, f in enumerate(entries) if j != i]
            if others:
                nn_d.append(min(others))
        rq = config.get("radius_q")
        model["radius"] = None if rq is None else quantile(nn_d, float(rq), default=float("inf"))
    elif family == "nearest_centroid":
        def centroid(parts: Sequence[Mapping[str, Any]]) -> Optional[List[float]]:
            if not parts:
                return None
            n = min(len(p["z"]) for p in parts)
            return [math.fsum(float(p["z"][j]) for p in parts) / len(parts) for j in range(n)]
        model["h10_centroid"] = centroid(positive)
        model["h15_centroid"] = centroid(negative)
    elif family == "positive_radius":
        pos_d: List[float] = []
        for i, e in enumerate(positive):
            others = [dist(e["z"], f["z"]) for j, f in enumerate(positive) if j != i]
            if others:
                pos_d.append(min(others))
        model["radius"] = quantile(pos_d, float(config.get("radius_q", 0.5)), default=0.0)
    return model


def predict_one(model: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    default = int(model.get("default", 15))
    entries = list(model.get("entries") or [])
    if not entries or int(model.get("positive_count", 0)) <= 0:
        return default
    z = featurize(row, model.get("standardizer") or {})
    if z is None:
        return default
    cfg = model.get("config") or {}
    family = str(cfg.get("family"))
    if family == "knn_vote":
        pairs = sorted((dist(z, e["z"]), int(e["label"])) for e in entries)
        if not pairs:
            return default
        radius = model.get("radius")
        if radius is not None and pairs[0][0] > float(radius):
            return 15
        k = max(1, int(cfg.get("k", 1)))
        chosen = pairs[: min(k, len(pairs))]
        pos_frac = sum(1 for _, lab in chosen if lab == 10) / float(len(chosen))
        d10s = [d for d, lab in pairs if lab == 10]
        d15s = [d for d, lab in pairs if lab != 10]
        ratio = float(cfg.get("h10_distance_ratio", 10.0))
        if d10s and d15s and min(d10s) > ratio * max(min(d15s), 1e-12):
            return 15
        return 10 if pos_frac >= float(cfg.get("vote_threshold", 1.0)) else 15
    if family == "nearest_centroid":
        c10 = model.get("h10_centroid")
        c15 = model.get("h15_centroid")
        if not isinstance(c10, list):
            return default
        d10 = dist(z, c10)
        d15 = dist(z, c15) if isinstance(c15, list) else float("inf")
        ratio = float(cfg.get("h10_distance_ratio", 1.0))
        return 10 if d10 <= ratio * d15 else 15
    if family == "positive_radius":
        positives = [e for e in entries if int(e["label"]) == 10]
        if not positives:
            return default
        dmin = min(dist(z, e["z"]) for e in positives)
        radius = float(model.get("radius", 0.0))
        return 10 if dmin <= radius else 15
    return default


def predict(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[int]:
    return [predict_one(model, r) for r in rows]


def eval_with_v0b(v0b: Any, rows: Sequence[Mapping[str, Any]], preds: Sequence[int], require_nonconstant: bool = True) -> Dict[str, Any]:
    return v0b.eval_policy(list(rows), list(preds), require_nonconstant=require_nonconstant)


def score_eval(ev: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
    saving = sf(ev.get("decision_relative_saving_vs_H15"), -1.0)
    solver_saving = sf(ev.get("solver_relative_saving_vs_H15"), -1.0)
    phys_delta = sf(ev.get("physical_delta_vs_H15"), 1e9)
    tol = max(sf(ev.get("physical_tolerance_vs_H15"), 0.0), 1e-9)
    regret = sf(ev.get("oracle_regret_physical_sum"), 1e9)
    unsafe = sf(ev.get("unsafe_chosen_groups"), 999.0)
    nonconstant = 1.0 if ev.get("nonconstant") else 0.0
    return (
        1.0 if ev.get("core_pass_5pct") else 0.0,
        1.0 if ev.get("core_pass_10pct") else 0.0,
        nonconstant,
        -unsafe,
        saving,
        solver_saving,
        -max(0.0, phys_delta / tol) - 0.001 * max(0.0, regret),
    )


def fixed_config_cv(v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    preds_by_id: Dict[str, int] = {}
    if len(folds) < 2:
        model = fit_model(rows, config)
        preds = predict(model, rows)
        ev = eval_with_v0b(v0b, rows, preds, require_nonconstant=True)
        ev.update({"fold_key": fold_key, "folds": len(folds), "fallback": "resubstitution_too_few_inner_folds"})
        return ev
    for fold in folds:
        train = [r for r in rows if str(r.get(fold_key)) != fold]
        test = [r for r in rows if str(r.get(fold_key)) == fold]
        if not train or not test:
            continue
        model = fit_model(train, config)
        for r, p in zip(test, predict(model, test)):
            preds_by_id[str(r["sample_id"])] = int(p)
    ordered = [r for r in rows if str(r["sample_id"]) in preds_by_id]
    ev = eval_with_v0b(v0b, ordered, [preds_by_id[str(r["sample_id"])] for r in ordered], require_nonconstant=True)
    ev.update({"fold_key": fold_key, "folds": len(folds)})
    return ev


def select_config(v0b: Any, train_rows: Sequence[Mapping[str, Any]], fold_key: str, configs: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    best_cfg: Optional[Dict[str, Any]] = None
    best_ev: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[float, float, float, float, float, float, float]] = None
    inner_evals = 0
    for cfg in configs:
        ev = fixed_config_cv(v0b, train_rows, fold_key, cfg)
        inner_evals += 1
        sc = score_eval(ev)
        if best_score is None or sc > best_score:
            best_score = sc
            best_cfg = dict(cfg)
            best_ev = dict(ev)
    assert best_cfg is not None and best_ev is not None
    return best_cfg, best_ev, inner_evals


def nested_cv(v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, configs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    if len(folds) < 3:
        return {"skipped": f"too_few_{fold_key}_folds", "fold_key": fold_key, "folds": len(folds), "n": len(rows)}
    preds_by_id: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []
    inner_eval_count = 0
    for fold in folds:
        train = [r for r in rows if str(r.get(fold_key)) != fold]
        test = [r for r in rows if str(r.get(fold_key)) == fold]
        if not train or not test:
            continue
        cfg, inner_ev, n_eval = select_config(v0b, train, fold_key, configs)
        inner_eval_count += n_eval
        model = fit_model(train, cfg)
        preds = predict(model, test)
        for r, p in zip(test, preds):
            preds_by_id[str(r["sample_id"])] = int(p)
        selected.append({
            "fold": fold,
            "test_n": len(test),
            "selected_config": cfg,
            "inner_eval_core5": bool(inner_ev.get("core_pass_5pct")),
            "inner_eval_decision_saving": inner_ev.get("decision_relative_saving_vs_H15"),
            "inner_eval_physical_delta": inner_ev.get("physical_delta_vs_H15"),
            "model_positive_count": model.get("positive_count"),
            "model_negative_count": model.get("negative_count"),
            "test_pred_counts": {"10": sum(1 for p in preds if int(p) == 10), "15": sum(1 for p in preds if int(p) == 15)},
        })
    ordered = [r for r in rows if str(r["sample_id"]) in preds_by_id]
    ev = eval_with_v0b(v0b, ordered, [preds_by_id[str(r["sample_id"])] for r in ordered], require_nonconstant=True)
    family_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for s in selected:
        cfg = s.get("selected_config") or {}
        family_counts[str(cfg.get("family"))] = family_counts.get(str(cfg.get("family")), 0) + 1
        mode_counts[str(cfg.get("mode"))] = mode_counts.get(str(cfg.get("mode")), 0) + 1
    ev.update({
        "fold_key": fold_key,
        "folds": len(folds),
        "outer_folds_completed": len(selected),
        "configs_considered_per_outer_fold": len(configs),
        "inner_config_evaluations": inner_eval_count,
        "outer_model_fits": len(selected),
        "selected_family_counts": family_counts,
        "selected_mode_counts": mode_counts,
        "selected_preview": selected[:24],
    })
    return ev


def dataset_variants(v0b: Any, samples: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    variants, flips = v0b.variants(samples)
    # Keep only rows with deployable branch observation, since the diagnostic is
    # explicitly about observable policy learning rather than source metadata.
    filtered = {name: [r for r in rows if obs_values(r) is not None] for name, rows in variants.items()}
    return filtered, flips


def analyze_dataset(v0b: Any, name: str, rows: Sequence[Dict[str, Any]], configs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset": name,
        "n": len(rows),
        "state_count": len({r["state_key"] for r in rows}),
        "case_count": len({r["case"] for r in rows}),
        "label_counts": {"10": sum(1 for r in rows if int(r["label"]) == 10), "15": sum(1 for r in rows if int(r["label"]) == 15)},
        "obs_dim_summary": finite(len(obs_values(r) or []) for r in rows),
    }
    out["baselines"] = {
        "fixed_H15": eval_with_v0b(v0b, rows, [15] * len(rows), require_nonconstant=False),
        "fixed_H10": eval_with_v0b(v0b, rows, [10] * len(rows), require_nonconstant=False),
        "oracle_H10H15": eval_with_v0b(v0b, rows, [int(r["label"]) for r in rows], require_nonconstant=False),
    }
    out["state_observable_nested_cv"] = {
        "leave_state_out": nested_cv(v0b, rows, "state_key", configs),
        "leave_case_out": nested_cv(v0b, rows, "case", configs),
    }
    ls = out["state_observable_nested_cv"]["leave_state_out"]
    lc = out["state_observable_nested_cv"]["leave_case_out"]
    out["passes_core5_both_folds"] = bool(ls.get("core_pass_5pct") and lc.get("core_pass_5pct"))
    out["passes_core10_both_folds"] = bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct"))
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
    v0b_done = completed_ok(V0B_DONE)
    feature_done = completed_ok(FEATURE_AUDIT_DONE)
    if not ((feature_done.get("headline") or {}).get("leakage_blocks_selector_rollout") is True):
        raise ContractError("feature audit did not establish the intended leakage-block precondition")
    v0b = load_v0b_module()
    samples, _aux = v0b.load_samples()
    variants, flips = dataset_variants(v0b, samples)
    configs = candidate_configs()
    results = {name: analyze_dataset(v0b, name, rows, configs) for name, rows in variants.items() if rows}
    pass_datasets = [name for name, res in results.items() if res.get("passes_core5_both_folds")]
    strong_pass_datasets = [name for name, res in results.items() if res.get("passes_core10_both_folds")]
    terminal_fixed_passes = [name for name in pass_datasets if name in ("matched_terminal_only", "shared_h15_terminal_only")]
    risk_passes = [name for name in pass_datasets if name.startswith("risk_anchor")]
    oracle_opp = {
        name: {
            "labels": res.get("label_counts"),
            "physical_delta_vs_H15": res["baselines"]["oracle_H10H15"].get("physical_delta_vs_H15"),
            "decision_relative_saving_vs_H15": res["baselines"]["oracle_H10H15"].get("decision_relative_saving_vs_H15"),
            "solver_relative_saving_vs_H15": res["baselines"]["oracle_H10H15"].get("solver_relative_saving_vs_H15"),
            "core_pass_5pct": res["baselines"]["oracle_H10H15"].get("core_pass_5pct"),
            "core_pass_10pct": res["baselines"]["oracle_H10H15"].get("core_pass_10pct"),
        }
        for name, res in results.items()
    }
    if terminal_fixed_passes:
        decision = "state-observable offline CV found a terminal-fixed pass; after backup freeze a tiny development rollout with selector overhead and blocked/randomized timing versus fixed true H10/H15/H25"
        proceed_to_rollout = True
    elif pass_datasets:
        decision = "state-observable CV passes only in pooled or diagnostic subsets; do not rollout yet, require terminal-consistent confirmation or terminal-value calibration first"
        proceed_to_rollout = False
    else:
        decision = "no state-observable H10/H15 model passed both leave-state and leave-case gates; selector rollout remains blocked, pivot to terminal-value/objective calibration or scenario/modeling intervention"
        proceed_to_rollout = False
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_STATE_OBSERVABLE_H10_H15_MODEL_CV_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
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
        "offline_model_candidate_configs": len(configs),
        "inputs": {"v0b_completed": rel(V0B_DONE), "feature_audit_completed": rel(FEATURE_AUDIT_DONE), "v0b_script": rel(V0B_SCRIPT)},
        "prerequisite_headlines": {"v0b": v0b_done.get("headline"), "feature_audit": feature_done.get("headline")},
        "terminal_label_flipped_states": len(flips),
        "oracle_opportunities_vs_fixed_H15": oracle_opp,
        "pass_datasets": pass_datasets,
        "strong_pass_datasets": strong_pass_datasets,
        "terminal_fixed_passes": terminal_fixed_passes,
        "risk_passes": risk_passes,
        "proceed_to_selector_rollout": proceed_to_rollout,
        "decision": decision,
        "results": results,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)

    def metric(obj: Mapping[str, Any], key: str) -> str:
        v = obj.get(key)
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    lines = [
        "# Vehicle true-variable-H state-observable H10/H15 model-CV v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation model-CV diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Candidate state-observable configs per outer fold: `{len(configs)}` (KNN/centroid/positive-radius using only `initial_observation_at_branch` transforms).",
        f"- Terminal-profile label-flipped states inherited from v0b: `{len(flips)}`.",
        f"- Datasets passing both leave-state and leave-case 5% gates: `{pass_datasets}`.",
        f"- Strong 10% passes: `{strong_pass_datasets}`.",
        f"- Terminal-fixed passes: `{terminal_fixed_passes}`.",
        f"- Proceed to selector rollout: `{proceed_to_rollout}`.",
        f"- Decision: {decision}",
        "",
        "## Per-dataset comparison against fixed true H15",
        "",
        "| dataset | n | labels | oracle dec save | oracle physΔ | LS pass | LS dec save | LS physΔ/tol | LS H counts | LC pass | LC dec save | LC physΔ/tol | LC H counts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, res in results.items():
        oracle = res["baselines"]["oracle_H10H15"]
        ls = res["state_observable_nested_cv"]["leave_state_out"]
        lc = res["state_observable_nested_cv"]["leave_case_out"]
        lines.append("| `%s` | %s | `%s` | %s | %s | `%s` | %s | %s/%s | `%s` | `%s` | %s | %s/%s | `%s` |" % (
            name, res.get("n"), res.get("label_counts"), metric(oracle, "decision_relative_saving_vs_H15"), metric(oracle, "physical_delta_vs_H15"),
            bool(ls.get("core_pass_5pct")), metric(ls, "decision_relative_saving_vs_H15"), metric(ls, "physical_delta_vs_H15"), metric(ls, "physical_tolerance_vs_H15"), ls.get("horizon_counts"),
            bool(lc.get("core_pass_5pct")), metric(lc, "decision_relative_saving_vs_H15"), metric(lc, "physical_delta_vs_H15"), metric(lc, "physical_tolerance_vs_H15"), lc.get("horizon_counts"),
        ))
    lines += [
        "",
        "## Interpretation rule frozen before execution",
        "",
        "A selector rollout is allowed only if a state-observable model (not source/role/window/terminal-profile metadata) passes both leave-state and leave-case gates with physical cost within the fixed-H15 tolerance, zero unsafe chosen groups, nonconstant choices, and >=5% measured whole-decision saving. A terminal-fixed pass is required for immediate rollout; pooled-only success remains a diagnostic clue because v0b already showed terminal-profile label dependence.",
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
## 2026-09-29 vehicle true-variable-H state-observable H10/H15 model-CV v0

UTC: {created.isoformat()}. Offline/no-simulation diagnostic using only initial branch observations and online-derived transforms; validation64 and sealed test stayed closed. Candidate configs per outer fold={len(configs)}. Passing datasets={pass_datasets}; strong passes={strong_pass_datasets}; terminal-fixed passes={terminal_fixed_passes}. Proceed to selector rollout={proceed_to_rollout}. Decision: {decision}. Artifacts: `{rel(summary_path)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE_PATH, req, V0B_DONE, FEATURE_AUDIT_DONE, V0B_SCRIPT]
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
        "offline_model_candidate_configs": len(configs),
        "headline": {
            "pass_datasets": pass_datasets,
            "strong_pass_datasets": strong_pass_datasets,
            "terminal_fixed_passes": terminal_fixed_passes,
            "proceed_to_selector_rollout": proceed_to_rollout,
            "decision": decision,
        },
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary_path), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
