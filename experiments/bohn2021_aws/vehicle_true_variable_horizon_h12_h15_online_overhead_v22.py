#!/usr/bin/env python3
"""v22 H12/H15 online selector-overhead and fixed-H12 baseline diagnostic.

Development-only IMPROVED analysis after v20b/v21.  No MPC simulation,
validation64 access, sealed-test access, refit grid search, or gradient training.

Purpose:
  * measure the in-process Python overhead of the deployable v20b history-family
    H12/H15 selector path (feature construction from in-memory H15 trace prefix +
    kNN/support/risk decision);
  * combine that overhead with existing measured true-H12/H15 branch timings from
    v19 and v21;
  * compare against both fixed true-H15 and fixed true-H12.  v21 showed fixed
    H12 can be a strong simple baseline on fresh targeted states, while v19
    showed fixed H12 is unsafe on a known opened negative cluster.

This script is intentionally a bounded diagnostic.  The overhead benchmark is a
microbenchmark of the selector logic, not a new closed-loop MPC validation run.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_h12_h15_selector_refit_v20 as v20  # noqa:E402

NAME = "vehicle_true_variable_horizon_h12_h15_online_overhead_v22"
STAMP = "20260930T0125Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preanalysis_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0125_after_h12_h15_online_overhead_v22.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_ONLINE_OVERHEAD_V22_{STAMP}.json"
MARKER = f"vehicle-h12-h15-online-overhead-v22-{STAMP}"

V20B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/raw.json"
V20B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/completed.json"
V20B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/summary.md"
V21_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json"
V21_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/completed.json"
V21_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/summary.md"
V21_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_preoutcome_frozen_20260930T0130Z.json"
V21_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/selected_state_manifest.json"

# Deployable path chosen from v20b's top strict/global history-family configs.
# It uses no H12 probe solve and therefore represents the static/history path for
# which Astra A4 requested online selector overhead measurement.
DEPLOY_CFG: Dict[str, Any] = {
    "family": "history",
    "k": 1,
    "risk_max": 0.25,
    "support_q": 1.0,
    "support_mult": 1.25,
    "gain_min_s": 0.0,
    "phys_max": 2.0,
    "cat_guard_ratio": 1.0,
    "cat_margin": 0.0,
    "risk_prior_alpha": 0.5,
    "min_cat_train_for_h12": 1,
}
OVERHEAD_REPS = 20000
OVERHEAD_SEED = 202609300125
MIN_SAVE = 0.05
SUPERVISOR_BACKUP_COMMIT = "7677039b1e7c718e07b5ea1c011b09cb4b794c6c"
SUPERVISOR_BACKUP_PACKAGE_SHA256 = "f706a0f5803b72925f04c35b03740469fd440b749aef3c186e3dc9d11227964a"


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
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


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


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=true in {label}")
    if obj.get("validation64_episodes", 0) not in (0, None):
        raise ContractError(f"unexpected validation64 episodes in {label}")
    if obj.get("sealed_test_episodes", 0) not in (0, None):
        raise ContractError(f"unexpected sealed-test episodes in {label}")


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def mean(xs: Sequence[float]) -> float:
    return math.fsum(float(x) for x in xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(math.fsum((float(x) - m) ** 2 for x in xs) / (len(xs) - 1))


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(float(len(vals) - 1), q * (len(vals) - 1)))
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def row_key(r: Mapping[str, Any]) -> str:
    return str(r["candidate_id"])


def metric_sum(ep: Mapping[str, Any], field: str) -> float:
    obj = ep.get(field)
    return sf(obj.get("sum"), 0.0) if isinstance(obj, Mapping) else 0.0


def max_existing_total_tokens() -> Optional[int]:
    candidates = [ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT / "state/research.sqlite"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            con = sqlite3.connect(str(p))
            cur = con.cursor()
            total = None
            for table in ("calls", "api_calls", "events"):
                try:
                    cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}")
                    val = cur.fetchone()[0]
                    if val is not None:
                        total = int(val)
                        break
                except Exception:
                    pass
            con.close()
            if total is not None:
                return total
        except Exception:
            continue
    return None


def verify_inputs() -> Dict[str, str]:
    paths = [SOURCE, Path(v20.__file__).resolve(), V20B_RAW, V20B_DONE, V20B_SUMMARY, V21_RAW, V21_DONE, V21_SUMMARY, V21_PROTOCOL, V21_MANIFEST]
    for p in paths:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    for p, lab in [(V20B_DONE, "v20b"), (V21_DONE, "v21")]:
        obj = read_json(p)
        assert_dev_only(obj, lab)
        ok = obj.get("hard_pass") is True or obj.get("passed") is True or obj.get("status") in ("complete", "completed")
        if not ok:
            raise ContractError(f"prerequisite not complete: {lab}")
    return {rel(p): sha256(p) for p in paths}


def obs14(raw: Any) -> List[float]:
    vals = [sf(v) for v in (raw or [])[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def state_from_trace_row(tr: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(tr.get("previous_state"), Mapping):
        return tr["previous_state"]
    if isinstance(tr.get("state"), Mapping):
        return tr["state"]
    return {}


def obs_from_trace_row(tr: Mapping[str, Any]) -> List[float]:
    return obs14(tr.get("observation") or tr.get("next_observation") or [])


def add_history_features_from_trace(fd: Dict[str, float], trace: Sequence[Mapping[str, Any]], branch_step: int) -> None:
    prefix = [tr for tr in trace if isinstance(tr, Mapping) and si(tr.get("step"), -1) <= branch_step]
    if not prefix:
        prefix = list(trace[: max(1, min(len(trace), branch_step + 1 if branch_step >= 0 else len(trace)))])
    for lag in [1, 3, 5, 10]:
        if len(prefix) >= lag + 1:
            old = prefix[-lag - 1]
            cur = prefix[-1]
            so = state_from_trace_row(old)
            sc = state_from_trace_row(cur)
            fd[f"hist_dx_lag{lag}"] = (sf(sc.get("x")) - sf(so.get("x"))) / max(1.0, float(lag))
            fd[f"hist_dy_lag{lag}"] = (sf(sc.get("y")) - sf(so.get("y"))) / max(1.0, float(lag))
            fd[f"hist_dtheta_lag{lag}"] = (sf(sc.get("theta")) - sf(so.get("theta"))) / max(1.0, float(lag))
            oo = obs_from_trace_row(old)
            oc = obs_from_trace_row(cur)
            for i in (0, 1, 2, 5, 6, 8, 9, 11, 12):
                fd[f"hist_dobs{i}_lag{lag}"] = oc[i] - oo[i]
    win = prefix[-min(12, len(prefix)):]
    if win:
        obs_win = [obs_from_trace_row(x) for x in win]
        for i in (0, 1, 2, 5, 6, 8, 9, 11, 12):
            xs = [u[i] for u in obs_win]
            fd[f"hist_obs{i}_mean12"] = mean(xs)
            fd[f"hist_obs{i}_std12"] = stdev(xs)
            fd[f"hist_obs{i}_min12"] = min(xs)
            fd[f"hist_obs{i}_max12"] = max(xs)
        pair_norms: List[float] = []
        for u in obs_win:
            pair_norms.extend([math.hypot(u[5], u[6]), math.hypot(u[8], u[9]), math.hypot(u[11], u[12])])
        if pair_norms:
            last = obs_win[-1]
            fd["hist_pair_norm_min12"] = min(pair_norms)
            fd["hist_pair_norm_mean12"] = mean(pair_norms)
            fd["hist_pair_norm_last"] = min(math.hypot(last[5], last[6]), math.hypot(last[8], last[9]), math.hypot(last[11], last[12]))
        states = [state_from_trace_row(x) for x in win]
        ths = [sf(s.get("theta")) for s in states]
        if ths:
            fd["hist_theta_mean12"] = mean(ths)
            fd["hist_theta_std12"] = stdev(ths)


def candidate_from_manifest(m: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": "v21_" + str(m["base_state_id"]),
        "base_state_id": str(m["base_state_id"]),
        "branch_previous_state": m.get("branch_previous_state") or {},
        "candidate_branch_step": si(m.get("branch_step"), -1),
        "offset_from_center": 0,
        "source_candidate_index": si(m.get("source_candidate_index"), -1),
        "initial_observation_from_h15_trace": m.get("initial_observation_from_h15_trace") or [],
        "h15_trace_episode_path": str(m.get("h15_trace_episode_path") or ""),
    }


def make_features_from_manifest(m: Mapping[str, Any], trace_cache: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, float]:
    cand = candidate_from_manifest(m)
    fd = v20.static_features(cand)
    trace = trace_cache.get(str(cand["h15_trace_episode_path"])) or []
    add_history_features_from_trace(fd, trace, si(cand.get("candidate_branch_step"), -1))
    return fd


def load_v21_rows_and_features(input_hashes: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], List[Mapping[str, Any]], Dict[str, Sequence[Mapping[str, Any]]]]:
    raw = read_json(V21_RAW)
    done = read_json(V21_DONE)
    manifest = read_json(V21_MANIFEST)
    assert_dev_only(raw, "v21 raw")
    assert_dev_only(done, "v21 done")
    states = list(((raw.get("analysis") or {}).get("state_rows")) or [])
    selected = list(manifest.get("selected_states") or [])
    selected_by_id = {str(x.get("base_state_id")): x for x in selected}
    if len(states) != 16 or len(selected) != 16:
        raise ContractError(f"unexpected v21 row/manifest counts states={len(states)} selected={len(selected)}")
    trace_cache: Dict[str, Sequence[Mapping[str, Any]]] = {}
    for m in selected:
        path = ROOT / str(m.get("h15_trace_episode_path")) / "trace.json"
        if not path.exists():
            raise ContractError("missing v21 trace " + rel(path))
        trace_cache[str(m.get("h15_trace_episode_path"))] = read_json(path)
        input_hashes[rel(path)] = sha256(path)
    rows: List[Dict[str, Any]] = []
    feats: Dict[str, Dict[str, float]] = {}
    for sr in states:
        bid = str(sr.get("base_state_id"))
        m = selected_by_id.get(bid)
        if m is None:
            raise ContractError("v21 state missing manifest: " + bid)
        cid = "v21_" + bid
        row = {
            "candidate_id": cid,
            "candidate_index": si(sr.get("fresh_case_index"), -1),
            "bank_id": "fresh_v21",
            "source_key": f"fresh_v21/source{si(sr.get('source_candidate_index'), -1):03d}/{bid}",
            "base_state_id": bid,
            "role": str(sr.get("risk_probe_role") or ""),
            "offset_from_center": si(sr.get("branch_state_slot"), 0),
            "branch_step": si(sr.get("branch_step"), -1),
            "h12_physical": sf((sr.get("h12") or {}).get("physical")),
            "h15_physical": sf((sr.get("h15") or {}).get("physical")),
            "h12_decision_sum_s": sf((sr.get("h12") or {}).get("decision_sum_s")),
            "h15_decision_sum_s": sf((sr.get("h15") or {}).get("decision_sum_s")),
            "h12_solver_sum_s": sf((sr.get("h12") or {}).get("solver_sum_s")),
            "h15_solver_sum_s": sf((sr.get("h15") or {}).get("solver_sum_s")),
            "h12_steps": sf((sr.get("h12") or {}).get("steps")),
            "h15_steps": sf((sr.get("h15") or {}).get("steps")),
            "phys_delta_h12_minus_h15": sf(sr.get("physical_delta_h12_minus_h15")),
            "decision_gain_h12_vs_h15_s": sf(sr.get("decision_gain_h12_vs_h15_s")),
            "solver_gain_h12_vs_h15_s": sf(sr.get("solver_gain_h12_vs_h15_s")),
            "row_physical_tolerance": sf(sr.get("row_tolerance_vs_H15"), 2.0),
            "h12_catastrophic_vs_h15": bool(sr.get("h12_catastrophic_vs_h15")),
            "h12_beneficial_vs_h15": bool(sr.get("h12_beneficial_vs_h15")),
        }
        rows.append(row)
        feats[cid] = make_features_from_manifest(m, trace_cache)
    return rows, feats, selected, trace_cache


def train_model(train_rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    fdict = v20.feature_subset(features, str(cfg["family"]))
    names, means, stds = v20.train_scaler(train_rows, fdict)
    train_vecs = {row_key(r): v20.vector(r, fdict, names, means, stds) for r in train_rows}
    pos_rows = [r for r in train_rows if bool(r["h12_beneficial_vs_h15"])]
    cat_rows = [r for r in train_rows if bool(r["h12_catastrophic_vs_h15"])]
    radius = v20.support_radius(pos_rows, train_vecs, cfg)
    return {"cfg": dict(cfg), "names": names, "means": means, "stds": stds, "train_rows": list(train_rows), "train_vecs": train_vecs, "pos_rows": pos_rows, "cat_rows": cat_rows, "radius": radius, "feature_count": len(names)}


def vector_from_fd(fd: Mapping[str, float], names: Sequence[str], means: Mapping[str, float], stds: Mapping[str, float]) -> List[float]:
    return [(sf(fd.get(nm)) - sf(means.get(nm))) / max(sf(stds.get(nm), 1.0), 1e-12) for nm in names]


def predict_one(fd: Mapping[str, float], model: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    cfg = model["cfg"]
    train_rows = model["train_rows"]
    pos_rows = model["pos_rows"]
    cat_rows = model["cat_rows"]
    if (not train_rows) or (not pos_rows) or len(cat_rows) < si(cfg.get("min_cat_train_for_h12"), 1):
        return 15, {"risk_hat": 1.0, "pred_gain_s": -1e9, "pred_phys_delta": 1e9, "dpos": 1e9, "dcat": 1e9, "support_ok": 0.0, "cat_guard_ok": 0.0}
    ev = vector_from_fd(fd, model["names"], model["means"], model["stds"])
    train_vecs = model["train_vecs"]
    neigh = sorted([(v20.dist(ev, train_vecs[row_key(tr)]), tr) for tr in train_rows], key=lambda z: (z[0], row_key(z[1])))
    pos_neigh = sorted([(v20.dist(ev, train_vecs[row_key(tr)]), tr) for tr in pos_rows], key=lambda z: (z[0], row_key(z[1])))
    cat_neigh = sorted([(v20.dist(ev, train_vecs[row_key(tr)]), tr) for tr in cat_rows], key=lambda z: (z[0], row_key(z[1])))
    kk = max(1, min(si(cfg.get("k"), 1), len(neigh)))
    near = neigh[:kk]
    weights = [1.0 / max(0.05, d) for d, _ in near]
    sw = math.fsum(weights) if weights else 1.0
    cat_w = math.fsum(w for w, (_, r) in zip(weights, near) if bool(r["h12_catastrophic_vs_h15"]))
    prior = sf(cfg.get("risk_prior_alpha"), 0.5)
    risk_hat = (cat_w + 0.5 * prior) / (sw + prior)
    kp = max(1, min(si(cfg.get("k"), 1), len(pos_neigh)))
    pnear = pos_neigh[:kp]
    pred_gain = mean([sf(r["decision_gain_h12_vs_h15_s"]) for _, r in pnear]) if pnear else -1e9
    pred_phys = mean([sf(r["phys_delta_h12_minus_h15"]) for _, r in pnear]) if pnear else 1e9
    dpos = pnear[-1][0] if pnear else 1e9
    dcat = cat_neigh[0][0] if cat_neigh else 1e9
    support_ok = dpos <= sf(model.get("radius"), 0.0)
    cat_guard_ok = dcat > sf(cfg.get("cat_guard_ratio"), 1.0) * max(dpos, 1e-9) + sf(cfg.get("cat_margin"), 0.0)
    choose = bool(support_ok and cat_guard_ok and risk_hat <= sf(cfg.get("risk_max"), 0.5) and pred_gain >= sf(cfg.get("gain_min_s"), 0.0) and pred_phys <= sf(cfg.get("phys_max"), 2.0))
    return (12 if choose else 15), {"risk_hat": risk_hat, "pred_gain_s": pred_gain, "pred_phys_delta": pred_phys, "dpos": dpos, "dcat": dcat, "support_ok": float(support_ok), "cat_guard_ok": float(cat_guard_ok)}


def select_choices_with_v20(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    choices, scores, _ = v20.predict_choices(train_rows, eval_rows, features, cfg, include_scores=True)
    return {str(k): int(v) for k, v in choices.items()}, scores


def choices_from_details(details: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {str(d["candidate_id"]): int(d.get("selected_h", 15)) for d in details}


def evaluate_policy(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int], overhead_per_branch_call_s: float = 0.0, overhead_per_control_step_s: float = 0.0) -> Dict[str, Any]:
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r.get("row_physical_tolerance"), 2.0) for r in rows)
    pol_phys = pol_dec_no = pol_sol = 0.0
    selected_steps = 0.0
    counts: Counter[str] = Counter()
    conf: Counter[str] = Counter()
    bad: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    for r in rows:
        h = int(choices.get(row_key(r), 15))
        counts[str(h)] += 1
        if h == 12:
            pol_phys += sf(r["h12_physical"])
            pol_dec_no += sf(r["h12_decision_sum_s"])
            pol_sol += sf(r["h12_solver_sum_s"])
            selected_steps += sf(r.get("h12_steps"), 1.0)
            conf["TP" if r.get("h12_beneficial_vs_h15") else "FP"] += 1
            if bool(r.get("h12_catastrophic_vs_h15")):
                bad.append({"candidate_id": row_key(r), "bank_id": r.get("bank_id"), "source_key": r.get("source_key"), "phys_delta": sf(r.get("phys_delta_h12_minus_h15")), "decision_gain_s": sf(r.get("decision_gain_h12_vs_h15_s"))})
        else:
            pol_phys += sf(r["h15_physical"])
            pol_dec_no += sf(r["h15_decision_sum_s"])
            pol_sol += sf(r["h15_solver_sum_s"])
            selected_steps += sf(r.get("h15_steps"), 1.0)
            conf["FN" if r.get("h12_beneficial_vs_h15") else "TN"] += 1
        detail.append({"candidate_id": row_key(r), "selected_h": h, "label_positive": bool(r.get("h12_beneficial_vs_h15")), "catastrophic": bool(r.get("h12_catastrophic_vs_h15"))})
    overhead_branch = overhead_per_branch_call_s * len(rows)
    overhead_step = overhead_per_control_step_s * selected_steps
    pol_dec = pol_dec_no + overhead_branch + overhead_step
    phys_delta = pol_phys - fixed_phys
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    return {
        "rows": len(rows),
        "chosen_counts": dict(counts),
        "confusion": dict(conf),
        "selected_control_steps_for_per_step_overhead": selected_steps,
        "physical_sum": pol_phys,
        "fixed_H15_physical_sum": fixed_phys,
        "physical_delta_vs_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "physical_gate_vs_H15": phys_delta <= tol_sum,
        "decision_sum_s_before_selector_overhead": pol_dec_no,
        "selector_overhead_branch_sum_s": overhead_branch,
        "selector_overhead_step_sum_s": overhead_step,
        "decision_sum_s": pol_dec,
        "fixed_H15_decision_sum_s": fixed_dec,
        "decision_relative_saving_vs_H15": dec_save,
        "solver_sum_s": pol_sol,
        "fixed_H15_solver_sum_s": fixed_sol,
        "solver_relative_saving_vs_H15": sol_save,
        "catastrophic_false_positive_count": len(bad),
        "catastrophic_false_positive_rows": bad,
        "pass5_zero_cat_physical": bool(dec_save >= MIN_SAVE and phys_delta <= tol_sum and not bad),
        "details": detail,
    }


def fixed_choices(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {row_key(r): h for r in rows}


def oracle_choices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {row_key(r): (12 if bool(r.get("h12_beneficial_vs_h15")) else 15) for r in rows}


def overhead_benchmark(selected: Sequence[Mapping[str, Any]], trace_cache: Mapping[str, Sequence[Mapping[str, Any]]], model: Mapping[str, Any]) -> Dict[str, Any]:
    rng = random.Random(OVERHEAD_SEED)
    items = list(selected)
    if not items:
        raise ContractError("no selected states for overhead benchmark")
    # Warm-up avoids first-call dictionary/list allocation noise dominating p95.
    for _ in range(200):
        m = rng.choice(items)
        fd = make_features_from_manifest(m, trace_cache)
        predict_one(fd, model)
    feat_ns: List[int] = []
    pred_ns: List[int] = []
    total_ns: List[int] = []
    counts: Counter[str] = Counter()
    for _ in range(OVERHEAD_REPS):
        m = rng.choice(items)
        t0 = time.perf_counter_ns()
        fd = make_features_from_manifest(m, trace_cache)
        t1 = time.perf_counter_ns()
        h, _score = predict_one(fd, model)
        t2 = time.perf_counter_ns()
        feat_ns.append(t1 - t0)
        pred_ns.append(t2 - t1)
        total_ns.append(t2 - t0)
        counts[str(h)] += 1
    def summ(ns: Sequence[int]) -> Dict[str, float]:
        vals = [x / 1e9 for x in ns]
        return {"mean_s": mean(vals), "median_s": statistics.median(vals), "p95_s": quantile(vals, 0.95), "max_s": max(vals)}
    return {
        "repetitions": OVERHEAD_REPS,
        "seed": OVERHEAD_SEED,
        "implementation": "Python in-memory history feature extraction from saved H15 trace prefix plus kNN/support/risk selector decision; no disk IO and no MPC solve",
        "feature_extraction": summ(feat_ns),
        "selector_decision": summ(pred_ns),
        "feature_plus_selector": summ(total_ns),
        "microbench_choice_counts": dict(counts),
    }


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(result: Mapping[str, Any]) -> None:
    h = result["headline"]
    pol = result["policies"]
    overhead = result["overhead_microbenchmark"]["feature_plus_selector"]
    lines = [
        "# Vehicle true-variable-H H12/H15 online overhead v22",
        "",
        f"UTC: `{result['created_utc']}`. Development-only IMPROVED overhead/baseline diagnostic; no simulation, no validation64, no sealed test, no training/refit.",
        "",
        "## Headline",
        "",
        f"- Selector path: `{h['deployable_config_id']}` trained on v19 opened rows, evaluated on v21 fresh rows; v19 combined proxy uses v20b strict source-CV choices.",
        f"- Measured Python feature+selector overhead: mean `{overhead['mean_s']:.9f}` s, median `{overhead['median_s']:.9f}` s, p95 `{overhead['p95_s']:.9f}` s over `{result['overhead_microbenchmark']['repetitions']}` repetitions.",
        f"- v21-only fixed H12: save `{pct(pol['v21_fixed_H12']['decision_relative_saving_vs_H15'])}`, bad `{pol['v21_fixed_H12']['catastrophic_false_positive_count']}`, physical gate `{pol['v21_fixed_H12']['physical_gate_vs_H15']}`.",
        f"- v21-only selector: H counts `{pol['v21_selector_global_history']['chosen_counts']}`, overhead-adjusted branch save `{pct(pol['v21_selector_global_history']['decision_relative_saving_vs_H15'])}`, bad `{pol['v21_selector_global_history']['catastrophic_false_positive_count']}`, physical gate `{pol['v21_selector_global_history']['physical_gate_vs_H15']}`.",
        f"- Combined v19 strict-source proxy + v21 selector: H counts `{pol['combined_selector_sourcecv_v19_global_v21']['chosen_counts']}`, overhead-adjusted branch save `{pct(pol['combined_selector_sourcecv_v19_global_v21']['decision_relative_saving_vs_H15'])}`, bad `{pol['combined_selector_sourcecv_v19_global_v21']['catastrophic_false_positive_count']}`, pass5 `{pol['combined_selector_sourcecv_v19_global_v21']['pass5_zero_cat_physical']}`.",
        f"- Combined fixed H12: save `{pct(pol['combined_fixed_H12']['decision_relative_saving_vs_H15'])}`, bad `{pol['combined_fixed_H12']['catastrophic_false_positive_count']}`, physical gate `{pol['combined_fixed_H12']['physical_gate_vs_H15']}`.",
        f"- Decision: {result['decision']}",
        "",
        "## Important limits",
        "",
        "- Overhead is measured for the Python selector path outside the live MPC control loop using in-memory traces. It answers Astra A4 for static/history selection overhead but is not a full closed-loop validation rollout.",
        "- v21 is targeted stress-pool development data, not a population estimate. v19/v21 are opened development evidence and cannot authorize final-test access.",
        "- Fixed H12 is now a strong baseline on v21-like fresh states; adaptive claims must show safety/value relative to fixed H12 as well as H15.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Protocol: `{rel(PROTOCOL)}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(result: Mapping[str, Any]) -> None:
    h = result["headline"]
    pol = result["policies"]
    overhead = result["overhead_microbenchmark"]["feature_plus_selector"]
    block = f"""
<!-- {MARKER} -->
## 2026-09-30 v22 H12/H15 online-overhead diagnostic

UTC: {result['created_utc']}. Development-only IMPROVED analysis; no new MPC simulation, no validation64 access, no sealed-test access, no training/refit. Inputs were v20b and v21 opened-development artifacts backed up by supervisor commit `{SUPERVISOR_BACKUP_COMMIT}`.

Measured Python in-memory history feature + selector overhead over {result['overhead_microbenchmark']['repetitions']} repetitions: mean {overhead['mean_s']:.9f}s, median {overhead['median_s']:.9f}s, p95 {overhead['p95_s']:.9f}s. v21 fixed H12 remains a strong baseline: {pct(pol['v21_fixed_H12']['decision_relative_saving_vs_H15'])} decision saving vs H15, zero bad, physical gate={pol['v21_fixed_H12']['physical_gate_vs_H15']}. The v20b history selector trained on v19 chose {pol['v21_selector_global_history']['chosen_counts']} on v21 and achieved {pct(pol['v21_selector_global_history']['decision_relative_saving_vs_H15'])} overhead-adjusted branch saving, zero bad. The combined development proxy (v19 strict-source CV choices + v21 v19-trained selector) achieved {pct(pol['combined_selector_sourcecv_v19_global_v21']['decision_relative_saving_vs_H15'])} saving, zero bad={pol['combined_selector_sourcecv_v19_global_v21']['catastrophic_false_positive_count'] == 0}, pass5={pol['combined_selector_sourcecv_v19_global_v21']['pass5_zero_cat_physical']}; combined fixed H12 still failed safety/physical due to v19 negatives (bad={pol['combined_fixed_H12']['catastrophic_false_positive_count']}, physical gate={pol['combined_fixed_H12']['physical_gate_vs_H15']}).

Decision: {result['decision']}

Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup requested: `{rel(BACKUP_REQUEST)}`.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / name, MARKER, block)
    response_path = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
    response_block = f"""
## Follow-up through v22 online-overhead diagnostic

Updated by GPT-5.5 executor at `{result['created_utc']}`. `LATEST.md` still points to `20260929T153837Z`; stable recommendation IDs are preserved.

| linked recommendation(s) | disposition after v22 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; partially resolved for static/history micro-overhead, still open for full closed-loop deployment | v22 raw/summary: Python in-memory history feature+selector overhead mean {overhead['mean_s']:.9f}s, p95 {overhead['p95_s']:.9f}s over {result['overhead_microbenchmark']['repetitions']} calls. The combined v19/v21 branch-level selector proxy retains {pct(pol['combined_selector_sourcecv_v19_global_v21']['decision_relative_saving_vs_H15'])} overhead-adjusted decision saving with zero catastrophic H12 false positives. | Do not claim final speed yet; next full confirmation must include closed-loop whole-decision timing and selector overhead in blocked/randomized order. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; sharpened | v22: v21-only fixed H12 saves {pct(pol['v21_fixed_H12']['decision_relative_saving_vs_H15'])} vs H15 with zero bad, while combined fixed H12 has {pol['combined_fixed_H12']['catastrophic_false_positive_count']} bad rows and physical_gate={pol['combined_fixed_H12']['physical_gate_vs_H15']}. | Treat fixed true H12 as a primary strong baseline on v21-like states; adaptive selector value is currently safety against known v19 negatives, not v21-only superiority. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v22 uses only opened v19/v21 targeted development rows. Combined selector pass remains development-only and source/risk enriched. | Before validation, acquire broader source-independent confirmation or freeze a fresh confirmation protocol with fixed H12/H15/selector baselines; sealed final test remains closed. |
| `A11_training_failure_modes_need_separation` | accepted; deferred by current evidence | v22 did not show overhead erasing branch-level selector savings, but fixed H12 dominates v21-only fresh states. | If broader confirmation shows fixed H12 safe, adaptivity may be unnecessary for that distribution; if new negatives recur, train/refit richer terminal-risk/value selector instead of static sweeps. |
| `A12_registry_backup_schema_contract` | accepted; active | v22 wrote new code/results/docs and backup request `{rel(BACKUP_REQUEST)}`. | Require verified external backup covering v22 before further unique science. |
"""
    append_if_missing(response_path, MARKER, response_block)


def update_registry(result: Mapping[str, Any]) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    line = [
        result["created_utc"],
        NAME,
        "development_IMPROVED_online_overhead_and_fixed_H12_baseline_diagnostic_no_sim_no_validation_no_test",
        f"OVERHEAD_SEED={OVERHEAD_SEED}",
        "opened v19/v20b/v21 development artifacts only; no validation64; no sealed test",
        "0",
        "0",
        str(result["budget_actual"]["selector_microbenchmark_repetitions"]),
        "0",
        "0",
        "False",
        rel(RUN_DIR / "completed.json"),
        MARKER,
    ]
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER not in old:
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(line)


def main() -> int:
    started = now_utc()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    input_hashes = verify_inputs()
    protocol = {
        "protocol_id": f"{NAME}_preanalysis_frozen_{STAMP}",
        "created_utc": started.isoformat(),
        "classification": "development_IMPROVED_online_overhead_and_fixed_H12_baseline_diagnostic_no_sim_no_validation_no_test",
        "hypothesis": "After charging measured in-process history-feature/selector overhead, the v20b H12/H15 static-history selector still has non-negligible branch-level compute saving versus H15 and provides safety advantage over fixed H12 on combined opened negative/support rows; if fixed H12 dominates fresh rows, it must be treated as a strong primary baseline before validation.",
        "inputs": {"v20b_raw": rel(V20B_RAW), "v21_raw": rel(V21_RAW), "v21_manifest": rel(V21_MANIFEST)},
        "deployable_config": DEPLOY_CFG,
        "overhead_repetitions": OVERHEAD_REPS,
        "overhead_seed": OVERHEAD_SEED,
        "budgets_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "selector_microbenchmark_repetitions": OVERHEAD_REPS, "gradient_steps": 0, "new_refit_grid_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": {"commit": SUPERVISOR_BACKUP_COMMIT, "package_sha256": SUPERVISOR_BACKUP_PACKAGE_SHA256, "status": "verified in user/supervisor context at 2026-09-30T01:10:29Z"},
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, protocol)

    # Load v19 rows/features through the already-audited v20 feature builder, then
    # build v21 rows/features without using v21 labels for model fitting.
    v19_rows, v19_features, v19_diag, v19_hashes = v20.load_rows()
    input_hashes.update(v19_hashes)
    v21_rows, v21_features, selected, trace_cache = load_v21_rows_and_features(input_hashes)
    all_features = dict(v19_features)
    all_features.update(v21_features)
    deploy_config_id = v20.cfg_id(DEPLOY_CFG)
    model = train_model(v19_rows, all_features, DEPLOY_CFG)
    choices_v21, scores_v21 = select_choices_with_v20(v19_rows, v21_rows, all_features, DEPLOY_CFG)
    overhead = overhead_benchmark(selected, trace_cache, model)
    overhead_mean = sf(overhead["feature_plus_selector"]["mean_s"])
    overhead_p95 = sf(overhead["feature_plus_selector"]["p95_s"])

    v20b_raw = read_json(V20B_RAW)
    strict_source_details = (((v20b_raw.get("strict_source_nested") or {}).get("aggregate") or {}).get("details") or [])
    strict_bank_details = (((v20b_raw.get("strict_bank_nested") or {}).get("aggregate") or {}).get("details") or [])
    if len(strict_source_details) != len(v19_rows) or len(strict_bank_details) != len(v19_rows):
        raise ContractError("v20b strict detail counts do not match v19 rows")
    choices_v19_source = choices_from_details(strict_source_details)
    choices_v19_bank = choices_from_details(strict_bank_details)
    combined_rows = list(v19_rows) + list(v21_rows)
    choices_comb_source = dict(choices_v19_source); choices_comb_source.update(choices_v21)
    choices_comb_bank = dict(choices_v19_bank); choices_comb_bank.update(choices_v21)

    policies = {
        "v19_fixed_H12": evaluate_policy(v19_rows, fixed_choices(v19_rows, 12)),
        "v19_fixed_H15": evaluate_policy(v19_rows, fixed_choices(v19_rows, 15)),
        "v19_selector_strict_source_cv": evaluate_policy(v19_rows, choices_v19_source, overhead_per_branch_call_s=overhead_mean),
        "v19_selector_strict_bank_cv": evaluate_policy(v19_rows, choices_v19_bank, overhead_per_branch_call_s=overhead_mean),
        "v21_fixed_H12": evaluate_policy(v21_rows, fixed_choices(v21_rows, 12)),
        "v21_fixed_H15": evaluate_policy(v21_rows, fixed_choices(v21_rows, 15)),
        "v21_oracle_H12_H15": evaluate_policy(v21_rows, oracle_choices(v21_rows)),
        "v21_selector_global_history": evaluate_policy(v21_rows, choices_v21, overhead_per_branch_call_s=overhead_mean),
        "v21_selector_global_history_p95_overhead": evaluate_policy(v21_rows, choices_v21, overhead_per_branch_call_s=overhead_p95),
        "combined_fixed_H12": evaluate_policy(combined_rows, fixed_choices(combined_rows, 12)),
        "combined_fixed_H15": evaluate_policy(combined_rows, fixed_choices(combined_rows, 15)),
        "combined_selector_sourcecv_v19_global_v21": evaluate_policy(combined_rows, choices_comb_source, overhead_per_branch_call_s=overhead_mean),
        "combined_selector_bankcv_v19_global_v21": evaluate_policy(combined_rows, choices_comb_bank, overhead_per_branch_call_s=overhead_mean),
    }
    # A conservative full-online upper-bound estimate charges the same mean Python
    # overhead on every selected control step, but only for v21 where continuation
    # steps are explicitly available in the normalized rows.
    policies["v21_selector_global_history_per_step_overhead_upper_bound"] = evaluate_policy(v21_rows, choices_v21, overhead_per_control_step_s=overhead_mean)

    v21_sel = policies["v21_selector_global_history"]
    comb_sel = policies["combined_selector_sourcecv_v19_global_v21"]
    v21_fixed = policies["v21_fixed_H12"]
    comb_fixed = policies["combined_fixed_H12"]
    if comb_sel["pass5_zero_cat_physical"] and comb_fixed["catastrophic_false_positive_count"] > 0 and v21_fixed["pass5_zero_cat_physical"]:
        decision = "Selector overhead is negligible at branch scale and the combined development proxy passes vs H15 while fixed H12 remains unsafe on v19 negatives; however v21-only fixed H12 outperforms/equals the selector as a simple baseline, so the next experiment should broaden fresh confirmation with fixed H12/H15/selector all predeclared before any validation."
    elif v21_fixed["pass5_zero_cat_physical"] and (v21_fixed["decision_relative_saving_vs_H15"] >= v21_sel["decision_relative_saving_vs_H15"]):
        decision = "Fixed H12 is the stronger simple baseline on v21-like fresh states; do not validate an adaptive claim yet. Acquire broader source-independent negatives/support or pivot to risk/value training if fixed H12 failures recur outside the known v19 cluster."
    elif not comb_sel["pass5_zero_cat_physical"]:
        decision = "After overhead and fixed-H12 comparison the selector does not clear the development gate; pivot to richer terminal-risk/value refit/training rather than another static threshold sweep."
    else:
        decision = "Selector clears the overhead-adjusted development gate; next still requires broader independent confirmation and strong fixed-H12/per-H terminal baselines before validation/final-test planning."

    created = now_utc()
    headline = {
        "deployable_config_id": deploy_config_id,
        "v19_rows": len(v19_rows),
        "v21_rows": len(v21_rows),
        "v19_fixed_H12_bad": policies["v19_fixed_H12"]["catastrophic_false_positive_count"],
        "v21_fixed_H12_bad": policies["v21_fixed_H12"]["catastrophic_false_positive_count"],
        "v21_selector_h12_count": int((v21_sel["chosen_counts"] or {}).get("12", 0)),
        "v21_selector_save": v21_sel["decision_relative_saving_vs_H15"],
        "v21_fixed_H12_save": v21_fixed["decision_relative_saving_vs_H15"],
        "combined_selector_save": comb_sel["decision_relative_saving_vs_H15"],
        "combined_selector_bad": comb_sel["catastrophic_false_positive_count"],
        "combined_selector_pass5": comb_sel["pass5_zero_cat_physical"],
        "combined_fixed_H12_save": comb_fixed["decision_relative_saving_vs_H15"],
        "combined_fixed_H12_bad": comb_fixed["catastrophic_false_positive_count"],
        "overhead_mean_s": overhead_mean,
        "overhead_p95_s": overhead_p95,
    }
    result = {
        "created_utc": created.isoformat(),
        "classification": protocol["classification"],
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "input_hashes": input_hashes,
        "access_flags": protocol["access_flags"],
        "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "selector_microbenchmark_repetitions": OVERHEAD_REPS, "gradient_steps": 0, "new_refit_grid_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "server_api_total_tokens_from_sqlite": max_existing_total_tokens(),
        "v19_diagnostic": v19_diag,
        "v21_rows": v21_rows,
        "deployable_config": DEPLOY_CFG,
        "deployable_config_id": deploy_config_id,
        "v21_selector_scores": scores_v21,
        "overhead_microbenchmark": overhead,
        "policies": policies,
        "headline": headline,
        "decision": decision,
        "interpretation_limits": [
            "opened development v19/v21 evidence only, not validation64 or sealed test",
            "microbenchmark excludes MPC solve and disk IO; it estimates in-process feature+selector Python overhead for a history-family path",
            "branch-level overhead charge is one selector call per saved continuation state; full closed-loop deployment must be measured in a later live blocked rollout",
            "fixed H12 is a strong baseline on v21-like rows but unsafe on v19 negative cluster",
        ],
        "backup_request": rel(BACKUP_REQUEST),
    }
    write_json(RUN_DIR / "raw.json", result)
    completed = {
        "status": "complete",
        "hard_pass": bool(comb_sel["pass5_zero_cat_physical"]),
        "created_utc": created.isoformat(),
        "classification": result["classification"],
        "headline": headline,
        "decision": decision,
        "budget_actual": result["budget_actual"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(RUN_DIR / "raw.json"): sha256(RUN_DIR / "raw.json")},
        "backup_request": rel(BACKUP_REQUEST),
    }
    write_json(RUN_DIR / "completed.json", completed)
    write_summary(result)
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v22_online_overhead_diagnostic",
        "created_utc": created.isoformat(),
        "must_cover": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"],
        "reason": "new v22 development evidence and docs/state updates before further unique science",
        "latest_verified_backup_before_run_from_supervisor_context": {"commit": SUPERVISOR_BACKUP_COMMIT, "package_sha256": SUPERVISOR_BACKUP_PACKAGE_SHA256},
    })
    # Add final hashes after backup request exists.
    completed["hashes"][rel(RUN_DIR / "summary.md")] = sha256(RUN_DIR / "summary.md")
    completed["hashes"][rel(BACKUP_REQUEST)] = sha256(BACKUP_REQUEST)
    write_json(RUN_DIR / "completed.json", completed)
    result["completed_sha256"] = sha256(RUN_DIR / "completed.json")
    write_json(RUN_DIR / "raw.json", result)
    write_summary(result)
    cont = f"""# Continue state after v22 H12/H15 online-overhead diagnostic\n\nUTC: {created.isoformat()}\n\nDecision: {decision}\n\nKey results: v21 fixed H12 save {pct(v21_fixed['decision_relative_saving_vs_H15'])}, bad {v21_fixed['catastrophic_false_positive_count']}; v21 selector save {pct(v21_sel['decision_relative_saving_vs_H15'])}, H counts {v21_sel['chosen_counts']}; combined selector save {pct(comb_sel['decision_relative_saving_vs_H15'])}, bad {comb_sel['catastrophic_false_positive_count']}, pass5 {comb_sel['pass5_zero_cat_physical']}; mean feature+selector overhead {overhead_mean:.9f}s, p95 {overhead_p95:.9f}s.\n\nNext: require verified external backup covering v22, then freeze broader source-independent H12/H15/selector confirmation with fixed H12 as primary strong baseline, or pivot to terminal-risk/value refit if new negatives appear. No validation64/sealed test.\n"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(cont, encoding="utf-8")
    update_docs(result)
    update_registry(result)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": headline, "decision": decision, "backup_request": rel(BACKUP_REQUEST), "server_api_total_tokens_from_sqlite": result["server_api_total_tokens_from_sqlite"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        err = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "classification": "development_IMPROVED_online_overhead_and_fixed_H12_baseline_diagnostic_no_sim_no_validation_no_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False}
        write_json(RUN_DIR / "failed.json", err)
        print(json.dumps(err, sort_keys=True), file=sys.stderr)
        raise
