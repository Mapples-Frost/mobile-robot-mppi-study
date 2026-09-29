#!/usr/bin/env python3
"""v16 boundary-augmented conservative risk/value refit for true-variable H10/H15.

Development-only IMPROVED diagnostic after v15 local boundary acquisition.  v15
produced safe-H15 paired H10/H15 labels around the two v14 false-positive risk
boundaries and their nearest safe/catastrophic lookalikes.  This script tests the
next falsifiable hypothesis:

    Adding those boundary labels to the opened development bank can repair the
    current calibrated risk/value selector in a strict held-out-bank protocol
    without leaking the same bank/source boundary rows into training.

The decision-relevant result is the strict nested opened-bank holdout over the
augmented rows.  In-sample/local repair and global LOBO rankings are diagnostic
only.  No MPC simulation, no gradient/RL training, no validation64 access and no
sealed-test access are performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_history_feature_refit_v13 as v13  # noqa:E402

NAME = "vehicle_true_variable_horizon_boundary_augmented_refit_v16"
STAMP = "20260929T2315Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2315_after_boundary_augmented_refit_v16.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_BOUNDARY_AUGMENTED_REFIT_V16_{STAMP}.json"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-boundary-augmented-refit-v16-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V15_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/raw.json"
V15_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/completed.json"
V15_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/summary.md"
V15_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_preoutcome_frozen_20260929T2325Z.json"

FAMILIES = ["base_no_history", "history_no_risk", "history_with_risk"]
MIN_ROW_TOL = 2.0


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {label}")


def row_key(row: Mapping[str, Any]) -> str:
    return str(row["unique_row_id"])


def mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs) if xs else 0.0


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, int(math.ceil(float(q) * len(vals))) - 1))
    return vals[idx]


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def parse_slot(text: str) -> int:
    if "slot1" in text:
        return 1
    if "slot0" in text:
        return 0
    return 0


def obs14_from_raw(raw: Any) -> List[float]:
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
    return obs14_from_raw(tr.get("observation") or tr.get("next_observation") or [])


def base_feature_dict(row: Mapping[str, Any]) -> Dict[str, float]:
    o = obs14_from_raw(row.get("obs14"))
    th = sf(row.get("prev_theta"))
    fd: Dict[str, float] = {
        "prev_x_30": sf(row.get("prev_x")) / 30.0,
        "prev_y_30": sf(row.get("prev_y")) / 30.0,
        "theta_sin": math.sin(th),
        "theta_cos": math.cos(th),
        "abs_theta_pi": abs(th) / math.pi,
        "branch_step_150": max(0.0, sf(row.get("branch_step"))) / 150.0,
        "slot": sf(row.get("branch_state_slot")),
        "trace_risk_20": sf(row.get("stage_a_trace_risk_score")) / 20.0,
        "obs01_norm": math.hypot(o[0], o[1]),
        "obs56_norm": math.hypot(o[5], o[6]),
        "obs89_norm": math.hypot(o[8], o[9]),
        "obs1112_norm": math.hypot(o[11], o[12]),
    }
    for i in range(14):
        fd[f"obs_{i}"] = o[i]
        fd[f"abs_obs_{i}"] = abs(o[i])
    for a, b in [(0, 1), (0, 2), (1, 2), (5, 6), (8, 9), (11, 12), (7, 10), (10, 13)]:
        fd[f"obs_{a}_x_obs_{b}"] = o[a] * o[b]
    return fd


def add_history_features(fd: Dict[str, float], trace_path: Path, branch_step: int) -> bool:
    if not trace_path.exists():
        return False
    trace = read_json(trace_path)
    if not isinstance(trace, list) or not trace:
        return False
    prefix = [tr for tr in trace if isinstance(tr, Mapping) and si(tr.get("step"), -1) <= branch_step]
    if not prefix:
        prefix = trace[: max(1, min(len(trace), branch_step + 1 if branch_step >= 0 else len(trace)))]
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
            m = mean(xs)
            fd[f"hist_obs{i}_mean12"] = m
            fd[f"hist_obs{i}_std12"] = math.sqrt(math.fsum((x - m) * (x - m) for x in xs) / max(1, len(xs) - 1))
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
            mt = mean(ths)
            fd["hist_theta_mean12"] = mt
            fd["hist_theta_std12"] = math.sqrt(math.fsum((x - mt) * (x - mt) for x in ths) / max(1, len(ths) - 1))
    return True


def family_features(feature_by_key: Mapping[str, Mapping[str, float]], family: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, fd0 in feature_by_key.items():
        fd = dict(fd0)
        if family == "base_no_history":
            fd = {n: v for n, v in fd.items() if not n.startswith("hist_")}
        elif family == "history_no_risk":
            fd = {n: v for n, v in fd.items() if n != "trace_risk_20"}
        elif family == "history_with_risk":
            pass
        else:
            raise ContractError("unknown family " + family)
        out[k] = fd
    return out


def train_scaler(train_rows: Sequence[Mapping[str, Any]], fdict: Mapping[str, Mapping[str, float]]) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    names = sorted(set().union(*(fdict[row_key(r)].keys() for r in train_rows))) if train_rows else []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for nm in names:
        xs = [sf(fdict[row_key(r)].get(nm)) for r in train_rows]
        m = mean(xs)
        var = math.fsum((x - m) * (x - m) for x in xs) / max(1, len(xs) - 1)
        means[nm] = m
        stds[nm] = math.sqrt(var) if var > 1e-18 else 1.0
    return names, means, stds


def vec(row: Mapping[str, Any], fdict: Mapping[str, Mapping[str, float]], names: Sequence[str], means: Mapping[str, float], stds: Mapping[str, float]) -> List[float]:
    fd = fdict[row_key(row)]
    return [(sf(fd.get(nm)) - sf(means.get(nm))) / max(sf(stds.get(nm), 1.0), 1e-12) for nm in names]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def base_estimate(eval_vec: Sequence[float], train_rows: Sequence[Mapping[str, Any]], train_vecs: Mapping[str, Sequence[float]], k: int) -> Dict[str, float]:
    if not train_rows:
        return {"risk_hat": 1.0, "pred_gain": -1e9, "pred_phys": 1e9, "dpos": 1e9, "dcat": 0.0, "n_used": 0.0}
    neigh = sorted([(dist(eval_vec, train_vecs[row_key(r)]), r) for r in train_rows], key=lambda z: (z[0], row_key(z[1])))
    kk = max(1, min(int(k), len(neigh)))
    near = neigh[:kk]
    weights = [1.0 / max(0.05, sf(d)) for d, _ in near]
    sw = math.fsum(weights) if weights else 1.0
    cat_w = math.fsum(w for w, (_, r) in zip(weights, near) if bool(r["h10_catastrophic_vs_h15"]))
    alpha = 0.5
    risk_hat = (cat_w + alpha) / (sw + 2.0 * alpha)
    pred_gain = math.fsum(w * sf(r["decision_gain_h10_vs_h15_s"]) for w, (_, r) in zip(weights, near)) / sw
    pred_phys = math.fsum(w * sf(r["phys_delta_h10_minus_h15"]) for w, (_, r) in zip(weights, near)) / sw
    pos = [(d, r) for d, r in neigh if bool(r["h10_beneficial_vs_h15"])]
    cat = [(d, r) for d, r in neigh if bool(r["h10_catastrophic_vs_h15"])]
    return {
        "risk_hat": risk_hat,
        "pred_gain": pred_gain,
        "pred_phys": pred_phys,
        "dpos": sf(pos[0][0], 1e9) if pos else 1e9,
        "dcat": sf(cat[0][0], 1e9) if cat else 1e9,
        "n_used": kk,
    }


def cfg_grid() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for k in [1, 3, 5]:
            for calib_q in [0.50, 0.75, 0.90]:
                for risk_ucb_max in [0.15, 0.25, 0.40, 0.60]:
                    for phys_ucb_max in [2.0, 6.0]:
                        for gain_lcb_min in [0.0, 0.25, 0.50]:
                            for support_mult in [0.75, 1.25]:
                                cfgs.append({
                                    "family": family,
                                    "k": k,
                                    "calib_q": calib_q,
                                    "risk_ucb_max": risk_ucb_max,
                                    "phys_ucb_max": phys_ucb_max,
                                    "gain_lcb_min": gain_lcb_min,
                                    "support_mult": support_mult,
                                })
    return cfgs


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v16_%s_k%s_q%s_ru%s_pu%s_gl%s_sm%s" % (
        c["family"], c["k"], c["calib_q"], c["risk_ucb_max"], c["phys_ucb_max"], c["gain_lcb_min"], c["support_mult"]
    )


def support_radius(train_rows: Sequence[Mapping[str, Any]], train_vecs: Mapping[str, Sequence[float]], support_mult: float) -> float:
    pos = [r for r in train_rows if bool(r["h10_beneficial_vs_h15"])]
    pp: List[float] = []
    for i, r in enumerate(pos):
        ds = [dist(train_vecs[row_key(r)], train_vecs[row_key(q)]) for j, q in enumerate(pos) if j != i]
        if ds:
            pp.append(min(ds))
    if not pp:
        return 1e9
    pp = sorted(pp)
    idx = max(0, min(len(pp) - 1, int(math.floor(0.50 * (len(pp) - 1)))))
    return sf(pp[idx]) * sf(support_mult, 1.0)


def calibration(train_rows: Sequence[Mapping[str, Any]], train_vecs: Mapping[str, Sequence[float]], k: int, calib_q: float) -> Dict[str, float]:
    risk_resid: List[float] = []
    phys_resid: List[float] = []
    gain_resid: List[float] = []
    for r in train_rows:
        # Leave same source_key out where possible; otherwise leave the exact row out.
        source = str(r.get("source_key", row_key(r)))
        sub = [q for q in train_rows if str(q.get("source_key", row_key(q))) != source]
        if not sub:
            sub = [q for q in train_rows if row_key(q) != row_key(r)]
        if not sub:
            continue
        est = base_estimate(train_vecs[row_key(r)], sub, train_vecs, k)
        y_cat = 1.0 if bool(r["h10_catastrophic_vs_h15"]) else 0.0
        risk_resid.append(max(0.0, y_cat - sf(est["risk_hat"])))
        phys_resid.append(max(0.0, sf(r["phys_delta_h10_minus_h15"]) - sf(est["pred_phys"])))
        gain_resid.append(max(0.0, sf(est["pred_gain"]) - sf(r["decision_gain_h10_vs_h15_s"])))
    return {
        "risk_q": quantile(risk_resid, calib_q),
        "phys_q": quantile(phys_resid, calib_q),
        "gain_q": quantile(gain_resid, calib_q),
        "n_calibration": len(risk_resid),
        "risk_resid_mean": mean(risk_resid),
        "phys_resid_mean": mean(phys_resid),
        "gain_resid_mean": mean(gain_resid),
    }


def predict_split(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], feature_by_key: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], include_scores: bool = False) -> Tuple[Dict[str, int], Dict[str, Any], int]:
    family = str(cfg["family"])
    fdict = family_features(feature_by_key, family)
    names, means, stds = train_scaler(train_rows, fdict)
    train_vecs = {row_key(r): vec(r, fdict, names, means, stds) for r in train_rows}
    eval_vecs = {row_key(r): vec(r, fdict, names, means, stds) for r in eval_rows}
    k = si(cfg["k"], 3)
    cal = calibration(train_rows, train_vecs, k, sf(cfg["calib_q"], 0.75))
    radius = support_radius(train_rows, train_vecs, sf(cfg["support_mult"], 1.0))
    choices: Dict[str, int] = {}
    scores: Dict[str, Any] = {}
    for r in eval_rows:
        est = base_estimate(eval_vecs[row_key(r)], train_rows, train_vecs, k)
        risk_ucb = min(1.0, sf(est["risk_hat"], 1.0) + sf(cal["risk_q"]))
        phys_ucb = sf(est["pred_phys"], 1e9) + sf(cal["phys_q"])
        gain_lcb = sf(est["pred_gain"], -1e9) - sf(cal["gain_q"])
        support_ok = sf(est["dpos"], 1e9) <= radius
        choose = bool(
            support_ok
            and risk_ucb <= sf(cfg["risk_ucb_max"])
            and phys_ucb <= sf(cfg["phys_ucb_max"])
            and gain_lcb >= sf(cfg["gain_lcb_min"])
        )
        choices[row_key(r)] = 10 if choose else 15
        if include_scores:
            scores[row_key(r)] = {
                "risk_hat": est["risk_hat"], "risk_ucb": risk_ucb,
                "pred_phys": est["pred_phys"], "phys_ucb": phys_ucb,
                "pred_gain": est["pred_gain"], "gain_lcb": gain_lcb,
                "dpos": est["dpos"], "dcat": est["dcat"], "support_radius": radius,
                "support_ok": support_ok, "selected_h": choices[row_key(r)],
                "calibration": cal, "feature_count": len(names),
            }
    return choices, scores, len(names)


def eval_choices(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int]) -> Dict[str, Any]:
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0
    counts: Counter = Counter()
    conf: Counter = Counter()
    bad: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for r in rows:
        h = int(choices.get(row_key(r), 15))
        counts[str(h)] += 1
        pos = bool(r["h10_beneficial_vs_h15"])
        cat = bool(r["h10_catastrophic_vs_h15"])
        if h == 10:
            pol_phys += sf(r["h10_physical"]); pol_dec += sf(r["h10_decision_sum_s"]); pol_sol += sf(r["h10_solver_sum_s"])
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({
                    "unique_row_id": row_key(r),
                    "bank_id": r.get("bank_id"),
                    "base_state_id": r.get("base_state_id"),
                    "source_key": r.get("source_key"),
                    "row_origin": r.get("row_origin"),
                    "boundary_role": r.get("boundary_role"),
                    "offset_from_center": r.get("offset_from_center"),
                    "group": r.get("group"),
                    "window": r.get("window"),
                    "phys_delta": sf(r["phys_delta_h10_minus_h15"]),
                    "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"]),
                    "tol": sf(r["row_physical_tolerance"]),
                })
        else:
            pol_phys += sf(r["h15_physical"]); pol_dec += sf(r["h15_decision_sum_s"]); pol_sol += sf(r["h15_solver_sum_s"])
            conf["FN" if pos else "TN"] += 1
        details.append({
            "unique_row_id": row_key(r), "bank_id": r.get("bank_id"), "base_state_id": r.get("base_state_id"),
            "source_key": r.get("source_key"), "row_origin": r.get("row_origin"), "selected_h": h,
            "label_positive": pos, "catastrophic": cat, "phys_delta": sf(r["phys_delta_h10_minus_h15"]),
            "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"]), "group": r.get("group"), "window": r.get("window"),
        })
    phys_delta = pol_phys - fixed_phys
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    return {
        "rows": len(rows),
        "chosen_counts": dict(counts),
        "confusion": dict(conf),
        "policy_physical_sum": pol_phys,
        "fixed_H15_physical_sum": fixed_phys,
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "physical_gate": phys_delta <= tol_sum,
        "policy_decision_sum_s": pol_dec,
        "fixed_H15_decision_sum_s": fixed_dec,
        "decision_relative_saving_vs_fixed_H15": dec_save,
        "policy_solver_sum_s": pol_sol,
        "fixed_H15_solver_sum_s": fixed_sol,
        "solver_relative_saving_vs_fixed_H15": sol_save,
        "catastrophic_false_positive_rows": bad,
        "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad),
        "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad),
        "details": details,
    }


def eval_split(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], feature_by_key: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    choices, scores, feature_count = predict_split(train_rows, eval_rows, feature_by_key, cfg, include_scores=include_scores)
    ev = eval_choices(eval_rows, choices)
    ev["feature_count"] = feature_count
    if include_scores:
        ev["prediction_scores"] = scores
    return ev


def aggregate_from_details(rows: Sequence[Mapping[str, Any]], holdouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    choices: Dict[str, int] = {}
    for ev in holdouts.values():
        for d in ev.get("details") or []:
            choices[str(d["unique_row_id"])] = int(d["selected_h"])
    return eval_choices(rows, choices)


def eval_lobo_by_bank(rows: Sequence[Mapping[str, Any]], banks: Sequence[str], feature_by_key: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], blocked_banks: Sequence[str] = (), include_scores: bool = False) -> Dict[str, Any]:
    blocked = set(str(b) for b in blocked_banks)
    eval_banks = [b for b in banks if b not in blocked]
    holdouts: Dict[str, Any] = {}
    for hb in eval_banks:
        train = [r for r in rows if str(r["bank_id"]) not in blocked and str(r["bank_id"]) != hb]
        evrows = [r for r in rows if str(r["bank_id"]) == hb]
        holdouts[hb] = eval_split(train, evrows, feature_by_key, cfg, include_scores=include_scores)
    eval_rows = [r for r in rows if str(r["bank_id"]) in eval_banks]
    return {"holdouts": holdouts, "aggregate": aggregate_from_details(eval_rows, holdouts)}


def summary_from_lobo(lobo: Mapping[str, Any]) -> Dict[str, Any]:
    agg = lobo["aggregate"]
    holds = lobo.get("holdouts") or {}
    saves = [sf(e.get("decision_relative_saving_vs_fixed_H15")) for e in holds.values()]
    return {
        "aggregate_bad": len(agg.get("catastrophic_false_positive_rows") or []),
        "aggregate_physical_gate": bool(agg.get("physical_gate")),
        "aggregate_save": sf(agg.get("decision_relative_saving_vs_fixed_H15")),
        "aggregate_solver_save": sf(agg.get("solver_relative_saving_vs_fixed_H15")),
        "aggregate_h10": int((agg.get("chosen_counts") or {}).get("10", 0)),
        "aggregate_pass5": bool(agg.get("pass_5pct_no_cat_fp")),
        "aggregate_pass10": bool(agg.get("pass_10pct_no_cat_fp")),
        "min_holdout_save": min(saves) if saves else 0.0,
        "avg_holdout_save": mean(saves),
        "holdout_bad_total": sum(len(e.get("catastrophic_false_positive_rows") or []) for e in holds.values()),
        "holdout_phys_fail_count": sum(0 if e.get("physical_gate") else 1 for e in holds.values()),
    }


def rank_summary(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(s["aggregate_bad"]),
        0 if bool(s["aggregate_physical_gate"]) else 1,
        not bool(s["aggregate_pass10"]),
        not bool(s["aggregate_pass5"]),
        -sf(s["aggregate_save"]),
        -int(s["aggregate_h10"]),
        int(s["holdout_bad_total"]),
        int(s["holdout_phys_fail_count"]),
    )


def compact_eval(ev: Mapping[str, Any]) -> Dict[str, Any]:
    keep = dict(ev)
    if len(keep.get("details") or []) > 25:
        keep["details"] = list(keep["details"][:25])
        keep["details_truncated"] = True
    if "prediction_scores" in keep and len(keep["prediction_scores"]) > 20:
        keys = sorted(keep["prediction_scores"])[:20]
        keep["prediction_scores"] = {k: keep["prediction_scores"][k] for k in keys}
        keep["prediction_scores_truncated"] = True
    return keep


def load_augmented_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, str]]:
    old_rows0, hashes = v13.load_rows()
    old_features0, hist_diag, hist_hashes = v13.enrich_history(old_rows0)
    hashes.update(hist_hashes)
    rows: List[Dict[str, Any]] = []
    features: Dict[str, Dict[str, float]] = {}
    for r0 in old_rows0:
        r = dict(r0)
        uid = v13.row_key(r)
        r["unique_row_id"] = uid
        r["source_key"] = uid
        r["row_origin"] = "opened_bank_base_v12_aligned"
        r["boundary_role"] = None
        rows.append(r)
        features[uid] = dict(old_features0[uid])
    for p in [V15_RAW, V15_DONE, V15_PROTOCOL, V15_SUMMARY, SOURCE, Path(v13.__file__)]:
        if p.exists():
            hashes[rel(p)] = sha256(p)
    done = read_json(V15_DONE)
    raw = read_json(V15_RAW)
    protocol = read_json(V15_PROTOCOL)
    assert_dev_only(done, "v15 completed")
    assert_dev_only(raw, "v15 raw")
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("v15 boundary acquisition did not complete")
    h = done.get("headline") or {}
    if h.get("label_data_sufficient_for_refit") is not True or h.get("all_h15_safe") is not True:
        raise ContractError("v15 boundary labels are not sufficient/safe according to completed.json")
    pair_by_idx: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for pr in ((raw.get("analysis") or {}).get("pair_rows") or []):
        pair_by_idx[si(pr.get("candidate_index"), -1)].append(pr)
    candidates = list(raw.get("candidates_preoutcome") or [])
    if len(candidates) != 24 or sum(len(v) for v in pair_by_idx.values()) != 48:
        raise ContractError("unexpected v15 candidate/pair counts")
    trace_hash_count = 0
    trace_loaded = 0
    for c in candidates:
        idx = si(c.get("candidate_index"), -1)
        rr = pair_by_idx.get(idx) or []
        if len(rr) != 2:
            raise ContractError(f"candidate {idx} does not have two pair repeats")
        h15_safe = all(p.get("h15_safe") is True for p in rr)
        h10_safe = all(p.get("h10_safe") is True for p in rr)
        any_cat = any(p.get("h10_catastrophic_vs_h15") is True for p in rr)
        mean_h10_phys = mean([sf(p.get("h10_physical")) for p in rr])
        mean_h15_phys = mean([sf(p.get("h15_physical")) for p in rr])
        mean_h10_dec = mean([sf(p.get("h10_decision_sum_s")) for p in rr])
        mean_h15_dec = mean([sf(p.get("h15_decision_sum_s")) for p in rr])
        mean_h10_sol = mean([sf(p.get("h10_solver_sum_s")) for p in rr])
        mean_h15_sol = mean([sf(p.get("h15_solver_sum_s")) for p in rr])
        phys_delta = mean_h10_phys - mean_h15_phys
        decision_gain = mean_h15_dec - mean_h10_dec
        solver_gain = mean_h15_sol - mean_h10_sol
        tol = max(MIN_ROW_TOL, 0.05 * abs(mean_h15_phys))
        beneficial = bool(h15_safe and h10_safe and (not any_cat) and phys_delta <= tol and decision_gain > 0.0)
        catastrophic = bool(any_cat)
        prev = c.get("branch_previous_state") if isinstance(c.get("branch_previous_state"), Mapping) else {}
        uid = "v15_boundary/" + str(c.get("candidate_id"))
        row = {
            "unique_row_id": uid,
            "bank_id": str(c.get("bank_id")),
            "base_state_id": str(c.get("candidate_id")),
            "original_base_state_id": str(c.get("base_state_id")),
            "source_key": str(c.get("source_key")),
            "row_origin": "v15_boundary_candidate_mean2",
            "boundary_role": str(c.get("role")),
            "offset_from_center": si(c.get("offset_from_center"), 0),
            "group": "v15_" + str(c.get("role")),
            "window": str(c.get("source_window")),
            "branch_state_slot": parse_slot(str(c.get("base_state_id"))),
            "branch_step": si(c.get("candidate_branch_step"), -1),
            "obs14": obs14_from_raw(c.get("initial_observation_from_h15_trace")),
            "prev_x": sf(prev.get("x")),
            "prev_y": sf(prev.get("y")),
            "prev_theta": sf(prev.get("theta")),
            "stage_a_trace_risk_score": 0.0,
            "h10_physical": mean_h10_phys,
            "h15_physical": mean_h15_phys,
            "h10_decision_sum_s": mean_h10_dec,
            "h15_decision_sum_s": mean_h15_dec,
            "h10_solver_sum_s": mean_h10_sol,
            "h15_solver_sum_s": mean_h15_sol,
            "h10_safe_all": h10_safe,
            "h15_safe_all": h15_safe,
            "h10_success_all": all(p.get("h10_success") is True for p in rr),
            "h15_success_all": all(p.get("h15_success") is True for p in rr),
            "row_physical_tolerance": tol,
            "phys_delta_h10_minus_h15": phys_delta,
            "decision_gain_h10_vs_h15_s": decision_gain,
            "solver_gain_h10_vs_h15_s": solver_gain,
            "h10_beneficial_vs_h15": beneficial,
            "h10_catastrophic_vs_h15": catastrophic,
            "v15_pair_repeats": len(rr),
            "v15_catastrophic_repeats": sum(1 for p in rr if p.get("h10_catastrophic_vs_h15") is True),
            "v15_beneficial_repeats": sum(1 for p in rr if p.get("h10_beneficial_vs_h15") is True),
        }
        fd = base_feature_dict(row)
        trace_rel = str(c.get("h15_trace_episode_path") or "")
        trace_path = ROOT / trace_rel / "trace.json"
        if add_history_features(fd, trace_path, si(c.get("candidate_branch_step"), -1)):
            trace_loaded += 1
            if trace_hash_count < 24 and trace_path.exists():
                hashes[rel(trace_path)] = sha256(trace_path)
                trace_hash_count += 1
        rows.append(row)
        features[uid] = fd
    diag = {
        "old_row_count": len(old_rows0),
        "v15_boundary_row_count": len(candidates),
        "augmented_row_count": len(rows),
        "old_history_diag": hist_diag,
        "v15_trace_history_loaded_rows": trace_loaded,
        "v15_trace_risk_unavailable_rows": len(candidates),
        "banks": sorted(set(str(r["bank_id"]) for r in rows)),
        "source_key_count": len(set(str(r.get("source_key")) for r in rows)),
        "label_counts": {
            "positive": sum(1 for r in rows if bool(r["h10_beneficial_vs_h15"])),
            "catastrophic": sum(1 for r in rows if bool(r["h10_catastrophic_vs_h15"])),
            "v15_positive": sum(1 for r in rows if r.get("row_origin") == "v15_boundary_candidate_mean2" and bool(r["h10_beneficial_vs_h15"])),
            "v15_catastrophic": sum(1 for r in rows if r.get("row_origin") == "v15_boundary_candidate_mean2" and bool(r["h10_catastrophic_vs_h15"])),
        },
    }
    return rows, features, diag, hashes


def freeze_protocol(created: dt.datetime, rows_diag: Mapping[str, Any], cfgs: Sequence[Mapping[str, Any]], backup_commit: str, hashes: Mapping[str, str]) -> None:
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_boundary_augmented_risk_value_refit_no_sim_no_validation_no_test",
        "hypothesis": "The v15 boundary labels add enough local catastrophic/safe-beneficial supervision to let a conservative deployable H10/H15 risk-value selector satisfy zero-catastrophic-H10 and >=5% measured decision saving under strict held-out opened-bank evaluation. If only local/in-sample repair succeeds, boundary mining did not generalize and the next intervention should be richer learned risk/terminal-value estimation or observability diagnostics.",
        "inputs": {
            "old_opened_rows_source": "v13/v12 loader over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11",
            "v15_boundary_raw": rel(V15_RAW),
            "v15_boundary_completed": rel(V15_DONE),
            "v15_boundary_protocol": rel(V15_PROTOCOL),
        },
        "rows_diag": rows_diag,
        "families": FAMILIES,
        "config_count": len(cfgs),
        "strict_evaluation": "nested opened-bank holdout over augmented rows; for each outer bank, all rows from that bank, including v15 boundary rows from that bank/source, are excluded from training and used only for outer evaluation",
        "decision_gate_before_unused_source_confirmation": {"zero_catastrophic_h10_false_positives": True, "physical_gate": True, "minimum_measured_decision_saving_vs_fixed_H15": 0.05},
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 50000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": dict(hashes),
    }
    write_json(PROTOCOL, proto)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows, feature_by_key, rows_diag, hashes = load_augmented_rows()
    cfgs = cfg_grid()
    freeze_protocol(created, rows_diag, cfgs, args.backup_verified_commit, hashes)
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    selector_evals = 0

    # Development global LOBO over augmented rows.  This is model-selection evidence,
    # not the strict decision result.
    global_items: List[Dict[str, Any]] = []
    for cfg in cfgs:
        lobo = eval_lobo_by_bank(rows, banks, feature_by_key, cfg, include_scores=False)
        selector_evals += len(banks)
        global_items.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summary_from_lobo(lobo), "lobo": lobo})
    global_ranked = sorted(global_items, key=lambda x: rank_summary(x["summary"]))
    top_global = []
    for x in global_ranked[:20]:
        keep = {k: v for k, v in x.items() if k != "lobo"}
        keep["aggregate"] = x["lobo"]["aggregate"]
        top_global.append(keep)
    best_global = global_ranked[0]

    # Strict nested opened-bank selection over augmented rows.
    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    for outer_bank in banks:
        inner_items: List[Dict[str, Any]] = []
        for cfg in cfgs:
            lobo = eval_lobo_by_bank(rows, banks, feature_by_key, cfg, blocked_banks=[outer_bank], include_scores=False)
            selector_evals += len([b for b in banks if b != outer_bank])
            inner_items.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summary_from_lobo(lobo)})
        selected = sorted(inner_items, key=lambda x: rank_summary(x["summary"]))[0]
        train_outer = [r for r in rows if str(r["bank_id"]) != outer_bank]
        eval_outer = [r for r in rows if str(r["bank_id"]) == outer_bank]
        outer_eval = eval_split(train_outer, eval_outer, feature_by_key, selected["config"], include_scores=True)
        selector_evals += 1
        for d in outer_eval.get("details") or []:
            nested_choices[str(d["unique_row_id"])] = int(d["selected_h"])
        nested_outer[outer_bank] = {
            "selected_config_id": selected["config_id"],
            "selected_config": selected["config"],
            "inner_summary": selected["summary"],
            "outer_eval": compact_eval(outer_eval),
        }
    nested_aggregate = eval_choices(rows, nested_choices)

    # Local repair diagnostic only: train the best global config on all augmented
    # rows and re-evaluate the same rows.  Self-neighbour leakage is intentional
    # and disclosed; it tests representational separability, not generalization.
    in_choices, in_scores, in_feature_count = predict_split(rows, rows, feature_by_key, best_global["config"], include_scores=True)
    selector_evals += 1
    in_sample = eval_choices(rows, in_choices)
    in_sample["feature_count"] = in_feature_count
    in_sample["prediction_scores_sample"] = {k: in_scores[k] for k in sorted(in_scores)[:20]}
    in_sample["self_neighbor_leakage"] = True

    h = {
        "rows": len(rows),
        "old_rows": rows_diag["old_row_count"],
        "v15_boundary_rows": rows_diag["v15_boundary_row_count"],
        "banks": banks,
        "source_key_count": rows_diag["source_key_count"],
        "positive_rows": rows_diag["label_counts"]["positive"],
        "catastrophic_rows": rows_diag["label_counts"]["catastrophic"],
        "v15_positive_rows": rows_diag["label_counts"]["v15_positive"],
        "v15_catastrophic_rows": rows_diag["label_counts"]["v15_catastrophic"],
        "config_count": len(cfgs),
        "selector_refit_evaluations": selector_evals,
        "global_pass5_count": sum(1 for x in global_items if x["summary"]["aggregate_pass5"]),
        "global_pass10_count": sum(1 for x in global_items if x["summary"]["aggregate_pass10"]),
        "best_global_config": best_global["config_id"],
        "best_global_bad": best_global["summary"]["aggregate_bad"],
        "best_global_save": best_global["summary"]["aggregate_save"],
        "best_global_solver_save": best_global["summary"]["aggregate_solver_save"],
        "best_global_h10": best_global["summary"]["aggregate_h10"],
        "nested_bad": len(nested_aggregate.get("catastrophic_false_positive_rows") or []),
        "nested_physical_gate": bool(nested_aggregate.get("physical_gate")),
        "nested_save": sf(nested_aggregate.get("decision_relative_saving_vs_fixed_H15")),
        "nested_solver_save": sf(nested_aggregate.get("solver_relative_saving_vs_fixed_H15")),
        "nested_h10": int((nested_aggregate.get("chosen_counts") or {}).get("10", 0)),
        "nested_pass5": bool(nested_aggregate.get("pass_5pct_no_cat_fp")),
        "nested_pass10": bool(nested_aggregate.get("pass_10pct_no_cat_fp")),
        "in_sample_bad": len(in_sample.get("catastrophic_false_positive_rows") or []),
        "in_sample_save": sf(in_sample.get("decision_relative_saving_vs_fixed_H15")),
        "in_sample_h10": int((in_sample.get("chosen_counts") or {}).get("10", 0)),
    }
    if h["nested_pass5"] and h["nested_bad"] == 0:
        decision = "v16 strict boundary-augmented refit meets the opened-development zero-catastrophe >=5% decision-saving gate; next freeze an unused-source development confirmation with actual selector overhead before any validation64 rollout."
    elif h["global_pass5_count"] > 0 and not h["nested_pass5"]:
        decision = "v16 can be made to pass a development global LOBO ranking but fails strict nested selection; treat as model-selection overfit and do not confirm yet. Inspect selected folds, then prefer richer risk/terminal-value learning or a better predeclared model-selection rule."
    elif h["in_sample_bad"] == 0 and h["in_sample_save"] >= 0.05:
        decision = "v16 boundary labels repair local/in-sample separability but not held-out generalization; this supports boundary-risk information being relevant but insufficient for deployable generalization. Pivot to richer learned risk/terminal-value representation rather than another static selector sweep."
    else:
        decision = "v16 boundary-augmented static/refit selector still fails to provide a safe useful held-out tradeoff; stop static/refit sweeps and prioritize explicit learned risk/terminal-value training or observability/scenario diagnostics."

    raw = {
        "created_utc": now().isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(),
        "classification": "development_IMPROVED_boundary_augmented_risk_value_refit_no_sim_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "rows_diag": rows_diag,
        "headline": h,
        "decision": decision,
        "top_global_lobo": top_global,
        "nested_outer": nested_outer,
        "nested_aggregate": nested_aggregate,
        "in_sample_repair_diagnostic": in_sample,
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 50000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations": selector_evals, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "platform": platform.platform()},
        "input_hashes": hashes,
        "interpretation_limits": ["opened development only", "v15 rows are deliberately mined boundary labels", "strict nested opened-bank result is decision-relevant", "global/in-sample repair diagnostics are not confirmation", "no selector runtime overhead measured here"],
    }
    return raw


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v16 boundary-augmented refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED metadata/refit diagnostic; no MPC simulation, no validation64, no sealed test, no gradient/RL training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` = old opened rows `{h['old_rows']}` + v15 boundary rows `{h['v15_boundary_rows']}` across banks `{h['banks']}` and source keys `{h['source_key_count']}`.",
        f"- Labels: positives `{h['positive_rows']}` (v15 `{h['v15_positive_rows']}`); catastrophic H10 `{h['catastrophic_rows']}` (v15 `{h['v15_catastrophic_rows']}`).",
        f"- Configs `{h['config_count']}`; selector-refit evaluations `{h['selector_refit_evaluations']}`.",
        f"- Global LOBO pass5 `{h['global_pass5_count']}`, pass10 `{h['global_pass10_count']}`; best global `{h['best_global_config']}` save `{pct(h['best_global_save'])}`, solver save `{pct(h['best_global_solver_save'])}`, bad `{h['best_global_bad']}`, H10 `{h['best_global_h10']}`.",
        f"- Strict nested opened-bank aggregate: save `{pct(h['nested_save'])}`, solver save `{pct(h['nested_solver_save'])}`, bad `{h['nested_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 `{h['nested_h10']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.",
        f"- Local in-sample repair diagnostic: save `{pct(h['in_sample_save'])}`, bad `{h['in_sample_bad']}`, H10 `{h['in_sample_h10']}`; self-neighbour leakage intentionally makes this non-generalization evidence.",
        f"- Decision: {raw['decision']}",
        "",
        "## Strict nested outer-bank results",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer save | outer solver save |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, obj in raw["nested_outer"].items():
        ev = obj["outer_eval"]
        inn = obj["inner_summary"]
        lines.append(f"| `{b}` | `{obj['selected_config_id']}` | `{inn['aggregate_pass5']}` | {inn['aggregate_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Top global LOBO configs", "", "| rank | config | pass10 | pass5 | bad | physical gate | save | solver save | H10 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_lobo"][:15], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['aggregate_pass10']}` | `{s['aggregate_pass5']}` | {s['aggregate_bad']} | `{s['aggregate_physical_gate']}` | {pct(s['aggregate_save'])} | {pct(s['aggregate_solver_save'])} | {s['aggregate_h10']} |")
    fps = raw.get("nested_aggregate", {}).get("catastrophic_false_positive_rows") or []
    lines += ["", "## Strict nested catastrophic H10 false positives", ""]
    if fps:
        lines.append("| bank | row | source | origin | role | off | phys delta | tol | decision gain s |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for r in fps[:40]:
            lines.append(f"| `{r.get('bank_id')}` | `{r.get('base_state_id')}` | `{r.get('source_key')}` | `{r.get('row_origin')}` | `{r.get('boundary_role')}` | {si(r.get('offset_from_center'), 0)} | {sf(r.get('phys_delta')):.6g} | {sf(r.get('tol')):.6g} | {sf(r.get('decision_gain_s')):.6g} |")
    else:
        lines.append("No strict nested catastrophic false positives.")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is opened-development refit evidence only. The strict nested bank holdout prevents training on the same bank's v15 boundary rows, but the benchmark is still development-informed and intentionally enriched around known failures. A pass would justify only an unused-source development confirmation with actual selector overhead. A failure means the v15 boundary labels are informative but not sufficient for deployable generalization under this static/refit class.",
        "",
        f"Raw: `{rel(OUT / 'raw.json')}`; completed: `{rel(OUT / 'completed.json')}`; protocol: `{rel(PROTOCOL)}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v16 boundary-augmented refit

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED boundary-augmented refit over opened rows plus v15 mined boundary labels; no MPC simulation, no validation64/sealed-test access, no gradient training. rows={h['rows']}; configs={h['config_count']}; selector_refit_evaluations={h['selector_refit_evaluations']}; strict_nested_save={h['nested_save']:.6f}; strict_nested_bad={h['nested_bad']}; strict_nested_pass5={h['nested_pass5']}; global_pass5={h['global_pass5_count']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_boundary_augmented_refit,opened_rows_plus_v15_boundary_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-boundary-augmented-refit", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_boundary_augmented_refit:
        raise ContractError("requires --run and explicit development boundary-augmented refit acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise ContractError("partial v16 output exists; inspect before rerun: " + rel(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    completed_utc = now()
    write_json(BACKUP_REQUEST, {
        "requested_utc": completed_utc.isoformat(),
        "reason": "backup v16 boundary-augmented refit source/protocol/raw/docs before any unused-source confirmation or training pivot",
        "backup_required_before_more_science": True,
        "development_mpc_simulation_episodes": 0,
        "development_control_steps": 0,
        "selector_refit_evaluations": raw["budget_actual"]["selector_refit_evaluations"],
        "training_episodes": 0,
        "gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    h = raw["headline"]
    STATE.write_text(f"""# Continue state after v16 boundary-augmented refit

UTC: {completed_utc.isoformat()}
Elapsed since first supervisor event: {(completed_utc - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Completed `{NAME}` metadata/refit diagnostic. No MPC simulation, no validation64, no sealed test, no gradient training.

Budget actual: {raw['budget_actual']}
Headline: {h}
Decision: {raw['decision']}
Summary: `{rel(OUT / 'summary.md')}`
Raw: `{rel(OUT / 'raw.json')}`
Completed: `{rel(OUT / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action: verify external backup for v15+v16 artifacts. If strict nested_pass5 is true with zero bad, freeze unused-source confirmation with actual selector overhead. Otherwise do not rerun static/refit sweeps unchanged; pivot to richer learned risk/terminal-value representation or observability/scenario diagnostics while preserving this negative evidence.
""", encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, V15_RAW, V15_DONE, V15_PROTOCOL]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": completed_utc.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (completed_utc - FIRST_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "budget_actual": raw["budget_actual"],
        "headline": h,
        "decision": raw["decision"],
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQUEST),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": h, "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "failed_utc": now().isoformat(),
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 50000, "training_episodes": 0, "gradient_steps": 0},
            "next_recovery_hint": "Preserve partial outputs; if source repair is needed write a new versioned v16b script and do not open validation64/test.",
        })
        raise
