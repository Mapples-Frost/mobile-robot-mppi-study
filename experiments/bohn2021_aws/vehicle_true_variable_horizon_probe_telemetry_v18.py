#!/usr/bin/env python3
"""v18 H10 probe-telemetry risk/value diagnostic.

Development-only IMPROVED offline diagnostic after v17.

Hypothesis: current static/history state features cannot safely decide when to use
true H10, but telemetry available from a candidate H10 solve (objective, solver
iterations, residuals, first action/value traces) may expose terminal/risk-value
information that separates safe-beneficial from catastrophic H10.  The diagnostic
uses existing v15 paired true-H branch rollouts only; it performs no MPC
simulation, no validation64 access, no sealed-test access, and no gradient/RL
training.

Two overhead semantics are evaluated:
  * H10-probe families: H10 telemetry is obtained by solving H10 once.  If H10 is
    accepted, that solve is the first control decision; if H10 is rejected, the
    H10 probe is extra overhead before running H15.
  * dual-probe family: both H10 and H15 first solves are required before the
    decision, so the unused first-solve overhead is charged to the selected arm.

The decision-relevant result is strict nested leave-bank and leave-source model
selection.  Global/non-nested tables are diagnostics only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_probe_telemetry_v18"
STAMP = "20260930T0005Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260930T0005_after_probe_telemetry_v18.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_PROBE_TELEMETRY_V18_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-probe-telemetry-v18-{STAMP}"
V15_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/raw.json"
V15_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/completed.json"
V17_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_bank_consensus_v17_20260929T2345Z/completed.json"
V17_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_bank_consensus_v17_20260929T2345Z/summary.md"

class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=true in {label}")


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def mean(xs: Sequence[float]) -> float:
    return math.fsum(float(x) for x in xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(math.fsum((float(x) - m) ** 2 for x in xs) / (len(xs) - 1))


def qtile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def row_key(r: Mapping[str, Any]) -> str:
    return str(r["candidate_id"])


def first_control(trace: Sequence[Any]) -> Mapping[str, Any]:
    for tr in trace:
        if isinstance(tr, Mapping) and si(tr.get("step"), -999) == 0:
            return tr
    for tr in trace:
        if isinstance(tr, Mapping) and si(tr.get("step"), -999) >= 0:
            return tr
    return {}


def attempt(tr: Mapping[str, Any]) -> Mapping[str, Any]:
    rec = tr.get("recovery") if isinstance(tr.get("recovery"), Mapping) else {}
    at = rec.get("attempts") if isinstance(rec.get("attempts"), list) else []
    for a in at:
        if isinstance(a, Mapping) and a.get("accepted") is True:
            return a
    return at[0] if at and isinstance(at[0], Mapping) else {}


def vec_values(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    return obj if isinstance(obj, Mapping) else {}


def obs14_from(tr: Mapping[str, Any]) -> List[float]:
    raw = tr.get("observation") or []
    vals = [sf(x) for x in raw[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def summarize_trace(ep_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    d = ROOT / ep_path
    summary_p = d / "summary.json"
    trace_p = d / "trace.json"
    solver_p = d / "solver_calls.jsonl"
    hashes: Dict[str, str] = {}
    if not summary_p.exists() or not trace_p.exists():
        raise ContractError("missing episode trace: " + ep_path)
    summary = read_json(summary_p)
    trace = read_json(trace_p)
    if not isinstance(trace, list):
        raise ContractError("trace not list: " + ep_path)
    hashes[rel(summary_p)] = sha256(summary_p)
    hashes[rel(trace_p)] = sha256(trace_p)
    if solver_p.exists():
        hashes[rel(solver_p)] = sha256(solver_p)
    tr0 = first_control(trace)
    a0 = attempt(tr0)
    obs = obs14_from(tr0)
    pre = vec_values(a0.get("warm_start_pre_opt_x"))
    post = vec_values(a0.get("warm_start_post_opt_x"))
    inp = tr0.get("input") if isinstance(tr0.get("input"), Mapping) else {}
    us = inp.get("u_s") if isinstance(inp.get("u_s"), list) else []
    uo = inp.get("u_omega") if isinstance(inp.get("u_omega"), list) else []
    objectives = []
    iters = []
    residuals = []
    states_x = []
    states_y = []
    states_th = []
    for tr in trace:
        if not isinstance(tr, Mapping) or si(tr.get("step"), -999) < 0:
            continue
        aa = attempt(tr)
        objectives.append(sf(aa.get("objective_opt_f_num")))
        iters.append(sf(aa.get("iterations")))
        residuals.append(max(sf(aa.get("constraint_residual")), sf(aa.get("bound_residual"))))
        st = tr.get("state") if isinstance(tr.get("state"), Mapping) else {}
        states_x.append(sf(st.get("x"))); states_y.append(sf(st.get("y"))); states_th.append(sf(st.get("theta")))
    feats = {
        "steps": sf(summary.get("steps")),
        "physical": sf(summary.get("physical_constraint_cost")),
        "decision_sum_s": sf((summary.get("decision_timing_s") or {}).get("sum")),
        "solver_sum_s": sf((summary.get("solver_attempt_timing_s") or {}).get("sum")),
        "first_decision_s": sf((tr0.get("timing") or {}).get("decision_s")),
        "first_controller_s": sf((tr0.get("timing") or {}).get("controller_s")),
        "first_solver_s": sf(a0.get("solver_s")),
        "first_objective": sf(a0.get("objective_opt_f_num")),
        "first_iterations": sf(a0.get("iterations")),
        "first_constraint_residual_log10": math.log10(max(1e-12, sf(a0.get("constraint_residual"), 0.0))),
        "first_bound_residual_log10": math.log10(max(1e-12, sf(a0.get("bound_residual"), 0.0))),
        "pre_mean": sf(pre.get("mean")),
        "pre_max_abs": sf(pre.get("max_abs")),
        "post_mean": sf(post.get("mean")),
        "post_max_abs": sf(post.get("max_abs")),
        "post_minus_pre_mean": sf(post.get("mean")) - sf(pre.get("mean")),
        "post_minus_pre_max_abs": sf(post.get("max_abs")) - sf(pre.get("max_abs")),
        "first_u_s": sf(us[0] if us else 0.0),
        "first_u_omega": sf(uo[0] if uo else 0.0),
        "first_performance": sf(tr0.get("performance")),
        "first_compute_reward": sf(tr0.get("compute")),
        "first_constraint_reward": sf(tr0.get("constraint")),
        "obj_mean": mean(objectives),
        "obj_min": min(objectives) if objectives else 0.0,
        "obj_max": max(objectives) if objectives else 0.0,
        "obj_std": stdev(objectives),
        "obj_first_minus_mean": (objectives[0] - mean(objectives)) if objectives else 0.0,
        "iters_mean": mean(iters),
        "iters_max": max(iters) if iters else 0.0,
        "residual_log10_max": math.log10(max(1e-12, max(residuals) if residuals else 0.0)),
        "state_x_span": (max(states_x) - min(states_x)) if states_x else 0.0,
        "state_y_span": (max(states_y) - min(states_y)) if states_y else 0.0,
        "state_theta_span": (max(states_th) - min(states_th)) if states_th else 0.0,
    }
    for i, x in enumerate(obs):
        feats[f"obs_{i}"] = x
        feats[f"abs_obs_{i}"] = abs(x)
    feats["obs01_norm"] = math.hypot(obs[0], obs[1])
    feats["obs56_norm"] = math.hypot(obs[5], obs[6])
    feats["obs89_norm"] = math.hypot(obs[8], obs[9])
    feats["obs1112_norm"] = math.hypot(obs[11], obs[12])
    return feats, hashes


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, str]]:
    if not V15_RAW.exists() or not V15_DONE.exists():
        raise ContractError("v15 inputs missing")
    raw = read_json(V15_RAW); done = read_json(V15_DONE)
    assert_dev_only(raw, "v15 raw"); assert_dev_only(done, "v15 done")
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("v15 did not complete")
    hashes = {rel(SOURCE): sha256(SOURCE), rel(V15_RAW): sha256(V15_RAW), rel(V15_DONE): sha256(V15_DONE)}
    if V17_DONE.exists():
        v17 = read_json(V17_DONE); assert_dev_only(v17, "v17 done")
        hashes[rel(V17_DONE)] = sha256(V17_DONE)
    if V17_SUMMARY.exists():
        hashes[rel(V17_SUMMARY)] = sha256(V17_SUMMARY)
    pair_by_idx: Dict[int, List[Mapping[str, Any]]] = {}
    for pr in ((raw.get("analysis") or {}).get("pair_rows") or []):
        pair_by_idx.setdefault(si(pr.get("candidate_index"), -1), []).append(pr)
    cand = list((raw.get("analysis") or {}).get("candidate_rows") or [])
    if not cand:
        cand = list(raw.get("candidates_preoutcome") or [])
    rows: List[Dict[str, Any]] = []
    fdicts: Dict[str, Dict[str, float]] = {}
    trace_hashes_added = 0
    for c in cand:
        idx = si(c.get("candidate_index"), -1)
        prs = pair_by_idx.get(idx, [])
        if len(prs) != 2:
            continue
        # One candidate-level row; means over deterministic paired repeats.
        h10_feats_list = []; h15_feats_list = []
        for p in prs:
            f10, h10h = summarize_trace(str(p.get("h10_path")))
            f15, h15h = summarize_trace(str(p.get("h15_path")))
            h10_feats_list.append(f10); h15_feats_list.append(f15)
            if trace_hashes_added < 20:
                for k, v in list(h10h.items())[:3] + list(h15h.items())[:3]:
                    hashes[k] = v
                    trace_hashes_added += 1
        h10m = {k: mean([sf(x.get(k)) for x in h10_feats_list]) for k in h10_feats_list[0]}
        h15m = {k: mean([sf(x.get(k)) for x in h15_feats_list]) for k in h15_feats_list[0]}
        mean_h10_phys = mean([sf(p.get("h10_physical")) for p in prs])
        mean_h15_phys = mean([sf(p.get("h15_physical")) for p in prs])
        mean_h10_dec = mean([sf(p.get("h10_decision_sum_s")) for p in prs])
        mean_h15_dec = mean([sf(p.get("h15_decision_sum_s")) for p in prs])
        mean_h10_sol = mean([sf(p.get("h10_solver_sum_s")) for p in prs])
        mean_h15_sol = mean([sf(p.get("h15_solver_sum_s")) for p in prs])
        phys_delta = mean_h10_phys - mean_h15_phys
        gain = mean_h15_dec - mean_h10_dec
        tol = max(2.0, 0.05 * abs(mean_h15_phys))
        catastrophic = any(p.get("h10_catastrophic_vs_h15") is True for p in prs)
        beneficial = bool((not catastrophic) and phys_delta <= tol and gain > 0.0)
        uid = str(c.get("candidate_id") or f"v15c{idx:02d}")
        src = str(c.get("source_key") or prs[0].get("source_key") or uid)
        row = {
            "candidate_id": uid,
            "candidate_index": idx,
            "bank_id": str(c.get("bank_id") or prs[0].get("bank_id")),
            "source_key": src,
            "source_group": str(c.get("source_group") or prs[0].get("source_group") or c.get("role")),
            "role": str(c.get("role") or prs[0].get("role")),
            "offset_from_center": si(c.get("offset_from_center", prs[0].get("offset_from_center")), 0),
            "branch_step": si(c.get("branch_step", c.get("candidate_branch_step", prs[0].get("branch_step"))), -1),
            "h10_physical": mean_h10_phys,
            "h15_physical": mean_h15_phys,
            "h10_decision_sum_s": mean_h10_dec,
            "h15_decision_sum_s": mean_h15_dec,
            "h10_solver_sum_s": mean_h10_sol,
            "h15_solver_sum_s": mean_h15_sol,
            "h10_first_decision_s": mean([sf(x.get("first_decision_s")) for x in h10_feats_list]),
            "h15_first_decision_s": mean([sf(x.get("first_decision_s")) for x in h15_feats_list]),
            "phys_delta_h10_minus_h15": phys_delta,
            "decision_gain_h10_vs_h15_s": gain,
            "solver_gain_h10_vs_h15_s": mean_h15_sol - mean_h10_sol,
            "row_physical_tolerance": tol,
            "h10_catastrophic_vs_h15": catastrophic,
            "h10_beneficial_vs_h15": beneficial,
        }
        feats: Dict[str, float] = {
            "branch_step_150": max(0.0, sf(row["branch_step"])) / 150.0,
            "offset_from_center_10": sf(row["offset_from_center"]) / 10.0,
        }
        for k, v in h10m.items():
            feats["h10_" + k] = sf(v)
        for k, v in h15m.items():
            feats["h15_" + k] = sf(v)
            feats["diff_" + k] = sf(h10m.get(k)) - sf(v)
            feats["ratio_" + k] = sf(h10m.get(k)) / max(1e-9, abs(sf(v)))
        rows.append(row)
        fdicts[uid] = feats
    if len(rows) != 24:
        raise ContractError(f"expected 24 candidate rows, got {len(rows)}")
    diag = {
        "rows": len(rows),
        "banks": sorted({r["bank_id"] for r in rows}),
        "source_keys": len({r["source_key"] for r in rows}),
        "positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]),
        "catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]),
        "fixed_H10_decision_relative_saving_vs_H15": (math.fsum(r["h15_decision_sum_s"] - r["h10_decision_sum_s"] for r in rows) / math.fsum(r["h15_decision_sum_s"] for r in rows)),
        "fixed_H10_physical_delta_vs_H15": math.fsum(r["h10_physical"] - r["h15_physical"] for r in rows),
        "fixed_H10_bad_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]),
    }
    return rows, fdicts, diag, hashes


def family_names(family: str, fdicts: Mapping[str, Mapping[str, float]]) -> List[str]:
    all_names = sorted(set().union(*(fd.keys() for fd in fdicts.values())))
    if family == "h10_probe_objective_only":
        keep = [n for n in all_names if n.startswith("h10_first_") or n in ("h10_pre_mean", "h10_pre_max_abs", "h10_post_mean", "h10_post_max_abs", "h10_post_minus_pre_mean", "h10_post_minus_pre_max_abs", "h10_obj_mean", "h10_obj_std", "h10_obj_min", "h10_obj_max", "h10_iters_mean", "h10_iters_max", "h10_residual_log10_max", "h10_first_u_s", "h10_first_u_omega")]
    elif family == "h10_probe_plus_state":
        keep = [n for n in all_names if n.startswith("h10_") or n in ("branch_step_150", "offset_from_center_10")]
    elif family == "dual_probe_objective":
        keep = [n for n in all_names if n.startswith("h10_first_") or n.startswith("h15_first_") or n.startswith("diff_first_") or n.startswith("ratio_first_") or n.startswith("diff_obj") or n.startswith("ratio_obj") or n in ("branch_step_150",)]
    else:
        raise ContractError("unknown family " + family)
    return sorted(set(keep))


def scaler(train: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], names: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    means = {}; stds = {}
    for nm in names:
        xs = [sf(fdicts[row_key(r)].get(nm)) for r in train]
        m = mean(xs); s = stdev(xs)
        means[nm] = m; stds[nm] = s if s > 1e-12 else 1.0
    return means, stds


def vrow(r: Mapping[str, Any], fdicts: Mapping[str, Mapping[str, float]], names: Sequence[str], means: Mapping[str, float], stds: Mapping[str, float]) -> List[float]:
    fd = fdicts[row_key(r)]
    return [(sf(fd.get(nm)) - sf(means.get(nm))) / max(1e-12, sf(stds.get(nm), 1.0)) for nm in names]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


def predict(train: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    fam = str(cfg["family"]); names = family_names(fam, fdicts)
    means, stds = scaler(train, fdicts, names)
    tvec = {row_key(r): vrow(r, fdicts, names, means, stds) for r in train}
    choices: Dict[str, int] = {}; scores: Dict[str, Any] = {}
    for er in eval_rows:
        ev = vrow(er, fdicts, names, means, stds)
        neigh = sorted([(dist(ev, tvec[row_key(tr)]), tr) for tr in train], key=lambda z: (z[0], row_key(z[1])))
        kk = max(1, min(si(cfg.get("k"), 3), len(neigh)))
        near = neigh[:kk]
        weights = [1.0 / max(0.05, d) for d, _ in near]
        sw = math.fsum(weights) if weights else 1.0
        risk = (math.fsum(w for w, (_, r) in zip(weights, near) if r["h10_catastrophic_vs_h15"]) + 0.5) / (sw + 1.0)
        pgain = math.fsum(w * sf(r["decision_gain_h10_vs_h15_s"]) for w, (_, r) in zip(weights, near)) / sw
        pphys = math.fsum(w * sf(r["phys_delta_h10_minus_h15"]) for w, (_, r) in zip(weights, near)) / sw
        dpos = min([d for d, r in neigh if r["h10_beneficial_vs_h15"]] or [1e9])
        dcat = min([d for d, r in neigh if r["h10_catastrophic_vs_h15"]] or [1e9])
        choose = bool(risk <= sf(cfg.get("risk_max"), 0.25) and pgain >= sf(cfg.get("gain_min"), 0.0) and pphys <= sf(cfg.get("phys_max"), 2.0) and dpos <= sf(cfg.get("support_q"), 1e9))
        choices[row_key(er)] = 10 if choose else 15
        scores[row_key(er)] = {"risk_hat": risk, "pred_gain_s": pgain, "pred_phys_delta": pphys, "dpos": dpos, "dcat": dcat, "selected_h": choices[row_key(er)], "feature_count": len(names)}
    return choices, scores


def support_threshold(train: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], family: str, q: float) -> float:
    names = family_names(family, fdicts)
    means, stds = scaler(train, fdicts, names)
    vecs = {row_key(r): vrow(r, fdicts, names, means, stds) for r in train}
    pos = [r for r in train if r["h10_beneficial_vs_h15"]]
    ds = []
    for i, r in enumerate(pos):
        others = [dist(vecs[row_key(r)], vecs[row_key(qr)]) for j, qr in enumerate(pos) if j != i]
        if others:
            ds.append(min(others))
    return qtile(ds, q) if ds else 1e9


def cfg_grid(rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]]) -> List[Dict[str, Any]]:
    out = []
    for fam in ["h10_probe_objective_only", "h10_probe_plus_state", "dual_probe_objective"]:
        for k in [1, 3, 5]:
            for risk_max in [0.15, 0.25, 0.40]:
                for gain_min in [0.0, 0.2, 0.5]:
                    for phys_max in [2.0, 6.0, 20.0]:
                        for sq in [0.50, 0.75, 1.0]:
                            out.append({"family": fam, "k": k, "risk_max": risk_max, "gain_min": gain_min, "phys_max": phys_max, "support_quantile": sq})
    return out


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v18_%s_k%s_r%s_g%s_p%s_sq%s" % (c["family"], c["k"], c["risk_max"], c["gain_min"], c["phys_max"], c["support_quantile"])


def prepared_cfg(train: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    c = dict(cfg)
    c["support_q"] = support_threshold(train, fdicts, str(cfg["family"]), sf(cfg.get("support_quantile"), 1.0))
    return c


def evaluate(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int], overhead_mode: str) -> Dict[str, Any]:
    fixed_phys = math.fsum(r["h15_physical"] for r in rows)
    fixed_dec = math.fsum(r["h15_decision_sum_s"] for r in rows)
    fixed_sol = math.fsum(r["h15_solver_sum_s"] for r in rows)
    tol_sum = math.fsum(r["row_physical_tolerance"] for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0
    counts = Counter(); conf = Counter(); bad = []; details = []
    extra_probe_dec = 0.0; extra_probe_sol = 0.0
    for r in rows:
        h = int(choices.get(row_key(r), 15)); counts[str(h)] += 1
        pos = bool(r["h10_beneficial_vs_h15"]); cat = bool(r["h10_catastrophic_vs_h15"])
        extra_d = extra_s = 0.0
        if overhead_mode == "h10_probe" and h == 15:
            extra_d += sf(r["h10_first_decision_s"]); extra_s += sf(r["h10_first_decision_s"])
        elif overhead_mode == "dual_probe":
            if h == 10:
                extra_d += sf(r["h15_first_decision_s"]); extra_s += sf(r["h15_first_decision_s"])
            else:
                extra_d += sf(r["h10_first_decision_s"]); extra_s += sf(r["h10_first_decision_s"])
        extra_probe_dec += extra_d; extra_probe_sol += extra_s
        if h == 10:
            pol_phys += r["h10_physical"]; pol_dec += r["h10_decision_sum_s"] + extra_d; pol_sol += r["h10_solver_sum_s"] + extra_s
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({"candidate_id": row_key(r), "bank_id": r["bank_id"], "source_key": r["source_key"], "role": r["role"], "offset": r["offset_from_center"], "phys_delta": r["phys_delta_h10_minus_h15"], "decision_gain_s": r["decision_gain_h10_vs_h15_s"]})
        else:
            pol_phys += r["h15_physical"]; pol_dec += r["h15_decision_sum_s"] + extra_d; pol_sol += r["h15_solver_sum_s"] + extra_s
            conf["FN" if pos else "TN"] += 1
        details.append({"candidate_id": row_key(r), "bank_id": r["bank_id"], "source_key": r["source_key"], "selected_h": h, "label_positive": pos, "catastrophic": cat, "phys_delta": r["phys_delta_h10_minus_h15"], "decision_gain_s": r["decision_gain_h10_vs_h15_s"], "extra_probe_decision_s": extra_d})
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    phys_delta = pol_phys - fixed_phys
    return {"rows": len(rows), "overhead_mode": overhead_mode, "chosen_counts": dict(counts), "confusion": dict(conf), "policy_physical_sum": pol_phys, "fixed_H15_physical_sum": fixed_phys, "physical_delta_vs_fixed_H15": phys_delta, "physical_tolerance_sum": tol_sum, "physical_gate": phys_delta <= tol_sum, "policy_decision_sum_s": pol_dec, "fixed_H15_decision_sum_s": fixed_dec, "extra_probe_decision_sum_s": extra_probe_dec, "decision_relative_saving_vs_fixed_H15": dec_save, "policy_solver_sum_s": pol_sol, "fixed_H15_solver_sum_s": fixed_sol, "extra_probe_solver_sum_s": extra_probe_sol, "solver_relative_saving_vs_fixed_H15": sol_save, "catastrophic_false_positive_rows": bad, "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad), "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad), "details": details}


def overhead_for_family(family: str) -> str:
    return "dual_probe" if family == "dual_probe_objective" else "h10_probe"


def eval_cfg(train: Sequence[Mapping[str, Any]], evrows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    pcfg = prepared_cfg(train, fdicts, cfg)
    choices, scores = predict(train, evrows, fdicts, pcfg)
    ev = evaluate(evrows, choices, overhead_for_family(str(cfg["family"])))
    ev["config_id"] = cfg_id(cfg); ev["config"] = dict(cfg)
    if include_scores:
        ev["prediction_scores"] = scores
    return ev


def aggregate(rows: Sequence[Mapping[str, Any]], hold: Mapping[str, Mapping[str, Any]], overhead_mode: str) -> Dict[str, Any]:
    choices = {}
    for ev in hold.values():
        for d in ev.get("details") or []:
            choices[str(d["candidate_id"])] = int(d["selected_h"])
    return evaluate(rows, choices, overhead_mode)


def summary(ev: Mapping[str, Any]) -> Dict[str, Any]:
    return {"bad": len(ev.get("catastrophic_false_positive_rows") or []), "physical_gate": bool(ev.get("physical_gate")), "save": sf(ev.get("decision_relative_saving_vs_fixed_H15")), "solver_save": sf(ev.get("solver_relative_saving_vs_fixed_H15")), "h10": int((ev.get("chosen_counts") or {}).get("10", 0)), "pass5": bool(ev.get("pass_5pct_no_cat_fp")), "pass10": bool(ev.get("pass_10pct_no_cat_fp"))}


def rank(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (int(s["bad"]), 0 if s["physical_gate"] else 1, not bool(s["pass10"]), not bool(s["pass5"]), -sf(s["save"]), -int(s["h10"]))


def nested(rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], cfgs: Sequence[Mapping[str, Any]], group_field: str) -> Tuple[Dict[str, Any], int]:
    groups = sorted({str(r[group_field]) for r in rows})
    outer = {}; all_choices = {}; evals = 0
    for og in groups:
        train_outer = [r for r in rows if str(r[group_field]) != og]
        ev_outer = [r for r in rows if str(r[group_field]) == og]
        inner_groups = sorted({str(r[group_field]) for r in train_outer})
        best_cfg = None; best_inner = None
        for cfg in cfgs:
            hold = {}
            for ig in inner_groups:
                tr = [r for r in train_outer if str(r[group_field]) != ig]
                evr = [r for r in train_outer if str(r[group_field]) == ig]
                hold[ig] = eval_cfg(tr, evr, fdicts, cfg, include_scores=False)
                evals += 1
            mode = overhead_for_family(str(cfg["family"]))
            inn = aggregate(train_outer, hold, mode)
            if best_inner is None or rank(summary(inn)) < rank(summary(best_inner)):
                best_inner, best_cfg = inn, dict(cfg)
        assert best_cfg is not None and best_inner is not None
        ev = eval_cfg(train_outer, ev_outer, fdicts, best_cfg, include_scores=True)
        evals += 1
        for d in ev.get("details") or []:
            all_choices[str(d["candidate_id"])] = int(d["selected_h"])
        outer[og] = {"selected_config_id": cfg_id(best_cfg), "selected_config": best_cfg, "inner_eval": summary(best_inner), "outer_eval": compact_eval(ev)}
    # Mixed family overhead can vary by outer choice; aggregate by explicit choices
    # under the selected outer family's per-row details rather than one mode.
    fixed_phys = math.fsum(r["h15_physical"] for r in rows); fixed_dec = math.fsum(r["h15_decision_sum_s"] for r in rows); fixed_sol = math.fsum(r["h15_solver_sum_s"] for r in rows); tol_sum = math.fsum(r["row_physical_tolerance"] for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0; bad = []; counts = Counter(); conf = Counter(); detail = []
    details_by_id = {d["candidate_id"]: d for obj in outer.values() for d in (obj["outer_eval"].get("details") or [])}
    for r in rows:
        d = details_by_id.get(row_key(r), {"selected_h": 15, "extra_probe_decision_s": 0.0})
        h = int(d.get("selected_h", 15)); counts[str(h)] += 1
        extra = sf(d.get("extra_probe_decision_s")); pos = bool(r["h10_beneficial_vs_h15"]); cat = bool(r["h10_catastrophic_vs_h15"])
        if h == 10:
            pol_phys += r["h10_physical"]; pol_dec += r["h10_decision_sum_s"] + extra; pol_sol += r["h10_solver_sum_s"] + extra; conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({"candidate_id": row_key(r), "bank_id": r["bank_id"], "source_key": r["source_key"], "phys_delta": r["phys_delta_h10_minus_h15"], "decision_gain_s": r["decision_gain_h10_vs_h15_s"]})
        else:
            pol_phys += r["h15_physical"]; pol_dec += r["h15_decision_sum_s"] + extra; pol_sol += r["h15_solver_sum_s"] + extra; conf["FN" if pos else "TN"] += 1
        detail.append({"candidate_id": row_key(r), "bank_id": r["bank_id"], "source_key": r["source_key"], "selected_h": h, "extra_probe_decision_s": extra, "label_positive": pos, "catastrophic": cat})
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol else 0.0
    phys_delta = pol_phys - fixed_phys
    agg = {"rows": len(rows), "chosen_counts": dict(counts), "confusion": dict(conf), "policy_physical_sum": pol_phys, "fixed_H15_physical_sum": fixed_phys, "physical_delta_vs_fixed_H15": phys_delta, "physical_tolerance_sum": tol_sum, "physical_gate": phys_delta <= tol_sum, "policy_decision_sum_s": pol_dec, "fixed_H15_decision_sum_s": fixed_dec, "decision_relative_saving_vs_fixed_H15": dec_save, "policy_solver_sum_s": pol_sol, "fixed_H15_solver_sum_s": fixed_sol, "solver_relative_saving_vs_fixed_H15": sol_save, "catastrophic_false_positive_rows": bad, "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad), "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad), "details": detail}
    return {"group_field": group_field, "groups": groups, "outer": outer, "aggregate": agg}, evals


def compact_eval(ev: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(ev)
    if len(out.get("details") or []) > 20:
        out["details"] = list(out["details"][:20]); out["details_truncated"] = True
    if len(out.get("prediction_scores") or {}) > 12:
        keys = sorted(out["prediction_scores"])[:12]
        out["prediction_scores"] = {k: out["prediction_scores"][k] for k in keys}; out["prediction_scores_truncated"] = True
    return out


def freeze_protocol(created: dt.datetime, diag: Mapping[str, Any], cfgs: Sequence[Mapping[str, Any]], hashes: Mapping[str, str], backup_commit: str) -> None:
    write_json(PROTOCOL, {"protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}", "created_utc": created.isoformat(), "classification": "development_IMPROVED_probe_telemetry_risk_value_no_sim_no_validation_no_test", "hypothesis": "H10 candidate-solve terminal/objective telemetry may contain bank/source-invariant risk information missing from static/history observations; if strict nested leave-bank and leave-source gates pass after charging probe overhead, a bounded online overhead smoke is justified. If only dual-probe or global diagnostics pass, telemetry is informative but not yet a deployable speed-improving selector. If all nested gates fail, instrument richer terminal residuals or acquire source-independent coverage before another selector sweep.", "inputs": {"v15_boundary_raw": rel(V15_RAW), "v15_boundary_completed": rel(V15_DONE), "v17_completed": rel(V17_DONE)}, "data_diag": dict(diag), "candidate_config_count": len(cfgs), "families": ["h10_probe_objective_only", "h10_probe_plus_state", "dual_probe_objective"], "strict_splits": ["leave_bank", "leave_source_key"], "overhead_semantics": "H10-probe rejected rows add the first H10 decision time to fixed H15; dual-probe rows add the unused first-solve time to the selected arm.", "decision_gate": {"zero_catastrophic_H10_false_positives": True, "physical_gate": True, "minimum_overhead_charged_decision_saving_vs_fixed_H15": 0.05}, "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "candidate_configs": len(cfgs), "offline_model_evaluations_cap": 20000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False}, "latest_verified_backup_before_run_from_supervisor_context": backup_commit, "input_hashes": dict(hashes)})


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = ["# Vehicle true-variable-H v18 H10 probe-telemetry diagnostic", "", f"UTC `{raw['created_utc']}`. Development-only IMPROVED offline diagnostic; no MPC simulation, no validation64, no sealed test, no gradient/RL training.", "", "## Headline", "", f"- Candidate rows `{h['rows']}` from v15 boundary bank; positives `{h['positive_rows']}`, catastrophic `{h['catastrophic_rows']}`, source keys `{h['source_keys']}`.", f"- Fixed H10 versus H15 on these branch continuations: decision saving `{pct(h['fixed_H10_decision_relative_saving_vs_H15'])}`, physical delta `{h['fixed_H10_physical_delta_vs_H15']:.6g}`, bad rows `{h['fixed_H10_bad_rows']}`.", f"- Configs `{h['config_count']}`; offline model evaluations `{h['offline_model_evaluations']}`.", f"- Strict nested leave-bank: save `{pct(h['leave_bank_save'])}`, solver save `{pct(h['leave_bank_solver_save'])}`, bad `{h['leave_bank_bad']}`, H10 `{h['leave_bank_h10']}`, pass5 `{h['leave_bank_pass5']}`.", f"- Strict nested leave-source: save `{pct(h['leave_source_save'])}`, solver save `{pct(h['leave_source_solver_save'])}`, bad `{h['leave_source_bad']}`, H10 `{h['leave_source_h10']}`, pass5 `{h['leave_source_pass5']}`.", f"- Best global non-nested diagnostic: `{h['best_global_config']}` save `{pct(h['best_global_save'])}`, bad `{h['best_global_bad']}`, H10 `{h['best_global_h10']}`.", f"- Decision: {raw['decision']}", "", "## Strict nested split summaries", "", "| split | groups | save | solver save | bad | physical gate | H counts | pass5 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, obj in raw["nested"].items():
        agg = obj["aggregate"]
        lines.append(f"| `{name}` | {len(obj['groups'])} | {pct(agg['decision_relative_saving_vs_fixed_H15'])} | {pct(agg['solver_relative_saving_vs_fixed_H15'])} | {len(agg['catastrophic_false_positive_rows'])} | `{agg['physical_gate']}` | `{agg['chosen_counts']}` | `{agg['pass_5pct_no_cat_fp']}` |")
    lines += ["", "## Outer split selections", "", "| split | held-out group | selected config | inner save | inner bad | outer save | outer bad | outer H counts |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, obj in raw["nested"].items():
        for g, ev in obj["outer"].items():
            o = ev["outer_eval"]
            lines.append(f"| `{name}` | `{g}` | `{ev['selected_config_id']}` | {pct(ev['inner_eval']['save'])} | {ev['inner_eval']['bad']} | {pct(o['decision_relative_saving_vs_fixed_H15'])} | {len(o['catastrophic_false_positive_rows'])} | `{o['chosen_counts']}` |")
    fps = (raw["nested"]["leave_source_key"]["aggregate"].get("catastrophic_false_positive_rows") or []) + (raw["nested"]["leave_bank"]["aggregate"].get("catastrophic_false_positive_rows") or [])
    lines += ["", "## Catastrophic H10 false positives in nested splits", ""]
    if not fps:
        lines.append("No nested catastrophic H10 false positives.")
    else:
        lines.append("| candidate | bank | source | phys delta | decision gain |")
        lines.append("|---|---|---|---:|---:|")
        seen = set()
        for r in fps:
            if r['candidate_id'] in seen: continue
            seen.add(r['candidate_id'])
            lines.append(f"| `{r['candidate_id']}` | `{r.get('bank_id')}` | `{r.get('source_key')}` | {sf(r.get('phys_delta')):.6g} | {sf(r.get('decision_gain_s')):.6g} |")
    lines += ["", "## Interpretation", "", "A pass would justify an online H10-probe overhead smoke on fresh development states. A global-only pass or nested failure means probe telemetry is informative in-sample but not source/bank invariant enough. A zero-H10 nested result means the telemetry veto is safe but not useful, paralleling v17. Results are development/mechanism evidence only.", "", f"Protocol: `{rel(PROTOCOL)}`. Raw: `{rel(OUT/'raw.json')}`. Completed: `{rel(OUT/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H v18 H10 probe-telemetry diagnostic

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline diagnostic over existing v15 true-H branch traces; no MPC simulation, no validation64/sealed-test access, no gradient training. rows={h['rows']}; configs={h['config_count']}; leave_bank_save={h['leave_bank_save']:.6f}; leave_bank_bad={h['leave_bank_bad']}; leave_bank_h10={h['leave_bank_h10']}; leave_source_save={h['leave_source_save']:.6f}; leave_source_bad={h['leave_source_bad']}; leave_source_h10={h['leave_source_h10']}; best_global={h['best_global_config']} save={h['best_global_save']:.6f} bad={h['best_global_bad']} H10={h['best_global_h10']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT/'summary.md')}`, `{rel(OUT/'raw.json')}`, `{rel(OUT/'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    with reg.open("a", encoding="utf-8") as f:
        f.write(f"{STAMP},{NAME},development_probe_telemetry_diagnostic,v15_boundary_traces_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['offline_model_evaluations']},0,0,False,{rel(OUT/'completed.json')},{MARKER}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    args = ap.parse_args(argv)
    if not args.run:
        print("pass --run", file=sys.stderr); return 2
    OUT.mkdir(parents=True, exist_ok=True)
    created = now()
    rows, fdicts, diag, hashes = load_rows()
    cfgs = cfg_grid(rows, fdicts)
    freeze_protocol(created, diag, cfgs, hashes, args.backup_verified_commit)
    # Global non-nested LOBO by source is diagnostic only.
    global_items = []
    groups = sorted({r["source_key"] for r in rows})
    evals = 0
    for cfg in cfgs:
        hold = {}
        for g in groups:
            tr = [r for r in rows if r["source_key"] != g]
            ev = [r for r in rows if r["source_key"] == g]
            hold[g] = eval_cfg(tr, ev, fdicts, cfg)
            evals += 1
        mode = overhead_for_family(str(cfg["family"]))
        agg = aggregate(rows, hold, mode)
        global_items.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summary(agg), "aggregate": agg})
    ranked = sorted(global_items, key=lambda x: rank(x["summary"]))
    nb, eb = nested(rows, fdicts, cfgs, "bank_id"); evals += eb
    ns, es = nested(rows, fdicts, cfgs, "source_key"); evals += es
    hb = summary(nb["aggregate"]); hs = summary(ns["aggregate"]); hg = ranked[0]["summary"]
    headline = dict(diag)
    headline.update({"config_count": len(cfgs), "offline_model_evaluations": evals, "leave_bank_save": hb["save"], "leave_bank_solver_save": hb["solver_save"], "leave_bank_bad": hb["bad"], "leave_bank_h10": hb["h10"], "leave_bank_pass5": hb["pass5"], "leave_source_save": hs["save"], "leave_source_solver_save": hs["solver_save"], "leave_source_bad": hs["bad"], "leave_source_h10": hs["h10"], "leave_source_pass5": hs["pass5"], "best_global_config": ranked[0]["config_id"], "best_global_save": hg["save"], "best_global_bad": hg["bad"], "best_global_h10": hg["h10"]})
    if hb["pass5"] and hs["pass5"] and hb["bad"] == 0 and hs["bad"] == 0:
        decision = "H10 probe telemetry passes strict nested bank and source gates after overhead charging; next freeze a tiny online overhead smoke on fresh development states before any validation/test."
    elif (hb["bad"] == 0 and hb["h10"] == 0) or (hs["bad"] == 0 and hs["h10"] == 0):
        decision = "Probe telemetry can be made safe only by suppressing H10 in at least one strict split; this does not solve the useful speed/safety selector problem. Next instrument richer terminal residuals or train/refit a terminal-risk value model on source-independent coverage."
    elif hg["pass5"] and (not hb["pass5"] or not hs["pass5"]):
        decision = "Probe telemetry contains some in-development signal, but it fails strict bank/source generalization after overhead charging; do not run an online selector yet. Next acquire source-independent states or add explicit terminal residual instrumentation."
    else:
        decision = "Existing H10 probe telemetry is insufficient for a safe useful selector under strict bank/source splits; prioritize richer terminal-value/risk instrumentation or scenario/source-independent data, not another static threshold sweep."
    raw = {"created_utc": now().isoformat(), "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(), "classification": "development_IMPROVED_probe_telemetry_risk_value_no_sim_no_validation_no_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "data_diag": diag, "headline": headline, "decision": decision, "top_global_source_lobo": ranked[:20], "nested": {"leave_bank": nb, "leave_source_key": ns}, "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "offline_model_evaluations_cap": 20000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "offline_model_evaluations": evals, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}, "input_hashes": hashes, "platform": {"python": sys.version, "platform": platform.platform()}, "interpretation_limits": ["v15 mined boundary data only", "offline telemetry from existing rollouts", "selector overhead approximated from first-solve decision time", "no validation64/test access", "no online selector implementation yet"]}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {"created_utc": now().isoformat(), "reason": "backup after v18 H10 probe-telemetry diagnostic", "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT/'raw.json'), rel(OUT/'summary.md'), rel(OUT/'completed.json'), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "required_before_more_unique_science": True})
    raw_hash = sha256(OUT / "raw.json"); summary_hash = sha256(OUT / "summary.md")
    completed = {"created_utc": now().isoformat(), "passed": True, "classification": raw["classification"], "headline": headline, "decision": decision, "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "protocol": rel(PROTOCOL), "raw": rel(OUT/'raw.json'), "summary": rel(OUT/'summary.md'), "state": rel(STATE), "backup_request": rel(BACKUP_REQUEST), "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT/'raw.json'): raw_hash, rel(OUT/'summary.md'): summary_hash, rel(BACKUP_REQUEST): sha256(BACKUP_REQUEST)}}
    write_json(OUT / "completed.json", completed)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8") + "\nNext action: request/verify external backup, then if v18 failed strict gates freeze either a tiny instrumentation smoke for terminal residuals on preselected positive/neutral/catastrophic opened states or a source-independent coverage acquisition; do not open sealed test.\n", encoding="utf-8")
    print(json.dumps({"headline": headline, "decision": decision, "summary": rel(OUT/'summary.md'), "completed": rel(OUT/'completed.json')}, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
