#!/usr/bin/env python3
"""v20 H12/H15 selector/refit diagnostic after v19 intermediate-horizon evidence.

Development-only IMPROVED offline refit.  v19 showed that true H12 is a much
better tradeoff than H10 on most opened boundary states (oracle H12/H15 value5),
but fixed H12 is still unsafe/unacceptable because a localized fresh_v11 case05
cluster is high-cost and slower than H15.  This script tests the next falsifiable
intervention before spending more simulation budget:

    A conservative H12/H15 selector using deployable state/history features and,
    optionally, online H12 first-solve telemetry can keep zero catastrophic H12
    choices while retaining >=5% measured decision-time saving versus fixed H15
    under strict leave-bank and leave-source nested model selection.

Important semantics:
  * Inputs are existing v19 opened-development branch rollouts only.
  * No MPC simulation, no validation64 access, no sealed-test access, and no
    gradient/RL training are performed.
  * Candidate-level rows average the two v19 paired repeats; repeats are not
    treated as independent training examples.
  * Probe families are charged online overhead: if H12 is probed and rejected,
    the first H12 decision/solver time is added before executing the H15 branch.
  * The primary safety default requires at least one catastrophic H12 training row
    before any H12 can be selected.  This is predeclared here to avoid all-positive
    calibration folds learning unwarranted confidence; the summary calls out when
    a held-out negative source is protected only by this class-absence fallback.
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
NAME = "vehicle_true_variable_horizon_h12_h15_selector_refit_v20"
STAMP = "20260930T0045Z"
SOURCE = Path(__file__).resolve()
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0045_after_h12_h15_selector_refit_v20.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_SELECTOR_REFIT_V20_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-h12-h15-selector-refit-v20-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V19_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/completed.json"
V19_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_preoutcome_frozen_20260930T0015Z.json"
V18D_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_probe_telemetry_v18d_20260930T0035Z/completed.json"
V16B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast_20260929T2320Z/completed.json"

SHORT_H = 12
REF_H = 15
MIN_ROW_TOL = 2.0
MIN_DECISION_SAVING = 0.05
FAMILIES = ["static", "history", "h12_probe", "history_h12_probe"]


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


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = max(0.0, min(float(len(vals) - 1), float(q) * (len(vals) - 1)))
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def row_key(r: Mapping[str, Any]) -> str:
    return str(r["candidate_id"])


def metric_sum(ep: Mapping[str, Any], name: str) -> float:
    obj = ep.get(name)
    if isinstance(obj, Mapping):
        return sf(obj.get("sum"), 0.0)
    return 0.0


def first_control(trace: Sequence[Any]) -> Mapping[str, Any]:
    for tr in trace:
        if isinstance(tr, Mapping) and si(tr.get("step"), -999) == 0:
            return tr
    for tr in trace:
        if isinstance(tr, Mapping) and si(tr.get("step"), -999) >= 0:
            return tr
    return {}


def accepted_attempt(tr: Mapping[str, Any]) -> Mapping[str, Any]:
    rec = tr.get("recovery") if isinstance(tr.get("recovery"), Mapping) else {}
    attempts = rec.get("attempts") if isinstance(rec.get("attempts"), list) else []
    for a in attempts:
        if isinstance(a, Mapping) and a.get("accepted") is True:
            return a
    return attempts[0] if attempts and isinstance(attempts[0], Mapping) else {}


def obs14_from_any(raw: Any) -> List[float]:
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
    return obs14_from_any(tr.get("observation") or tr.get("next_observation") or [])


def first_probe_features(ep_path: str, prefix: str, hashes: Dict[str, str], hash_budget: List[int]) -> Dict[str, float]:
    """Read only first-solve telemetry and summary timing from a v19 episode."""
    d = ROOT / ep_path
    summary_p = d / "summary.json"
    trace_p = d / "trace.json"
    solver_p = d / "solver_calls.jsonl"
    if not summary_p.exists() or not trace_p.exists():
        raise ContractError("missing probe episode trace: " + ep_path)
    summary = read_json(summary_p)
    trace = read_json(trace_p)
    if not isinstance(trace, list):
        raise ContractError("trace not list: " + ep_path)
    if hash_budget[0] < 36:
        for p in (summary_p, trace_p, solver_p):
            if p.exists() and hash_budget[0] < 36:
                hashes[rel(p)] = sha256(p)
                hash_budget[0] += 1
    tr0 = first_control(trace)
    a0 = accepted_attempt(tr0)
    pre = a0.get("warm_start_pre_opt_x") if isinstance(a0.get("warm_start_pre_opt_x"), Mapping) else {}
    post = a0.get("warm_start_post_opt_x") if isinstance(a0.get("warm_start_post_opt_x"), Mapping) else {}
    inp = tr0.get("input") if isinstance(tr0.get("input"), Mapping) else {}
    us = inp.get("u_s") if isinstance(inp.get("u_s"), list) else []
    uo = inp.get("u_omega") if isinstance(inp.get("u_omega"), list) else []
    feats = {
        f"{prefix}_first_decision_s": sf((tr0.get("timing") or {}).get("decision_s")),
        f"{prefix}_first_solver_s": sf(a0.get("solver_s")),
        f"{prefix}_first_objective": sf(a0.get("objective_opt_f_num")),
        f"{prefix}_first_iterations": sf(a0.get("iterations")),
        f"{prefix}_first_constraint_residual_log10": math.log10(max(1e-12, sf(a0.get("constraint_residual"), 0.0))),
        f"{prefix}_first_bound_residual_log10": math.log10(max(1e-12, sf(a0.get("bound_residual"), 0.0))),
        f"{prefix}_pre_mean": sf(pre.get("mean")),
        f"{prefix}_pre_max_abs": sf(pre.get("max_abs")),
        f"{prefix}_post_mean": sf(post.get("mean")),
        f"{prefix}_post_max_abs": sf(post.get("max_abs")),
        f"{prefix}_post_minus_pre_mean": sf(post.get("mean")) - sf(pre.get("mean")),
        f"{prefix}_post_minus_pre_max_abs": sf(post.get("max_abs")) - sf(pre.get("max_abs")),
        f"{prefix}_first_u_s": sf(us[0] if us else 0.0),
        f"{prefix}_first_u_omega": sf(uo[0] if uo else 0.0),
        f"{prefix}_first_performance_reward": sf(tr0.get("performance")),
        f"{prefix}_first_compute_reward": sf(tr0.get("compute")),
        f"{prefix}_first_constraint_reward": sf(tr0.get("constraint")),
        f"{prefix}_summary_decision_sum_s": metric_sum(summary, "decision_timing_s"),
        f"{prefix}_summary_solver_sum_s": metric_sum(summary, "solver_attempt_timing_s"),
    }
    obs = obs14_from_any(tr0.get("observation"))
    for i, x in enumerate(obs):
        feats[f"{prefix}_obs_{i}"] = x
        feats[f"{prefix}_abs_obs_{i}"] = abs(x)
    feats[f"{prefix}_obs01_norm"] = math.hypot(obs[0], obs[1])
    feats[f"{prefix}_obs56_norm"] = math.hypot(obs[5], obs[6])
    feats[f"{prefix}_obs89_norm"] = math.hypot(obs[8], obs[9])
    feats[f"{prefix}_obs1112_norm"] = math.hypot(obs[11], obs[12])
    return feats


def static_features(candidate: Mapping[str, Any]) -> Dict[str, float]:
    o = obs14_from_any(candidate.get("initial_observation_from_h15_trace"))
    prev = candidate.get("branch_previous_state") if isinstance(candidate.get("branch_previous_state"), Mapping) else {}
    th = sf(prev.get("theta"))
    fd: Dict[str, float] = {
        "prev_x_30": sf(prev.get("x")) / 30.0,
        "prev_y_30": sf(prev.get("y")) / 30.0,
        "theta_sin": math.sin(th),
        "theta_cos": math.cos(th),
        "abs_theta_pi": abs(th) / math.pi,
        "branch_step_150": max(0.0, sf(candidate.get("candidate_branch_step"))) / 150.0,
        "offset_from_center_10": sf(candidate.get("offset_from_center")) / 10.0,
        "slot": 1.0 if "slot1" in str(candidate.get("base_state_id")) else 0.0,
        "source_candidate_index_300": sf(candidate.get("source_candidate_index")) / 300.0,
        "obs01_norm": math.hypot(o[0], o[1]),
        "obs56_norm": math.hypot(o[5], o[6]),
        "obs89_norm": math.hypot(o[8], o[9]),
        "obs1112_norm": math.hypot(o[11], o[12]),
        "obs_min_obstacle_norm": min(math.hypot(o[5], o[6]), math.hypot(o[8], o[9]), math.hypot(o[11], o[12])),
    }
    for i in range(14):
        fd[f"obs_{i}"] = o[i]
        fd[f"abs_obs_{i}"] = abs(o[i])
    for a, b in [(0, 1), (0, 2), (1, 2), (5, 6), (8, 9), (11, 12), (7, 10), (10, 13)]:
        fd[f"obs_{a}_x_obs_{b}"] = o[a] * o[b]
    return fd


def add_history_features(fd: Dict[str, float], trace_rel: str, branch_step: int, hashes: Dict[str, str], hash_budget: List[int]) -> bool:
    if not trace_rel:
        return False
    trace_path = ROOT / trace_rel / "trace.json"
    if not trace_path.exists():
        return False
    trace = read_json(trace_path)
    if not isinstance(trace, list) or not trace:
        return False
    if hash_budget[0] < 48:
        hashes[rel(trace_path)] = sha256(trace_path)
        hash_budget[0] += 1
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
            fd[f"hist_obs{i}_std12"] = stdev(xs)
            fd[f"hist_obs{i}_min12"] = min(xs)
            fd[f"hist_obs{i}_max12"] = max(xs)
        pair_norms = []
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
    return True


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, str]]:
    if not V19_RAW.exists() or not V19_DONE.exists() or not V19_PROTOCOL.exists():
        raise ContractError("missing v19 prerequisite artifacts")
    raw = read_json(V19_RAW)
    done = read_json(V19_DONE)
    proto = read_json(V19_PROTOCOL)
    assert_dev_only(raw, "v19 raw")
    assert_dev_only(done, "v19 completed")
    assert_dev_only(proto, "v19 protocol")
    h = done.get("headline") or done.get("result") or {}
    if done.get("hard_pass") is not True and done.get("passed") is not True:
        raise ContractError("v19 not complete")
    if h.get("oracle_H12_H15_value5") is not True:
        raise ContractError("v19 oracle did not justify H12/H15 selector refit")
    hashes: Dict[str, str] = {rel(SOURCE): sha256(SOURCE), rel(V19_RAW): sha256(V19_RAW), rel(V19_DONE): sha256(V19_DONE), rel(V19_PROTOCOL): sha256(V19_PROTOCOL)}
    for p in (V18D_DONE, V16B_DONE):
        if p.exists():
            obj = read_json(p)
            assert_dev_only(obj, rel(p))
            hashes[rel(p)] = sha256(p)
    analysis = raw.get("analysis") if isinstance(raw.get("analysis"), Mapping) else {}
    pair_rows = list(analysis.get("pair_rows") or [])
    cand_rows = list(analysis.get("candidate_rows") or [])
    candidates = list(proto.get("candidates") or [])
    if len(pair_rows) != 48 or len(cand_rows) != 24 or len(candidates) != 24:
        raise ContractError(f"unexpected v19 row counts pair={len(pair_rows)} cand={len(cand_rows)} proto={len(candidates)}")
    proto_by_idx = {si(c.get("candidate_index"), -1): c for c in candidates}
    pairs_by_idx: Dict[int, List[Mapping[str, Any]]] = {}
    for pr in pair_rows:
        pairs_by_idx.setdefault(si(pr.get("candidate_index"), -1), []).append(pr)
    rows: List[Dict[str, Any]] = []
    features: Dict[str, Dict[str, float]] = {}
    trace_hash_budget = [0]
    history_loaded = 0
    for cr in cand_rows:
        idx = si(cr.get("candidate_index"), -1)
        cand = proto_by_idx.get(idx)
        prs = pairs_by_idx.get(idx) or []
        if cand is None or len(prs) != 2:
            raise ContractError(f"missing candidate/pairs for idx {idx}")
        h12_first_decisions = []
        h12_first_solvers = []
        h15_first_decisions = []
        h15_first_solvers = []
        probe_accum: Dict[str, List[float]] = {}
        for pr in prs:
            h12_feats = first_probe_features(str(pr.get("short_path")), "h12", hashes, trace_hash_budget)
            h15_feats = first_probe_features(str(pr.get("ref_path")), "h15", hashes, trace_hash_budget)
            h12_first_decisions.append(sf(h12_feats.get("h12_first_decision_s")))
            h12_first_solvers.append(sf(h12_feats.get("h12_first_solver_s")))
            h15_first_decisions.append(sf(h15_feats.get("h15_first_decision_s")))
            h15_first_solvers.append(sf(h15_feats.get("h15_first_solver_s")))
            # Only first-solve telemetry is used as candidate features; full summary
            # fields are intentionally not copied into predictor features except
            # first timing already above.
            for k, v in {**h12_feats, **h15_feats}.items():
                if "summary_" in k:
                    continue
                probe_accum.setdefault(k, []).append(sf(v))
        pair_h12_phys = [sf(p.get("short_physical")) for p in prs]
        pair_h15_phys = [sf(p.get("ref_physical")) for p in prs]
        pair_h12_dec = [sf(p.get("short_decision_sum_s")) for p in prs]
        pair_h15_dec = [sf(p.get("ref_decision_sum_s")) for p in prs]
        pair_h12_sol = [sf(p.get("short_solver_sum_s")) for p in prs]
        pair_h15_sol = [sf(p.get("ref_solver_sum_s")) for p in prs]
        phys_delta = mean(pair_h12_phys) - mean(pair_h15_phys)
        decision_gain = mean(pair_h15_dec) - mean(pair_h12_dec)
        solver_gain = mean(pair_h15_sol) - mean(pair_h12_sol)
        tol = max(MIN_ROW_TOL, 0.05 * abs(mean(pair_h15_phys)))
        catastrophic = any(p.get("short_catastrophic_vs_ref") is True for p in prs)
        beneficial = bool((not catastrophic) and all(p.get("short_beneficial_vs_ref") is True for p in prs) and phys_delta <= tol and decision_gain > 0.0)
        row = {
            "candidate_id": str(cr.get("candidate_id")),
            "candidate_index": idx,
            "bank_id": str(cr.get("bank_id")),
            "source_key": str(cr.get("source_key")),
            "base_state_id": str(cr.get("base_state_id")),
            "role": str(cr.get("role")),
            "offset_from_center": si(cr.get("offset_from_center"), 0),
            "branch_step": si(cr.get("branch_step"), si(cand.get("candidate_branch_step"), -1)),
            "h12_physical": mean(pair_h12_phys),
            "h15_physical": mean(pair_h15_phys),
            "h12_decision_sum_s": mean(pair_h12_dec),
            "h15_decision_sum_s": mean(pair_h15_dec),
            "h12_solver_sum_s": mean(pair_h12_sol),
            "h15_solver_sum_s": mean(pair_h15_sol),
            "h12_first_decision_s": mean(h12_first_decisions),
            "h12_first_solver_s": mean(h12_first_solvers),
            "h15_first_decision_s": mean(h15_first_decisions),
            "h15_first_solver_s": mean(h15_first_solvers),
            "phys_delta_h12_minus_h15": phys_delta,
            "decision_gain_h12_vs_h15_s": decision_gain,
            "solver_gain_h12_vs_h15_s": solver_gain,
            "row_physical_tolerance": tol,
            "h12_catastrophic_vs_h15": catastrophic,
            "h12_beneficial_vs_h15": beneficial,
            "pair_repeats": len(prs),
            "catastrophic_repeats": sum(1 for p in prs if p.get("short_catastrophic_vs_ref") is True),
            "beneficial_repeats": sum(1 for p in prs if p.get("short_beneficial_vs_ref") is True),
        }
        fd = static_features(cand)
        if add_history_features(fd, str(cand.get("h15_trace_episode_path") or ""), si(cand.get("candidate_branch_step"), -1), hashes, trace_hash_budget):
            history_loaded += 1
        for k, xs in probe_accum.items():
            fd[k] = mean(xs)
        rows.append(row)
        features[row_key(row)] = fd
    diag = {
        "rows": len(rows),
        "banks": sorted({r["bank_id"] for r in rows}),
        "source_key_count": len({r["source_key"] for r in rows}),
        "positive_rows": sum(1 for r in rows if r["h12_beneficial_vs_h15"]),
        "catastrophic_rows": sum(1 for r in rows if r["h12_catastrophic_vs_h15"]),
        "history_loaded_rows": history_loaded,
        "fixed_H12_decision_relative_saving_vs_H15": (math.fsum(r["h15_decision_sum_s"] - r["h12_decision_sum_s"] for r in rows) / math.fsum(r["h15_decision_sum_s"] for r in rows)),
        "fixed_H12_solver_relative_saving_vs_H15": (math.fsum(r["h15_solver_sum_s"] - r["h12_solver_sum_s"] for r in rows) / math.fsum(r["h15_solver_sum_s"] for r in rows)),
        "fixed_H12_physical_delta_vs_H15": math.fsum(r["h12_physical"] - r["h15_physical"] for r in rows),
        "fixed_H12_bad_rows": sum(1 for r in rows if r["h12_catastrophic_vs_h15"]),
        "negative_sources": sorted({r["source_key"] for r in rows if r["h12_catastrophic_vs_h15"]}),
    }
    return rows, features, diag, hashes


def feature_subset(all_features: Mapping[str, Mapping[str, float]], family: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, fd0 in all_features.items():
        if family == "static":
            fd = {n: v for n, v in fd0.items() if not n.startswith("hist_") and not n.startswith("h12_") and not n.startswith("h15_")}
        elif family == "history":
            fd = {n: v for n, v in fd0.items() if not n.startswith("h12_") and not n.startswith("h15_")}
        elif family == "h12_probe":
            fd = {n: v for n, v in fd0.items() if (not n.startswith("hist_") and not n.startswith("h15_"))}
        elif family == "history_h12_probe":
            fd = {n: v for n, v in fd0.items() if not n.startswith("h15_")}
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
        s = stdev(xs)
        means[nm] = m
        stds[nm] = s if s > 1e-12 else 1.0
    return names, means, stds


def vector(row: Mapping[str, Any], fdict: Mapping[str, Mapping[str, float]], names: Sequence[str], means: Mapping[str, float], stds: Mapping[str, float]) -> List[float]:
    fd = fdict[row_key(row)]
    return [(sf(fd.get(nm)) - sf(means.get(nm))) / max(sf(stds.get(nm), 1.0), 1e-12) for nm in names]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def support_radius(pos_rows: Sequence[Mapping[str, Any]], vecs: Mapping[str, Sequence[float]], cfg: Mapping[str, Any]) -> float:
    if len(pos_rows) < 2:
        return 0.0
    ds: List[float] = []
    for i, r in enumerate(pos_rows):
        others = [dist(vecs[row_key(r)], vecs[row_key(q)]) for j, q in enumerate(pos_rows) if i != j]
        if others:
            ds.append(min(others))
    if not ds:
        return 0.0
    return quantile(ds, sf(cfg.get("support_q"), 0.75)) * sf(cfg.get("support_mult"), 1.0)


def predict_choices(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], all_features: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], include_scores: bool = False) -> Tuple[Dict[str, int], Dict[str, Any], int]:
    family = str(cfg["family"])
    fdict = feature_subset(all_features, family)
    names, means, stds = train_scaler(train_rows, fdict)
    train_vecs = {row_key(r): vector(r, fdict, names, means, stds) for r in train_rows}
    pos_rows = [r for r in train_rows if bool(r["h12_beneficial_vs_h15"])]
    cat_rows = [r for r in train_rows if bool(r["h12_catastrophic_vs_h15"])]
    radius = support_radius(pos_rows, train_vecs, cfg)
    choices: Dict[str, int] = {}
    scores: Dict[str, Any] = {}
    if not train_rows or not pos_rows or len(cat_rows) < si(cfg.get("min_cat_train_for_h12"), 1):
        for er in eval_rows:
            choices[row_key(er)] = REF_H
            if include_scores:
                scores[row_key(er)] = {"selected_h": REF_H, "reason": "insufficient_positive_or_catastrophic_training_support", "feature_count": len(names), "train_cat_count": len(cat_rows), "train_pos_count": len(pos_rows)}
        return choices, scores, len(names)
    for er in eval_rows:
        ev = vector(er, fdict, names, means, stds)
        neigh = sorted([(dist(ev, train_vecs[row_key(tr)]), tr) for tr in train_rows], key=lambda z: (z[0], row_key(z[1])))
        pos_neigh = sorted([(dist(ev, train_vecs[row_key(tr)]), tr) for tr in pos_rows], key=lambda z: (z[0], row_key(z[1])))
        cat_neigh = sorted([(dist(ev, train_vecs[row_key(tr)]), tr) for tr in cat_rows], key=lambda z: (z[0], row_key(z[1])))
        kk = max(1, min(si(cfg.get("k"), 1), len(neigh)))
        near = neigh[:kk]
        weights = [1.0 / max(0.05, d) for d, _ in near]
        sw = math.fsum(weights) if weights else 1.0
        cat_w = math.fsum(w for w, (_, r) in zip(weights, near) if bool(r["h12_catastrophic_vs_h15"]))
        # A weak beta-prior prevents tiny neighbour sets from becoming exactly zero-risk.
        prior = sf(cfg.get("risk_prior_alpha"), 0.5)
        risk_hat = (cat_w + 0.5 * prior) / (sw + prior)
        kp = max(1, min(si(cfg.get("k"), 1), len(pos_neigh)))
        pnear = pos_neigh[:kp]
        pred_gain = mean([sf(r["decision_gain_h12_vs_h15_s"]) for _, r in pnear]) if pnear else -1e9
        pred_phys = mean([sf(r["phys_delta_h12_minus_h15"]) for _, r in pnear]) if pnear else 1e9
        dpos = pnear[-1][0] if pnear else 1e9
        dcat = cat_neigh[0][0] if cat_neigh else 1e9
        support_ok = dpos <= radius
        cat_guard_ok = dcat > sf(cfg.get("cat_guard_ratio"), 1.0) * max(dpos, 1e-9) + sf(cfg.get("cat_margin"), 0.0)
        choose = bool(
            support_ok
            and cat_guard_ok
            and risk_hat <= sf(cfg.get("risk_max"), 0.5)
            and pred_gain >= sf(cfg.get("gain_min_s"), 0.0)
            and pred_phys <= sf(cfg.get("phys_max"), 2.0)
        )
        choices[row_key(er)] = SHORT_H if choose else REF_H
        if include_scores:
            scores[row_key(er)] = {
                "selected_h": choices[row_key(er)],
                "risk_hat": risk_hat,
                "pred_gain_s": pred_gain,
                "pred_phys_delta": pred_phys,
                "dpos": dpos,
                "dcat": dcat,
                "support_radius": radius,
                "support_ok": support_ok,
                "cat_guard_ok": cat_guard_ok,
                "feature_count": len(names),
                "train_cat_count": len(cat_rows),
                "train_pos_count": len(pos_rows),
            }
    return choices, scores, len(names)


def overhead_mode_for_family(family: str) -> str:
    return "h12_probe" if "probe" in family else "none"


def eval_choices(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int], overhead_mode: str) -> Dict[str, Any]:
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0
    extra_probe_dec = extra_probe_sol = 0.0
    counts: Counter[str] = Counter()
    conf: Counter[str] = Counter()
    bad: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for r in rows:
        h = int(choices.get(row_key(r), REF_H))
        counts[str(h)] += 1
        pos = bool(r["h12_beneficial_vs_h15"])
        cat = bool(r["h12_catastrophic_vs_h15"])
        extra_d = extra_s = 0.0
        if overhead_mode == "h12_probe" and h == REF_H:
            extra_d = sf(r["h12_first_decision_s"])
            extra_s = sf(r["h12_first_solver_s"])
        if h == SHORT_H:
            pol_phys += sf(r["h12_physical"])
            pol_dec += sf(r["h12_decision_sum_s"])
            pol_sol += sf(r["h12_solver_sum_s"])
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({
                    "candidate_id": row_key(r),
                    "bank_id": r.get("bank_id"),
                    "source_key": r.get("source_key"),
                    "role": r.get("role"),
                    "offset_from_center": r.get("offset_from_center"),
                    "phys_delta": sf(r["phys_delta_h12_minus_h15"]),
                    "decision_gain_s": sf(r["decision_gain_h12_vs_h15_s"]),
                })
        else:
            pol_phys += sf(r["h15_physical"])
            pol_dec += sf(r["h15_decision_sum_s"]) + extra_d
            pol_sol += sf(r["h15_solver_sum_s"]) + extra_s
            extra_probe_dec += extra_d
            extra_probe_sol += extra_s
            conf["FN" if pos else "TN"] += 1
        details.append({
            "candidate_id": row_key(r),
            "bank_id": r.get("bank_id"),
            "source_key": r.get("source_key"),
            "selected_h": h,
            "label_positive": pos,
            "catastrophic": cat,
            "phys_delta": sf(r["phys_delta_h12_minus_h15"]),
            "decision_gain_s": sf(r["decision_gain_h12_vs_h15_s"]),
            "extra_probe_decision_s": extra_d,
            "extra_probe_solver_s": extra_s,
        })
    phys_delta = pol_phys - fixed_phys
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    return {
        "rows": len(rows),
        "overhead_mode": overhead_mode,
        "chosen_counts": dict(counts),
        "confusion": dict(conf),
        "policy_physical_sum": pol_phys,
        "fixed_H15_physical_sum": fixed_phys,
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "physical_gate": phys_delta <= tol_sum,
        "policy_decision_sum_s": pol_dec,
        "fixed_H15_decision_sum_s": fixed_dec,
        "extra_probe_decision_sum_s": extra_probe_dec,
        "decision_relative_saving_vs_fixed_H15": dec_save,
        "policy_solver_sum_s": pol_sol,
        "fixed_H15_solver_sum_s": fixed_sol,
        "extra_probe_solver_sum_s": extra_probe_sol,
        "solver_relative_saving_vs_fixed_H15": sol_save,
        "catastrophic_false_positive_rows": bad,
        "pass_5pct_no_cat_fp": bool(dec_save >= MIN_DECISION_SAVING and phys_delta <= tol_sum and not bad),
        "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad),
        "details": details,
    }


def eval_cfg(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    choices, scores, feature_count = predict_choices(train_rows, eval_rows, features, cfg, include_scores=include_scores)
    ev = eval_choices(eval_rows, choices, overhead_mode_for_family(str(cfg["family"])))
    ev["feature_count"] = feature_count
    if include_scores:
        ev["prediction_scores"] = scores
    return ev


def aggregate_from_details(rows: Sequence[Mapping[str, Any]], holdouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    choices: Dict[str, int] = {}
    overhead_modes = set()
    # In a given aggregate call all holdouts come from the same config/family.
    for ev in holdouts.values():
        overhead_modes.add(str(ev.get("overhead_mode", "none")))
        for d in ev.get("details") or []:
            choices[str(d["candidate_id"])] = int(d["selected_h"])
    mode = sorted(overhead_modes)[0] if overhead_modes else "none"
    return eval_choices(rows, choices, mode)


def eval_group_lobo(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], group_field: str, features: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], blocked_groups: Sequence[str] = (), include_scores: bool = False) -> Dict[str, Any]:
    blocked = set(str(g) for g in blocked_groups)
    eval_groups = [g for g in groups if g not in blocked]
    holdouts: Dict[str, Any] = {}
    for hg in eval_groups:
        train = [r for r in rows if str(r[group_field]) not in blocked and str(r[group_field]) != hg]
        evrows = [r for r in rows if str(r[group_field]) == hg]
        holdouts[hg] = eval_cfg(train, evrows, features, cfg, include_scores=include_scores)
    eval_rows = [r for r in rows if str(r[group_field]) in eval_groups]
    return {"holdouts": holdouts, "aggregate": aggregate_from_details(eval_rows, holdouts)}


def summary(ev: Mapping[str, Any]) -> Dict[str, Any]:
    agg = ev["aggregate"] if "aggregate" in ev else ev
    holds = ev.get("holdouts") if isinstance(ev.get("holdouts"), Mapping) else {}
    saves = [sf(x.get("decision_relative_saving_vs_fixed_H15")) for x in holds.values()]
    return {
        "aggregate_bad": len(agg.get("catastrophic_false_positive_rows") or []),
        "aggregate_physical_gate": bool(agg.get("physical_gate")),
        "aggregate_save": sf(agg.get("decision_relative_saving_vs_fixed_H15")),
        "aggregate_solver_save": sf(agg.get("solver_relative_saving_vs_fixed_H15")),
        "aggregate_h12": int((agg.get("chosen_counts") or {}).get(str(SHORT_H), 0)),
        "aggregate_pass5": bool(agg.get("pass_5pct_no_cat_fp")),
        "aggregate_pass10": bool(agg.get("pass_10pct_no_cat_fp")),
        "min_holdout_save": min(saves) if saves else sf(agg.get("decision_relative_saving_vs_fixed_H15")),
        "avg_holdout_save": mean(saves) if saves else sf(agg.get("decision_relative_saving_vs_fixed_H15")),
        "holdout_bad_total": sum(len(x.get("catastrophic_false_positive_rows") or []) for x in holds.values()),
        "holdout_phys_fail_count": sum(0 if x.get("physical_gate") else 1 for x in holds.values()),
    }


def rank_summary(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(s["aggregate_bad"]),
        0 if bool(s["aggregate_physical_gate"]) else 1,
        not bool(s["aggregate_pass10"]),
        not bool(s["aggregate_pass5"]),
        -sf(s["aggregate_save"]),
        -sf(s["aggregate_solver_save"]),
        -int(s["aggregate_h12"]),
        int(s["holdout_bad_total"]),
        int(s["holdout_phys_fail_count"]),
    )


def cfg_grid() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for k in [1, 3]:
            for risk_max in [0.25, 0.50, 0.75]:
                for support_q in [0.50, 0.75, 1.00]:
                    for support_mult in [0.75, 1.25]:
                        for gain_min_s in [0.0, 0.20]:
                            for phys_max in [2.0, 6.0]:
                                for cat_guard_ratio in [1.0, 1.5, 3.0]:
                                    cfgs.append({
                                        "family": family,
                                        "k": k,
                                        "risk_max": risk_max,
                                        "support_q": support_q,
                                        "support_mult": support_mult,
                                        "gain_min_s": gain_min_s,
                                        "phys_max": phys_max,
                                        "cat_guard_ratio": cat_guard_ratio,
                                        "cat_margin": 0.0,
                                        "risk_prior_alpha": 0.5,
                                        "min_cat_train_for_h12": 1,
                                    })
    return cfgs


def cfg_id(cfg: Mapping[str, Any]) -> str:
    return "v20_%s_k%s_r%s_q%s_m%s_g%s_p%s_cg%s" % (
        cfg["family"], cfg["k"], cfg["risk_max"], cfg["support_q"], cfg["support_mult"], cfg["gain_min_s"], cfg["phys_max"], cfg["cat_guard_ratio"]
    )


def nested_selection(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], cfgs: Sequence[Mapping[str, Any]], group_field: str) -> Tuple[Dict[str, Any], int]:
    groups = sorted({str(r[group_field]) for r in rows})
    outer: Dict[str, Any] = {}
    choices: Dict[str, int] = {}
    eval_count = 0
    for outer_group in groups:
        inner_rows = [r for r in rows if str(r[group_field]) != outer_group]
        inner_groups = sorted({str(r[group_field]) for r in inner_rows})
        candidates: List[Dict[str, Any]] = []
        for cfg in cfgs:
            lobo = eval_group_lobo(inner_rows, inner_groups, group_field, features, cfg, include_scores=False)
            eval_count += len(inner_groups)
            candidates.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summary(lobo)})
        selected = sorted(candidates, key=lambda x: rank_summary(x["summary"]))[0]
        train_outer = inner_rows
        eval_outer = [r for r in rows if str(r[group_field]) == outer_group]
        outer_eval = eval_cfg(train_outer, eval_outer, features, selected["config"], include_scores=True)
        eval_count += 1
        for d in outer_eval.get("details") or []:
            choices[str(d["candidate_id"])] = int(d["selected_h"])
        outer[outer_group] = {
            "selected_config_id": selected["config_id"],
            "selected_config": selected["config"],
            "inner_summary": selected["summary"],
            "outer_eval": compact_eval(outer_eval),
            "outer_train_cat_count": sum(1 for r in train_outer if r["h12_catastrophic_vs_h15"]),
            "outer_train_pos_count": sum(1 for r in train_outer if r["h12_beneficial_vs_h15"]),
        }
    # Use per-row overhead from the selected family.  Because different outer
    # groups can select different families, aggregate manually from details with
    # already charged outer_eval row decisions.
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = extra_dec = extra_sol = 0.0
    counts: Counter[str] = Counter(); conf: Counter[str] = Counter(); bad: List[Dict[str, Any]] = []; details: List[Dict[str, Any]] = []
    detail_by_id: Dict[str, Mapping[str, Any]] = {}
    # Find exact outer details and their charged overheads.
    for obj in outer.values():
        for d in obj["outer_eval"].get("details") or []:
            detail_by_id[str(d["candidate_id"])] = d
    for r in rows:
        d = detail_by_id.get(row_key(r), {"selected_h": REF_H, "extra_probe_decision_s": 0.0, "extra_probe_solver_s": 0.0})
        h = int(d.get("selected_h", REF_H)); counts[str(h)] += 1
        pos = bool(r["h12_beneficial_vs_h15"]); cat = bool(r["h12_catastrophic_vs_h15"])
        ed = sf(d.get("extra_probe_decision_s")); es = sf(d.get("extra_probe_solver_s"))
        if h == SHORT_H:
            pol_phys += sf(r["h12_physical"]); pol_dec += sf(r["h12_decision_sum_s"]); pol_sol += sf(r["h12_solver_sum_s"])
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({"candidate_id": row_key(r), "bank_id": r.get("bank_id"), "source_key": r.get("source_key"), "role": r.get("role"), "offset_from_center": r.get("offset_from_center"), "phys_delta": sf(r["phys_delta_h12_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h12_vs_h15_s"])})
        else:
            pol_phys += sf(r["h15_physical"]); pol_dec += sf(r["h15_decision_sum_s"]) + ed; pol_sol += sf(r["h15_solver_sum_s"]) + es
            extra_dec += ed; extra_sol += es
            conf["FN" if pos else "TN"] += 1
        details.append({"candidate_id": row_key(r), "bank_id": r.get("bank_id"), "source_key": r.get("source_key"), "selected_h": h, "label_positive": pos, "catastrophic": cat, "extra_probe_decision_s": ed, "extra_probe_solver_s": es})
    phys_delta = pol_phys - fixed_phys
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    aggregate = {
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
        "extra_probe_decision_sum_s": extra_dec,
        "decision_relative_saving_vs_fixed_H15": dec_save,
        "policy_solver_sum_s": pol_sol,
        "fixed_H15_solver_sum_s": fixed_sol,
        "extra_probe_solver_sum_s": extra_sol,
        "solver_relative_saving_vs_fixed_H15": sol_save,
        "catastrophic_false_positive_rows": bad,
        "pass_5pct_no_cat_fp": bool(dec_save >= MIN_DECISION_SAVING and phys_delta <= tol_sum and not bad),
        "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad),
        "details": details,
    }
    return {"group_field": group_field, "groups": groups, "outer": outer, "aggregate": aggregate}, eval_count


def compact_eval(ev: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(ev)
    if len(out.get("details") or []) > 20:
        out["details"] = list(out["details"][:20])
        out["details_truncated"] = True
    if "prediction_scores" in out and len(out["prediction_scores"]) > 12:
        keys = sorted(out["prediction_scores"])[:12]
        out["prediction_scores"] = {k: out["prediction_scores"][k] for k in keys}
        out["prediction_scores_truncated"] = True
    return out


def feature_counts(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for fam in FAMILIES:
        fdict = feature_subset(features, fam)
        names, _, _ = train_scaler(rows, fdict)
        out[fam] = len(names)
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows, features, diag, hashes = load_rows()
    cfgs = cfg_grid()
    banks = sorted({str(r["bank_id"]) for r in rows})
    sources = sorted({str(r["source_key"]) for r in rows})
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_H12_H15_selector_refit_no_sim_no_validation_no_test",
        "hypothesis": "A conservative true-H12/H15 selector can exploit v19's oracle H12/H15 opportunity while avoiding the localized fresh_v11 case05 H12 high-cost/slower cluster under strict bank/source nested splits.",
        "before_evidence": [
            "v16b/v17/v18d show H10 static/probe selectors fail strict useful-safety gates or collapse to H15.",
            "v19 fixed H12 saves 14.32% decision time but has six catastrophic/high-cost rows; oracle H12/H15 saves 15.72% decision time with physical gate.",
            "All v19 H12 catastrophic rows are in one opened-development source, fresh_v11/fresh_case05_slot1_mid_late_control; this script tests whether conservative support/uncertainty can handle that under leave-bank/source splits without new simulation.",
        ],
        "inputs": {"v19_raw": rel(V19_RAW), "v19_completed": rel(V19_DONE), "v19_protocol": rel(V19_PROTOCOL)},
        "row_diagnostic_preoutcome": diag,
        "candidate_rows_are_repeats_averaged": True,
        "families": FAMILIES,
        "config_count": len(cfgs),
        "primary_safety_default": "min_cat_train_for_h12=1 for every config; if an outer training fold has zero H12-catastrophic examples, all rows in that outer fold are assigned H15 as an uncertainty veto",
        "strict_splits": ["leave_bank_nested", "leave_source_key_nested"],
        "decision_gate": {"both_strict_splits_pass5": True, "zero_catastrophic_H12_false_positives": True, "physical_gate": True, "minimum_measured_decision_saving_vs_fixed_H15": MIN_DECISION_SAVING},
        "overhead_semantics": "probe families add mean first-H12 decision/solver time when H12 is rejected and H15 is executed; static/history families have no measured online selector overhead yet",
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 250000, "offline_model_evaluations_cap": 250000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": args.backup_verified_commit,
        "input_hashes": hashes,
    }
    write_json(PROTOCOL, proto)

    selector_evals = 0
    global_bank_items: List[Dict[str, Any]] = []
    global_source_items: List[Dict[str, Any]] = []
    for cfg in cfgs:
        lb = eval_group_lobo(rows, banks, "bank_id", features, cfg)
        selector_evals += len(banks)
        ls = eval_group_lobo(rows, sources, "source_key", features, cfg)
        selector_evals += len(sources)
        global_bank_items.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summary(lb), "aggregate": lb["aggregate"]})
        global_source_items.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summary(ls), "aggregate": ls["aggregate"]})
    top_global_bank = sorted(global_bank_items, key=lambda x: rank_summary(x["summary"]))[:20]
    top_global_source = sorted(global_source_items, key=lambda x: rank_summary(x["summary"]))[:20]

    strict_bank, n_bank = nested_selection(rows, features, cfgs, "bank_id")
    strict_source, n_source = nested_selection(rows, features, cfgs, "source_key")
    selector_evals += n_bank + n_source

    hbank = summary(strict_bank)
    hsource = summary(strict_source)
    pass_both = bool(strict_bank["aggregate"]["pass_5pct_no_cat_fp"] and strict_source["aggregate"]["pass_5pct_no_cat_fp"])
    # Identify when the unique negative source is protected only by the safety default.
    negative_source = diag.get("negative_sources", [])
    source_outer_notes: Dict[str, Any] = {}
    for s in negative_source:
        obj = strict_source["outer"].get(s)
        if obj:
            source_outer_notes[s] = {
                "outer_train_cat_count": obj.get("outer_train_cat_count"),
                "selected_config_id": obj.get("selected_config_id"),
                "outer_chosen_counts": obj.get("outer_eval", {}).get("chosen_counts"),
                "outer_bad": len(obj.get("outer_eval", {}).get("catastrophic_false_positive_rows") or []),
                "outer_save": obj.get("outer_eval", {}).get("decision_relative_saving_vs_fixed_H15"),
            }
    headline = {
        "rows": len(rows),
        "banks": banks,
        "source_key_count": len(sources),
        "positive_rows": diag["positive_rows"],
        "catastrophic_rows": diag["catastrophic_rows"],
        "negative_sources": negative_source,
        "history_loaded_rows": diag["history_loaded_rows"],
        "feature_counts": feature_counts(rows, features),
        "config_count": len(cfgs),
        "selector_refit_evaluations": selector_evals,
        "fixed_H12_save": diag["fixed_H12_decision_relative_saving_vs_H15"],
        "fixed_H12_solver_save": diag["fixed_H12_solver_relative_saving_vs_H15"],
        "fixed_H12_bad": diag["fixed_H12_bad_rows"],
        "strict_bank_save": hbank["aggregate_save"],
        "strict_bank_solver_save": hbank["aggregate_solver_save"],
        "strict_bank_bad": hbank["aggregate_bad"],
        "strict_bank_h12": hbank["aggregate_h12"],
        "strict_bank_physical_gate": hbank["aggregate_physical_gate"],
        "strict_bank_pass5": hbank["aggregate_pass5"],
        "strict_source_save": hsource["aggregate_save"],
        "strict_source_solver_save": hsource["aggregate_solver_save"],
        "strict_source_bad": hsource["aggregate_bad"],
        "strict_source_h12": hsource["aggregate_h12"],
        "strict_source_physical_gate": hsource["aggregate_physical_gate"],
        "strict_source_pass5": hsource["aggregate_pass5"],
        "global_bank_pass5_count": sum(1 for x in global_bank_items if x["summary"]["aggregate_pass5"]),
        "global_source_pass5_count": sum(1 for x in global_source_items if x["summary"]["aggregate_pass5"]),
        "pass_both_strict_splits": pass_both,
    }
    if pass_both:
        decision = "v20 H12/H15 conservative selector passes both strict opened bank and source nested >=5% zero-catastrophe gates, but the single held-out negative source is protected by a no-cat-training uncertainty fallback. Next do not validate yet: freeze either an online overhead smoke and/or source-independent H12-negative acquisition to test whether this safety rule generalizes beyond one negative cluster."
    elif hbank["aggregate_pass5"] or hsource["aggregate_pass5"]:
        decision = "v20 partially passes one strict split but not both; this indicates H12/H15 opportunity is real but current selector evidence is not source/bank robust enough. Next inspect failing outer groups and acquire source-independent H12 negative/support data or richer terminal-risk features before validation."
    elif hbank["aggregate_bad"] == 0 and hsource["aggregate_bad"] == 0:
        decision = "v20 avoids catastrophic H12 selections but collapses below the 5% measured decision-saving gate; H12/H15 static/probe uncertainty is too conservative. Next prioritize source-independent H12 opportunity/negative acquisition or terminal-risk value training, not validation."
    else:
        decision = "v20 H12/H15 selector still makes catastrophic held-out choices; H12 improved the horizon grid but selector representation/data remain insufficient. Next acquire independent negative H12 boundary states and/or train richer risk/terminal-value models."
    raw = {
        "created_utc": now().isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(),
        "classification": proto["classification"],
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "row_diag": diag,
        "headline": headline,
        "negative_source_outer_notes": source_outer_notes,
        "strict_bank_nested": strict_bank,
        "strict_source_nested": strict_source,
        "top_global_bank_lobo": top_global_bank,
        "top_global_source_lobo": top_global_source,
        "decision": decision,
        "budget_declared": proto["budget_declared"],
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations": selector_evals, "offline_model_evaluations": selector_evals, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "input_hashes": hashes,
        "platform": {"python": sys.version, "platform": platform.platform()},
        "interpretation_limits": [
            "opened-development v19 boundary rows only",
            "candidate-level repeats are averaged; n=24 candidate states",
            "only one H12-catastrophic source exists, so source-independent negative generalization cannot be proven",
            "static/history families do not include measured online selector overhead",
            "probe-family overhead is approximated from first-H12 solve in saved v19 traces",
            "no validation64 or sealed final-test evidence",
        ],
    }
    return raw


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v20 H12/H15 selector/refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED offline H12/H15 selector diagnostic; no MPC simulation, no validation64, no sealed test, no gradient/RL training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` candidate states (two v19 repeats averaged), banks `{h['banks']}`, source keys `{h['source_key_count']}`.",
        f"- Labels: H12 beneficial `{h['positive_rows']}`, H12 catastrophic/high-cost `{h['catastrophic_rows']}`, negative sources `{h['negative_sources']}`.",
        f"- Fixed H12 vs fixed H15 on these rows: decision saving `{pct(h['fixed_H12_save'])}`, solver saving `{pct(h['fixed_H12_solver_save'])}`, bad `{h['fixed_H12_bad']}`.",
        f"- Features: history loaded `{h['history_loaded_rows']}/{h['rows']}`, counts `{h['feature_counts']}`; configs `{h['config_count']}`; selector/offline evaluations `{h['selector_refit_evaluations']}`.",
        f"- Strict leave-bank nested: save `{pct(h['strict_bank_save'])}`, solver save `{pct(h['strict_bank_solver_save'])}`, bad `{h['strict_bank_bad']}`, H12 `{h['strict_bank_h12']}`, physical gate `{h['strict_bank_physical_gate']}`, pass5 `{h['strict_bank_pass5']}`.",
        f"- Strict leave-source nested: save `{pct(h['strict_source_save'])}`, solver save `{pct(h['strict_source_solver_save'])}`, bad `{h['strict_source_bad']}`, H12 `{h['strict_source_h12']}`, physical gate `{h['strict_source_physical_gate']}`, pass5 `{h['strict_source_pass5']}`.",
        f"- Global diagnostic pass5 counts: bank `{h['global_bank_pass5_count']}`, source `{h['global_source_pass5_count']}`. Primary both-split pass: `{h['pass_both_strict_splits']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Strict leave-bank outer groups",
        "",
        "| outer bank | train cats | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer save | outer solver save |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g, obj in raw["strict_bank_nested"]["outer"].items():
        ev = obj["outer_eval"]; inn = obj["inner_summary"]
        lines.append(f"| `{g}` | {obj['outer_train_cat_count']} | `{obj['selected_config_id']}` | `{inn['aggregate_pass5']}` | {inn['aggregate_bad']} | `{ev['chosen_counts']}` | {len(ev.get('catastrophic_false_positive_rows') or [])} | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Strict leave-source outer groups", "", "| outer source | train cats | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer save | outer solver save |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g, obj in raw["strict_source_nested"]["outer"].items():
        ev = obj["outer_eval"]; inn = obj["inner_summary"]
        lines.append(f"| `{g}` | {obj['outer_train_cat_count']} | `{obj['selected_config_id']}` | `{inn['aggregate_pass5']}` | {inn['aggregate_bad']} | `{ev['chosen_counts']}` | {len(ev.get('catastrophic_false_positive_rows') or [])} | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Negative-source uncertainty note", ""]
    if raw.get("negative_source_outer_notes"):
        for s, note in raw["negative_source_outer_notes"].items():
            lines.append(f"- `{s}`: train_cat_count `{note['outer_train_cat_count']}`, selected `{note['selected_config_id']}`, outer H counts `{note['outer_chosen_counts']}`, bad `{note['outer_bad']}`, save `{pct(note['outer_save'])}`.")
    else:
        lines.append("No negative source note available.")
    lines += ["", "## Top global leave-bank configs (diagnostic only)", "", "| rank | config | pass5 | bad | physical gate | save | solver save | H12 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_bank_lobo"][:10], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['aggregate_pass5']}` | {s['aggregate_bad']} | `{s['aggregate_physical_gate']}` | {pct(s['aggregate_save'])} | {pct(s['aggregate_solver_save'])} | {s['aggregate_h12']} |")
    lines += ["", "## Top global leave-source configs (diagnostic only)", "", "| rank | config | pass5 | bad | physical gate | save | solver save | H12 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_source_lobo"][:10], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['aggregate_pass5']}` | {s['aggregate_bad']} | `{s['aggregate_physical_gate']}` | {pct(s['aggregate_save'])} | {pct(s['aggregate_solver_save'])} | {s['aggregate_h12']} |")
    bad_bank = raw["strict_bank_nested"]["aggregate"].get("catastrophic_false_positive_rows") or []
    bad_source = raw["strict_source_nested"]["aggregate"].get("catastrophic_false_positive_rows") or []
    lines += ["", "## Strict catastrophic false positives", ""]
    if not bad_bank and not bad_source:
        lines.append("No strict nested catastrophic H12 false positives.")
    else:
        for label, rows in [("bank", bad_bank), ("source", bad_source)]:
            if rows:
                lines.append(f"### {label}")
                lines.append("| candidate | bank | source | phys delta | decision gain s |")
                lines.append("|---|---|---|---:|---:|")
                for r in rows:
                    lines.append(f"| `{r.get('candidate_id')}` | `{r.get('bank_id')}` | `{r.get('source_key')}` | {sf(r.get('phys_delta')):.6g} | {sf(r.get('decision_gain_s')):.6g} |")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is still opened-development mechanism evidence. It does not authorize validation64 or final-test claims. Passing both strict splits would justify only a fresh development confirmation/overhead smoke and likely additional H12-negative-source acquisition, because v19 contains only one H12-catastrophic source. Failure means intermediate H12 helped the horizon design but not yet the deployable selector/data problem.",
        "",
        f"Protocol: `{rel(PROTOCOL)}`. Raw: `{rel(OUT / 'raw.json')}`. Completed: `{rel(OUT / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H v20 H12/H15 selector/refit

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline H12/H15 selector/refit over v19 opened boundary rows; no MPC simulation, validation64, sealed test, or gradient training. rows={h['rows']}; positives={h['positive_rows']}; catastrophics={h['catastrophic_rows']}; configs={h['config_count']}; selector_refit_evaluations={h['selector_refit_evaluations']}; strict_bank_save={h['strict_bank_save']:.6f}, bad={h['strict_bank_bad']}, H12={h['strict_bank_h12']}, pass5={h['strict_bank_pass5']}; strict_source_save={h['strict_source_save']:.6f}, bad={h['strict_source_bad']}, H12={h['strict_source_h12']}, pass5={h['strict_source_pass5']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-h12-selector-v20", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_h12_selector_v20:
        raise ContractError("requires --run and --i-accept-development-h12-selector-v20")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = {
        "requested_utc": raw["created_utc"],
        "reason": "backup after v20 H12/H15 selector/refit before any more unique diagnostics or simulations",
        "required_before_more_unique_science": True,
        "artifacts": [
            rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQUEST),
            "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv",
        ],
        "development_mpc_simulation_episodes": 0,
        "development_control_steps": 0,
        "selector_refit_evaluations": raw["budget_actual"]["selector_refit_evaluations"],
        "offline_model_evaluations": raw["budget_actual"]["offline_model_evaluations"],
        "training_episodes": 0,
        "gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(BACKUP_REQUEST, req)
    done = {
        "passed": True,
        "status": "complete",
        "marker": MARKER,
        "classification": raw["classification"],
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQUEST),
        "headline": raw["headline"],
        "decision": raw["decision"],
        "budget_actual": raw["budget_actual"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {
            rel(SOURCE): sha256(SOURCE),
            rel(PROTOCOL): sha256(PROTOCOL),
            rel(OUT / "raw.json"): sha256(OUT / "raw.json"),
            rel(OUT / "summary.md"): sha256(OUT / "summary.md"),
            rel(BACKUP_REQUEST): sha256(BACKUP_REQUEST),
        },
    }
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    state = {
        "utc": raw["created_utc"],
        "headline": raw["headline"],
        "decision": raw["decision"],
        "budget_actual": raw["budget_actual"],
        "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST)},
        "backup_status": "not verified after v20; supervisor backup required before more unique science",
        "next_action": "After verified backup, if both strict splits passed, freeze fresh development online-overhead/source-independent H12-negative confirmation; otherwise inspect failing outer groups and acquire H12 negative/support data or richer terminal-risk features. No validation64 or sealed test.",
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v20 H12/H15 selector/refit\n\n" + json.dumps(clean(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
