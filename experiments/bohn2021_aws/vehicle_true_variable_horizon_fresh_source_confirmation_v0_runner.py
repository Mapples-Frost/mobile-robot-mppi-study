#!/usr/bin/env python3
"""Run the frozen vehicle true-variable-H fresh-source confirmation v0.

Development-only IMPROVED diagnostic.  It executes the protocol frozen by
vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0.py after the
supervisor-reported external backup covering the freeze artifacts.

Design:
  * Stage A: run one true-H15 trace from each frozen fresh source case.  Select
    two branch states from H15 traces only, before any H10 outcome is observed;
    write a selected_state_manifest before Stage B begins.
  * Stage B: run the frozen blocked true-H10/H15 branch schedule under matched
    and shared-H15 terminal profiles, with two repeats.
  * Analysis: evaluate fresh oracle opportunity and a conservative source-trained
    online-observable H10-abstention guard reconstructed from existing oracle and
    risk-anchor development labels.  This is development evidence only, not
    validation, not final test, and not ORIGINAL SAC.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_case5_smoke_v0_runner as v0  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_v0"
STAMP = "20260929T1025Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json"
FREEZE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_20260929T1015Z/completed.json"
TRANSFER_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_domain_profile_transfer_audit_v0_20260929T1000Z/completed.json"
ORACLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
RISK_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json"
RISK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json"

RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
CONTINUE_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T1025_after_fresh_source_confirmation_run.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER_RUN = f"vehicle-true-variable-H-fresh-source-confirmation-v0-run-{STAMP}"

TRUE_HORIZONS = [10, 15]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
MAX_STEPS = 150
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
MIN_DECISION_SAVING = 0.05
STRONG_DECISION_SAVING = 0.10


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: " + rel(path))
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "historical_validation64_bank_opened"):
        if key in obj and obj.get(key) is not False:
            raise ContractError(f"{key} flag must be false in {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError("hash mismatch from %s for %s" % (rel(path), name))
    return obj


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}

    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * q
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)

    return {"n": len(vals), "min": vals[0], "median": pct(0.5), "mean": float(math.fsum(vals) / len(vals)), "p95": pct(0.95), "max": vals[-1], "sum": float(math.fsum(vals))}


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def metric_sum(e: Mapping[str, Any], field: str) -> float:
    return sf((e.get(field) or {}).get("sum"), 0.0)


def is_safe_episode(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 999) == 0 and si(e.get("initial_failed_steps"), 999) == 0 and si(e.get("final_failed_steps"), 999) == 0


def verify_inputs() -> Dict[str, Any]:
    for p in (PROTOCOL, FREEZE_DONE, TRANSFER_DONE, ORACLE_RAW, ORACLE_DONE, RISK_RAW, RISK_DONE):
        if not p.exists():
            raise ContractError("required input missing: " + rel(p))
    freeze_done = completed_ok(FREEZE_DONE, check_hashes=False)
    completed_ok(TRANSFER_DONE, check_hashes=False)
    completed_ok(ORACLE_DONE, check_hashes=False)
    completed_ok(RISK_DONE, check_hashes=False)
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != "vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z":
        raise ContractError("unexpected fresh-source protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("protocol access flags invalid")
    budget = protocol.get("budget_declared") or {}
    if int(budget.get("total_episodes_exact", -1)) != 136:
        raise ContractError("unexpected frozen episode budget")
    selected = ((protocol.get("case_source") or {}).get("selected_fresh_cases") or [])
    if len(selected) != 8:
        raise ContractError("expected exactly 8 fresh cases")
    return {"protocol": protocol, "freeze_done": freeze_done}


def scalar_from_obj(obj: Any, names: Sequence[str]) -> float:
    if isinstance(obj, Mapping):
        for n in names:
            if n in obj:
                return sf(obj.get(n), 0.0)
        # common vehicle control input schemas
        for n in ("omega", "w", "u_omega", "steer", "v", "velocity"):
            if n in obj and n in names:
                return sf(obj.get(n), 0.0)
    if isinstance(obj, list) and obj:
        idx = 1 if len(obj) > 1 else 0
        return sf(obj[idx], 0.0)
    return 0.0


def trace_risk_score(row: Mapping[str, Any], total_steps: int) -> float:
    obs = row.get("observation") or []
    state = row.get("previous_state") or row.get("state") or {}
    attempts = (row.get("recovery") or {}).get("attempts") or []
    iters = 0.0
    for a in attempts:
        iters = max(iters, sf(a.get("iterations"), 0.0))
    step = si(row.get("step"), 0)
    perf = max(sf(row.get("performance"), 0.0), 0.0)
    theta_obs = abs(sf(obs[2], 0.0)) if isinstance(obs, list) and len(obs) > 2 else abs(sf(state.get("theta"), 0.0))
    u_omega = abs(scalar_from_obj(row.get("input"), ["omega", "u_omega", "w", "steer"]))
    constraint = max(sf(row.get("constraint"), 0.0), 0.0)
    y = abs(sf(state.get("y"), 0.0)) if isinstance(state, Mapping) else 0.0
    theta = abs(sf(state.get("theta"), 0.0)) if isinstance(state, Mapping) else 0.0
    phase = step / float(max(total_steps - 1, 1))
    return float(8.0 * math.sqrt(perf) + 2.0 * theta_obs + 0.6 * u_omega + 0.04 * iters + 1.0 * min(10.0, constraint) + 0.6 * y + 0.5 * theta + 1.2 * phase)


def select_stage_a_states(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    stage_cfg = protocol.get("stage_A_h15_trace_state_selection") or {}
    windows = stage_cfg.get("windows") or [
        {"slot": 0, "name": "early_mid", "low_fraction": 0.12, "high_fraction": 0.36},
        {"slot": 1, "name": "mid_late", "low_fraction": 0.38, "high_fraction": 0.75},
    ]
    bounds = stage_cfg.get("eligible_step_bounds") or {}
    min_step = si(bounds.get("min_step"), 8)
    max_step_abs = si(bounds.get("max_step"), 90)
    reserve = si(bounds.get("reserve_terminal_margin_steps"), 3)
    selected: List[Dict[str, Any]] = []
    for ep in stage_a_episodes:
        trace_path = ROOT / str(ep.get("path")) / "trace.json"
        trace = read_json(trace_path)
        n = len(trace)
        if n < min_step + reserve + 2:
            raise ContractError("Stage-A trace too short for state selection: " + str(ep.get("state_id")))
        chosen_steps: List[int] = []
        for window in windows:
            slot = si(window.get("slot"), len(chosen_steps))
            lo = max(min_step, int(math.floor(n * sf(window.get("low_fraction"), 0.1))))
            hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * sf(window.get("high_fraction"), 0.8))))
            candidates = [r for r in trace if lo <= si(r.get("step"), -1) <= hi]
            if not candidates:
                candidates = [r for r in trace if min_step <= si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
            if slot == 1 and chosen_steps:
                separated = [r for r in candidates if all(abs(si(r.get("step"), -1) - s) >= 8 for s in chosen_steps)]
                if separated:
                    candidates = separated
            if not candidates:
                raise ContractError("no eligible Stage-A state for " + str(ep.get("state_id")))
            row = max(candidates, key=lambda r: trace_risk_score(r, n))
            step = si(row.get("step"), -1)
            chosen_steps.append(step)
            base_state_id = f"fresh_case{si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{window.get('name', 'window')}"
            obs = row.get("observation") or []
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": si(ep.get("fresh_case_index"), -1),
                "source_candidate_index": si(ep.get("source_candidate_index"), -1),
                "fresh_confirmation_group": ep.get("fresh_confirmation_group"),
                "branch_state_slot": slot,
                "window": window.get("name"),
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(row.get("previous_state") or row.get("state")),
                "initial_observation_from_h15_trace": copy.deepcopy(obs),
                "h15_trace_episode_path": ep.get("path"),
                "stage_a_trace_risk_score": trace_risk_score(row, n),
                "selection_rule": "selected from H15 trace before any Stage-B H10/H15 branch outcome was run",
            })
    if len(selected) != 16:
        raise ContractError("expected 16 selected branch states, got %d" % len(selected))
    return selected


def feature_from_observation_state(obs: Any, state: Any) -> List[float]:
    vals: List[float] = []
    if isinstance(obs, list) and obs:
        vals.extend(sf(x, 0.0) for x in obs[:14])
    if not vals and isinstance(state, Mapping):
        vals.extend([sf(state.get("x"), 0.0) / 30.0, sf(state.get("y"), 0.0) / 30.0, sf(state.get("theta"), 0.0) / math.pi])
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def feature_from_episode_summary(e: Mapping[str, Any]) -> List[float]:
    br = e.get("branch_reset") or {}
    return feature_from_observation_state(br.get("initial_observation_at_branch"), br.get("branch_state_target"))


def feature_from_manifest_state(s: Mapping[str, Any]) -> List[float]:
    return feature_from_observation_state(s.get("initial_observation_from_h15_trace"), s.get("branch_previous_state"))


def transform_feature(x: Sequence[float], mode: str) -> np.ndarray:
    arr = np.asarray([sf(v, 0.0) for v in x], dtype=float)
    if mode == "raw_abs_l2":
        arr = np.concatenate([arr, np.abs(arr)])
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
    return arr


def source_training_examples() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"sources": [], "excluded_no_terminal_agreement": 0, "label_counts": {}}
    for source_name, raw_path in (("oracle_bank", ORACLE_RAW), ("risk_anchor", RISK_RAW)):
        raw = read_json(raw_path)
        rows = (((raw.get("analysis") or {}).get("state_profile_rows")) or [])
        episodes = raw.get("episodes") or []
        labels_by_sid: Dict[str, Dict[str, Optional[int]]] = {}
        for r in rows:
            sid = str(r.get("state_id"))
            profile = str(r.get("terminal_profile"))
            label = r.get("oracle_label")
            labels_by_sid.setdefault(sid, {})[profile] = None if label is None else int(label)
        ep_by_sid_profile_h: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
        for e in episodes:
            sid = str(e.get("state_id"))
            profile = str(e.get("terminal_profile") or e.get("terminal_mode"))
            h = si(e.get("true_mpc_n_horizon"), -1)
            if h == 15:
                ep_by_sid_profile_h.setdefault((sid, profile, h), e)
        kept = 0
        for sid, profs in sorted(labels_by_sid.items()):
            values = {p: lab for p, lab in profs.items() if lab is not None}
            if set(TERMINAL_PROFILES).issubset(values.keys()) and len(set(values[p] for p in TERMINAL_PROFILES)) == 1:
                label = int(values[TERMINAL_PROFILES[0]])
                ep = ep_by_sid_profile_h.get((sid, PRIMARY_TERMINAL_PROFILE, 15)) or ep_by_sid_profile_h.get((sid, "matched_terminal", 15))
                if ep is None:
                    diagnostics["excluded_no_h15_episode"] = diagnostics.get("excluded_no_h15_episode", 0) + 1
                    continue
                examples.append({
                    "source": source_name,
                    "state_id": sid,
                    "label": label,
                    "binary_h10": int(label == 10),
                    "feature": feature_from_episode_summary(ep),
                    "episode_path": ep.get("path"),
                })
                kept += 1
                diagnostics["label_counts"][str(label)] = diagnostics["label_counts"].get(str(label), 0) + 1
            else:
                diagnostics["excluded_no_terminal_agreement"] += 1
        diagnostics["sources"].append({"source": source_name, "raw": rel(raw_path), "states_seen": len(labels_by_sid), "states_kept_terminal_agreement": kept})
    return examples, diagnostics


def fit_positive_support_guard(examples: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(config.get("mode", "raw"))
    pos = [transform_feature(e["feature"], mode) for e in examples if int(e.get("binary_h10", 0)) == 1]
    neg = [transform_feature(e["feature"], mode) for e in examples if int(e.get("binary_h10", 0)) != 1]
    min_support = si(config.get("min_positive_support"), 2)
    q = sf(config.get("positive_radius_quantile"), 0.5)
    margin = sf(config.get("negative_margin"), 1.25)
    if len(pos) >= 2:
        nn: List[float] = []
        for i, p in enumerate(pos):
            ds = [float(np.linalg.norm(p - qv)) for j, qv in enumerate(pos) if j != i]
            if ds:
                nn.append(min(ds))
        radius = float(np.quantile(np.asarray(nn, dtype=float), min(max(q, 0.0), 1.0))) if nn else 0.0
        if radius <= 1e-12:
            all_ds = [float(np.linalg.norm(a - b)) for i, a in enumerate(pos) for j, b in enumerate(pos) if i < j]
            radius = max(all_ds) if all_ds else 1e-6
        radius = max(radius * margin, 1e-6)
    else:
        radius = 0.0
    return {"mode": mode, "config": dict(config), "positive_features": [p.tolist() for p in pos], "negative_features": [n.tolist() for n in neg], "positive_radius": radius, "min_positive_support": min_support, "negative_margin": margin, "positive_count": len(pos), "negative_count": len(neg)}


def predict_guard(model: Mapping[str, Any], feature: Sequence[float]) -> Tuple[int, Dict[str, Any]]:
    x = transform_feature(feature, str(model.get("mode", "raw")))
    pos = [np.asarray(v, dtype=float) for v in model.get("positive_features") or []]
    neg = [np.asarray(v, dtype=float) for v in model.get("negative_features") or []]
    if len(pos) < si(model.get("min_positive_support"), 2):
        return 15, {"reason": "insufficient_positive_support", "positive_count": len(pos), "negative_count": len(neg)}
    dpos = sorted(float(np.linalg.norm(x - p)) for p in pos)
    dneg = sorted(float(np.linalg.norm(x - n)) for n in neg)
    radius = sf(model.get("positive_radius"), 0.0)
    support = sum(1 for d in dpos if d <= radius)
    nearest_pos = dpos[0] if dpos else float("inf")
    nearest_neg = dneg[0] if dneg else float("inf")
    nearest_positive_wins = nearest_pos < nearest_neg
    negative_margin_ok = nearest_neg >= sf(model.get("negative_margin"), 1.25) * max(nearest_pos, 1e-12)
    support_ok = support >= si(model.get("min_positive_support"), 2)
    choose_h10 = bool(nearest_positive_wins and negative_margin_ok and support_ok)
    return (10 if choose_h10 else 15), {"nearest_positive_distance": nearest_pos, "nearest_negative_distance": nearest_neg, "positive_radius": radius, "positive_support_within_radius": support, "nearest_positive_wins": nearest_positive_wins, "negative_margin_ok": negative_margin_ok, "support_ok": support_ok, "reason": "h10_guard_pass" if choose_h10 else "abstain_to_h15"}


def group_key(e: Mapping[str, Any]) -> Tuple[str, str]:
    return str(e.get("base_state_id") or e.get("state_id")), str(e.get("terminal_profile") or e.get("terminal_mode"))


def analyze_branch_episodes(episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_by_base = {str(s["base_state_id"]): s for s in selected_states}
    source_examples, source_diag = source_training_examples()
    policy_cfgs = (((protocol.get("source_trained_selector_confirmation") or {}).get("policies_frozen_before_fresh_outcomes")) or [])
    primary_cfg = None
    for p in policy_cfgs:
        if p.get("primary_gate") is True:
            primary_cfg = p
            break
    if primary_cfg is None:
        primary_cfg = {"policy_id": "fallback_primary_raw_guard", "config": {"mode": "raw", "min_positive_support": 2, "positive_radius_quantile": 0.5, "negative_margin": 1.25}}
    primary_model = fit_positive_support_guard(source_examples, primary_cfg.get("config") or {})

    rows: List[Dict[str, Any]] = []
    unsafe_groups: List[Dict[str, Any]] = []
    groups = sorted(set(group_key(e) for e in episodes))
    for base_sid, profile in groups:
        med_by_h: Dict[int, Dict[str, Any]] = {}
        for h in TRUE_HORIZONS:
            reps = [e for e in episodes if group_key(e) == (base_sid, profile) and si(e.get("true_mpc_n_horizon"), -1) == h]
            safe_all = bool(reps) and all(is_safe_episode(e) for e in reps)
            med_by_h[h] = {
                "n": len(reps),
                "safe_all": safe_all,
                "physical": median([sf(e.get("physical_constraint_cost"), 0.0) for e in reps]),
                "total_cost": median([sf(e.get("total_cost"), 0.0) for e in reps]),
                "decision_sum_s": median([metric_sum(e, "decision_timing_s") for e in reps]),
                "solver_sum_s": median([metric_sum(e, "solver_attempt_timing_s") for e in reps]),
                "steps": median([si(e.get("steps"), 0) for e in reps]),
                "success_all": bool(reps) and all(bool(e.get("success")) for e in reps),
                "constraint_any": any(bool(e.get("constraint")) for e in reps),
                "solver_failure_steps_sum": int(sum(si(e.get("solver_failure_steps"), 0) for e in reps)),
                "opt_x_size": sorted(set(si(x) for e in reps for x in (e.get("opt_x_sizes_observed") or []))),
            }
        h10 = med_by_h[10]; h15 = med_by_h[15]
        if not h15["safe_all"]:
            unsafe_groups.append({"base_state_id": base_sid, "terminal_profile": profile, "reason": "H15 baseline not safe", "median_by_h": med_by_h})
        best_phys = None
        near_best_horizons: List[int] = []
        oracle_label: Optional[int] = None
        safe_hs = [h for h, m in med_by_h.items() if m["safe_all"] and m["physical"] is not None]
        if safe_hs:
            best_phys = min(float(med_by_h[h]["physical"]) for h in safe_hs)
            tol = max(2.0, 0.05 * abs(best_phys))
            near_best_horizons = [h for h in safe_hs if float(med_by_h[h]["physical"]) <= best_phys + tol]
            oracle_label = sorted(near_best_horizons, key=lambda hh: (float(med_by_h[hh]["decision_sum_s"]), float(med_by_h[hh]["solver_sum_s"]), hh))[0]
        else:
            tol = None
        h10_beneficial_vs_h15 = bool(h15["safe_all"] and h10["safe_all"] and h15["physical"] is not None and h10["physical"] is not None and float(h10["physical"]) - float(h15["physical"]) <= max(2.0, 0.05 * abs(float(h15["physical"]))) and h15["decision_sum_s"] and h10["decision_sum_s"] and float(h10["decision_sum_s"]) < float(h15["decision_sum_s"]))
        feature = feature_from_manifest_state(manifest_by_base[base_sid])
        pred_h, pred_diag = predict_guard(primary_model, feature)
        rows.append({
            "base_state_id": base_sid,
            "fresh_case_index": manifest_by_base[base_sid].get("fresh_case_index"),
            "source_candidate_index": manifest_by_base[base_sid].get("source_candidate_index"),
            "fresh_confirmation_group": manifest_by_base[base_sid].get("fresh_confirmation_group"),
            "branch_state_slot": manifest_by_base[base_sid].get("branch_state_slot"),
            "terminal_profile": profile,
            "median_by_h": {str(h): med_by_h[h] for h in TRUE_HORIZONS},
            "best_physical": best_phys,
            "near_best_tolerance": tol,
            "near_best_horizons": near_best_horizons,
            "oracle_label": oracle_label,
            "h10_beneficial_vs_h15": h10_beneficial_vs_h15,
            "primary_policy_prediction": pred_h,
            "primary_policy_diagnostics": pred_diag,
        })

    def aggregate_for(rows_in: Sequence[Mapping[str, Any]], chooser: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"groups": len(rows_in)}
        for h in TRUE_HORIZONS:
            out[f"fixed_H{h}"] = {
                "physical_sum": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("physical"), 0.0) for r in rows_in)),
                "decision_sum_s": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("decision_sum_s"), 0.0) for r in rows_in)),
                "solver_sum_s": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("solver_sum_s"), 0.0) for r in rows_in)),
            }
        phys: List[float] = []
        dec: List[float] = []
        solver: List[float] = []
        chosen_counts: Dict[str, int] = {}
        for r in rows_in:
            if chooser == "oracle":
                h = r.get("oracle_label")
            elif chooser == "primary_policy":
                h = r.get("primary_policy_prediction")
            else:
                raise ValueError(chooser)
            if h is None:
                continue
            h = int(h)
            chosen_counts[str(h)] = chosen_counts.get(str(h), 0) + 1
            m = r["median_by_h"][str(h)]
            phys.append(sf(m.get("physical"), 0.0)); dec.append(sf(m.get("decision_sum_s"), 0.0)); solver.append(sf(m.get("solver_sum_s"), 0.0))
        out[chooser] = {"groups_used": len(phys), "chosen_counts": chosen_counts, "physical_sum": float(math.fsum(phys)), "decision_sum_s": float(math.fsum(dec)), "solver_sum_s": float(math.fsum(solver))}
        return out

    primary_rows = [r for r in rows if r.get("terminal_profile") == PRIMARY_TERMINAL_PROFILE]
    matched_rows = [r for r in rows if r.get("terminal_profile") == "matched_terminal"]
    ag_primary_policy = aggregate_for(primary_rows, "primary_policy")
    ag_primary_oracle = aggregate_for(primary_rows, "oracle")
    ag_matched_policy = aggregate_for(matched_rows, "primary_policy")
    ag_matched_oracle = aggregate_for(matched_rows, "oracle")

    def comp(ag: Mapping[str, Any], chooser: str, base_h: int = 15) -> Dict[str, Any]:
        c = ag[chooser]; b = ag[f"fixed_H{base_h}"]
        return {
            "physical_delta_vs_fixed_H15": float(c["physical_sum"] - b["physical_sum"]),
            "decision_relative_saving_vs_fixed_H15": None if b["decision_sum_s"] <= 0 else float((b["decision_sum_s"] - c["decision_sum_s"]) / b["decision_sum_s"]),
            "solver_relative_saving_vs_fixed_H15": None if b["solver_sum_s"] <= 0 else float((b["solver_sum_s"] - c["solver_sum_s"]) / b["solver_sum_s"]),
        }

    primary_policy_comp = comp(ag_primary_policy, "primary_policy")
    primary_oracle_comp = comp(ag_primary_oracle, "oracle")
    matched_policy_comp = comp(ag_matched_policy, "primary_policy")
    matched_oracle_comp = comp(ag_matched_oracle, "oracle")

    false_positive_rows: List[Dict[str, Any]] = []
    confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in primary_rows:
        pred_pos = int(r.get("primary_policy_prediction")) == 10
        true_pos = bool(r.get("h10_beneficial_vs_h15"))
        if pred_pos and true_pos:
            confusion["TP"] += 1
        elif pred_pos and not true_pos:
            confusion["FP"] += 1
            h10 = r["median_by_h"]["10"]; h15 = r["median_by_h"]["15"]
            false_positive_rows.append({"base_state_id": r["base_state_id"], "fresh_confirmation_group": r.get("fresh_confirmation_group"), "h10_safe_all": h10.get("safe_all"), "h15_safe_all": h15.get("safe_all"), "physical_delta_h10_minus_h15": None if h10.get("physical") is None or h15.get("physical") is None else float(h10["physical"] - h15["physical"]), "decision_delta_h10_minus_h15": None if h10.get("decision_sum_s") is None or h15.get("decision_sum_s") is None else float(h10["decision_sum_s"] - h15["decision_sum_s"])})
        elif (not pred_pos) and true_pos:
            confusion["FN"] += 1
        else:
            confusion["TN"] += 1

    n_primary = len(primary_rows)
    fixed15_phys = ag_primary_policy["fixed_H15"]["physical_sum"]
    physical_tol = max(2.0 * n_primary, 0.05 * abs(fixed15_phys)) if n_primary else 0.0
    catastrophic_fp = [x for x in false_positive_rows if x.get("h15_safe_all") and ((not x.get("h10_safe_all")) or (x.get("physical_delta_h10_minus_h15") is not None and float(x["physical_delta_h10_minus_h15"]) > max(2.0, 0.25 * abs(sf(next(r for r in primary_rows if r["base_state_id"] == x["base_state_id"])["median_by_h"]["15"].get("physical"), 0.0)))))]
    h10_predictions = sum(1 for r in primary_rows if int(r.get("primary_policy_prediction")) == 10)
    decision_saving = primary_policy_comp.get("decision_relative_saving_vs_fixed_H15")
    solver_saving = primary_policy_comp.get("solver_relative_saving_vs_fixed_H15")
    primary_pass = bool(
        n_primary > 0
        and h10_predictions > 0
        and not catastrophic_fp
        and primary_policy_comp["physical_delta_vs_fixed_H15"] <= physical_tol
        and decision_saving is not None
        and float(decision_saving) >= MIN_DECISION_SAVING
    )
    primary_strong = bool(primary_pass and decision_saving is not None and float(decision_saving) >= STRONG_DECISION_SAVING)

    disagreements: List[Dict[str, Any]] = []
    for sid in sorted({r["base_state_id"] for r in rows}):
        prof_labels = {r["terminal_profile"]: r.get("oracle_label") for r in rows if r["base_state_id"] == sid}
        if len(set(v for v in prof_labels.values() if v is not None)) > 1:
            disagreements.append({"base_state_id": sid, "oracle_labels_by_profile": prof_labels, "fresh_confirmation_group": manifest_by_base[sid].get("fresh_confirmation_group")})

    if primary_pass:
        decision = "fresh source confirmation supports a terminal-profile-aware source-trained selector signal; next run a tiny closed-loop selector-overhead smoke with the true-H controller cache before any independent validation"
    elif h10_predictions == 0:
        decision = "fresh source guard abstained to all-H15; block selector rollout and prioritize representation/value/terminal calibration or scenario redesign rather than another unchanged CV sweep"
    elif catastrophic_fp or primary_policy_comp["physical_delta_vs_fixed_H15"] > physical_tol:
        decision = "fresh source H10 false positives/physical regression block selector rollout; prioritize terminal-value/objective/representation calibration before validation"
    else:
        decision = "fresh source result did not meet compute-saving gate; inspect measured timing noise and fixed-H tradeoff before selector rollout"

    return {
        "source_training_examples": source_diag,
        "primary_policy": {"protocol_policy_id": primary_cfg.get("policy_id"), "model": {k: v for k, v in primary_model.items() if k not in ("positive_features", "negative_features")}},
        "state_profile_rows": rows,
        "unsafe_groups": unsafe_groups,
        "terminal_profile_disagreements": disagreements,
        "aggregates": {
            "shared_h15_terminal_primary_policy": ag_primary_policy,
            "shared_h15_terminal_oracle": ag_primary_oracle,
            "matched_terminal_primary_policy": ag_matched_policy,
            "matched_terminal_oracle": ag_matched_oracle,
        },
        "comparisons": {
            "shared_h15_terminal_primary_policy_vs_fixed_H15": primary_policy_comp,
            "shared_h15_terminal_oracle_vs_fixed_H15": primary_oracle_comp,
            "matched_terminal_primary_policy_vs_fixed_H15": matched_policy_comp,
            "matched_terminal_oracle_vs_fixed_H15": matched_oracle_comp,
        },
        "primary_gate": {
            "scope": PRIMARY_TERMINAL_PROFILE,
            "groups": n_primary,
            "h10_predictions": h10_predictions,
            "confusion_vs_fresh_h10_beneficial_label": confusion,
            "false_positive_rows": false_positive_rows,
            "catastrophic_false_positive_rows": catastrophic_fp,
            "physical_tolerance_vs_fixed_H15": physical_tol,
            "physical_gate": bool(primary_policy_comp["physical_delta_vs_fixed_H15"] <= physical_tol),
            "decision_saving_gate_5pct": bool(decision_saving is not None and float(decision_saving) >= MIN_DECISION_SAVING),
            "decision_saving_gate_10pct_strong": bool(decision_saving is not None and float(decision_saving) >= STRONG_DECISION_SAVING),
            "primary_pass_5pct": primary_pass,
            "primary_strong_10pct": primary_strong,
            "solver_saving_reported_not_primary": solver_saving,
            "train_or_refit_now": False,
        },
        "decision": decision,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    g = a["primary_gate"]
    lines = [
        "# Vehicle true-variable-H fresh-source confirmation v0 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only fresh-source confirmation; no validation64 bank, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Primary shared-H15-terminal source-trained guard gate",
        "",
        f"- Groups: `{g['groups']}`; H10 predictions: `{g['h10_predictions']}`.",
        f"- Confusion vs fresh H10-beneficial labels: `{g['confusion_vs_fresh_h10_beneficial_label']}`.",
        f"- Physical gate: `{g['physical_gate']}`; delta/tolerance `{a['comparisons']['shared_h15_terminal_primary_policy_vs_fixed_H15']['physical_delta_vs_fixed_H15']}` / `{g['physical_tolerance_vs_fixed_H15']}`.",
        f"- Decision saving gate >=5%: `{g['decision_saving_gate_5pct']}`; comparison `{a['comparisons']['shared_h15_terminal_primary_policy_vs_fixed_H15']}`.",
        f"- Strong >=10% gate: `{g['decision_saving_gate_10pct_strong']}`.",
        f"- Catastrophic false positives: `{len(g['catastrophic_false_positive_rows'])}`.",
        f"- Primary pass: `{g['primary_pass_5pct']}`; strong pass: `{g['primary_strong_10pct']}`.",
        "",
        "## Oracle opportunity on fresh states",
        "",
        f"- Shared-H15 terminal oracle vs fixed H15: `{a['comparisons']['shared_h15_terminal_oracle_vs_fixed_H15']}`; chosen counts `{a['aggregates']['shared_h15_terminal_oracle']['oracle']['chosen_counts']}`.",
        f"- Matched terminal oracle vs fixed H15: `{a['comparisons']['matched_terminal_oracle_vs_fixed_H15']}`; chosen counts `{a['aggregates']['matched_terminal_oracle']['oracle']['chosen_counts']}`.",
        f"- Terminal-profile oracle disagreements: `{len(a['terminal_profile_disagreements'])}` / 16 states.",
        "",
        "## Per-state shared-H15 rows",
        "",
        "| state | group | pred | oracle | H10 beneficial | H10 phys | H15 phys | H10 dec | H15 dec |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in a["state_profile_rows"]:
        if r["terminal_profile"] != PRIMARY_TERMINAL_PROFILE:
            continue
        m10 = r["median_by_h"]["10"]; m15 = r["median_by_h"]["15"]
        def fmt(v: Any) -> str:
            return "NA" if v is None else "%.6g" % float(v)
        lines.append("| `%s` | `%s` | %s | %s | `%s` | %s | %s | %s | %s |" % (r["base_state_id"], r.get("fresh_confirmation_group"), r.get("primary_policy_prediction"), r.get("oracle_label"), bool(r.get("h10_beneficial_vs_h15")), fmt(m10.get("physical")), fmt(m15.get("physical")), fmt(m10.get("decision_sum_s")), fmt(m15.get("decision_sum_s"))))
    lines += [
        "",
        "## Decision",
        "",
        str(a["decision"]),
        "",
        "Interpretation limits: development-only; source-trained guard is a protocol-compatible reconstruction from existing oracle/risk labels; timing claims use measured whole-decision and solver timings, not nominal H. Final validation/test remain unopened.",
        "",
        f"Backup request after run: `{raw['backup_request_after_run']}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER_RUN not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if NAME not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},development_fresh_source_confirmation,development_no_validation_no_test,136,0,0,0,0,False,{rel(RUN_DIR / 'completed.json')}\n", encoding="utf-8")


def terminal_for(profile: str, h: int, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    if profile == "matched_terminal":
        return terminals[h]
    if profile == "shared_h15_terminal":
        return terminals[15]
    raise ContractError("unknown terminal profile " + profile)


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", type=str, required=True)
    ap.add_argument("--i-accept-development-fresh-source-confirmation-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_fresh_source_confirmation_v0:
        raise ContractError("requires --run and explicit fresh-source confirmation acknowledgement")
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    info = verify_inputs()
    protocol = info["protocol"]
    selected_cases = (protocol.get("case_source") or {}).get("selected_fresh_cases") or []
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v0.SMOKE_DIR = RUN_DIR

    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    for h in TRUE_HORIZONS:
        if h not in terminals:
            raise ContractError("terminal grid missing H%d" % h)

    started = now_utc()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    # Stage A: H15 trace scans only.  These run before any Stage-B H10 branch outcome.
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {
            "execution_index": i,
            "state_id": f"stageA_fresh_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
            "case": i,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": 0,
            "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
            "true_mpc_n_horizon": 15,
            "commanded_horizon": 15,
            "terminal_mode": "matched_terminal",
            "initialization": "fresh_source_stageA_trueH15_trace_from_initial_state",
        }
        summary = v0.run_true_h_episode(item, case, terminals[15])
        summary["stage"] = "A_h15_trace_scan"
        summary["fresh_case_index"] = i
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        stage_a_eps.append(summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "stage": "A", "episodes_done": len(episodes), "episodes_expected": 136, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("state_id", "true_mpc_n_horizon", "steps", "success", "termination")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    selected_states = select_stage_a_states(stage_a_eps, protocol)
    manifest = {"created_utc": now_utc().isoformat(), "stage_A_complete_before_any_stage_B": True, "stage_A_episode_count": len(stage_a_eps), "selected_states": selected_states, "sha256_protocol": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(RUN_DIR / "selected_state_manifest.json", manifest)
    write_json(RUN_DIR / "manifest_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "selected_state_count": len(selected_states), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    state_by_case_slot = {(si(s["fresh_case_index"], -1), si(s["branch_state_slot"], -1)): s for s in selected_states}
    branch_template = ((protocol.get("stage_B_branch_confirmation") or {}).get("branch_arm_template_after_state_selection") or [])
    if len(branch_template) != 128:
        raise ContractError("unexpected Stage-B template length")
    write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
    branch_episodes: List[Dict[str, Any]] = []
    for j, tmpl in enumerate(branch_template):
        fc = si(tmpl.get("fresh_case_index"), -1)
        slot = si(tmpl.get("branch_state_slot"), -1)
        st = state_by_case_slot.get((fc, slot))
        if st is None:
            raise ContractError("missing selected state for Stage-B template")
        case_meta = selected_cases[fc]
        case = case_meta["case_snapshot_from_candidate_pool"]
        h = si(tmpl.get("true_mpc_n_horizon"), -1)
        profile = str(tmpl.get("terminal_profile"))
        repeat = si(tmpl.get("repeat"), 0)
        unique_state_id = f"B{j:03d}_{st['base_state_id']}_{profile}_r{repeat}_H{h}"
        item = {
            "execution_index": 1000 + j,
            "state_id": unique_state_id,
            "case": fc,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": int(st["branch_step"]),
            "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
            "true_mpc_n_horizon": h,
            "commanded_horizon": h,
            "terminal_mode": profile,
            "initialization": "fresh_source_stageB_direct_branch_state_from_predeclared_H15_trace_manifest",
        }
        summary = v0.run_true_h_episode(item, case, terminal_for(profile, h, terminals))
        summary["stage"] = "B_blocked_branch_trueH10_H15"
        summary["template_execution_index"] = int(tmpl.get("execution_index", j))
        summary["repeat"] = repeat
        summary["fresh_case_index"] = fc
        summary["base_state_id"] = st["base_state_id"]
        summary["branch_state_slot"] = slot
        summary["terminal_profile"] = profile
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["terminal_source_horizon"] = 15 if profile == "shared_h15_terminal" else h
        summary["terminal_receipt_effective"] = terminal_receipts.get(str(summary["terminal_source_horizon"])) or terminal_receipts.get(summary["terminal_source_horizon"])
        branch_episodes.append(summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "stage": "B", "episodes_done": len(episodes), "episodes_expected": 136, "stage_B_done": len(branch_episodes), "stage_B_expected": 128, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("base_state_id", "repeat", "terminal_profile", "true_mpc_n_horizon", "steps", "success", "termination")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    declared = protocol.get("budget_declared") or {}
    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    if len(stage_a_eps) != int(declared.get("stage_a_h15_trace_episodes", -1)) or len(branch_episodes) != int(declared.get("stage_b_branch_episodes", -1)) or len(episodes) != int(declared.get("total_episodes_exact", -1)):
        raise ContractError("episode budget mismatch")
    if control_steps > int(declared.get("control_step_upper_bound", -1)):
        raise ContractError("control-step budget exceeded")

    analysis = analyze_branch_episodes(branch_episodes, selected_states, protocol)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_V0_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup fresh-source confirmation runner/source/raw outputs before selector-overhead smoke, value calibration or further simulation", "backup_required_before_more_simulations": True, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(RUN_DIR), rel(STATE_RUN), rel(CONTINUE_STATE), rel(PROTOCOL), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_H_fresh_source_confirmation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {"episodes": len(episodes), "stage_A_h15_trace_episodes": len(stage_a_eps), "stage_B_branch_episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(episodes), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "stage_A_episodes": stage_a_eps,
        "selected_state_manifest": manifest,
        "branch_episodes": branch_episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development-only", "fresh source but not validation/test", "source-trained guard is reconstructed from existing development labels", "not ORIGINAL SAC", "does not infer speed from H alone; measured decision/solver timing reported"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(f"""# Continue state after fresh-source confirmation v0 run

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.

Completed: `{NAME}` development run: {len(episodes)} episodes, {control_steps} control steps. No validation64, no sealed test, no training/refit.

Primary gate: {analysis['primary_gate']}
Decision: {analysis['decision']}
Summary: `{rel(RUN_DIR / 'summary.md')}`
Raw: `{rel(RUN_DIR / 'raw.json')}`
Completed: `{rel(RUN_DIR / 'completed.json')}`
Backup request: `{rel(req)}`

Next action: if primary pass, freeze a tiny closed-loop selector-overhead smoke; if failed, use the failure mode to choose terminal-value/objective/representation calibration or scenario-design intervention, not another unchanged label-density sweep. Backup this run before more simulations/training/refit.
""", encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation v0 run

UTC: {created.isoformat()}. Development-only fresh-source confirmation completed: {len(episodes)} episodes ({len(stage_a_eps)} H15 traces + {len(branch_episodes)} blocked H10/H15 branches), {control_steps} control steps. No validation64, no sealed test, no training/refit. Primary shared-H15 source-trained guard gate: {analysis['primary_gate']}. Decision: {analysis['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""")
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, FREEZE_DONE, TRANSFER_DONE, ORACLE_DONE, RISK_DONE, STATE_RUN, CONTINUE_STATE, req]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "classification": raw["classification"], "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": analysis["primary_gate"], "decision": analysis["decision"], "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": analysis["primary_gate"], "decision": analysis["decision"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except BaseException as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial outputs. Audit completed episode directories, selected_state_manifest, progress.json and hashes before any rerun; if source repair is needed, version it and re-backup before additional simulation."})
        raise
