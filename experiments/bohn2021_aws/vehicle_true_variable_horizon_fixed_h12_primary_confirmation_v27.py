#!/usr/bin/env python3
"""v27 repaired fixed-H12-primary development confirmation.

Concrete bounded diagnostic following v26b repair and v27 preflight.

This script executes the preserved v26/v26b fixed-H12-primary case selection:
12 source-independent stress-bank cases, one H15 Stage-A trace per case, two
pre-outcome H15-trace branch states per case, then true H12/H15 branch rollouts
with a shared H15 terminal.  Fixed true H12 is the primary comparator; the
pre-existing v20b/v22 H12/H15 selector and oracle H12/H15 are secondary
diagnostics.  No validation64 or sealed test is opened; no training/refit/grid
search is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_selector_refit_v20 as v20  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_online_overhead_v22 as v22  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21 as v21  # noqa:E402

NAME = "vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27"
STAMP = "20260930T0250Z"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0250_after_v27_fixed_h12_confirmation.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_FIXED_H12_CONFIRMATION_{STAMP}.json"
MARKER = f"vehicle-fixed-h12-primary-confirmation-v27-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V26_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26_preoutcome_fixed_H12_primary_confirmation_20260930T0225Z.json"
V26B_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_corrected_fixed_H12_primary_confirmation_amendment_20260930T0235Z.json"
V26B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_20260930T0235Z/completed.json"
V27_PREFLIGHT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_preflight_20260930T0240Z/completed.json"
DEFAULT_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T022626_from_supervisor_context_after_v26b_and_v27_source.json"

TRUE_HORIZONS = [12, 15]
SHORT_H = 12
REF_H = 15
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
TARGET_CASES = 12
BRANCH_STATES_PER_CASE = 2
REPEATS = 1
MAX_STEPS = 150
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS
CONTROL_STEP_CAP = TOTAL_EPISODES * MAX_STEPS
RNG_SEED = 202609300250
MIN_SAVE = 0.05

WINDOWS = [
    {"slot": 0, "name": "early_risk", "low_fraction": 0.12, "high_fraction": 0.34, "selection": "max_h15_trace_risk_score"},
    {"slot": 1, "name": "mid_late_risk", "low_fraction": 0.45, "high_fraction": 0.74, "selection": "max_h15_trace_risk_score_min_step_separation_8"},
]

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
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
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

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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

def pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "NA"

def q(xs: Sequence[float], quant: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = quant * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return math.fsum(vals) / len(vals) if vals else 0.0

def completed_ok(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    ok = obj.get("passed") is True or obj.get("hard_pass") is True or obj.get("status") in ("complete", "completed")
    if not ok:
        raise ContractError(f"prerequisite did not pass/complete: {label} {rel(path)}")
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(key) is True:
            raise ContractError(f"forbidden {key}=true in prerequisite {label}")
    return obj

def parse_time(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str):
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def verify_backup(path: Path, v26b_time: Optional[dt.datetime]) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("backup proof missing: " + rel(path))
    obj = read_json(path)
    if obj.get("status") != "verified" or obj.get("backup_verified") is not True:
        raise ContractError("backup proof not verified: " + rel(path))
    if obj.get("remaining_changed_files") not in (0, "0"):
        raise ContractError("backup proof reports remaining changed files: " + rel(path))
    if not obj.get("commit") or not obj.get("packages_this_run"):
        raise ContractError("backup proof lacks commit/package evidence: " + rel(path))
    if v26b_time is not None:
        t = parse_time(obj.get("time"))
        if t is None or t <= v26b_time:
            raise ContractError("backup proof does not postdate v26b repair")
    return obj

def obs14(raw: Any) -> List[float]:
    vals = [sf(v) for v in (raw or [])[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]

def branch_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("previous_state") or row.get("state") or {}

def load_protocol_inputs() -> Dict[str, Any]:
    v26b_done = completed_ok(V26B_DONE, "v26b repair")
    v26 = read_json(V26_PROTOCOL)
    v26b = read_json(V26B_PROTOCOL)
    if v26b.get("original_v26_protocol", {}).get("sha256") != sha256(V26_PROTOCOL):
        raise ContractError("v26 protocol sha mismatch against v26b repair amendment")
    selected_cases = list(v26.get("selected_cases") or [])
    selected_indices = [si(c.get("source_candidate_index"), -1) for c in selected_cases]
    if selected_indices != list(v26b.get("selected_source_candidate_indices_preserved_from_v26") or []):
        raise ContractError("selected case indices do not match v26b preserved selection")
    budget = v26b.get("budget_declared_for_future_confirmation") or {}
    if budget.get("total_episodes_exact") != TOTAL_EPISODES or budget.get("control_step_upper_bound") != CONTROL_STEP_CAP:
        raise ContractError("v26b future budget does not match v27 runner constants")
    fresh = (v26b.get("corrected_before_evidence") or {}).get("fresh_v21_v23_v25") or {}
    all_opened = (v26b.get("corrected_before_evidence") or {}).get("all_opened_v19_v21_v23_v25") or {}
    if fresh.get("rows") != 56 or fresh.get("h12_bad") != 0 or fresh.get("pass5") is not True:
        raise ContractError("v26b corrected fresh evidence sanity check failed")
    if all_opened.get("rows") != 80 or all_opened.get("h12_bad") != 3 or all_opened.get("pass5") is not False:
        raise ContractError("v26b corrected all-opened evidence sanity check failed")
    return {"v26": v26, "v26b": v26b, "v26b_done": v26b_done, "selected_cases": selected_cases}

def select_stage_a_states(stage_a_episodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    min_step = 8; max_step_abs = 90; reserve = 3
    for ep in stage_a_episodes:
        trace_path = ROOT / str(ep.get("path")) / "trace.json"
        trace = read_json(trace_path)
        n = len(trace)
        if n < min_step + reserve + 2:
            raise ContractError("Stage-A trace too short: " + str(ep.get("state_id")))
        chosen_steps: List[int] = []
        for window in WINDOWS:
            slot = si(window.get("slot"), len(chosen_steps))
            lo = max(min_step, int(math.floor(n * sf(window.get("low_fraction"), 0.12))))
            hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * sf(window.get("high_fraction"), 0.74))))
            candidates = [r for r in trace if lo <= si(r.get("step"), -1) <= hi]
            if slot == 1 and chosen_steps:
                separated = [r for r in candidates if all(abs(si(r.get("step"), -1) - s) >= 8 for s in chosen_steps)]
                if separated:
                    candidates = separated
            if not candidates:
                candidates = [r for r in trace if min_step <= si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
            if not candidates:
                raise ContractError("no eligible H15 trace state for " + str(ep.get("state_id")))
            row = max(candidates, key=lambda r: (v21.trace_risk_score(r, n), -abs(si(r.get("step"), -1) - int((lo + hi) / 2)), -si(r.get("step"), 9999)))
            step = si(row.get("step"), -1)
            chosen_steps.append(step)
            fc = si(ep.get("fresh_case_index"), -1)
            base_state_id = f"v27_case{fc:02d}_slot{slot}_{window.get('name','window')}"
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": fc,
                "source_candidate_index": si(ep.get("source_candidate_index"), -1),
                "role": ep.get("role"),
                "branch_state_slot": slot,
                "window": window.get("name"),
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(branch_state(row)),
                "initial_observation_from_h15_trace": copy.deepcopy(row.get("observation") or []),
                "h15_trace_episode_path": ep.get("path"),
                "stage_a_trace_risk_score": v21.trace_risk_score(row, n),
                "selection_rule": "v27: selected from H15 trace by predeclared risk windows before selector choices and before any H12/H15 Stage-B outcome",
            })
    if len(selected) != len(stage_a_episodes) * BRANCH_STATES_PER_CASE:
        raise ContractError("unexpected selected state count")
    return selected

def selector_features(state: Mapping[str, Any], trace_cache: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, float]:
    cand = {
        "candidate_id": str(state["base_state_id"]),
        "base_state_id": str(state["base_state_id"]),
        "branch_previous_state": state.get("branch_previous_state") or {},
        "candidate_branch_step": si(state.get("branch_step"), -1),
        "offset_from_center": 0,
        "source_candidate_index": si(state.get("source_candidate_index"), -1),
        "initial_observation_from_h15_trace": obs14(state.get("initial_observation_from_h15_trace") or []),
        "h15_trace_episode_path": str(state.get("h15_trace_episode_path") or ""),
    }
    fd = v20.static_features(cand)
    trace = trace_cache.get(str(cand["h15_trace_episode_path"])) or []
    v22.add_history_features_from_trace(fd, trace, si(cand.get("candidate_branch_step"), -1))
    return fd

def make_preoutcome_selector_choices(selected_states: Sequence[Mapping[str, Any]], input_hashes: Dict[str, str]) -> Dict[str, Any]:
    v19_rows, v19_features, v19_diag, v19_hashes = v20.load_rows()
    input_hashes.update(v19_hashes)
    model = v22.train_model(v19_rows, v19_features, v22.DEPLOY_CFG)
    trace_cache: Dict[str, Sequence[Mapping[str, Any]]] = {}
    for st in selected_states:
        ep_path = str(st.get("h15_trace_episode_path") or "")
        tp = ROOT / ep_path / "trace.json"
        if ep_path not in trace_cache:
            trace_cache[ep_path] = read_json(tp)
            input_hashes[rel(tp)] = sha256(tp)
    choices: List[Dict[str, Any]] = []
    for st in selected_states:
        t0 = time.perf_counter_ns()
        fd = selector_features(st, trace_cache)
        t1 = time.perf_counter_ns()
        h, score = v22.predict_one(fd, model)
        t2 = time.perf_counter_ns()
        choices.append({
            "base_state_id": st["base_state_id"],
            "source_candidate_index": st.get("source_candidate_index"),
            "role": st.get("role"),
            "selected_h_preoutcome": int(h),
            "feature_time_s": (t1 - t0) / 1e9,
            "selector_time_s": (t2 - t1) / 1e9,
            "feature_plus_selector_time_s": (t2 - t0) / 1e9,
            "feature_count": len(fd),
            "score": score,
        })
    times = [sf(c["feature_plus_selector_time_s"]) for c in choices]
    out = {
        "created_utc": now_utc().isoformat(),
        "stage_B_started": False,
        "no_v27_h12_outcome_seen_before_choices": True,
        "deployable_config": dict(v22.DEPLOY_CFG),
        "deployable_config_id": v20.cfg_id(v22.DEPLOY_CFG),
        "fit_source": "opened v19 rows via v20.load_rows; no grid search/refit/validation/test",
        "v19_diag": v19_diag,
        "choices": choices,
        "choice_counts": dict(Counter(str(c["selected_h_preoutcome"]) for c in choices)),
        "overhead_summary": {"n": len(times), "sum_s": math.fsum(times), "mean_s": mean(times), "median_s": q(times, 0.5), "p95_s": q(times, 0.95), "max_s": max(times) if times else 0.0},
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(RUN_DIR / "preoutcome_selector_choices.json", out)
    return out

def make_stage_b_template(selected_states: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for st in selected_states:
            hs = list(TRUE_HORIZONS)
            rng.shuffle(hs)
            for h in hs:
                rows.append({
                    "execution_index": exe,
                    "repeat": rep,
                    "fresh_case_index": si(st.get("fresh_case_index"), -1),
                    "source_candidate_index": si(st.get("source_candidate_index"), -1),
                    "base_state_id": st["base_state_id"],
                    "branch_state_slot": si(st.get("branch_state_slot"), -1),
                    "true_mpc_n_horizon": int(h),
                    "terminal_profile": PRIMARY_TERMINAL_PROFILE,
                    "blocked_randomization_unit": f"v27|rep{rep}|case{si(st.get('fresh_case_index'), -1)}|slot{si(st.get('branch_state_slot'), -1)}|shared_h15",
                })
                exe += 1
    return rows

def to_eval_rows(state_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in state_rows:
        out.append({
            "candidate_id": str(r["base_state_id"]),
            "bank_id": "v27_fixed_H12_primary",
            "source_key": f"v27/source{si(r.get('source_candidate_index'), -1):03d}/{r['base_state_id']}",
            "base_state_id": r["base_state_id"],
            "role": r.get("role") or r.get("risk_probe_role"),
            "h12_physical": sf((r.get("h12") or {}).get("physical")),
            "h15_physical": sf((r.get("h15") or {}).get("physical")),
            "h12_decision_sum_s": sf((r.get("h12") or {}).get("decision_sum_s")),
            "h15_decision_sum_s": sf((r.get("h15") or {}).get("decision_sum_s")),
            "h12_solver_sum_s": sf((r.get("h12") or {}).get("solver_sum_s")),
            "h15_solver_sum_s": sf((r.get("h15") or {}).get("solver_sum_s")),
            "h12_steps": sf((r.get("h12") or {}).get("steps"), 1.0),
            "h15_steps": sf((r.get("h15") or {}).get("steps"), 1.0),
            "phys_delta_h12_minus_h15": sf(r.get("physical_delta_h12_minus_h15")),
            "decision_gain_h12_vs_h15_s": sf(r.get("decision_gain_h12_vs_h15_s")),
            "solver_gain_h12_vs_h15_s": sf(r.get("solver_gain_h12_vs_h15_s")),
            "row_physical_tolerance": sf(r.get("row_tolerance_vs_H15"), 2.0),
            "h12_catastrophic_vs_h15": bool(r.get("h12_catastrophic_vs_h15")),
            "h12_beneficial_vs_h15": bool(r.get("h12_beneficial_vs_h15")),
        })
    return out

def fixed_choices(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {str(r["candidate_id"]): int(h) for r in rows}

def oracle_choices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {str(r["candidate_id"]): (SHORT_H if bool(r.get("h12_beneficial_vs_h15")) else REF_H) for r in rows}

def selector_choice_map(choices: Mapping[str, Any]) -> Dict[str, int]:
    return {str(d["base_state_id"]): int(d["selected_h_preoutcome"]) for d in choices.get("choices", [])}

def analyze(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], choices: Mapping[str, Any]) -> Dict[str, Any]:
    label = v21.analyze(branch_episodes, selected_states)
    rows = to_eval_rows(label["state_rows"])
    overhead = sf((choices.get("overhead_summary") or {}).get("mean_s"))
    policies = {
        "fixed_H15": v22.evaluate_policy(rows, fixed_choices(rows, REF_H)),
        "fixed_H12": v22.evaluate_policy(rows, fixed_choices(rows, SHORT_H)),
        "oracle_H12_H15": v22.evaluate_policy(rows, oracle_choices(rows)),
        "selector_preoutcome_actual_overhead": v22.evaluate_policy(rows, selector_choice_map(choices), overhead_per_branch_call_s=overhead),
        "selector_preoutcome_no_overhead": v22.evaluate_policy(rows, selector_choice_map(choices), overhead_per_branch_call_s=0.0),
    }
    fx = policies["fixed_H12"]; sel = policies["selector_preoutcome_actual_overhead"]; oracle = policies["oracle_H12_H15"]
    if fx["pass5_zero_cat_physical"]:
        decision = "v27 fixed-H12-primary confirmation passes: fixed true H12 remains the stronger simple baseline on this fresh stress-bank batch; do not validate adaptive H12/H15 on this distribution without new scenario-opportunity evidence."
    elif fx["catastrophic_false_positive_count"] > 0 and sel["pass5_zero_cat_physical"]:
        decision = "v27 found fresh fixed-H12 failures while selector passes; adaptive mechanism has renewed value, inspect failure morphology and consider terminal-risk/value refit or bounded training before validation."
    elif fx["catastrophic_false_positive_count"] > 0 and oracle["pass5_zero_cat_physical"]:
        decision = "v27 found fixed-H12 failures with oracle opportunity but current selector does not pass; pivot to terminal-risk/value refit or bounded training."
    elif not fx["physical_gate_vs_H15"]:
        decision = "v27 fixed H12 fails physical/safety gate without deployable selector rescue; diagnose scenario/reward/terminal/training mismatch before validation."
    else:
        decision = "v27 yields ambiguous fixed-H12 tradeoff; no adaptive validation claim, continue scenario/comparison diagnosis."
    return {"label_analysis": label, "evaluation_rows": rows, "policies": policies, "decision": decision, "overhead_charged_s": {"selector_actual_mean": overhead}}

def write_summary(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    pol = raw["analysis"]["policies"]
    fx = pol["fixed_H12"]; sel = pol["selector_preoutcome_actual_overhead"]; oracle = pol["oracle_H12_H15"]
    lines = [
        "# v27 fixed-H12-primary development confirmation",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; validation64 closed, sealed test closed, no training/refit/grid search.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` / `{raw['budget_declared']['total_episodes_exact']}` episodes; `{raw['budget_actual']['control_steps']}` / `{raw['budget_declared']['control_step_upper_bound']}` control steps; selector choices `{raw['budget_actual']['selector_choice_evaluations']}`.",
        "",
        "## Headline",
        "",
        f"- States: `{ag['states']}`; H12-beneficial `{ag['beneficial_H12_count']}`; H12-catastrophic/high-cost `{ag['catastrophic_H12_count']}`; H15 unsafe `{ag['h15_unsafe_count']}`.",
        f"- Fixed H12 vs H15: decision saving `{pct(fx['decision_relative_saving_vs_H15'])}`, solver saving `{pct(fx['solver_relative_saving_vs_H15'])}`, bad `{fx['catastrophic_false_positive_count']}`, physical gate `{fx['physical_gate_vs_H15']}`, pass5 `{fx['pass5_zero_cat_physical']}`.",
        f"- Pre-outcome selector: H counts `{sel['chosen_counts']}`, overhead-adjusted decision saving `{pct(sel['decision_relative_saving_vs_H15'])}`, solver saving `{pct(sel['solver_relative_saving_vs_H15'])}`, bad `{sel['catastrophic_false_positive_count']}`, physical gate `{sel['physical_gate_vs_H15']}`, pass5 `{sel['pass5_zero_cat_physical']}`.",
        f"- Oracle H12/H15: H counts `{oracle['chosen_counts']}`, decision saving `{pct(oracle['decision_relative_saving_vs_H15'])}`, bad `{oracle['catastrophic_false_positive_count']}`, pass5 `{oracle['pass5_zero_cat_physical']}`.",
        f"- Selector feature+decision overhead: mean `{raw['preoutcome_selector_choices']['overhead_summary']['mean_s']:.9f}` s, p95 `{raw['preoutcome_selector_choices']['overhead_summary']['p95_s']:.9f}` s over `{raw['preoutcome_selector_choices']['overhead_summary']['n']}` states.",
        f"- Decision: {raw['analysis']['decision']}",
        "",
        "## Per-state results",
        "",
        "| state | source | role | selector H | H12 ben | H12 bad | physΔ H12-H15 | H12 decision | H15 decision | H12 solver | H15 solver |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    choice_by = {str(c["base_state_id"]): c for c in raw["preoutcome_selector_choices"].get("choices", [])}
    for r in raw["analysis"]["label_analysis"]["state_rows"]:
        ch = choice_by.get(str(r["base_state_id"]), {})
        lines.append("| `%s` | %d | `%s` | %s | `%s` | `%s` | %.6g | %.6g | %.6g | %.6g | %.6g |" % (
            r["base_state_id"], si(r.get("source_candidate_index"), -1), r.get("role") or r.get("risk_probe_role"), ch.get("selected_h_preoutcome", "NA"), r.get("h12_beneficial_vs_h15"), r.get("h12_catastrophic_vs_h15"), sf(r.get("physical_delta_h12_minus_h15")), sf((r.get("h12") or {}).get("decision_sum_s")), sf((r.get("h15") or {}).get("decision_sum_s")), sf((r.get("h12") or {}).get("solver_sum_s")), sf((r.get("h15") or {}).get("solver_sum_s"))))
    lines += ["", "## Limits", "", "This is stress-pool development evidence only, not validation64, not sealed-test evidence, not a population estimate, and not ORIGINAL SAC. Fixed H12 is now evaluated as the primary simple comparator; any final claim still requires fresh independent validation/test, complete strong baselines, and full closed-loop timing.", "", f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def update_docs(raw: Mapping[str, Any]) -> None:
    pol = raw["analysis"]["policies"]; fx = pol["fixed_H12"]; sel = pol["selector_preoutcome_actual_overhead"]
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v27 fixed-H12-primary development confirmation

UTC: {raw['created_utc']}. Development-only v26b-repaired fixed-H12-primary confirmation completed; validation64 closed, sealed test closed, no training/refit/grid search. Budget {raw['budget_actual']['episodes']} episodes / {raw['budget_actual']['control_steps']} control steps plus {raw['budget_actual']['selector_choice_evaluations']} preoutcome selector choices. States={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic/high-cost={ag['catastrophic_H12_count']}. Fixed H12 save={fx['decision_relative_saving_vs_H15']}, bad={fx['catastrophic_false_positive_count']}, pass5={fx['pass5_zero_cat_physical']}; selector save={sel['decision_relative_saving_vs_H15']}, H counts={sel['chosen_counts']}, bad={sel['catastrophic_false_positive_count']}, pass5={sel['pass5_zero_cat_physical']}. Decision: {raw['analysis']['decision']} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
## Follow-up through v27 fixed-H12-primary confirmation

Updated by GPT-5.5 executor at `{raw['created_utc']}`. v27 did not access validation64 or sealed test and performed no training/refit.

| linked recommendation(s) | disposition after v27 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; acted with fixed H12 primary | v27 fixed H12 save {pct(fx['decision_relative_saving_vs_H15'])}, bad {fx['catastrophic_false_positive_count']}, physical gate {fx['physical_gate_vs_H15']}, pass5 {fx['pass5_zero_cat_physical']}; selector save {pct(sel['decision_relative_saving_vs_H15'])}, H counts {sel['chosen_counts']}, bad {sel['catastrophic_false_positive_count']}. | Use v27 decision rule: {raw['analysis']['decision']} |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; partially addressed, still not final speed | v27 charged measured preoutcome feature+selector overhead mean {raw['preoutcome_selector_choices']['overhead_summary']['mean_s']:.9f}s/p95 {raw['preoutcome_selector_choices']['overhead_summary']['p95_s']:.9f}s over {raw['preoutcome_selector_choices']['overhead_summary']['n']} states and reports branch whole-decision/solver sums. | Still no final deployed speed claim without later closed-loop validation/final timing. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v27 uses the v26/v26b stress-bank development selection; it is not validation64/population/final-test evidence. | Keep claims development-scoped and require fresh independent confirmation before final test. |
| `A11_training_failure_modes_need_separation` | accepted; next branch depends on fixed-H12 result | v27 states={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic={ag['catastrophic_H12_count']}. | If fixed H12 passes, prioritize scenario/comparison design; if failures recur and selector/oracle helps, pivot to terminal-risk/value refit or bounded training. |
| `A12_registry_backup_schema_contract` | accepted; active | v27 wrote new source/results/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified external backup before further unique science. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    old = (ROOT / "EXPERIMENT_REGISTRY.csv").read_text(encoding="utf-8", errors="replace") if (ROOT / "EXPERIMENT_REGISTRY.csv").exists() else ""
    if MARKER not in old:
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], f"RNG_SEED={RNG_SEED}", "development_fixed_H12_primary_no_validation64_no_test", raw["budget_actual"]["episodes"], raw["budget_actual"]["control_steps"], raw["budget_actual"]["selector_choice_evaluations"], 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v27 fixed-H12-primary confirmation\n\nUTC: {raw['created_utc']}\n\nDecision: {raw['analysis']['decision']}\n\nKey results: states={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic={ag['catastrophic_H12_count']}; fixed H12 save={pct(fx['decision_relative_saving_vs_H15'])}, bad={fx['catastrophic_false_positive_count']}, pass5={fx['pass5_zero_cat_physical']}; selector save={pct(sel['decision_relative_saving_vs_H15'])}, H counts={sel['chosen_counts']}, bad={sel['catastrophic_false_positive_count']}, pass5={sel['pass5_zero_cat_physical']}.\n\nNext: require verified external backup covering v27. Then follow decision rule: if fixed H12 passed, scenario/comparison design is higher-value than another static selector sweep; if fixed-H12 failures recurred with selector/oracle value, inspect failure telemetry and run terminal-risk/value refit or bounded training.\n", encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-proof", default=str(DEFAULT_BACKUP_PROOF))
    ap.add_argument("--i-accept-development-v27", action="store_true", required=True)
    args = ap.parse_args(argv)
    delattr(args, "run")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if (RUN_DIR / "completed.json").exists():
            done = completed_ok(RUN_DIR / "completed.json", "existing v27")
            print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
            return 0
        inputs = load_protocol_inputs()
        v26b_time = parse_time(inputs["v26b_done"].get("created_utc"))
        backup = verify_backup(Path(args.backup_proof), v26b_time)
        preflight = completed_ok(V27_PREFLIGHT_DONE, "v27 preflight") if V27_PREFLIGHT_DONE.exists() else {"input_sufficient_for_unique_simulation_now": None}
        started = now_utc()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "classification": "development_fixed_H12_primary_no_validation64_no_test", "backup_proof": rel(Path(args.backup_proof)), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0})
        v8.engine.case_runner.SMOKE_DIR = RUN_DIR
        _, stage1_runner, _ = v1d.import_legacy_modules()
        runtime_preflight = stage1_runner.runtime_preflight()
        if not runtime_preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (runtime_preflight,))
        stage1_runner.base.v1.latency_verify()
        terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
        if REF_H not in terminals:
            raise ContractError("terminal grid missing H15 terminal")
        write_json(RUN_DIR / "runtime_preflight.json", runtime_preflight)
        write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
        input_hashes = {rel(p): sha256(p) for p in [Path(__file__).resolve(), V26_PROTOCOL, V26B_PROTOCOL, V26B_DONE, Path(args.backup_proof)] if p.exists()}
        if V27_PREFLIGHT_DONE.exists():
            input_hashes[rel(V27_PREFLIGHT_DONE)] = sha256(V27_PREFLIGHT_DONE)
        selected_cases = inputs["selected_cases"]
        episodes: List[Dict[str, Any]] = []
        stage_a_eps: List[Dict[str, Any]] = []
        for i, case_meta in enumerate(selected_cases):
            case = case_meta["case_snapshot_from_candidate_pool"]
            item = {
                "execution_index": i,
                "state_id": f"v27_stageA_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
                "case": i,
                "source_candidate_index": int(case_meta["source_candidate_index"]),
                "branch_step": 0,
                "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
                "true_mpc_n_horizon": REF_H,
                "commanded_horizon": REF_H,
                "terminal_mode": "matched_terminal",
                "initialization": "v27_stageA_H15_trace_before_selector_and_H12_outcome",
            }
            summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
            summary["stage"] = "A_h15_trace_scan"; summary["fresh_case_index"] = i; summary["role"] = case_meta.get("role")
            stage_a_eps.append(summary); episodes.append(summary)
            write_json(RUN_DIR / "progress.json", {"stage": "A", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})
        selected_states = select_stage_a_states(stage_a_eps)
        role_by_case = {int(c["fresh_case_index"]): c.get("role") for c in selected_cases}
        for st in selected_states:
            st["role"] = role_by_case.get(si(st.get("fresh_case_index"), -1))
        write_json(RUN_DIR / "selected_state_manifest.json", {"created_utc": now_utc().isoformat(), "stage_A_complete_before_selector_and_stage_B": True, "windows": WINDOWS, "selected_states": selected_states, "v26b_protocol_sha256": sha256(V26B_PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False})
        write_json(RUN_DIR / "manifest_completed_before_selector_and_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "selector_started": False, "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})
        choices = make_preoutcome_selector_choices(selected_states, input_hashes)
        write_json(RUN_DIR / "selector_choices_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "preoutcome_selector_choices": rel(RUN_DIR / "preoutcome_selector_choices.json"), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})
        state_by_id = {str(s["base_state_id"]): s for s in selected_states}
        branch_template = make_stage_b_template(selected_states)
        branch_episodes: List[Dict[str, Any]] = []
        write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "selector_choices_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
        for j, tmpl in enumerate(branch_template):
            st = state_by_id[str(tmpl["base_state_id"])]
            fc = si(tmpl.get("fresh_case_index"), -1)
            case_meta = selected_cases[fc]
            case = case_meta["case_snapshot_from_candidate_pool"]
            h = si(tmpl.get("true_mpc_n_horizon"), -1)
            item = {
                "execution_index": 7000 + j,
                "state_id": f"v27_B{j:03d}_{st['base_state_id']}_H{h}",
                "case": fc,
                "source_candidate_index": int(case_meta["source_candidate_index"]),
                "branch_step": int(st["branch_step"]),
                "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
                "true_mpc_n_horizon": h,
                "commanded_horizon": h,
                "terminal_mode": PRIMARY_TERMINAL_PROFILE,
                "initialization": "v27_stageB_direct_branch_state_from_predeclared_H15_manifest",
            }
            summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
            summary.update({"stage": "B_shared_h15_trueH12_H15", "template_execution_index": int(tmpl.get("execution_index", j)), "repeat": si(tmpl.get("repeat"), 0), "fresh_case_index": fc, "base_state_id": st["base_state_id"], "branch_state_slot": si(st.get("branch_state_slot"), -1), "terminal_profile": PRIMARY_TERMINAL_PROFILE, "terminal_source_horizon": REF_H, "role": case_meta.get("role")})
            branch_episodes.append(summary); episodes.append(summary)
            write_json(RUN_DIR / "progress.json", {"stage": "B", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "stage_B_done": len(branch_episodes), "stage_B_expected": len(branch_template), "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})
        control_steps = int(sum(si(e.get("steps")) for e in episodes))
        if len(episodes) != TOTAL_EPISODES or control_steps > CONTROL_STEP_CAP:
            raise ContractError("budget mismatch")
        analysis = analyze(branch_episodes, selected_states, choices)
        created = now_utc()
        raw = {"created_utc": created.isoformat(), "started_utc": started.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "development_IMPROVED_fixed_H12_primary_confirmation_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "mobile_robot_mppi_resumed": False, "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit")}, "preflight_used": preflight, "v26b_corrected_before_evidence": inputs["v26b"].get("corrected_before_evidence"), "budget_declared": {"stage_A_h15_trace_episodes": TARGET_CASES, "stage_B_branch_episodes": TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS), "total_episodes_exact": TOTAL_EPISODES, "control_step_upper_bound": CONTROL_STEP_CAP, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "selector_choice_evaluations": len(choices.get("choices", [])), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "selected_cases": selected_cases, "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "preoutcome_selector_choices": choices, "episodes": episodes, "analysis": analysis, "input_hashes": input_hashes, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}, "interpretation_limits": ["development stress-pool evidence only", "not validation64", "not sealed test", "not population estimate", "not ORIGINAL SAC", "no new training/refit/grid search"], "backup_request": rel(BACKUP_REQUEST)}
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v27_fixed_H12_primary_confirmation", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v27 development simulation evidence, source, docs/state/registry/response-log", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False})
        update_docs(raw)
        pol = analysis["policies"]
        headline = {"states": analysis["label_analysis"]["aggregate"]["states"], "beneficial_H12_count": analysis["label_analysis"]["aggregate"]["beneficial_H12_count"], "catastrophic_H12_count": analysis["label_analysis"]["aggregate"]["catastrophic_H12_count"], "fixed_H12_save": pol["fixed_H12"]["decision_relative_saving_vs_H15"], "fixed_H12_bad": pol["fixed_H12"]["catastrophic_false_positive_count"], "fixed_H12_pass5": pol["fixed_H12"]["pass5_zero_cat_physical"], "selector_save": pol["selector_preoutcome_actual_overhead"]["decision_relative_saving_vs_H15"], "selector_bad": pol["selector_preoutcome_actual_overhead"]["catastrophic_false_positive_count"], "selector_pass5": pol["selector_preoutcome_actual_overhead"]["pass5_zero_cat_physical"], "selector_h_counts": pol["selector_preoutcome_actual_overhead"]["chosen_counts"], "decision": analysis["decision"]}
        completed = {"status": "complete", "hard_pass": bool(pol["fixed_H12"]["pass5_zero_cat_physical"] or pol["selector_preoutcome_actual_overhead"]["pass5_zero_cat_physical"]), "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": headline, "backup_request": rel(BACKUP_REQUEST), "hashes": {}}
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), V26_PROTOCOL, V26B_PROTOCOL, V26B_DONE, STATE, BACKUP_REQUEST, Path(args.backup_proof), ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
        completed["hashes"] = {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}
        write_json(RUN_DIR / "completed.json", completed)
        raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": headline, "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_fixed_H12_primary_confirmation_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
