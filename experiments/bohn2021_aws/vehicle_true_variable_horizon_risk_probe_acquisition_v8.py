#!/usr/bin/env python3
"""Targeted source-independent H10 risk-probe acquisition v8.

Development-only IMPROVED diagnostic.  This is not validation64 and not sealed
final-test work.  It is a bounded data-acquisition intervention after the v5/v7
negative evidence: fresh-bank catastrophics appear to be intrinsic short-H risk
and pure deployable-history selectors still leave false positives.  The script
freezes a fresh-source target manifest before any new H10 outcome, then runs a
small true-H10/H15 paired branch block under the primary shared-H15 terminal.

Budget: 4 fresh cases x 2 H15-trace-selected states x 2 true horizons x 2
repeats + 4 Stage-A H15 traces = 36 development episodes, <=5400 control steps.
No gradient training, selector refit, validation64, or sealed test access.
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
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0 as freeze0  # noqa:E402
import vehicle_true_variable_horizon_fresh_source_confirmation_v1_runner as engine  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_probe_acquisition_v8"
STAMP = "20260929T1258Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1258Z_after_risk_probe_acquisition_v8.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-risk-probe-acquisition-v8-{STAMP}"

V0_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json"
V1_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json"
V2_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json"
V0_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json"
V2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/raw.json"
V7_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_representation_v7_lobo_20260929T122843Z/completed.json"
V7_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_representation_v7_lobo_20260929T122843Z/summary.md"
V5_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_causality_fresh_v5_20260929T121533Z/completed.json"

TRUE_HORIZONS = [10, 15]
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
REPEATS = 2
TARGET_CASES = 4
BRANCH_STATES_PER_CASE = 2
MAX_STEPS = 150
RNG_SEED = 202609291258
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS
CONTROL_STEP_CAP = TOTAL_EPISODES * MAX_STEPS

DESIRED_ROLES = [
    ("fresh_high_heading_long_or_medium", "risk_high_heading_long_or_medium"),
    ("fresh_high_heading_short", "risk_high_heading_short_prior_catastrophic_mode"),
    ("fresh_low_heading_low_clearance_control", "risk_low_heading_low_clearance_prior_catastrophic_mode"),
    ("fresh_lower_stress_control", "lower_stress_specificity_control"),
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
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


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


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("prerequisite did not pass: " + rel(path))
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if key in obj and obj.get(key) is not False:
            raise ContractError(f"unexpected {key} in {rel(path)}")
    return obj


def collect_indices(obj: Any) -> List[int]:
    out: List[int] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k in ("source_candidate_index", "candidate_index"):
                try:
                    x = int(v)
                    if x >= 0:
                        out.append(x)
                except Exception:
                    pass
            out.extend(collect_indices(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_indices(v))
    return out


def indices_from_paths(paths: Sequence[Path]) -> List[int]:
    vals: List[int] = []
    for p in paths:
        if p.exists():
            try:
                vals.extend(collect_indices(read_json(p)))
            except Exception:
                pass
    return sorted(set(vals))


def verify_inputs() -> Dict[str, str]:
    required_done = [freeze0.BANK_DONE, freeze0.STAGE1_DONE, freeze0.ORACLE_DONE, freeze0.RISK_DONE, V5_DONE, V7_DONE]
    for p in required_done:
        completed_ok(p)
    required = [freeze0.BANK_PATH, freeze0.ORACLE_PROTOCOL, freeze0.ORACLE_RAW, freeze0.RISK_PROTOCOL, freeze0.RISK_RAW, V0_PROTOCOL, V1_PROTOCOL, V2_PROTOCOL, V0_RAW, V1_RAW, V2_RAW, V7_SUMMARY]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise ContractError("missing required inputs: " + ", ".join(missing))
    return {rel(p): sha256(p) for p in required + required_done + [SOURCE] if p.exists()}


def choose_cases(bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metas = list((bank.get("selection") or {}).get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) < 64:
        raise ContractError("candidate bank dimensions invalid")
    excluded = set(freeze0.selected_stage1_indices(bank))
    excluded.update(freeze0.used_candidate_indices(read_json(freeze0.ORACLE_PROTOCOL), read_json(freeze0.ORACLE_RAW), read_json(freeze0.RISK_PROTOCOL), read_json(freeze0.RISK_RAW)))
    excluded.update(indices_from_paths([V0_PROTOCOL, V1_PROTOCOL, V2_PROTOCOL, V0_RAW, V1_RAW, V2_RAW]))
    thr = freeze0.thresholds_from_bank(bank, metas)
    enriched: List[Dict[str, Any]] = []
    for row0 in metas:
        row = dict(row0)
        idx = si(row.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases) or idx in excluded:
            continue
        row["strict_group"] = freeze0.classify(row, thr, relaxed=False)
        row["relaxed_group"] = freeze0.classify(row, thr, relaxed=True)
        enriched.append(row)
    chosen: List[Dict[str, Any]] = []
    used: set = set()
    fallbacks: List[str] = []

    def sort_pool(pool: Sequence[Mapping[str, Any]], group: str) -> List[Mapping[str, Any]]:
        if "lower_stress" in group:
            return sorted(pool, key=lambda r: (sf(r.get("stress_v1_score")), sf(r.get("abs_theta_r")), -si(r.get("candidate_index"))))
        if "low_clearance" in group:
            return sorted(pool, key=lambda r: (sf(r.get("min_reference_obstacle_clearance"), 1e9), -sf(r.get("abs_theta_r")), -sf(r.get("stress_v1_score"))))
        if "short" in group:
            return sorted(pool, key=lambda r: (-sf(r.get("abs_theta_r")), sf(r.get("traj_steps")), -sf(r.get("stress_v1_score"))))
        return sorted(pool, key=lambda r: (-sf(r.get("stress_v1_score")), -sf(r.get("abs_theta_r")), -sf(r.get("traj_steps"))))

    for group, role in DESIRED_ROLES:
        pools = [
            ("strict", [r for r in enriched if r["strict_group"] == group and si(r.get("candidate_index"), -1) not in used]),
            ("relaxed", [r for r in enriched if r["relaxed_group"] == group and si(r.get("candidate_index"), -1) not in used]),
            ("global", [r for r in enriched if si(r.get("candidate_index"), -1) not in used]),
        ]
        selected_row = None
        selected_mode = None
        for mode, pool in pools:
            if pool:
                selected_row = dict(sort_pool(pool, group)[0])
                selected_mode = mode
                break
        if selected_row is None:
            raise ContractError("no eligible candidate for role " + role)
        idx = si(selected_row.get("candidate_index"), -1)
        used.add(idx)
        if selected_mode != "strict":
            fallbacks.append(role + "_" + str(selected_mode))
        selected_row["fresh_case_index"] = len(chosen)
        selected_row["fresh_confirmation_group"] = group
        selected_row["risk_probe_role"] = role
        selected_row["selection_mode"] = selected_mode
        chosen.append(selected_row)
    selected_cases: List[Dict[str, Any]] = []
    for row in chosen:
        idx = si(row.get("candidate_index"), -1)
        selected_cases.append({
            "fresh_case_index": int(row["fresh_case_index"]),
            "source_candidate_index": idx,
            "fresh_confirmation_group": row["fresh_confirmation_group"],
            "risk_probe_role": row["risk_probe_role"],
            "selection_mode": row.get("selection_mode"),
            "strict_group": row.get("strict_group"),
            "relaxed_group": row.get("relaxed_group"),
            "theta_r": sf(row.get("theta_r")),
            "abs_theta_r": sf(row.get("abs_theta_r")),
            "traj_steps": sf(row.get("traj_steps")),
            "min_reference_obstacle_clearance": sf(row.get("min_reference_obstacle_clearance"), 0.0),
            "stress_v1_score": sf(row.get("stress_v1_score")),
            "case_snapshot_from_candidate_pool": cases[idx],
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(cases[idx]), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
        })
    diag = {
        "excluded_source_candidate_indices_count": len(excluded),
        "eligible_after_exclusion_count": len(enriched),
        "selected_source_candidate_indices": [int(c["source_candidate_index"]) for c in selected_cases],
        "fallbacks": fallbacks,
        "strict_pool_counts": {g: sum(1 for r in enriched if r["strict_group"] == g) for g, _ in DESIRED_ROLES},
        "relaxed_pool_counts": {g: sum(1 for r in enriched if r["relaxed_group"] == g) for g, _ in DESIRED_ROLES},
    }
    return selected_cases, diag


def make_stage_a(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{"stage": "A_h15_trace_scan", "execution_index": i, "fresh_case_index": int(c["fresh_case_index"]), "source_candidate_index": int(c["source_candidate_index"]), "true_mpc_n_horizon": 15, "terminal_profile": "matched_terminal", "purpose": "collect H15 trace before any H10 outcome"} for i, c in enumerate(selected_cases)]


def make_stage_b_template(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for c in selected_cases:
            for slot in range(BRANCH_STATES_PER_CASE):
                hs = list(TRUE_HORIZONS)
                rng.shuffle(hs)
                for h in hs:
                    rows.append({"stage": "B_shared_h15_trueH10_H15", "execution_index": exe, "repeat": rep, "fresh_case_index": int(c["fresh_case_index"]), "source_candidate_index": int(c["source_candidate_index"]), "branch_state_slot": slot, "terminal_profile": PRIMARY_TERMINAL_PROFILE, "true_mpc_n_horizon": int(h), "blocked_randomization_unit": f"v8|rep{rep}|case{c['fresh_case_index']}|slot{slot}|shared_h15"})
                    exe += 1
    return rows


def build_protocol(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str]) -> Dict[str, Any]:
    stage_a = make_stage_a(selected_cases)
    stage_b = make_stage_b_template(selected_cases)
    if len(stage_a) + len(stage_b) != TOTAL_EPISODES:
        raise ContractError("unexpected v8 budget")
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_targeted_risk_probe_acquisition_preoutcome_not_validation_not_test",
        "hypothesis": "Fresh-bank failures are caused by state-local intrinsic H10 risk concentrated in early high-heading/low-clearance modes. A small source-independent shared-H15-terminal probe should reproduce or refute that risk before another selector sweep; if reproduced, next step is risk/value representation training or value-refit using these anchors.",
        "before_evidence": ["v5: shared-H15 oracle opportunity remains strong but fixed H10 has catastrophic rows", "v7: online-history deployable trees still leave a false-positive despite >10% timing potential"],
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "stage_A_h15_trace_state_selection": {
            "schedule": stage_a,
            "horizon": 15,
            "terminal_profile": "matched_terminal",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 90, "reserve_terminal_margin_steps": 3},
            "windows": [
                {"slot": 0, "name": "early_risk", "low_fraction": 0.12, "high_fraction": 0.34, "selection": "max_h15_trace_risk_score"},
                {"slot": 1, "name": "mid_late_control", "low_fraction": 0.45, "high_fraction": 0.74, "selection": "max_h15_trace_risk_score_with_min_step_separation_8"},
            ],
            "manifest_rule": "selected_state_manifest is written after all Stage-A traces and before any Stage-B H10 branch outcome",
        },
        "stage_B_branch_confirmation": {"template": stage_b, "true_horizons": TRUE_HORIZONS, "terminal_profile": PRIMARY_TERMINAL_PROFILE, "repeats": REPEATS, "primary_measurements": ["physical_constraint_cost", "whole_decision_time", "solver_attempt_time", "success", "solver failures"]},
        "budget_declared": {"stage_A_h15_trace_episodes": len(stage_a), "stage_B_branch_episodes": len(stage_b), "total_episodes_exact": TOTAL_EPISODES, "control_step_upper_bound": CONTROL_STEP_CAP, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rule": {"if_reproduces_catastrophic_early_risk": "do not validate current selector; train/refit a risk/value representation or acquire more risk anchors", "if_no_catastrophes_and_H10_safe": "reassess prior failure specificity and scenario distribution/timing before training", "if_fixed_H15_or_fixed_H10_absorbs_tradeoff": "document weak adaptive opportunity rather than force switching"},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, protocol)
    return protocol


def terminal_for_shared_h15(h: int, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    if 15 not in terminals:
        raise ContractError("terminal grid missing H15")
    return terminals[15]


def summarize_for_h(episodes: Sequence[Mapping[str, Any]], h: int) -> Dict[str, Any]:
    reps = [e for e in episodes if si(e.get("true_mpc_n_horizon"), -1) == h]
    return {"n": len(reps), "safe_all": bool(reps) and all(is_safe_episode(e) for e in reps), "physical": median([sf(e.get("physical_constraint_cost"), 0.0) for e in reps]), "total_cost": median([sf(e.get("total_cost"), 0.0) for e in reps]), "decision_sum_s": median([metric_sum(e, "decision_timing_s") for e in reps]), "solver_sum_s": median([metric_sum(e, "solver_attempt_timing_s") for e in reps]), "steps": median([si(e.get("steps"), 0) for e in reps]), "success_all": bool(reps) and all(bool(e.get("success")) for e in reps), "solver_failure_steps_sum": int(sum(si(e.get("solver_failure_steps"), 0) for e in reps)), "opt_x_size": sorted(set(si(x) for e in reps for x in (e.get("opt_x_sizes_observed") or [])))}


def analyze(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    state_meta = {str(s["base_state_id"]): s for s in selected_states}
    rows: List[Dict[str, Any]] = []
    for sid in sorted(state_meta):
        eps = [e for e in branch_episodes if str(e.get("base_state_id")) == sid]
        h10 = summarize_for_h(eps, 10)
        h15 = summarize_for_h(eps, 15)
        tol = max(2.0, 0.05 * abs(sf(h15.get("physical"), 0.0))) if h15.get("physical") is not None else 2.0
        delta_phys = None if h10.get("physical") is None or h15.get("physical") is None else float(h10["physical"] - h15["physical"])
        delta_dec = None if h10.get("decision_sum_s") is None or h15.get("decision_sum_s") is None else float(h10["decision_sum_s"] - h15["decision_sum_s"])
        catastrophic_threshold = max(2.0, 0.25 * abs(sf(h15.get("physical"), 0.0)))
        catastrophic = bool(h15.get("safe_all") and ((not h10.get("safe_all")) or (delta_phys is not None and delta_phys > catastrophic_threshold)))
        beneficial = bool(h15.get("safe_all") and h10.get("safe_all") and delta_phys is not None and delta_phys <= tol and delta_dec is not None and delta_dec < 0.0)
        st = state_meta[sid]
        rows.append({"base_state_id": sid, "fresh_case_index": st.get("fresh_case_index"), "source_candidate_index": st.get("source_candidate_index"), "fresh_confirmation_group": st.get("fresh_confirmation_group"), "risk_probe_role": st.get("risk_probe_role"), "branch_state_slot": st.get("branch_state_slot"), "slot_name": "early_risk" if si(st.get("branch_state_slot"), 0) == 0 else "mid_late_control", "branch_step": st.get("branch_step"), "h10": h10, "h15": h15, "row_tolerance_vs_H15": tol, "physical_delta_h10_minus_h15": delta_phys, "decision_delta_h10_minus_h15": delta_dec, "catastrophic_threshold": catastrophic_threshold, "h10_catastrophic_vs_h15": catastrophic, "h10_beneficial_vs_h15": beneficial})
    fixed15_phys = math.fsum(sf(r["h15"].get("physical"), 0.0) for r in rows)
    fixed10_phys = math.fsum(sf(r["h10"].get("physical"), 0.0) for r in rows)
    fixed15_dec = math.fsum(sf(r["h15"].get("decision_sum_s"), 0.0) for r in rows)
    fixed10_dec = math.fsum(sf(r["h10"].get("decision_sum_s"), 0.0) for r in rows)
    oracle_phys = math.fsum(sf((r["h10"] if r["h10_beneficial_vs_h15"] else r["h15"]).get("physical"), 0.0) for r in rows)
    oracle_dec = math.fsum(sf((r["h10"] if r["h10_beneficial_vs_h15"] else r["h15"]).get("decision_sum_s"), 0.0) for r in rows)
    tol_sum = math.fsum(sf(r.get("row_tolerance_vs_H15"), 2.0) for r in rows)
    catastrophic_rows = [r for r in rows if r["h10_catastrophic_vs_h15"]]
    early_cats = [r for r in catastrophic_rows if r["slot_name"] == "early_risk"]
    risk_cats = [r for r in catastrophic_rows if "risk_" in str(r.get("risk_probe_role"))]
    oracle_saving = None if fixed15_dec <= 0 else (fixed15_dec - oracle_dec) / fixed15_dec
    fixed10_saving = None if fixed15_dec <= 0 else (fixed15_dec - fixed10_dec) / fixed15_dec
    fixed10_physical_gate = bool((fixed10_phys - fixed15_phys) <= tol_sum)
    if len(early_cats) >= 2 and len(risk_cats) >= 2:
        decision = "targeted fresh risk probe reproduced source-independent early H10 catastrophic modes; next run bounded risk/value representation training or value-refit, not another selector threshold sweep"
    elif not catastrophic_rows and fixed10_physical_gate and fixed10_saving is not None and fixed10_saving >= 0.10:
        decision = "targeted fresh probe did not reproduce H10 catastrophics and fixed H10 absorbs the shared-H15 tradeoff here; reassess scenario distribution and fixed-H comparison before training"
    elif oracle_saving is not None and oracle_saving >= 0.10 and len(catastrophic_rows) > 0:
        decision = "adaptive opportunity exists but requires risk-aware value/representation learning before selector rollout"
    else:
        decision = "probe is mixed/low-opportunity; inspect rows and choose scenario/value intervention rather than validation rollout"
    return {"state_rows": rows, "catastrophic_rows": catastrophic_rows, "early_catastrophic_rows": early_cats, "risk_catastrophic_rows": risk_cats, "beneficial_count": sum(1 for r in rows if r["h10_beneficial_vs_h15"]), "aggregate": {"fixed_H15_physical_sum": fixed15_phys, "fixed_H10_physical_sum": fixed10_phys, "physical_tolerance_sum": tol_sum, "fixed_H10_physical_gate_vs_H15": fixed10_physical_gate, "fixed_H15_decision_sum_s": fixed15_dec, "fixed_H10_decision_sum_s": fixed10_dec, "fixed_H10_decision_relative_saving_vs_H15": fixed10_saving, "oracle_physical_sum": oracle_phys, "oracle_decision_sum_s": oracle_dec, "oracle_decision_relative_saving_vs_H15": oracle_saving}, "decision": decision}


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = ["# Vehicle true-variable-H targeted risk-probe acquisition v8", "", f"UTC: `{raw['created_utc']}`. Development-only fresh-source risk probe; no validation64, no sealed test, no training/refit.", "", f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.", "", "## Headline", "", f"- States: `{len(a['state_rows'])}`; H10-beneficial `{a['beneficial_count']}`; catastrophic H10 rows `{len(a['catastrophic_rows'])}`; early catastrophics `{len(a['early_catastrophic_rows'])}`; risk-role catastrophics `{len(a['risk_catastrophic_rows'])}`.", f"- Aggregate: `{a['aggregate']}`.", f"- Decision: {a['decision']}", "", "## Per-state rows", "", "| state | role | slot | H10 ben | H10 cat | H10 phys | H15 phys | H10 dec | H15 dec |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(v: Any) -> str:
        return "NA" if v is None else "%.6g" % float(v)
    for r in a["state_rows"]:
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s |" % (r["base_state_id"], r.get("risk_probe_role"), r.get("slot_name"), r.get("h10_beneficial_vs_h15"), r.get("h10_catastrophic_vs_h15"), fmt(r["h10"].get("physical")), fmt(r["h15"].get("physical")), fmt(r["h10"].get("decision_sum_s")), fmt(r["h15"].get("decision_sum_s"))))
    lines += ["", "## Interpretation", "", "This diagnostic deliberately tested a fresh, source-independent subset of risk/control cases under true variable H10/H15 with measured decision and solver timing. It is development evidence only and does not open validation64 or the sealed final test. The intended next action is determined by the decision rule frozen in the preoutcome protocol, not by post-hoc validation acceptance.", "", f"Preoutcome protocol: `{raw['protocol']['json']}`.", f"Backup request: `{raw['backup_request_after_run']}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H targeted risk-probe acquisition v8

UTC: {raw['created_utc']}. Development-only fresh-source risk probe completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, validation64 closed, sealed test closed, no training/refit. Catastrophic H10 rows: {len(raw['analysis']['catastrophic_rows'])}; early catastrophics: {len(raw['analysis']['early_catastrophic_rows'])}; beneficial H10 rows: {raw['analysis']['beneficial_count']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if MARKER not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{raw['created_utc']},{NAME},development_targeted_risk_probe,true_variable_H_shared_h15_terminal,{raw['budget_actual']['episodes']},{raw['budget_actual']['control_steps']},0,0,0,False,{rel(RUN_DIR / 'completed.json')}\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", type=str, required=True)
    ap.add_argument("--i-accept-development-risk-probe-v8", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_risk_probe_v8:
        raise ContractError("requires --run and explicit development risk-probe v8 acknowledgement")
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    created0 = now_utc()
    input_hashes = verify_inputs()
    bank = read_json(freeze0.BANK_PATH)
    selected_cases, case_diag = choose_cases(bank)
    protocol = build_protocol(created0, selected_cases, case_diag, input_hashes)
    engine.case_runner.SMOKE_DIR = RUN_DIR
    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    if 15 not in terminals or 10 not in terminals:
        raise ContractError("terminal grid missing H10/H15")
    write_json(RUN_DIR / "run_started.json", {"started_utc": created0.isoformat(), "pid": os.getpid(), "method": NAME, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {"execution_index": i, "state_id": f"v8_stageA_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}", "case": i, "source_candidate_index": int(case_meta["source_candidate_index"]), "branch_step": 0, "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})), "true_mpc_n_horizon": 15, "commanded_horizon": 15, "terminal_mode": "matched_terminal", "initialization": "risk_probe_v8_stageA_H15_trace_before_any_H10_outcome"}
        summary = engine.case_runner.run_true_h_episode(item, case, terminals[15])
        summary["stage"] = "A_h15_trace_scan"
        summary["fresh_case_index"] = i
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["risk_probe_role"] = case_meta.get("risk_probe_role")
        stage_a_eps.append(summary); episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "A", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    selected_states = engine.fresh0.select_stage_a_states(stage_a_eps, protocol)
    role_by_case = {int(c["fresh_case_index"]): c.get("risk_probe_role") for c in selected_cases}
    for st in selected_states:
        st["risk_probe_role"] = role_by_case.get(si(st.get("fresh_case_index"), -1))
    manifest = {"created_utc": now_utc().isoformat(), "stage_A_complete_before_any_stage_B": True, "selected_states": selected_states, "protocol_sha256": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(RUN_DIR / "selected_state_manifest.json", manifest)
    write_json(RUN_DIR / "manifest_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    state_by_case_slot = {(si(s.get("fresh_case_index"), -1), si(s.get("branch_state_slot"), -1)): s for s in selected_states}
    branch_template = ((protocol.get("stage_B_branch_confirmation") or {}).get("template") or [])
    branch_episodes: List[Dict[str, Any]] = []
    write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
    for j, tmpl in enumerate(branch_template):
        fc = si(tmpl.get("fresh_case_index"), -1); slot = si(tmpl.get("branch_state_slot"), -1)
        st = state_by_case_slot.get((fc, slot))
        if st is None:
            raise ContractError("missing selected state for branch template")
        case_meta = selected_cases[fc]
        case = case_meta["case_snapshot_from_candidate_pool"]
        h = si(tmpl.get("true_mpc_n_horizon"), -1)
        rep = si(tmpl.get("repeat"), 0)
        item = {"execution_index": 3000 + j, "state_id": f"v8_B{j:03d}_{st['base_state_id']}_r{rep}_H{h}", "case": fc, "source_candidate_index": int(case_meta["source_candidate_index"]), "branch_step": int(st["branch_step"]), "branch_previous_state": copy.deepcopy(st["branch_previous_state"]), "true_mpc_n_horizon": h, "commanded_horizon": h, "terminal_mode": PRIMARY_TERMINAL_PROFILE, "initialization": "risk_probe_v8_stageB_direct_branch_state_from_predeclared_H15_manifest"}
        summary = engine.case_runner.run_true_h_episode(item, case, terminal_for_shared_h15(h, terminals))
        summary["stage"] = "B_shared_h15_trueH10_H15"
        summary["template_execution_index"] = int(tmpl.get("execution_index", j))
        summary["repeat"] = rep
        summary["fresh_case_index"] = fc
        summary["base_state_id"] = st["base_state_id"]
        summary["branch_state_slot"] = slot
        summary["terminal_profile"] = PRIMARY_TERMINAL_PROFILE
        summary["terminal_source_horizon"] = 15
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["risk_probe_role"] = case_meta.get("risk_probe_role")
        branch_episodes.append(summary); episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "B", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "stage_B_done": len(branch_episodes), "stage_B_expected": len(branch_template), "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    if len(episodes) != TOTAL_EPISODES or control_steps > CONTROL_STEP_CAP:
        raise ContractError("budget mismatch")
    analysis = analyze(branch_episodes, selected_states)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_RISK_PROBE_ACQUISITION_V8_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup v8 targeted fresh-source risk-probe outputs before any risk/value training or further simulations", "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(RUN_DIR), rel(PROTOCOL), rel(STATE), rel(CONTINUE_STATE), rel(SOURCE), rel(req)]})
    raw = {"created_utc": created.isoformat(), "started_utc": created0.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "development_IMPROVED_targeted_true_variable_H_risk_probe_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False, "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}, "budget_declared": protocol["budget_declared"], "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "selected_cases": selected_cases, "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "episodes": episodes, "analysis": analysis, "backup_request_after_run": rel(req), "interpretation_limits": ["development-only", "fresh source from existing stress-v1 candidate pool", "not validation64", "not sealed test", "not ORIGINAL SAC", "measured timing only"]}
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text("# Continue state after risk-probe acquisition v8\n\nUTC: %s\n\nDecision: %s\n\nBudgets: %s\n\nNext: backup these artifacts, then execute the indicated risk/value representation training or scenario/control-compute diagnostic; do not open validation64 or sealed test.\n" % (created.isoformat(), analysis["decision"], json.dumps(raw["budget_actual"], sort_keys=True)), encoding="utf-8")
    append_docs(raw)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, CONTINUE_STATE, req]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"states": len(analysis["state_rows"]), "beneficial_count": analysis["beneficial_count"], "catastrophic_count": len(analysis["catastrophic_rows"]), "early_catastrophic_count": len(analysis["early_catastrophic_rows"]), "decision": analysis["decision"]}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": completed["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failed.json", {"failed_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
