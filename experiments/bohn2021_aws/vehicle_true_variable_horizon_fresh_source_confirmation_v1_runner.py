#!/usr/bin/env python3
"""Run v1 independent fresh-source confirmation for the guard/veto true-H selector.

Development-only IMPROVED diagnostic.  Requires the v1 freeze protocol and a
verified external backup covering that freeze/source before execution.  It does
not open validation64 or sealed test and performs no training/refit.

Protocol summary:
  * Stage A: run H15-only traces for the v1 fresh source cases, then write a
    selected_state_manifest before any Stage-B branch outcome.
  * Stage B: run blocked true-H10/H15 branches under matched and shared-H15
    terminal profiles with two repeats.
  * Analysis: evaluate the pre-frozen outcome-informed variant
    disagreementVeto_s1_raw_abs_l2_m1.5_v1.0, trained only from the existing
    oracle/risk source banks, against fixed true H15 using measured decision and
    solver timings.  This is confirmation development evidence, not validation
    or final test.
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
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_case5_smoke_v0_runner as case_runner  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_fresh_source_confirmation_v0_runner as fresh0  # noqa:E402
import vehicle_true_variable_horizon_guard_veto_diagnostic_v0 as gv  # noqa:E402

NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_v1"
STAMP = "20260929T1110Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json"
FREEZE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_20260929T1055Z/completed.json"
GUARD_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/completed.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
RISK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json"

RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1110_after_fresh_source_confirmation_v1_run.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER_RUN = f"vehicle-true-variable-H-fresh-source-confirmation-v1-run-{STAMP}"

TRUE_HORIZONS = [10, 15]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
PRIMARY_VARIANT = "disagreementVeto_s1_raw_abs_l2_m1.5_v1.0"
MIN_DECISION_SAVING_WEAK = 0.05
MIN_DECISION_SAVING_STRONG = 0.10


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
    return fresh0.sf(value, default)


def si(value: Any, default: int = 0) -> int:
    return fresh0.si(value, default)


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    obj = fresh0.completed_ok(path, check_hashes=check_hashes)
    if obj.get("validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
        raise ContractError("validation/test access flags invalid in " + rel(path))
    return obj


def verify_inputs() -> Mapping[str, Any]:
    for p in (PROTOCOL, FREEZE_DONE, GUARD_DONE, ORACLE_DONE, RISK_DONE):
        if not p.exists():
            raise ContractError("required input missing: " + rel(p))
    completed_ok(FREEZE_DONE, check_hashes=False)
    completed_ok(GUARD_DONE, check_hashes=False)
    completed_ok(ORACLE_DONE, check_hashes=False)
    completed_ok(RISK_DONE, check_hashes=False)
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != "vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z":
        raise ContractError("unexpected v1 protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("protocol access flags invalid")
    primary = ((protocol.get("source_trained_selector_confirmation") or {}).get("primary_policy_frozen_before_fresh_v1_outcomes") or {})
    if primary.get("policy_id") != PRIMARY_VARIANT:
        raise ContractError("v1 protocol primary variant mismatch")
    budget = protocol.get("budget_declared") or {}
    if int(budget.get("total_episodes_exact", -1)) != 136 or int(budget.get("stage_b_branch_episodes", -1)) != 128:
        raise ContractError("unexpected v1 budget")
    selected = ((protocol.get("case_source") or {}).get("selected_fresh_cases") or [])
    if len(selected) != 8:
        raise ContractError("expected exactly 8 v1 fresh cases")
    return protocol


def terminal_for(profile: str, h: int, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    if profile == "matched_terminal":
        return terminals[h]
    if profile == "shared_h15_terminal":
        return terminals[15]
    raise ContractError("unknown terminal profile " + profile)


def source_guard_model(protocol: Mapping[str, Any]) -> Tuple[Mapping[str, Any], List[Dict[str, Any]]]:
    primary = ((protocol.get("source_trained_selector_confirmation") or {}).get("primary_policy_frozen_before_fresh_v1_outcomes") or {})
    spec = dict(primary.get("config") or {})
    spec.setdefault("variant_id", PRIMARY_VARIANT)
    if spec.get("variant_id") != PRIMARY_VARIANT:
        raise ContractError("primary model spec variant mismatch")
    examples = gv.load_source_examples(gv.ORACLE_RAW, "oracle_bank") + gv.load_source_examples(gv.RISK_RAW, "risk_anchor")
    model = gv.fit_predictor(examples, spec)
    return model, examples


def median_summary_for_h(episodes: Sequence[Mapping[str, Any]], h: int) -> Dict[str, Any]:
    reps = [e for e in episodes if si(e.get("true_mpc_n_horizon"), -1) == h]
    return {
        "n": len(reps),
        "safe_all": bool(reps) and all(fresh0.is_safe_episode(e) for e in reps),
        "physical": fresh0.median([sf(e.get("physical_constraint_cost"), 0.0) for e in reps]),
        "total_cost": fresh0.median([sf(e.get("total_cost"), 0.0) for e in reps]),
        "decision_sum_s": fresh0.median([fresh0.metric_sum(e, "decision_timing_s") for e in reps]),
        "solver_sum_s": fresh0.median([fresh0.metric_sum(e, "solver_attempt_timing_s") for e in reps]),
        "steps": fresh0.median([si(e.get("steps"), 0) for e in reps]),
        "success_all": bool(reps) and all(bool(e.get("success")) for e in reps),
        "constraint_any": any(bool(e.get("constraint")) for e in reps),
        "solver_failure_steps_sum": int(sum(si(e.get("solver_failure_steps"), 0) for e in reps)),
        "opt_x_size": sorted(set(si(x) for e in reps for x in (e.get("opt_x_sizes_observed") or []))),
    }


def branch_group_key(e: Mapping[str, Any]) -> Tuple[str, str]:
    return str(e.get("base_state_id") or e.get("state_id")), str(e.get("terminal_profile") or e.get("terminal_mode"))


def build_state_profile_rows(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_by_base = {str(s["base_state_id"]): s for s in selected_states}
    model, examples = source_guard_model(protocol)
    rows: List[Dict[str, Any]] = []
    unsafe_h15_groups: List[Dict[str, Any]] = []
    for base_sid, profile in sorted(set(branch_group_key(e) for e in branch_episodes)):
        group_eps = [e for e in branch_episodes if branch_group_key(e) == (base_sid, profile)]
        med_by_h = {h: median_summary_for_h(group_eps, h) for h in TRUE_HORIZONS}
        h10 = med_by_h[10]
        h15 = med_by_h[15]
        if not h15["safe_all"]:
            unsafe_h15_groups.append({"base_state_id": base_sid, "terminal_profile": profile, "reason": "H15 baseline not safe", "median_by_h": {str(h): med_by_h[h] for h in TRUE_HORIZONS}})
        safe_hs = [h for h in TRUE_HORIZONS if med_by_h[h]["safe_all"] and med_by_h[h]["physical"] is not None]
        best_phys = None
        tol_best = None
        near_best_horizons: List[int] = []
        oracle_label: Optional[int] = None
        if safe_hs:
            best_phys = min(float(med_by_h[h]["physical"]) for h in safe_hs)
            tol_best = max(2.0, 0.05 * abs(best_phys))
            near_best_horizons = [h for h in safe_hs if float(med_by_h[h]["physical"]) <= best_phys + tol_best]
            oracle_label = sorted(near_best_horizons, key=lambda hh: (float(med_by_h[hh]["decision_sum_s"]), float(med_by_h[hh]["solver_sum_s"]), hh))[0]
        row_tol_h15 = max(2.0, 0.05 * abs(float(h15["physical"]))) if h15["physical"] is not None else 2.0
        h10_beneficial = bool(
            h15["safe_all"]
            and h10["safe_all"]
            and h15["physical"] is not None
            and h10["physical"] is not None
            and float(h10["physical"]) - float(h15["physical"]) <= row_tol_h15
            and h15["decision_sum_s"] is not None
            and h10["decision_sum_s"] is not None
            and float(h10["decision_sum_s"]) < float(h15["decision_sum_s"])
        )
        st = manifest_by_base[base_sid]
        feature = gv.feature_from_observation_state(st.get("initial_observation_from_h15_trace"), st.get("branch_previous_state"))
        pred, diag = gv.predict(model, feature)
        rows.append({
            "base_state_id": base_sid,
            "fresh_case_index": st.get("fresh_case_index"),
            "source_candidate_index": st.get("source_candidate_index"),
            "fresh_confirmation_group": st.get("fresh_confirmation_group"),
            "branch_state_slot": st.get("branch_state_slot"),
            "terminal_profile": profile,
            "median_by_h": {str(h): med_by_h[h] for h in TRUE_HORIZONS},
            "best_physical": best_phys,
            "near_best_tolerance": tol_best,
            "near_best_horizons": near_best_horizons,
            "oracle_label": oracle_label,
            "row_physical_tolerance_vs_H15": row_tol_h15,
            "h10_beneficial_vs_h15": h10_beneficial,
            "primary_policy_prediction": pred,
            "primary_policy_diagnostics": diag,
        })
    source_counts: Dict[str, int] = {"agreement_positive_h10": 0, "agreement_non_h10": 0, "terminal_disagreement_or_missing": 0}
    for e in examples:
        source_counts[str(e.get("category"))] = source_counts.get(str(e.get("category")), 0) + 1
    return {"rows": rows, "unsafe_h15_groups": unsafe_h15_groups, "source_model": {"variant_id": PRIMARY_VARIANT, "counts": model.get("counts"), "radius": model.get("radius"), "mode": model.get("mode")}, "source_counts": source_counts}


def aggregate(rows_in: Sequence[Mapping[str, Any]], chooser: str) -> Dict[str, Any]:
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
        h = r.get("oracle_label") if chooser == "oracle" else r.get("primary_policy_prediction")
        if h is None:
            continue
        h = int(h)
        chosen_counts[str(h)] = chosen_counts.get(str(h), 0) + 1
        m = r["median_by_h"][str(h)]
        phys.append(sf(m.get("physical"), 0.0))
        dec.append(sf(m.get("decision_sum_s"), 0.0))
        solver.append(sf(m.get("solver_sum_s"), 0.0))
    out[chooser] = {"groups_used": len(phys), "chosen_counts": chosen_counts, "physical_sum": float(math.fsum(phys)), "decision_sum_s": float(math.fsum(dec)), "solver_sum_s": float(math.fsum(solver))}
    return out


def comparison(ag: Mapping[str, Any], chooser: str) -> Dict[str, Any]:
    c = ag[chooser]
    b = ag["fixed_H15"]
    return {
        "physical_delta_vs_fixed_H15": float(c["physical_sum"] - b["physical_sum"]),
        "decision_relative_saving_vs_fixed_H15": None if b["decision_sum_s"] <= 0 else float((b["decision_sum_s"] - c["decision_sum_s"]) / b["decision_sum_s"]),
        "solver_relative_saving_vs_fixed_H15": None if b["solver_sum_s"] <= 0 else float((b["solver_sum_s"] - c["solver_sum_s"]) / b["solver_sum_s"]),
    }


def evaluate_policy_gate(rows: Sequence[Mapping[str, Any]], profile: str) -> Dict[str, Any]:
    prof_rows = [r for r in rows if r.get("terminal_profile") == profile]
    ag_policy = aggregate(prof_rows, "primary_policy")
    ag_oracle = aggregate(prof_rows, "oracle")
    comp_policy = comparison(ag_policy, "primary_policy")
    comp_oracle = comparison(ag_oracle, "oracle")
    confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    false_positive_rows: List[Dict[str, Any]] = []
    catastrophic_rows: List[Dict[str, Any]] = []
    unsafe_rows: List[Dict[str, Any]] = []
    for r in prof_rows:
        pred_h = int(r.get("primary_policy_prediction"))
        pred_pos = pred_h == 10
        true_pos = bool(r.get("h10_beneficial_vs_h15"))
        if pred_pos and true_pos:
            confusion["TP"] += 1
        elif pred_pos and not true_pos:
            confusion["FP"] += 1
        elif (not pred_pos) and true_pos:
            confusion["FN"] += 1
        else:
            confusion["TN"] += 1
        h10 = r["median_by_h"]["10"]
        h15 = r["median_by_h"]["15"]
        phys_delta = None if h10.get("physical") is None or h15.get("physical") is None else float(h10["physical"] - h15["physical"])
        dec_delta = None if h10.get("decision_sum_s") is None or h15.get("decision_sum_s") is None else float(h10["decision_sum_s"] - h15["decision_sum_s"])
        if pred_pos and not h10.get("safe_all"):
            unsafe_rows.append({"base_state_id": r["base_state_id"], "reason": "predicted H10 not safe_all", "h10": h10, "h15": h15})
        if pred_pos and not true_pos:
            fp = {"base_state_id": r["base_state_id"], "fresh_confirmation_group": r.get("fresh_confirmation_group"), "h10_safe_all": h10.get("safe_all"), "h15_safe_all": h15.get("safe_all"), "physical_delta_h10_minus_h15": phys_delta, "decision_delta_h10_minus_h15": dec_delta, "row_tolerance": r.get("row_physical_tolerance_vs_H15"), "diagnostics": r.get("primary_policy_diagnostics")}
            false_positive_rows.append(fp)
            catastrophic_threshold = max(2.0, 0.25 * abs(sf(h15.get("physical"), 0.0)))
            if h15.get("safe_all") and ((not h10.get("safe_all")) or (phys_delta is not None and phys_delta > catastrophic_threshold)):
                fp["catastrophic_threshold"] = catastrophic_threshold
                catastrophic_rows.append(fp)
    fixed15_phys = ag_policy["fixed_H15"]["physical_sum"]
    physical_tol = float(math.fsum(sf(r.get("row_physical_tolerance_vs_H15"), 2.0) for r in prof_rows))
    decision_saving = comp_policy.get("decision_relative_saving_vs_fixed_H15")
    solver_saving = comp_policy.get("solver_relative_saving_vs_fixed_H15")
    h10_predictions = int(ag_policy["primary_policy"]["chosen_counts"].get("10", 0))
    h15_predictions = int(ag_policy["primary_policy"]["chosen_counts"].get("15", 0))
    nonconstant = h10_predictions > 0 and h15_predictions > 0
    no_bad_h10 = not unsafe_rows and not catastrophic_rows
    physical_gate = bool(comp_policy["physical_delta_vs_fixed_H15"] <= physical_tol)
    weak_pass = bool(nonconstant and no_bad_h10 and physical_gate and decision_saving is not None and float(decision_saving) >= MIN_DECISION_SAVING_WEAK)
    strong_pass = bool(weak_pass and decision_saving is not None and float(decision_saving) >= MIN_DECISION_SAVING_STRONG)
    return {
        "scope": profile,
        "groups": len(prof_rows),
        "h10_predictions": h10_predictions,
        "h15_predictions": h15_predictions,
        "nonconstant_horizons": nonconstant,
        "confusion_vs_fresh_h10_beneficial_label": confusion,
        "false_positive_rows": false_positive_rows,
        "catastrophic_false_positive_rows": catastrophic_rows,
        "unsafe_rows": unsafe_rows,
        "physical_tolerance_sum_vs_fixed_H15": physical_tol,
        "physical_gate": physical_gate,
        "fixed_H15_physical_sum": fixed15_phys,
        "comparison_policy_vs_fixed_H15": comp_policy,
        "comparison_oracle_vs_fixed_H15": comp_oracle,
        "oracle_chosen_counts": ag_oracle["oracle"]["chosen_counts"],
        "policy_chosen_counts": ag_policy["primary_policy"]["chosen_counts"],
        "decision_saving_gate_5pct": bool(decision_saving is not None and float(decision_saving) >= MIN_DECISION_SAVING_WEAK),
        "decision_saving_gate_10pct_strong": bool(decision_saving is not None and float(decision_saving) >= MIN_DECISION_SAVING_STRONG),
        "primary_pass_5pct": weak_pass,
        "primary_strong_10pct": strong_pass,
        "solver_saving_reported_not_primary": solver_saving,
    }


def analyze(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    built = build_state_profile_rows(branch_episodes, selected_states, protocol)
    rows = built["rows"]
    shared_gate = evaluate_policy_gate(rows, PRIMARY_TERMINAL_PROFILE)
    matched_gate = evaluate_policy_gate(rows, "matched_terminal")
    disagreements: List[Dict[str, Any]] = []
    for sid in sorted({r["base_state_id"] for r in rows}):
        labels = {r["terminal_profile"]: r.get("oracle_label") for r in rows if r["base_state_id"] == sid}
        vals = [v for v in labels.values() if v is not None]
        if len(set(vals)) > 1:
            r0 = next(r for r in rows if r["base_state_id"] == sid)
            disagreements.append({"base_state_id": sid, "oracle_labels_by_profile": labels, "fresh_confirmation_group": r0.get("fresh_confirmation_group")})
    if shared_gate["primary_strong_10pct"]:
        decision = "v1 independent fresh-source confirmation strongly supports the guard/veto true-H10/H15 selector; next freeze a validation64 plan with fair fixed true-H baselines and actual overhead accounting, after backup"
    elif shared_gate["primary_pass_5pct"]:
        decision = "v1 independent fresh-source confirmation gives only weak measured-decision saving; inspect timing uncertainty and consider value/objective/representation calibration before validation"
    elif shared_gate["catastrophic_false_positive_rows"] or shared_gate["unsafe_rows"] or not shared_gate["physical_gate"]:
        decision = "v1 independent fresh-source confirmation failed safety/physical gates; block selector rollout and prioritize value/objective/representation refit or training ablation"
    elif not shared_gate["nonconstant_horizons"]:
        decision = "v1 independent fresh-source guard collapsed to a constant horizon; block validation and prioritize representation/value/training diagnosis rather than more unchanged label sweeps"
    else:
        decision = "v1 independent fresh-source confirmation failed compute-saving gate; compare oracle opportunity and timing noise before any validation rollout"
    return {
        "source_summary": {"model": built["source_model"], "label_counts": built["source_counts"]},
        "state_profile_rows": rows,
        "unsafe_h15_groups": built["unsafe_h15_groups"],
        "terminal_profile_disagreements": disagreements,
        "primary_gate": shared_gate,
        "matched_terminal_gate": matched_gate,
        "decision": decision,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    g = a["primary_gate"]
    lines = [
        "# Vehicle true-variable-H fresh-source confirmation v1 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only independent fresh-source confirmation for `{PRIMARY_VARIANT}`; no validation64 bank, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Primary shared-H15-terminal guard/veto gate",
        "",
        f"- Groups: `{g['groups']}`; H10 predictions `{g['h10_predictions']}`, H15 abstentions `{g['h15_predictions']}`, nonconstant `{g['nonconstant_horizons']}`.",
        f"- Confusion vs fresh H10-beneficial labels: `{g['confusion_vs_fresh_h10_beneficial_label']}`.",
        f"- Physical gate: `{g['physical_gate']}`; policy comparison `{g['comparison_policy_vs_fixed_H15']}`; tolerance sum `{g['physical_tolerance_sum_vs_fixed_H15']}`.",
        f"- Decision saving gates: >=5% `{g['decision_saving_gate_5pct']}`, >=10% `{g['decision_saving_gate_10pct_strong']}`; solver saving `{g['solver_saving_reported_not_primary']}`.",
        f"- Unsafe H10 rows: `{len(g['unsafe_rows'])}`; catastrophic false positives: `{len(g['catastrophic_false_positive_rows'])}`.",
        f"- Weak pass: `{g['primary_pass_5pct']}`; strong pass: `{g['primary_strong_10pct']}`.",
        "",
        "## Oracle opportunity and matched-terminal diagnostic",
        "",
        f"- Shared-H15 oracle comparison: `{g['comparison_oracle_vs_fixed_H15']}`; chosen counts `{g['oracle_chosen_counts']}`.",
        f"- Matched-terminal policy gate: `{a['matched_terminal_gate']}`.",
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
        m10 = r["median_by_h"]["10"]
        m15 = r["median_by_h"]["15"]
        def fmt(v: Any) -> str:
            return "NA" if v is None else "%.6g" % float(v)
        lines.append("| `%s` | `%s` | %s | %s | `%s` | %s | %s | %s | %s |" % (r["base_state_id"], r.get("fresh_confirmation_group"), r.get("primary_policy_prediction"), r.get("oracle_label"), bool(r.get("h10_beneficial_vs_h15")), fmt(m10.get("physical")), fmt(m15.get("physical")), fmt(m10.get("decision_sum_s")), fmt(m15.get("decision_sum_s"))))
    if g["catastrophic_false_positive_rows"] or g["unsafe_rows"]:
        lines += ["", "## Failure anatomy", ""]
        for row in g["catastrophic_false_positive_rows"]:
            lines.append(f"- Catastrophic FP `{row['base_state_id']}`: phys delta {row.get('physical_delta_h10_minus_h15')}, decision delta {row.get('decision_delta_h10_minus_h15')}, tol {row.get('row_tolerance')}, threshold {row.get('catastrophic_threshold')}." )
        for row in g["unsafe_rows"]:
            lines.append(f"- Unsafe H10 `{row['base_state_id']}`: {row.get('reason')}.")
    lines += [
        "",
        "## Decision",
        "",
        str(a["decision"]),
        "",
        "Interpretation limits: development-only independent fresh-source confirmation; variant was selected after v0 fresh-source outcomes and is IMPROVED, not ORIGINAL SAC. Timing claims use measured whole-decision and solver timings, not nominal H. Final validation/test remain unopened.",
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
        if MARKER_RUN not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},development_guard_veto_fresh_source_confirmation,development_no_validation_no_test,136,0,0,0,0,False,{rel(RUN_DIR / 'completed.json')}\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", type=str, required=True)
    ap.add_argument("--i-accept-development-fresh-source-confirmation-v1", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_fresh_source_confirmation_v1:
        raise ContractError("requires --run and explicit v1 fresh-source confirmation acknowledgement")
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    protocol = verify_inputs()
    selected_cases = (protocol.get("case_source") or {}).get("selected_fresh_cases") or []
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    case_runner.SMOKE_DIR = RUN_DIR

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
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {
            "execution_index": i,
            "state_id": f"v1_stageA_fresh_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
            "case": i,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": 0,
            "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
            "true_mpc_n_horizon": 15,
            "commanded_horizon": 15,
            "terminal_mode": "matched_terminal",
            "initialization": "fresh_source_v1_stageA_trueH15_trace_from_initial_state",
        }
        summary = case_runner.run_true_h_episode(item, case, terminals[15])
        summary["stage"] = "A_h15_trace_scan"
        summary["fresh_case_index"] = i
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        stage_a_eps.append(summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "stage": "A", "episodes_done": len(episodes), "episodes_expected": 136, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("state_id", "true_mpc_n_horizon", "steps", "success", "termination")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    selected_states = fresh0.select_stage_a_states(stage_a_eps, protocol)
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
        unique_state_id = f"v1_B{j:03d}_{st['base_state_id']}_{profile}_r{repeat}_H{h}"
        item = {
            "execution_index": 2000 + j,
            "state_id": unique_state_id,
            "case": fc,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": int(st["branch_step"]),
            "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
            "true_mpc_n_horizon": h,
            "commanded_horizon": h,
            "terminal_mode": profile,
            "initialization": "fresh_source_v1_stageB_direct_branch_state_from_predeclared_H15_trace_manifest",
        }
        summary = case_runner.run_true_h_episode(item, case, terminal_for(profile, h, terminals))
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

    analysis = analyze(branch_episodes, selected_states, protocol)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_V1_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup v1 independent fresh-source confirmation runner/source/raw outputs before validation planning or further simulation", "backup_required_before_more_simulations": True, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(RUN_DIR), rel(STATE_RUN), rel(CONTINUE_STATE), rel(PROTOCOL), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_guard_veto_true_variable_H_fresh_source_confirmation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "primary_variant": PRIMARY_VARIANT,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {"episodes": len(episodes), "stage_A_h15_trace_episodes": len(stage_a_eps), "stage_B_branch_episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(episodes), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "stage_A_episodes": stage_a_eps,
        "selected_state_manifest": manifest,
        "branch_episodes": branch_episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development-only", "independent fresh source but not validation/test", "variant selected after v0 fresh outcomes", "not ORIGINAL SAC", "does not infer speed from H alone; measured decision/solver timing reported"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(f"""# Continue state after fresh-source confirmation v1 run

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.

Completed: `{NAME}` development run for `{PRIMARY_VARIANT}`: {len(episodes)} episodes, {control_steps} control steps. No validation64, no sealed test, no training/refit.

Primary gate: {analysis['primary_gate']}
Decision: {analysis['decision']}
Summary: `{rel(RUN_DIR / 'summary.md')}`
Raw: `{rel(RUN_DIR / 'raw.json')}`
Completed: `{rel(RUN_DIR / 'completed.json')}`
Backup request: `{rel(req)}`

Next action: backup this run before any additional simulation/training/refit. If strong pass, freeze validation64 with fair fixed true-H baselines and overhead accounting. If fail or weak-only, prioritize value/objective/representation refit or training ablation instead of another unchanged label-density sweep.
""", encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation v1 run

UTC: {created.isoformat()}. Development-only independent fresh-source confirmation completed for `{PRIMARY_VARIANT}`: {len(episodes)} episodes ({len(stage_a_eps)} H15 traces + {len(branch_episodes)} blocked H10/H15 branches), {control_steps} control steps. No validation64, no sealed test, no training/refit. Primary gate: {analysis['primary_gate']}. Decision: {analysis['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""")
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, FREEZE_DONE, GUARD_DONE, ORACLE_DONE, RISK_DONE, STATE_RUN, CONTINUE_STATE, req]
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
