#!/usr/bin/env python3
"""Freeze a source-independent true-variable-H risk-anchor acquisition protocol.

Development-only, no simulation, no validation64, no sealed test, no training
or refit.  This follows the selector-feasibility postdiagnostic: the oracle
bank showed real measured compute/control opportunity, but deployable
leave-one-state-out selector learning failed because conservative H25 labels
were supported by only one unique state and labels disagreed across terminal
profiles.  This script freezes the next discriminating acquisition protocol
from already-opened stress-v1 H15 traces without using any new non-H15 branch
outcomes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0"
STAMP = "20260929T0810Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_RISK_ANCHOR_ACQUISITION_FREEZE_V0_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SELECTOR_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0_20260929T0755Z/raw.json"
SELECTOR_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0_20260929T0755Z/completed.json"
ORACLE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_frozen_20260929T0725Z.json"
ORACLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"
STAGE1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/raw.json"
STAGE1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
STAGE1_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json"
STAGE1_POST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/completed.json"
COVERAGE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0_20260929T0435Z/raw.json"
COVERAGE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0_20260929T0435Z/completed.json"

TRUE_HORIZONS = [10, 15, 25]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
REPEATS = 2
TARGET_COUNT = 10
MAX_BRANCH_STEPS = 150
RNG_SEED = 202609290810
MIN_STEP_SEPARATION = 8


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def first_scalar(x: Any, default: float = 0.0) -> float:
    if isinstance(x, list) and x:
        return first_scalar(x[0], default)
    if isinstance(x, tuple) and x:
        return first_scalar(x[0], default)
    return sf(x, default)


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: " + rel(path))
    if obj.get("sealed_test_accessed") is not False and obj.get("sealed_test_bank_opened") is not False:
        raise ContractError("sealed test flag not false in " + rel(path))
    if obj.get("validation64_bank_opened") not in (False, None):
        raise ContractError("validation64 flag not false in " + rel(path))
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError("historical validation64 flag not false in " + rel(path))
    return obj


def require_inputs() -> Dict[str, str]:
    required = [SELECTOR_RAW, SELECTOR_DONE, ORACLE_PROTOCOL, ORACLE_RAW, STAGE1_RAW, STAGE1_DONE, STAGE1_POST_RAW, STAGE1_POST_DONE, COVERAGE_RAW, COVERAGE_DONE]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise ContractError("missing required development inputs: " + ", ".join(missing))
    for p in [SELECTOR_DONE, STAGE1_DONE, STAGE1_POST_DONE, COVERAGE_DONE]:
        completed_ok(p)
    return {rel(p): sha256(p) for p in required + [Path(__file__).resolve()]}


def selected_metadata(stage1_raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    metas = ((stage1_raw.get("bank_selection") or {}).get("selected_metadata") or [])
    if len(metas) < 20:
        raise ContractError("stage1 selected_metadata missing/too short")
    return metas


def post_rows(stage1_post: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {si(r.get("case")): r for r in (stage1_post.get("postdiagnostic_case_table") or []) if si(r.get("case")) >= 0}


def coverage_rows(coverage: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {si(r.get("case")): r for r in (coverage.get("case_coverage_rows") or []) if si(r.get("case")) >= 0}


def oracle_used_cases(protocol: Mapping[str, Any]) -> List[int]:
    cases = []
    for t in ((protocol.get("target_selection") or {}).get("targets") or []):
        c = si(t.get("case"), -1)
        if c >= 0:
            cases.append(c)
    return sorted(set(cases))


def trace_path_for(stage1_raw: Mapping[str, Any], case: int, horizon: int = 15) -> Path:
    rows = [e for e in (stage1_raw.get("episodes") or []) if si(e.get("case")) == case and si(e.get("horizon")) == horizon]
    if len(rows) != 1:
        raise ContractError(f"expected exactly one Stage1 H{horizon} episode for case={case}, got {len(rows)}")
    p = ROOT / str(rows[0].get("path"))
    if p.is_dir():
        p = p / "trace.json"
    if not p.exists():
        raise ContractError("missing trace: " + rel(p))
    return p


def obs_component(row: Mapping[str, Any], idx: int) -> float:
    obs = row.get("observation") or []
    try:
        return sf(obs[idx], 0.0)
    except Exception:
        return 0.0


def omega(row: Mapping[str, Any]) -> float:
    inp = row.get("input") or {}
    v = inp.get("u_omega") if isinstance(inp, Mapping) else None
    return first_scalar(v, 0.0)


def solver_iter_proxy(row: Mapping[str, Any]) -> float:
    attempts = ((row.get("recovery") or {}).get("attempts") or [])
    vals = [sf(a.get("iterations"), 0.0) for a in attempts if isinstance(a, Mapping)]
    return max(vals) if vals else 0.0


def state_xyz(row: Mapping[str, Any]) -> Dict[str, float]:
    s = row.get("previous_state") or row.get("state") or {}
    if not isinstance(s, Mapping):
        return {"x": 0.0, "y": 0.0, "theta": 0.0}
    return {"x": first_scalar(s.get("x"), 0.0), "y": first_scalar(s.get("y"), 0.0), "theta": first_scalar(s.get("theta"), 0.0)}


def risk_score(row: Mapping[str, Any], step: int, n: int) -> Tuple[float, Dict[str, float]]:
    perf = max(0.0, sf(row.get("performance"), 0.0))
    constraint = max(0.0, sf(row.get("constraint"), 0.0))
    st = state_xyz(row)
    phase = step / float(max(1, n - 1))
    heading_obs = abs(obs_component(row, 2))
    turn = abs(omega(row))
    iters = solver_iter_proxy(row)
    # Purely H15-trace features: favor high transient cost/heading, later/high-y
    # states similar to the lone H25-positive anchor, without using non-H15 outcomes.
    score = 8.0 * math.sqrt(perf) + 2.0 * heading_obs + 0.6 * turn + 0.04 * iters + 1.0 * min(10.0, constraint) + 0.6 * abs(st["y"]) + 0.5 * abs(st["theta"]) + 1.2 * phase
    return float(score), {"performance_step_cost": float(perf), "sqrt_performance": float(math.sqrt(perf)), "constraint_step_cost_capped": float(min(10.0, constraint)), "abs_heading_obs2": float(heading_obs), "abs_turn_u_omega": float(turn), "solver_iteration_proxy": float(iters), "abs_y": float(abs(st["y"])), "abs_theta": float(abs(st["theta"])), "phase_fraction": float(phase), "risk_score": float(score)}


def choose_step(trace: Sequence[Mapping[str, Any]], case: int, window: str, lo_frac: float, hi_frac: float, used_steps: set) -> Tuple[int, Mapping[str, Any], float, Dict[str, float]]:
    if len(trace) < 12:
        raise ContractError(f"trace too short for case {case}: {len(trace)}")
    lo = max(8, int(math.floor(lo_frac * len(trace))))
    hi = min(max(8, len(trace) - 3), 90, int(math.ceil(hi_frac * len(trace))))
    if hi < lo:
        lo, hi = 8, min(max(8, len(trace) - 3), 90)
    candidates = []
    for step in range(lo, hi + 1):
        if any(abs(step - u) < MIN_STEP_SEPARATION for u in used_steps):
            continue
        sc, comps = risk_score(trace[step], step, len(trace))
        candidates.append((sc, step, comps))
    if not candidates:
        for step in range(8, min(max(8, len(trace) - 3), 90) + 1):
            if step in used_steps:
                continue
            sc, comps = risk_score(trace[step], step, len(trace))
            candidates.append((sc, step, comps))
    if not candidates:
        raise ContractError(f"no branch-step candidates for case {case} window {window}")
    sc, step, comps = max(candidates, key=lambda x: (x[0], x[1]))
    used_steps.add(step)
    return step, trace[step], sc, comps


def material_info(case: int, post: Mapping[int, Mapping[str, Any]], cov: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    prow = dict(post.get(case, {})); crow = dict(cov.get(case, {}))
    mats = prow.get("material_strict_horizons_vs_H15") or crow.get("material_strict_horizons_vs_H15") or []
    return {"material": bool(mats), "physical_gain_ge3": bool(crow.get("episode_physical_gain_ge_3_case")), "runner_material": bool(crow.get("runner_episode_material")), "material_horizons": mats, "v1d_selected_target_count": si(crow.get("v1d_selected_target_count"), 0)}


def build_case_plan(metas: Sequence[Mapping[str, Any]], post: Mapping[int, Mapping[str, Any]], cov: Mapping[int, Mapping[str, Any]], used: Sequence[int]) -> List[Dict[str, Any]]:
    used_set = set(int(c) for c in used)
    case_rows = []
    for c in range(len(metas)):
        meta = metas[c]
        group = str(meta.get("selection_group") or meta.get("stratum") or post.get(c, {}).get("group") or "unknown")
        info = material_info(c, post, cov)
        case_rows.append({"case": c, "group": group, **info, "oracle_used_case": c in used_set, "candidate_index": si(meta.get("candidate_index"), -1)})
    plan: List[Dict[str, Any]] = []
    def add(case: int, role: str, windows: Sequence[Tuple[str, float, float]]) -> None:
        for w, lo, hi in windows:
            if len(plan) < TARGET_COUNT:
                plan.append({"case": int(case), "role": role, "window": w, "low_frac": lo, "high_frac": hi})
    fresh_material = [r for r in case_rows if (r["material"] or r["runner_material"] or r["physical_gain_ge3"]) and not r["oracle_used_case"]]
    fresh_material.sort(key=lambda r: (not r["physical_gain_ge3"], not r["material"], r["v1d_selected_target_count"], r["case"]))
    for r in fresh_material:
        if len(plan) >= 5:
            break
        if r["physical_gain_ge3"]:
            add(r["case"], "fresh_unseen_stage1_physical_gain_risk_anchor", [("early", 0.12, 0.32), ("middle", 0.36, 0.58), ("late", 0.60, 0.82)])
        else:
            add(r["case"], "fresh_unseen_stage1_runner_material_risk_anchor", [("middle", 0.30, 0.60), ("late", 0.58, 0.82)])
    high_controls = [r for r in case_rows if (not r["material"] and not r["runner_material"] and not r["physical_gain_ge3"] and not r["oracle_used_case"] and "high_heading_long_or_medium" in r["group"])]
    high_controls.sort(key=lambda r: r["case"])
    for r in high_controls[:2]:
        add(r["case"], "fresh_same_stratum_nonmaterial_specificity_control", [("middle", 0.25, 0.68)])
    other_controls = [r for r in case_rows if (not r["material"] and not r["runner_material"] and not r["physical_gain_ge3"] and not r["oracle_used_case"] and ("high_heading_short" in r["group"] or "low_heading" in r["group"]))]
    other_controls.sort(key=lambda r: r["case"])
    for r in other_controls[:2]:
        add(r["case"], "fresh_other_heading_specificity_control", [("middle", 0.25, 0.68)])
    lower_controls = [r for r in case_rows if (not r["oracle_used_case"] and "lower_stress_control" in r["group"])]
    lower_controls.sort(key=lambda r: r["case"])
    for r in lower_controls:
        if len(plan) >= TARGET_COUNT:
            break
        add(r["case"], "fresh_lower_stress_specificity_control", [("middle", 0.25, 0.68)])
    if len(plan) < TARGET_COUNT:
        fallback = [r for r in case_rows if not r["oracle_used_case"] and all(p["case"] != r["case"] for p in plan)]
        for r in fallback:
            if len(plan) >= TARGET_COUNT:
                break
            add(r["case"], "fresh_unfilled_fallback_h15_trace_state", [("middle", 0.25, 0.68)])
    if len(plan) != TARGET_COUNT:
        raise ContractError(f"could not build {TARGET_COUNT} fresh target plan; got {len(plan)}")
    return plan


def build_targets(stage1_raw: Mapping[str, Any], metas: Sequence[Mapping[str, Any]], post: Mapping[int, Mapping[str, Any]], cov: Mapping[int, Mapping[str, Any]], used_cases: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    plan = build_case_plan(metas, post, cov, used_cases)
    used_steps_by_case: Dict[int, set] = {}
    trace_cache: Dict[int, Tuple[Path, List[Mapping[str, Any]]]] = {}
    targets: List[Dict[str, Any]] = []
    for item in plan:
        case = int(item["case"])
        if case not in trace_cache:
            tp = trace_path_for(stage1_raw, case, 15)
            trace_cache[case] = (tp, read_json(tp))
            used_steps_by_case[case] = set()
        tp, trace = trace_cache[case]
        step, row, sc, comps = choose_step(trace, case, str(item["window"]), float(item["low_frac"]), float(item["high_frac"]), used_steps_by_case[case])
        meta = metas[case]
        info = material_info(case, post, cov)
        st = state_xyz(row)
        idx = len(targets)
        targets.append({
            "target_index": idx,
            "state_id": f"risk_anchor_v0_t{idx:02d}_case{case:02d}_cand{si(meta.get('candidate_index'), -1):03d}_b{step:03d}_{item['window']}",
            "case": case,
            "source_candidate_index": si(meta.get("candidate_index"), -1),
            "selection_group": meta.get("selection_group") or meta.get("stratum") or post.get(case, {}).get("group") or "unknown",
            "target_role": item["role"],
            "window": item["window"],
            "branch_step": int(step),
            "branch_previous_state": st,
            "source_h15_trace_path": rel(tp),
            "source_h15_trace_sha256": sha256(tp),
            "source_independent_from_oracle_bank_cases": case not in set(used_cases),
            "oracle_bank_used_cases_excluded_rule": sorted(set(int(c) for c in used_cases)),
            "h15_trace_selection_score": float(sc),
            "h15_trace_selection_score_components": comps,
            "stage1_material_info": info,
            "selection_rule": "chosen from existing stress-v1 Stage1 H15 trace only, before any new true-H branch outcomes; oracle-bank cases excluded for source independence",
        })
    return targets, {"plan": plan, "case_counts": {str(c): sum(1 for t in targets if int(t["case"]) == c) for c in sorted({int(t["case"]) for t in targets})}, "role_counts": {role: sum(1 for t in targets if str(t["target_role"]) == role) for role in sorted({str(t["target_role"]) for t in targets})}, "all_targets_source_independent_from_oracle_bank_cases": all(bool(t["source_independent_from_oracle_bank_cases"]) for t in targets)}


def build_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for t in targets:
            for profile in TERMINAL_PROFILES:
                hs = list(TRUE_HORIZONS)
                rng.shuffle(hs)
                for h in hs:
                    rows.append({"execution_index": exe, "repeat": rep, "state_id": t["state_id"], "target_index": int(t["target_index"]), "case": int(t["case"]), "source_candidate_index": int(t["source_candidate_index"]), "selection_group": t.get("selection_group"), "target_role": t.get("target_role"), "branch_step": int(t["branch_step"]), "terminal_profile": profile, "true_mpc_n_horizon": int(h), "blocked_randomization_unit": f"repeat{rep}|{t['state_id']}|{profile}"})
                    exe += 1
    return rows


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    d = raw["diagnosis"]
    prot = raw["protocol_snapshot"]
    lines = [
        "# Vehicle true-variable-H risk-anchor acquisition freeze v0",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata/protocol freeze only: no simulation, no training/refit, no validation64, no sealed test.",
        "",
        "## Why this diagnostic",
        "",
        "The oracle bank passed the measured compute/control opportunity gate, but the offline deployable selector gate failed.  The failure is informative rather than a reason to repeat the same label-density sweep: H25 risk labels were supported by one unique primary state, and multiple states changed label across terminal profiles.  The next acquisition must therefore be source-independent and explicitly test risk-anchor and terminal-consistency coverage before any selector refit.",
        "",
        "## Input evidence snapshot",
        "",
        f"- Selector accepted for fresh rollout: `{d['selector_accepted_for_fresh_rollout']}`.",
        f"- Primary label counts: `{d['primary_label_counts']}`; H25 unique primary states: `{d['h25_primary_unique_state_count']}`; H15 unique primary states: `{d['h15_primary_unique_state_count']}`.",
        f"- Terminal-profile label disagreements: `{d['profile_label_disagreement_count']}`.",
        f"- Oracle-bank used cases excluded: `{d['oracle_bank_used_cases']}`.",
        "",
        "## Frozen fresh target plan",
        "",
        f"- Targets: `{len(raw['targets'])}`; all source-independent from oracle-bank cases: `{raw['target_plan_diagnostics']['all_targets_source_independent_from_oracle_bank_cases']}`.",
        f"- Case counts: `{raw['target_plan_diagnostics']['case_counts']}`.",
        f"- Role counts: `{raw['target_plan_diagnostics']['role_counts']}`.",
        f"- Future schedule: `{prot['planned_development_branch_episodes_exact']}` branch episodes, cap `{prot['development_control_step_upper_bound']}` control steps; horizons `{TRUE_HORIZONS}`, profiles `{TERMINAL_PROFILES}`, repeats `{REPEATS}`.",
        "",
        "| target | case | role | branch | group | score | x | y | theta |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for t in raw["targets"]:
        st = t["branch_previous_state"]
        lines.append("| `%s` | %d | `%s` | %d | `%s` | %.6g | %.6g | %.6g | %.6g |" % (t["state_id"], int(t["case"]), t["target_role"], int(t["branch_step"]), t.get("selection_group"), sf(t.get("h15_trace_selection_score")), sf(st.get("x")), sf(st.get("y")), sf(st.get("theta"))))
    lines += [
        "",
        "## Future run acceptance (frozen now)",
        "",
        "Proceed to selector/value-model refit only if the future acquisition yields non-control H15/H25 risk anchors across at least two fresh cases, no safety/solver regressions, measured decision-time saving >=15% vs H25, and deployable leave-one-case/state-out rules are not absorbed by fixed H10/H15. If labels remain terminal-profile dependent or concentrated, pivot to terminal-value/objective/modeling repair or scenario design rather than training a sparse selector.",
        "",
        f"Protocol: `{raw['protocol']}`.",
        f"Backup request before any future simulation/training/refit: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--i-accept-development-risk-anchor-freeze-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.freeze or not args.i_accept_development_risk_anchor_freeze_v0:
        raise ContractError("requires --freeze and explicit development risk-anchor freeze acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "planned_episodes": done.get("planned_development_branch_episodes_exact"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    hashes = require_inputs()
    selector = read_json(SELECTOR_RAW)
    oracle_protocol = read_json(ORACLE_PROTOCOL)
    stage1_raw = read_json(STAGE1_RAW)
    stage1_post = read_json(STAGE1_POST_RAW)
    coverage = read_json(COVERAGE_RAW)
    ev = selector.get("evidence_diagnostics") or {}
    used_cases = oracle_used_cases(oracle_protocol)
    metas = selected_metadata(stage1_raw)
    post = post_rows(stage1_post)
    cov = coverage_rows(coverage)
    targets, target_diag = build_targets(stage1_raw, metas, post, cov, used_cases)
    schedule = build_schedule(targets)
    protocol_obj = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_H_risk_anchor_acquisition_freeze_no_simulation",
        "hypothesis": "If fresh stress-v1 H15-trace states outside the oracle-bank cases produce H15/H25 risk labels across multiple cases with consistent terminal-profile behavior, then a conservative deployable selector/value model may be learnable; if labels remain concentrated or terminal-dependent, terminal-value/objective/modeling or scenario design is the limiting factor.",
        "access_rules": {"development_only": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_external_backup_before_any_future_run": True},
        "before_evidence": {"selector_accepted_for_fresh_rollout": selector.get("selector_feasibility", {}).get("accepted_for_fresh_rollout"), "primary_label_counts": ev.get("primary_label_counts"), "h25_primary_unique_state_count": ev.get("h25_primary_unique_state_count"), "profile_label_disagreement_count": ev.get("profile_label_disagreement_count"), "oracle_bank_used_cases_excluded": used_cases},
        "target_selection": {"selection_source": "stress-v1 Stage1 H15 traces only; no new non-H15 outcomes", "targets_exact": len(targets), "diagnostics": target_diag, "targets": targets},
        "arms": {"true_horizons": TRUE_HORIZONS, "terminal_profiles": TERMINAL_PROFILES, "repeats": REPEATS, "blocked_randomization": "horizon order shuffled within repeat/state/profile block", "randomization_seed": RNG_SEED, "schedule": schedule},
        "budget_declared": {"planned_development_branch_episodes_exact": len(schedule), "development_control_step_upper_bound": len(schedule) * MAX_BRANCH_STEPS, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "acceptance_before_selector_or_value_refit": {"fresh_risk_anchor_requirement": "H15/H25 oracle labels in >=2 fresh non-control cases or a documented reason that all fresh risk states are safely H10", "terminal_consistency_requirement": "profile-disagreement rate reported; selector refit blocked if disagreements dominate labels without terminal-value/objective repair", "compute_requirement": ">=15% measured whole-decision saving vs fixed true H25, with solver timing also reported", "comparison_requirement": "compare against true fixed H10/H15/H25 and reject if fixed H10/H15 absorbs the tradeoff", "cv_requirement": "deployable leave-one-case/state-out selector gate before any closed-loop rollout"},
        "if_future_gate_fails": "pivot to terminal-value/objective/modeling repair or scenario-opportunity redesign; do not train on sparse anchor-dependent labels",
    }
    write_json(PROTOCOL, protocol_obj)
    write_json(BACKUP_REQ, {"requested_utc": created.isoformat(), "reason": "backup risk-anchor acquisition frozen protocol/source before any future development true-H branch simulation, selector refit, or training", "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "planned_development_branch_episodes_exact": len(schedule), "development_control_step_upper_bound": len(schedule) * MAX_BRANCH_STEPS, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(Path(__file__).resolve()), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQ)]})
    raw = {"created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "metadata_no_simulation_protocol_freeze", "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_actual": {"new_simulations": 0, "new_control_steps": 0, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0}, "input_hashes": hashes, "diagnosis": {"selector_accepted_for_fresh_rollout": selector.get("selector_feasibility", {}).get("accepted_for_fresh_rollout"), "primary_label_counts": ev.get("primary_label_counts"), "h25_primary_unique_state_count": ev.get("h25_primary_unique_state_count"), "h15_primary_unique_state_count": ev.get("h15_primary_unique_state_count"), "profile_label_disagreement_count": ev.get("profile_label_disagreement_count"), "oracle_bank_used_cases": used_cases}, "targets": targets, "target_plan_diagnostics": target_diag, "protocol": rel(PROTOCOL), "protocol_snapshot": protocol_obj["budget_declared"], "backup_request": rel(BACKUP_REQ), "next_experiment_after_backup": "implement/run the frozen source-independent true-variable-H risk-anchor acquisition bank, then repeat offline deployable CV before any selector training/refit"}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    marker = f"vehicle-true-variable-H-risk-anchor-acquisition-freeze-v0-{STAMP}"
    append_docs(f"""<!-- {marker} -->
## 2026-09-29 vehicle true-variable-H risk-anchor acquisition freeze v0

UTC: {created.isoformat()}. Metadata-only/no-simulation protocol freeze completed after selector feasibility failed. Selector accepted for fresh rollout: {selector.get('selector_feasibility', {}).get('accepted_for_fresh_rollout')}; primary labels {ev.get('primary_label_counts')}; H25 unique primary states {ev.get('h25_primary_unique_state_count')}; terminal-profile disagreements {ev.get('profile_label_disagreement_count')}. Frozen source-independent protocol `{rel(PROTOCOL)}` selects {len(targets)} stress-v1 H15-trace targets outside oracle-bank cases, with {len(schedule)} planned true-H branch episodes and cap {len(schedule) * MAX_BRANCH_STEPS} control steps. No validation64/test/training/refit. Further simulation/training/refit requires verified backup covering `{rel(BACKUP_REQ)}` and this new source/protocol.
""", marker)
    files = [Path(__file__).resolve(), PROTOCOL, OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, SELECTOR_RAW, SELECTOR_DONE, ORACLE_PROTOCOL, ORACLE_RAW, STAGE1_RAW, STAGE1_DONE, STAGE1_POST_RAW, STAGE1_POST_DONE, COVERAGE_RAW, COVERAGE_DONE]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "classification": "metadata_no_simulation_protocol_freeze", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "planned_development_branch_episodes_exact": len(schedule), "development_control_step_upper_bound": len(schedule) * MAX_BRANCH_STEPS, "target_count": len(targets), "all_targets_source_independent_from_oracle_bank_cases": target_diag["all_targets_source_independent_from_oracle_bank_cases"], "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "hashes": {rel(p): sha256(p) for p in files if p.exists()}}
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "target_count": len(targets), "planned_episodes": len(schedule), "all_targets_source_independent_from_oracle_bank_cases": target_diag["all_targets_source_independent_from_oracle_bank_cases"], "new_simulations": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
