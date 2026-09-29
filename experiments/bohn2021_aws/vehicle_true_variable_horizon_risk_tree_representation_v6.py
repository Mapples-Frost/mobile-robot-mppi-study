#!/usr/bin/env python3
"""Cost-sensitive nonlinear risk-representation diagnostic for true-variable-H H10/H15.

Development-only IMPROVED diagnostic.  It consumes only already-opened fresh-source
banks v0/v1/v2 and asks whether a bounded nonlinear risk-value policy class
(shallow cost-sensitive trees over online/deployable features, plus explicit
scenario-metadata diagnostic controls) can detect intrinsic H10 risk without
collapsing compute savings.  This is a representation/policy-class diagnostic,
not another nearest-neighbour label-density sweep.

No validation64 bank, sealed test, MPC simulation, gradient training, or
checkpoint writing is performed.
"""
from __future__ import annotations

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
NAME = "vehicle_true_variable_horizon_risk_tree_representation_v6"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_risk_tree_representation_v6.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-risk-tree-representation-v6-{STAMP}"
BANK_DIRS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z",
}
PROFILE = "shared_h15_terminal"
MIN_ROW_TOL = 2.0

class ContractError(RuntimeError):
    pass


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
    return x


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def med_h(row: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    med = row.get("median_by_h") or {}
    return med.get(str(h)) or med.get(h) or {}


def assert_dev_only(raw: Mapping[str, Any], done: Mapping[str, Any], bank: str) -> None:
    for name, obj in (("raw", raw), ("completed", done)):
        for flag in ("sealed_test_accessed", "sealed_test_bank_opened", "test_accessed", "validation64_bank_opened"):
            if obj.get(flag) is True:
                raise ContractError(f"forbidden {flag}=True in {bank} {name}")
    if done.get("passed") is not True and done.get("hard_pass") is not True and done.get("status") not in ("complete", "completed"):
        raise ContractError(f"bank did not complete cleanly: {bank}")


def load_manifest(raw: Mapping[str, Any], path: Path) -> List[Mapping[str, Any]]:
    man = raw.get("selected_state_manifest")
    if isinstance(man, Mapping) and isinstance(man.get("selected_states"), list):
        return list(man["selected_states"])
    if isinstance(man, list):
        return list(man)
    if path.exists():
        obj = read_json(path)
        if isinstance(obj, Mapping):
            return list(obj.get("selected_states") or [])
        if isinstance(obj, list):
            return list(obj)
    raise ContractError("missing selected_state_manifest: " + rel(path))


def obs14(st: Mapping[str, Any]) -> List[float]:
    raw = st.get("initial_observation_from_h15_trace") or st.get("initial_observation_at_branch") or []
    vals = [sf(v) for v in raw[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    rows: List[Dict[str, Any]] = []
    hashes = {rel(Path(__file__)): sha256(Path(__file__))}
    groups = set()
    for bank, d in BANK_DIRS.items():
        raw_path = d / "raw.json"; done_path = d / "completed.json"; man_path = d / "selected_state_manifest.json"
        raw = read_json(raw_path); done = read_json(done_path)
        assert_dev_only(raw, done, bank)
        hashes[rel(raw_path)] = sha256(raw_path); hashes[rel(done_path)] = sha256(done_path)
        if man_path.exists():
            hashes[rel(man_path)] = sha256(man_path)
        by_base = {str(s.get("base_state_id")): s for s in load_manifest(raw, man_path)}
        for r in ((raw.get("analysis") or {}).get("state_profile_rows") or []):
            if str(r.get("terminal_profile")) != PROFILE:
                continue
            base = str(r.get("base_state_id") or r.get("state_id"))
            st = by_base.get(base, {})
            h10 = med_h(r, 10); h15 = med_h(r, 15)
            tol = max(MIN_ROW_TOL, sf(r.get("row_physical_tolerance_vs_H15", r.get("row_physical_tolerance")), MIN_ROW_TOL))
            h10_phys = sf(h10.get("physical")); h15_phys = sf(h15.get("physical"))
            h10_dec = sf(h10.get("decision_sum_s")); h15_dec = sf(h15.get("decision_sum_s"))
            h10_sol = sf(h10.get("solver_sum_s")); h15_sol = sf(h15.get("solver_sum_s"))
            h10_safe = bool(h10.get("safe_all")); h15_safe = bool(h15.get("safe_all"))
            phys_delta = h10_phys - h15_phys
            decision_gain = h15_dec - h10_dec
            beneficial = bool(h10_safe and h15_safe and phys_delta <= tol and decision_gain > 0.0)
            catastrophic = bool((not h10_safe) or (h15_safe and phys_delta > tol))
            prev = st.get("branch_previous_state") if isinstance(st.get("branch_previous_state"), Mapping) else {}
            group = str(r.get("fresh_confirmation_group") or st.get("fresh_confirmation_group") or "unknown")
            window = str(st.get("window") or ("early_mid" if si(r.get("branch_state_slot", st.get("branch_state_slot")), 0) == 0 else "mid_late"))
            groups.add(group); groups.add(f"window::{window}"); groups.add(f"gw::{group}::{window}")
            rows.append({
                "bank_id": bank, "base_state_id": base, "group": group, "window": window,
                "branch_state_slot": si(r.get("branch_state_slot", st.get("branch_state_slot")), 0),
                "branch_step": si(st.get("branch_step"), -1),
                "obs14": obs14(st), "prev_x": sf(prev.get("x")), "prev_y": sf(prev.get("y")), "prev_theta": sf(prev.get("theta")),
                "stage_a_trace_risk_score": sf(st.get("stage_a_trace_risk_score"), 0.0),
                "h10_physical": h10_phys, "h15_physical": h15_phys,
                "h10_decision_sum_s": h10_dec, "h15_decision_sum_s": h15_dec,
                "h10_solver_sum_s": h10_sol, "h15_solver_sum_s": h15_sol,
                "h10_safe_all": h10_safe, "h15_safe_all": h15_safe,
                "row_physical_tolerance": tol, "phys_delta_h10_minus_h15": phys_delta,
                "decision_gain_h10_vs_h15_s": decision_gain, "solver_gain_h10_vs_h15_s": h15_sol - h10_sol,
                "h10_beneficial_vs_h15": beneficial, "h10_catastrophic_vs_h15": catastrophic,
            })
    if len(rows) != 48:
        raise ContractError(f"expected 48 primary rows, got {len(rows)}")
    return rows, hashes, sorted(groups)


def feature_dict(row: Mapping[str, Any], all_groups: Sequence[str]) -> Dict[str, float]:
    o = [sf(v) for v in (row.get("obs14") or [])[:14]]
    while len(o) < 14:
        o.append(0.0)
    d: Dict[str, float] = {}
    for i, v in enumerate(o):
        d[f"obs_{i}"] = v; d[f"abs_obs_{i}"] = abs(v)
    d["prev_x"] = sf(row.get("prev_x")) / 30.0; d["prev_y"] = sf(row.get("prev_y")) / 30.0
    th = sf(row.get("prev_theta")); d["theta_sin"] = math.sin(th); d["theta_cos"] = math.cos(th); d["abs_theta"] = abs(th) / math.pi
    d["branch_step"] = max(0.0, sf(row.get("branch_step"))) / 150.0; d["slot"] = sf(row.get("branch_state_slot"))
    d["risk_score"] = sf(row.get("stage_a_trace_risk_score")) / 20.0
    d["obs01_norm"] = math.sqrt(o[0]*o[0] + o[1]*o[1]); d["obs56_norm"] = math.sqrt(o[5]*o[5] + o[6]*o[6])
    for g in all_groups:
        if g.startswith("window::"):
            d[g] = 1.0 if g == f"window::{row.get('window')}" else 0.0
        elif g.startswith("gw::"):
            d[g] = 1.0 if g == f"gw::{row.get('group')}::{row.get('window')}" else 0.0
        else:
            d[f"group::{g}"] = 1.0 if g == row.get("group") else 0.0
    return d


def feature_names(family: str, all_groups: Sequence[str]) -> List[str]:
    obs = [f"obs_{i}" for i in range(14)]
    absobs = [f"abs_obs_{i}" for i in range(14)]
    pose = ["prev_x", "prev_y", "theta_sin", "theta_cos", "abs_theta", "branch_step", "slot"]
    risk = ["risk_score", "obs01_norm", "obs56_norm"]
    group = [f"group::{g}" for g in all_groups if not g.startswith("window::") and not g.startswith("gw::")]
    win = [g for g in all_groups if g.startswith("window::")]
    gw = [g for g in all_groups if g.startswith("gw::")]
    if family == "deploy_obs_pose_step_no_risk": return obs + absobs + pose
    if family == "deploy_obs_pose_step_risk": return obs + absobs + pose + risk
    if family == "scenario_group_window_only": return group + win + gw
    if family == "deploy_plus_group_window": return obs + absobs + pose + risk + group + win + gw
    if family == "step_window_only": return ["branch_step", "slot"] + win
    raise ContractError("unknown feature family " + family)


def leaf_decision(train_rows: Sequence[Mapping[str, Any]], min_pos: int, max_cat: int, min_gain: float) -> Dict[str, Any]:
    pos = sum(1 for r in train_rows if r["h10_beneficial_vs_h15"])
    cat = sum(1 for r in train_rows if r["h10_catastrophic_vs_h15"])
    gain = sum(sf(r["decision_gain_h10_vs_h15_s"]) for r in train_rows)
    phys = sum(sf(r["phys_delta_h10_minus_h15"]) for r in train_rows)
    tol = sum(sf(r["row_physical_tolerance"]) for r in train_rows)
    choose = bool(pos >= min_pos and cat <= max_cat and gain >= min_gain and phys <= tol)
    return {"n": len(train_rows), "pos": pos, "cat": cat, "gain_sum": gain, "phys_delta_sum": phys, "tol_sum": tol, "choose_h10": choose}


def leaf_score(train_rows: Sequence[Mapping[str, Any]], min_pos: int, max_cat: int, min_gain: float) -> float:
    ld = leaf_decision(train_rows, min_pos, max_cat, min_gain)
    if not ld["choose_h10"]:
        return 0.0
    # Reward measured decision gain; penalize physical degradation even if inside tolerance.
    return sf(ld["gain_sum"]) - 0.05 * max(0.0, sf(ld["phys_delta_sum"])) - 1000.0 * max(0, int(ld["cat"]) - max_cat)


def candidate_thresholds(rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Dict[str, float]], names: Sequence[str]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for nm in names:
        vals = sorted(set(sf(fdicts[str(r["bank_id"])+"/"+str(r["base_state_id"])].get(nm)) for r in rows))
        if len(vals) <= 1:
            continue
        mids = [(vals[i] + vals[i+1]) / 2.0 for i in range(len(vals)-1)]
        if len(mids) > 18:
            idxs = sorted(set(int(round(k * (len(mids)-1) / 17.0)) for k in range(18)))
            mids = [mids[i] for i in idxs]
        out.extend((nm, m) for m in mids)
    return out


def row_key(r: Mapping[str, Any]) -> str:
    return str(r["bank_id"]) + "/" + str(r["base_state_id"])


def build_tree(rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Dict[str, float]], names: Sequence[str], depth_left: int, min_leaf: int, min_pos: int, max_cat: int, min_gain: float) -> Dict[str, Any]:
    rows = list(rows)
    base = leaf_score(rows, min_pos, max_cat, min_gain)
    if depth_left <= 0 or len(rows) < 2 * min_leaf:
        return {"leaf": True, **leaf_decision(rows, min_pos, max_cat, min_gain)}
    best: Optional[Tuple[float, str, float, List[Mapping[str, Any]], List[Mapping[str, Any]]]] = None
    for nm, thr in candidate_thresholds(rows, fdicts, names):
        left = [r for r in rows if sf(fdicts[row_key(r)].get(nm)) <= thr]
        right = [r for r in rows if sf(fdicts[row_key(r)].get(nm)) > thr]
        if len(left) < min_leaf or len(right) < min_leaf:
            continue
        score = leaf_score(left, min_pos, max_cat, min_gain) + leaf_score(right, min_pos, max_cat, min_gain)
        if best is None or score > best[0] + 1e-12:
            best = (score, nm, thr, left, right)
    if best is None or best[0] <= base + 1e-12:
        return {"leaf": True, **leaf_decision(rows, min_pos, max_cat, min_gain)}
    _, nm, thr, left, right = best
    return {"leaf": False, "feature": nm, "threshold": thr, "n": len(rows),
            "left": build_tree(left, fdicts, names, depth_left-1, min_leaf, min_pos, max_cat, min_gain),
            "right": build_tree(right, fdicts, names, depth_left-1, min_leaf, min_pos, max_cat, min_gain)}


def predict(tree: Mapping[str, Any], fd: Mapping[str, float]) -> int:
    node = tree
    guard = 0
    while not node.get("leaf") and guard < 20:
        guard += 1
        node = node["left"] if sf(fd.get(str(node["feature"]))) <= sf(node["threshold"]) else node["right"]
    return 10 if node.get("choose_h10") else 15


def eval_choices(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int]) -> Dict[str, Any]:
    fixed_phys = sum(sf(r["h15_physical"]) for r in rows); fixed_dec = sum(sf(r["h15_decision_sum_s"]) for r in rows); fixed_sol = sum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = sum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0; counts = Counter(); conf = Counter(); bad = []; details = []
    for r in rows:
        h = int(choices.get(row_key(r), 15)); counts[str(h)] += 1
        pos = bool(r["h10_beneficial_vs_h15"])
        if h == 10:
            pol_phys += sf(r["h10_physical"]); pol_dec += sf(r["h10_decision_sum_s"]); pol_sol += sf(r["h10_solver_sum_s"])
            conf["TP" if pos else "FP"] += 1
            if bool(r["h10_catastrophic_vs_h15"]):
                bad.append({"bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "group": r["group"], "window": r["window"], "phys_delta": sf(r["phys_delta_h10_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"]), "tol": sf(r["row_physical_tolerance"])})
        else:
            pol_phys += sf(r["h15_physical"]); pol_dec += sf(r["h15_decision_sum_s"]); pol_sol += sf(r["h15_solver_sum_s"])
            conf["FN" if pos else "TN"] += 1
        details.append({"bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "selected_h": h, "label_positive": pos, "catastrophic": bool(r["h10_catastrophic_vs_h15"]), "group": r["group"], "window": r["window"], "phys_delta": sf(r["phys_delta_h10_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"])})
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    phys_delta = pol_phys - fixed_phys
    return {"groups": len(rows), "chosen_counts": dict(counts), "confusion": dict(conf), "physical_delta_vs_fixed_H15": phys_delta, "physical_tolerance_sum": tol_sum, "physical_gate": phys_delta <= tol_sum, "decision_relative_saving_vs_fixed_H15": dec_save, "solver_relative_saving_vs_fixed_H15": sol_save, "catastrophic_false_positive_rows": bad, "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad), "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad), "details": details}


def compact_tree(t: Mapping[str, Any], depth: int = 0) -> Any:
    if t.get("leaf"):
        return {"leaf": True, "n": t.get("n"), "pos": t.get("pos"), "cat": t.get("cat"), "choose_h10": t.get("choose_h10"), "gain_sum": t.get("gain_sum"), "phys_delta_sum": t.get("phys_delta_sum")}
    return {"feature": t.get("feature"), "threshold": t.get("threshold"), "n": t.get("n"), "left": compact_tree(t.get("left", {}), depth+1), "right": compact_tree(t.get("right", {}), depth+1)}


def pct(x: Any) -> str:
    return f"{100.0*sf(x):.2f}%"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hashes, all_groups = load_rows()
    fdicts = {row_key(r): feature_dict(r, all_groups) for r in rows}
    banks = sorted(BANK_DIRS)
    by_bank = {b: [r for r in rows if r["bank_id"] == b] for b in banks}
    families = ["deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk", "step_window_only", "scenario_group_window_only", "deploy_plus_group_window"]
    variants = []
    tree_fits = 0
    for fam in families:
        names = feature_names(fam, all_groups)
        for depth in [1, 2, 3]:
            for min_leaf in [2, 3, 4]:
                for min_pos in [1, 2, 3]:
                    for max_cat in [0, 1]:
                        for min_gain in [0.0, 0.5, 1.0]:
                            spec = {"family": fam, "max_depth": depth, "min_leaf": min_leaf, "min_pos_leaf": min_pos, "max_train_cat_leaf": max_cat, "min_leaf_gain_s": min_gain}
                            hold = {}; trees = {}; total_bad = 0; phys_fail = 0; total_h10 = 0; min_save = 1e9; avg_save = 0.0; strong = True; weak = True
                            for hb in banks:
                                train = [r for r in rows if r["bank_id"] != hb]
                                tree = build_tree(train, fdicts, names, depth, min_leaf, min_pos, max_cat, min_gain)
                                tree_fits += 1
                                choices = {row_key(r): predict(tree, fdicts[row_key(r)]) for r in by_bank[hb]}
                                ev = eval_choices(by_bank[hb], choices)
                                hold[hb] = ev; trees[hb] = compact_tree(tree)
                                total_bad += len(ev["catastrophic_false_positive_rows"]); phys_fail += 0 if ev["physical_gate"] else 1
                                total_h10 += int(ev["chosen_counts"].get("10", 0)); min_save = min(min_save, sf(ev["decision_relative_saving_vs_fixed_H15"])); avg_save += sf(ev["decision_relative_saving_vs_fixed_H15"])
                                strong = strong and bool(ev["pass_10pct_no_cat_fp"]); weak = weak and bool(ev["pass_5pct_no_cat_fp"])
                            avg_save /= len(banks)
                            variants.append({"variant_id": f"v6_tree_{fam}_d{depth}_ml{min_leaf}_mp{min_pos}_mc{max_cat}_mg{min_gain:g}", "spec": spec, "holdout_evaluations": hold, "trees_by_holdout": trees, "all_holdouts_strong": strong, "all_holdouts_weak": weak, "total_catastrophic_fp": total_bad, "physical_gate_fail_count": phys_fail, "min_decision_saving": min_save, "avg_decision_saving": avg_save, "total_h10": total_h10, "feature_count": len(names)})
    def rank_key(c: Mapping[str, Any]) -> Tuple[Any, ...]:
        pure_deploy = c["spec"]["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk")
        return (not bool(c["all_holdouts_strong"]), not bool(c["all_holdouts_weak"]), int(c["total_catastrophic_fp"]), int(c["physical_gate_fail_count"]), not pure_deploy, -sf(c["min_decision_saving"]), -sf(c["avg_decision_saving"]), -int(c["total_h10"]))
    ranked = sorted(variants, key=rank_key)
    top = ranked[:20]
    deploy_only = [v for v in variants if v["spec"]["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk")]
    scenario = [v for v in variants if v["spec"]["family"] in ("scenario_group_window_only", "deploy_plus_group_window", "step_window_only")]
    best = ranked[0]
    best_deploy = sorted(deploy_only, key=rank_key)[0]
    best_scenario = sorted(scenario, key=rank_key)[0]
    headline = {"rows": len(rows), "tree_variants": len(variants), "tree_fits": tree_fits, "strong_candidates_total": sum(1 for v in variants if v["all_holdouts_strong"]), "weak_candidates_total": sum(1 for v in variants if v["all_holdouts_weak"]), "deploy_strong_candidates": sum(1 for v in deploy_only if v["all_holdouts_strong"]), "deploy_weak_candidates": sum(1 for v in deploy_only if v["all_holdouts_weak"]), "scenario_or_hybrid_strong_candidates": sum(1 for v in scenario if v["all_holdouts_strong"]), "best_variant": best["variant_id"], "best_deploy_variant": best_deploy["variant_id"], "best_deploy_min_save": best_deploy["min_decision_saving"], "best_deploy_bad": best_deploy["total_catastrophic_fp"], "best_scenario_variant": best_scenario["variant_id"], "best_scenario_min_save": best_scenario["min_decision_saving"], "best_scenario_bad": best_scenario["total_catastrophic_fp"]}
    if headline["deploy_strong_candidates"] > 0:
        decision = "Deployable nonlinear risk-tree representation can meet opened-bank strong gates; next freeze an unused fresh-source confirmation for the best pure-deployable tree with actual closed-loop selector overhead before any validation64."
    elif headline["scenario_or_hybrid_strong_candidates"] > 0 and headline["deploy_strong_candidates"] == 0:
        decision = "Scenario/window metadata can separate the opened risk modes but pure online deployable features cannot; next add/learn explicit risk-state representation or acquire targeted risk data, not a terminal-only refit or unchanged selector sweep."
    elif best_deploy["total_catastrophic_fp"] == 0 and best_deploy["min_decision_saving"] >= 0.05:
        decision = "Pure deployable tree reaches only weak opened-bank safety/compute tradeoff; next run a small fresh-source confirmation or augment representation only if the weak tradeoff is worth testing against overhead."
    else:
        decision = "Even nonlinear deployable trees fail opened-bank weak/strong gates; next bounded intervention should be targeted risk-data acquisition or value-function training/refit with richer state history/clearance features, not another threshold/NN selector sweep."
    raw = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "classification": "development_IMPROVED_opened_fresh_bank_risk_tree_representation_no_simulation_no_validation_no_test", "hypothesis": "Intrinsic H10 risk may be predictable only with richer nonlinear or scenario-aware state representation. If pure deployable shallow trees pass LOBO gates, representation/policy class is sufficient; if only scenario/hybrid passes, online features are missing risk information; if none pass, target data acquisition/value training.", "budgets": {"new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "gradient_steps": 0, "closed_form_tree_fits": tree_fits, "validation64_episodes": 0, "sealed_test_episodes": 0}, "headline": headline, "top_variants": top, "best_deployable_variant": best_deploy, "best_scenario_or_hybrid_variant": best_scenario, "bank_label_summary": {b: {"rows": len(by_bank[b]), "positive": sum(1 for r in by_bank[b] if r["h10_beneficial_vs_h15"]), "catastrophic": sum(1 for r in by_bank[b] if r["h10_catastrophic_vs_h15"])} for b in banks}, "hashes": hashes, "access_flags": {"sealed_test_accessed": False, "validation64_bank_opened": False, "mobile_robot_mppi_resumed": False}, "decision": decision, "platform": {"python": sys.version, "platform": platform.platform()}}
    raw_path = OUT / "raw.json"; summary_path = OUT / "summary.md"; done_path = OUT / "completed.json"
    write_json(raw_path, raw)
    lines = ["# Vehicle true-variable-H risk-tree representation v6", "", f"UTC: `{raw['created_utc']}`. Development-only opened-bank nonlinear risk-representation diagnostic; no simulations, no validation64, no sealed test, no gradient training.", "", "## Headline", "", f"- Rows `{headline['rows']}`; tree variants `{headline['tree_variants']}`; leave-bank tree fits `{headline['tree_fits']}`.", f"- Strong candidates total `{headline['strong_candidates_total']}`, weak `{headline['weak_candidates_total']}`; pure deployable strong `{headline['deploy_strong_candidates']}`, weak `{headline['deploy_weak_candidates']}`; scenario/hybrid strong `{headline['scenario_or_hybrid_strong_candidates']}`.", f"- Best overall `{headline['best_variant']}`; best deployable `{headline['best_deploy_variant']}` min save `{pct(headline['best_deploy_min_save'])}`, bad `{headline['best_deploy_bad']}`; best scenario/hybrid `{headline['best_scenario_variant']}` min save `{pct(headline['best_scenario_min_save'])}`, bad `{headline['best_scenario_bad']}`.", "", "## Top variants", "", "| rank | family | strong | weak | bad | phys fails | min save | avg save | H10 total | variant |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i, v in enumerate(top[:15], 1):
        lines.append(f"| {i} | `{v['spec']['family']}` | `{v['all_holdouts_strong']}` | `{v['all_holdouts_weak']}` | {v['total_catastrophic_fp']} | {v['physical_gate_fail_count']} | {pct(v['min_decision_saving'])} | {pct(v['avg_decision_saving'])} | {v['total_h10']} | `{v['variant_id']}` |")
    lines += ["", "## Best deployable held-out banks", "", "| bank | H counts | confusion | bad | phys Δ/tol | decision save | solver save | pass10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b, ev in best_deploy["holdout_evaluations"].items():
        lines.append(f"| `{b}` | `{ev['chosen_counts']}` | `{ev['confusion']}` | {len(ev['catastrophic_false_positive_rows'])} | {ev['physical_delta_vs_fixed_H15']:.4g}/{ev['physical_tolerance_sum']:.4g} | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} | `{ev['pass_10pct_no_cat_fp']}` |")
    lines += ["", "## Best scenario/hybrid held-out banks", "", "| bank | H counts | confusion | bad | phys Δ/tol | decision save | solver save | pass10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b, ev in best_scenario["holdout_evaluations"].items():
        lines.append(f"| `{b}` | `{ev['chosen_counts']}` | `{ev['confusion']}` | {len(ev['catastrophic_false_positive_rows'])} | {ev['physical_delta_vs_fixed_H15']:.4g}/{ev['physical_tolerance_sum']:.4g} | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} | `{ev['pass_10pct_no_cat_fp']}` |")
    lines += ["", "## Decision", "", decision, "", "Interpretation: opened-development evidence only. A positive tree result would still need unused fresh-source confirmation with actual selector overhead before validation; a negative result supports targeted risk representation/value training or risk-state acquisition. This remains IMPROVED, not ORIGINAL SAC reproduction."]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    backup_req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_RISK_TREE_REPRESENTATION_V6_{STAMP}.json"
    write_json(backup_req, {"created_utc": raw["created_utc"], "reason": "backup after risk-tree representation v6 diagnostic", "paths": [rel(raw_path), rel(summary_path), rel(done_path), rel(STATE)], "sealed_test_accessed": False, "validation64_bank_opened": False})
    done = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(summary_path), "raw": rel(raw_path), "backup_request": rel(backup_req), "headline": headline, "decision": decision, "sealed_test_accessed": False, "validation64_bank_opened": False, "new_simulations": 0, "gradient_steps": 0, "closed_form_tree_fits": tree_fits, "hashes": {rel(Path(__file__)): sha256(Path(__file__)), rel(raw_path): sha256(raw_path), rel(summary_path): sha256(summary_path)}}
    write_json(done_path, done)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after {NAME}\n\nUTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(headline), sort_keys=True)}\n\nDecision: {decision}\n\nBudgets: no simulations/control steps, no validation64, no sealed test, no gradient training; closed_form_tree_fits={tree_fits}.\n\nArtifacts: {rel(summary_path)}, {rel(raw_path)}, {rel(done_path)}.\n", encoding="utf-8")
    audit_line = f"\n- {raw['created_utc']} `{NAME}`: nonlinear risk-tree representation diagnostic; deploy_strong={headline['deploy_strong_candidates']}, deploy_weak={headline['deploy_weak_candidates']}, best_deploy_min_save={headline['best_deploy_min_save']:.4f}, best_deploy_bad={headline['best_deploy_bad']}; no simulation/validation/test; artifacts `{rel(summary_path)}`.\n"
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + audit_line, encoding="utf-8")
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8") as f:
        f.write(f"{STAMP},{NAME},development_risk_tree_representation,no_simulation_no_rng,opened_fresh_v0_v1_v2_no_validation_no_test,0,0,{tree_fits},0,0,False,{rel(done_path)},{MARKER}\n")
    print("SUMMARY " + rel(summary_path))
    print(json.dumps({"headline": headline, "decision": decision, "best_deploy_holdouts": {b: {"save": ev["decision_relative_saving_vs_fixed_H15"], "bad": len(ev["catastrophic_false_positive_rows"]), "h_counts": ev["chosen_counts"], "pass10": ev["pass_10pct_no_cat_fp"]} for b, ev in best_deploy["holdout_evaluations"].items()}, "best_scenario_holdouts": {b: {"save": ev["decision_relative_saving_vs_fixed_H15"], "bad": len(ev["catastrophic_false_positive_rows"]), "h_counts": ev["chosen_counts"], "pass10": ev["pass_10pct_no_cat_fp"]} for b, ev in best_scenario["holdout_evaluations"].items()}}, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "completed.json", {"passed": False, "status": "failed", "marker": MARKER, "error": type(exc).__name__, "message": str(exc), "sealed_test_accessed": False, "validation64_bank_opened": False})
        raise
