#!/usr/bin/env python3
"""Offline selector-feasibility diagnostic for the true-variable-H oracle bank.

Development-only, no simulation, no validation64, no sealed test, no gradient
training, and no persistent selector refit.  The diagnostic asks whether the
oracle-bank labels are learnable by low-capacity/deployable selectors under a
leave-one-state-out test, or whether the apparent oracle opportunity is too
sparse/anchor-dependent to justify closed-loop selector training yet.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0"
STAMP = "20260929T0755Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_frozen_20260929T0725Z.json"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"{NAME}-{STAMP}"
HORIZONS = [10, 15, 25]
MIN_DECISION_SAVING = 0.15


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


def completed_access_ok() -> None:
    for path in (RAW, ORACLE_DONE, PROTOCOL):
        if not path.exists():
            raise ContractError("missing required input: " + rel(path))
    done = read_json(ORACLE_DONE)
    if done.get("hard_pass") is not True and done.get("passed") is not True:
        raise ContractError("oracle bank completed marker did not pass")
    if done.get("validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise ContractError("oracle bank access flags invalid")
    raw = read_json(RAW)
    if raw.get("validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("raw oracle access flags invalid")


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(v for v in (sf(x, float("nan")) for x in xs) if math.isfinite(v))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(v for v in (sf(x, float("nan")) for x in xs) if math.isfinite(v))
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * q
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {"n": len(vals), "min": vals[0], "median": pct(0.5), "mean": math.fsum(vals) / len(vals), "p95": pct(0.95), "max": vals[-1], "sum": math.fsum(vals)}


def load_groups() -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    completed_access_ok()
    raw = read_json(RAW)
    protocol = read_json(PROTOCOL)
    targets = {str(t.get("state_id")): t for t in (protocol.get("target_selection") or {}).get("targets", [])}
    groups: List[Dict[str, Any]] = []
    for row in raw.get("analysis", {}).get("state_profile_rows", []):
        label = row.get("oracle_label")
        if label is None:
            continue
        sid = str(row.get("state_id"))
        profile = str(row.get("terminal_profile"))
        target = targets.get(sid, {})
        med = row.get("median_by_h") or {}
        physical: Dict[int, float] = {}
        decision: Dict[int, float] = {}
        solver: Dict[int, float] = {}
        safe: Dict[int, bool] = {}
        for h in HORIZONS:
            m = med.get(str(h), {})
            physical[h] = sf(m.get("physical"), 0.0)
            decision[h] = sf(m.get("decision_sum_s"), 0.0)
            solver[h] = sf(m.get("solver_sum_s"), 0.0)
            safe[h] = bool(m.get("safe_all")) and bool(m.get("success_all")) and not bool(m.get("constraint_any")) and si(m.get("solver_failure_steps_sum"), 0) == 0
        branch_state = target.get("branch_previous_state") or {}
        score_components = target.get("score_components") or {}
        feats: Dict[str, Any] = {
            "terminal_profile": profile,
            "matched_terminal": profile == "matched_terminal",
            "shared_h15_terminal": profile == "shared_h15_terminal",
            "branch_step": si(target.get("branch_step"), -1),
            "case": si(target.get("case"), -1),
            "window": target.get("window") or "none",
            "source_stratum": row.get("source_stratum") or target.get("source_stratum") or "unknown",
            "target_role": row.get("target_role") or target.get("target_role") or "unknown",
            "selection_group": target.get("selection_group") or "unknown",
            "previously_used_anchor": bool(row.get("previously_used_in_true_h_broader_block") or target.get("previously_used_in_true_h_broader_block")),
            "noncontrol": bool(row.get("noncontrol_for_primary_gate")),
            "x": sf(branch_state.get("x"), 0.0),
            "y": sf(branch_state.get("y"), 0.0),
            "theta": sf(branch_state.get("theta"), 0.0),
            "score": sf(target.get("score"), 0.0),
            "score_performance": sf(score_components.get("performance_step_cost"), 0.0),
            "score_heading": sf(score_components.get("abs_heading_proxy"), 0.0),
            "score_turn": sf(score_components.get("abs_turn_effort_u_omega"), 0.0),
            "score_solver_iter": sf(score_components.get("solver_iteration_proxy"), 0.0),
        }
        groups.append({
            "key": sid + "|" + profile,
            "state_id": sid,
            "terminal_profile": profile,
            "label": int(label),
            "noncontrol": bool(row.get("noncontrol_for_primary_gate")),
            "previously_used_anchor": bool(row.get("previously_used_in_true_h_broader_block")),
            "physical": physical,
            "decision": decision,
            "solver": solver,
            "safe": safe,
            "features": feats,
        })
    if not groups:
        raise ContractError("no oracle rows loaded")
    return groups, raw, protocol


def count_hs(hs: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for h in hs:
        out[str(int(h))] = out.get(str(int(h)), 0) + 1
    return out


def fixed_metrics(groups: Sequence[Mapping[str, Any]], h: int) -> Dict[str, float]:
    return {
        "physical_sum": float(math.fsum(float(g["physical"][h]) for g in groups)),
        "decision_sum_s": float(math.fsum(float(g["decision"][h]) for g in groups)),
        "solver_sum_s": float(math.fsum(float(g["solver"][h]) for g in groups)),
    }


def eval_predictions(groups: Sequence[Mapping[str, Any]], pred: Mapping[str, int], name: str = "selector") -> Dict[str, Any]:
    n = len(groups)
    hs: List[int] = []
    phys = dec = sol = 0.0
    unsafe: List[str] = []
    oracle_phys = oracle_dec = oracle_sol = 0.0
    for g in groups:
        h = int(pred.get(str(g["key"]), 25))
        if h not in HORIZONS:
            h = 25
        hs.append(h)
        phys += float(g["physical"][h]); dec += float(g["decision"][h]); sol += float(g["solver"][h])
        if not bool(g["safe"][h]):
            unsafe.append(str(g["key"]))
        oh = int(g["label"])
        oracle_phys += float(g["physical"][oh]); oracle_dec += float(g["decision"][oh]); oracle_sol += float(g["solver"][oh])
    f10 = fixed_metrics(groups, 10); f15 = fixed_metrics(groups, 15); f25 = fixed_metrics(groups, 25)
    tol = max(2.0 * n, 0.05 * abs(f25["physical_sum"])) if n else 0.0
    def saving(base: float, val: float) -> Optional[float]:
        return None if base <= 0 else float((base - val) / base)
    out = {
        "name": name,
        "n_groups": n,
        "horizon_counts": count_hs(hs),
        "adaptive_nonconstant": len(set(hs)) >= 2,
        "physical_sum": float(phys),
        "decision_sum_s": float(dec),
        "solver_sum_s": float(sol),
        "unsafe_count": len(unsafe),
        "unsafe_keys": unsafe[:20],
        "oracle_physical_sum": float(oracle_phys),
        "oracle_decision_sum_s": float(oracle_dec),
        "oracle_solver_sum_s": float(oracle_sol),
        "oracle_regret_physical_sum": float(phys - oracle_phys),
        "oracle_regret_decision_sum_s": float(dec - oracle_dec),
        "fixed_H10": f10,
        "fixed_H15": f15,
        "fixed_H25": f25,
        "physical_tolerance_vs_H25": tol,
        "physical_delta_vs_H25": float(phys - f25["physical_sum"]),
        "physical_delta_vs_H15": float(phys - f15["physical_sum"]),
        "physical_delta_vs_H10": float(phys - f10["physical_sum"]),
        "decision_relative_saving_vs_H25": saving(f25["decision_sum_s"], dec),
        "decision_relative_saving_vs_H15": saving(f15["decision_sum_s"], dec),
        "decision_relative_saving_vs_H10": saving(f10["decision_sum_s"], dec),
    }
    out["safe_gate"] = (len(unsafe) == 0)
    out["physical_gate_vs_H25"] = bool(out["physical_delta_vs_H25"] <= tol)
    out["decision_saving_gate_vs_H25"] = bool(out["decision_relative_saving_vs_H25"] is not None and float(out["decision_relative_saving_vs_H25"]) >= MIN_DECISION_SAVING)
    return out


def selector_score(res: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
    unsafe = 0.0 if res.get("safe_gate") else 1.0
    phys_excess = max(0.0, sf(res.get("physical_delta_vs_H25")) - sf(res.get("physical_tolerance_vs_H25")))
    speed = sf(res.get("decision_relative_saving_vs_H25"), -999.0)
    speed_deficit = max(0.0, MIN_DECISION_SAVING - speed)
    # Prefer safe/physical-feasible, then speed-feasible, then lower physical regret, then faster decisions.
    return (unsafe, 1.0 if phys_excess > 0 else 0.0, 1.0 if speed_deficit > 0 else 0.0, phys_excess, speed_deficit, sf(res.get("physical_sum")), sf(res.get("decision_sum_s")))


def choose_h_gate(groups: Sequence[Mapping[str, Any]]) -> int:
    if not groups:
        return 25
    n = len(groups)
    base_phys = math.fsum(float(g["physical"][25]) for g in groups)
    tol = max(2.0 * n, 0.05 * abs(base_phys))
    feasible: List[Tuple[float, float, int]] = []
    fallback: List[Tuple[float, float, int]] = []
    for h in HORIZONS:
        phys = math.fsum(float(g["physical"][h]) for g in groups)
        dec = math.fsum(float(g["decision"][h]) for g in groups)
        safe = all(bool(g["safe"][h]) for g in groups)
        if safe and phys - base_phys <= tol:
            feasible.append((dec, phys, h))
        fallback.append((phys + (0.0 if safe else 1e9), dec, h))
    if feasible:
        return int(sorted(feasible)[0][2])
    return int(sorted(fallback)[0][2])


def choose_h_majority(groups: Sequence[Mapping[str, Any]]) -> int:
    if not groups:
        return 25
    labels = [int(g["label"]) for g in groups]
    counts = {h: labels.count(h) for h in HORIZONS}
    max_count = max(counts.values())
    tied = [h for h, c in counts.items() if c == max_count]
    return int(sorted(tied, key=lambda h: (math.fsum(float(g["decision"][h]) for g in groups), h))[0])


def make_leaf(groups: Sequence[Mapping[str, Any]], chooser: str) -> Dict[str, Any]:
    return {"kind": "leaf", "h": choose_h_majority(groups) if chooser == "majority" else choose_h_gate(groups), "n": len(groups), "labels": count_hs([int(g["label"]) for g in groups])}


def pred_tree(tree: Mapping[str, Any], g: Mapping[str, Any]) -> int:
    node = tree
    while node.get("kind") == "node":
        cond = node["cond"]
        if cond_eval(cond, g):
            node = node["left"]
        else:
            node = node["right"]
    return int(node.get("h", 25))


def pred_map_from_func(groups: Sequence[Mapping[str, Any]], f: Callable[[Mapping[str, Any]], int]) -> Dict[str, int]:
    return {str(g["key"]): int(f(g)) for g in groups}


def cond_eval(cond: Mapping[str, Any], g: Mapping[str, Any]) -> bool:
    val = g["features"].get(cond["feature"])
    if cond["op"] == "<=":
        return sf(val, 0.0) <= sf(cond["value"], 0.0)
    return val == cond["value"]


def cond_text(cond: Mapping[str, Any]) -> str:
    if cond["op"] == "<=":
        return f"{cond['feature']} <= {sf(cond['value']):.6g}"
    return f"{cond['feature']} == {cond['value']}"


def candidate_splits(groups: Sequence[Mapping[str, Any]], features: Sequence[Tuple[str, str]], min_leaf: int = 1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, typ in features:
        vals = [g["features"].get(name) for g in groups]
        if typ == "num":
            nums = sorted(set(sf(v, float("nan")) for v in vals if math.isfinite(sf(v, float("nan")))))
            thresholds = [(nums[i] + nums[i + 1]) / 2.0 for i in range(len(nums) - 1)]
            for thr in thresholds:
                cond = {"feature": name, "op": "<=", "value": float(thr)}
                n_left = sum(1 for g in groups if cond_eval(cond, g))
                if n_left >= min_leaf and len(groups) - n_left >= min_leaf:
                    out.append(cond)
        else:
            for v in sorted(set(str(v) for v in vals)):
                cond = {"feature": name, "op": "==", "value": v}
                n_left = sum(1 for g in groups if cond_eval(cond, g))
                if n_left >= min_leaf and len(groups) - n_left >= min_leaf:
                    out.append(cond)
    return out


def tree_conditions(tree: Mapping[str, Any]) -> List[str]:
    if tree.get("kind") != "node":
        return []
    return [cond_text(tree["cond"])] + tree_conditions(tree["left"]) + tree_conditions(tree["right"])


def build_tree(groups: Sequence[Mapping[str, Any]], features: Sequence[Tuple[str, str]], chooser: str, max_depth: int, depth: int = 0) -> Dict[str, Any]:
    leaf = make_leaf(groups, chooser)
    if depth >= max_depth or len(groups) <= 2 or len(set(int(g["label"]) for g in groups)) == 1:
        return leaf
    base_pred = {str(g["key"]): int(leaf["h"]) for g in groups}
    base_score = selector_score(eval_predictions(groups, base_pred, "leaf"))
    best_tree: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[float, float, float, float, float, float, float]] = None
    for cond in candidate_splits(groups, features, min_leaf=1):
        left = [g for g in groups if cond_eval(cond, g)]
        right = [g for g in groups if not cond_eval(cond, g)]
        if not left or not right:
            continue
        lt = build_tree(left, features, chooser, max_depth, depth + 1)
        rt = build_tree(right, features, chooser, max_depth, depth + 1)
        tree = {"kind": "node", "cond": cond, "left": lt, "right": rt, "n": len(groups)}
        pred = {str(g["key"]): pred_tree(tree, g) for g in groups}
        score = selector_score(eval_predictions(groups, pred, "tree"))
        if best_score is None or score < best_score:
            best_score = score; best_tree = tree
    if best_tree is not None and best_score is not None and best_score < base_score:
        return best_tree
    return leaf


DEPLOYABLE_FEATURES: List[Tuple[str, str]] = [
    ("terminal_profile", "cat"), ("matched_terminal", "cat"), ("branch_step", "num"),
    ("x", "num"), ("y", "num"), ("theta", "num"), ("window", "cat"),
]
DIAGNOSTIC_FEATURES: List[Tuple[str, str]] = DEPLOYABLE_FEATURES + [
    ("source_stratum", "cat"), ("target_role", "cat"), ("selection_group", "cat"),
    ("previously_used_anchor", "cat"), ("case", "cat"), ("score", "num"),
    ("score_performance", "num"), ("score_heading", "num"), ("score_turn", "num"), ("score_solver_iter", "num"),
]


def fit_family(family: Mapping[str, Any], train: Sequence[Mapping[str, Any]]) -> Tuple[Callable[[Mapping[str, Any]], int], Dict[str, Any]]:
    kind = family["kind"]
    if kind == "fixed":
        h = int(family["h"])
        return (lambda g, h=h: h), {"rule": f"fixed_H{h}"}
    if kind == "group":
        feature = str(family["feature"])
        chooser = str(family.get("chooser", "gate"))
        fallback = choose_h_majority(train) if chooser == "majority" else choose_h_gate(train)
        mapping: Dict[str, int] = {}
        vals = sorted(set(str(g["features"].get(feature)) for g in train))
        for v in vals:
            subset = [g for g in train if str(g["features"].get(feature)) == v]
            mapping[v] = choose_h_majority(subset) if chooser == "majority" else choose_h_gate(subset)
        def pred(g: Mapping[str, Any], feature: str = feature, mapping: Mapping[str, int] = mapping, fallback: int = fallback) -> int:
            return int(mapping.get(str(g["features"].get(feature)), fallback))
        return pred, {"rule": f"group_by_{feature}_{chooser}", "mapping": mapping, "fallback": fallback}
    if kind == "tree":
        chooser = str(family.get("chooser", "gate"))
        max_depth = int(family.get("max_depth", 2))
        features = DEPLOYABLE_FEATURES if family.get("feature_set") == "deployable" else DIAGNOSTIC_FEATURES
        tree = build_tree(train, features, chooser, max_depth=max_depth)
        return (lambda g, tree=tree: pred_tree(tree, g)), {"tree": tree, "conditions": tree_conditions(tree), "feature_set": family.get("feature_set"), "chooser": chooser, "max_depth": max_depth}
    raise ContractError("unknown family kind: " + str(kind))


def family_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for h in HORIZONS:
        specs.append({"name": f"fixed_H{h}", "kind": "fixed", "h": h, "deployable": True, "adaptive_candidate": False, "uses_dev_only_features": False})
    for feature in ("terminal_profile", "window"):
        specs.append({"name": f"{feature}_constant_gate", "kind": "group", "feature": feature, "chooser": "gate", "deployable": True, "adaptive_candidate": True, "uses_dev_only_features": False})
        specs.append({"name": f"{feature}_constant_majority", "kind": "group", "feature": feature, "chooser": "majority", "deployable": True, "adaptive_candidate": True, "uses_dev_only_features": False})
    for depth in (1, 2):
        specs.append({"name": f"deployable_tree_d{depth}_gate", "kind": "tree", "feature_set": "deployable", "chooser": "gate", "max_depth": depth, "deployable": True, "adaptive_candidate": True, "uses_dev_only_features": False})
        specs.append({"name": f"deployable_tree_d{depth}_majority", "kind": "tree", "feature_set": "deployable", "chooser": "majority", "max_depth": depth, "deployable": True, "adaptive_candidate": True, "uses_dev_only_features": False})
        specs.append({"name": f"diagnostic_source_tree_d{depth}_gate", "kind": "tree", "feature_set": "diagnostic", "chooser": "gate", "max_depth": depth, "deployable": False, "adaptive_candidate": True, "uses_dev_only_features": True})
    for feature in ("source_stratum", "target_role", "previously_used_anchor", "selection_group", "case"):
        specs.append({"name": f"diagnostic_{feature}_constant_gate", "kind": "group", "feature": feature, "chooser": "gate", "deployable": False, "adaptive_candidate": True, "uses_dev_only_features": True})
    return specs


def cross_validate(family: Mapping[str, Any], groups_all: Sequence[Mapping[str, Any]], eval_groups: Sequence[Mapping[str, Any]], fold_key: str = "state_id") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    folds = sorted(set(str(g[fold_key]) for g in groups_all))
    pred: Dict[str, int] = {}
    fold_details: List[Dict[str, Any]] = []
    for fold in folds:
        train = [g for g in groups_all if str(g[fold_key]) != fold]
        test = [g for g in groups_all if str(g[fold_key]) == fold]
        f, meta = fit_family(family, train)
        for g in test:
            pred[str(g["key"])] = int(f(g))
        fold_details.append({"fold": fold, "train_groups": len(train), "test_groups": len(test), "predictions": {str(g["key"]): int(f(g)) for g in test}, "meta": meta})
    return eval_predictions(eval_groups, pred, family["name"]), {"fold_key": fold_key, "folds": fold_details}


def evaluate_all(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    primary = [g for g in groups if bool(g["noncontrol"])]
    specs = family_specs()
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for spec in specs:
        f, meta = fit_family(spec, groups)
        full_pred = pred_map_from_func(groups, f)
        full_primary = eval_predictions(primary, {k: v for k, v in full_pred.items() if k in {str(g["key"]) for g in primary}}, spec["name"])
        cv_primary, cv_detail = cross_validate(spec, groups, primary, fold_key="state_id")
        row = {
            "name": spec["name"],
            "deployable": bool(spec.get("deployable")),
            "adaptive_candidate": bool(spec.get("adaptive_candidate")),
            "uses_dev_only_features": bool(spec.get("uses_dev_only_features")),
            "full_primary": {k: full_primary[k] for k in ("horizon_counts", "physical_sum", "decision_sum_s", "physical_delta_vs_H25", "physical_tolerance_vs_H25", "decision_relative_saving_vs_H25", "oracle_regret_physical_sum", "safe_gate", "physical_gate_vs_H25", "decision_saving_gate_vs_H25", "adaptive_nonconstant")},
            "cv_loso_primary": {k: cv_primary[k] for k in ("horizon_counts", "physical_sum", "decision_sum_s", "physical_delta_vs_H25", "physical_tolerance_vs_H25", "decision_relative_saving_vs_H25", "oracle_regret_physical_sum", "safe_gate", "physical_gate_vs_H25", "decision_saving_gate_vs_H25", "adaptive_nonconstant", "physical_delta_vs_H15", "physical_delta_vs_H10")},
            "full_meta": meta,
        }
        row["cv_acceptance_core"] = bool(row["deployable"] and row["adaptive_candidate"] and cv_primary["safe_gate"] and cv_primary["physical_gate_vs_H25"] and cv_primary["decision_saving_gate_vs_H25"] and cv_primary["adaptive_nonconstant"])
        rows.append(row)
        details[spec["name"]] = {"full_meta": meta, "cv": cv_detail}
    rows_sorted = sorted(rows, key=lambda r: selector_score({**r["cv_loso_primary"], "safe_gate": r["cv_loso_primary"]["safe_gate"]}))
    fixed_by_name = {r["name"]: r["cv_loso_primary"] for r in rows}
    fixed_short_absorption = {
        "fixed_H10_core_pass": bool(fixed_by_name.get("fixed_H10", {}).get("safe_gate") and fixed_by_name.get("fixed_H10", {}).get("physical_gate_vs_H25") and fixed_by_name.get("fixed_H10", {}).get("decision_saving_gate_vs_H25")),
        "fixed_H15_core_pass": bool(fixed_by_name.get("fixed_H15", {}).get("safe_gate") and fixed_by_name.get("fixed_H15", {}).get("physical_gate_vs_H25") and fixed_by_name.get("fixed_H15", {}).get("decision_saving_gate_vs_H25")),
    }
    deployable_passes = [r for r in rows if r["cv_acceptance_core"]]
    accepted = bool(deployable_passes and not (fixed_short_absorption["fixed_H10_core_pass"] or fixed_short_absorption["fixed_H15_core_pass"]))
    return {
        "primary_group_count": len(primary),
        "all_group_count": len(groups),
        "candidate_rows": rows,
        "candidate_rows_sorted_by_cv_score": [r["name"] for r in rows_sorted],
        "best_cv_by_score": rows_sorted[0] if rows_sorted else None,
        "deployable_cv_core_passes": [r["name"] for r in deployable_passes],
        "fixed_short_absorption": fixed_short_absorption,
        "accepted_for_fresh_rollout": accepted,
        "details": details,
    }


def evidence_diagnostics(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    primary = [g for g in groups if bool(g["noncontrol"])]
    by_state: Dict[str, List[Mapping[str, Any]]] = {}
    for g in groups:
        by_state.setdefault(str(g["state_id"]), []).append(g)
    profile_disagreements = []
    for sid, rows in sorted(by_state.items()):
        labels = {str(g["terminal_profile"]): int(g["label"]) for g in rows}
        if len(set(labels.values())) > 1:
            profile_disagreements.append({"state_id": sid, "labels_by_profile": labels})
    rare = {str(h): sorted(set(str(g["state_id"]) for g in primary if int(g["label"]) == h)) for h in HORIZONS}
    oracle_pred = {str(g["key"]): int(g["label"]) for g in primary}
    oracle_eval = eval_predictions(primary, oracle_pred, "oracle")
    fixed_evals = {f"fixed_H{h}": eval_predictions(primary, {str(g["key"]): h for g in primary}, f"fixed_H{h}") for h in HORIZONS}
    return {
        "primary_label_counts": count_hs([int(g["label"]) for g in primary]),
        "all_label_counts": count_hs([int(g["label"]) for g in groups]),
        "unique_primary_states_by_label": rare,
        "profile_label_disagreements": profile_disagreements,
        "profile_label_disagreement_count": len(profile_disagreements),
        "h25_primary_unique_state_count": len(rare.get("25", [])),
        "h15_primary_unique_state_count": len(rare.get("15", [])),
        "oracle_primary_eval": oracle_eval,
        "fixed_primary_evals": {k: {kk: v[kk] for kk in ("physical_sum", "decision_sum_s", "physical_delta_vs_H25", "decision_relative_saving_vs_H25", "safe_gate", "physical_gate_vs_H25", "decision_saving_gate_vs_H25")} for k, v in fixed_evals.items()},
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    ev = raw["evidence_diagnostics"]
    analysis = raw["selector_feasibility"]
    lines = [
        "# Vehicle true-variable-H selector feasibility postdiagnostic v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only offline diagnostic: no simulation, no validation64, no sealed test, no training/refit checkpoint.",
        "",
        "## Oracle-bank structure",
        "",
        f"- Primary non-control groups: `{analysis['primary_group_count']}` / all groups `{analysis['all_group_count']}`.",
        f"- Primary label counts: `{ev['primary_label_counts']}`; all label counts: `{ev['all_label_counts']}`.",
        f"- Primary unique states by label: `{ev['unique_primary_states_by_label']}`.",
        f"- Terminal-profile label disagreements: `{ev['profile_label_disagreement_count']}`: `{ev['profile_label_disagreements']}`.",
        f"- H25 primary labels occur in `{ev['h25_primary_unique_state_count']}` unique state(s); H15 primary labels in `{ev['h15_primary_unique_state_count']}` unique state(s).",
        "",
        "## Leave-one-state-out selector feasibility",
        "",
        "Acceptance for a fresh rollout was frozen inside this script before execution: deployable, adaptive/nonconstant, safe, physical delta vs fixed H25 within tolerance, >=15% measured decision-time saving vs fixed H25, and not absorbed by fixed H10/H15. Metrics use logged measured decision time, not H alone.",
        "",
        "| candidate | deployable | CV counts | CV physΔ vs H25 | tol | CV decision saving vs H25 | CV oracle phys regret | CV core pass | full-fit counts | full-fit physΔ | full-fit decision saving |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in raw["selector_feasibility"]["candidate_rows"]:
        cv = row["cv_loso_primary"]; full = row["full_primary"]
        lines.append("| `%s` | `%s` | `%s` | %.6g | %.6g | %.6g | %.6g | `%s` | `%s` | %.6g | %.6g |" % (
            row["name"], row["deployable"], cv["horizon_counts"], sf(cv["physical_delta_vs_H25"]), sf(cv["physical_tolerance_vs_H25"]), sf(cv["decision_relative_saving_vs_H25"], float("nan")), sf(cv["oracle_regret_physical_sum"]), row["cv_acceptance_core"], full["horizon_counts"], sf(full["physical_delta_vs_H25"]), sf(full["decision_relative_saving_vs_H25"], float("nan"))
        ))
    lines += [
        "",
        "## Decision",
        "",
        f"- Accepted for fresh closed-loop rollout: `{analysis['accepted_for_fresh_rollout']}`.",
        f"- Deployable CV core passes: `{analysis['deployable_cv_core_passes']}`.",
        f"- Fixed-short absorption flags: `{analysis['fixed_short_absorption']}`.",
        f"- Best CV-by-score candidate: `{(analysis['best_cv_by_score'] or {}).get('name')}` with CV `{(analysis['best_cv_by_score'] or {}).get('cv_loso_primary')}`.",
        "",
        str(raw["decision"]),
        "",
        "This remains development evidence only and cannot be reported as validation/test success or ORIGINAL reproduction.",
        f"Backup request before further simulation/training/refit: `{raw['backup_request_after_diagnostic']}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    if (RUN_DIR / "completed.json").exists():
        done = read_json(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "accepted_for_fresh_rollout": done.get("accepted_for_fresh_rollout"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    groups, oracle_raw, protocol = load_groups()
    evidence = evidence_diagnostics(groups)
    feasibility = evaluate_all(groups)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_SELECTOR_FEASIBILITY_POSTDIAGNOSTIC_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    decision: str
    blockers: List[str] = []
    if feasibility["accepted_for_fresh_rollout"]:
        decision = "Proceed next to a small fresh development confirmation rollout for the selected deployable IMPROVED true-variable-H selector, after verified backup. Keep validation64 and sealed test closed; compare against true fixed H10/H15/H25 with blocked timing."
        next_experiment = "freeze and run a small fresh development confirmation bank for the selected deployable selector after backup"
    else:
        if evidence["h25_primary_unique_state_count"] <= 1:
            blockers.append("H25-positive primary label is supported by <=1 unique development state, so leave-one-state-out cannot learn a conservative H25 trigger.")
        if evidence["profile_label_disagreement_count"]:
            blockers.append("Terminal-profile label disagreements indicate possible terminal/value artifact or profile-dependent objective mismatch.")
        decision = "Do not launch selector training/refit yet from this oracle bank: the offline deployable leave-one-state-out gate did not pass. Next highest-information action is a versioned fresh development risk-anchor/terminal-consistency bank (true H10/H15/H25, matched/shared terminal, blocked timing) or terminal-value/objective repair, not scaling a selector on sparse anchor-dependent labels."
        next_experiment = "after backup, freeze a small source-independent true-H risk-anchor/terminal-consistency acquisition diagnostic before any selector refit"
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup offline selector-feasibility diagnostic outputs/source before further simulation/training/refit", "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "offline_selector_fits": len(family_specs()) * (1 + len(set(g['state_id'] for g in groups))), "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(RUN_DIR), rel(STATE_PATH), rel(Path(__file__)), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_IMPROVED_selector_feasibility_not_validation_not_final_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "budget_actual": {"new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "offline_selector_family_fits": len(family_specs()) * (1 + len(set(g['state_id'] for g in groups))), "candidate_pool_resets": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "inputs": {"oracle_raw": rel(RAW), "oracle_completed": rel(ORACLE_DONE), "protocol": rel(PROTOCOL), "oracle_raw_sha256": sha256(RAW), "protocol_sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        "acceptance_rule_frozen": {"deployable_required": True, "adaptive_nonconstant_required": True, "safe_gate": True, "physical_delta_vs_H25_within_tolerance": True, "decision_saving_vs_H25_min": MIN_DECISION_SAVING, "not_absorbed_by_fixed_H10_or_H15": True, "cv_split": "leave_one_state_out"},
        "evidence_diagnostics": evidence,
        "selector_feasibility": feasibility,
        "decision": decision,
        "blockers": blockers,
        "next_experiment": next_experiment,
        "backup_request_after_diagnostic": rel(req),
        "interpretation_limits": ["development-only oracle-bank postdiagnostic", "no closed-loop selector deployment", "no validation64 or sealed test", "not ORIGINAL SAC", "offline CV over source-supported states may still over/under-estimate fresh generalization"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H selector feasibility postdiagnostic v0

UTC: {created.isoformat()}. Offline development diagnostic completed with no simulation/training/refit, validation64 closed, sealed test closed. Accepted for fresh rollout: {feasibility['accepted_for_fresh_rollout']}. Primary label counts: {evidence['primary_label_counts']}. H25 unique primary states: {evidence['h25_primary_unique_state_count']}. Decision: {decision} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`. Backup request: `{rel(req)}`.
""")
    files = [p for p in [Path(__file__), RAW, ORACLE_DONE, PROTOCOL, RUN_DIR / "raw.json", RUN_DIR / "summary.md", STATE_PATH, req] if p.exists()]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "accepted_for_fresh_rollout": feasibility["accepted_for_fresh_rollout"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0, "backup_request": rel(req), "decision": decision, "hashes": {rel(p): sha256(p) for p in files}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "accepted_for_fresh_rollout": feasibility["accepted_for_fresh_rollout"], "primary_label_counts": evidence["primary_label_counts"], "h25_primary_unique_state_count": evidence["h25_primary_unique_state_count"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
        raise
