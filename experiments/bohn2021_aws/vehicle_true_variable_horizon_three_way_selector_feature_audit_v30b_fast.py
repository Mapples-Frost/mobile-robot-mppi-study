#!/usr/bin/env python3
"""v30b fast three-way selector feature/separability audit after v29.

This is a performance-bounded replacement for v30, which timed out while
brute-forcing two-threshold rules.  It is still development-only analysis over
already opened v29 outputs: no MPC simulation, no validation64, no sealed test,
and no training/refit/gradient updates.

Question: given the v29 identical-state H12/H15/H25/H35 outcomes, can a very
simple deployable pre-outcome feature rule choose among H12, H15, and H35 with
absolute success-sensitive safety while saving whole-decision time relative to a
strong fixed longer-H fallback?  Source/campaign/category/role/base-state IDs and
any outcome fields are excluded from prediction features.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast"
STAMP = "20260930T0410Z"
MARKER = f"vehicle-three-way-selector-feature-audit-v30b-fast-{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0410_after_v30b_three_way_feature_audit.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_THREE_WAY_SELECTOR_FEATURE_AUDIT_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
# v30 run was terminated by the bounded runner at/after this timestamp.  A new
# backup proof should postdate it so the failed-run registry, failure note and
# v30b source are recoverable before v30b executes.
REQUIRE_BACKUP_AFTER = dt.datetime.fromisoformat("2026-09-30T03:27:34.201806+00:00")

V29_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json"
V29_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/completed.json"
V29_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/summary.md"
V30_FAILED_NOTE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30_20260930T0355Z/failed_timeout_posthoc.md"
V30_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260930T032734_52f6627b/registry.json"

HORIZONS = [12, 15, 25, 35]
SELECTOR_HORIZONS = [12, 15, 35]

class ContractError(RuntimeError):
    pass

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

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

def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default

def parse_time(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str):
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def verify_completed(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing v29 completion marker: " + rel(path))
    obj = read_json(path)
    if obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError("v29 completion marker is not complete")
    if obj.get("validation64_bank_opened") is True or obj.get("sealed_test_accessed") is True or obj.get("test_accessed") is True:
        raise ContractError("v29 marker unexpectedly reports validation/test access")
    return obj

def verify_backup(path: Path, after_time: dt.datetime) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("backup proof missing: " + rel(path))
    obj = read_json(path)
    if obj.get("status") != "verified" or obj.get("backup_verified") is not True:
        raise ContractError("backup proof is not verified")
    if obj.get("remaining_changed_files") not in (0, "0"):
        raise ContractError("backup proof has remaining changed files")
    if not obj.get("commit") or not obj.get("packages_this_run"):
        raise ContractError("backup proof lacks commit/package evidence")
    t = parse_time(obj.get("time"))
    if t is None or t <= after_time:
        raise ContractError(f"backup proof time {obj.get('time')} does not postdate required {after_time.isoformat()}")
    return obj

def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def outcome(row: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    return row["per_horizon"][str(h)]

def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("safe_success_no_solver_fail")) and not bool(e.get("hit_step_cap"))

def build_features(raw29: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    state_specs = {str(s.get("base_state_id")): s for s in raw29.get("selected_states") or []}
    features: Dict[str, Dict[str, float]] = {}
    for row in raw29["analysis"]["state_rows"]:
        bid = str(row["base_state_id"])
        spec = state_specs.get(bid)
        if not spec:
            raise ContractError("missing selected-state spec for " + bid)
        obs = spec.get("initial_observation") or []
        feats: Dict[str, float] = {}
        vals: List[float] = []
        for i, val in enumerate(obs):
            x = sf(val)
            vals.append(x)
            feats[f"obs_{i:02d}"] = x
            feats[f"abs_obs_{i:02d}"] = abs(x)
        prev = spec.get("branch_previous_state") or {}
        for k in ("x", "y", "theta"):
            x = sf(prev.get(k))
            feats[f"prev_{k}"] = x
            feats[f"abs_prev_{k}"] = abs(x)
        feats["branch_step"] = sf(row.get("branch_step"))
        feats["obs_l2"] = math.sqrt(sum(v * v for v in vals)) if vals else 0.0
        feats["obs_linf"] = max([abs(v) for v in vals] or [0.0])
        feats["prev_xy_radius"] = math.sqrt(feats.get("prev_x", 0.0) ** 2 + feats.get("prev_y", 0.0) ** 2)
        features[bid] = feats
    return features

def make_arrays(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    all_mask = (1 << n) - 1
    decision: Dict[int, List[float]] = {h: [] for h in HORIZONS}
    solver: Dict[int, List[float]] = {h: [] for h in HORIZONS}
    physical: Dict[int, List[float]] = {h: [] for h in HORIZONS}
    safe_mask: Dict[int, int] = {h: 0 for h in HORIZONS}
    bad_mask: Dict[int, int] = {h: 0 for h in HORIZONS}
    bad_reasons: Dict[Tuple[int, int], str] = {}
    excesses: Dict[Tuple[int, int], Optional[float]] = {}
    for i, row in enumerate(rows):
        safe_phys = []
        for h in HORIZONS:
            e = outcome(row, h)
            if is_safe(e):
                safe_phys.append(sf(e.get("physical_constraint_cost")))
        min_phys = min(safe_phys) if safe_phys else float("inf")
        tol = max(2.0, 0.25 * abs(min_phys)) if math.isfinite(min_phys) else float("inf")
        for h in HORIZONS:
            e = outcome(row, h)
            decision[h].append(sf(e.get("decision_sum_s")))
            solver[h].append(sf(e.get("solver_sum_s")))
            phys = sf(e.get("physical_constraint_cost"))
            physical[h].append(phys)
            if is_safe(e):
                safe_mask[h] |= 1 << i
            bad = False
            reason = "ok"
            excess: Optional[float] = None
            if not is_safe(e):
                bad = True
                reason = "unsafe_or_step_cap"
            elif math.isfinite(min_phys):
                excess_val = phys - min_phys
                excess = excess_val
                if excess_val > tol:
                    bad = True
                    reason = "large_physical_excess_vs_best_safe"
            if bad:
                bad_mask[h] |= 1 << i
            bad_reasons[(i, h)] = reason
            excesses[(i, h)] = excess
    # Precompute sums for every row mask. n=11 in v29, so this is tiny and makes
    # two-threshold/LOO enumeration bounded and deterministic.
    maxmask = 1 << n
    sum_decision: Dict[int, List[float]] = {h: [0.0] * maxmask for h in HORIZONS}
    sum_solver: Dict[int, List[float]] = {h: [0.0] * maxmask for h in HORIZONS}
    sum_physical: Dict[int, List[float]] = {h: [0.0] * maxmask for h in HORIZONS}
    for h in HORIZONS:
        for mask in range(1, maxmask):
            bit = mask & -mask
            i = bit.bit_length() - 1
            prev = mask ^ bit
            sum_decision[h][mask] = sum_decision[h][prev] + decision[h][i]
            sum_solver[h][mask] = sum_solver[h][prev] + solver[h][i]
            sum_physical[h][mask] = sum_physical[h][prev] + physical[h][i]
    return {
        "n": n,
        "all_mask": all_mask,
        "decision": decision,
        "solver": solver,
        "physical": physical,
        "safe_mask": safe_mask,
        "bad_mask": bad_mask,
        "bad_reasons": bad_reasons,
        "excesses": excesses,
        "sum_decision": sum_decision,
        "sum_solver": sum_solver,
        "sum_physical": sum_physical,
    }

def condition_candidates(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], train_mask: Optional[int] = None) -> List[Dict[str, Any]]:
    ids = [str(r["base_state_id"]) for r in rows]
    if train_mask is None:
        train_mask = (1 << len(ids)) - 1
    names = sorted(next(iter(features.values())).keys())
    conds: List[Dict[str, Any]] = [{"feature": "__false__", "op": "false", "threshold": 0.0, "mask": 0, "repr": "FALSE"}]
    for name in names:
        vals = sorted({features[ids[i]][name] for i in range(len(ids)) if (train_mask >> i) & 1 and math.isfinite(features[ids[i]][name])})
        if len(vals) <= 1:
            continue
        thresholds = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:]) if a != b]
        for thr in thresholds:
            le = 0
            gt = 0
            for i, bid in enumerate(ids):
                val = features[bid][name]
                if val <= thr:
                    le |= 1 << i
                else:
                    gt |= 1 << i
            conds.append({"feature": name, "op": "<=", "threshold": thr, "mask": le, "repr": f"{name} <= {thr:.12g}"})
            conds.append({"feature": name, "op": ">", "threshold": thr, "mask": gt, "repr": f"{name} > {thr:.12g}"})
    return conds

def metric_from_masks(rows: Sequence[Mapping[str, Any]], arr: Mapping[str, Any], h12_mask: int, h15_mask: int, h35_mask: int, eval_mask: Optional[int] = None) -> Dict[str, Any]:
    if eval_mask is None:
        eval_mask = arr["all_mask"]
    h12 = h12_mask & eval_mask
    h15 = h15_mask & eval_mask
    h35 = h35_mask & eval_mask
    n = arr["n"]
    preds: Dict[str, int] = {}
    for i, row in enumerate(rows):
        bit = 1 << i
        if not (eval_mask & bit):
            continue
        if h35 & bit:
            preds[str(row["base_state_id"])] = 35
        elif h15 & bit:
            preds[str(row["base_state_id"])] = 15
        elif h12 & bit:
            preds[str(row["base_state_id"])] = 12
        else:
            raise ContractError("row assigned to no horizon")
    bad_bits_by_h = {12: h12 & arr["bad_mask"][12], 15: h15 & arr["bad_mask"][15], 35: h35 & arr["bad_mask"][35]}
    safe_count = ((h12 & arr["safe_mask"][12]).bit_count() + (h15 & arr["safe_mask"][15]).bit_count() + (h35 & arr["safe_mask"][35]).bit_count())
    bad_rows: List[Dict[str, Any]] = []
    for h, bits in bad_bits_by_h.items():
        m = bits
        while m:
            bit = m & -m
            i = bit.bit_length() - 1
            row = rows[i]
            bad_rows.append({
                "base_state_id": str(row["base_state_id"]),
                "state_label": row.get("state_label"),
                "category_for_audit_only": row.get("category"),
                "predicted_horizon": h,
                "reason": arr["bad_reasons"].get((i, h)),
                "physical_excess_vs_best_safe": arr["excesses"].get((i, h)),
            })
            m ^= bit
    return {
        "states": eval_mask.bit_count(),
        "h_counts": {"12": h12.bit_count(), "15": h15.bit_count(), "35": h35.bit_count()},
        "safe_success_count": safe_count,
        "bad_count": sum(v.bit_count() for v in bad_bits_by_h.values()),
        "bad_rows": bad_rows,
        "decision_sum_s": arr["sum_decision"][12][h12] + arr["sum_decision"][15][h15] + arr["sum_decision"][35][h35],
        "solver_sum_s": arr["sum_solver"][12][h12] + arr["sum_solver"][15][h15] + arr["sum_solver"][35][h35],
        "physical_sum": arr["sum_physical"][12][h12] + arr["sum_physical"][15][h15] + arr["sum_physical"][35][h35],
        "predictions": preds,
    }

def metric_fixed(rows: Sequence[Mapping[str, Any]], arr: Mapping[str, Any], h: int) -> Dict[str, Any]:
    mask = arr["all_mask"]
    if h == 12:
        return metric_from_masks(rows, arr, mask, 0, 0)
    if h == 15:
        return metric_from_masks(rows, arr, 0, mask, 0)
    if h == 35:
        return metric_from_masks(rows, arr, 0, 0, mask)
    # H25 is not a deployable action for the three-way rule, but include fixed-H25
    # comparator using a direct generic summary rather than metric_from_masks.
    bad_bits = mask & arr["bad_mask"][25]
    safe_count = (mask & arr["safe_mask"][25]).bit_count()
    bad_rows = []
    m = bad_bits
    while m:
        bit = m & -m
        i = bit.bit_length() - 1
        row = rows[i]
        bad_rows.append({"base_state_id": str(row["base_state_id"]), "state_label": row.get("state_label"), "category_for_audit_only": row.get("category"), "predicted_horizon": 25, "reason": arr["bad_reasons"].get((i, 25)), "physical_excess_vs_best_safe": arr["excesses"].get((i, 25))})
        m ^= bit
    return {
        "states": mask.bit_count(),
        "h_counts": {"12": 0, "15": 0, "25": mask.bit_count(), "35": 0},
        "safe_success_count": safe_count,
        "bad_count": bad_bits.bit_count(),
        "bad_rows": bad_rows,
        "decision_sum_s": arr["sum_decision"][25][mask],
        "solver_sum_s": arr["sum_solver"][25][mask],
        "physical_sum": arr["sum_physical"][25][mask],
        "predictions": {str(row["base_state_id"]): 25 for row in rows},
    }

def metric_oracle_fastest_safe(rows: Sequence[Mapping[str, Any]], arr: Mapping[str, Any], allowed: Sequence[int]) -> Dict[str, Any]:
    h_masks = {12: 0, 15: 0, 35: 0}
    for i, row in enumerate(rows):
        safe = [h for h in allowed if is_safe(outcome(row, h))]
        if not safe:
            chosen = int(allowed[0])
        else:
            chosen = min(safe, key=lambda h: sf(outcome(row, h).get("decision_sum_s")))
        if chosen not in h_masks:
            # For this script allowed excludes H25; keep guard for clarity.
            raise ContractError("oracle chose unsupported deployable horizon")
        h_masks[chosen] |= 1 << i
    return metric_from_masks(rows, arr, h_masks[12], h_masks[15], h_masks[35])

def score(metric: Mapping[str, Any], eval_rows: int) -> Tuple[float, float, float, float]:
    # Safety first; then prefer more successes; then lower decision sum; then lower
    # physical cost.  This enforces success-sensitive semantics before speed.
    return (
        float(metric["bad_count"]),
        -float(metric["safe_success_count"]),
        float(metric["decision_sum_s"]),
        float(metric["physical_sum"]),
    )

def eval_rule_masks(rows: Sequence[Mapping[str, Any]], arr: Mapping[str, Any], c35: Mapping[str, Any], c15: Mapping[str, Any], order: str, eval_mask: Optional[int] = None) -> Dict[str, Any]:
    all_mask = arr["all_mask"] if eval_mask is None else eval_mask
    m35 = int(c35["mask"])
    m15 = int(c15["mask"])
    if order == "H35_first":
        h35 = m35 & all_mask
        h15 = m15 & (~h35) & all_mask
    elif order == "H15_first":
        h15 = m15 & all_mask
        h35 = m35 & (~h15) & all_mask
    else:
        raise ContractError("bad order")
    h12 = all_mask & ~(h35 | h15)
    return metric_from_masks(rows, arr, h12, h15, h35, all_mask)

def learn_rule(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], arr: Mapping[str, Any], train_mask: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if train_mask is None:
        train_mask = arr["all_mask"]
    conds = condition_candidates(rows, features, train_mask)
    if len(conds) > 2500:
        raise ContractError(f"too many condition candidates {len(conds)}; refuse unbounded audit")
    best_rule: Optional[Dict[str, Any]] = None
    best_metric: Optional[Dict[str, Any]] = None
    evaluated = 0
    for c35, c15 in itertools.product(conds, conds):
        for order in ("H35_first", "H15_first"):
            metric = eval_rule_masks(rows, arr, c35, c15, order, train_mask)
            evaluated += 1
            if best_metric is None or score(metric, train_mask.bit_count()) < score(best_metric, train_mask.bit_count()):
                best_metric = metric
                best_rule = {"default_horizon": 12, "h35_condition": {k: c35[k] for k in ("feature", "op", "threshold", "repr")}, "h15_condition": {k: c15[k] for k in ("feature", "op", "threshold", "repr")}, "order": order}
    assert best_rule is not None and best_metric is not None
    diagnostics = {"condition_candidates": len(conds), "rules_evaluated": evaluated, "train_mask": train_mask}
    return best_rule, best_metric, diagnostics

def apply_rule(rows: Sequence[Mapping[str, Any]], arr: Mapping[str, Any], features: Mapping[str, Mapping[str, float]], rule: Mapping[str, Any], eval_mask: Optional[int] = None) -> Dict[str, Any]:
    ids = [str(r["base_state_id"]) for r in rows]
    def cond_mask(cond: Mapping[str, Any]) -> int:
        if cond.get("feature") == "__false__":
            return 0
        name = str(cond["feature"])
        op = str(cond["op"])
        thr = sf(cond["threshold"])
        mask = 0
        for i, bid in enumerate(ids):
            val = features[bid][name]
            if (op == "<=" and val <= thr) or (op == ">" and val > thr):
                mask |= 1 << i
        return mask
    c35 = {"mask": cond_mask(rule["h35_condition"]), **dict(rule["h35_condition"])}
    c15 = {"mask": cond_mask(rule["h15_condition"]), **dict(rule["h15_condition"])}
    return eval_rule_masks(rows, arr, c35, c15, str(rule["order"]), eval_mask)

def leave_one_out(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]], arr: Mapping[str, Any]) -> Dict[str, Any]:
    held_masks = {12: 0, 15: 0, 35: 0}
    folds: List[Dict[str, Any]] = []
    n = arr["n"]
    for i, row in enumerate(rows):
        held_bit = 1 << i
        train_mask = arr["all_mask"] ^ held_bit
        rule, train_metric, diag = learn_rule(rows, features, arr, train_mask)
        pred_metric = apply_rule(rows, arr, features, rule, held_bit)
        pred = pred_metric["predictions"][str(row["base_state_id"])]
        held_masks[pred] |= held_bit
        folds.append({
            "heldout_state": str(row["base_state_id"]),
            "heldout_category_for_audit_only": row.get("category"),
            "predicted_horizon": pred,
            "heldout_bad_count": pred_metric["bad_count"],
            "heldout_bad_rows": pred_metric["bad_rows"],
            "train_bad_count": train_metric["bad_count"],
            "train_decision_sum_s": train_metric["decision_sum_s"],
            "rule": rule,
            "rule_search": diag,
        })
    metric = metric_from_masks(rows, arr, held_masks[12], held_masks[15], held_masks[35])
    metric["folds"] = folds
    return metric

def pct_saving(candidate: float, reference: float) -> float:
    return 0.0 if reference <= 0 else (reference - candidate) / reference

def write_summary(raw: Mapping[str, Any]) -> None:
    c = raw["comparators"]
    lines = [
        "# v30b fast three-way selector feature/separability audit",
        "",
        f"UTC: `{raw['created_utc']}`. Analysis-only over v29; no MPC simulations, no validation64, no sealed test, no training/refit.",
        "",
        "## Why this exists",
        "",
        "The preceding v30 implementation timed out under the 600 s bound before writing scientific outputs. v30b replaces it with bit-mask precomputation; the v30 timeout is recorded as an implementation/performance failure, not a scientific result.",
        "",
        "## Question",
        "",
        "Can deployable pre-outcome observation/state/step features choose H35 for the source242 both-fail rescue rows, H15 for v19 H12-risk rows, and H12 for fresh safe controls, without using source/category/role IDs or outcome fields as predictors?",
        "",
        "## Comparator metrics on 11 opened v29 states",
        "",
        "Bad rows are absolute success-sensitive: unsafe/step-cap rows are bad, and safe horizons with large physical-cost excess versus the best safe continuation are also bad. Failed-row timing is not speed evidence.",
        "",
        "| policy | H counts | bad rows | safe successes | decision sum s | physical sum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["fixed_H12", "fixed_H15", "fixed_H25", "fixed_H35", "oracle_fastest_safe_H12_H15_H35"]:
        m = c[name]
        lines.append(f"| `{name}` | `{m['h_counts']}` | {m['bad_count']} | {m['safe_success_count']} | {m['decision_sum_s']:.6g} | {m['physical_sum']:.6g} |")
    ins = raw["learned_in_sample"]
    loo = raw["leave_one_out"]
    lines += [
        "",
        "## Simple two-threshold rule audit",
        "",
        f"- Best in-sample rule: `{raw['learned_in_sample_rule']}`",
        f"- In-sample: bad rows `{ins['bad_count']}`, safe successes `{ins['safe_success_count']}`, H counts `{ins['h_counts']}`, decision sum `{ins['decision_sum_s']:.6g}` s.",
        f"- Leave-one-out: bad rows `{loo['bad_count']}`, safe successes `{loo['safe_success_count']}`, H counts `{loo['h_counts']}`, decision sum `{loo['decision_sum_s']:.6g}` s.",
        f"- LOO decision saving vs fixed H35: `{100.0 * raw['headline']['loo_decision_saving_vs_fixed_H35']:.2f}%` (valid only if LOO bad rows are zero).",
        f"- Oracle fastest-safe H12/H15/H35 saving vs fixed H35: `{100.0 * raw['headline']['oracle_decision_saving_vs_fixed_H35']:.2f}%`.",
        "",
        "## Decision",
        "",
        raw["scientific_decision"],
        "",
        "## Limits",
        "",
        "This is a tiny opened development audit after v29 outcome discovery. A pass only justifies a fresh confirmation protocol; it is not validation64, final test, population evidence, or ORIGINAL SAC reproduction. A fail does not prove adaptivity impossible; it indicates that the current simple feature-rule class is insufficient or overfit for the observed regimes.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v30b three-way selector feature/separability audit

UTC: {raw['created_utc']}. Analysis-only over v29; validation64 closed, sealed test closed, simulations=0, control_steps=0, training/refit=0. This supersedes the failed v30 brute-force analysis attempt, which timed out with exit_status=-15 and no scientific outputs. On the 11 opened v29 states, success-sensitive bad rows: fixed H12={raw['comparators']['fixed_H12']['bad_count']}, fixed H15={raw['comparators']['fixed_H15']['bad_count']}, fixed H25={raw['comparators']['fixed_H25']['bad_count']}, fixed H35={raw['comparators']['fixed_H35']['bad_count']}, oracle fastest-safe H12/H15/H35={raw['comparators']['oracle_fastest_safe_H12_H15_H35']['bad_count']}. Two-threshold feature rule: in-sample bad={raw['learned_in_sample']['bad_count']}, LOO bad={raw['leave_one_out']['bad_count']}, LOO saving vs fixed H35={100.0*h['loo_decision_saving_vs_fixed_H35']:.2f}% if safety holds. Decision: {raw['scientific_decision']} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
<!-- {MARKER} -->
## Follow-up through v30b three-way selector feature/separability audit

Updated by GPT-5.5 executor at `{raw['created_utc']}`. v30b is analysis-only over v29 and did not access validation64 or sealed test. It replaces the timed-out v30 implementation-performance failure.

| linked recommendation(s) | disposition after v30b | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; broadened to H12/H15/H25/H35 evidence | v30b compares fixed H12/H15/H25/H35 and oracle/feature H12/H15/H35 rules on v29 states; success-sensitive bad rows fixed H12={raw['comparators']['fixed_H12']['bad_count']}, fixed H15={raw['comparators']['fixed_H15']['bad_count']}, fixed H25={raw['comparators']['fixed_H25']['bad_count']}, fixed H35={raw['comparators']['fixed_H35']['bad_count']}. | Any further adaptive claim must compare against fixed H35 and fixed H12/H15, and must separate absolute safety from relative timing. |
| `A11_training_failure_modes_need_separation` | accepted; refit/training gate updated | v30b two-threshold feature audit: in-sample bad={raw['learned_in_sample']['bad_count']}; leave-one-out bad={raw['leave_one_out']['bad_count']}; LOO saving vs fixed H35={100.0*raw['headline']['loo_decision_saving_vs_fixed_H35']:.2f}% conditional on zero bad rows. | Use this outcome to decide whether a fresh triage-selector confirmation, terminal-risk/value refit, or scenario redesign is next. |
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; preserved | Source242 rows require H35 in v29/v30b accounting; failed H12/H15/H25 timing is excluded from speed claims. | Continue absolute success-sensitive accounting. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v30b uses only the 11 opened v29 states and selects rules after v29 outcomes. | Development-only mechanism audit; require fresh independent confirmation before validation/final claims. |
| `A12_registry_backup_schema_contract` | accepted; active | v30b wrote analysis/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified backup before more unique simulation/refit/validation. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "deterministic_v29_analysis", "opened_development_v29_only_no_validation64_no_test", 0, 0, 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v30b three-way feature audit\n\nUTC: {raw['created_utc']}\n\nDecision: {raw['scientific_decision']}\n\nHeadline: {json.dumps(clean(raw['headline']), sort_keys=True)}\n\nNext: verified backup covering v30b, then follow the decision. If LOO is safe and saves >=5% vs fixed H35, freeze fresh source-independent triage-selector confirmation. If not, do not validate the current selector; acquire more labels or pivot to terminal-risk/value refit or scenario/comparison redesign.\n", encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-proof", required=True)
    ap.add_argument("--i-accept-development-v30b", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if (RUN_DIR / "completed.json").exists():
            done = read_json(RUN_DIR / "completed.json")
            print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
            return 0
        v29_done = verify_completed(V29_DONE)
        v29_time = parse_time(v29_done.get("created_utc")) or REQUIRE_BACKUP_AFTER
        backup = verify_backup(Path(args.backup_proof), max(REQUIRE_BACKUP_AFTER, v29_time))
        raw29 = read_json(V29_RAW)
        if raw29.get("validation64_bank_opened") is True or raw29.get("sealed_test_accessed") is True or raw29.get("test_accessed") is True:
            raise ContractError("v29 raw unexpectedly reports validation/test access")
        rows = list(raw29["analysis"]["state_rows"])
        if len(rows) != 11:
            raise ContractError(f"unexpected v29 state count {len(rows)}")
        features = build_features(raw29)
        arr = make_arrays(rows)
        comparators: Dict[str, Any] = {f"fixed_H{h}": metric_fixed(rows, arr, h) for h in HORIZONS}
        comparators["oracle_fastest_safe_H12_H15_H35"] = metric_oracle_fastest_safe(rows, arr, SELECTOR_HORIZONS)
        rule, ins_metric, ins_diag = learn_rule(rows, features, arr)
        full_rule_metric = apply_rule(rows, arr, features, rule)
        # Keep the in-sample metric from applying the rule to all rows; the train
        # metric returned by learn_rule is identical here but this checks masks.
        ins_metric = full_rule_metric
        loo_metric = leave_one_out(rows, features, arr)
        fixed_h35 = comparators["fixed_H35"]
        oracle = comparators["oracle_fastest_safe_H12_H15_H35"]
        oracle_saving_vs_h35 = pct_saving(oracle["decision_sum_s"], fixed_h35["decision_sum_s"])
        ins_saving_vs_h35 = pct_saving(ins_metric["decision_sum_s"], fixed_h35["decision_sum_s"])
        loo_saving_vs_h35 = pct_saving(loo_metric["decision_sum_s"], fixed_h35["decision_sum_s"])
        if loo_metric["bad_count"] == 0 and loo_saving_vs_h35 >= 0.05:
            decision = "Feature separability is promising on this tiny opened audit. After backup, freeze a fresh development-only source-independent H12/H15/H35 triage-selector confirmation with fixed H35 and fixed H12/H15 baselines; do not open validation64 or sealed test yet."
        elif oracle["bad_count"] == 0 and oracle_saving_vs_h35 >= 0.05:
            decision = "Oracle triage opportunity is real on v29, but the simple deployable feature rule did not pass leave-one-out. After backup, acquire more pre-outcome source-independent labels or run a bounded terminal-risk/value refit before validating any selector."
        else:
            decision = "v29 does not show enough success-sensitive deployable triage value beyond strong fixed H35; prioritize scenario/comparison redesign or a negative adaptive-opportunity conclusion for this stress-v1 slice."
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_analysis_only_fast_three_way_selector_feature_audit_no_sim_no_validation_no_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "new_simulation_episodes": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit"), "time": backup.get("time")},
            "required_backup_after_utc": REQUIRE_BACKUP_AFTER.isoformat(),
            "v30_timeout_context": {"failed_note": rel(V30_FAILED_NOTE), "failed_note_sha256": sha256(V30_FAILED_NOTE) if V30_FAILED_NOTE.exists() else None, "run_registry": rel(V30_RUN_REGISTRY), "run_registry_sha256": sha256(V30_RUN_REGISTRY) if V30_RUN_REGISTRY.exists() else None},
            "v29_inputs": {"raw": rel(V29_RAW), "raw_sha256": sha256(V29_RAW), "completed": rel(V29_DONE), "completed_sha256": sha256(V29_DONE), "summary": rel(V29_SUMMARY), "summary_sha256": sha256(V29_SUMMARY)},
            "features_excluded_to_avoid_leakage": ["source_candidate_index", "source_campaign", "category", "role", "base_state_id", "state_label", "per_horizon_outcomes", "success/cost/timing fields"],
            "features_used": sorted(next(iter(features.values())).keys()),
            "feature_values_by_state": {bid: features[bid] for bid in sorted(features)},
            "bad_semantics": "bad if unsafe/step-cap, or safe but physical_constraint_cost exceeds best safe horizon for that row by max(2.0, 25% of best-safe physical)",
            "comparators": comparators,
            "learned_in_sample_rule": rule,
            "learned_in_sample_rule_search": ins_diag,
            "learned_in_sample": ins_metric,
            "leave_one_out": loo_metric,
            "headline": {
                "states": len(rows),
                "fixed_H12_bad": comparators["fixed_H12"]["bad_count"],
                "fixed_H15_bad": comparators["fixed_H15"]["bad_count"],
                "fixed_H25_bad": comparators["fixed_H25"]["bad_count"],
                "fixed_H35_bad": comparators["fixed_H35"]["bad_count"],
                "oracle_bad": oracle["bad_count"],
                "oracle_decision_saving_vs_fixed_H35": oracle_saving_vs_h35,
                "in_sample_bad": ins_metric["bad_count"],
                "in_sample_decision_saving_vs_fixed_H35": ins_saving_vs_h35,
                "loo_bad": loo_metric["bad_count"],
                "loo_decision_saving_vs_fixed_H35": loo_saving_vs_h35,
                "loo_h_counts": loo_metric["h_counts"],
            },
            "scientific_decision": decision,
            "interpretation_limits": ["opened v29 development states only", "tiny sample", "rules selected after observing v29 outcomes", "not validation64", "not sealed test", "not population estimate", "not ORIGINAL SAC"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
            "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), V29_RAW, V29_DONE, V29_SUMMARY, V30_FAILED_NOTE, V30_RUN_REGISTRY, Path(args.backup_proof)] if p.exists()},
            "backup_request": rel(BACKUP_REQUEST),
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v30b_three_way_selector_feature_audit", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v30b development analysis/docs/state/registry/response-log", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "new_control_steps": 0, "new_simulation_episodes": 0, "new_training_or_gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        update_docs(raw)
        completed = {"status": "complete", "hard_pass": True, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "new_control_steps": 0, "new_simulation_episodes": 0, "new_training_or_gradient_steps": 0, "headline": raw["headline"], "scientific_decision": decision, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "backup_request": rel(BACKUP_REQUEST), "hashes": {}}
        files = [Path(__file__).resolve(), RUN_DIR / "raw.json", RUN_DIR / "summary.md", V29_RAW, V29_DONE, V29_SUMMARY, V30_FAILED_NOTE, V30_RUN_REGISTRY, STATE, BACKUP_REQUEST, ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
        completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
        write_json(RUN_DIR / "completed.json", completed)
        raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": raw["headline"], "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_analysis_only_fast_three_way_selector_feature_audit_no_sim_no_validation_no_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
