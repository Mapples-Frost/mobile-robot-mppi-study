#!/usr/bin/env python3
"""v34l/A13b residual, eps and label-accessor dissection (zero solve).

Active-lead A13b task after v34k: use already-opened v34g/v34k evidence and a
rebuilt no-solve MPC object to dissect the tiny v34k reconstruction residual,
verify eps/rterm label paths, explain raw_live_label_match:false, and check
whether the displayed max_colloc truncation actually means k>=8 is missing.

Budgets are deliberately zero: no lower-level solver calls, no plant rollouts,
no env.reset/env.step after construction, no selector search/refit, no training,
no validation64 and no sealed/final-test access.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0 as kacc  # noqa:E402
import vehicle_true_variable_horizon_v34i_objective_contract_localization_v0 as i  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34l_residual_eps_label_dissection.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34L_RESIDUAL_EPS_LABEL_DISSECTION_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34l-residual-eps-label-dissection-{STAMP}"
MARKER = f"vehicle-v34l-residual-eps-label-dissection-{STAMP}"

V34G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/raw.json"
V34K_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/completed.json"
V34K_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/raw.json"
V34K_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/candidate_residuals.csv"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T104228Z_96cbf7.md"
OPUS_REPORT_SHA = "f078602ff045367cb2c27b14aa605edff81ac0c075501fd987f138a1335104b3"
OPUS_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
FORMULA_SPEC = {
    "candidate_id": "stage=x0_then_last_node|term=author_last_node|eps=False|r=False|discount=n_horizon_parameter_or_H|z=same_k_last",
    "stage_rule": "x0_then_last_node",
    "terminal_rule": "author_last_node",
    "include_eps": False,
    "include_rterm": False,
    "discount_rule": "n_horizon_parameter_or_H",
    "z_rule": "same_k_last",
}


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(v: Any) -> Any:
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if hasattr(v, "tolist"):
        return clean(v.tolist())
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arr(value: Any) -> np.ndarray:
    return kacc.arr(value)


def finite_float(value: Any) -> Optional[float]:
    return kacc.finite_float(value)


def get_labels(obj: Any) -> List[str]:
    return kacc.get_labels(obj)


def parse_label_any(label: str) -> Tuple[Any, ...]:
    return tuple(kacc.parse_label(label))


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify_inputs() -> Dict[str, Any]:
    for p in [V34G_RAW, V34K_DONE, V34K_RAW, V34K_CSV, OPUS_REPORT, OPUS_READY]:
        if not p.exists():
            raise ContractError(f"missing required input {rel(p)}")
    done = read_json(V34K_DONE)
    raw = read_json(V34K_RAW)
    if done.get("hard_pass") is not True or (raw.get("gates") or {}).get("both_pass") is not True:
        raise ContractError("v34k hard/both pass not present in raw completed artifacts")
    b = done.get("budget_actual") or {}
    for key in ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]:
        if int(b.get(key, 0)) != 0:
            raise ContractError("v34k predecessor was not zero-budget: %s=%r" % (key, b.get(key)))
    if sha256(OPUS_REPORT) != OPUS_REPORT_SHA:
        raise ContractError("active Opus A13b report sha mismatch")
    rows = read_csv_rows(V34K_CSV)
    best_rows = [r for r in rows if r.get("candidate_id") == FORMULA_SPEC["candidate_id"]]
    if not best_rows:
        raise ContractError("frozen best formula not found in v34k candidate CSV")
    return {
        "v34k_completed": rel(V34K_DONE),
        "v34k_raw": rel(V34K_RAW),
        "v34k_candidate_csv": rel(V34K_CSV),
        "v34g_raw": rel(V34G_RAW),
        "opus_report": rel(OPUS_REPORT),
        "opus_report_sha256": OPUS_REPORT_SHA,
        "v34k_hashes": {rel(p): sha256(p) for p in [V34K_DONE, V34K_RAW, V34K_CSV, V34G_RAW]},
        "v34k_budget_actual": b,
        "v34k_headline": done.get("headline"),
        "frozen_formula_row": best_rows[0],
    }


def records_dict(records: Sequence[Tuple[Tuple[Any, ...], float]]) -> Dict[Tuple[Any, ...], float]:
    return {tuple(parts): float(value) for parts, value in records}


def trace_scalar(records: Sequence[Tuple[Tuple[Any, ...], float]], exact: Tuple[Any, ...], fallback_prefix: str, numeric_prefix: Tuple[Any, ...], var: str) -> Dict[str, Any]:
    rd = records_dict(records)
    if exact in rd:
        return {"path": "exact", "exact": list(exact), "value": rd[exact], "fallback_hit_count": None}
    hits: List[Tuple[Tuple[Any, ...], float]] = []
    for parts, value in records:
        if not parts or parts[0] != fallback_prefix:
            continue
        if tuple(parts[1:1 + len(numeric_prefix)]) != tuple(numeric_prefix):
            continue
        if var in parts[1 + len(numeric_prefix):]:
            hits.append((tuple(parts), float(value)))
    if len(hits) == 1:
        return {"path": "fallback", "exact": list(exact), "numeric_prefix": list(numeric_prefix), "value": hits[0][1], "fallback_hit_count": 1, "fallback_label": list(hits[0][0])}
    return {"path": "missing_or_ambiguous", "exact": list(exact), "numeric_prefix": list(numeric_prefix), "value": None, "fallback_hit_count": len(hits), "fallback_labels_first5": [list(h[0]) for h in hits[:5]]}


def eps_dissection(mpc: Any, acc: kacc.LabelAccessor, H: int) -> Dict[str, Any]:
    ns = kacc.n_scenarios(mpc, H)
    rows: List[Dict[str, Any]] = []
    counts = {"exact": 0, "fallback": 0, "missing_or_ambiguous": 0, "zero_values": 0, "nonzero_values": 0}
    for kk in range(H):
        for ss in range(int(ns[kk])):
            for var in acc.eps_vars:
                tr = trace_scalar(acc.unscaled_records, ("_eps", int(kk), int(ss), var, 0), "_eps", (int(kk), int(ss)), str(var))
                val = tr.get("value")
                counts[str(tr["path"])] = counts.get(str(tr["path"]), 0) + 1
                if val is not None and abs(float(val)) <= 1e-12:
                    counts["zero_values"] += 1
                elif val is not None:
                    counts["nonzero_values"] += 1
                if len(rows) < 80:
                    rows.append({"k": kk, "scenario": ss, "var": var, **tr})
    model_eps_labels: List[str] = []
    try:
        model_eps_labels = get_labels(mpc.model._eps)
    except Exception:
        model_eps_labels = []
    current_epsterm_values: List[float] = []
    for kk in range(H):
        for ss in range(int(ns[kk])):
            try:
                current_epsterm_values.append(float(kacc.epsterm_value(mpc, acc, kk, ss, 1.0)))
            except Exception:
                pass
    epsterm_fn_probe: Dict[str, Any] = {"exists": hasattr(mpc, "epsterm_fun")}
    if hasattr(mpc, "epsterm_fun"):
        try:
            n_in = int(mpc.epsterm_fun.n_in()) if hasattr(mpc.epsterm_fun, "n_in") else None
            cur = acc.eps(0, 0)
            zeros = np.zeros_like(cur, dtype=float)
            ones = np.ones_like(cur, dtype=float)
            epsterm_fn_probe.update({
                "n_in": n_in,
                "current_eps_k0": arr(cur).tolist(),
                "value_at_current_k0_gamma_unweighted": finite_float(mpc.epsterm_fun(cur)),
                "value_at_zero_eps": finite_float(mpc.epsterm_fun(zeros)),
                "value_at_one_eps": finite_float(mpc.epsterm_fun(ones)),
            })
        except Exception as exc:
            epsterm_fn_probe.update({"error": repr(exc)})
    rterm_probe: Dict[str, Any] = {"has_rterm_factor": hasattr(mpc, "rterm_factor")}
    if hasattr(mpc, "rterm_factor"):
        try:
            factors = arr(getattr(mpc.rterm_factor, "cat", mpc.rterm_factor)).reshape(-1)
            rterm_probe.update({"factor_values": factors.tolist(), "factor_abs_sum": float(np.sum(np.abs(factors))), "factor_nonzero_count": int(np.sum(np.abs(factors) > 1e-12))})
            u0 = arr(acc.control(0, 0, True)).reshape(-1)
            u_prev = arr(acc.u_prev()).reshape(-1)
            n = min(len(u0), len(u_prev), len(factors))
            rterm_probe["current_k0_unweighted"] = float(np.sum(factors[:n] * (u0[:n] - u_prev[:n]) ** 2)) if n else 0.0
            rterm_probe["unit_delta_unweighted"] = float(np.sum(factors[:n] * np.ones(n))) if n else 0.0
        except Exception as exc:
            rterm_probe.update({"error": repr(exc)})
    if counts["exact"] > 0 and counts["nonzero_values"] == 0 and epsterm_fn_probe.get("value_at_one_eps") not in (None, 0.0):
        classification = "eps_labels_exact_and_objective_branch_active_but_current_solution_eps_zero"
    elif counts["exact"] > 0 and counts["nonzero_values"] == 0 and abs(float(epsterm_fn_probe.get("value_at_one_eps") or 0.0)) <= 1e-12:
        classification = "eps_labels_exact_but_epsterm_function_zero_for_unit_eps"
    elif counts["fallback"] > 0 and counts["exact"] == 0:
        classification = "eps_values_obtained_by_fallback_not_exact"
    elif counts["missing_or_ambiguous"] > 0:
        classification = "eps_accessor_missing_or_ambiguous_for_some_labels"
    else:
        classification = "eps_has_nonzero_current_values_or_mixed"
    return {
        "H": H,
        "n_scenarios_by_k": ns,
        "eps_vars": acc.eps_vars,
        "model_eps_label_count": len(model_eps_labels),
        "model_eps_labels_first20": model_eps_labels[:20],
        "path_counts": counts,
        "trace_rows_first80": rows,
        "current_epsterm_values_abs_sum_gamma1": float(np.sum(np.abs(current_epsterm_values))) if current_epsterm_values else None,
        "current_epsterm_nonzero_count_gamma1": int(np.sum(np.abs(current_epsterm_values) > 1e-12)) if current_epsterm_values else 0,
        "epsterm_function_probe": epsterm_fn_probe,
        "rterm_probe": rterm_probe,
        "classification": classification,
    }


def detailed_terms(mpc: Any, acc: kacc.LabelAccessor, H: int) -> Dict[str, Any]:
    gamma, gsrc = i.gamma_and_source(mpc)
    ns = kacc.n_scenarios(mpc, H)
    by_k: List[Dict[str, Any]] = []
    all_stage_terms: List[float] = []
    for kk in range(H):
        vals: List[float] = []
        for ss in range(int(ns[kk])):
            val = kacc.lterm_value(mpc, acc, kk, ss, gamma, FORMULA_SPEC["stage_rule"], FORMULA_SPEC["z_rule"])
            vals.append(float(val))
            all_stage_terms.append(float(val))
        by_k.append({
            "k": kk,
            "scenario_count": int(ns[kk]),
            "discounted_lterm_sum": float(math.fsum(vals)),
            "discounted_lterm_abs_sum": float(math.fsum(abs(v) for v in vals)),
            "values": vals,
        })
    vf_raw, terminal_disc, terminal_meta = kacc.terminal_value(mpc, acc, H, gamma, FORMULA_SPEC["terminal_rule"], FORMULA_SPEC["discount_rule"])
    solver = finite_float(getattr(mpc, "opt_f_num", None))
    stage_fsum = float(math.fsum(all_stage_terms))
    terminal_disc = float(terminal_disc)
    total_fsum = float(math.fsum(all_stage_terms + [terminal_disc]))
    total_builtin_order = float(sum(all_stage_terms) + terminal_disc)
    total_reversed = float(sum(reversed(all_stage_terms)) + terminal_disc)
    total_numpy = float(np.asarray(all_stage_terms + [terminal_disc], dtype=np.float64).sum())
    total_float32 = float(np.asarray(all_stage_terms + [terminal_disc], dtype=np.float32).sum(dtype=np.float32))
    residual = None if solver is None else float(total_fsum - float(solver))
    abs_residual = None if residual is None else abs(residual)
    sum_abs_terms = float(math.fsum(abs(v) for v in all_stage_terms) + abs(terminal_disc))
    eps64_round_bound = float(sum_abs_terms * np.finfo(np.float64).eps)
    eps32_round_bound = float(sum_abs_terms * np.finfo(np.float32).eps)
    allocation_by_abs: List[Dict[str, Any]] = []
    if residual is not None and sum_abs_terms > 0:
        for row in by_k:
            allocation_by_abs.append({"k": row["k"], "abs_term_sum": row["discounted_lterm_abs_sum"], "proportional_abs_residual_share": float(abs_residual * row["discounted_lterm_abs_sum"] / sum_abs_terms)})
        terminal_share = float(abs_residual * abs(terminal_disc) / sum_abs_terms)
    else:
        terminal_share = None
    by_k_abs_values = [row["discounted_lterm_abs_sum"] for row in by_k]
    solver_stats = (((read_json(V34G_RAW).get("arm") or {}).get("solver_event") or {}).get("solver_stats") or {})
    return {
        "formula_spec": FORMULA_SPEC,
        "gamma": gamma,
        "gamma_source": gsrc,
        "stage_by_k": by_k,
        "terminal": {"vf_raw": vf_raw, "discounted": terminal_disc, "meta": terminal_meta},
        "stage_total_fsum": stage_fsum,
        "total_fsum": total_fsum,
        "solver_objective": solver,
        "residual_total_minus_solver": residual,
        "absolute_residual": abs_residual,
        "relative_residual": None if solver is None or abs_residual is None else float(abs_residual / max(1.0, abs(float(solver)))),
        "alternative_summations": {
            "builtin_order_total": total_builtin_order,
            "builtin_order_minus_fsum": float(total_builtin_order - total_fsum),
            "reversed_order_total": total_reversed,
            "reversed_order_minus_fsum": float(total_reversed - total_fsum),
            "numpy_float64_total": total_numpy,
            "numpy_float64_minus_fsum": float(total_numpy - total_fsum),
            "numpy_float32_total": total_float32,
            "numpy_float32_minus_fsum": float(total_float32 - total_fsum),
        },
        "scale_comparison": {
            "sum_abs_terms": sum_abs_terms,
            "double_precision_one_eps_abs_sum_bound": eps64_round_bound,
            "single_precision_one_eps_abs_sum_bound": eps32_round_bound,
            "abs_residual_over_terminal_discounted": None if terminal_disc == 0 or abs_residual is None else float(abs_residual / abs(terminal_disc)),
            "abs_residual_over_stage_total": None if stage_fsum == 0 or abs_residual is None else float(abs_residual / abs(stage_fsum)),
            "abs_residual_over_min_abs_k_stage": None if not by_k_abs_values or abs_residual is None else float(abs_residual / max(min(x for x in by_k_abs_values if x > 0), 1e-300)),
            "abs_residual_over_max_abs_k_stage": None if not by_k_abs_values or abs_residual is None else float(abs_residual / max(max(by_k_abs_values), 1e-300)),
        },
        "residual_distribution_by_k_proportional_to_abs_term_not_unique": allocation_by_abs,
        "terminal_proportional_abs_residual_share_not_unique": terminal_share,
        "solver_stats_from_v34g": solver_stats,
        "interpretation_note": "No per-k solver-side component is exposed in the saved evidence; stage_by_k is the reconstructed additive distribution. The residual can only be allocated across k by a stated convention unless a solver internal term decomposition is added.",
    }


def raw_live_label_match_dissection(mpc: Any, raw_v34g: Mapping[str, Any]) -> Dict[str, Any]:
    source = Path(i.__file__).resolve()
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    for idx, line in enumerate(lines, start=1):
        if "raw_live_label_match" in line:
            hits.append({"line": idx, "text": line.strip()})
    src_excerpt = []
    if hits:
        lo = max(1, hits[0]["line"] - 8); hi = min(len(lines), hits[0]["line"] + 8)
        src_excerpt = [{"line": n, "text": lines[n - 1]} for n in range(lo, hi + 1)]
    arm = raw_v34g.get("arm") or {}
    post = ((arm.get("solver_event") or {}).get("post") or {})
    raw_sources = {
        "opt_x_num": (((post.get("complete_opt_x_num") or {}).get("labeled_values") or {})),
        "opt_x_num_unscaled": (((post.get("complete_opt_x_num_unscaled") or {}).get("labeled_values") or {})),
        "opt_p_num": (((post.get("complete_opt_p_num") or {}).get("labeled_values") or {})),
    }
    out: Dict[str, Any] = {"write_points": hits, "source_excerpt_around_first_write_point": src_excerpt, "objects": {}}
    for name, raw_labeled in raw_sources.items():
        obj = getattr(mpc, name)
        raw_labels = list(raw_labeled.keys())
        live_labels = get_labels(obj)
        raw_norm = [parse_label_any(x) for x in raw_labels]
        live_norm = [parse_label_any(x) for x in live_labels]
        exact_match = raw_labels == live_labels
        normalized_sequence_match = raw_norm == live_norm
        normalized_set_match = set(raw_norm) == set(live_norm)
        first_mismatch: Optional[Dict[str, Any]] = None
        for n, (rr, ll) in enumerate(zip(raw_labels, live_labels)):
            if rr != ll:
                first_mismatch = {"index": n, "raw_label": rr, "live_label": ll, "raw_norm": list(raw_norm[n]), "live_norm": list(live_norm[n])}
                break
        if first_mismatch is None and len(raw_labels) != len(live_labels):
            first_mismatch = {"index": min(len(raw_labels), len(live_labels)), "reason": "length_mismatch"}
        out["objects"][name] = {
            "raw_label_count": len(raw_labels),
            "live_label_count": len(live_labels),
            "exact_string_sequence_match": exact_match,
            "normalized_sequence_match": normalized_sequence_match,
            "normalized_set_match": normalized_set_match,
            "first_raw_labels": raw_labels[:5],
            "first_live_labels": live_labels[:5],
            "first_mismatch": first_mismatch,
        }
    if all(v.get("normalized_sequence_match") for v in out["objects"].values()) and not all(v.get("exact_string_sequence_match") for v in out["objects"].values()):
        semantics = "diagnostic_string_format_difference_only_normalized_labels_match"
    elif all(v.get("normalized_set_match") for v in out["objects"].values()):
        semantics = "label_set_same_but_order_or_string_format_differs"
    else:
        semantics = "normalized_label_mismatch_possible_live_label_change"
    out["semantic_classification"] = semantics
    return out


def collocation_dissection(acc: kacc.LabelAccessor, H: int) -> Dict[str, Any]:
    table: Dict[int, Dict[str, Any]] = {}
    for parts, _ in acc.unscaled_records:
        if len(parts) >= 6 and parts[0] == "_x" and isinstance(parts[1], int) and isinstance(parts[2], int) and isinstance(parts[3], int):
            kk = int(parts[1]); ss = int(parts[2]); cc = int(parts[3])
            row = table.setdefault(kk, {"scenarios": {}, "colloc_values": set(), "record_count": 0})
            row["record_count"] += 1
            row["colloc_values"].add(cc)
            skey = str(ss)
            row["scenarios"].setdefault(skey, set()).add(cc)
    serial: Dict[str, Any] = {}
    for kk, row in sorted(table.items()):
        serial[str(kk)] = {
            "record_count": int(row["record_count"]),
            "colloc_values": sorted(int(x) for x in row["colloc_values"]),
            "scenarios": {str(s): sorted(int(x) for x in vals) for s, vals in sorted(row["scenarios"].items())},
        }
    missing_display = [kk for kk in range(H + 1) if kk not in table]
    full_keys = {f"{kk}:{ss}": int(cc) for (kk, ss), cc in sorted(acc.max_colloc.items())}
    if not missing_display and len(full_keys) > len((acc.diagnostics().get("max_colloc_first") or {})):
        reason = "v34k_raw_display_truncated_max_colloc_first_to_first_8_entries; full label table contains all k values"
    elif missing_display:
        reason = "some_k_values_absent_from_unscaled_x_labels"
    else:
        reason = "all_k_values_present_without_truncation_issue"
    return {
        "H": H,
        "x_label_k_table": serial,
        "missing_k_values_in_full_unscaled_x_labels": missing_display,
        "full_max_colloc_keys": full_keys,
        "full_max_colloc_key_count": len(full_keys),
        "v34k_diagnostics_max_colloc_first_only": acc.diagnostics().get("max_colloc_first"),
        "classification": reason,
    }


def fixed_h_fairness_note(raw_v34k: Mapping[str, Any]) -> Dict[str, Any]:
    """Zero-cost text/evidence scan for the D2 carry-forward note.

    This deliberately does not assert a final fixed-H fairness conclusion. It
    records whether the v34k raw contains the H50/15000-step clue cited by Opus.
    """
    raw_text = json.dumps(raw_v34k, sort_keys=True)
    snippets: List[str] = []
    for pat in ["vehicle_fixed_h50", "terminal_source_horizon", "15000", "H=50", "horizon\": 50"]:
        idx = raw_text.find(pat)
        if idx >= 0:
            snippets.append(raw_text[max(0, idx - 180): idx + 240])
    return {
        "status": "carry_forward_not_full_fairness_audit",
        "searched_v34k_raw_for_cited_clue": True,
        "found_snippet_count": len(snippets),
        "snippets_first5": snippets[:5],
        "conclusion": "D2 remains open: this run only preserves the clue and does not establish strong fixed-H fairness across H/seed budgets.",
    }


def write_outputs(raw: Mapping[str, Any]) -> None:
    dtl = raw["residual_dissection"]
    eps = raw["eps_dissection"]
    lbl = raw["raw_live_label_match_dissection"]
    col = raw["collocation_dissection"]
    flags = raw["mechanical_flags"]
    lines = [
        "# v34l/A13b residual, eps and label dissection",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve/zero-plant diagnostic using already-opened v34g/v34k evidence.",
        "",
        "## Budget",
        f"- solver calls: `{raw['budget_actual']['solver_calls']}`; plant steps: `{raw['budget_actual']['plant_steps']}`; env_reset/env_step after construction: `{raw['budget_actual']['env_reset_calls_after_construction']}`/`{raw['budget_actual']['env_step_calls_after_construction']}`; validation64: `0`; sealed test: `0`.",
        "",
        "## Residual dissection",
        f"- formula: `{FORMULA_SPEC['candidate_id']}`",
        f"- solver objective: `{dtl['solver_objective']}`; reconstructed total: `{dtl['total_fsum']}`; total-minus-solver: `{dtl['residual_total_minus_solver']}`; abs residual: `{dtl['absolute_residual']}`; relative: `{dtl['relative_residual']}`.",
        f"- stage total: `{dtl['stage_total_fsum']}`; terminal discounted: `{dtl['terminal']['discounted']}`.",
        f"- alternative summation deltas: `{dtl['alternative_summations']}`.",
        f"- scale comparison: `{dtl['scale_comparison']}`.",
        "- per-k reconstructed discounted stage sums are in raw.json under `residual_dissection.stage_by_k`; proportional residual allocation is explicitly labelled non-unique.",
        "",
        "## eps/rterm dissection",
        f"- eps classification: `{eps['classification']}`; eps path counts: `{eps['path_counts']}`; epsterm probe: `{eps['epsterm_function_probe']}`.",
        f"- rterm probe: `{eps['rterm_probe']}`.",
        "",
        "## raw_live_label_match and max_colloc",
        f"- raw_live_label_match semantics: `{lbl['semantic_classification']}`; write points: `{lbl['write_points']}`.",
        f"- max_colloc classification: `{col['classification']}`; full key count: `{col['full_max_colloc_key_count']}`; missing k values: `{col['missing_k_values_in_full_unscaled_x_labels']}`.",
        "",
        "## Mechanical next-gate flags",
        f"- `{flags}`",
        "",
        f"Artifacts: `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    block = f"""
<!-- {MARKER} -->
## v34l/A13b residual, eps and label dissection

UTC: {raw['created_utc']}. Ran Opus A13b as a zero-solve/zero-plant diagnostic over already-opened v34g/v34k evidence. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0/0, training/refit=0, validation64=0, sealed_test=0. Formula `{FORMULA_SPEC['candidate_id']}` produced solver={dtl['solver_objective']}, reconstructed={dtl['total_fsum']}, abs_residual={dtl['absolute_residual']}, rel_residual={dtl['relative_residual']}; alternative summation deltas={dtl['alternative_summations']}. eps classification={eps['classification']} with path_counts={eps['path_counts']}; rterm probe={eps['rterm_probe']}. raw_live_label_match semantics={lbl['semantic_classification']}; max_colloc classification={col['classification']}. Mechanical flags={flags}. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "gate_evidence_ready" if flags.get("a13c_allowed_by_mechanical_checks") else "analysis_requested",
        "trigger": "v34l/A13b zero-solve residual/eps/label dissection completed",
        "experiment_id": NAME,
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "question": "Review A13b dissection. If mechanical flags support it, the active plan already authorizes A13c <=6-solve non-optimal reconstruction consistency; otherwise diagnose the listed blocker. Preserve development-only/no validation64/no sealed-test limits.",
        "evidence_paths": [rel(RUN_DIR/"summary.md"), rel(RUN_DIR/"raw.json"), rel(RUN_DIR/"completed.json"), rel(V34K_RAW), rel(V34K_CSV), rel(OPUS_REPORT), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "mechanical_flags": flags,
        "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
    })
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34l_residual_eps_label_dissection",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new A13b zero-solve diagnostic source/output/docs after v34k hard-pass must be externally recoverable before any A13c solver calls or further science",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "after verified backup, run A13c <=6-solve non-optimal reconstruction consistency only if mechanical flags allow; do not run 23-call Task-C probe under the latest Opus plan before A13c gate",
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Continue state after v34l/A13b residual eps label dissection\n\nUTC: {raw['created_utc']}\n\nMechanical flags: {json.dumps(clean(flags), sort_keys=True)}\n\nArtifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'completed.json')}\n\nNext: verify external backup for {rel(BACKUP_REQUEST)}. Latest Opus plan says run A13c <=6-solve non-optimal reconstruction consistency after A13b if not blocked; do NOT run the 23-call Task-C/v34j probe until A13c passes.\n",
        encoding="utf-8",
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--i-accept-v34l-zero-solve-a13b-dissection", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        inputs = verify_inputs()
        # Rebuild and assign the already-opened v34g solution arrays; patch counter
        # guards come from the v34h/i no-solve runtime.
        i.h.patch_no_solve_runtime()
        raw_v34g = read_json(V34G_RAW)
        raw_v34k = read_json(V34K_RAW)
        mpc, build_meta = i.h.build_mpc_without_solve()
        assign_meta = i.assign_saved_arrays(mpc, raw_v34g)
        accessor = kacc.LabelAccessor(mpc)
        H = int(kacc.TARGET["horizon"])
        residual = detailed_terms(mpc, accessor, H)
        eps = eps_dissection(mpc, accessor, H)
        label_match = raw_live_label_match_dissection(mpc, raw_v34g)
        colloc = collocation_dissection(accessor, H)
        fixedh = fixed_h_fairness_note(raw_v34k)
        forbidden = build_meta.get("forbidden_call_counter", {})
        budget_actual = {
            "solver_calls": int(forbidden.get("solve", 0)),
            "plant_steps": 0,
            "env_reset_calls_after_construction": int(forbidden.get("env_reset", 0)),
            "env_step_calls_after_construction": int(forbidden.get("env_step", 0)),
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        }
        if any(int(budget_actual[k]) != 0 for k in ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]):
            raise ContractError("zero-budget contract violated: " + repr(budget_actual))
        # Mechanical, non-scientific gating flags. The active lead owns causal
        # interpretation; these flags only state whether a hard operational
        # blocker was found by A13b.
        abs_res = residual.get("absolute_residual")
        eps_blocker = eps.get("classification") in ("eps_values_obtained_by_fallback_not_exact", "eps_accessor_missing_or_ambiguous_for_some_labels")
        label_blocker = label_match.get("semantic_classification") == "normalized_label_mismatch_possible_live_label_change"
        colloc_blocker = colloc.get("classification") == "some_k_values_absent_from_unscaled_x_labels"
        summation_explains_residual = False
        try:
            deltas = residual.get("alternative_summations") or {}
            max_alt = max(abs(float(deltas.get(k, 0.0))) for k in ["builtin_order_minus_fsum", "reversed_order_minus_fsum", "numpy_float64_minus_fsum"])
            summation_explains_residual = abs_res is not None and max_alt >= 0.5 * float(abs_res)
        except Exception:
            max_alt = None
        missing_formula_blocker = bool(eps_blocker or label_blocker or colloc_blocker)
        flags = {
            "a13b_completed": True,
            "eps_accessor_or_zero_fill_blocker": bool(eps_blocker),
            "raw_live_label_normalized_mismatch_blocker": bool(label_blocker),
            "collocation_missing_k_blocker": bool(colloc_blocker),
            "missing_formula_blocker_detected_by_a13b": bool(missing_formula_blocker),
            "double_precision_summation_explains_residual": bool(summation_explains_residual),
            "max_float64_order_delta_vs_abs_residual": None if max_alt is None else float(max_alt),
            "a13c_allowed_by_mechanical_checks": not missing_formula_blocker,
            "v34j_23_call_probe_still_blocked_until_a13c_passes_under_latest_opus_plan": True,
        }
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_zero_solve_a13b_residual_eps_label_dissection_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(OPUS_REPORT),
            "lead_report_sha256": OPUS_REPORT_SHA,
            "hypothesis_frozen": "A13b should determine whether the v34k 3.66e-4 residual and eps/rterm zero axes are operational artifacts before any non-optimal reconstruction consistency solves or 23-call objective-vs-basin probe.",
            "input_gate": inputs,
            "runtime_build_meta": build_meta,
            "assignment_meta": assign_meta,
            "residual_dissection": residual,
            "eps_dissection": eps,
            "raw_live_label_match_dissection": label_match,
            "collocation_dissection": colloc,
            "fixed_h_fairness_carryforward_note": fixedh,
            "mechanical_flags": flags,
            "budget_declared": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": budget_actual,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "interpretation_limits": ["development-only", "already-opened v34g/v34k evidence", "zero new solver calls", "not validation64", "not sealed/final test", "no selector policy or closed-loop claim", "A13c and 23-call Task-C remain separate dependent tasks"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        }
        write_json(RUN_DIR / "raw.json", raw)
        # Separate compact CSV for per-k terms and non-unique residual allocation.
        with (RUN_DIR / "stage_by_k.csv").open("w", encoding="utf-8", newline="") as fcsv:
            fields = ["k", "scenario_count", "discounted_lterm_sum", "discounted_lterm_abs_sum", "proportional_abs_residual_share_not_unique"]
            writer = csv.DictWriter(fcsv, fieldnames=fields)
            writer.writeheader()
            alloc_by_k = {int(r["k"]): r.get("proportional_abs_residual_share") for r in residual.get("residual_distribution_by_k_proportional_to_abs_term_not_unique", [])}
            for row in residual.get("stage_by_k", []):
                writer.writerow({
                    "k": row.get("k"),
                    "scenario_count": row.get("scenario_count"),
                    "discounted_lterm_sum": row.get("discounted_lterm_sum"),
                    "discounted_lterm_abs_sum": row.get("discounted_lterm_abs_sum"),
                    "proportional_abs_residual_share_not_unique": alloc_by_k.get(int(row.get("k"))),
                })
        write_outputs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {
            "status": "complete",
            "passed": True,
            "hard_pass": True,
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": budget_actual,
            "headline": {
                "absolute_residual": residual.get("absolute_residual"),
                "relative_residual": residual.get("relative_residual"),
                "eps_classification": eps.get("classification"),
                "eps_path_counts": eps.get("path_counts"),
                "raw_live_label_match_semantics": label_match.get("semantic_classification"),
                "collocation_classification": colloc.get("classification"),
                "mechanical_flags": flags,
            },
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "stage_by_k_csv": rel(RUN_DIR / "stage_by_k.csv"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
            "hashes": hash_existing(files),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "stage_by_k_csv": rel(RUN_DIR / "stage_by_k.csv"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_zero_solve_a13b_residual_eps_label_dissection_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34l/A13b residual eps label dissection failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo validation64 or sealed test access was requested. Preserve failure; repair operational script bug before A13c or any 23-call probe.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
