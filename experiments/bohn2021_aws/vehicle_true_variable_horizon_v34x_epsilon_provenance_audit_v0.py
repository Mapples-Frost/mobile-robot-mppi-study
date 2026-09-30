#!/usr/bin/env python3
"""v34x / T-C1 OC-epsilon-1a zero-solve epsilon provenance audit.

This development diagnostic implements the active Opus T-C1 instruction from
`docs/bohn2021_takeover/opus_lead/20260930T130507Z_6f3456.md`.

It performs no optimization, no plant rollout, no env.step/reset, no training,
no validation64 access and no sealed/final-test access. It inspects source text
and already-written v34u/v34v artifacts to determine whether the epsilon/slack
objective contribution was computed from the solver-state slack variables and
fixed source coefficients rather than back-derived from the residual.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v34x_epsilon_provenance_audit_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34x_epsilon_provenance_audit.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34X_EPSILON_PROVENANCE_AUDIT_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T130507Z_6f3456.md"
OPUS_REQUEST = "execution-result:20260930T130525_b585d5b0"
OPUS_REPORT_SHA = "d600ffca526977daac9169b97fe7b5ae1bff18befeb9428b93b9d2e590d8c276"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V34U_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z"
V34V_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34v_eps_objective_localization_postdiagnostic_v0_20260930T130526Z"
V34U_RAW = V34U_DIR / "raw.json"
V34U_DONE = V34U_DIR / "completed.json"
V34V_RAW = V34V_DIR / "raw.json"
V34V_CSV = V34V_DIR / "eps_alias_localization.csv"
V34V_DONE = V34V_DIR / "completed.json"

SOURCE_CONTROLLER = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/controller.py"
SOURCE_OPTIMIZER = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/optimizer.py"
SOURCE_GYM_CONTROLLER = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-horizon/gym_let_mpc/controllers.py"
SOURCE_V34K = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0.py"
MARKER = f"vehicle-v34x-epsilon-provenance-audit-{STAMP}"
REQUEST_ID = f"v34x-epsilon-provenance-audit-{STAMP}"


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
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
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


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def fnum(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        out = float(v)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def find_snippets(path: Path, patterns: Mapping[str, str], context: int = 2) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out: Dict[str, Any] = {"path": rel(path), "sha256": sha256(path), "matches": {}}
    for label, pattern in patterns.items():
        hits: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines, start=1):
            if pattern in line:
                lo = max(1, idx - context)
                hi = min(len(lines), idx + context)
                hits.append({
                    "line": idx,
                    "pattern": pattern,
                    "snippet": [{"line": n, "text": lines[n - 1]} for n in range(lo, hi + 1)],
                })
        out["matches"][label] = hits
    return out


def verify_active_plan() -> Dict[str, Any]:
    ready = read_json(PLAN_READY)
    latest_text = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    report_text = OPUS_REPORT.read_text(encoding="utf-8", errors="replace") if OPUS_REPORT.exists() else ""
    actual_sha = sha256(OPUS_REPORT) if OPUS_REPORT.exists() else None
    checks = {
        "plan_ready_request_matches": ready.get("request_id") == OPUS_REQUEST,
        "plan_ready_report_sha_matches_expected": ready.get("report_sha256") == OPUS_REPORT_SHA,
        "report_file_sha_matches_expected": actual_sha == OPUS_REPORT_SHA,
        "latest_points_to_report": rel(OPUS_REPORT) in latest_text,
        "report_authorizes_T_C1": "T-C1" in report_text and "OC-ε-1a" in report_text,
        "report_authorizes_T_C2_dependency": "T-C2" in report_text and "Depends on T-C1 passing" in report_text,
    }
    return {
        "ready": ready,
        "report": rel(OPUS_REPORT),
        "report_sha256_actual": actual_sha,
        "checks": checks,
        "pass": all(checks.values()),
    }


def parse_candidate_id(candidate_id: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for token in str(candidate_id).split("|"):
        if "=" in token:
            k, v = token.split("=", 1)
            parts[k] = v
    return parts


def normalized_pair_key(candidate_id: str) -> str:
    parts = parse_candidate_id(candidate_id)
    parts["eps"] = "*"
    return "|".join(f"{k}={parts.get(k, '')}" for k in ["stage", "term", "eps", "r", "discount", "z"])


def scan_for_serialized_eps_vectors(obj: Any, path: str = "root", hits: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if hits is None:
        hits = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k)
            new_path = f"{path}.{key}"
            if "eps" in key.lower() or "slack" in key.lower():
                if isinstance(v, (int, float, str, type(None))):
                    preview = v
                    size = 1
                elif isinstance(v, list):
                    preview = v[:8]
                    size = len(v)
                elif isinstance(v, Mapping):
                    preview = list(v.keys())[:8]
                    size = len(v)
                else:
                    preview = repr(type(v))
                    size = None
                hits.append({"path": new_path, "type": type(v).__name__, "size": size, "preview": preview})
            scan_for_serialized_eps_vectors(v, new_path, hits)
    elif isinstance(obj, list):
        for idx, v in enumerate(obj[:500]):
            scan_for_serialized_eps_vectors(v, f"{path}[{idx}]", hits)
    return hits


def source_audit() -> Dict[str, Any]:
    controller_patterns = {
        "epsilon_variables_in_nlp_decision_vector": "entry('_eps', repeat=[self.n_horizon, n_max_scenarios], struct=self._eps)",
        "stage_lterm_objective": "obj += self.discount_factor ** k * omega[k] * self.lterm_fun",
        "stage_epsterm_objective": "obj += self.discount_factor ** k * self.epsterm_fun(opt_x_unscaled['_eps', k, s])",
        "terminal_objective_addition": "obj += term_cost",
        "rterm_objective_addition": "obj += self.discount_factor ** k * self.rterm_factor.cat.T@",
        "epsilon_lower_bound": "self.lb_opt_x['_eps'] = self._eps_lb.cat",
        "epsilon_upper_bound": "self.ub_opt_x['_eps'] = self._eps_ub.cat",
    }
    optimizer_patterns = {
        "soft_constraint_symbol": "epsilon = SX.sym('eps_'+expr_name,*expr.shape)",
        "soft_constraint_expression_minus_epsilon": "expr = expr-epsilon",
        "slack_variable_registered": "{'slack_name': expr_name, 'var': epsilon, 'ub': maximum_violation}",
        "slack_cost_penalty_term": "self.slack_cost += sum1(penalty_term_cons*epsilon)",
        "epsterm_fun_definition": "self.epsterm_fun = Function('epsterm', [_eps], [self.slack_cost])",
    }
    gym_patterns = {
        "mpc_set_constraints_passes_cost": "mpc.set_nl_cons(expr_name, eval(expr), ub=c[\"value\"], soft_constraint=c.get(\"soft\", False), penalty_term_cons=c[\"cost\"])",
        "tta_obstacle_soft_loop": "for soft_constraint in [True, False]:",
        "tta_obstacle_cost_1000": '"cost": 1000',
        "tta_soft_name": '"obj_{}_distance-{}".format(obj_i, "s" if soft_constraint else "h")',
    }
    v34k_patterns = {
        "label_accessor_eps_from_unscaled_opt_x": "def eps(self, k: int, s: int = 0) -> np.ndarray:",
        "formula_calls_epsterm_fun": "eps_terms.append(epsterm_value(mpc, acc, k, s, gamma))",
        "epsterm_value_function": "def epsterm_value(mpc: Any, acc: LabelAccessor, k: int, s: int, gamma: float) -> float:",
        "epsterm_fun_eval_on_acc_eps": "call_scalar(mpc.epsterm_fun, (acc.eps(k, s),), f\"epsterm k={k} s={s}\")",
    }
    audit = {
        "controller": find_snippets(SOURCE_CONTROLLER, controller_patterns),
        "optimizer": find_snippets(SOURCE_OPTIMIZER, optimizer_patterns),
        "gym_controller": find_snippets(SOURCE_GYM_CONTROLLER, gym_patterns),
        "v34k_formula_source": find_snippets(SOURCE_V34K, v34k_patterns),
    }
    checks = {
        "controller_has_epsilon_decision_variables": bool(audit["controller"]["matches"]["epsilon_variables_in_nlp_decision_vector"]),
        "controller_adds_discounted_epsterm_to_objective": bool(audit["controller"]["matches"]["stage_epsterm_objective"]),
        "optimizer_creates_epsilon_for_soft_constraints": bool(audit["optimizer"]["matches"]["soft_constraint_symbol"] and audit["optimizer"]["matches"]["soft_constraint_expression_minus_epsilon"]),
        "optimizer_slack_cost_is_penalty_times_epsilon": bool(audit["optimizer"]["matches"]["slack_cost_penalty_term"]),
        "optimizer_defines_epsterm_fun_from_slack_cost": bool(audit["optimizer"]["matches"]["epsterm_fun_definition"]),
        "vehicle_passes_constraint_cost_to_do_mpc": bool(audit["gym_controller"]["matches"]["mpc_set_constraints_passes_cost"]),
        "vehicle_obstacle_soft_constraint_cost_1000_present": bool(audit["gym_controller"]["matches"]["tta_obstacle_cost_1000"] and audit["gym_controller"]["matches"]["tta_soft_name"]),
        "v34k_epsterm_evaluator_uses_solver_state_epsilon": bool(audit["v34k_formula_source"]["matches"]["label_accessor_eps_from_unscaled_opt_x"] and audit["v34k_formula_source"]["matches"]["epsterm_fun_eval_on_acc_eps"]),
    }
    audit["checks"] = checks
    audit["pass"] = all(checks.values())
    audit["algebraic_identity"] = (
        "For each prediction stage k and scenario s, do-mpc source adds "
        "discount_factor**k * epsterm_fun(opt_x_unscaled['_eps', k, s]) to the NLP objective. "
        "Optimizer.set_nl_cons creates epsilon slack variables for soft constraints, changes each soft constraint to expr - epsilon <= ub, "
        "and defines epsterm_fun(_eps)=sum(penalty_term_cons*epsilon). The vehicle TTAHMPC source supplies penalty_term_cons=c['cost']; "
        "the obstacle soft-constraint block sets cost=1000. Therefore for the opened vehicle obstacle soft constraints, "
        "J_epsilon = sum_k gamma^k * sum_j 1000 * epsilon_{k,j}, with epsilon constrained by the do-mpc _eps bounds and not by any fitted residual coefficient."
    )
    return audit


def raw_artifact_audit() -> Dict[str, Any]:
    raw_u = read_json(V34U_RAW)
    done_u = read_json(V34U_DONE)
    raw_v = read_json(V34V_RAW)
    done_v = read_json(V34V_DONE)
    v34v_rows = read_csv(V34V_CSV)
    v34v_by_arm = {str(r.get("arm_id")): r for r in v34v_rows}
    per_arm: List[Dict[str, Any]] = []
    for arm in raw_u.get("arms", []):
        arm_id = str(arm.get("arm_id"))
        main = arm.get("objective_reconstruction") or {}
        eps = (arm.get("alias_branch_variants") or {}).get("eps_only") or {}
        both = (arm.get("alias_branch_variants") or {}).get("eps_and_rterm") or {}
        rterm = (arm.get("alias_branch_variants") or {}).get("rterm_only") or {}
        alias_metrics = main.get("alias_16_candidate_metrics") or []
        grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        for row in alias_metrics:
            cid = str(row.get("candidate_id") or "")
            parts = parse_candidate_id(cid)
            if "eps" not in parts:
                continue
            grouped.setdefault(normalized_pair_key(cid), {})[parts["eps"]] = row
        pair_deltas: List[float] = []
        pair_ids: List[str] = []
        for key, grp in grouped.items():
            if "True" in grp and "False" in grp:
                te = fnum(grp["True"].get("total"))
                tn = fnum(grp["False"].get("total"))
                if te is not None and tn is not None:
                    pair_deltas.append(float(te - tn))
                    pair_ids.append(key)
        epsterm_total = fnum(eps.get("epsterm_total"))
        eps_delta_v34u = fnum((arm.get("alias_separation") or {}).get("eps_delta_vs_main"))
        eps_delta_v34v = fnum((v34v_by_arm.get(arm_id) or {}).get("eps_delta_vs_main"))
        pair_min = min(pair_deltas) if pair_deltas else None
        pair_max = max(pair_deltas) if pair_deltas else None
        pair_width = None if pair_min is None or pair_max is None else float(pair_max - pair_min)
        pair_vs_epsterm_max_abs = None
        if epsterm_total is not None and pair_deltas:
            pair_vs_epsterm_max_abs = max(abs(d - epsterm_total) for d in pair_deltas)
        v34v_vs_epsterm_abs = None if epsterm_total is None or eps_delta_v34v is None else abs(epsterm_total - eps_delta_v34v)
        per_arm.append({
            "arm_id": arm_id,
            "role": arm.get("cell_role"),
            "horizon": arm.get("horizon"),
            "initialization": arm.get("initialization"),
            "return_status": (arm.get("solver_event_summary") or {}).get("return_status"),
            "solver_success": (arm.get("solver_event_summary") or {}).get("success"),
            "solver_objective_from_eps_variant": eps.get("solver_objective"),
            "J_noeps_main": main.get("total"),
            "J_eps_only": eps.get("total"),
            "J_eps_and_rterm": both.get("total"),
            "J_rterm_only": rterm.get("total"),
            "epsterm_total_from_v34u_formula_components": epsterm_total,
            "eps_nonzero_count_from_v34u_formula_components": eps.get("eps_nonzero_count"),
            "rterm_total_from_v34u_formula_components": rterm.get("input_regularization_total"),
            "eps_delta_vs_main_v34u_alias_separation": eps_delta_v34u,
            "eps_delta_vs_main_v34v_csv": eps_delta_v34v,
            "alias_pair_delta_count": len(pair_deltas),
            "alias_pair_delta_min": pair_min,
            "alias_pair_delta_max": pair_max,
            "alias_pair_delta_width": pair_width,
            "alias_pair_delta_example_keys": pair_ids[:4],
            "max_abs_alias_pair_delta_minus_epsterm_total": pair_vs_epsterm_max_abs,
            "abs_v34v_eps_delta_minus_epsterm_total": v34v_vs_epsterm_abs,
            "direct_f_rel_error_vs_solver": (arm.get("direct_nlp_eval") or {}).get("direct_f_rel_error_vs_solver"),
            "direct_g_linf_vs_saved": (arm.get("direct_nlp_eval") or {}).get("g_eval_minus_saved_opt_g_linf"),
            "residual_not_used_for_this_check": True,
            "independent_contribution_reproduced_without_abs_residual": bool(
                epsterm_total is not None
                and pair_deltas
                and pair_vs_epsterm_max_abs is not None and pair_vs_epsterm_max_abs <= 1e-8
                and v34v_vs_epsterm_abs is not None and v34v_vs_epsterm_abs <= 1e-8
            ),
        })
    eps_hits = scan_for_serialized_eps_vectors(raw_u)
    likely_vector_hits = [h for h in eps_hits if any(tok in h["path"].lower() for tok in ["opt_x", "eps_vector", "epsilon_vector", "_eps"])]
    return {
        "v34u_raw": rel(V34U_RAW),
        "v34u_completed": rel(V34U_DONE),
        "v34v_raw": rel(V34V_RAW),
        "v34v_completed": rel(V34V_DONE),
        "v34v_eps_csv": rel(V34V_CSV),
        "input_hashes": {rel(p): sha256(p) for p in [V34U_RAW, V34U_DONE, V34V_RAW, V34V_CSV, V34V_DONE]},
        "v34u_headline": done_u.get("headline"),
        "v34v_headline": done_v.get("headline"),
        "per_arm": per_arm,
        "serialized_epsilon_key_hits_sample": eps_hits[:60],
        "likely_full_epsilon_vector_hits_sample": likely_vector_hits[:20],
        "individual_epsilon_vector_values_serialized_in_v34u_raw": bool(likely_vector_hits),
        "note_on_vector_values": "v34u raw preserves epsterm_total and eps_nonzero_count from formula_components, and alias candidate totals for all 16 formulas. It does not appear to serialize the full per-stage _eps numeric vector; T-C2 should explicitly persist that vector at converged and truncated points.",
    }


def write_outputs(raw: Mapping[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    per_arm = raw["artifact_audit"]["per_arm"]
    csv_path = RUN_DIR / "epsilon_provenance_per_arm.csv"
    fields = [
        "arm_id", "role", "horizon", "initialization", "return_status", "solver_success",
        "J_noeps_main", "J_eps_only", "J_eps_and_rterm", "J_rterm_only",
        "epsterm_total_from_v34u_formula_components", "eps_nonzero_count_from_v34u_formula_components",
        "eps_delta_vs_main_v34u_alias_separation", "eps_delta_vs_main_v34v_csv",
        "alias_pair_delta_count", "alias_pair_delta_min", "alias_pair_delta_max", "alias_pair_delta_width",
        "max_abs_alias_pair_delta_minus_epsterm_total", "abs_v34v_eps_delta_minus_epsterm_total",
        "direct_f_rel_error_vs_solver", "direct_g_linf_vs_saved", "residual_not_used_for_this_check",
        "independent_contribution_reproduced_without_abs_residual",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_arm:
            w.writerow({k: clean(row.get(k)) for k in fields})

    h = raw["headline"]
    source = raw["source_audit"]
    lines = [
        "# v34x / T-C1 epsilon provenance audit",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve source/artifact provenance audit under active Opus T-C1 (`OC-epsilon-1a`).",
        "",
        "## Budget and isolation",
        "- New solver calls: `0`; plant/env.step/reset after construction: `0`; training/refit: `0`; validation64: `0`; sealed/final test: `0`.",
        f"- Active plan verified: `{raw['plan_check']['pass']}` for `{OPUS_REQUEST}`.",
        "",
        "## Gate result",
        f"- T-C1 provenance pass: `{h['T_C1_pass']}`.",
        f"- Source identity pass: `{h['source_identity_pass']}`.",
        f"- Independent epsilon-contribution reproduction count: `{h['independent_contribution_reproduced_count']}` / `{h['scheduled_cells']}`.",
        f"- Individual full epsilon vectors serialized in v34u raw: `{h['individual_epsilon_vector_values_serialized_in_v34u_raw']}`.",
        "",
        "## Source-level objective identity",
        source["algebraic_identity"],
        "",
        "Relevant line matches are stored in `raw.json` with exact source paths, line numbers and snippets. Key checks:",
    ]
    for k, v in source["checks"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Per-arm non-circular epsilon contribution check",
        "",
        "The comparison below uses v34u formula-components totals and paired eps/no-eps alias totals; it does not use `abs_residual` as an input.",
        "",
        "| role | H | status | epsterm_total | v34v eps_delta | alias pair width | max |pair-epsterm| | reproduced |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in per_arm:
        lines.append(
            f"| `{r['role']}` | {r['horizon']} | `{r['return_status']}` | "
            f"{r['epsterm_total_from_v34u_formula_components']} | {r['eps_delta_vs_main_v34v_csv']} | "
            f"{r['alias_pair_delta_width']} | {r['max_abs_alias_pair_delta_minus_epsterm_total']} | "
            f"`{r['independent_contribution_reproduced_without_abs_residual']}` |"
        )
    lines += [
        "",
        "## Limitation carried to T-C2",
        raw["artifact_audit"]["note_on_vector_values"],
        "This limitation does not make the epsilon contribution circular: v34u source computes `epsterm_total` by evaluating `mpc.epsterm_fun(acc.eps(k,s))` on solver-state `_eps` labels, and the paired alias deltas reproduce the same value without subtracting `abs_residual`. However, the next solver-bearing T-C2 run should explicitly persist per-stage epsilon values.",
        "",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`; CSV: `{rel(csv_path)}`; completed: `{rel(RUN_DIR/'completed.json')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    summary_path = RUN_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34x_epsilon_provenance_audit",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "T-C1 zero-solve provenance audit created new source/output/docs; preserve before the T-C2 solver-bearing gate.",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "env_step_calls_after_construction": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "If active plan remains unchanged, T-C2 is authorized after this pass and external backup. T-C2 must persist per-stage epsilon vectors. Do not open validation64/sealed test.",
    })

    if not h["T_C1_pass"]:
        write_json(NEXT_REVIEW_REQUEST, {
            "request_id": REQUEST_ID,
            "created": raw["created_utc"],
            "status": "analysis_requested",
            "trigger": "T-C1 epsilon provenance audit failed or was incomplete",
            "experiment_id": NAME,
            "active_lead_expected": "claude-opus-5-5",
            "question": "T-C1 did not satisfy the provenance gate. Review raw evidence and decide whether epsilon attribution is circular/incomplete or whether a repaired data-capture diagnostic is authorized. Do not open validation64/sealed test.",
            "evidence_paths": [rel(summary_path), rel(RUN_DIR/"raw.json"), rel(csv_path), rel(RUN_DIR/"completed.json")],
            "headline": h,
            "budget_actual": raw["budget_actual"],
            "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
        })

    block = f"""
<!-- {MARKER} -->
## v34x / T-C1 epsilon provenance audit

UTC: {raw['created_utc']}. Zero-solve source/artifact provenance audit. Budget: solver=0, plant/env.step/reset-after-construction=0, training/refit=0, validation64=0, sealed/final test=0. Active Opus plan `{OPUS_REQUEST}` verified={raw['plan_check']['pass']}. T-C1_pass={h['T_C1_pass']}; source_identity_pass={h['source_identity_pass']}; independent epsilon-contribution reproduction={h['independent_contribution_reproduced_count']}/{h['scheduled_cells']}; full individual epsilon vectors serialized in v34u raw={h['individual_epsilon_vector_values_serialized_in_v34u_raw']}. Evidence: `{rel(summary_path)}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(csv_path)}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`. Next safe action if plan remains unchanged and backup is verified: run T-C2 bounded converged objective-contract gate and explicitly serialize per-stage epsilon values; do not open validation64 or sealed test.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34x / T-C1 epsilon provenance audit\n\n"
        + f"UTC: {raw['created_utc']}\n\n"
        + f"Headline: {json.dumps(clean(h), sort_keys=True)}\n\n"
        + f"Artifacts: `{rel(summary_path)}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(csv_path)}`, `{rel(RUN_DIR/'completed.json')}`.\n\n"
        + "Next: verify external backup for the v34x backup request. If active PLAN_READY remains the same, execute Opus-approved T-C2 bounded converged objective-contract gate. T-C2 must persist per-stage epsilon vectors and timing, with <=6 low-level solver calls and no validation64/sealed-test access.\n",
        encoding="utf-8",
    )

    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, "development_IMPROVED_T_C1_zero_solve_provenance_audit_not_validation_not_test", "source+v34u/v34v artifact audit; no solver", "opened_development_no_validation_no_test", h["scheduled_cells"], 0, 0, 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])

    files = [Path(__file__).resolve(), RUN_DIR / "raw.json", summary_path, csv_path, BACKUP_REQUEST, STATE, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
    completed = {
        "status": "complete",
        "hard_pass": bool(h["T_C1_pass"]),
        "T_C1_pass": bool(h["T_C1_pass"]),
        "created_utc": raw["created_utc"],
        "classification": raw["classification"],
        "summary": rel(summary_path),
        "raw": rel(RUN_DIR / "raw.json"),
        "epsilon_provenance_per_arm_csv": rel(csv_path),
        "backup_request": rel(BACKUP_REQUEST),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "budget_actual": raw["budget_actual"],
        "headline": h,
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    }
    write_json(RUN_DIR / "completed.json", completed)
    return completed


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    plan_check = verify_active_plan()
    if not plan_check["pass"]:
        raise RuntimeError("active plan self-consistency check failed: " + repr(plan_check["checks"]))
    for p in [V34U_RAW, V34U_DONE, V34V_RAW, V34V_CSV, V34V_DONE, SOURCE_CONTROLLER, SOURCE_OPTIMIZER, SOURCE_GYM_CONTROLLER, SOURCE_V34K]:
        if not p.exists():
            raise RuntimeError("missing required input: " + rel(p))
    source = source_audit()
    artifact = raw_artifact_audit()
    reproduced_count = sum(1 for r in artifact["per_arm"] if r["independent_contribution_reproduced_without_abs_residual"])
    source_pass = bool(source["pass"])
    t_c1_pass = bool(source_pass and reproduced_count >= 3)
    created = now_utc()
    headline = {
        "T_C1_pass": t_c1_pass,
        "source_identity_pass": source_pass,
        "scheduled_cells": len(artifact["per_arm"]),
        "independent_contribution_reproduced_count": int(reproduced_count),
        "independent_contribution_threshold": ">=3/4",
        "individual_epsilon_vector_values_serialized_in_v34u_raw": artifact["individual_epsilon_vector_values_serialized_in_v34u_raw"],
        "max_abs_alias_pair_delta_minus_epsterm_total": max([r["max_abs_alias_pair_delta_minus_epsterm_total"] for r in artifact["per_arm"] if r["max_abs_alias_pair_delta_minus_epsterm_total"] is not None] or [None]),
        "max_abs_v34v_eps_delta_minus_epsterm_total": max([r["abs_v34v_eps_delta_minus_epsterm_total"] for r in artifact["per_arm"] if r["abs_v34v_eps_delta_minus_epsterm_total"] is not None] or [None]),
        "objective_coefficient_from_source": "penalty_term_cons=c['cost']; TTAHMPC soft obstacle constraints set cost=1000",
        "epsilon_where_enters": "stage objective only: sum over k,s of discount_factor**k * epsterm_fun(_eps[k,s]); not terminal; rterm separate",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_T_C1_zero_solve_epsilon_provenance_audit_not_validation_not_test",
        "active_lead": "claude-opus-5-5",
        "plan_check": plan_check,
        "purpose": "Verify source-level epsilon/slack objective provenance and non-circular reproduction of v34v epsilon deltas before any T-C2 solver-bearing gate.",
        "source_audit": source,
        "artifact_audit": artifact,
        "headline": headline,
        "budget_actual": {
            "new_solver_calls": 0,
            "plant_steps": 0,
            "env_reset_calls_after_construction": 0,
            "env_step_calls_after_construction": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "interpretation_limits": [
            "opened development diagnostic artifacts only",
            "no new solver state was generated",
            "v34u remains failed under the frozen no-epsilon formula",
            "the full individual epsilon vector was not serialized by v34u; only epsterm_total and eps_nonzero_count are available from the prior solver-state evaluator",
            "T-C2 must persist per-stage epsilon vector values at both converged and truncated points",
            "does not validate adaptive policy performance and does not support timing claims",
        ],
    }
    write_json(RUN_DIR / "raw.json", raw)
    completed = write_outputs(raw)
    print(json.dumps(clean({"completed": rel(RUN_DIR/"completed.json"), "summary": completed["summary"], "headline": headline, "backup_request": rel(BACKUP_REQUEST)}), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
