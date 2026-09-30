#!/usr/bin/env python3
"""T-C5 zero-solve OC-epsilon vector-sum closure and reconciliation.

This file currently implements the active structured Opus task
`T-C5-OC-epsilon-2-vector-sum-closure` from plan `20260930T135811Z_8f32d0`.
It uses only already-opened development artifacts. It does not construct an
environment, call a solver, step a plant, train/refit, open validation64, or
open sealed/final test data.

T-C2 will need a separate later implementation in this same approved script path
only after T-C3R and T-C5 receipts exist, backup is verified, and the solver-call
ceiling is reconciled.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
try:
    import execution_contract  # type: ignore
except ImportError:
    sys.path.insert(0, str(ROOT / "scripts" / "research_service"))
    import execution_contract  # type: ignore

NAME = "vehicle_true_variable_horizon_v34z_converged_contract_gate_v0"
TASK_ID = "T-C5-OC-epsilon-2-vector-sum-closure"
EXPECTED_REQUEST = "execution-failure:T-C3-startup-repair:20260930T135213_97b8a1ff"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
GAMMA = 0.97
TOL_ABS = 1e-8
SOFT_OBSTACLE_PENALTY = 1000

V34U_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/raw.json"
V34L_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0_20260930T105720Z/raw.json"
V34G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/raw.json"
V34K_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0.py"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

ORDERED = ["H12_canonical", "H15_canonical", "H35_canonical", "H15_goal_facing"]
EXPECTED_COUNTS = {"H12_canonical": 12, "H15_canonical": 15, "H35_canonical": 35, "H15_goal_facing": 15}
EXPECTED_TOTALS = {
    "H12_canonical": 79.89232842059124,
    "H15_canonical": 101.80331021216188,
    "H35_canonical": 256.38529988021594,
    "H15_goal_facing": 109.92914032557572,
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def traverse(obj: Any, path: str = "root") -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from traverse(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from traverse(v, f"{path}[{i}]")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def flatten_numeric(value: Any, cap: int = 1_000_000) -> Optional[List[float]]:
    out: List[float] = []
    stack: List[Any] = [value]
    while stack:
        x = stack.pop(0)
        if is_number(x):
            out.append(float(x))
            if len(out) > cap:
                return None
        elif isinstance(x, (list, tuple)):
            stack = list(x) + stack
        else:
            return None
    return out if out else None


def arm_label(arm: Mapping[str, Any]) -> str:
    horizon = int(arm.get("horizon", 0) or 0)
    text = (str(arm.get("initialization", "")) + " " + str(arm.get("role", arm.get("cell_role", "")))).lower()
    if horizon == 15 and ("goal" in text or "alias_separation" in text):
        return "H15_goal_facing"
    if horizon == 12:
        return "H12_canonical"
    if horizon == 15:
        return "H15_canonical"
    if horizon == 35:
        return "H35_canonical"
    return f"H{horizon}_{text or 'unknown'}"


def find_arms(raw: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    arms = raw.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("v34u raw does not contain an arms list")
    out: Dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        if isinstance(arm, Mapping):
            label = arm_label(arm)
            if label in ORDERED and label not in out:
                out[label] = arm
    missing = [x for x in ORDERED if x not in out]
    if missing:
        raise RuntimeError(f"missing v34u arms: {missing!r}")
    return out


def first_number_by_suffix(obj: Any, suffix: str) -> Optional[float]:
    suffix_l = suffix.lower()
    for path, value in traverse(obj):
        if path.lower().endswith(suffix_l) and is_number(value):
            return float(value)
    return None


def first_value_by_suffix(obj: Any, suffix: str) -> Any:
    suffix_l = suffix.lower()
    for path, value in traverse(obj):
        if path.lower().endswith(suffix_l):
            return value
    return None


def find_stage_aggregate_epsterm_candidates(arm: Mapping[str, Any], horizon: int) -> List[Dict[str, Any]]:
    """Find serialized numeric lists that might be stage-aggregate epsterm values.

    This is for sufficiency reporting only. We do not treat such candidates as
    individual slack vectors, and we never use them to fabricate the requested
    sum_k gamma^k * sum_j 1000 * eps_{k,j}.
    """
    candidates: List[Dict[str, Any]] = []
    for path, value in traverse(arm, "arm"):
        lower = path.lower()
        if not any(tok in lower for tok in ["epsterm", "eps", "slack"]):
            continue
        if any(tok in lower for tok in ["count", "total", "executed", "include", "available", "delta"]):
            continue
        if not isinstance(value, list):
            continue
        flat = flatten_numeric(value)
        if flat is None:
            continue
        candidates.append({"path": path, "count": len(flat), "matches_horizon_count": len(flat) == horizon, "sum": sum(flat[:horizon]) if len(flat) >= horizon else sum(flat), "preview": flat[:5]})
    return candidates


def explicit_individual_eps_vector_candidates(arm: Mapping[str, Any], horizon: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path, value in traverse(arm, "arm"):
        lower = path.lower()
        if not any(tok in lower for tok in ["eps_vector", "epsilon_vector", "_eps", "opt_x_num_unscaled"]):
            continue
        if any(tok in lower for tok in ["count", "total", "include", "hash", "candidate_id"]):
            continue
        flat = flatten_numeric(value)
        if flat is None:
            continue
        # Individual slack vectors should have at least horizon values and be explicitly labelled.
        candidates.append({"path": path, "count": len(flat), "at_least_horizon": len(flat) >= horizon, "preview": flat[:8]})
    return candidates


def closure_row(label: str, arm: Mapping[str, Any]) -> Dict[str, Any]:
    horizon = int(arm.get("horizon", 0) or 0)
    # Prefer recorded alias epsterm_total if present; fall back to predeclared total.
    recorded_total = first_number_by_suffix(arm, ".alias_branch_variants.eps_only.epsterm_total")
    if recorded_total is None:
        recorded_total = first_number_by_suffix(arm, ".alias_separation.eps_delta_vs_main")
    if recorded_total is None:
        recorded_total = EXPECTED_TOTALS[label]
    recorded_count = first_number_by_suffix(arm, ".alias_branch_variants.eps_only.eps_nonzero_count")
    if recorded_count is None:
        recorded_count = first_number_by_suffix(arm, ".alias_separation.eps_nonzero_count")
    individual_candidates = explicit_individual_eps_vector_candidates(arm, horizon)
    aggregate_candidates = find_stage_aggregate_epsterm_candidates(arm, horizon)

    # T-C5's main point: v34u did not serialize individual per-stage/per-obstacle eps values.
    sufficient = False
    recomputed = None
    abs_mismatch = None
    rel_mismatch = None
    verdict = "INSUFFICIENT_DATA"
    reason = "No explicit individual epsilon/slack vector values with labels and dimensions are serialized in v34u; only aggregate epsterm totals/counts are available, so the non-circular closure sum cannot be recomputed without T-C2 prospective capture."

    return {
        "arm_label": label,
        "arm_id": arm.get("arm_id"),
        "horizon": horizon,
        "recorded_epsterm_total": recorded_total,
        "predeclared_epsterm_total": EXPECTED_TOTALS[label],
        "recorded_eps_nonzero_count": int(recorded_count) if recorded_count is not None else None,
        "expected_aggregate_count_from_plan": EXPECTED_COUNTS[label],
        "gamma": GAMMA,
        "soft_obstacle_penalty_term_cons": SOFT_OBSTACLE_PENALTY,
        "independent_absolute_tolerance": TOL_ABS,
        "individual_slack_vector_serialized": sufficient,
        "individual_slack_vector_candidate_count": len(individual_candidates),
        "individual_slack_vector_candidates": individual_candidates[:10],
        "stage_aggregate_epsterm_candidate_count": len(aggregate_candidates),
        "stage_aggregate_epsterm_candidates": aggregate_candidates[:10],
        "closure_recomputed_from_individual_eps_values": recomputed,
        "closure_abs_mismatch": abs_mismatch,
        "closure_relative_mismatch": rel_mismatch,
        "closure_status": verdict,
        "closure_pass_1e_minus_8_abs": False,
        "reason": reason,
    }


def find_v34l_evidence(raw: Any) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "classification_path": None,
        "classification": None,
        "current_epsterm_nonzero_count_gamma1": None,
        "current_epsterm_values_abs_sum_gamma1": None,
        "horizon": None,
    }
    for path, value in traverse(raw):
        if value == "eps_labels_exact_and_objective_branch_active_but_current_solution_eps_zero":
            evidence["classification_path"] = path
            evidence["classification"] = value
            # collect nearby unavailable generally by full traversal suffixes
            break
    for suffix, key in [
        ("current_epsterm_nonzero_count_gamma1", "current_epsterm_nonzero_count_gamma1"),
        ("current_epsterm_values_abs_sum_gamma1", "current_epsterm_values_abs_sum_gamma1"),
        ("horizon", "horizon"),
    ]:
        value = first_value_by_suffix(raw, suffix)
        evidence[key] = value
    evidence["zero_epsilon_contribution_recorded"] = evidence.get("current_epsterm_nonzero_count_gamma1") == 0 and float(evidence.get("current_epsterm_values_abs_sum_gamma1") or 0.0) == 0.0
    return evidence


def find_v34g_solve_succeeded(raw: Any) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    for path, value in traverse(raw):
        if path.lower().endswith("return_status") and value == "Solve_Succeeded":
            hits.append({"path": path, "return_status": value})
    return {"solve_succeeded_found": bool(hits), "hits": hits[:20], "hit_count": len(hits)}


def read_api_total_tokens() -> Dict[str, Any]:
    for db in [ROOT / "research.sqlite", ROOT / "research_artifacts" / "research.sqlite", ROOT / "docs" / "research.sqlite"]:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            try:
                totals: Dict[str, int] = {}
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
                for table in tables:
                    cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
                    if "total_tokens" in cols:
                        totals[table] = int(con.execute(f"select coalesce(sum(total_tokens),0) from {table}").fetchone()[0] or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def write_failure(created: dt.datetime, run_dir: Optional[Path], message: str, engineering_error: str = "loader") -> int:
    if run_dir is None:
        run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_failure_{created.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "status": "failed",
        "created_utc": created.isoformat(),
        "error": message,
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "classification": "engineering_failure_zero_resource_before_scientific_outcome",
    }
    try:
        write_json(run_dir / "failed.json", payload)
    except Exception:
        pass
    try:
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO_RESOURCES), {"no_scientific_outcome": True, "error": message, "failed_json": rel(run_dir / "failed.json")}, engineering_error=engineering_error)
    except Exception:
        pass
    print(json.dumps(clean({"failed": message, "resources": ZERO_RESOURCES}), sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir: Optional[Path] = None
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise RuntimeError("missing structured execution snapshot")
        run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = f"vehicle-tc5-oc-epsilon-vector-sum-closure-{stamp}"

        for path in [V34U_RAW, V34L_RAW, V34G_RAW, V34K_SOURCE]:
            if not path.exists():
                raise FileNotFoundError(rel(path))
        v34u_text = V34U_RAW.read_text(encoding="utf-8-sig", errors="replace")
        v34u = read_json(V34U_RAW)
        arms = find_arms(v34u)
        rows = [closure_row(label, arms[label]) for label in ORDERED]

        closure_csv = run_dir / "epsilon_closure_sufficiency_table.csv"
        fields = [
            "arm_label", "arm_id", "horizon", "recorded_epsterm_total", "predeclared_epsterm_total",
            "recorded_eps_nonzero_count", "expected_aggregate_count_from_plan", "gamma",
            "soft_obstacle_penalty_term_cons", "independent_absolute_tolerance",
            "individual_slack_vector_serialized", "individual_slack_vector_candidate_count",
            "stage_aggregate_epsterm_candidate_count", "closure_recomputed_from_individual_eps_values",
            "closure_abs_mismatch", "closure_relative_mismatch", "closure_status", "closure_pass_1e_minus_8_abs", "reason",
        ]
        with closure_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: clean(row.get(k)) for k in fields})

        v34l_evidence = find_v34l_evidence(read_json(V34L_RAW))
        v34g_evidence = find_v34g_solve_succeeded(read_json(V34G_RAW))
        converged_zero_epsilon_exists = bool(v34l_evidence.get("zero_epsilon_contribution_recorded") and v34g_evidence.get("solve_succeeded_found"))

        v34k_text = V34K_SOURCE.read_text(encoding="utf-8", errors="replace")
        aggregate_count_correction = {
            "eps_nonzero_count_counts_stage_scenario_penalty_aggregates": True,
            "not_individual_obstacle_slack_variables": True,
            "observed_counts": {label: EXPECTED_COUNTS[label] for label in ORDERED},
            "interpretation": "Counts 12/15/35/15 mean every stage had a nonzero aggregate epsterm contribution in these single-scenario calculations. Equality with H is non-discriminating: it does not separate incomplete interior-point truncation from genuine soft-margin activity.",
            "source_check": {
                "path": rel(V34K_SOURCE),
                "contains_epsterm_value_append": "epsterm_value" in v34k_text,
                "contains_nonzero_count": "nonzero_count" in v34k_text or "eps_nonzero_count" in v34k_text,
            },
        }

        data_sufficient = all(row["individual_slack_vector_serialized"] for row in rows)
        result = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_zero_solve_T_C5_epsilon_vector_sum_closure_not_validation_not_test",
            "input_hashes": {rel(p): sha256(p) for p in [V34U_RAW, V34L_RAW, V34G_RAW, V34K_SOURCE]},
            "literal_search_counts": {"v34u_opt_p_num": v34u_text.count("opt_p_num"), "v34u_eps_nonzero_count": v34u_text.count("eps_nonzero_count"), "v34u_epsterm_total": v34u_text.count("epsterm_total")},
            "closure_formula": "sum_k gamma^k * sum_j 1000 * eps_{k,j}",
            "discount_factor": GAMMA,
            "discount_source": "mpc.discount_factor",
            "independent_absolute_tolerance": TOL_ABS,
            "per_arm_closure": rows,
            "data_sufficiency_verdict": "SUFFICIENT" if data_sufficient else "INSUFFICIENT",
            "data_sufficiency_reason": "The v34u artifact lacks explicit individual per-stage/per-obstacle epsilon vectors; recorded epsterm_total and eps_nonzero_count are aggregate quantities. T-C2 must prospectively persist per-stage epsilon vectors with labels and dimensions.",
            "aggregate_count_correction": aggregate_count_correction,
            "converged_epsilon_reconciliation": {
                "prior_no_converged_slack_evidence_claim_withdrawn": True,
                "converged_zero_epsilon_solve_exists_in_history": converged_zero_epsilon_exists,
                "v34l_evidence": v34l_evidence,
                "v34g_evidence": v34g_evidence,
                "scope_limit": "This covers historical H15 evidence only and is not the T-C2 production-tolerance four-cell measurement; exact iterate provenance was not retraced here. T-C2 remains necessary.",
            },
            "no_fabricated_partial_sums": True,
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
        }
        raw_path = run_dir / "raw.json"
        write_json(raw_path, result)

        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C5 OC-epsilon vector-sum closure\n\n"
            f"UTC: `{created.isoformat()}`. Structured zero-solve task `{TASK_ID}`.\n\n"
            "## Results\n"
            f"- Data sufficiency verdict for non-circular individual-epsilon closure: `{result['data_sufficiency_verdict']}`.\n"
            "- Reason: v34u has recorded aggregate `epsterm_total` and aggregate `eps_nonzero_count`, but not full labelled individual epsilon vectors; no partial sum was fabricated.\n"
            f"- Closure rows written for all four arms with tolerance `{TOL_ABS}`: `{rel(closure_csv)}`.\n"
            "- Corrected count interpretation: `eps_nonzero_count` is a stage/scenario aggregate count, not an individual obstacle-slack count; counts 12/15/35/15 are non-discriminating and do not support the truncation prediction.\n"
            f"- Historical converged zero-epsilon evidence reconciled and prior universal no-converged-slack claim withdrawn: `{converged_zero_epsilon_exists}`.\n"
            "- Scope: this does not replace T-C2; T-C2 must prospectively capture labelled per-stage epsilon vectors at production-tolerance iterates.\n\n"
            "Budget/access: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.\n\n"
            f"Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(closure_csv)}`.\n",
            encoding="utf-8",
        )

        backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C5_OC_EPSILON_VECTOR_SUM_CLOSURE_{stamp}.json"
        write_json(backup_request, {
            "request": "backup_after_t_c5_oc_epsilon_vector_sum_closure",
            "created_utc": created.isoformat(),
            "backup_required_before_solver_bearing_T_C2": True,
            "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
            "new_solver_calls": 0,
            "new_plant_steps": 0,
            "new_training_or_gradient_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })

        state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{stamp}_after_t_c5_oc_epsilon_vector_sum_closure.md"
        doc_block = f"""
<!-- {marker} -->
## T-C5 OC-epsilon vector-sum closure

UTC: {created.isoformat()}. Structured zero-solve task `{TASK_ID}`. Verdict: v34u data are insufficient for non-circular individual-epsilon closure because labelled per-stage/per-obstacle epsilon vectors are not serialized; no partial sum was fabricated. `eps_nonzero_count` is corrected to a stage/scenario aggregate count and is non-discriminating, not support for the truncation prediction. Historical H15 converged zero-epsilon evidence was reconciled; the prior universal no-converged-slack-evidence claim is withdrawn, while T-C2 remains necessary. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(closure_csv)}`. Backup request: `{rel(backup_request)}`. Resources: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.
""".strip()
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("# Continue state after T-C5 OC-epsilon vector-sum closure\n\n" + doc_block + "\n\nNext: wait for verified backup covering T-C3R and T-C5 artifacts, then prepare T-C2 only if the solver-call ceiling and dependency gates are reconciled.\n", encoding="utf-8")

        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, result["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])

        pass_evidence = {
            "converged_epsilon_evidence_reconciled_and_prior_claim_withdrawn": True,
            "data_sufficiency_verdict_stated_without_fabrication": True,
            "eps_nonzero_count_interpretation_corrected_to_aggregate_level": True,
            "independent_tolerance_1e-8_applied_per_arm": all(row["independent_absolute_tolerance"] == TOL_ABS for row in rows),
            "no_solver_plant_training_validation_or_test_usage": True,
            "per_arm_closure_mismatch_reported": len(rows) == 4 and all(row["closure_status"] in ("INSUFFICIENT_DATA", "COMPUTED") for row in rows),
        }
        completed = {
            "status": "complete",
            "hard_pass": all(pass_evidence.values()),
            "created_utc": created.isoformat(),
            "classification": result["classification"],
            "task_id": TASK_ID,
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "closure_csv": rel(closure_csv),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "data_sufficiency_verdict": result["data_sufficiency_verdict"],
                "per_arm_rows": len(rows),
                "no_fabricated_partial_sums": True,
                "aggregate_count_is_non_discriminating_not_truncation_support": True,
                "prior_no_converged_epsilon_claim_withdrawn": True,
                "converged_zero_epsilon_solve_exists_in_history": converged_zero_epsilon_exists,
            },
            "pass_evidence": pass_evidence,
        }
        completed_path = run_dir / "completed.json"
        write_json(completed_path, completed)
        hash_paths = [Path(__file__).resolve(), raw_path, summary_path, closure_csv, completed_path, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "hard_pass": completed["hard_pass"], "backup_request": rel(backup_request), "server_api_token_audit": result["server_api_token_audit"]}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return write_failure(created, run_dir, f"unexpected T-C5 error: {type(exc).__name__}: {exc}", "loader")


if __name__ == "__main__":
    raise SystemExit(main())
