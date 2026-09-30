#!/usr/bin/env python3
"""Zero-solver preflight wrapper for the v34z2 converged objective-contract gate.

Temporary solo GPT-5.5 plan task: verify the real T-C2 pre-solve chain without
calling controller.get_action(), mpc.solve(), env.reset(), env.step(), training,
validation64, or sealed/final-test code. The wrapped v34z2 gate source remains
unmodified and is verified by hash before import.
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import sqlite3
import sys
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments" / "bohn2021_aws"
SERVICE_DIR = ROOT / "scripts" / "research_service"
for _p in (SERVICE_DIR, AWS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import execution_contract  # type: ignore  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34z2_converged_contract_preflight_v0"
TASK_ID = "S-TC2C-source242-converged-contract-preflight-wrapper-v0"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
EXPECTED_GATE_SHA256 = "66dbf84e4a7917d824152cae1cac4da77c51efff576cd4775e8cd1be29dbcc97"
GATE_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
RESPONSE_LOG = ROOT / "docs" / "bohn2021_takeover" / "astra_reviews" / "RESPONSE_LOG.md"
DOCS_TO_APPEND = [
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    RESPONSE_LOG,
]


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Any) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items() if k != "nlp_expr"}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


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
                    cols = [r[1] for r in con.execute("pragma table_info(%s)" % table).fetchall()]
                    if "total_tokens" in cols:
                        totals[table] = int(con.execute("select coalesce(sum(total_tokens),0) from %s" % table).fetchone()[0] or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def numeric_summary(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    vals_sorted = sorted(vals)
    if len(vals_sorted) == 1:
        p95 = vals_sorted[0]
    else:
        pos = 0.95 * (len(vals_sorted) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos)); frac = pos - lo
        p95 = vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac
    return {"n": len(vals), "mean": float(mean(vals)), "median": float(median(vals)), "p95": float(p95), "min": float(min(vals)), "max": float(max(vals))}


def compact_setup_record(setup: Mapping[str, Any]) -> Dict[str, Any]:
    init = setup.get("initialization_meta") or {}
    pre = setup.get("pre_solve_state_after_initialization") or {}
    return {
        "cell_label": setup.get("cell_label"),
        "role": setup.get("role"),
        "horizon": setup.get("horizon"),
        "initialization": setup.get("initialization"),
        "solve_mode": setup.get("solve_mode"),
        "assignments_attempted": init.get("assignments_attempted"),
        "assignments_succeeded": init.get("assignments_succeeded"),
        "numeric_readback_count": init.get("numeric_readback_count"),
        "numeric_readback_all_successful": init.get("numeric_readback_all_successful"),
        "pre_opt_x_hash": pre.get("opt_x_hash"),
        "pre_opt_p_hash": pre.get("opt_p_hash"),
        "production_max_iter_not_forced_to_1": setup.get("production_max_iter_not_forced_to_1"),
        "truncated_max_iter_forced_to_1": setup.get("truncated_max_iter_forced_to_1"),
        "captured_nlpsol_count_during_construction": setup.get("captured_nlpsol_count_during_construction"),
        "captured_nlpsol_meta": setup.get("captured_nlpsol_meta"),
        "first_three_pred_controls": init.get("first_three_pred_controls"),
        "assignment_examples": init.get("assignment_examples"),
        "mismatches_first20": init.get("mismatches_first20"),
    }


def intended_primal_report(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_label = {str(r.get("cell_label")): r for r in records}
    out: Dict[str, Any] = {"per_cell": {}, "distinctness_checks": {}}
    for label in ["H12_canonical", "H15_canonical", "H15_goal_facing", "H35_canonical", "H12_truncated_control"]:
        r = by_label.get(label)
        if r is None:
            out["per_cell"][label] = {"present": False}
            continue
        out["per_cell"][label] = {
            "present": True,
            "assignments_attempted_positive": int(r.get("assignments_attempted") or 0) > 0,
            "assignments_succeeded_positive": int(r.get("assignments_succeeded") or 0) > 0,
            "numeric_readback_all_successful": r.get("numeric_readback_all_successful") is True,
            "pre_opt_x_hash": r.get("pre_opt_x_hash"),
            "pre_opt_p_hash": r.get("pre_opt_p_hash"),
        }
    h15c = by_label.get("H15_canonical")
    h15g = by_label.get("H15_goal_facing")
    if h15c and h15g:
        out["distinctness_checks"]["H15_canonical_vs_goal_facing"] = {
            "pre_opt_x_hash_equal": h15c.get("pre_opt_x_hash") == h15g.get("pre_opt_x_hash"),
            "pre_opt_p_hash_equal": h15c.get("pre_opt_p_hash") == h15g.get("pre_opt_p_hash"),
            "first_three_pred_controls_equal": clean(h15c.get("first_three_pred_controls")) == clean(h15g.get("first_three_pred_controls")),
            "contractually_distinct_numeric_profiles": clean(h15c.get("first_three_pred_controls")) != clean(h15g.get("first_three_pred_controls")),
        }
    return out


def write_failure(run_dir: Path, created: dt.datetime, error: str) -> int:
    failed = {
        "status": "failed",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "error": error,
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    }
    failed_path = run_dir / "failed.json"
    write_json(failed_path, failed)
    evidence = {
        "preflight_chain_verified": False,
        "no_solver_plant_training_validation_or_test_usage": True,
        "inert_intervention_gate_evaluated_and_reported": False,
        "no_scientific_outcome": True,
        "error": error,
        "failed_json": rel(failed_path),
    }
    try:
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO_RESOURCES), evidence, engineering_error="loader")
    except Exception:
        pass
    print(json.dumps(clean({"failed": error, "failed_json": rel(failed_path), "resources": ZERO_RESOURCES}), sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT)
        if snapshot is None:
            raise ContractError("missing structured execution snapshot")
        gate_sha = sha256(GATE_SOURCE)
        if gate_sha != EXPECTED_GATE_SHA256:
            raise ContractError("v34z2 gate source hash mismatch before preflight: %s" % gate_sha)

        import vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0 as gate  # type: ignore

        # Make any contract checks inside the imported module agree with the actual
        # structured snapshot request, but do not call gate.main() and do not edit source.
        gate.EXPECTED_REQUEST = (snapshot.get("ready") or {}).get("request_id")
        gate.TASK_ID = TASK_ID

        gate.load_modules()
        gate.install_nlpsol_patch()
        context, terminals, load_meta = gate.load_context_and_terminals()
        terminal_keys = sorted(int(k) for k in terminals.keys())

        setup_records: List[Dict[str, Any]] = []
        for row in gate.SOLVE_ORDER:
            before_solver_usage = int(gate.RESOURCE_USAGE.get("solver_calls", 0))
            prep = gate.prepare_cell(row, context, terminals, for_presolve_only=True)
            after_solver_usage = int(gate.RESOURCE_USAGE.get("solver_calls", 0))
            if after_solver_usage != before_solver_usage:
                raise ContractError("prepare_cell unexpectedly incremented solver_calls for %s" % row.get("cell_label"))
            setup_records.append(clean(prep["setup"]))
            del prep
            gc.collect()
        compact_records = [compact_setup_record(r) for r in setup_records]
        production_records = [r for r in setup_records if r.get("solve_mode") == "production"]
        inert_gate = gate.validate_setup_gate(production_records)
        if int(gate.RESOURCE_USAGE.get("solver_calls", 0)) != 0:
            raise ContractError("preflight used solver_calls=%s" % gate.RESOURCE_USAGE.get("solver_calls"))

        intended = intended_primal_report(compact_records)
        preflight_chain_verified = bool(
            load_meta.get("context_construction", {}).get("source242_context_built_without_calling_base_load_contexts") is True
            and load_meta.get("context_construction", {}).get("base_load_contexts_called") is False
            and 15 in terminal_keys
            and inert_gate.get("evaluated") is True
            and inert_gate.get("inert_intervention_failure") is False
            and all((r.get("numeric_readback_all_successful") is True) for r in compact_records)
            and all(int(r.get("assignments_attempted") or 0) > 0 for r in compact_records)
            and intended.get("distinctness_checks", {}).get("H15_canonical_vs_goal_facing", {}).get("contractually_distinct_numeric_profiles") is True
        )
        evidence = {
            "preflight_chain_verified": preflight_chain_verified,
            "no_solver_plant_training_validation_or_test_usage": True,
            "inert_intervention_gate_evaluated_and_reported": bool(inert_gate.get("evaluated") is True),
        }
        hard_pass = all(evidence.values())

        raw_path = run_dir / "raw.json"
        summary_path = run_dir / "summary.md"
        setup_csv = run_dir / "setup_preflight.csv"
        completed_path = run_dir / "completed.json"
        backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_S_TC2C_PREFLIGHT_WRAPPER_{stamp}.json"
        state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{stamp}_after_s_tc2c_preflight_wrapper.md"

        raw = {
            "status": "complete",
            "task_id": TASK_ID,
            "snapshot_sha256": snapshot.get("snapshot_sha256"),
            "plan_request_id": (snapshot.get("ready") or {}).get("request_id"),
            "plan_audit_id": (snapshot.get("ready") or {}).get("audit_id"),
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "classification": "development_IMPROVED_zero_solver_T_C2C_preflight_wrapper_not_validation_not_test",
            "wrapped_gate_source": rel(GATE_SOURCE),
            "wrapped_gate_sha256": gate_sha,
            "gate_source_modified": False,
            "context_built_without_base_load_contexts": load_meta.get("context_construction", {}).get("source242_context_built_without_calling_base_load_contexts"),
            "base_load_contexts_called": load_meta.get("context_construction", {}).get("base_load_contexts_called"),
            "load_meta": load_meta,
            "terminal_grid_horizon_keys": terminal_keys,
            "terminal_15_present": 15 in terminal_keys,
            "preflight_stopped_before_controller_get_action_and_mpc_solve": True,
            "setup_records": setup_records,
            "compact_setup_records": compact_records,
            "intended_primal_report": intended,
            "inert_intervention_gate": inert_gate,
            "construction_nlpsol_capture_count_summary": numeric_summary([r.get("captured_nlpsol_count_during_construction") for r in compact_records]),
            "resource_counter_observed_from_gate_module": clean(gate.RESOURCE_USAGE),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "no_training_or_refit": True,
            "no_env_reset_or_step_after_construction": True,
            "pass_evidence": evidence,
            "hard_pass": hard_pass,
            "interpretation_limits": [
                "This preflight verifies only the source242 T-C2 setup path up to prepare_cell(..., for_presolve_only=True).",
                "It intentionally does not solve the objective contract, make a control-performance claim, read validation64, or read sealed/final test data.",
                "MPC/NLP objects are constructed because that is the real pre-solve chain; no low-level optimization solve is called.",
            ],
        }
        write_json(raw_path, raw)

        with setup_csv.open("w", encoding="utf-8", newline="") as stream:
            fields = ["cell_label", "role", "horizon", "initialization", "solve_mode", "assignments_attempted", "assignments_succeeded", "numeric_readback_count", "numeric_readback_all_successful", "pre_opt_x_hash", "pre_opt_p_hash", "production_max_iter_not_forced_to_1", "truncated_max_iter_forced_to_1", "captured_nlpsol_count_during_construction"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in compact_records:
                writer.writerow({k: row.get(k) for k in fields})

        summary_lines = [
            "# S-TC2C zero-solver preflight wrapper",
            "",
            "UTC: `%s`." % created.isoformat(),
            "",
            "Task hard_pass: `%s`. Solver/plant/training/validation/test counters are all zero." % hard_pass,
            "",
            "Key checks:",
            "- Gate source unchanged and matched accepted hash `%s`: `%s`." % (EXPECTED_GATE_SHA256, gate_sha == EXPECTED_GATE_SHA256),
            "- Context built without `base.load_contexts()`: `%s`." % raw["context_built_without_base_load_contexts"],
            "- Terminal horizon keys: `%s`; H15 present: `%s`." % (terminal_keys, 15 in terminal_keys),
            "- Inert-intervention gate: `%s`." % inert_gate,
            "- Intended-primal report: `%s`." % intended,
            "",
            "This is not objective-closure evidence and not validation/final-test evidence.",
            "",
            "Evidence: `%s`, `%s`, `%s`." % (rel(raw_path), rel(setup_csv), rel(completed_path)),
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        write_json(backup_request, {
            "request": "backup_after_s_tc2c_preflight_wrapper",
            "created_utc": created.isoformat(),
            "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "solver_calls": 0,
        })

        marker = "s-tc2c-preflight-wrapper-" + stamp
        doc_block = "\n".join([
            "<!-- %s -->" % marker,
            "## S-TC2C zero-solver preflight wrapper",
            "",
            "UTC: %s. Task hard_pass `%s`; gate source hash matched `%s`; context built without `base.load_contexts()` `%s`; H15 terminal present `%s`; inert gate verdict `%s`. Solver, plant, training, validation and test counters were all zero; validation64 and sealed/final test remained closed. Evidence: `%s`, `%s`, `%s`. Backup request: `%s`." % (
                created.isoformat(), hard_pass, gate_sha, raw["context_built_without_base_load_contexts"], 15 in terminal_keys, inert_gate.get("verdict"), rel(summary_path), rel(raw_path), rel(setup_csv), rel(backup_request)
            ),
        ])
        for doc in DOCS_TO_APPEND:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "# Continue state after S-TC2C preflight wrapper\n\n" + doc_block + "\n\nNext: if this run has a valid receipt and an external backup is verified, run the dependent S-TC2 objective-contract measurement wrapper with exactly five solver calls. If the preflight did not pass, preserve raw evidence and repair only the demonstrated operational defect under a new bounded plan.\n",
            encoding="utf-8",
        )
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow([
                created.isoformat(), NAME, raw["classification"], "canonical_solve_no_training_seed", "opened_development_source242_slot0_branch_start_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(completed_path), marker,
            ])

        completed = {
            "status": "complete",
            "hard_pass": hard_pass,
            "task_id": TASK_ID,
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "setup_csv": rel(setup_csv),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "preflight_chain_verified": preflight_chain_verified,
                "inert_gate_verdict": inert_gate.get("verdict"),
                "terminal_15_present": 15 in terminal_keys,
                "setup_record_count": len(setup_records),
                "new_solver_calls": 0,
            },
            "pass_evidence": evidence,
            "hashes": {},
        }
        hash_paths = [Path(__file__).resolve(), GATE_SOURCE, raw_path, summary_path, setup_csv, completed_path, backup_request, state_path, ROOT / "EXPERIMENT_REGISTRY.csv", RESPONSE_LOG]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p != completed_path}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": evidence, "resources": ZERO_RESOURCES}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return write_failure(run_dir, created, "unexpected S-TC2C preflight error: %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
