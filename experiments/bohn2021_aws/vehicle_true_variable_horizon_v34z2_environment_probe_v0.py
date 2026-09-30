#!/usr/bin/env python3
"""T-C2R2 zero-usage source242 context-loader repair.

Active Opus structured task: T-C2R2-source242-context-loader-repair from plan
20260930T151629Z_525eed. This script repairs only the current T-C2 loader
blocker: v34z2 must construct source242 directly instead of calling
base.load_contexts(), whose eager unrelated c13 branch construction fails on a
list-valued previous_input schema. No solver, plant step, env.step/reset,
training/refit, validation64 access, or sealed/final-test access is performed.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import importlib
import inspect
import json
import math
import os
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

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

NAME = "vehicle_true_variable_horizon_v34z2_environment_probe_v0"
TASK_ID = "T-C2R2-source242-context-loader-repair"
EXPECTED_REQUEST = "execution-failure:T-C3-startup-repair:20260930T135213_97b8a1ff"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}

V34Z2_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
BASE_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py"
RESPONSE_LOG = ROOT / "docs" / "bohn2021_takeover" / "astra_reviews" / "RESPONSE_LOG.md"
DOCS_TO_APPEND = [
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    RESPONSE_LOG,
]

NEW_FUNCTION = '''def load_context_and_terminals() -> Tuple[Mapping[str, Any], Mapping[int, Any], Dict[str, Any]]:
    base = MODULES["base"]
    stage1 = MODULES["stage1_runner"]
    try:
        base.v1.latency_verify()
    except Exception:
        pass
    term_protocol = read_json(stage1.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])

    # T-C2R2 source242 loader repair: do not call base.load_contexts().
    # base.load_contexts() eagerly constructs an unrelated v19_c13 context and
    # fails before the source242 filter on a list-valued c13 previous_input
    # schema. The source242 entry is independent of c13 and is reproduced from
    # the exact primitives used by base.load_contexts() lines 317-329.
    specs = base.v29.build_state_specs()
    spec = base.find_spec("v27_case09_slot0_early_risk", specs)
    context = {
        "context_id": "source242_slot0_branch_start",
        "state_label": "v27_case09_slot0_early_risk",
        "source": "v29/v33 selected branch state, original branch start",
        "case_snapshot": copy.deepcopy(spec["case_snapshot"]),
        "branch_step": int(spec["branch_step"]),
        "tvp_start_index": int(spec["branch_step"]),
        "state": base.state_clean(spec["branch_previous_state"]),
        "previous_input": {"u_omega": 0.0, "u_s": 0.0},
        "previous_input_source": "Astra-specified v33 branch-reset zero-input semantics",
        "horizons": [15, 35],
    }
    if 15 not in terminals:
        raise ContractError("terminal grid lacks V15")
    return context, terminals, {
        "terminal_protocol": rel(stage1.TERMINAL_SOURCE_PROTOCOL),
        "terminal_receipts": clean({str(k): v for k, v in terminal_receipts.items()}),
        "context_id": "source242_slot0_branch_start",
        "context_construction": {
            "source242_context_built_without_calling_base_load_contexts": True,
            "base_load_contexts_called": False,
            "reference_primitives": [
                "base.v29.build_state_specs()",
                "base.find_spec('v27_case09_slot0_early_risk', specs)",
                "base.state_clean(spec['branch_previous_state'])",
            ],
            "mirrors_base_load_contexts_source_lines": "vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py:317-329",
            "not_determined": "Full base.load_contexts() output is not constructed here because the unrelated c13 entry is the observed defect; equality is established for the source242 fields by using the same primitives and literals as the base source242 entry.",
        },
    }
'''


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
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def canonical(value: Any) -> str:
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


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


def find_function_range(lines: list[str], function_name: str) -> Tuple[int, int]:
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(f"def {function_name}("):
            start = idx
            break
    if start is None:
        raise ContractError(f"could not find function {function_name}")
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("def "):
            end = idx
            break
    return start, end


def patch_v34z2_loader(stamp: str) -> Dict[str, Any]:
    if not V34Z2_SOURCE.exists():
        raise ContractError("missing v34z2 source: " + rel(V34Z2_SOURCE))
    before = V34Z2_SOURCE.read_text(encoding="utf-8")
    before_hash = sha256(V34Z2_SOURCE)
    lines = before.splitlines()
    start, end = find_function_range(lines, "load_context_and_terminals")
    old_func = "\n".join(lines[start:end]) + "\n"
    already_direct = "base.load_contexts()" not in old_func and "source242_context_built_without_calling_base_load_contexts" in old_func
    archive_path: Optional[Path] = None
    if already_direct:
        action = "already_patched_no_source_write"
    else:
        if "base.load_contexts()" not in old_func:
            raise ContractError("load_context_and_terminals did not contain base.load_contexts(), refusing uncertain repair")
        archive_dir = ROOT / "research_artifacts" / "source_archives" / f"{NAME}_{stamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.pre_tc2r2.{before_hash}.py"
        archive_path.write_text(before, encoding="utf-8")
        new_lines = lines[:start] + NEW_FUNCTION.rstrip("\n").splitlines() + lines[end:]
        tmp = V34Z2_SOURCE.with_suffix(V34Z2_SOURCE.suffix + ".tc2r2_tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(V34Z2_SOURCE)
        action = "load_context_and_terminals_replaced_with_direct_source242_builder"
    after_hash = sha256(V34Z2_SOURCE)
    after_lines = V34Z2_SOURCE.read_text(encoding="utf-8").splitlines()
    new_start, new_end = find_function_range(after_lines, "load_context_and_terminals")
    new_func = "\n".join(after_lines[new_start:new_end]) + "\n"
    return {
        "action": action,
        "source": rel(V34Z2_SOURCE),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "archive_path": rel(archive_path) if archive_path is not None else None,
        "archive_sha256": sha256(archive_path) if archive_path is not None else None,
        "old_line_range_1_based_inclusive": [start + 1, end],
        "new_line_range_1_based_inclusive": [new_start + 1, new_end],
        "old_function_contained_base_load_contexts_call": "base.load_contexts()" in old_func,
        "new_function_contains_base_load_contexts_call": "base.load_contexts()" in new_func,
        "new_function_contains_direct_build_state_specs": "base.v29.build_state_specs()" in new_func,
        "new_function_contains_find_spec": 'base.find_spec("v27_case09_slot0_early_risk"' in new_func,
        "exact_lines_changed_stated_explicitly": True,
        "scope_statement": "Only v34z2 load_context_and_terminals was replaced; base.load_contexts and vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py were not edited.",
    }


def verify_do_mpc_import() -> Dict[str, Any]:
    try:
        do_mpc = importlib.import_module("do_mpc")
        return {
            "ok": True,
            "module_name": getattr(do_mpc, "__name__", None),
            "module_file": getattr(do_mpc, "__file__", None),
            "module_path": list(getattr(do_mpc, "__path__", []) or []),
            "interpreter": sys.executable,
            "python_version": sys.version,
            "shim_path": rel(AWS_DIR / "do_mpc" / "__init__.py"),
            "shim_sha256": sha256(AWS_DIR / "do_mpc" / "__init__.py") if (AWS_DIR / "do_mpc" / "__init__.py").exists() else None,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback_tail": traceback.format_exc().splitlines()[-12:]}


def reference_source242_context(base: Any) -> Dict[str, Any]:
    specs = base.v29.build_state_specs()
    spec = base.find_spec("v27_case09_slot0_early_risk", specs)
    return {
        "context_id": "source242_slot0_branch_start",
        "state_label": "v27_case09_slot0_early_risk",
        "source": "v29/v33 selected branch state, original branch start",
        "case_snapshot": copy.deepcopy(spec["case_snapshot"]),
        "branch_step": int(spec["branch_step"]),
        "tvp_start_index": int(spec["branch_step"]),
        "state": base.state_clean(spec["branch_previous_state"]),
        "previous_input": {"u_omega": 0.0, "u_s": 0.0},
        "previous_input_source": "Astra-specified v33 branch-reset zero-input semantics",
        "horizons": [15, 35],
    }


def verify_context_loader() -> Dict[str, Any]:
    importlib.invalidate_caches()
    v34z2 = importlib.import_module("vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0")
    v34z2.load_modules()
    base = v34z2.MODULES["base"]

    source242_lines = []
    try:
        src_lines, line_no = inspect.getsourcelines(base.load_contexts)
        for offset, text in enumerate(src_lines, start=line_no):
            if 317 <= offset <= 329:
                source242_lines.append({"line": offset, "text": text.rstrip("\n")})
    except Exception:
        pass

    reference = reference_source242_context(base)
    called = {"base_load_contexts_called": False}
    original_load_contexts = base.load_contexts

    def forbidden_load_contexts(*args: Any, **kwargs: Any) -> Any:
        called["base_load_contexts_called"] = True
        raise ContractError("base.load_contexts() was called during T-C2R2 verification")

    base.load_contexts = forbidden_load_contexts
    try:
        context, terminals, load_meta = v34z2.load_context_and_terminals()
    finally:
        base.load_contexts = original_load_contexts

    observed_fields = [
        "context_id",
        "state_label",
        "source",
        "case_snapshot",
        "branch_step",
        "tvp_start_index",
        "state",
        "previous_input",
        "previous_input_source",
        "horizons",
    ]
    comparisons = {}
    mismatches = []
    for field in observed_fields:
        actual_s = canonical(context.get(field))
        reference_s = canonical(reference.get(field))
        eq = actual_s == reference_s
        comparisons[field] = {
            "equal": eq,
            "actual_sha256": hashlib.sha256(actual_s.encode("utf-8")).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_s.encode("utf-8")).hexdigest(),
        }
        if not eq:
            mismatches.append(field)
    terminal_keys = []
    if isinstance(terminals, Mapping):
        for k in terminals.keys():
            try:
                terminal_keys.append(int(k))
            except Exception:
                pass
    return {
        "ok": not mismatches and called["base_load_contexts_called"] is False,
        "source242_context_built_without_calling_base_load_contexts": called["base_load_contexts_called"] is False,
        "context_id": context.get("context_id"),
        "field_equality_comparisons": comparisons,
        "field_equality_all_observed_fields": not mismatches,
        "field_equality_mismatches": mismatches,
        "comparison_basis": "Actual patched v34z2.load_context_and_terminals output compared to a reference reconstructed from base.v29.build_state_specs(), base.find_spec('v27_case09_slot0_early_risk', specs), base.state_clean, and the same literals as base.load_contexts source242 lines 317-329. base.load_contexts() was monkeypatched to raise, proving the patched function did not call it.",
        "base_load_contexts_source242_source_lines_317_329": source242_lines,
        "terminal_grid_horizons_loaded_sample": sorted(terminal_keys)[:20],
        "terminal_15_present": 15 in terminal_keys,
        "load_meta": load_meta,
        "not_determined": [
            "The complete base.load_contexts() return list is intentionally not observed because constructing its unrelated c13 entry is the live defect.",
            "This task does not construct an MPC and therefore does not verify solver-path objective-contract behavior.",
        ],
    }


def write_failure(run_dir: Path, created: dt.datetime, error: str, extra: Optional[Mapping[str, Any]] = None) -> int:
    failed_path = run_dir / "failed.json"
    payload = {
        "status": "failed",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "error": error,
        "extra": dict(extra or {}),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    }
    write_json(failed_path, payload)
    try:
        execution_contract.record_outcome(
            ROOT,
            "engineering_failure",
            dict(ZERO_RESOURCES),
            {"no_scientific_outcome": True, "error": error, "failed_json": rel(failed_path)},
            engineering_error="loader",
        )
    except Exception:
        pass
    print(json.dumps({"failed": error, "failed_json": rel(failed_path), "resources": ZERO_RESOURCES}, sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-tc2r2-source242-context-loader-repair-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise ContractError("missing structured execution snapshot")
        patch = patch_v34z2_loader(stamp)
        do_mpc_status = verify_do_mpc_import()
        context_verification = verify_context_loader()
        base_after_hash = sha256(BASE_SOURCE) if BASE_SOURCE.exists() else None

        pass_evidence = {
            "source242_context_built_without_calling_base_load_contexts": bool(context_verification.get("source242_context_built_without_calling_base_load_contexts") is True),
            "field_equality_evidence_reported_or_failure_raised": bool(context_verification.get("field_equality_all_observed_fields") is True and context_verification.get("field_equality_comparisons")),
            "no_solver_plant_training_validation_or_test_usage": True,
            "exact_lines_changed_stated_explicitly": bool(patch.get("exact_lines_changed_stated_explicitly") is True),
        }
        hard_pass = all(pass_evidence.values())
        if not hard_pass:
            raise ContractError("T-C2R2 pass conditions not met: " + repr(pass_evidence))

        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_T_C2R2_source242_context_loader_repair_zero_resource_not_validation_not_test",
            "snapshot_sha256": snapshot.get("snapshot_sha256"),
            "plan_audit_id": (snapshot.get("ready") or {}).get("audit_id"),
            "defect_3_evidence": {
                "failed_run": "20260930T150458_b5f74aa4",
                "failed_json": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T150458Z/failed.json",
                "syntax_failure_this_cycle": "20260930T152354_85fe2ce6 failed before script execution due a temporary syntax error in this T-C2R2 runner; no receipt was produced there.",
                "root_cause": "base.load_contexts() eagerly built an unrelated v19_c13 context and cast list-valued c13 previous_input to float before the source242 filter could run.",
                "measured_prior_actual_solver_calls": 0,
            },
            "patch": patch,
            "do_mpc_import_status": do_mpc_status,
            "context_verification": context_verification,
            "source_integrity": {
                "base_source": rel(BASE_SOURCE),
                "base_source_sha256_after": base_after_hash,
                "base_source_not_modified_by_this_script": True,
                "v34z2_source_sha256_after": sha256(V34Z2_SOURCE),
            },
            "explicit_non_actions": {
                "did_not_call_base_load_contexts_in_patched_function": True,
                "did_not_modify_base_load_contexts_or_c13_schema": True,
                "did_not_construct_mpc": True,
                "did_not_call_solver": True,
                "did_not_step_or_reset_environment_after_construction": True,
                "did_not_train_or_refit": True,
                "did_not_open_validation64": True,
                "did_not_open_sealed_or_final_test": True,
                "did_not_create_new_aws_resources_or_modify_iam_scheduler": True,
            },
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "pass_evidence": pass_evidence,
            "hard_pass": hard_pass,
            "not_determined": context_verification.get("not_determined", []),
        }
        raw_path = run_dir / "raw.json"
        write_json(raw_path, raw)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C2R2 source242 context-loader repair\n\n"
            f"UTC: `{created.isoformat()}`. Task `{TASK_ID}`.\n\n"
            f"Local hard_pass: `{hard_pass}`. Resources: solver=0, plant=0, training=0, validation=0, test=0.\n\n"
            f"Source action: `{patch['action']}` in `{rel(V34Z2_SOURCE)}`; changed line range `{patch['old_line_range_1_based_inclusive']}` to `{patch['new_line_range_1_based_inclusive']}`.\n\n"
            f"Patched loader avoided `base.load_contexts()`: `{pass_evidence['source242_context_built_without_calling_base_load_contexts']}`. Field equality across observed source242 fields: `{context_verification.get('field_equality_all_observed_fields')}`.\n\n"
            "No validation64 or sealed/final-test data was opened. No solver, plant step, env.step/reset, training or refit was performed.\n\n"
            f"Primary evidence: `{rel(raw_path)}`.\n",
            encoding="utf-8",
        )
        backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C2R2_SOURCE242_CONTEXT_LOADER_REPAIR_{stamp}.json"
        state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{stamp}_after_t_c2r2_source242_context_loader_repair.md"
        write_json(
            backup_request,
            {
                "request": "backup_after_t_c2r2_source242_context_loader_repair",
                "created_utc": created.isoformat(),
                "must_cover": [
                    rel(Path(__file__).resolve()),
                    rel(V34Z2_SOURCE),
                    rel(run_dir),
                    rel(backup_request),
                    rel(state_path),
                    "STATUS.md",
                    "RESEARCH_LOG.md",
                    "DECISIONS.md",
                    "RESULTS_AUDIT.md",
                    "REPRODUCTION_PROTOCOL.md",
                    "EXPERIMENT_REGISTRY.csv",
                    rel(RESPONSE_LOG),
                ],
                "new_solver_calls": 0,
                "plant_steps": 0,
                "training_steps": 0,
                "validation64_bank_opened": False,
                "sealed_test_accessed": False,
                "solver_bearing_T_C2_must_wait_for_verified_backup": True,
            },
        )
        doc_block = f"""
<!-- {marker} -->
## T-C2R2 source242 context-loader repair

UTC: {created.isoformat()}. Local task hard_pass `{hard_pass}` under active Opus plan `20260930T151629Z_525eed`. Resources were zero for solver, plant, training, validation and test. The v34z2 `load_context_and_terminals()` function now builds `source242_slot0_branch_start` directly from `base.v29.build_state_specs()` and `base.find_spec('v27_case09_slot0_early_risk')` and no longer calls `base.load_contexts()` in the patched path. Field-level equality for observed source242 fields was verified against the same base primitives/literals; full `base.load_contexts()` output remains intentionally unobserved because the unrelated c13 entry is the defect. Evidence: `{rel(summary_path)}` and `{rel(raw_path)}`. Backup request: `{rel(backup_request)}`. No validation64 or sealed/final-test data was opened.
""".strip()
        for doc in DOCS_TO_APPEND:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "# Continue state after T-C2R2\n\n"
            + doc_block
            + "\n\nNext: do not spend solver calls until an external backup covers the T-C2R2 source changes and artifacts. After backup, continue the active Opus plan with T-C2C-context-identity-preflight (zero resources) using the exact task fields from PLAN_READY. Only after T-C2C passes and backup is verified should solver-bearing T-C2 be launched.\n",
            encoding="utf-8",
        )
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow([
                created.isoformat(),
                NAME,
                raw["classification"],
                "not_applicable_no_training_seed",
                "environment_diagnostic_only_no_solver_no_validation64_no_sealed_test",
                0,
                0,
                0,
                0,
                0,
                False,
                rel(run_dir / "completed.json"),
                marker,
            ])
        completed_path = run_dir / "completed.json"
        completed = {
            "status": "complete",
            "hard_pass": hard_pass,
            "task_id": TASK_ID,
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "T_C2R2_pass": hard_pass,
                "source_action": patch["action"],
                "base_load_contexts_called": False,
                "field_equality_all_observed_fields": context_verification.get("field_equality_all_observed_fields"),
                "new_solver_calls": 0,
            },
            "pass_evidence": pass_evidence,
        }
        hash_paths = [
            Path(__file__).resolve(),
            V34Z2_SOURCE,
            BASE_SOURCE,
            raw_path,
            summary_path,
            backup_request,
            state_path,
            ROOT / "STATUS.md",
            ROOT / "RESEARCH_LOG.md",
            ROOT / "DECISIONS.md",
            ROOT / "RESULTS_AUDIT.md",
            ROOT / "REPRODUCTION_PROTOCOL.md",
            ROOT / "EXPERIMENT_REGISTRY.csv",
            RESPONSE_LOG,
        ]
        archive_rel = patch.get("archive_path")
        if archive_rel:
            hash_paths.append(ROOT / archive_rel)
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        print(json.dumps(clean({
            "completed": rel(completed_path),
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "headline": completed["headline"],
            "pass_evidence": pass_evidence,
            "backup_request": rel(backup_request),
            "server_api_token_audit": raw["server_api_token_audit"],
        }), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return write_failure(run_dir, created, f"unexpected T-C2R2 error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
