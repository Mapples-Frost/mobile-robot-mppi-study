#!/usr/bin/env python3
"""T-C2R2 zero-resource source242 context-loader repair.

This task implements the current Opus structured task
T-C2R2-source242-context-loader-repair from plan 20260930T152426Z_847e54.
It repairs only the loader path used by the T-C2 objective-contract gate. It
performs no solver call, plant step, env.step/reset, training/refit,
validation64 access, sealed/final-test access, or AWS/IAM/scheduler change.
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
PLAN_AUDIT_ID = "20260930T152426Z_847e54"
EXPECTED_REQUEST = "execution-result:20260930T152354_85fe2ce6"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
PASS_EVIDENCE_KEYS = (
    "source242_context_built_without_calling_base_load_contexts",
    "field_equality_evidence_reported_or_failure_raised",
    "no_solver_plant_training_validation_or_test_usage",
    "exact_lines_changed_stated_explicitly",
    "parse_gate_compile_passed_before_source_write",
    "second_run_idempotent",
)

V34Z2_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
BASE_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py"
TEMPLATE_SOURCE = ROOT / "research_artifacts" / "source_archives" / "vehicle_true_variable_horizon_v34z2_load_context_and_terminals_template_v0.txt"
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


def syntax_error_payload(exc: SyntaxError, source_text: str) -> Dict[str, Any]:
    line = None
    if exc.lineno is not None:
        all_lines = source_text.splitlines()
        if 1 <= int(exc.lineno) <= len(all_lines):
            line = all_lines[int(exc.lineno) - 1]
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "lineno": exc.lineno,
        "offset": exc.offset,
        "text": line,
    }


def compile_or_raise(source_text: str, label: str) -> Dict[str, Any]:
    try:
        compile(source_text, label, "exec")
        return {"ok": True, "label": label}
    except SyntaxError as exc:
        payload = syntax_error_payload(exc, source_text)
        raise ContractError("parse gate failed for " + label + ": " + json.dumps(payload, sort_keys=True))


def load_loader_template() -> Tuple[str, Dict[str, Any]]:
    if not TEMPLATE_SOURCE.exists():
        raise ContractError("missing external loader template: " + rel(TEMPLATE_SOURCE))
    text = TEMPLATE_SOURCE.read_text(encoding="utf-8")
    triple_double = chr(34) * 3
    triple_single = chr(39) * 3
    escaped_double = chr(92) + chr(34)
    checks = {
        "template_path": rel(TEMPLATE_SOURCE),
        "template_sha256": sha256(TEMPLATE_SOURCE),
        "template_contains_triple_double_quote": triple_double in text,
        "template_contains_triple_single_quote": triple_single in text,
        "template_contains_backslash_escaped_double_quote": escaped_double in text,
        "template_contains_base_load_contexts_call": "base.load_contexts()" in text,
        "template_contains_find_spec_marker": 'find_spec("v27_case09_slot0_early_risk"' in text,
    }
    if checks["template_contains_triple_double_quote"] or checks["template_contains_triple_single_quote"]:
        raise ContractError("loader template contains a triple-quoted string marker")
    if checks["template_contains_backslash_escaped_double_quote"]:
        raise ContractError("loader template contains a backslash-escaped double quote")
    if checks["template_contains_base_load_contexts_call"]:
        raise ContractError("loader template still contains base.load_contexts()")
    if not checks["template_contains_find_spec_marker"]:
        raise ContractError("loader template lacks the source242 find_spec marker")
    parse = compile_or_raise(text, rel(TEMPLATE_SOURCE))
    checks["function_template_compile_passed"] = bool(parse["ok"])
    return text.rstrip("\n") + "\n", checks


def replace_function_text(source_text: str, function_name: str, replacement: str) -> Tuple[str, Tuple[int, int], str]:
    lines = source_text.splitlines()
    start, end = find_function_range(lines, function_name)
    old_func = "\n".join(lines[start:end]) + "\n"
    new_lines = lines[:start] + replacement.rstrip("\n").splitlines() + lines[end:]
    return "\n".join(new_lines) + "\n", (start, end), old_func


def patch_v34z2_loader(stamp: str, phase: str) -> Dict[str, Any]:
    if not V34Z2_SOURCE.exists():
        raise ContractError("missing v34z2 source: " + rel(V34Z2_SOURCE))
    before = V34Z2_SOURCE.read_text(encoding="utf-8")
    before_hash = sha256(V34Z2_SOURCE)
    template, template_checks = load_loader_template()
    candidate, range_zero_based, old_func = replace_function_text(before, "load_context_and_terminals", template)
    old_start, old_end = range_zero_based
    parse_function = compile_or_raise(template, rel(TEMPLATE_SOURCE) + "::load_context_and_terminals")
    parse_full = compile_or_raise(candidate, rel(V34Z2_SOURCE) + "::candidate_patched_full_file")
    already_direct = "base.load_contexts()" not in old_func and "source242_context_built_without_calling_base_load_contexts" in old_func
    parse_gate = {
        "phase": phase,
        "compile_replacement_function_passed": bool(parse_function["ok"]),
        "compile_full_candidate_file_passed": bool(parse_full["ok"]),
        "parse_gate_compile_passed_before_source_write": True,
        "template_checks": template_checks,
    }
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
        V34Z2_SOURCE.write_text(candidate, encoding="utf-8")
        action = "load_context_and_terminals_replaced_with_direct_source242_builder"
    after = V34Z2_SOURCE.read_text(encoding="utf-8")
    after_hash = sha256(V34Z2_SOURCE)
    after_lines = after.splitlines()
    new_start, new_end = find_function_range(after_lines, "load_context_and_terminals")
    new_func = "\n".join(after_lines[new_start:new_end]) + "\n"
    return {
        "phase": phase,
        "action": action,
        "source": rel(V34Z2_SOURCE),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_changed": before_hash != after_hash,
        "archive_path": rel(archive_path) if archive_path is not None else None,
        "archive_sha256": sha256(archive_path) if archive_path is not None else None,
        "old_line_range_1_based_inclusive": [old_start + 1, old_end],
        "new_line_range_1_based_inclusive": [new_start + 1, new_end],
        "old_function_contained_base_load_contexts_call": "base.load_contexts()" in old_func,
        "new_function_contains_base_load_contexts_call": "base.load_contexts()" in new_func,
        "new_function_contains_direct_build_state_specs": "base.v29.build_state_specs()" in new_func,
        "new_function_contains_find_spec_marker": 'base.find_spec("v27_case09_slot0_early_risk"' in new_func,
        "exact_lines_changed_stated_explicitly": True,
        "parse_gate": parse_gate,
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

    try:
        base.load_contexts = forbidden_load_contexts
        context, terminals, load_meta = v34z2.load_context_and_terminals()
    except Exception as exc:
        context = {}
        terminals = {}
        load_meta = {}
        loader_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
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
    out = {
        "ok": not mismatches and called["base_load_contexts_called"] is False and "loader_error" not in locals(),
        "source242_context_built_without_calling_base_load_contexts": called["base_load_contexts_called"] is False and "loader_error" not in locals(),
        "base_load_contexts_called": called["base_load_contexts_called"],
        "context_id": context.get("context_id"),
        "field_equality_comparisons": comparisons,
        "field_equality_all_observed_fields": not mismatches,
        "field_equality_mismatches": mismatches,
        "comparison_basis": "Actual patched v34z2.load_context_and_terminals output compared to a reference reconstructed from base.v29.build_state_specs(), base.find_spec('v27_case09_slot0_early_risk', specs), base.state_clean, and the same literals as base.load_contexts source242 lines 317-329. base.load_contexts() was monkeypatched to raise, proving the patched function did not call it when the proof succeeds.",
        "base_load_contexts_source242_source_lines_317_329": source242_lines,
        "terminal_grid_horizons_loaded_sample": sorted(terminal_keys)[:20],
        "terminal_15_present": 15 in terminal_keys,
        "load_meta": load_meta,
        "not_determined": [
            "The complete base.load_contexts() return list is intentionally not observed because constructing its unrelated c13 entry is the live defect.",
            "This task does not construct an MPC and therefore does not verify solver-path objective-contract behavior.",
        ],
    }
    if "loader_error" in locals():
        out["loader_error"] = loader_error
    return out


def receipt_evidence(repair_verified: bool, pass_evidence: Mapping[str, bool]) -> Dict[str, Any]:
    exact_pass = {k: bool(pass_evidence.get(k) is True) for k in PASS_EVIDENCE_KEYS}
    return {"repair_verified": bool(repair_verified), "pass_evidence": exact_pass}


def write_failure(run_dir: Optional[Path], created: dt.datetime, error: str, extra: Optional[Mapping[str, Any]] = None) -> int:
    if run_dir is None:
        run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_failure_{created.strftime('%Y%m%dT%H%M%SZ')}"
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


def write_scientific_completion(
    run_dir: Path,
    created: dt.datetime,
    marker: str,
    snapshot: Mapping[str, Any],
    patch_first: Mapping[str, Any],
    patch_second: Mapping[str, Any],
    do_mpc_status: Mapping[str, Any],
    context_verification: Mapping[str, Any],
    raw_extra: Mapping[str, Any],
) -> int:
    first_archive_ok = bool(patch_first.get("action") == "already_patched_no_source_write" or patch_first.get("archive_path"))
    second_idempotent = bool(
        patch_second.get("action") == "already_patched_no_source_write"
        and patch_second.get("source_changed") is False
        and patch_second.get("source_sha256_before") == patch_second.get("source_sha256_after")
    )
    parse_gate_ok = bool(
        (patch_first.get("parse_gate") or {}).get("parse_gate_compile_passed_before_source_write") is True
        and (patch_second.get("parse_gate") or {}).get("parse_gate_compile_passed_before_source_write") is True
    )
    pass_evidence = {
        "source242_context_built_without_calling_base_load_contexts": bool(context_verification.get("source242_context_built_without_calling_base_load_contexts") is True),
        "field_equality_evidence_reported_or_failure_raised": bool(context_verification.get("field_equality_all_observed_fields") is True and context_verification.get("field_equality_comparisons")),
        "no_solver_plant_training_validation_or_test_usage": True,
        "exact_lines_changed_stated_explicitly": bool(patch_first.get("exact_lines_changed_stated_explicitly") is True),
        "parse_gate_compile_passed_before_source_write": parse_gate_ok,
        "second_run_idempotent": second_idempotent,
    }
    hard_pass = bool(first_archive_ok and all(pass_evidence.values()))
    outcome_evidence = receipt_evidence(hard_pass, pass_evidence)
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "server_api_token_audit": read_api_total_tokens(),
        "task_id": TASK_ID,
        "classification": "development_IMPROVED_T_C2R2_source242_context_loader_repair_zero_resource_not_validation_not_test",
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "plan_audit_id": (snapshot.get("ready") or {}).get("audit_id"),
        "expected_plan_audit_id_at_script_write": PLAN_AUDIT_ID,
        "defect_3_evidence": {
            "failed_run": "20260930T150458_b5f74aa4",
            "failed_json": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T150458Z/failed.json",
            "syntax_failure_predecessor": "20260930T152354_85fe2ce6 failed before script execution due a temporary syntax error in this T-C2R2 runner; no receipt was produced there.",
            "root_cause": "base.load_contexts() eagerly built an unrelated v19_c13 context and cast list-valued c13 previous_input to float before the source242 filter could run.",
            "measured_prior_actual_solver_calls": 0,
        },
        "patch_first_application": patch_first,
        "patch_second_idempotence_check": patch_second,
        "pre_replacement_archive_requirement_satisfied": first_archive_ok,
        "do_mpc_import_status": do_mpc_status,
        "context_verification": context_verification,
        "source_integrity": {
            "base_source": rel(BASE_SOURCE),
            "base_source_sha256_after": sha256(BASE_SOURCE) if BASE_SOURCE.exists() else None,
            "base_source_not_modified_by_this_script": True,
            "v34z2_source_sha256_after": sha256(V34Z2_SOURCE),
            "template_source": rel(TEMPLATE_SOURCE),
            "template_source_sha256": sha256(TEMPLATE_SOURCE) if TEMPLATE_SOURCE.exists() else None,
        },
        "explicit_non_actions": {
            "did_not_call_solver": True,
            "did_not_construct_mpc": True,
            "did_not_step_or_reset_environment_after_construction": True,
            "did_not_train_or_refit": True,
            "did_not_open_validation64": True,
            "did_not_open_sealed_or_final_test": True,
            "did_not_create_new_aws_resources_or_modify_iam_scheduler": True,
            "did_not_modify_base_load_contexts_or_c13_schema": True,
        },
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "repair_verified": hard_pass,
        "pass_evidence": pass_evidence,
        "outcome_receipt_evidence_expected_by_scheduler": outcome_evidence,
        "hard_pass": hard_pass,
        "not_determined": context_verification.get("not_determined", []),
        "raw_extra": dict(raw_extra),
    }
    raw_path = run_dir / "raw.json"
    write_json(raw_path, raw)
    summary_path = run_dir / "summary.md"
    summary_path.write_text(
        "# T-C2R2 source242 context-loader repair\n\n"
        f"UTC: `{created.isoformat()}`. Task `{TASK_ID}`.\n\n"
        f"Local hard_pass: `{hard_pass}`. Resources: solver=0, plant=0, training=0, validation=0, test=0.\n\n"
        f"First source action: `{patch_first['action']}` in `{rel(V34Z2_SOURCE)}`; old line range `{patch_first['old_line_range_1_based_inclusive']}`, new line range `{patch_first['new_line_range_1_based_inclusive']}`.\n\n"
        f"Second idempotence action: `{patch_second['action']}`; idempotent=`{second_idempotent}`.\n\n"
        f"Parse gate before source write: `{parse_gate_ok}`. Patched loader avoided `base.load_contexts()`: `{pass_evidence['source242_context_built_without_calling_base_load_contexts']}`. Field equality across observed source242 fields: `{context_verification.get('field_equality_all_observed_fields')}`.\n\n"
        "No validation64 or sealed/final-test data was opened. No solver, plant step, env.step/reset, training or refit was performed.\n\n"
        f"Primary evidence: `{rel(raw_path)}`.\n",
        encoding="utf-8",
    )
    backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C2R2_SOURCE242_CONTEXT_LOADER_REPAIR_{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{created.strftime('%Y%m%dT%H%M%SZ')}_after_t_c2r2_source242_context_loader_repair.md"
    write_json(
        backup_request,
        {
            "request": "backup_after_t_c2r2_source242_context_loader_repair",
            "created_utc": created.isoformat(),
            "must_cover": [
                rel(Path(__file__).resolve()),
                rel(TEMPLATE_SOURCE),
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
            "dependent_solver_bearing_T_C2_must_wait_for_verified_backup_and_current_lead_plan": True,
        },
    )
    doc_block = f"""
<!-- {marker} -->
## T-C2R2 source242 context-loader repair

UTC: {created.isoformat()}. Local task hard_pass `{hard_pass}` under active Opus plan `{(snapshot.get('ready') or {}).get('audit_id')}`. Resources were zero for solver, plant, training, validation and test. The v34z2 `load_context_and_terminals()` function now builds `source242_slot0_branch_start` directly from `base.v29.build_state_specs()` and `base.find_spec('v27_case09_slot0_early_risk')` and no longer calls `base.load_contexts()` in the patched path. Field-level equality for observed source242 fields was checked against the same base primitives/literals. The probe used a template file plus compile() parse gate before writing and an immediate second idempotence call. Full `base.load_contexts()` output remains intentionally unobserved because the unrelated c13 entry is the defect. Evidence: `{rel(summary_path)}` and `{rel(raw_path)}`. Backup request: `{rel(backup_request)}`. No validation64 or sealed/final-test data was opened.
""".strip()
    for doc in DOCS_TO_APPEND:
        append_if_missing(doc, marker, doc_block)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        "# Continue state after T-C2R2\n\n"
        + doc_block
        + "\n\nNext: verify external backup coverage for the T-C2R2 source/template changes, run artifacts, docs, registry and this state. Because the current structured task has continue_without_review=false, do not start T-C2C or solver-bearing T-C2 from prose alone; re-read PLAN_READY and execute the next Opus-published task if one is available. No validation64 or sealed/final test access is authorized.\n",
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
            "source_action": patch_first["action"],
            "second_run_idempotent": second_idempotent,
            "base_load_contexts_called": context_verification.get("base_load_contexts_called"),
            "field_equality_all_observed_fields": context_verification.get("field_equality_all_observed_fields"),
            "new_solver_calls": 0,
        },
        "repair_verified": hard_pass,
        "pass_evidence": pass_evidence,
        "outcome_receipt_evidence": outcome_evidence,
    }
    hash_paths = [
        Path(__file__).resolve(),
        TEMPLATE_SOURCE,
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
    for archive_rel in (patch_first.get("archive_path"), patch_second.get("archive_path")):
        if archive_rel:
            hash_paths.append(ROOT / str(archive_rel))
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
    write_json(completed_path, completed)
    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), outcome_evidence)
    print(json.dumps(clean({
        "completed": rel(completed_path),
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "headline": completed["headline"],
        "repair_verified": hard_pass,
        "pass_evidence": pass_evidence,
        "backup_request": rel(backup_request),
        "server_api_token_audit": raw["server_api_token_audit"],
    }), sort_keys=True), flush=True)
    return 0


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir: Optional[Path] = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-tc2r2-source242-context-loader-repair-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise ContractError("missing structured execution snapshot")
        patch_first = patch_v34z2_loader(stamp, "first_application")
        patch_second = patch_v34z2_loader(stamp + "_idempotence", "second_idempotence_check")
        do_mpc_status = verify_do_mpc_import()
        context_verification = verify_context_loader()
        return write_scientific_completion(
            run_dir=run_dir,
            created=created,
            marker=marker,
            snapshot=snapshot,
            patch_first=patch_first,
            patch_second=patch_second,
            do_mpc_status=do_mpc_status,
            context_verification=context_verification,
            raw_extra={"do_mpc_import_ok": do_mpc_status.get("ok")},
        )
    except Exception as exc:
        return write_failure(run_dir, created, f"unexpected T-C2R2 error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
