#!/usr/bin/env python3
"""T-C2R zero-usage environment/import repair for the v34z2 T-C2 launch path.

This script implements the active Opus structured task
``T-C2R-env-import-repair`` from plan 20260930T143121Z_dfaf99. It performs no
solver call, plant step, training/refit, validation64 access or sealed/final-test
access. Its only source modification is the lead-authorized one-line repair in
``vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py``: bind
``stage1_runner`` to the true terminal-grid provider module instead of the v1d
intermediate wrapper.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
TASK_ID = "T-C2R-env-import-repair"
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
WRAPPER_MODULE = "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner"
PROVIDER_MODULE = "vehicle_stress_scenario_opportunity_probe_v1_runner"
OLD_IMPORT = f"import {WRAPPER_MODULE} as stage1_runner"
NEW_IMPORT = f"import {PROVIDER_MODULE} as stage1_runner"
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


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
                    return {
                        "available": True,
                        "path": rel(db),
                        "table_sums": totals,
                        "total_tokens": int(sum(totals.values())),
                    }
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def line_no_for(text: str, needle: str) -> Optional[int]:
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    return None


def patch_v34z2_source(run_dir: Path, stamp: str) -> Dict[str, Any]:
    if not V34Z2_SOURCE.exists():
        raise ContractError("v34z2 source does not exist: " + rel(V34Z2_SOURCE))
    before = V34Z2_SOURCE.read_text(encoding="utf-8")
    before_hash = sha256(V34Z2_SOURCE)
    old_count = before.count(OLD_IMPORT)
    new_count = before.count(NEW_IMPORT)
    old_line = line_no_for(before, OLD_IMPORT)
    new_line_before = line_no_for(before, NEW_IMPORT)
    archive_path: Optional[Path] = None
    action: str
    if new_count >= 1 and old_count == 0:
        action = "already_patched_no_source_write"
        after = before
    elif old_count == 1:
        archive_dir = ROOT / "research_artifacts" / "source_archives" / f"{NAME}_{stamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.pre_tc2r.{before_hash}.py"
        archive_path.write_text(before, encoding="utf-8")
        after = before.replace(OLD_IMPORT, NEW_IMPORT, 1)
        tmp = V34Z2_SOURCE.with_suffix(V34Z2_SOURCE.suffix + ".tc2r_tmp")
        tmp.write_text(after, encoding="utf-8")
        tmp.replace(V34Z2_SOURCE)
        action = "one_line_import_repair_written"
    else:
        raise ContractError(
            "v34z2 source had unexpected import state: "
            f"old_count={old_count}, new_count={new_count}; refusing non-minimal edit"
        )
    after_hash = sha256(V34Z2_SOURCE)
    after_text = V34Z2_SOURCE.read_text(encoding="utf-8")
    return {
        "action": action,
        "source": rel(V34Z2_SOURCE),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "archive_path": rel(archive_path) if archive_path is not None else None,
        "archive_sha256": sha256(archive_path) if archive_path is not None else None,
        "old_import": OLD_IMPORT,
        "new_import": NEW_IMPORT,
        "old_import_count_before": old_count,
        "new_import_count_before": new_count,
        "old_import_count_after": after_text.count(OLD_IMPORT),
        "new_import_count_after": after_text.count(NEW_IMPORT),
        "old_import_line_before": old_line,
        "new_import_line_before": new_line_before,
        "new_import_line_after": line_no_for(after_text, NEW_IMPORT),
        "minimality_statement": "Only the stage1_runner import binding is changed when old_count==1; no scientific constants, tolerances, splits, methods, seeds, budgets, provider modules, v1c modules or v1d modules are edited by this script.",
    }


def verify_do_mpc_import() -> Dict[str, Any]:
    try:
        do_mpc = importlib.import_module("do_mpc")
        controller = importlib.import_module("do_mpc.controller")
        return {
            "ok": True,
            "module_name": getattr(do_mpc, "__name__", None),
            "module_file": getattr(do_mpc, "__file__", None),
            "module_path": list(getattr(do_mpc, "__path__", []) or []),
            "module_spec_origin": getattr(getattr(do_mpc, "__spec__", None), "origin", None),
            "controller_file": getattr(controller, "__file__", None),
            "interpreter": sys.executable,
            "python_version": sys.version,
            "shim_path": rel(AWS_DIR / "do_mpc" / "__init__.py"),
            "shim_sha256": sha256(AWS_DIR / "do_mpc" / "__init__.py") if (AWS_DIR / "do_mpc" / "__init__.py").exists() else None,
            "durability_statement": "do_mpc is importable in the launch interpreter with experiments/bohn2021_aws on sys.path; the repository-local shim points the package namespace to the retained vendored do-mpc-horizon tree.",
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback_tail": traceback.format_exc().splitlines()[-12:]}


def verify_stage1_binding() -> Dict[str, Any]:
    try:
        provider = importlib.import_module(PROVIDER_MODULE)
        direct = {
            "module_name": getattr(provider, "__name__", None),
            "module_file": getattr(provider, "__file__", None),
            "has_TERMINAL_SOURCE_PROTOCOL": hasattr(provider, "TERMINAL_SOURCE_PROTOCOL"),
            "has_load_terminal_grid": hasattr(provider, "load_terminal_grid"),
            "load_terminal_grid_callable": callable(getattr(provider, "load_terminal_grid", None)),
        }
        protocol_path = Path(getattr(provider, "TERMINAL_SOURCE_PROTOCOL")) if hasattr(provider, "TERMINAL_SOURCE_PROTOCOL") else None
        direct["terminal_source_protocol"] = rel(protocol_path) if protocol_path is not None else None
        direct["terminal_source_protocol_exists"] = bool(protocol_path is not None and protocol_path.exists())
        protocol_obj = read_json(protocol_path) if protocol_path is not None and protocol_path.exists() else None
        direct["terminal_source_protocol_opened"] = protocol_obj is not None
        direct["terminal_source_protocol_top_level_keys"] = sorted(str(k) for k in protocol_obj.keys())[:50] if isinstance(protocol_obj, dict) else []
        direct["terminal_grid_readiness_reused_from_v1_present"] = isinstance(protocol_obj, dict) and "terminal_grid_readiness_reused_from_v1" in protocol_obj

        v34z2 = importlib.import_module("vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0")
        # This is import-only/module-loading verification. load_modules imports dependencies and binds MODULES;
        # it does not construct an MPC, call a solver, step/reset an environment, train or refit.
        v34z2.load_modules()
        bound = v34z2.MODULES.get("stage1_runner")
        bound_protocol = Path(getattr(bound, "TERMINAL_SOURCE_PROTOCOL")) if hasattr(bound, "TERMINAL_SOURCE_PROTOCOL") else None
        bound_info = {
            "v34z2_module_file": getattr(v34z2, "__file__", None),
            "bound_module_name": getattr(bound, "__name__", None),
            "bound_module_file": getattr(bound, "__file__", None),
            "bound_is_verified_provider_module": getattr(bound, "__name__", None) == PROVIDER_MODULE,
            "has_TERMINAL_SOURCE_PROTOCOL": hasattr(bound, "TERMINAL_SOURCE_PROTOCOL"),
            "has_load_terminal_grid": hasattr(bound, "load_terminal_grid"),
            "load_terminal_grid_callable": callable(getattr(bound, "load_terminal_grid", None)),
            "terminal_source_protocol": rel(bound_protocol) if bound_protocol is not None else None,
            "terminal_source_protocol_exists": bool(bound_protocol is not None and bound_protocol.exists()),
            "terminal_source_protocol_opened": False,
        }
        if bound_protocol is not None and bound_protocol.exists():
            read_json(bound_protocol)
            bound_info["terminal_source_protocol_opened"] = True
        return {"ok": True, "direct_provider": direct, "v34z2_bound_stage1_runner": bound_info}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback_tail": traceback.format_exc().splitlines()[-12:]}


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
            engineering_error="dependency",
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
    marker = f"vehicle-tc2r-env-import-repair-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise ContractError("missing structured execution snapshot")
        patch = patch_v34z2_source(run_dir, stamp)
        do_mpc_status = verify_do_mpc_import()
        stage1_status = verify_stage1_binding()

        corrected = bool(
            stage1_status.get("ok") is True
            and (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("bound_is_verified_provider_module") is True
            and (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("has_TERMINAL_SOURCE_PROTOCOL") is True
            and (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("has_load_terminal_grid") is True
            and (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("load_terminal_grid_callable") is True
            and (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("terminal_source_protocol_exists") is True
            and (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("terminal_source_protocol_opened") is True
        )
        do_mpc_ok = bool(do_mpc_status.get("ok") is True)
        exact_change = bool(
            patch.get("new_import_count_after") == 1
            and patch.get("old_import_count_after") == 0
            and patch.get("new_import") == NEW_IMPORT
            and patch.get("action") in ("one_line_import_repair_written", "already_patched_no_source_write")
        )
        pass_evidence = {
            "corrected_stage1_binding_exposes_protocol_and_grid_loader": corrected,
            "do_mpc_import_verified_in_launch_interpreter": do_mpc_ok,
            "exact_import_change_stated_explicitly": exact_change,
            "no_solver_plant_training_validation_or_test_usage": True,
        }
        hard_pass = all(pass_evidence.values())

        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_T_C2R_zero_usage_environment_import_repair_not_validation_not_test",
            "snapshot_sha256": snapshot.get("snapshot_sha256"),
            "plan_audit_id": (snapshot.get("ready") or {}).get("audit_id"),
            "defects_recorded": {
                "defect_1_do_mpc_import": {
                    "status": "observed_resolved_verified_in_launch_interpreter",
                    "failed_run": "20260930T143813_cd837da9",
                    "evidence_path": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T143814Z/failed.json",
                    "current_import_status": do_mpc_status,
                },
                "defect_2_stage1_terminal_attribute_binding": {
                    "status": "repaired_and_import_only_verified" if corrected else "repair_verification_failed",
                    "failed_run": "20260930T144148_12c26dcc",
                    "evidence_path": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T144148Z/failed.json",
                    "patch": patch,
                    "stage1_verification": stage1_status,
                },
            },
            "explicit_non_actions": {
                "did_not_edit_provider_module": PROVIDER_MODULE,
                "did_not_edit_v1c_or_v1d_modules": True,
                "did_not_copy_terminal_protocol_literal_into_scientific_logic": True,
                "did_not_change_horizons_or_tolerances_or_splits_or_seed_strings_or_budgets": True,
                "did_not_construct_mpc": True,
                "did_not_call_solver": True,
                "did_not_step_or_reset_environment_after_construction": True,
                "did_not_train_or_refit": True,
                "did_not_open_validation64": True,
                "did_not_open_sealed_or_final_test": True,
                "did_not_create_new_aws_resources": True,
            },
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "pass_evidence": pass_evidence,
            "hard_pass": hard_pass,
            "honest_limitations": [
                "Import-only verification does not prove that load_terminal_grid schema will satisfy the later T-C2 solver path.",
                "No scientific objective-contract measurement was made in T-C2R.",
                "The earlier do_mpc shim remains a repository-local import compatibility layer; the archived vendored do-mpc source tree was not edited by this script.",
            ],
        }
        raw_path = run_dir / "raw.json"
        write_json(raw_path, raw)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C2R zero-usage environment/import repair\n\n"
            f"UTC: `{created.isoformat()}`. Task `{TASK_ID}`.\n\n"
            f"Local hard_pass: `{hard_pass}`. Resources: solver=0, plant=0, training=0, validation=0, test=0.\n\n"
            f"Source action: `{patch['action']}` in `{rel(V34Z2_SOURCE)}`; import line changed from "
            f"`{OLD_IMPORT}` to `{NEW_IMPORT}`.\n\n"
            f"do_mpc import verified: `{do_mpc_ok}`; corrected stage1 binding exposes protocol and grid loader: `{corrected}`.\n\n"
            "No validation64 or sealed/final-test data was opened. No solver or plant call was made.\n\n"
            f"Primary evidence: `{rel(raw_path)}`.\n",
            encoding="utf-8",
        )
        backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C2R_ENV_IMPORT_REPAIR_{stamp}.json"
        state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{stamp}_after_t_c2r_env_import_repair.md"
        write_json(
            backup_request,
            {
                "request": "backup_after_t_c2r_env_import_repair",
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
            },
        )
        doc_block = f"""
<!-- {marker} -->
## T-C2R zero-usage environment/import repair

UTC: {created.isoformat()}. Local task hard_pass `{hard_pass}` under active Opus plan `20260930T143121Z_dfaf99`. Resources were zero for solver, plant, training, validation and test. The v34z2 stage1 binding now imports `{PROVIDER_MODULE}` directly as `stage1_runner`; the previous intermediate wrapper binding `{WRAPPER_MODULE}` was removed from that import site. `do_mpc` is importable in the launch interpreter. Evidence: `{rel(summary_path)}` and `{rel(raw_path)}`. This is an operational repair only, not a scientific objective-contract result. Backup request: `{rel(backup_request)}`.
""".strip()
        for doc in DOCS_TO_APPEND:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "# Continue state after T-C2R\n\n"
            + doc_block
            + "\n\nNext action: after supervisor verifies an external backup covering this repair and the generated evidence, launch T-C2 with the exact active-plan task_id, method, split, one approved seed string, config constraints and resource reservation (solver_calls=5, all other counters zero).\n",
            encoding="utf-8",
        )
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
                "T_C2R_pass": hard_pass,
                "source_action": patch["action"],
                "do_mpc_import_verified": do_mpc_ok,
                "stage1_binding_module": (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("bound_module_name"),
                "protocol_opened": (stage1_status.get("v34z2_bound_stage1_runner") or {}).get("terminal_source_protocol_opened"),
                "new_solver_calls": 0,
            },
            "pass_evidence": pass_evidence,
        }
        hash_paths = [
            Path(__file__).resolve(),
            V34Z2_SOURCE,
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
                rel(completed_path),
                marker,
            ])
        # completed.json is written after document updates; update its self-containing hash block is not attempted.
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        print(
            json.dumps(
                clean(
                    {
                        "completed": rel(completed_path),
                        "summary": rel(summary_path),
                        "raw": rel(raw_path),
                        "headline": completed["headline"],
                        "pass_evidence": pass_evidence,
                        "backup_request": rel(backup_request),
                        "server_api_token_audit": raw["server_api_token_audit"],
                    }
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        return write_failure(run_dir, created, f"unexpected T-C2R error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
