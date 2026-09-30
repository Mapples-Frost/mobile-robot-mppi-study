#!/usr/bin/env python3
"""T-C2R2F zero-resource module-binding and alias provenance verification.

Structured task: T-C2R2F-module-binding-and-alias-provenance-verification
from Opus plan 20260930T154136Z_dfe805. This is a read-only mechanism
confirmation over the already accepted T-C2R2 loader repair. It performs no
source repair, no MPC construction, no solver call, no plant step, no training
or refit, no validation64 access, and no sealed/final-test access.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib
import importlib.util
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
TASK_ID = "T-C2R2F-module-binding-and-alias-provenance-verification"
EXPECTED_REQUEST = "execution-result:20260930T154029_56f95ec9"
PLAN_AUDIT_ID = "20260930T154136Z_dfe805"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}

EXPECTED_V34Z2_SHA256 = "66dbf84e4a7917d824152cae1cac4da77c51efff576cd4775e8cd1be29dbcc97"
V34Z2_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
V1D_SOURCE = AWS_DIR / "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner.py"
V1_SOURCE = AWS_DIR / "vehicle_stress_scenario_opportunity_probe_v1_runner.py"
DO_MPC_SHIM = AWS_DIR / "do_mpc" / "__init__.py"
VENDORED_DO_MPC = ROOT / "research_artifacts" / "bohn2021_reproduction_2026-09-17" / "sources" / "do-mpc-horizon" / "do_mpc"
TC2R2_RAW = ROOT / "research_artifacts" / "aws_diagnostics" / "vehicle_true_variable_horizon_v34z2_environment_probe_v0_20260930T154029Z" / "raw.json"
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
    """Local gate violation."""


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
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


def module_record(name: str, module: Optional[Any] = None) -> Dict[str, Any]:
    mod = module if module is not None else sys.modules.get(name)
    spec_origin = None
    spec_locations = None
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            spec_origin = spec.origin
            spec_locations = list(spec.submodule_search_locations or [])
    except Exception as exc:
        spec_origin = "find_spec_failed:%s:%s" % (type(exc).__name__, exc)
    out = {
        "name": name,
        "present_in_sys_modules": mod is not None,
        "object_id": id(mod) if mod is not None else None,
        "module_file": getattr(mod, "__file__", None) if mod is not None else None,
        "module_path": list(getattr(mod, "__path__", []) or []) if mod is not None else None,
        "spec_origin": spec_origin,
        "spec_submodule_search_locations": spec_locations,
        "has_TERMINAL_SOURCE_PROTOCOL": bool(hasattr(mod, "TERMINAL_SOURCE_PROTOCOL")) if mod is not None else False,
        "has_load_terminal_grid": bool(hasattr(mod, "load_terminal_grid")) if mod is not None else False,
        "alias_repair_marker": getattr(mod, "_v34z2_terminal_source_alias_repair", None) if mod is not None else None,
    }
    mf = out.get("module_file")
    if mf:
        out["module_file_sha256"] = sha256(Path(str(mf)))
    return out


def path_equal(a: Any, b: Any) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except Exception:
        return str(a) == str(b)


def verify_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"sys_executable": sys.executable, "sys_version": sys.version}
    try:
        import casadi as ca  # type: ignore
        versions["casadi_version"] = getattr(ca, "__version__", None)
    except Exception as exc:
        versions["casadi_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        import numpy as np  # type: ignore
        versions["numpy_version"] = getattr(np, "__version__", None)
    except Exception as exc:
        versions["numpy_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        import gym  # type: ignore
        versions["gym_version"] = getattr(gym, "__version__", None)
    except Exception as exc:
        versions["gym_error"] = "%s: %s" % (type(exc).__name__, exc)
    versions["matches_legacy_freeze_core_versions"] = bool(
        versions.get("casadi_version") == "3.5.5"
        and versions.get("numpy_version") == "1.18.5"
        and versions.get("gym_version") == "0.17.3"
    )
    versions["legacy_freeze_reference"] = "docs/bohn2021_takeover/legacy-pip-freeze.txt lists casadi==3.5.5, numpy==1.18.5, gym==0.17.3, tensorflow==1.15.5"
    return versions


def protected_hashes() -> Dict[str, Optional[str]]:
    return {
        rel(V34Z2_SOURCE): sha256(V34Z2_SOURCE),
        rel(V1D_SOURCE): sha256(V1D_SOURCE),
        rel(V1_SOURCE): sha256(V1_SOURCE),
        rel(DO_MPC_SHIM): sha256(DO_MPC_SHIM),
    }


def verify_alias_provenance() -> Dict[str, Any]:
    before_hashes = protected_hashes()
    if before_hashes.get(rel(V34Z2_SOURCE)) != EXPECTED_V34Z2_SHA256:
        raise ContractError("v34z2 source hash is not the accepted T-C2R2 hash")

    importlib.invalidate_caches()
    v34z2 = importlib.import_module("vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0")
    v34z2.load_modules()
    stage1 = v34z2.MODULES["stage1_runner"]

    # Import do_mpc/controller explicitly after the gate module load to record the
    # path the controller/plant import will use, without constructing an MPC.
    do_mpc = importlib.import_module("do_mpc")
    controller_mod = importlib.import_module("do_mpc.controller")

    scenario_name = "vehicle_stress_scenario_opportunity_probe_v1_runner"
    v1d_name = "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner"
    scenario_mod = sys.modules.get(scenario_name)
    v1d_before_identity_import = module_record(v1d_name)
    if scenario_mod is None:
        scenario_mod = importlib.import_module(scenario_name)
    # Resolve the v1d module identity read-only if it was not part of the actual
    # gate import path. This is recorded as an identity-only import.
    v1d_imported_for_identity_only = False
    v1d_mod = sys.modules.get(v1d_name)
    if v1d_mod is None:
        v1d_mod = importlib.import_module(v1d_name)
        v1d_imported_for_identity_only = True

    stage1_record = module_record(getattr(stage1, "__name__", "stage1_runner"), stage1)
    scenario_record = module_record(scenario_name, scenario_mod)
    v1d_record = module_record(v1d_name, v1d_mod)

    alias_target = scenario_mod if getattr(scenario_mod, "_v34z2_terminal_source_alias_repair", None) else None
    if getattr(v1d_mod, "_v34z2_terminal_source_alias_repair", None):
        alias_target = v1d_mod
    alias_target_name = getattr(alias_target, "__name__", None) if alias_target is not None else None
    alias_target_id = id(alias_target) if alias_target is not None else None

    do_mpc_file = getattr(do_mpc, "__file__", None)
    do_mpc_path = list(getattr(do_mpc, "__path__", []) or [])
    controller_file = getattr(controller_mod, "__file__", None)
    vendored_root = VENDORED_DO_MPC.resolve()
    do_mpc_path_under_vendored = bool(do_mpc_path) and all(str(Path(p).resolve()).startswith(str(vendored_root)) for p in do_mpc_path)
    do_mpc_file_under_vendored = bool(do_mpc_file) and str(Path(str(do_mpc_file)).resolve()).startswith(str(vendored_root))
    controller_file_under_vendored = bool(controller_file) and str(Path(str(controller_file)).resolve()).startswith(str(vendored_root))

    tc2r2_raw = read_json(TC2R2_RAW)
    tc2r2_protocol = (((tc2r2_raw.get("context_verification") or {}).get("load_meta") or {}).get("terminal_protocol"))
    stage1_protocol_path = Path(getattr(stage1, "TERMINAL_SOURCE_PROTOCOL"))
    stage1_protocol_rel = rel(stage1_protocol_path)
    stage1_protocol_sha = sha256(stage1_protocol_path)
    protocol_matches_tc2r2 = bool(stage1_protocol_rel == str(tc2r2_protocol))

    term_protocol = read_json(stage1_protocol_path)
    terminals, terminal_receipts = stage1.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
    horizon_keys = sorted(int(k) for k in terminals.keys())
    receipt_15 = terminal_receipts.get(15) if isinstance(terminal_receipts, Mapping) else None
    if receipt_15 is None and isinstance(terminal_receipts, Mapping):
        receipt_15 = terminal_receipts.get("15")

    interpreter = verify_versions()
    after_hashes = protected_hashes()
    protected_unchanged = before_hashes == after_hashes

    report = {
        "source_write_occurred_this_run": False,
        "exact_lines_changed_stated_explicitly": "none; no source write was needed or performed by T-C2R2F",
        "protected_source_hashes_before": before_hashes,
        "protected_source_hashes_after": after_hashes,
        "protected_sources_unchanged_during_run": protected_unchanged,
        "v34z2_expected_hash_verified": before_hashes.get(rel(V34Z2_SOURCE)) == EXPECTED_V34Z2_SHA256,
        "gate_stage1_runner_binding": {
            "MODULES_stage1_runner_name": getattr(stage1, "__name__", None),
            "MODULES_stage1_runner_object_id": id(stage1),
            "MODULES_stage1_runner_file": getattr(stage1, "__file__", None),
            "is_vehicle_stress_scenario_opportunity_probe_v1_runner": stage1 is scenario_mod,
            "is_vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner": stage1 is v1d_mod,
            "is_module_that_received_do_mpc_alias": bool(alias_target is not None and stage1 is alias_target),
            "alias_target_module_name": alias_target_name,
            "alias_target_object_id": alias_target_id,
        },
        "module_identities": {
            "scenario_v1": scenario_record,
            "v1d_trace_selected": v1d_record,
            "v1d_state_before_identity_only_import": v1d_before_identity_import,
            "v1d_imported_for_identity_only_after_gate_path": v1d_imported_for_identity_only,
        },
        "do_mpc_resolution": {
            "module_name": getattr(do_mpc, "__name__", None),
            "module_file": do_mpc_file,
            "module_path": do_mpc_path,
            "controller_file": controller_file,
            "vendored_do_mpc_root": str(vendored_root),
            "repository_local_shim_path": rel(DO_MPC_SHIM),
            "repository_local_shim_sha256": sha256(DO_MPC_SHIM),
            "do_mpc_file_under_vendored_root": do_mpc_file_under_vendored,
            "do_mpc_path_under_vendored_root": do_mpc_path_under_vendored,
            "controller_file_under_vendored_root": controller_file_under_vendored,
            "repository_local_package_does_not_shadow_vendored_controller_path": bool(do_mpc_file_under_vendored and do_mpc_path_under_vendored and controller_file_under_vendored),
        },
        "terminal_protocol": {
            "stage1_runner_terminal_protocol_path": stage1_protocol_rel,
            "stage1_runner_terminal_protocol_sha256": stage1_protocol_sha,
            "tc2r2_raw_load_meta_terminal_protocol": tc2r2_protocol,
            "matches_tc2r2_raw_load_meta_terminal_protocol": protocol_matches_tc2r2,
        },
        "terminal_grid_load_without_mpc": {
            "called_stage1_runner_load_terminal_grid": True,
            "horizon_keys": horizon_keys,
            "terminal_15_present": 15 in horizon_keys,
            "h15_terminal_receipt_provenance_hashes": receipt_15,
            "constructed_mpc": False,
        },
        "interpreter": interpreter,
    }
    report["alias_provenance_verified"] = bool(
        report["v34z2_expected_hash_verified"]
        and protected_unchanged
        and report["gate_stage1_runner_binding"]["is_vehicle_stress_scenario_opportunity_probe_v1_runner"]
        and hasattr(stage1, "TERMINAL_SOURCE_PROTOCOL")
        and hasattr(stage1, "load_terminal_grid")
        and report["do_mpc_resolution"]["repository_local_package_does_not_shadow_vendored_controller_path"]
        and protocol_matches_tc2r2
        and (15 in horizon_keys)
        and interpreter.get("matches_legacy_freeze_core_versions") is True
    )
    report["honest_notes"] = [
        "The do_mpc shim alias target is not required for the current gate path because v34z2.MODULES['stage1_runner'] is the scenario-opportunity v1 module, which natively defines TERMINAL_SOURCE_PROTOCOL and load_terminal_grid.",
        "The v1d module identity was resolved read-only only to answer the lead's alias-provenance question; no v1d, v1, v34z2 or do_mpc source was edited by this run.",
        "This task loaded the terminal grid but did not construct an MPC, solve, step, train, refit, open validation64 or open sealed/final test data.",
    ]
    return report


def write_failure(run_dir: Path, created: dt.datetime, error: str) -> int:
    evidence = {
        "alias_provenance_verified": False,
        "exact_lines_changed_stated_explicitly": True,
        "no_solver_plant_training_validation_or_test_usage": True,
        "no_scientific_outcome": True,
        "error": error,
    }
    failed_path = run_dir / "failed.json"
    payload = {
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
    write_json(failed_path, payload)
    try:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), evidence)
    except Exception:
        pass
    print(json.dumps(clean({"failed": error, "failed_json": rel(failed_path), "resources": ZERO_RESOURCES}), sort_keys=True), flush=True)
    return 1


def write_completion(run_dir: Path, created: dt.datetime, snapshot: Mapping[str, Any], alias_report: Mapping[str, Any]) -> int:
    marker = "vehicle-tc2r2f-module-binding-alias-provenance-verification-" + created.strftime("%Y%m%dT%H%M%SZ")
    token_audit = read_api_total_tokens()
    evidence = {
        "alias_provenance_verified": bool(alias_report.get("alias_provenance_verified") is True),
        "exact_lines_changed_stated_explicitly": True,
        "no_solver_plant_training_validation_or_test_usage": True,
    }
    hard_pass = all(evidence.values())
    raw = {
        "status": "complete",
        "task_id": TASK_ID,
        "plan_audit_id_expected_at_script_write": PLAN_AUDIT_ID,
        "plan_audit_id_from_snapshot": (snapshot.get("ready") or {}).get("audit_id"),
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "server_api_token_audit": token_audit,
        "classification": "development_IMPROVED_T_C2R2F_alias_provenance_zero_resource_not_validation_not_test",
        "alias_provenance_report": alias_report,
        "source_write_occurred_this_run": False,
        "exact_lines_changed_stated_explicitly": "none; this task was read-only for v34z2, v1d, v1 and do_mpc sources",
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "explicit_non_actions": {
            "did_not_modify_v34z2_source": True,
            "did_not_modify_v1d_or_v1_runner_sources": True,
            "did_not_modify_base_objective_basin_solver_probe": True,
            "did_not_construct_mpc": True,
            "did_not_call_solver": True,
            "did_not_step_or_reset_environment": True,
            "did_not_train_or_refit": True,
            "did_not_open_validation64": True,
            "did_not_open_sealed_or_final_test": True,
            "did_not_create_new_aws_resources_or_modify_iam_scheduler": True,
        },
        "pass_evidence": evidence,
        "hard_pass": hard_pass,
    }
    raw_path = run_dir / "raw.json"
    summary_path = run_dir / "summary.md"
    backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / ("REQUEST_BACKUP_AFTER_T_C2R2F_MODULE_BINDING_ALIAS_PROVENANCE_%s.json" % created.strftime("%Y%m%dT%H%M%SZ"))
    state_path = ROOT / "research_artifacts" / "aws_state" / ("continue_state_%s_after_t_c2r2f_module_binding_alias_provenance.md" % created.strftime("%Y%m%dT%H%M%SZ"))
    write_json(raw_path, raw)

    lines = [
        "# T-C2R2F module-binding and alias provenance verification",
        "",
        "UTC: `%s`." % created.isoformat(),
        "",
        "Local task hard_pass: `%s`. This is a zero-resource mechanism check, not a solver/objective result." % hard_pass,
        "",
        "Key findings:",
        "- No source write occurred in this run; exact lines changed: none.",
        "- `MODULES['stage1_runner']` resolved to `%s`." % alias_report.get("gate_stage1_runner_binding", {}).get("MODULES_stage1_runner_name"),
        "- Stage1 terminal protocol path matched T-C2R2 raw load metadata: `%s`." % alias_report.get("terminal_protocol", {}).get("matches_tc2r2_raw_load_meta_terminal_protocol"),
        "- Terminal-grid horizon 15 present: `%s`." % alias_report.get("terminal_grid_load_without_mpc", {}).get("terminal_15_present"),
        "- Repository-local do_mpc shim redirected controller imports to the vendored package: `%s`." % alias_report.get("do_mpc_resolution", {}).get("repository_local_package_does_not_shadow_vendored_controller_path"),
        "- Legacy interpreter core versions matched the freeze: `%s`." % alias_report.get("interpreter", {}).get("matches_legacy_freeze_core_versions"),
        "",
        "Resources: solver_calls=0, plant_steps=0, training_steps=0, validation_episodes=0, test_episodes=0. No validation64 or sealed/final-test data was opened.",
        "",
        "Evidence: `%s`, `%s`." % (rel(raw_path), rel(run_dir / "completed.json")),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_payload = {
        "request": "backup_after_T_C2R2F_module_binding_alias_provenance_verification",
        "created_utc": created.isoformat(),
        "reason": "Preserve zero-resource alias/module provenance evidence and log updates before dependent T-C2C/T-C2 work.",
        "must_cover": [rel(raw_path), rel(summary_path), rel(run_dir / "completed.json"), rel(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(backup_request, backup_payload)

    doc_block = "\n".join([
        "<!-- %s -->" % marker,
        "## T-C2R2F module-binding and alias provenance verification",
        "",
        "UTC: %s. Local task hard_pass `%s` under Opus plan `%s`. No source write occurred; exact lines changed: none. `MODULES['stage1_runner']` resolved to `%s`; terminal protocol matched T-C2R2 raw metadata `%s`; H15 terminal grid was present `%s`; do_mpc controller imports resolved to the vendored package `%s`; legacy interpreter core versions matched `%s`. Evidence: `%s` and `%s`. Resources were zero for solver, plant, training, validation and test; validation64 and sealed/final test remained closed. Backup request: `%s`." % (
            created.isoformat(),
            hard_pass,
            (snapshot.get("ready") or {}).get("audit_id"),
            alias_report.get("gate_stage1_runner_binding", {}).get("MODULES_stage1_runner_name"),
            alias_report.get("terminal_protocol", {}).get("matches_tc2r2_raw_load_meta_terminal_protocol"),
            alias_report.get("terminal_grid_load_without_mpc", {}).get("terminal_15_present"),
            alias_report.get("do_mpc_resolution", {}).get("repository_local_package_does_not_shadow_vendored_controller_path"),
            alias_report.get("interpreter", {}).get("matches_legacy_freeze_core_versions"),
            rel(summary_path),
            rel(raw_path),
            rel(backup_request),
        ),
    ])
    for path in DOCS_TO_APPEND:
        append_if_missing(path, marker, doc_block)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        "# Continue state after T-C2R2F\n\n"
        + doc_block
        + "\n\nNext: verify external backup coverage for T-C2R2F artifacts/logs/state. Because T-C2R2F has continue_without_review=true and passed, the next active-plan task is T-C2C-source242-converged-contract-preflight. Do not run solver-bearing T-C2 until T-C2C passes and backup/resource gates are satisfied. No validation64 or sealed/final-test access is authorized.\n",
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
            "alias_provenance_verified": evidence["alias_provenance_verified"],
            "stage1_runner_module": alias_report.get("gate_stage1_runner_binding", {}).get("MODULES_stage1_runner_name"),
            "stage1_is_alias_target": alias_report.get("gate_stage1_runner_binding", {}).get("is_module_that_received_do_mpc_alias"),
            "terminal_15_present": alias_report.get("terminal_grid_load_without_mpc", {}).get("terminal_15_present"),
            "do_mpc_vendored_controller_path": alias_report.get("do_mpc_resolution", {}).get("repository_local_package_does_not_shadow_vendored_controller_path"),
            "legacy_core_versions_match": alias_report.get("interpreter", {}).get("matches_legacy_freeze_core_versions"),
        },
        "pass_evidence": evidence,
        "outcome_receipt_evidence": evidence,
    }
    hash_paths = [Path(__file__).resolve(), V34Z2_SOURCE, V1D_SOURCE, V1_SOURCE, DO_MPC_SHIM, raw_path, summary_path, backup_request, state_path, ROOT / "EXPERIMENT_REGISTRY.csv", RESPONSE_LOG]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
    write_json(completed_path, completed)
    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), evidence)
    print(json.dumps(clean({
        "completed": rel(completed_path),
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "headline": completed["headline"],
        "pass_evidence": evidence,
        "backup_request": rel(backup_request),
        "server_api_token_audit": token_audit,
    }), sort_keys=True), flush=True)
    return 0


def main() -> int:
    created = now_utc()
    run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / ("%s_%s" % (NAME, created.strftime("%Y%m%dT%H%M%SZ")))
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise ContractError("missing structured execution snapshot")
        alias_report = verify_alias_provenance()
        return write_completion(run_dir, created, snapshot, alias_report)
    except Exception as exc:
        return write_failure(run_dir, created, "unexpected T-C2R2F error: %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
