#!/usr/bin/env python3
"""v34g one-cell objective-reconstruction smoke for Opus Task B.

This is the dependency gate between the v34f backup-gate repair and the full
24-call objective-vs-basin probe.  It spends exactly one lower-level MPC solve
on the frozen cell requested by the active Opus lead:

    execution_index=0; source242_slot0_branch_start; H15; V15_shared; canonical

It does not run plant rollouts, env.reset/env.step after construction,
selector refit/search, training, validation64, or sealed/final test.  The smoke
records the solver objective, reconstructed stage+terminal objective, residuals,
and complete initial/solution/parameter/bounds arrays so that G-B can pass or
fail before the remaining 23 solves are considered.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

import vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0 as f

base = f.base
ROOT = f.ROOT
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34g_objective_reconstruction_smoke.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34G_OBJECTIVE_RECONSTRUCTION_SMOKE_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34g-objective-reconstruction-smoke-{STAMP}"
MARKER = f"vehicle-v34g-objective-reconstruction-smoke-{STAMP}"
TARGET_CELL = {
    "execution_index": 0,
    "context_id": "source242_slot0_branch_start",
    "state_label": "v27_case09_slot0_early_risk",
    "horizon": 15,
    "terminal_mode": "V15_shared",
    "initialization": "canonical",
    "blocked_randomization_unit": "v34g|source242_slot0_branch_start|H15",
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    return base.rel(path)


def sha256(path: Path) -> str:
    return base.sha256(path)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def hash_existing(paths: Sequence[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def finite(value: Any) -> Optional[float]:
    try:
        if hasattr(value, "full"):
            a = np.asarray(value.full(), dtype=float).reshape(-1)
        elif hasattr(value, "cat"):
            a = np.asarray(value.cat, dtype=float).reshape(-1)
        else:
            a = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if a.size != 1:
        return None
    out = float(a[0])
    return out if math.isfinite(out) else None


def struct_snapshot(obj: Any, label_limit: int = 20000) -> Dict[str, Any]:
    flat = base.arr(obj)
    labels = base.get_struct_labels(obj)
    snap: Dict[str, Any] = {
        "flat_size": int(flat.size),
        "flat_hash": base.arr_hash(obj),
        "flat_values": flat.tolist(),
        "labels_count": len(labels),
        "labels_truncated": len(labels) > label_limit,
    }
    labeled: Dict[str, Optional[float]] = {}
    for label in labels[:label_limit]:
        parts = base.parse_label(label)
        try:
            labeled[label] = finite(obj[tuple(parts)])
        except Exception:
            labeled[label] = None
    if labels:
        snap["labeled_values"] = labeled
    return snap


def install_full_capture() -> None:
    original = base.capture_mpc_state

    def capture_mpc_state_full(mpc: Any) -> Dict[str, Any]:
        out = original(mpc)
        out["complete_scalar_opt_f_num"] = finite(getattr(mpc, "opt_f_num", None))
        for attr in (
            "opt_x_num",
            "opt_x_num_unscaled",
            "opt_p_num",
            "opt_g_num",
            "opt_x_lb",
            "opt_x_ub",
            "opt_g_lb",
            "opt_g_ub",
            "lam_g_num",
        ):
            if hasattr(mpc, attr):
                try:
                    out["complete_" + attr] = struct_snapshot(getattr(mpc, attr))
                except Exception as exc:
                    out["complete_" + attr] = {"error": repr(exc)}
        return out

    base.capture_mpc_state = capture_mpc_state_full


def patch_runtime() -> None:
    f.patch_runtime()
    base.NAME = NAME
    base.STAMP = STAMP
    base.RUN_DIR = RUN_DIR
    base.STATE = STATE
    base.BACKUP_REQUEST = BACKUP_REQUEST
    base.REQUEST_ID = REQUEST_ID
    base.MARKER = MARKER
    base.__file__ = str(Path(__file__).resolve())
    install_full_capture()


def summarize_required_outputs(arm: Mapping[str, Any]) -> Dict[str, Any]:
    ev = arm.get("solver_event") or {}
    recon = arm.get("objective_reconstruction_current") or {}
    pre = ev.get("pre") or {}
    post = ev.get("post") or {}
    residual = {"constraint_residual": ev.get("constraint_residual"), "bound_residual": ev.get("bound_residual")}
    rel_err = recon.get("relative_error_vs_solver")
    residual_gate = (
        residual["constraint_residual"] is not None
        and residual["bound_residual"] is not None
        and float(residual["constraint_residual"]) <= base.ACCEPT_RESIDUAL_TOL
        and float(residual["bound_residual"]) <= base.ACCEPT_RESIDUAL_TOL
    )
    objective_gate = rel_err is not None and float(rel_err) <= base.RECON_REL_TOL
    solver_success_gate = bool(ev.get("success")) and ev.get("objective_opt_f_num") is not None
    failure_reasons = list(recon.get("failure_reasons") or [])
    if not solver_success_gate:
        failure_reasons.append("solver_success_gate_failed")
    if not residual_gate:
        failure_reasons.append("residual_gate_failed_non_none_and_le_1e-5")
    if not objective_gate:
        failure_reasons.append("objective_reconstruction_relative_error_gate_failed_le_1e-6")
    return {
        "J_solver": ev.get("objective_opt_f_num"),
        "stage_lterm_total": recon.get("stage_lterm_total"),
        "gamma": recon.get("gamma"),
        "gamma_source": recon.get("gamma_source"),
        "terminal_value_current_vf": recon.get("terminal_value_current_vf"),
        "gamma_pow_H_terminal": recon.get("gamma_pow_H_terminal"),
        "J_reconstructed_stage_plus_terminal": recon.get("reconstructed_total_current_objective"),
        "relative_error_vs_solver": rel_err,
        "failure_reasons": failure_reasons,
        "status": {"return_status": ev.get("return_status"), "success": ev.get("success"), "iterations": ev.get("iterations"), "solver_exception": ev.get("solver_exception")},
        "residual": residual,
        "x_initial": pre.get("complete_opt_x_num"),
        "x_initial_unscaled": pre.get("complete_opt_x_num_unscaled"),
        "x_solution": post.get("complete_opt_x_num"),
        "x_solution_unscaled": post.get("complete_opt_x_num_unscaled"),
        "opt_p": post.get("complete_opt_p_num"),
        "bounds": {
            "opt_x_lb": post.get("complete_opt_x_lb"),
            "opt_x_ub": post.get("complete_opt_x_ub"),
            "opt_g_lb": post.get("complete_opt_g_lb"),
            "opt_g_ub": post.get("complete_opt_g_ub"),
        },
        "attempt_ledger": {
            "solve_calls": arm.get("solve_calls"),
            "plant_steps": arm.get("plant_steps"),
            "env_reset_calls": arm.get("env_reset_calls"),
            "controller_get_action_wall_s": arm.get("controller_get_action_wall_s"),
            "solver_wall_s": ev.get("solver_wall_s"),
            "terminal_mode": arm.get("terminal_mode"),
            "horizon": arm.get("horizon"),
            "initialization": arm.get("initialization"),
            "accepted_by_v34_legacy_rule": arm.get("accepted"),
        },
        "objective_gate_pass": bool(objective_gate),
        "residual_gate_pass": bool(residual_gate),
        "solver_success_gate_pass": bool(solver_success_gate),
        "hard_pass": bool(objective_gate and residual_gate and solver_success_gate),
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    g = raw["objective_smoke_gate"]
    lines = [
        "# v34g one-cell objective-reconstruction smoke",
        "",
        f"Created UTC: `{raw['created_utc']}`. Development-only Opus Task B gate; no validation64 or sealed/final test access.",
        "",
        "## Gate result",
        "",
        f"- hard_pass: `{g['hard_pass']}`",
        f"- solver calls: `{raw['budget_actual']['solver_calls']}`; plant steps: `{raw['budget_actual']['plant_steps']}`; env_reset calls: `{raw['budget_actual']['env_reset_calls']}`",
        f"- J_solver: `{g['J_solver']}`",
        f"- stage_lterm_total: `{g['stage_lterm_total']}`",
        f"- gamma_pow_H_terminal: `{g['gamma_pow_H_terminal']}`",
        f"- reconstructed total: `{g['J_reconstructed_stage_plus_terminal']}`",
        f"- relative_error_vs_solver: `{g['relative_error_vs_solver']}` (gate <= {base.RECON_REL_TOL})",
        f"- residual: `{g['residual']}` (both non-None and <= {base.ACCEPT_RESIDUAL_TOL})",
        f"- solver status: `{g['status']}`",
        f"- failure_reasons: `{g['failure_reasons']}`",
        "",
        "## Next gate",
        "",
    ]
    if g["hard_pass"]:
        lines.append("G-B passed. After external backup covering this smoke source/output, the already-approved next task is the remaining fixed objective-vs-basin probe budget; do not open validation64 or sealed test.")
    else:
        lines.append("G-B failed. Do not spend the remaining 23 solver calls. Localize objective reconstruction/residual mismatch before scientific attribution.")
    lines += ["", f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    g = raw["objective_smoke_gate"]
    block = f"""
<!-- {MARKER} -->
## v34g one-cell objective-reconstruction smoke

UTC: {raw['created_utc']}. Opus Task B executed one fixed-context solver call for `{raw['selected_cell']['context_id']}|H{raw['selected_cell']['horizon']}|{raw['selected_cell']['terminal_mode']}|{raw['selected_cell']['initialization']}`. hard_pass={g['hard_pass']}; J_solver={g['J_solver']}; reconstructed={g['J_reconstructed_stage_plus_terminal']}; relative_error={g['relative_error_vs_solver']}; residual={g['residual']}; solver_status={g['status']}. Budgets: solver_calls={raw['budget_actual']['solver_calls']}, plant_steps=0, env_reset_calls=0, training/refit=0, validation64=false, sealed_test=false. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(ROOT / doc if isinstance(doc, str) else doc, MARKER, block)
    review_status = "gate_passed_dependent_task_authorized_after_backup" if g["hard_pass"] else "gate_failed_requires_objective_localization"
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": review_status,
        "trigger": "v34g one-cell objective reconstruction smoke completed",
        "experiment_id": NAME,
        "active_lead_report": rel(f.CURRENT_OPUS_REPORT),
        "active_lead_report_sha256": f.CURRENT_OPUS_SHA,
        "question": "Review the Task B one-cell objective reconstruction smoke. If hard_pass is false, diagnose objective/residual reconstruction before any remaining solver calls. If hard_pass is true, the next dependent task is the fixed objective-vs-basin probe after backup; preserve no validation64/sealed-test access.",
        "evidence_paths": [rel(RUN_DIR / "summary.md"), rel(RUN_DIR / "raw.json"), rel(RUN_DIR / "completed.json"), rel(f.CURRENT_OPUS_REPORT), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "gate": {k: g.get(k) for k in ["hard_pass", "relative_error_vs_solver", "residual", "J_solver", "J_reconstructed_stage_plus_terminal", "failure_reasons"]},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "plant_steps": 0, "training_or_refit": 0},
        "operational_note": f"Post-smoke backup required via {rel(BACKUP_REQUEST)} before additional unique solver/scientific work.",
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Continue state after v34g objective-reconstruction smoke\n\nUTC: {raw['created_utc']}\n\nGate: {json.dumps(g, sort_keys=True, default=str)[:4000]}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'completed.json')}\n\nNext: external backup for {rel(BACKUP_REQUEST)}. If hard_pass true, run the approved fixed objective-vs-basin probe; if false, localize objective reconstruction/residual before spending remaining calls.\n",
        encoding="utf-8",
    )


def make_backup_request(raw: Mapping[str, Any]) -> None:
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34g_objective_reconstruction_smoke",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new one-cell objective reconstruction smoke source, raw arrays, docs, registry handoff must be externally recoverable before remaining objective-vs-basin solver calls or localization reruns",
        "must_cover": [
            rel(Path(__file__).resolve()),
            rel(RUN_DIR),
            rel(STATE),
            rel(BACKUP_REQUEST),
            rel(NEXT_REVIEW_REQUEST),
            rel(RESPONSE_LOG),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
        ],
        "new_solver_calls": raw["budget_actual"]["solver_calls"],
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "G-B pass allows approved fixed objective-vs-basin probe after backup; G-B fail requires objective/residual localization before remaining calls",
    })


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, default=0)
    ap.add_argument("--i-accept-v34g-one-cell-objective-smoke", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    patch_runtime()
    try:
        started = now_utc()
        gates = f.verify_opus_and_backup_gates(args)
        write_json(RUN_DIR / "run_started.json", {
            "started_utc": started.isoformat(),
            "pid": os.getpid(),
            "method": NAME,
            "selected_cell": TARGET_CELL,
            "budget_cap_solver_calls": 1,
            "plant_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "gates": gates,
        })
        _, stage1_runner, _ = base.v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise base.ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        term_protocol = base.read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        contexts = base.load_contexts()
        context_by_id = {str(c["context_id"]): c for c in contexts}
        if TARGET_CELL["context_id"] not in context_by_id:
            raise base.ContractError("target context missing from v34/v34c strict context loader")
        coeffs = base.parse_terminal_coeff_csv()
        write_json(RUN_DIR / "frozen_smoke_cell.json", {
            "created_utc": now_utc().isoformat(),
            "selected_cell": TARGET_CELL,
            "contexts": [{k: v for k, v in c.items() if k != "case_snapshot"} for c in contexts],
            "solver_call_cap": 1,
            "plant_steps": 0,
            "env_reset_calls": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        write_json(RUN_DIR / "parsed_terminal_coefficients.json", coeffs)
        arm = base.solve_arm(copy.deepcopy(TARGET_CELL), context_by_id[TARGET_CELL["context_id"]], terminals, coeffs)
        if int(arm.get("solve_calls", 0)) != 1:
            raise base.ContractError("one-cell smoke did not record exactly one solve call")
        if int(arm.get("plant_steps", 0)) != 0 or int(arm.get("env_reset_calls", 0)) != 0:
            raise base.ContractError("forbidden plant/reset call count nonzero in one-cell smoke")
        gate = summarize_required_outputs(arm)
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_one_cell_objective_reconstruction_smoke_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(f.CURRENT_OPUS_REPORT),
            "lead_report_sha256": f.CURRENT_OPUS_SHA,
            "hypothesis_frozen": "A single fixed-context H15/V15/canonical lower-level solve can verify that reconstructed stage+terminal objective matches solver opt_f and residuals are explicit before spending the remaining objective-vs-basin probe calls.",
            "selected_cell": TARGET_CELL,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_declared": {"solver_call_cap": 1, "plant_steps": 0, "env_reset_calls": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"arms": 1, "solver_calls": int(arm.get("solve_calls", 0)), "plant_steps": 0, "env_reset_calls": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "gates": gates,
            "runtime_preflight": preflight,
            "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
            "contexts": [{k: v for k, v in c.items() if k != "case_snapshot"} for c in contexts],
            "arm": arm,
            "objective_smoke_gate": gate,
            "interpretation_limits": ["opened development context only", "one lower-level solve only", "not validation64", "not sealed test", "no closed-loop adaptive selector", "no plant continuation"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "input_hashes": hash_existing([
                Path(__file__).resolve(),
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0.py",
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0.py",
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py",
                f.OPUS_PLAN_READY,
                f.OPUS_LATEST,
                f.CURRENT_OPUS_REPORT,
                f.d.V0E_DONE,
            ]),
            "backup_request_after_run": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        update_docs(raw)
        make_backup_request(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [
            Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG,
            ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
        ]
        completed = {
            "status": "complete",
            "passed": bool(gate["hard_pass"]),
            "hard_pass": bool(gate["hard_pass"]),
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": raw["budget_actual"],
            "headline": {k: gate.get(k) for k in ["hard_pass", "J_solver", "J_reconstructed_stage_plus_terminal", "relative_error_vs_solver", "residual", "status", "failure_reasons"]},
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_gate": "task C full objective-vs-basin probe after backup" if gate["hard_pass"] else "objective/residual localization before remaining solver calls",
            "next_review_request_id": REQUEST_ID,
            "hashes": hash_existing(sorted(set(files))),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST), "next_review_request_id": REQUEST_ID}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {
            "status": "failed",
            "created_utc": now_utc().isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "classification": "development_IMPROVED_one_cell_objective_reconstruction_smoke_not_validation_not_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_caps": {"solver_call_cap": 1, "plant_steps": 0, "env_reset_calls": 0},
            "selected_cell": TARGET_CELL,
        }
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# v34g objective reconstruction smoke failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR / 'failed.json')}\n\nNo validation64 or sealed test access was requested. Inspect traceback and exact primitive before retrying; do not spend remaining 23 calls.\n",
            encoding="utf-8",
        )
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
