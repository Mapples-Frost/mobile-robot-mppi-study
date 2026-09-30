#!/usr/bin/env python3
"""v34r / T-B2 zero-solve loader gate after v34q goal-source enumeration.

This is a narrow operational continuation of v34o.  It keeps v34o's strict
previous_input singleton scalarization and fail-loud _u0 path unchanged, but
patches only goal reconstruction to the authoritative saved-case
reference/trajectory endpoint source established by v34q/T-B1 and the already
successful v34g->v34f->v34d->v34c/v0e/v0c construction chain.

Budget: 0 solver calls, 0 plant/env steps, 0 training/refit, validation64=0,
sealed/final test=0.  Passing this gate authorizes a backup request before the
already approved A13c-3 solver probe; it is not itself scientific validation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34o_loader_gate_v0 as v34o  # noqa:E402
import vehicle_true_variable_horizon_v34q_goal_source_enumeration_v0 as v34q  # noqa:E402

base = v34o.base
NAME = "vehicle_true_variable_horizon_v34r_loader_gate_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34r_loader_gate.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34R_LOADER_GATE_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T114431Z_6f8902.md"
OPUS_SHA = "f0494a123570fd389a81ffd563bdf610964741b6f578e7f5f410cd23c27bbe3c"
OPUS_REQUEST = "execution-result:20260930T114327_085d17fb"
EXPECTED_PREVIOUS_HASHES = {
    "source242_slot0_branch_start": "a5d5f196566b4558fc156886da3a05d2ff86f2ffa2977ed37c89adb61860b660",
    "v19_c13_step1_after_V15_H35_common_state": "f3ebf9a7c07101f14105b8afeff83421166f75d617357cbc09716e09341b5139",
}
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-v34r-loader-gate-{STAMP}"
CHECK_HORIZONS = (12, 15, 35)


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    return v34o.rel(path)


def clean(value: Any) -> Any:
    return v34o.clean(value)


def write_json(path: Path, value: Any) -> None:
    return v34o.write_json(path, value)


def read_json(path: Path) -> Any:
    return v34o.read_json(path)


def sha256(path: Path) -> str:
    return v34o.sha256(path)


def canonical_hash(value: Any) -> str:
    return v34o.canonical_hash(value)


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    return v34o.hash_existing(paths)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def latest_v34q_completed() -> Path:
    paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34q_goal_source_enumeration_v0_*/completed.json")))]
    if not paths:
        raise ContractError("v34q/T-B1 completed.json not found")
    return paths[-1]


def verify_active_plan_and_v34q() -> Dict[str, Any]:
    for p in [PLAN_READY, OPUS_REPORT, OPUS_LATEST]:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    ready = read_json(PLAN_READY)
    actual = sha256(OPUS_REPORT)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_SHA or actual != OPUS_SHA:
        raise ContractError(f"active Opus plan mismatch: request={ready.get('request_id')} ready_sha={ready.get('report_sha256')} actual_sha={actual}")
    latest = OPUS_LATEST.read_text(encoding="utf-8", errors="replace")
    if rel(OPUS_REPORT) not in latest:
        raise ContractError("Opus LATEST.md does not point to active report")
    q_path = latest_v34q_completed()
    q_done = read_json(q_path)
    if q_done.get("hard_pass") is not True or q_done.get("passed") is not True:
        raise ContractError("latest v34q/T-B1 did not pass G-GOAL")
    qb = q_done.get("budget_actual") or {}
    for key in ["solver_calls", "plant_steps", "env_step_calls_after_construction", "env_reset_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]:
        if int(qb.get(key, 0)) != 0:
            raise ContractError(f"v34q budget not zero for {key}: {qb.get(key)}")
    return {
        "plan_ready": rel(PLAN_READY),
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_SHA,
        "active_lead_request": OPUS_REQUEST,
        "v34q_completed": rel(q_path),
        "v34q_completed_sha256": sha256(q_path),
        "v34q_headline": q_done.get("headline"),
        "G_GOAL_pass": True,
    }


def robust_extract_goal_xy_v34r(case: Mapping[str, Any], shifted_tvp: Mapping[str, Any]) -> Tuple[float, float, str]:
    # Copy-only use of the source established by v34q/v34c-v0c: full saved-case
    # reference.traj_steps plus saved trajectory TVP endpoint.  shifted_tvp is
    # ignored for authority but included in the source string for auditability.
    del shifted_tvp
    got = v34q.robust_goal_from_saved_case(case)
    if got.get("ok") is not True:
        raise base.ContractError("v34r authoritative saved-case goal extraction failed: " + repr(got))
    return float(got["goal_x"]), float(got["goal_y"]), str(got.get("source")) + "|v34q_authoritative_copy"


def patch_runtime() -> Dict[str, Any]:
    base.extract_goal_xy = robust_extract_goal_xy_v34r  # type: ignore[assignment]
    base.load_contexts = v34o.load_contexts_strict  # type: ignore[assignment]
    base.configure_context_no_reset = v34o.configure_context_no_reset_strict  # type: ignore[assignment]
    return {
        "patch_scope": "in_process_extension_file_only",
        "patched_symbols": ["base.extract_goal_xy", "base.load_contexts", "base.configure_context_no_reset"],
        "goal_source": "v34q.robust_goal_from_saved_case / saved reference.traj_steps + trajectory TVP endpoint",
        "previous_input_scalarization": "unchanged from v34o.load_contexts_strict and v34o.normalize_previous_input",
        "source_file": rel(Path(__file__).resolve()),
        "v34o_file": rel(Path(v34o.__file__).resolve()),
        "v34q_file": rel(Path(v34q.__file__).resolve()),
        "base_file": rel(Path(base.__file__).resolve()),
    }


def write_summary_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# v34r / T-B2 zero-solve loader gate",
        "",
        f"UTC: `{raw['created_utc']}`. Operational loader gate after v34q G-GOAL under active Opus plan `{raw['active_lead_report']}`.",
        "",
        "## Budget",
        "- Solver calls: `0`; plant steps: `0`; env.step/reset after construction: `0`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Headline",
        f"- passed: `{h['passed']}`; checks: `{h['context_horizon_checks']}`; exceptions: `{h['exception_count']}`; solver calls: `{h['solver_calls']}`.",
        f"- Previous-input hash match: `{h['previous_input_hashes_match_expected']}`; hashes: `{h['previous_input_scalar_hashes']}`.",
        f"- _u0 norm range: `{h['u0_norm_min']}` to `{h['u0_norm_max']}`.",
        "",
        "| context | H | ok | goal_source | goal_x | goal_y | _u0 shape | _u0 norm | input vector source | state distance |",
        "|---|---:|---|---|---:|---:|---|---:|---|---:|",
    ]
    for r in raw["context_horizon_checks"]:
        meta = r.get("u0_vector_meta") or {}
        lines.append(f"| `{r['context_id']}` | {r['horizon']} | `{r.get('ok')}` | `{r.get('goal_source')}` | {r.get('goal_x')} | {r.get('goal_y')} | `{r.get('u0_shape')}` | {r.get('u0_norm2')} | `{meta.get('vector_source')}` | {r.get('state_distance_after_config')} |")
    lines += [
        "",
        "This passes only the loader/configuration contract. It does not spend A13c-3 solver calls and is not validation/final-test evidence.",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`; completed: `{rel(RUN_DIR/'completed.json')}`; previous-input CSV: `{rel(RUN_DIR/'previous_input_normalization.csv')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (RUN_DIR / "previous_input_normalization.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["context_id", "channel", "scalar", "rule", "raw_type", "flat_size", "raw_preview"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in raw["contexts_no_case_snapshot"]:
            norm = c.get("previous_input_normalization") or {}
            for ch, meta in (norm.get("channel_rules") or {}).items():
                w.writerow({"context_id": c.get("context_id"), "channel": ch, "scalar": meta.get("scalar"), "rule": meta.get("rule"), "raw_type": meta.get("raw_type"), "flat_size": meta.get("flat_size"), "raw_preview": json.dumps(clean(meta.get("raw_preview")), ensure_ascii=False)})
    block = f"""
<!-- {MARKER} -->
## v34r/T-B2 zero-solve loader gate

UTC: {raw['created_utc']}. Ran the Opus-authorized loader gate after v34q G-GOAL. Only goal extraction changed from v34o, using the v34q/v34c-v0c saved reference/trajectory endpoint source; v34o previous_input scalarization is unchanged. Budget: solver_calls=0, plant/env steps=0, training/refit=0, validation64=0, sealed_test=0. passed={h['passed']}; checks={h['context_horizon_checks']}; exceptions={h['exception_count']}; previous_input_hashes_match_expected={h['previous_input_hashes_match_expected']}; u0_norm_range=[{h['u0_norm_min']}, {h['u0_norm_max']}]. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`, `{rel(RUN_DIR/'previous_input_normalization.csv')}`. Backup request before A13c-3 solver calls: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "T-B2; H=12,15,35; contexts=source242,c13; v34q goal source", "zero_solve_loader_gate_no_validation_no_test", h["context_horizon_checks"], 0, 0, 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34r/T-B2 zero-solve loader gate\n\n"
        + f"UTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(h), sort_keys=True)}\n\n"
        + f"Artifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'completed.json')}, {rel(RUN_DIR/'previous_input_normalization.csv')}\n\n"
        + f"Backup request before A13c-3 solver calls: {rel(BACKUP_REQUEST)}\n\n"
        + "Next: obtain verified external backup covering v34q/v34r sources and outputs, then run v34p/A13c-3 <=6 low-level solver attempts under the active Opus plan. Do not open validation64 or sealed test.\n",
        encoding="utf-8",
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--i-accept-zero-solve-loader-gate", action="store_true", required=True)
    args = ap.parse_args(argv)
    del args
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        gates = verify_active_plan_and_v34q()
        patch_info = patch_runtime()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "gates": gates, "patch_info": patch_info, "budget": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "training_or_refit": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}})
        _, stage1_runner, _ = base.v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        term_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        contexts = v34o.load_contexts_strict()
        context_summaries: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []
        exceptions: List[Dict[str, Any]] = []
        for c in contexts:
            context_summaries.append({k: v for k, v in c.items() if k != "case_snapshot"})
            for h in CHECK_HORIZONS:
                rec: Dict[str, Any] = {"context_id": c["context_id"], "state_label": c["state_label"], "horizon": int(h), "solver_call_attempts": 0, "ok": False}
                try:
                    shifted = base.shift_case_tvp(c["case_snapshot"], int(c["tvp_start_index"]), int(h))
                    rec["tvp_lengths"] = {k: len(v) for k, v in shifted.items()}
                    rec["tvp_min_len"] = min(rec["tvp_lengths"].values()) if rec["tvp_lengths"] else None
                    rec["tvp_hash"] = canonical_hash(shifted)
                    terminal, _, _ = base.terminal_for_mode(15, "V15_shared", terminals)
                    env = base.create_env(int(h), terminal)
                    ctrl = env.control_system.controller
                    v34o.install_zero_solve_guard(ctrl.mpc, rec)
                    meta = v34o.configure_context_no_reset_strict(env, c, int(h))
                    rec["previous_input"] = clean(meta["previous_input_applied"])
                    rec["previous_input_hash"] = canonical_hash(meta["previous_input_applied"])
                    rec["previous_input_normalization"] = meta.get("previous_input_normalization")
                    rec["u0_shape"] = (meta.get("u0_vector_meta") or {}).get("shape")
                    rec["u0_norm2"] = (meta.get("u0_vector_meta") or {}).get("norm2")
                    rec["u0_hash"] = (meta.get("u0_vector_meta") or {}).get("hash")
                    rec["u0_vector_meta"] = meta.get("u0_vector_meta")
                    rec["u0_master_fingerprint"] = meta.get("u0_master_fingerprint")
                    rec["state_distance_after_config"] = meta.get("state_distance_after_config")
                    rec["goal_source"] = meta.get("goal_source")
                    rec["goal_x"] = meta.get("goal_x")
                    rec["goal_y"] = meta.get("goal_y")
                    rec["ok"] = True
                except Exception as exc:
                    rec["ok"] = False
                    rec["error"] = repr(exc)
                    rec["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
                    exceptions.append(rec)
                checks.append(rec)
                write_json(RUN_DIR / "progress.json", {"checks_done": len(checks), "checks_expected": len(contexts) * len(CHECK_HORIZONS), "last_check": rec, "solver_calls": sum(int(x.get("solver_call_attempts", 0)) for x in checks), "validation64_bank_opened": False, "sealed_test_accessed": False})
                print(json.dumps(clean({"checks_done": len(checks), "context": c["context_id"], "H": h, "ok": rec["ok"], "solver_call_attempts": rec.get("solver_call_attempts", 0), "goal_source": rec.get("goal_source"), "u0_norm2": rec.get("u0_norm2")}), sort_keys=True), flush=True)
        total_solve_attempts = int(sum(int(x.get("solver_call_attempts", 0)) for x in checks))
        context_hashes = {c["context_id"]: c.get("previous_input_normalization", {}).get("hash") for c in context_summaries}
        previous_match = context_hashes == EXPECTED_PREVIOUS_HASHES
        u0_norms = [float(x["u0_norm2"]) for x in checks if x.get("u0_norm2") is not None]
        passed = bool(len(contexts) == 2 and len(checks) == 6 and all(x.get("ok") for x in checks) and total_solve_attempts == 0 and previous_match)
        headline = {
            "passed": passed,
            "contexts_loaded": len(contexts),
            "context_horizon_checks": len(checks),
            "exception_count": len(exceptions),
            "solver_calls": total_solve_attempts,
            "plant_steps": 0,
            "env_step_calls_after_construction": 0,
            "env_reset_calls_after_construction": 0,
            "training_or_refit": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
            "previous_input_scalar_hashes": context_hashes,
            "expected_previous_input_scalar_hashes": EXPECTED_PREVIOUS_HASHES,
            "previous_input_hashes_match_expected": previous_match,
            "u0_norm_min": min(u0_norms) if u0_norms else None,
            "u0_norm_max": max(u0_norms) if u0_norms else None,
            "goal_sources": sorted({str(x.get("goal_source")) for x in checks if x.get("goal_source")}),
        }
        created = now_utc()
        raw: Dict[str, Any] = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_operational_zero_solve_loader_gate_after_goal_source_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "active_lead_report": rel(OPUS_REPORT),
            "active_lead_report_sha256": OPUS_SHA,
            "active_lead_request": OPUS_REQUEST,
            "hypothesis_frozen": "Using v34q's authoritative saved reference/trajectory goal source, v34o's strict previous_input loader can configure both opened contexts for H=12/15/35 and _u0 without any solver/plant access, with previous_input hashes unchanged.",
            "gates": gates,
            "patch_info": patch_info,
            "runtime_preflight": preflight,
            "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
            "contexts_no_case_snapshot": context_summaries,
            "context_horizon_checks": checks,
            "exceptions": exceptions,
            "headline": headline,
            "budget_declared": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"solver_calls": total_solve_attempts, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "input_hashes": hash_existing([Path(__file__).resolve(), Path(v34o.__file__).resolve(), Path(v34q.__file__).resolve(), Path(base.__file__).resolve(), PLAN_READY, OPUS_LATEST, OPUS_REPORT, Path(gates["v34q_completed"])]),
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
            "interpretation_limits": ["operational loader/configuration gate only", "no low-level solves", "no plant rollouts", "not validation64", "not sealed/final test", "does not test objective reconstruction A13c-3"],
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary_docs(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v34r_loader_gate", "created_utc": created.isoformat(), "backup_required_before_a13c3_solver_calls": True, "reason": "v34q/v34r source, zero-solve artifacts and state must be externally recoverable before v34p/A13c-3 solver calls", "must_cover": [rel(Path(__file__).resolve()), rel(Path(v34q.__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_solver_calls": 0, "new_plant_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "next_gate": "After verified backup, run v34p/A13c-3 <=6 low-level solver attempts under active Opus plan; no validation64 or sealed test."})
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), Path(v34q.__file__).resolve(), STATE, BACKUP_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {"status": "complete", "passed": passed, "hard_pass": passed, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": headline, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "previous_input_normalization_csv": rel(RUN_DIR / "previous_input_normalization.csv"), "backup_request": rel(BACKUP_REQUEST), "next_if_backup_verified": "vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py", "hashes": hash_existing(files)}
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR/"completed.json"), "summary": rel(RUN_DIR/"summary.md"), "raw": rel(RUN_DIR/"raw.json"), "headline": headline, "backup_request": rel(BACKUP_REQUEST), "next_if_backup_verified": "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py"}, sort_keys=True), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_operational_zero_solve_loader_gate_after_goal_source_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34r/T-B2 loader gate failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo solver, plant, validation64, sealed-test, training or refit access was requested. Preserve this operational failure; do not run A13c-3 until repaired or lead-reviewed.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR/"failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls": 0}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
