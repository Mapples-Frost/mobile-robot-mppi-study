#!/usr/bin/env python3
"""Zero-solve preflight for the v34f backup-gate repair.

Active Opus task A requires verifying the repaired backup gate before any new
objective-vs-basin solver calls are spent.  This script imports the v34f solver
wrapper but does not call ``base.run`` or any controller/solver construction. It
only executes the approved gate checks, records the matched backup proof, writes
a backup request for the newly written wrapper/preflight evidence, and exits.

Budgets by construction: 0 lower-level solver calls, 0 plant steps,
0 env.reset/env.step calls, 0 training/refit, 0 validation64, 0 sealed test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0 as probe

base = probe.base
ROOT = probe.ROOT
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34f_backup_gate_preflight_v0"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34f_backup_gate_preflight.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34F_BACKUP_GATE_PREFLIGHT_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34f-backup-gate-preflight-{STAMP}"
MARKER = f"vehicle-v34f-backup-gate-preflight-{STAMP}"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    return base.rel(path)


def sha256(path: Path) -> str:
    return base.sha256(path)


def clean(value: Any) -> Any:
    return base.clean(value)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def read_json(path: Path) -> Any:
    return base.read_json(path)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def hash_existing(paths: Sequence[Path]) -> Mapping[str, str]:
    out = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# v34f backup-gate preflight (zero-solve)",
        "",
        f"Created UTC: {raw['created_utc']}",
        "",
        "This is an operational gate-preflight only. It did not construct a controller, call a solver, advance the plant, open validation64, or access sealed/final test data.",
        "",
        "## Headline",
        "",
        f"- hard_pass: {h['hard_pass']}",
        f"- matched backup proof: {h.get('matched_backup_proof')}",
        f"- matched commit: {h.get('matched_commit')}",
        f"- matched package sha256: {h.get('matched_package_sha256')}",
        f"- lower_level_solver_calls: {raw['budget_actual']['lower_level_solver_calls']}",
        f"- plant_steps: {raw['budget_actual']['plant_steps']}",
        f"- validation64_bank_opened: {raw['validation64_bank_opened']}",
        f"- sealed_test_accessed: {raw['sealed_test_accessed']}",
        "",
        "## Next gate",
        "",
        "External backup is required for this v34f wrapper/preflight source and outputs before any objective-vs-basin solver smoke or 24-call probe.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    marker = MARKER
    block = f"""
### {marker}

UTC: {raw['created_utc']}

Opus task A backup-gate preflight passed with zero solver/plant/training/validation/test budget. Matched verified backup proof `{h.get('matched_backup_proof')}` for commit `{h.get('matched_commit')}` and package `{h.get('matched_package_sha256')}`. This only verifies the repaired gate; it is not objective-vs-basin science. External backup covering v34f source/preflight outputs is required before the one-cell objective-reconstruction smoke.

Artifacts: `{rel(RUN_DIR / 'completed.json')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'summary.md')}`.
"""
    for p in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", RESPONSE_LOG]:
        append_if_missing(p, marker, block)


def make_backup_request(raw: Mapping[str, Any]) -> None:
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34f_backup_gate_preflight",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new v34f wrapper source, backup-gate preflight evidence, docs and registry must be externally recoverable before objective-vs-basin solver smoke/probe",
        "must_cover": [
            rel(ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0.py"),
            rel(Path(__file__).resolve()),
            rel(RUN_DIR),
            rel(STATE),
            rel(BACKUP_REQUEST),
            rel(RESPONSE_LOG),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
        ],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "after verified external backup, run Opus task B one-cell objective reconstruction smoke; do not spend remaining 23 calls unless G-B passes",
    })


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, default=0)
    ap.add_argument("--i-accept-v34f-backup-gate-preflight", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        created = now_utc()
        write_json(RUN_DIR / "run_started.json", {
            "started_utc": created.isoformat(),
            "pid": os.getpid(),
            "method": NAME,
            "budget_cap_solver_calls": 0,
            "plant_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        gates = probe.verify_opus_and_backup_gates(args)
        matched = gates["matched_verified_backup_proof"]
        raw = {
            "created_utc": now_utc().isoformat(),
            "started_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (now_utc() - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_zero_solve_backup_gate_preflight_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(probe.CURRENT_OPUS_REPORT),
            "lead_report_sha256": probe.CURRENT_OPUS_SHA,
            "hypothesis_frozen": "The v34d backup-gate failure was operational over-constraint; scanning verified backup proofs and matching current CLI args permits a fail-closed zero-solve gate without altering the Task2 scientific design.",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_declared": {
                "lower_level_solver_calls": 0,
                "plant_steps": 0,
                "env_reset_calls": 0,
                "new_training_or_gradient_steps": 0,
                "selector_refits": 0,
                "validation64_episodes": 0,
                "sealed_test_episodes": 0,
            },
            "budget_actual": {
                "lower_level_solver_calls": 0,
                "plant_steps": 0,
                "env_reset_calls": 0,
                "new_training_or_gradient_steps": 0,
                "selector_refits": 0,
                "validation64_episodes": 0,
                "sealed_test_episodes": 0,
            },
            "gates": gates,
            "headline": {
                "hard_pass": True,
                "matched_backup_proof": matched.get("path"),
                "matched_backup_proof_sha256": matched.get("sha256"),
                "matched_commit": matched.get("commit"),
                "matched_package_sha256": (matched.get("package") or {}).get("sha256"),
                "matched_package_bytes": (matched.get("package") or {}).get("bytes"),
                "lower_level_solver_calls": 0,
                "plant_steps": 0,
                "validation64_bank_opened": False,
                "sealed_test_accessed": False,
            },
            "input_hashes": hash_existing([
                Path(__file__).resolve(),
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0.py",
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0.py",
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py",
                probe.OPUS_PLAN_READY,
                probe.OPUS_LATEST,
                probe.CURRENT_OPUS_REPORT,
                probe.d.V0E_DONE,
            ]),
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "next_step_if_backed_up": "Run task B one-cell objective reconstruction smoke only; halt scientific attribution if J reconstruction or residual gate fails.",
            "backup_request_after_run": rel(BACKUP_REQUEST),
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# Continue state after v34f backup-gate preflight\n\nUTC: {raw['created_utc']}\n\nHeadline: {raw['headline']}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'completed.json')}\n\nNext: obtain verified external backup for {rel(BACKUP_REQUEST)} and v34f source/preflight outputs; then run one-cell objective reconstruction smoke, not full 24-call probe unless G-B passes.\n",
            encoding="utf-8",
        )
        update_docs(raw)
        make_backup_request(raw)
        files = [
            RUN_DIR / "run_started.json",
            RUN_DIR / "raw.json",
            RUN_DIR / "summary.md",
            STATE,
            BACKUP_REQUEST,
            RESPONSE_LOG,
            ROOT / "STATUS.md",
            ROOT / "RESEARCH_LOG.md",
            ROOT / "DECISIONS.md",
            ROOT / "RESULTS_AUDIT.md",
            ROOT / "REPRODUCTION_PROTOCOL.md",
            ROOT / "EXPERIMENT_REGISTRY.csv",
            Path(__file__).resolve(),
            ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0.py",
        ]
        completed = {
            "status": "complete",
            "passed": True,
            "hard_pass": True,
            "created_utc": raw["created_utc"],
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": raw["budget_actual"],
            "headline": raw["headline"],
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_gate": raw["next_step_if_backed_up"],
            "hashes": hash_existing(files),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({
            "completed": rel(RUN_DIR / "completed.json"),
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "headline": raw["headline"],
            "backup_request": rel(BACKUP_REQUEST),
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {
            "status": "failed",
            "created_utc": now_utc().isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "classification": "development_IMPROVED_zero_solve_backup_gate_preflight_not_validation_not_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_caps": {"lower_level_solver_calls": 0, "plant_steps": 0, "env_reset_calls": 0},
        }
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# v34f backup-gate preflight failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR / 'failed.json')}\n\nNo solver/plant/training/validation/test budget was spent. Inspect the gate traceback/proof candidates; do not run objective-vs-basin solver smoke until this gate passes and is backed up.\n",
            encoding="utf-8",
        )
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
