#!/usr/bin/env python3
"""Vehicle safe-shortening v1 engineering smoke, v3 runtime recovery harness.

This is a versioned source amendment for the v1/v2 smoke runner.  The first v1
execution reached one smoke episode and failed in summary accounting because a
local variable named ``decision`` shadowed a timing list.  The v2 recovery source
fixed only that summary bug, but the 2026-09-27T07:50Z execution was launched
with the modern interpreter and failed before simulations with
``ModuleNotFoundError: tensorflow``.  That is preserved as launch/runtime
evidence, not a controller result.

v3 writes to a fresh output directory, keeps the v2 summary-accounting patch,
records v1/v2 failure provenance, and is intended to be run with the legacy
Python/TF1 interpreter.  Controllers, policies, bank IDs, seeds, selection rules
and acceptance criteria remain the frozen IMPROVED safe-shortening v1 smoke
protocol.

It remains engineering smoke only: registered vehicle smoke bank, no validation64
bank content, and no sealed final-test access.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# The v1 module contains the frozen runner and the failed implementation.  Import
# it instead of overwriting it so the original failure remains auditable.
import vehicle_safe_shortening_v1_smoke as v1  # noqa:E402

ROOT = v1.ROOT
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3"
V1_FAILED_OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927"
V2_FAILED_OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v2"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_SMOKE_V3_20260927T000000Z.json"
MARKER = "vehicle-safe-shortening-v1-smoke-20260927-v3"

# Patch the v1 module globals used by its helper functions.
v1.OUT_DIR = OUT_DIR
v1.MARKER = MARKER


def fixed_summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                            terminal_load_s: float, episode_wall_s: float, case_id: int,
                            repeat: int, arm: Dict[str, Any]) -> Dict[str, Any]:
    """Patched copy of v1.summarize_episode with timing/meta variables separated."""
    metric = v1.case_metrics(v1.TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross_times = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"]["selection_s"]) for r in trace]
    solver_times: List[float] = []
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    clamps = 0
    fallbacks = 0
    for row in trace:
        horizons[str(row["horizon"])] = horizons.get(str(row["horizon"]), 0) + 1
        decision_meta = row.get("decision", {})
        raw = decision_meta.get("raw_horizon", row["horizon"])
        raw_horizons[str(raw)] = raw_horizons.get(str(raw), 0) + 1
        clamps += int(bool(decision_meta.get("clamped_to_base", False)))
        fallbacks += int(bool(decision_meta.get("fallback", False)))
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    out = dict(metric)
    out.update({
        "arm_id": arm["arm_id"],
        "family": arm["family"],
        "seed": arm["seed"],
        "case": int(case_id),
        "repeat": int(repeat),
        "episode_failure": not bool(metric.get("success")),
        "construction_s": float(construction_s),
        "terminal_load_s_amortized_reference": float(terminal_load_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": v1.values_summary(decision_times),
        "decision_gross_timing_s": v1.values_summary(decision_gross_times),
        "controller_timing_s_logging_deducted": v1.values_summary(controller_times),
        "selection_timing_s": v1.values_summary(selection_times),
        "solver_attempt_timing_s": v1.values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "clamped_steps": int(clamps),
        "solver_failure_fallback_steps": int(fallbacks),
    })
    return out


# Install the one-variable accounting patch.  v1.run_episode resolves this name
# from the v1 module global namespace, so assignment here changes only the called
# summary helper while preserving all other v1 behavior and source provenance.
v1.summarize_episode = fixed_summarize_episode


def _dir_provenance(path: Path, label: str) -> Dict[str, Any]:
    item: Dict[str, Any] = {"label": label, "path": v1.rel(path), "exists": path.exists()}
    if not path.exists():
        return item
    files = []
    for name in ("failure.json", "run_started.json", "schedule.json", "terminal_sources.json", "runtime_preflight.json"):
        p = path / name
        if p.exists():
            files.append({"path": v1.rel(p), "sha256": v1.sha256(p), "bytes": p.stat().st_size})
    ep_root = path / "episodes"
    item.update({
        "hashes": files,
        "episode_dirs_present": sorted(p.name for p in ep_root.iterdir()) if ep_root.exists() else [],
    })
    return item


def prior_failure_provenance() -> Dict[str, Any]:
    return {
        "v1_failed_smoke": _dir_provenance(
            V1_FAILED_OUT,
            "v1 failed after first episode trace in summarize_episode variable shadowing; no validation/test access",
        ),
        "v2_failed_launch": _dir_provenance(
            V2_FAILED_OUT,
            "v2 launched with modern interpreter and failed before simulations because TensorFlow/TF1 is unavailable there; no validation/test access",
        ),
        "v3_change": "fresh output directory plus explicit legacy-runtime preflight; no controller/policy/seed/bank/selection-rule change beyond the v2 summary-accounting repair",
    }


def write_backup_request(raw: Dict[str, Any]) -> None:
    v1.write_json(BACKUP_REQUEST, {
        "requested_utc": raw["created_utc"],
        "reason": "backup v1/v2 failed smoke evidence plus v3 patched smoke source/artifacts before fresh validation",
        "artifacts": [
            v1.rel(OUT_DIR),
            v1.rel(V1_FAILED_OUT),
            v1.rel(V2_FAILED_OUT),
            v1.rel(v1.PROTOCOL),
            v1.rel(Path(__file__).resolve()),
            v1.rel(Path(v1.__file__).resolve()),
        ],
        "validation_accessed": False,
        "test_accessed": False,
    })


def _tensorflow_preflight() -> Dict[str, Any]:
    """Fail early and informatively if the script is not run under legacy TF1."""
    try:
        import tensorflow as tf  # noqa:F401
    except BaseException as exc:
        return {
            "passed": False,
            "exception": repr(exc),
            "diagnosis": "TensorFlow is unavailable in this interpreter; rerun this smoke with run_experiment interpreter='legacy'.",
            "python": sys.version,
            "executable": sys.executable,
        }
    return {
        "passed": True,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "python": sys.version,
        "executable": sys.executable,
    }


def main() -> int:
    assert v1.TASK in v1.TASKS
    v1.assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    v1.write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "validation_accessed": False,
        "test_accessed": False,
        "method": "IMPROVED_safe_shortening_v1_smoke_v3_legacy_runtime_recovery",
        "source_amendment": "fresh output directory after v2 wrong-interpreter launch; keep v2 summarize_episode accounting repair; require legacy TF1 runtime",
        "prior_failures": prior_failure_provenance(),
    })
    runtime_preflight = _tensorflow_preflight()
    v1.write_json(OUT_DIR / "runtime_preflight.json", runtime_preflight)
    if not runtime_preflight.get("passed"):
        raise RuntimeError(runtime_preflight["diagnosis"] + " " + runtime_preflight.get("exception", ""))

    v1.latency_verify()
    assert v1.PROTOCOL.exists(), "v1 protocol file must be frozen before smoke"
    bank = v1.bank_name(v1.TASK, "smoke")
    banks_done = v1.read_json(bank.parent / "completed.json")
    assert v1.sha256(bank) == banks_done["hashes"][str(bank)]
    bank_data = v1.read_json(bank)
    assert bank_data["task"] == v1.TASK and bank_data["split"] == "smoke" and len(bank_data["cases"]) == v1.CASES_EXPECTED
    arms = v1.build_arms()
    schedule = v1.randomized_schedule(arms)
    v1.write_json(OUT_DIR / "schedule.json", {"order_seed": v1.ORDER_SEED, "episodes": schedule, "arms": arms})
    terminal_hashes: Dict[str, Any] = {}
    terminals: Dict[int, Tuple[Any, Any]] = {}
    terminal_load_s: Dict[int, float] = {}
    for seed in v1.SEEDS:
        start = time.perf_counter()
        terminals[seed] = v1.load_terminal(seed, terminal_hashes)
        terminal_load_s[seed] = time.perf_counter() - start
    v1.write_json(OUT_DIR / "terminal_sources.json", terminal_hashes)
    arm_by_id = {a["arm_id"]: a for a in arms}
    episodes: List[Dict[str, Any]] = []
    trace_index: Dict[Tuple[str, int, int], Path] = {}
    replay = {"passed": True, "pairs_checked": 0, "mismatches": []}
    for item in schedule:
        arm = arm_by_id[item["arm_id"]]
        summary = v1.run_episode(item, arm, bank_data["cases"][item["case"]], terminals[arm["seed"]], terminal_load_s[arm["seed"]])
        episodes.append(summary)
        trace_path = ROOT / summary["path"] / "trace.json"
        trace_index[(item["arm_id"], item["case"], item["repeat"])] = trace_path
        if item["repeat"] == 1:
            base = trace_index[(item["arm_id"], item["case"], 0)]
            replay["pairs_checked"] += 1
            if v1.clean_trace(v1.read_json(base)) != v1.clean_trace(v1.read_json(trace_path)):
                replay["passed"] = False
                replay["mismatches"].append({"arm_id": item["arm_id"], "case": item["case"], "repeat0": v1.rel(base), "repeat1": v1.rel(trace_path)})
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(e["steps"] for e in episodes)),
            "last_episode": {k: summary[k] for k in ("episode_index", "arm_id", "case", "repeat", "steps", "success", "termination")},
            "validation_accessed": False,
            "test_accessed": False,
        }
        v1.write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    assert replay["passed"], replay["mismatches"][:3]
    control_steps = int(sum(e["steps"] for e in episodes))
    assert len(episodes) == 24 and control_steps <= 3600
    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = v1.aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({"seed": arm["seed"], "family": arm["family"], "policy": arm["policy"], "policy_path": arm["policy_path"], "policy_sha256": arm["policy_sha256"]})
        aggregates[arm["arm_id"]] = agg
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_safe_shortening_v1_reused_gated_policies_not_original_SAC_v3_legacy_runtime_recovery",
        "source_amendment": {
            "v1_failure": "ValueError converting decision metadata key 'kind' to float because v1 shadowed timing list with per-step decision dict in summarize_episode",
            "v2_failure": "ModuleNotFoundError('tensorflow') because v2 was launched with the modern interpreter; no simulations occurred",
            "changed_variable_only": "fresh output directory and explicit legacy TensorFlow preflight; keep the v2 summarize_episode local-variable repair; no controller/policy/seed/bank/selection rule change",
            "prior_failures": prior_failure_provenance(),
        },
        "runtime_preflight": runtime_preflight,
        "formal_scientific_evidence": False,
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "split": "vehicle_smoke_bank_only",
        "protocol": {"path": v1.rel(v1.PROTOCOL), "sha256": v1.sha256(v1.PROTOCOL)},
        "bank": {"path": v1.rel(bank), "sha256": v1.sha256(bank), "cases": v1.CASES_EXPECTED},
        "source_hashes": {
            v1.rel(Path(__file__).resolve()): v1.sha256(Path(__file__).resolve()),
            v1.rel(Path(v1.__file__).resolve()): v1.sha256(Path(v1.__file__).resolve()),
            v1.rel(v1.REPRO / "gated_horizon_policy.py"): v1.sha256(v1.REPRO / "gated_horizon_policy.py"),
            v1.rel(v1.REPRO / "latency_tree_protocol.py"): v1.sha256(v1.REPRO / "latency_tree_protocol.py"),
            v1.rel(v1.REPRO / "conservative_canonical_reset.py"): v1.sha256(v1.REPRO / "conservative_canonical_reset.py"),
            v1.rel(v1.REPRO / "conservative_solver_recovery.py"): v1.sha256(v1.REPRO / "conservative_solver_recovery.py"),
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "budget_declared": {"episodes_exact": 24, "control_step_upper_bound": 3600, "new_gradient_steps": 0, "validation_episodes": 0, "test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "resets": int(sum(e["resets_metered"] for e in episodes)), "environment_constructions": len(episodes), "new_gradient_steps": 0, "validation_episodes": 0, "test_episodes": 0},
        "arms": arms,
        "randomized_schedule": schedule,
        "terminal_sources": terminal_hashes,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired_comparisons": v1.compare(aggregates),
        "replay": replay,
        "interpretation_limits": ["engineering smoke only", "not validation/model selection", "reuses unequal historical adaptive search budget", "H distribution is not timing evidence"],
    }
    v1.write_json(OUT_DIR / "raw.json", raw)
    v1.write_summary(raw)
    v1.append_docs(raw)
    write_backup_request(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [BACKUP_REQUEST, v1.PROTOCOL, Path(__file__).resolve(), Path(v1.__file__).resolve()]
    v1.write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hashes": {v1.rel(p): v1.sha256(p) for p in sorted(set(files))},
        "backup_request": v1.rel(BACKUP_REQUEST),
        "formal_scientific_evidence": False,
        "validation_accessed": False,
        "test_accessed": False,
        "source_amendment": raw["source_amendment"],
    })
    print(json.dumps({
        "completed": v1.rel(OUT_DIR / "completed.json"),
        "summary": v1.rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "replay_passed": replay["passed"],
        "validation_accessed": False,
        "test_accessed": False,
        "comparisons": raw["paired_comparisons"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        v1.write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "validation_accessed": False,
            "test_accessed": False,
            "prior_failures": prior_failure_provenance(),
            "next_recovery_hint": "If this failure is TensorFlow-unavailable, rerun a fresh versioned smoke source with run_experiment interpreter='legacy'; otherwise inspect failure and version the next one-variable amendment.",
        })
        raise
