#!/usr/bin/env python3
"""Vehicle safe-shortening v1 case9 H10 hold-length sensitivity diagnostic.

Development-only one-variable follow-up on already-opened fresh devval
shard13/case9, seed2.  Earlier deterministic diagnostics established:

* the frozen adaptive trace fails when it uses H10 at step57 and immediately
  returns to H25;
* forcing H25 at that step exactly recovers the matched-H25 success;
* one-step H15/H20 and H10-from-step57-onward are safe on this case;
* short ramps that include a single H10 then return to H25 still fail.

This script tests whether holding H10 for a bounded number of steps after the
unsafe step repairs the transition before returning to H25.  It does not train,
mutate checkpoints/controllers, open the historical validation64 bank, or touch
sealed/final test data.  It is diagnostic development evidence only.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

import vehicle_safe_shortening_v1_smoke as v1  # noqa:E402

ROOT = v1.ROOT
TASK = "vehicle"
SEED = 2
CASE_ID = 9
TARGET_STEP = 57
BASE_H = 25
SHORT_H = 10
MAX_STEPS = 150
HOLD_LENGTHS = (3, 5, 10, 15)
DEVVAL_ROOT = ROOT / "research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1"
BANK = DEVVAL_ROOT / "bank/vehicle_safe_shortening_v1_devval64_bank.json"
BANK_COMPLETED = DEVVAL_ROOT / "bank/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_safe_shortening_v1_protocol_20260927.md"
SCRIPT = Path(__file__).resolve()
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_" + STAMP)
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_H10_HOLD_SENSITIVITY_V1_" + STAMP + ".json")
MARKER = "vehicle-safe-shortening-v1-case9-h10-hold-sensitivity-v1-" + STAMP


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return rel(value)
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=serial) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(xs.size), "sum": float(xs.sum()), "mean": float(xs.mean()), "median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)), "max": float(xs.max())}


def verify_bank() -> Dict[str, Any]:
    done = read_json(BANK_COMPLETED)
    if done.get("passed") is not True:
        raise RuntimeError("devval bank completed marker did not pass")
    expected = done.get("hashes", {}).get(rel(BANK)) or done.get("hashes", {}).get(str(BANK))
    if expected is not None and sha256(BANK) != expected:
        raise RuntimeError("devval bank hash mismatch")
    bank = read_json(BANK)
    if bank.get("task") != TASK or len(bank.get("cases", [])) <= CASE_ID:
        raise RuntimeError("unexpected devval bank metadata")
    return bank


def runtime_preflight() -> Dict[str, Any]:
    try:
        import tensorflow as tf  # noqa:F401
    except BaseException as exc:
        return {"passed": False, "exception": repr(exc), "diagnosis": "legacy Python/TF1 interpreter required", "python": sys.version, "executable": sys.executable}
    return {"passed": True, "tensorflow_version": getattr(tf, "__version__", "unknown"), "python": sys.version, "executable": sys.executable}


def stable_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(dict(row))
    for key in ("timing", "recovery", "decision", "policy_context"):
        d.pop(key, None)
    return d


def first_divergence(trace: List[Mapping[str, Any]], reference: Optional[List[Mapping[str, Any]]]) -> Optional[int]:
    if reference is None:
        return None
    limit = min(len(trace), len(reference))
    for i in range(limit):
        if stable_row(trace[i]) != stable_row(reference[i]):
            return i
    if len(trace) != len(reference):
        return limit
    return None


def reference_trace_and_summary() -> Dict[str, Any]:
    """Load already-opened shard13 matched-H25 and adaptive summaries if present."""
    out: Dict[str, Any] = {"matched_H25_trace": None, "matched_H25_summary": None, "adaptive_summary": None}
    shard13 = DEVVAL_ROOT / "shard13/episodes"
    h25 = sorted(shard13.glob("*case09_matched_terminal_fixed_H25_vehicle_s2"))
    adaptive = sorted(shard13.glob("*case09_safe_shortening_v1_vehicle_s2"))
    if h25:
        ep = h25[0]
        out["matched_H25_trace"] = read_json(ep / "trace.json")
        out["matched_H25_summary"] = read_json(ep / "summary.json")
        out["matched_H25_path"] = rel(ep)
    if adaptive:
        ep = adaptive[0]
        out["adaptive_summary"] = read_json(ep / "summary.json")
        out["adaptive_path"] = rel(ep)
    return out


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                      terminal_load_s: float, episode_wall_s: float, mode: str, hold_length: int,
                      forced_schedule: Mapping[int, int], reference: Optional[List[Mapping[str, Any]]]) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"]["selection_s"]) for r in trace]
    solver_times: List[float] = []
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    clamps = 0
    fallbacks = 0
    for row in trace:
        h = str(row["horizon"])
        horizons[h] = horizons.get(h, 0) + 1
        decision = row.get("decision", {})
        raw = decision.get("raw_horizon", row["horizon"])
        raw_horizons[str(raw)] = raw_horizons.get(str(raw), 0) + 1
        clamps += int(bool(decision.get("clamped_to_base", False)))
        fallbacks += int(bool(decision.get("fallback", False)))
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    out = dict(metric)
    out.update({
        "mode": mode,
        "case": CASE_ID,
        "seed": SEED,
        "hold_length": int(hold_length),
        "target_step": TARGET_STEP,
        "forced_schedule": {str(k): int(v) for k, v in sorted(forced_schedule.items())},
        "episode_failure": not bool(metric.get("success")),
        "construction_s": float(construction_s),
        "terminal_load_s_reference": float(terminal_load_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision_times),
        "decision_gross_timing_s": values_summary(decision_gross),
        "controller_timing_s_logging_deducted": values_summary(controller_times),
        "selection_timing_s": values_summary(selection_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "clamped_steps": int(clamps),
        "solver_failure_fallback_steps": int(fallbacks),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "first_divergence_vs_existing_matched_H25": first_divergence(trace, reference),
    })
    return out


def run_forced_episode(index: int, mode: str, hold_length: int, case: Mapping[str, Any], terminal: Tuple[Any, Any],
                       terminal_load_s: float, reference: Optional[List[Mapping[str, Any]]]) -> Dict[str, Any]:
    forced = {t: SHORT_H for t in range(TARGET_STEP, TARGET_STEP + int(hold_length))}
    ep_dir = OUT_DIR / "episodes" / ("exec%04d_case%02d_%s" % (9400 + index, CASE_ID, mode))
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = v1.make_env(TASK, SEED, aligned=True, scaled_obs=True)
    counts = v1.meter(env, ep_dir)
    env.set_value_function_weights_and_biases(*terminal)
    construction_s = time.perf_counter() - construct_start
    controller = env.control_system.controller
    original = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    with v1.LoggingTimer(ep_dir) as logging:
        recovery = v1.recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            if not (0.0 <= logged < gross):
                raise RuntimeError("invalid logging timing bounds")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=CASE_ID, step=-1)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_gross_s = time.perf_counter() - reset_start
        if obs is None or len(measured) != 1:
            raise RuntimeError("unexpected reset/controller warmup state")
        reset_record = {"reset_gross_s": float(reset_gross_s)}
        reset_record.update(measured.pop())
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        previous_initial = False
        previous_final = False
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                recovery["step"] = t
                ctx = v1.context(env, TASK)
                ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                selected_h = int(forced.get(t, BASE_H))
                select_start = time.perf_counter()
                decision = {
                    "kind": "case9_h10_hold_sensitivity_v1",
                    "mode": mode,
                    "raw_horizon": selected_h,
                    "selected_horizon": selected_h,
                    "base_horizon": BASE_H,
                    "short_horizon": SHORT_H,
                    "hold_length": int(hold_length),
                    "forced_short": bool(t in forced),
                    "fallback": False,
                    "clamped_to_base": False,
                }
                selection_s = time.perf_counter() - select_start
                _, terminated, row = v1.observed_step(env, TASK, selected_h, dict(case), t)
                if len(measured) != 1:
                    raise RuntimeError("expected exactly one measured controller call")
                timing = measured.pop()
                timing.update({"selection_s": float(selection_s), "decision_s": float(selection_s + timing["controller_s"]), "decision_gross_s": float(selection_s + timing["controller_gross_s"])})
                row.update({"policy_context": ctx, "decision": decision, "recovery": recovery["events"][-1], "timing": timing})
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                previous_initial = not bool(row["recovery"]["attempts"][0]["success"])
                previous_final = not bool(row["solver_success"])
                if terminated:
                    break
    if not trace or not trace[-1].get("termination"):
        raise RuntimeError("episode did not terminate within max steps")
    v1.audit_trace(TASK, dict(case), trace)
    write_json(ep_dir / "trace.json", trace)
    summary = summarize_episode(trace, reset_record, construction_s, terminal_load_s, time.perf_counter() - episode_start, mode, hold_length, forced, reference)
    summary.update({"path": rel(ep_dir), "steps_metered": counts["step_calls"], "resets_metered": counts["reset_calls"], "solver_counts": recovery["counts"], "logging_operations": logging.operations, "logging_total_s": float(logging.seconds)})
    write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


def brief(summary: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("success", "termination", "steps", "physical_constraint_cost", "total_cost", "horizon_counts", "raw_horizon_counts_before_clamp", "solver_failure_steps", "initial_failed_steps", "retries", "clamped_steps", "solver_failure_fallback_steps")
    return {k: summary.get(k) for k in keys}


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle safe-shortening v1 case9 seed2 H10 hold-length sensitivity v1",
        "",
        "Created UTC: `%s`." % raw["created_utc"],
        "",
        "Development diagnostic only on already-opened fresh devval shard13/case9; no training, no controller/checkpoint mutation, no historical validation64 reopen, and no sealed-test access.",
        "",
        "## Budget/access",
        "",
        "- new_rollout_episodes: `%d`." % raw["budget_actual"]["new_rollout_episodes"],
        "- new_control_steps: `%d`." % raw["budget_actual"]["new_control_steps"],
        "- new_training_episodes: `0`.",
        "- new_gradient_steps: `0`.",
        "- sealed_test_episodes: `0`.",
        "- sealed_test_control_steps: `0`.",
        "- already_opened_existing_reference_traces_loaded: `%d`." % raw["budget_actual"]["already_opened_existing_reference_traces_loaded"],
        "",
        "## Existing reference outcomes",
        "",
        "- matched_H25_brief: `%s`." % raw["reference_briefs"].get("matched_H25"),
        "- adaptive_brief: `%s`." % raw["reference_briefs"].get("adaptive"),
        "",
        "## New H10 hold counterfactuals",
        "",
        "| replay | forced H10 steps | success | termination | steps | phys+constraint | total | horizons | first divergence vs H25 |",
        "|---|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for s in raw["episode_summaries"]:
        forced_steps = "%d-%d" % (TARGET_STEP, TARGET_STEP + int(s["hold_length"]) - 1)
        lines.append("| `%s` | `%s` | %s | %s | %d | %.6g | %.6g | %s | %s |" % (
            s["mode"], forced_steps, bool(s.get("success")), s.get("termination"), int(s.get("steps", -1)),
            float(s.get("physical_constraint_cost", float("nan"))), float(s.get("total_cost", float("nan"))),
            s.get("horizon_counts"), s.get("first_divergence_vs_existing_matched_H25")))
    lines += [
        "",
        "## Interpretation",
        "",
        raw["diagnostic_conclusion"],
        "",
        "## Next action",
        "",
        raw["next_action"],
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(raw: Mapping[str, Any]) -> None:
    write_json(BACKUP_REQUEST, {
        "requested_utc": raw["created_utc"],
        "reason": "backup case9 H10 hold-length diagnostic artifacts before v2 amendment work",
        "artifacts": [rel(OUT_DIR), rel(SCRIPT), rel(PROTOCOL), rel(BANK), rel(BANK_COMPLETED)],
        "validation_accessed": True,
        "validation_access_note": "only already-opened fresh safe-shortening v1 devval shard13/case9 development diagnostic traces/case snapshot; no historical validation64 reopen",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "final_test_authorization_requested": False,
    })


def append_docs(raw: Mapping[str, Any]) -> None:
    token = "<!-- %s -->" % MARKER
    body = (
        token + "\n"
        "## 2026-09-28 vehicle safe-shortening v1 case9 H10 hold diagnostic\n\n"
        "Development-only case9 seed2 H10 hold-length diagnostic completed: %d new episodes, %d new control steps, no training and no sealed-test access. "
        "Result: %s Artifacts: `%s`, `%s`, `%s`.\n" % (
            raw["budget_actual"]["new_rollout_episodes"], raw["budget_actual"]["new_control_steps"], raw["diagnostic_conclusion"],
            rel(OUT_DIR / "summary.md"), rel(OUT_DIR / "raw.json"), rel(OUT_DIR / "completed.json"))
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if token not in old:
                path.write_text(old.rstrip() + "\n\n" + body, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "IMPROVED_vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_development_only", "sealed_test_accessed": False, "historical_validation64_bank_opened": False})
    preflight = runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise RuntimeError(preflight.get("diagnosis", "runtime preflight failed"))
    v1.latency_verify()
    bank = verify_bank()
    case = bank["cases"][CASE_ID]
    refs = reference_trace_and_summary()
    terminal_hashes: Dict[str, Any] = {}
    terminal_start = time.perf_counter()
    terminal = v1.load_terminal(SEED, terminal_hashes)
    terminal_load_s = time.perf_counter() - terminal_start
    episode_summaries: List[Dict[str, Any]] = []
    for i, hold in enumerate(HOLD_LENGTHS):
        mode = "H10_hold%d_steps_then_H25" % int(hold)
        summary = run_forced_episode(i, mode, int(hold), case, terminal, terminal_load_s, refs.get("matched_H25_trace"))
        episode_summaries.append(summary)
        write_json(OUT_DIR / "progress.json", {"episodes_done": len(episode_summaries), "episodes_expected": len(HOLD_LENGTHS), "control_steps_done": int(sum(int(s["steps"]) for s in episode_summaries)), "last_episode": brief(summary), "sealed_test_accessed": False})
        print(json.dumps({"mode": mode, "brief": brief(summary)}, sort_keys=True), flush=True)

    successes = [s for s in episode_summaries if bool(s.get("success"))]
    if successes:
        min_hold = min(int(s["hold_length"]) for s in successes)
        conclusion = "At least one finite H10 hold repaired the case9 transition; shortest successful hold among tested lengths is %d steps. This supports considering transition dwell/hysteresis in v2, but only as development evidence from one already-opened case." % min_hold
        next_action = "Freeze a conservative v2 amendment that treats H10 transitions as high-risk and either requires evidence-based dwell/hysteresis or avoids H10 in comparable states; then smoke on fresh non-test cases."
    else:
        conclusion = "All tested finite H10 holds (3, 5, 10, 15 steps) still failed after returning to H25 on case9. Together with prior H10-from-step57-onward success, the failure is a path-dependent return-to-H25 issue, not a generic H10 feasibility issue. v2 should avoid one-off H10 excursions unless a stronger continuation/risk model justifies the return."
        next_action = "Freeze a conservative transition-aware v2 amendment: no isolated H10 decisions; require robust state/continuation criteria before dispatching H10, and prioritize fresh smoke/validation over rerunning unchanged v1."

    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_development_only",
        "formal_scientific_evidence": False,
        "split": "already_opened_fresh_devval_shard13_case9_development_diagnostic_only_no_sealed_test",
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "final_test_authorization_requested": False,
        "source_hashes": {rel(SCRIPT): sha256(SCRIPT), rel(PROTOCOL): sha256(PROTOCOL), rel(BANK): sha256(BANK), rel(BANK_COMPLETED): sha256(BANK_COMPLETED)},
        "runtime_preflight": preflight,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "config": {"case_id": CASE_ID, "seed": SEED, "target_step": TARGET_STEP, "base_horizon": BASE_H, "short_horizon": SHORT_H, "hold_lengths": list(HOLD_LENGTHS)},
        "reference_paths": {k: v for k, v in refs.items() if k.endswith("path")},
        "reference_briefs": {"matched_H25": brief(refs["matched_H25_summary"]) if refs.get("matched_H25_summary") else None, "adaptive": brief(refs["adaptive_summary"]) if refs.get("adaptive_summary") else None},
        "terminal_hashes": terminal_hashes,
        "episode_summaries": episode_summaries,
        "budget_actual": {"new_rollout_episodes": len(episode_summaries), "new_control_steps": int(sum(int(s["steps"]) for s in episode_summaries)), "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_episodes": 0, "sealed_test_control_steps": 0, "already_opened_existing_reference_traces_loaded": int(refs.get("matched_H25_trace") is not None) + int(refs.get("matched_H25_summary") is not None) + int(refs.get("adaptive_summary") is not None)},
        "diagnostic_conclusion": conclusion,
        "next_action": next_action,
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    write_backup_request(raw)
    raw["backup_request"] = rel(BACKUP_REQUEST)
    write_json(OUT_DIR / "raw.json", raw)
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [BACKUP_REQUEST, SCRIPT, PROTOCOL, BANK, BANK_COMPLETED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "formal_scientific_evidence": False,
        "method": raw["method"],
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"],
        "new_control_steps": raw["budget_actual"]["new_control_steps"],
        "headline": {"diagnostic_conclusion": conclusion, "episode_briefs": {s["mode"]: brief(s) for s in episode_summaries}},
        "summary": rel(OUT_DIR / "summary.md"),
        "raw": rel(OUT_DIR / "raw.json"),
        "backup_request": rel(BACKUP_REQUEST),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
    })
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "raw": rel(OUT_DIR / "raw.json"), "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"], "new_control_steps": raw["budget_actual"]["new_control_steps"], "sealed_test_accessed": False, "diagnostic_conclusion": conclusion}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "next_recovery_hint": "Inspect failure; if runtime issue rerun with legacy interpreter; otherwise version a one-variable repair."})
        raise
