#!/usr/bin/env python3
"""Case9 seed2 H10-return/ramp sensitivity diagnostic for safe-shortening v1.

Development-only deterministic replay on already-opened vehicle safe-shortening
v1 devval shard13/case9.  Prior diagnostics established:

* the existing adaptive seed2 run differs from matched H25 only by one H10
  decision at step57 and then fails;
* forcing H25 at step57 rescues exactly to the matched-H25 trajectory;
* injecting a single H10 at step57 into the H25 path reproduces the adaptive
  failure exactly;
* one-step H15/H20 at step57 are safe, and continuing H10 from step57 onward is
  safe.

This script tests whether an intermediate ramp after the H10 decision mitigates
that specific unsafe H10->H25 transition.  It runs three new replays from the
matched-H25 arm/path with fixed horizon schedules:

1. H10 at step57, H20 at step58, then H25.
2. H10 at step57, H15 at step58, H20 at step59, then H25.
3. H10 at steps57-58, then H25.

No training, no controller/checkpoint mutation, no new validation split, no
historical validation64-bank open, and no sealed-test access.  The evidence is
contaminated development diagnostics only and is meant to inform a versioned
IMPROVED v2 amendment.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import vehicle_safe_shortening_v1_case9_counterfactual_v1 as cf1  # noqa:E402

runner = cf1.runner
ROOT = cf1.ROOT
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT_REL = "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1.py"
MARKER = "vehicle-safe-shortening-v1-case9-ramp-sensitivity-v1-20260928"
CASE_ID = 9
SEED = 2
TARGET_STEP = 57
BASE_H = 25
SERVICE_START = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MODES = (
    ("H10_step57_H20_step58_then_H25", {57: 10, 58: 20}),
    ("H10_step57_H15_step58_H20_step59_then_H25", {57: 10, 58: 15, 59: 20}),
    ("H10_steps57_58_then_H25", {57: 10, 58: 10}),
)


def rel(path: Path) -> str:
    return cf1.rel(path)


def read_json(path: Path) -> Any:
    return cf1.read_json(path)


def write_json(path: Path, value: Any) -> None:
    return cf1.write_json(path, value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not data:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    n = len(data)
    def pct(p: float) -> float:
        if n == 1:
            return data[0]
        pos = p * (n - 1); lo = int(math.floor(pos)); hi = int(math.ceil(pos)); frac = pos - lo
        return data[lo] * (1.0 - frac) + data[hi] * frac
    return {"count": n, "sum": float(math.fsum(data)), "mean": float(math.fsum(data) / n), "median": float(pct(0.5)), "p95": float(pct(0.95)), "min": float(data[0]), "max": float(data[-1])}


def make_decider(original_decide, mode: str, schedule: Mapping[int, int]):
    def decide(arm: Mapping[str, Any], env: Any, previous_initial: bool, previous_final: bool):
        h, decision, ctx, selection_s = original_decide(arm, env, previous_initial, previous_final)
        elapsed = int(ctx.get("elapsed", -1))
        arm_id = str(arm.get("arm_id"))
        if arm_id != "matched_terminal_fixed_H25_vehicle_s2" or elapsed not in schedule:
            return h, decision, ctx, selection_s
        forced_h = int(schedule[elapsed])
        new_decision = dict(decision)
        new_decision.update({
            "kind": "fixed_grid_counterfactual_ramp_sensitivity",
            "ablation": mode,
            "ablation_original_selected_horizon": int(h),
            "ablation_forced_horizon": forced_h,
            "raw_horizon": forced_h,
            "selected_horizon": forced_h,
            "fallback": False,
            "clamped_to_base": False,
            "transition_schedule": {str(k): int(v) for k, v in sorted(schedule.items())},
            "target_step": TARGET_STEP,
        })
        return forced_h, new_decision, ctx, selection_s
    return decide


def run_mode(out_dir: Path, mode: str, schedule: Mapping[int, int], row: Mapping[str, Any], arm: Mapping[str, Any], case: Mapping[str, Any], terminal: Tuple[Any, Any], terminal_load_s: float):
    old_decide = runner.decide_for_arm
    runner.decide_for_arm = make_decider(old_decide, mode, schedule)
    try:
        summary = runner.run_episode(out_dir, row, arm, case, terminal, terminal_load_s)
    finally:
        runner.decide_for_arm = old_decide
    ep_dir = out_dir / "episodes" / ("exec%04d_case%02d_%s" % (int(row["execution_index"]), int(row["case_index"]), runner.safe_name(str(arm["arm_id"]))))
    trace = cf1.load_trace(ep_dir / "trace.json")
    return summary, trace, ep_dir


def brief(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return cf1.episode_brief(summary)


def first_divergence(trace: Any, ref: Any, eps: float = 1e-9) -> Any:
    for i, (a, b) in enumerate(zip(trace, ref)):
        d = cf1.state_distance(a, b)
        if d is not None and d > eps:
            return i
    return None


def outcome_label(summary: Mapping[str, Any], h25_summary: Mapping[str, Any]) -> str:
    b = brief(summary); h25 = brief(h25_summary)
    if b["success"] and b["steps"] == h25["steps"] and abs(b["physical_constraint_cost"] - h25["physical_constraint_cost"]) < 1e-9:
        return "identical_to_H25_success"
    if b["success"]:
        return "success_changed_cost_or_steps"
    return "failure"


def write_summary(raw: Mapping[str, Any], out_dir: Path) -> None:
    lines = []
    lines.append("# Vehicle safe-shortening v1 case9 seed2 H10-return/ramp sensitivity v1")
    lines.append("")
    lines.append("Created UTC: `%s`." % raw["created_utc"])
    lines.append("")
    lines.append("Development diagnostic only on already-opened fresh devval shard13/case9; no training and no sealed-test access.")
    lines.append("")
    lines.append("## Budget/access")
    lines.append("")
    for k, v in raw["budget_actual"].items():
        lines.append("- %s: `%s`." % (k, v))
    lines.append("")
    lines.append("## Reference outcomes")
    lines.append("")
    for name, ref in raw["existing_reference"].items():
        if name.endswith("_brief"):
            lines.append("- %s: `%s`." % (name, ref))
    lines.append("")
    lines.append("## New transition/ramp counterfactuals")
    lines.append("")
    lines.append("| replay | forced schedule | success | termination | steps | phys+constraint | total | horizons | label | first divergence vs H25 |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|---|---:|")
    for mode, _ in MODES:
        item = raw["counterfactuals"][mode]
        b = item["brief"]
        lines.append("| `%s` | `%s` | %s | %s | %d | %.6g | %.6g | %s | %s | %s |" % (
            mode, item["forced_schedule"], b["success"], b["termination"], b["steps"], b["physical_constraint_cost"], b["total_cost"], b["horizon_counts"], item["outcome_label"], item["first_divergence_step_vs_H25"]
        ))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(raw["diagnostic_conclusion"])
    lines.append("")
    lines.append("## Next action")
    lines.append("")
    lines.append(raw["next_action"])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backup_request(raw: Mapping[str, Any], out_dir: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_RAMP_SENSITIVITY_V1_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup bounded case9 seed2 H10-return/ramp diagnostic before IMPROVED v2 amendment",
        "artifacts": [rel(out_dir / "summary.md"), rel(out_dir / "raw.json"), rel(out_dir / "completed.json"), rel(out_dir / "episodes"), SCRIPT_REL, rel(runner.PROTOCOL), rel(runner.GATE_PATH), rel(runner.BANK_PATH)],
        "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"],
        "new_control_steps": raw["budget_actual"]["new_control_steps"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    })
    return path


def append_docs(raw: Mapping[str, Any], out_dir: Path, backup: Path) -> None:
    text = (
        "\n<!-- %s -->\n"
        "## 2026-09-28 vehicle safe-shortening v1 case9 seed2 H10-return/ramp sensitivity v1\n\n"
        "UTC: %s. Ran three deterministic transition/ramp counterfactual episodes on already-opened fresh devval shard13/case9 seed2. Budget: %d episodes, %d control steps, 0 training/gradient steps; sealed test closed. Conclusion: %s Artifacts: `%s`, `%s`, `%s`; backup request `%s`.\n"
        % (MARKER, raw["created_utc"], raw["budget_actual"]["new_rollout_episodes"], raw["budget_actual"]["new_control_steps"], raw["diagnostic_conclusion"], rel(out_dir / "summary.md"), rel(out_dir / "raw.json"), rel(out_dir / "completed.json"), rel(backup))
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")
    decision = (
        "\n<!-- %s-decision -->\n"
        "### Decision input: case9 H10-return/ramp sensitivity for IMPROVED v2\n\n"
        "Before evidence: one-step H10 at step57 caused the case9 seed2 failure; one-step H15/H20 and continued H10 were safe on the same case.\n\n"
        "New diagnostic: transition/ramp schedules after H10 on the same already-opened case. No training, no controller/terminal mutation, no sealed test.\n\n"
        "Outcome: `%s`. Use only as development evidence for v2 transition-safety design.\n" % (MARKER, raw["diagnostic_conclusion"])
    )
    path = ROOT / "DECISIONS.md"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if MARKER + "-decision" not in old:
            path.write_text(old.rstrip() + "\n" + decision, encoding="utf-8")


def append_registry(raw: Mapping[str, Any], out_dir: Path) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    if not path.exists() or MARKER in path.read_text(encoding="utf-8"):
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f); header = next(reader)
    row = {h: "" for h in header}; lower = {h.lower(): h for h in header}
    def put(names, value):
        for n in names:
            if n in lower:
                row[lower[n]] = str(value); return
    put(("timestamp", "created_utc", "utc", "time"), raw["created_utc"])
    put(("method",), raw["method"])
    put(("script", "source", "script_path"), SCRIPT_REL)
    put(("split",), raw["split"])
    put(("status", "result", "outcome"), "completed_development_ramp_sensitivity")
    put(("episodes", "new_rollout_episodes"), raw["budget_actual"]["new_rollout_episodes"])
    put(("control_steps", "new_control_steps"), raw["budget_actual"]["new_control_steps"])
    put(("training_episodes", "new_training_episodes"), 0)
    put(("gradient_steps", "new_gradient_steps"), 0)
    put(("sealed_test", "sealed_test_accessed"), False)
    put(("artifacts", "artifact", "summary"), rel(out_dir / "summary.md"))
    if all(v == "" for v in row.values()):
        row[header[-1]] = MARKER + " " + rel(out_dir / "summary.md")
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1_%s" % now.strftime("%Y%m%dT%H%M%SZ"))
    if out_dir.exists():
        raise RuntimeError("Refusing to overwrite existing output: %s" % rel(out_dir))
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {"started_utc": now.isoformat(), "pid": os.getpid(), "method": "IMPROVED_vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1", "sealed_test_accessed": False})
    runtime = runner.verify_runtime()
    if not runtime.get("passed"):
        raise RuntimeError(runtime.get("diagnosis") + ": " + runtime.get("exception", ""))
    runner.v1.latency_verify()
    gate = runner.freeze_gate_if_needed()
    bank = read_json(runner.BANK_PATH)
    case = bank["cases"][CASE_ID]
    arms = {a["arm_id"]: a for a in gate["schedule"]["arms"]}
    fixed_arm = arms["matched_terminal_fixed_H25_vehicle_s2"]
    terminal_source = fixed_arm["terminal_source"]
    load_start = time.perf_counter()
    terminal, terminal_receipt = runner.load_terminal(terminal_source)
    terminal_load_s = time.perf_counter() - load_start

    _, adaptive_summary_path, adaptive_trace, adaptive_summary = cf1.find_existing_episode("shard13/episodes/exec*_case09_safe_shortening_v1_vehicle_s2")
    _, h25_summary_path, h25_trace, h25_summary = cf1.find_existing_episode("shard13/episodes/exec*_case09_matched_terminal_fixed_H25_vehicle_s2")
    _, h10_summary_path, h10_trace, h10_summary = cf1.find_existing_episode("shard13/episodes/exec*_case09_matched_terminal_fixed_H10_vehicle_s2")
    short_steps = [i for i, r in enumerate(adaptive_trace) if int(r.get("horizon", BASE_H)) < BASE_H]
    if short_steps != [TARGET_STEP]:
        raise RuntimeError("Expected adaptive short step [%d], got %s" % (TARGET_STEP, short_steps))

    counterfactuals: Dict[str, Any] = {}
    traces: Dict[str, Any] = {}
    total_steps = 0
    for j, (mode, schedule) in enumerate(MODES):
        row = {"case_index": CASE_ID, "case_block_index": 13, "arm_index": -1, "local_arm_order": -1, "terminal_source": terminal_source, "execution_index": 9300 + j, "arm_id": fixed_arm["arm_id"], "family": fixed_arm["family"], "seed": SEED, "controller_h": mode}
        summary, trace, ep_dir = run_mode(out_dir, mode, schedule, row, fixed_arm, case, terminal, terminal_load_s)
        total_steps += int(summary.get("steps", 0))
        b = brief(summary)
        counterfactuals[mode] = {
            "forced_schedule": {str(k): int(v) for k, v in sorted(schedule.items())},
            "summary_path": rel(ep_dir / "summary.json"),
            "trace_path": rel(ep_dir / "trace.json"),
            "brief": b,
            "outcome_label": outcome_label(summary, h25_summary),
            "first_divergence_step_vs_H25": first_divergence(trace, h25_trace),
            "trace_compare_vs_H25": cf1.compare_trace(trace, h25_trace, mode + "_vs_H25"),
            "trace_compare_vs_existing_adaptive": cf1.compare_trace(trace, adaptive_trace, mode + "_vs_adaptive"),
        }
        traces[mode] = trace

    labels = {m: counterfactuals[m]["outcome_label"] for m, _ in MODES}
    if all(labels[m] != "failure" for m, _ in MODES):
        conclusion = "All tested H10-return schedules succeeded on this case, including H10->H20->H25, H10->H15->H20->H25, and two H10 steps then H25. The earlier one-step H10->H25 catastrophe is a narrow transition/path-dependence failure rather than a generic H10 or generic return-to-H25 failure; v2 should include transition-risk checks or smoothing but need not ban all short horizons."
    elif labels["H10_step57_H20_step58_then_H25"] != "failure" or labels["H10_step57_H15_step58_H20_step59_then_H25"] != "failure":
        conclusion = "At least one intermediate ramp after H10 succeeded while another H10-return schedule failed; this supports adding conservative horizon-transition smoothing/risk logic in v2 instead of one-step jumps from H10 to H25."
    else:
        conclusion = "The H10-return/ramp schedules still failed on this case; v2 should avoid H10 in comparable states unless a stronger continuation/risk model justifies it."

    created = dt.datetime.now(dt.timezone.utc)
    elapsed = (created - SERVICE_START).total_seconds()
    raw: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1_development_only",
        "split": "already_opened_fresh_devval_shard13_case9_development_diagnostic_only",
        "formal_scientific_evidence": False,
        "access_control": {"sealed_test_accessed": False, "historical_validation64_bank_opened": False, "final_test_authorization_requested": False},
        "budget_actual": {"new_rollout_episodes": len(MODES), "new_control_steps": total_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_episodes": 0, "sealed_test_control_steps": 0, "already_opened_existing_reference_traces_loaded": 3},
        "runtime_preflight": runtime,
        "service_elapsed_at_diagnostic": {"seconds_since_2026-09-26T10:55:29.419331Z": elapsed, "hours": elapsed / 3600.0, "research_sqlite_total_tokens": "unknown; repository tools cannot access supervisor sqlite"},
        "case": {"case_id": CASE_ID, "seed": SEED, "target_step": TARGET_STEP, "base_h": BASE_H, "bank": {"path": rel(runner.BANK_PATH), "sha256": sha256(runner.BANK_PATH)}},
        "protocol": {"path": rel(runner.PROTOCOL), "sha256": sha256(runner.PROTOCOL)},
        "gate": {"path": rel(runner.GATE_PATH), "sha256": sha256(runner.GATE_PATH)},
        "terminal_source": terminal_receipt,
        "source_hashes": {SCRIPT_REL: sha256(ROOT / SCRIPT_REL), "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case9_counterfactual_v1.py": sha256(ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case9_counterfactual_v1.py"), "experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py": sha256(ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py")},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "existing_reference": {"adaptive_summary_path": rel(adaptive_summary_path), "adaptive_brief": brief(adaptive_summary), "matched_H25_summary_path": rel(h25_summary_path), "matched_H25_brief": brief(h25_summary), "matched_H10_summary_path": rel(h10_summary_path), "matched_H10_brief": brief(h10_summary), "adaptive_short_steps": short_steps},
        "counterfactuals": counterfactuals,
        "diagnostic_conclusion": conclusion,
        "next_action": "Freeze IMPROVED vehicle adaptive-horizon v2 amendment using case9 transition diagnostics and v1 guard-collapse evidence; implement a conservative transition-aware policy/training smoke before fresh multi-seed validation. Keep sealed test closed.",
    }
    write_json(out_dir / "raw.json", raw)
    write_summary(raw, out_dir)
    backup = backup_request(raw, out_dir)
    append_docs(raw, out_dir, backup)
    append_registry(raw, out_dir)
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"] + [backup, ROOT / SCRIPT_REL, runner.PROTOCOL, runner.GATE_PATH, runner.BANK_PATH]
    write_json(out_dir / "completed.json", {"passed": True, "method": raw["method"], "formal_scientific_evidence": False, "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"], "new_control_steps": raw["budget_actual"]["new_control_steps"], "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "summary": rel(out_dir / "summary.md"), "raw": rel(out_dir / "raw.json"), "backup_request": rel(backup), "headline": {"counterfactuals": {m: counterfactuals[m]["brief"] for m, _ in MODES}, "diagnostic_conclusion": conclusion}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(out_dir / "completed.json"), "summary": rel(out_dir / "summary.md"), "raw": rel(out_dir / "raw.json"), "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"], "new_control_steps": raw["budget_actual"]["new_control_steps"], "sealed_test_accessed": False, "counterfactuals": {m: counterfactuals[m]["brief"] for m, _ in MODES}, "diagnostic_conclusion": conclusion}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        fail_dir = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1_failure_%s" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        fail_dir.mkdir(parents=True, exist_ok=True)
        write_json(fail_dir / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "method": "IMPROVED_vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1", "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "next_recovery_hint": "Preserve this failed diagnostic; inspect only if needed and version a one-variable fix. Do not open sealed test."})
        raise
