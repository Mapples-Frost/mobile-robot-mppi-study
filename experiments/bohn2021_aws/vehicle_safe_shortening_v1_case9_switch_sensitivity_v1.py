#!/usr/bin/env python3
"""Case9 seed2 switch-sensitivity counterfactuals for vehicle safe-shortening v1.

This development diagnostic extends the completed case9 one-step H10 causal
counterfactual without reopening sealed test data or changing the frozen v1
controller/checkpoints.  The prior diagnostic established that, on already-opened
fresh devval shard13/case9 seed2, a single H10 decision at step57 on the H25
prefix reproduces the adaptive failure, while forcing H25 at that decision
rescues the episode.

New question: is the catastrophe specific to the very short H10 action, to the
return from H10 back to H25/warm-started H25 control, or to horizon switching in
general?  We run three bounded one-variable replays from the same H25 prefix:

1. one_step_H15_at_step57_then_H25
2. one_step_H20_at_step57_then_H25
3. switch_to_H10_from_step57_onward

No training, no source/controller mutation, no new validation split, and no
sealed-test access.  This remains contaminated development diagnostics only.
"""
from __future__ import annotations

import csv
import datetime as dt
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
DEVVAL = cf1.DEVVAL
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT_REL = "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case9_switch_sensitivity_v1.py"
MARKER = "vehicle-safe-shortening-v1-case9-switch-sensitivity-v1-20260928"
CASE_ID = 9
SEED = 2
TARGET_STEP = 57
BASE_H = 25
SERVICE_START = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MODES = (
    ("one_step_H15_at_step57_then_H25", 15, "one_step"),
    ("one_step_H20_at_step57_then_H25", 20, "one_step"),
    ("switch_to_H10_from_step57_onward", 10, "from_step_onward"),
)


def rel(path: Path) -> str:
    return cf1.rel(path)


def read_json(path: Path) -> Any:
    return cf1.read_json(path)


def write_json(path: Path, value: Any) -> None:
    return cf1.write_json(path, value)


def sha256(path: Path) -> str:
    return cf1.sha256(path)


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    return cf1.values_summary(values)


def make_decider(original_decide, mode: str, forced_h: int, kind: str):
    def decide(arm: Mapping[str, Any], env: Any, previous_initial: bool, previous_final: bool):
        h, decision, ctx, selection_s = original_decide(arm, env, previous_initial, previous_final)
        elapsed = int(ctx.get("elapsed", -1))
        arm_id = str(arm.get("arm_id"))
        should_force = False
        if arm_id == "matched_terminal_fixed_H25_vehicle_s2":
            if kind == "one_step" and elapsed == TARGET_STEP:
                should_force = True
            elif kind == "from_step_onward" and elapsed >= TARGET_STEP:
                should_force = True
        if not should_force:
            return h, decision, ctx, selection_s
        new_decision = dict(decision)
        new_decision.update({
            "kind": "fixed_grid_counterfactual_switch_sensitivity",
            "ablation": mode,
            "ablation_original_selected_horizon": int(h),
            "ablation_forced_horizon": int(forced_h),
            "raw_horizon": int(forced_h),
            "selected_horizon": int(forced_h),
            "fallback": False,
            "clamped_to_base": False,
            "switch_rule": kind,
            "target_step": TARGET_STEP,
        })
        return int(forced_h), new_decision, ctx, selection_s
    return decide


def run_mode(out_dir: Path, mode: str, forced_h: int, kind: str, row: Mapping[str, Any], arm: Mapping[str, Any], case: Mapping[str, Any], terminal: Tuple[Any, Any], terminal_load_s: float):
    old_decide = runner.decide_for_arm
    runner.decide_for_arm = make_decider(old_decide, mode, forced_h, kind)
    try:
        summary = runner.run_episode(out_dir, row, arm, case, terminal, terminal_load_s)
    finally:
        runner.decide_for_arm = old_decide
    ep_dir = out_dir / "episodes" / ("exec%04d_case%02d_%s" % (int(row["execution_index"]), int(row["case_index"]), runner.safe_name(str(arm["arm_id"]))))
    trace = cf1.load_trace(ep_dir / "trace.json")
    return summary, trace


def brief(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return cf1.episode_brief(summary)


def outcome_label(summary: Mapping[str, Any], h25_summary: Mapping[str, Any]) -> str:
    b = brief(summary)
    h25 = brief(h25_summary)
    if b["success"] and b["steps"] == h25["steps"] and abs(b["physical_constraint_cost"] - h25["physical_constraint_cost"]) < 1e-9:
        return "identical_to_H25_success"
    if b["success"]:
        return "success_changed_cost_or_steps"
    return "failure"


def write_summary(raw: Mapping[str, Any], out_dir: Path) -> None:
    lines = []
    lines.append("# Vehicle safe-shortening v1 case9 seed2 switch-sensitivity v1")
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
    lines.append("## Existing references")
    lines.append("")
    for name, ref in raw["existing_reference"].items():
        if name.endswith("_brief"):
            lines.append("- %s: `%s`." % (name, ref))
    lines.append("")
    lines.append("## New switch-sensitivity counterfactuals")
    lines.append("")
    lines.append("| replay | success | termination | steps | phys+constraint | total | horizons | label |")
    lines.append("|---|---:|---|---:|---:|---:|---|---|")
    for mode, _, _ in MODES:
        b = raw["counterfactuals"][mode]["brief"]
        lines.append("| `%s` | %s | %s | %d | %.6g | %.6g | %s | %s |" % (
            mode, b["success"], b["termination"], b["steps"], b["physical_constraint_cost"], b["total_cost"], b["horizon_counts"], raw["counterfactuals"][mode]["outcome_label"]
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
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_SWITCH_SENSITIVITY_V1_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup bounded case9 seed2 horizon switch-sensitivity diagnostic before IMPROVED v2 amendment",
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
        "## 2026-09-28 vehicle safe-shortening v1 case9 seed2 switch-sensitivity v1\n\n"
        "UTC: %s. Ran three deterministic case9 seed2 switch-sensitivity counterfactual episodes on the already-opened fresh devval case: one-step H15, one-step H20, and H10 from step57 onward. Budget: %d episodes, %d control steps, 0 training/gradient steps; sealed test closed. Conclusion: %s Artifacts: `%s`, `%s`, `%s`; backup request `%s`.\n"
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
        "### Decision: case9 switch-sensitivity informs IMPROVED v2 horizon-change safety\n\n"
        "Before evidence: v1 full devval64 failed the gate; case9 one-step H10 at step57 was causally implicated by force/inject counterfactuals.\n\n"
        "New diagnostic: one-step H15, one-step H20, and H10-from-step57 onward replays on the same already-opened case. No training, no controller/terminal mutation, no sealed test.\n\n"
        "Outcome: `%s`. Use this as development-only evidence for v2 design; it cannot serve as fresh validation.\n" % (MARKER, rel(out_dir / "summary.md"))
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
    put(("status", "result", "outcome"), "completed_development_switch_sensitivity")
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
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_%s" % now.strftime("%Y%m%dT%H%M%SZ"))
    if out_dir.exists():
        raise RuntimeError("Refusing to overwrite existing output: %s" % rel(out_dir))
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {"started_utc": now.isoformat(), "pid": os.getpid(), "method": "IMPROVED_vehicle_safe_shortening_v1_case9_switch_sensitivity_v1", "sealed_test_accessed": False})
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
    for j, (mode, forced_h, kind) in enumerate(MODES):
        row = {"case_index": CASE_ID, "case_block_index": 13, "arm_index": -1, "local_arm_order": -1, "terminal_source": terminal_source, "execution_index": 9200 + j, "arm_id": fixed_arm["arm_id"], "family": fixed_arm["family"], "seed": SEED, "controller_h": mode}
        summary, trace = run_mode(out_dir, mode, forced_h, kind, row, fixed_arm, case, terminal, terminal_load_s)
        counterfactuals[mode] = {"forced_horizon": forced_h, "switch_rule": kind, "summary_path": rel(out_dir / "episodes" / ("exec%04d_case%02d_%s" % (9200 + j, CASE_ID, runner.safe_name(fixed_arm["arm_id"]))) / "summary.json"), "brief": brief(summary), "outcome_label": outcome_label(summary, h25_summary)}
        traces[mode] = trace

    labels = {m: counterfactuals[m]["outcome_label"] for m, _, _ in MODES}
    h15_safe = labels["one_step_H15_at_step57_then_H25"] != "failure"
    h20_safe = labels["one_step_H20_at_step57_then_H25"] != "failure"
    h10_cont_safe = labels["switch_to_H10_from_step57_onward"] != "failure"
    if h15_safe and h20_safe and h10_cont_safe:
        conclusion = "On this diagnosed case, H15/H20 one-step switches and continuing H10 from step57 all avoid the prior single-H10-then-H25 failure; the v1 catastrophe is most consistent with an unsafe abrupt return to H25 after an H10 step rather than horizon switching in general."
    elif h15_safe and h20_safe and not h10_cont_safe:
        conclusion = "On this diagnosed case, moderate one-step shortening to H15/H20 is safe while H10 remains unsafe even if continued; v2 should avoid aggressive H10 in comparable states and learn horizon magnitude/risk, not just short-vs-base."
    elif h20_safe and not h15_safe:
        conclusion = "On this diagnosed case, only mild H20 one-step shortening is safe; v2 needs conservative horizon-change limits and state-risk/value checks before short horizons."
    else:
        conclusion = "On this diagnosed case, additional switch-sensitivity replays still show failures for moderate/continued short horizons; v2 should treat comparable states as unsafe for shortening and diagnose solver/warm-start dynamics further."

    created = dt.datetime.now(dt.timezone.utc)
    elapsed = (created - SERVICE_START).total_seconds()
    total_steps = int(sum(counterfactuals[m]["brief"]["steps"] for m, _, _ in MODES))
    raw: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_development_only",
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
        "trace_comparisons": {"H15_one_step_vs_existing_H25_prefix": cf1.compare_trace(traces["one_step_H15_at_step57_then_H25"], h25_trace, "H15_one_step_vs_existing_H25"), "H20_one_step_vs_existing_H25_prefix": cf1.compare_trace(traces["one_step_H20_at_step57_then_H25"], h25_trace, "H20_one_step_vs_existing_H25"), "H10_from_step57_vs_existing_adaptive": cf1.compare_trace(traces["switch_to_H10_from_step57_onward"], adaptive_trace, "H10_from_step57_vs_existing_adaptive")},
        "diagnostic_conclusion": conclusion,
        "next_action": "Freeze an IMPROVED v2 amendment using these case9/case43 causal diagnostics: introduce explicit horizon-change/risk/value logic and a conservative smoke before any new multi-seed training/validation; keep sealed test closed.",
    }
    write_json(out_dir / "raw.json", raw)
    write_summary(raw, out_dir)
    backup = backup_request(raw, out_dir)
    append_docs(raw, out_dir, backup)
    append_registry(raw, out_dir)
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"] + [backup, ROOT / SCRIPT_REL, runner.PROTOCOL, runner.GATE_PATH, runner.BANK_PATH]
    write_json(out_dir / "completed.json", {"passed": True, "method": raw["method"], "formal_scientific_evidence": False, "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"], "new_control_steps": raw["budget_actual"]["new_control_steps"], "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "summary": rel(out_dir / "summary.md"), "raw": rel(out_dir / "raw.json"), "backup_request": rel(backup), "headline": {"counterfactuals": counterfactuals, "diagnostic_conclusion": conclusion}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(out_dir / "completed.json"), "summary": rel(out_dir / "summary.md"), "raw": rel(out_dir / "raw.json"), "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"], "new_control_steps": raw["budget_actual"]["new_control_steps"], "sealed_test_accessed": False, "counterfactuals": counterfactuals, "diagnostic_conclusion": conclusion}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        fail_dir = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_failure_%s" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        fail_dir.mkdir(parents=True, exist_ok=True)
        write_json(fail_dir / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "method": "IMPROVED_vehicle_safe_shortening_v1_case9_switch_sensitivity_v1", "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "next_recovery_hint": "Preserve this failed diagnostic; inspect schema/output conflict and rerun only a versioned fix. Do not open sealed test."})
        raise
