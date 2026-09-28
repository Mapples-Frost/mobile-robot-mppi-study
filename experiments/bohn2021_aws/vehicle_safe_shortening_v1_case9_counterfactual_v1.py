#!/usr/bin/env python3
"""Bounded deterministic counterfactual replay for safe-shortening v1 case9 seed2.

Context
-------
The completed fresh development-validation campaign for vehicle safe-shortening
v1 found one large seed2 failure on devval shard13/case9.  The trace/policy
metadata diagnostic showed the adaptive and matched-H25 trajectories were
identical before step 57, where the adaptive controller made its only H10
selection; the matched same-seed H25 comparator then succeeded in 77 steps while
adaptive failed at the 150-step cap.

This diagnostic performs two one-variable deterministic replays on the already
opened development-validation case9 only:

1. adaptive_force_H25_at_step57: use the same safe-shortening seed2 arm, but
   override the selected horizon to H25 only at the saved H10 step.
2. fixed_H25_inject_H10_at_step57: use the matched terminal fixed-H25 seed2 arm,
   but override the selected horizon to H10 only at that same step.

No controller code, terminal model, case data, solver settings, training, or
final-test content is changed.  The result is contaminated development/causal
engineering evidence only; it must not be used as fresh independent validation.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import vehicle_safe_shortening_v1_devval_shard_runner as runner  # noqa:E402

ROOT = runner.ROOT
DEVVAL = runner.OUT_ROOT
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT_REL = "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case9_counterfactual_v1.py"
MARKER = "vehicle-safe-shortening-v1-case9-counterfactual-v1-20260928"
CASE_ID = 9
SEED = 2
H10_STEP = 57
BASE_H = 25
SHORT_H = 10
SERVICE_START = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")


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
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = sorted(float(v) for v in values if math.isfinite(float(v)))
    n = len(data)
    if not n:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    def pct(p: float) -> float:
        if n == 1:
            return data[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos)); frac = pos - lo
        return data[lo] * (1.0 - frac) + data[hi] * frac
    return {"count": n, "sum": float(math.fsum(data)), "mean": float(math.fsum(data) / n), "median": float(pct(0.5)), "p95": float(pct(0.95)), "min": float(data[0]), "max": float(data[-1])}


def state_xytheta(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    st = row.get("state") or {}
    try:
        return float(st["x"]), float(st["y"]), float(st["theta"])
    except Exception:
        return None


def state_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> Optional[float]:
    aa = state_xytheta(a); bb = state_xytheta(b)
    if aa is None or bb is None:
        return None
    return math.hypot(aa[0] - bb[0], aa[1] - bb[1])


def heading_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> Optional[float]:
    aa = state_xytheta(a); bb = state_xytheta(b)
    if aa is None or bb is None:
        return None
    d = aa[2] - bb[2]
    return math.atan2(math.sin(d), math.cos(d))


def load_trace(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    out.append(json.loads(line))
        return out
    return read_json(path)


def find_existing_episode(pattern: str) -> Tuple[Path, Path, List[Dict[str, Any]], Dict[str, Any]]:
    summaries = sorted(DEVVAL.glob(pattern + "/summary.json"))
    if len(summaries) != 1:
        raise RuntimeError("Expected exactly one existing episode for %s, found %d" % (pattern, len(summaries)))
    summary_path = summaries[0]
    trace_path = summary_path.with_name("trace.json")
    if not trace_path.exists():
        trace_path = summary_path.with_name("trace.jsonl")
    return summary_path.parent, summary_path, load_trace(trace_path), read_json(summary_path)


def compare_trace(a: List[Mapping[str, Any]], b: List[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    n = min(len(a), len(b))
    dists = []
    headings = []
    reward_diffs = []
    perf_diffs = []
    constraint_diffs = []
    horizon_mismatches = []
    first_state_distance_gt_1e_9 = None
    for i in range(n):
        d = state_distance(a[i], b[i])
        if d is not None:
            dists.append(d)
            if first_state_distance_gt_1e_9 is None and d > 1e-9:
                first_state_distance_gt_1e_9 = i
        hd = heading_delta(a[i], b[i])
        if hd is not None:
            headings.append(abs(hd))
        for key, dest in (("reward", reward_diffs), ("performance", perf_diffs), ("constraint", constraint_diffs)):
            av = safe_float(a[i].get(key)); bv = safe_float(b[i].get(key))
            if av is not None and bv is not None:
                dest.append(abs(av - bv))
        if int(a[i].get("horizon", -999)) != int(b[i].get("horizon", -998)):
            horizon_mismatches.append({"step": i, "a_h": a[i].get("horizon"), "b_h": b[i].get("horizon")})
    return {
        "label": label,
        "len_a": len(a),
        "len_b": len(b),
        "common_prefix": n,
        "state_distance": values_summary(dists),
        "heading_abs_delta": values_summary(headings),
        "reward_abs_delta": values_summary(reward_diffs),
        "performance_abs_delta": values_summary(perf_diffs),
        "constraint_abs_delta": values_summary(constraint_diffs),
        "first_state_distance_gt_1e_9": first_state_distance_gt_1e_9,
        "horizon_mismatch_count_common_prefix": len(horizon_mismatches),
        "horizon_mismatch_first_20": horizon_mismatches[:20],
    }


def horizon_counts(summary: Mapping[str, Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in (summary.get("horizon_counts") or {}).items()}


def episode_brief(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "success": bool(summary.get("success")),
        "termination": summary.get("termination"),
        "steps": int(summary.get("steps", 0)),
        "physical_constraint_cost": float(summary.get("physical_constraint_cost", 0.0)),
        "total_cost": float(summary.get("total_cost", 0.0)),
        "horizon_counts": horizon_counts(summary),
        "raw_horizon_counts_before_clamp": {str(k): int(v) for k, v in (summary.get("raw_horizon_counts_before_clamp") or {}).items()},
        "solver_failure_steps": int(summary.get("solver_failure_steps", 0)),
        "initial_failed_steps": int(summary.get("initial_failed_steps", 0)),
        "retries": int(summary.get("retries", 0)),
        "clamped_steps": int(summary.get("clamped_steps", 0)),
        "solver_failure_fallback_steps": int(summary.get("solver_failure_fallback_steps", 0)),
        "decision_sum_s": float((summary.get("decision_timing_s") or {}).get("sum", 0.0)),
        "decision_mean_s": float((summary.get("decision_timing_s") or {}).get("mean", 0.0)),
        "decision_p95_s": float((summary.get("decision_timing_s") or {}).get("p95", 0.0)),
    }


def make_counterfactual_decider(original_decide, mode: str):
    def decide(arm: Mapping[str, Any], env: Any, previous_initial: bool, previous_final: bool):
        h, decision, ctx, selection_s = original_decide(arm, env, previous_initial, previous_final)
        elapsed = int(ctx.get("elapsed", -1))
        arm_id = str(arm.get("arm_id"))
        if mode == "adaptive_force_H25_at_step57" and arm_id == "safe_shortening_v1_vehicle_s2" and elapsed == H10_STEP:
            new_decision = dict(decision)
            new_decision.update({
                "ablation": mode,
                "ablation_original_selected_horizon": int(h),
                "ablation_original_raw_horizon": int(decision.get("raw_horizon", h)) if isinstance(decision, dict) else int(h),
                "ablation_forced_horizon": BASE_H,
                "selected_horizon": BASE_H,
                "clamped_to_base": False,
            })
            return BASE_H, new_decision, ctx, selection_s
        if mode == "fixed_H25_inject_H10_at_step57" and arm_id == "matched_terminal_fixed_H25_vehicle_s2" and elapsed == H10_STEP:
            new_decision = dict(decision)
            new_decision.update({
                "kind": "fixed_grid_counterfactual_injection",
                "ablation": mode,
                "ablation_original_selected_horizon": int(h),
                "ablation_forced_horizon": SHORT_H,
                "raw_horizon": SHORT_H,
                "selected_horizon": SHORT_H,
                "fallback": False,
                "clamped_to_base": False,
            })
            return SHORT_H, new_decision, ctx, selection_s
        return h, decision, ctx, selection_s
    return decide


def run_counterfactual(out_dir: Path, mode: str, row: Mapping[str, Any], arm: Mapping[str, Any], case: Mapping[str, Any], terminal: Tuple[Any, Any], terminal_load_s: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    old_decide = runner.decide_for_arm
    runner.decide_for_arm = make_counterfactual_decider(old_decide, mode)
    try:
        summary = runner.run_episode(out_dir, row, arm, case, terminal, terminal_load_s)
    finally:
        runner.decide_for_arm = old_decide
    ep_dir = out_dir / "episodes" / ("exec%04d_case%02d_%s" % (int(row["execution_index"]), int(row["case_index"]), runner.safe_name(str(arm["arm_id"]))))
    trace = load_trace(ep_dir / "trace.json")
    return summary, trace


def write_summary(raw: Mapping[str, Any], out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Vehicle safe-shortening v1 case9 seed2 counterfactual v1")
    lines.append("")
    lines.append("Created UTC: `%s`." % raw["created_utc"])
    lines.append("")
    lines.append("Development causal diagnostic only on already-opened fresh devval shard13/case9; no training and no sealed-test access.")
    lines.append("")
    lines.append("## Budget/access")
    lines.append("")
    for k, v in raw["budget_actual"].items():
        lines.append("- %s: `%s`." % (k, v))
    lines.append("")
    lines.append("## Existing reference episodes")
    lines.append("")
    lines.append("- Existing adaptive: `%s`." % raw["existing_reference"]["adaptive_brief"])
    lines.append("- Existing matched H25: `%s`." % raw["existing_reference"]["fixed_H25_brief"])
    lines.append("")
    lines.append("## Counterfactual outcomes")
    lines.append("")
    lines.append("| replay | success | termination | steps | phys+constraint | total | horizons | solver failures | interpretation |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|---|")
    for mode in ("adaptive_force_H25_at_step57", "fixed_H25_inject_H10_at_step57"):
        b = raw["counterfactuals"][mode]["brief"]
        lines.append("| `%s` | %s | %s | %d | %.6g | %.6g | %s | %d | %s |" % (
            mode, b["success"], b["termination"], b["steps"], b["physical_constraint_cost"], b["total_cost"],
            b["horizon_counts"], b["solver_failure_steps"], raw["counterfactuals"][mode]["interpretation"],
        ))
    lines.append("")
    lines.append("## Trace identity checks")
    lines.append("")
    lines.append("- Forced-H25 vs existing matched-H25: `%s`." % raw["trace_comparisons"]["adaptive_force_H25_vs_existing_fixed_H25"])
    lines.append("- Inject-H10 vs existing adaptive: `%s`." % raw["trace_comparisons"]["fixed_inject_H10_vs_existing_adaptive"])
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(raw["diagnostic_conclusion"])
    lines.append("")
    lines.append("## Next action")
    lines.append("")
    lines.append(raw["next_action"])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backup_request(raw: Mapping[str, Any], out_dir: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_COUNTERFACTUAL_V1_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup bounded deterministic case9 seed2 counterfactual diagnostic before IMPROVED v2 method revision",
        "artifacts": [
            rel(out_dir / "summary.md"), rel(out_dir / "raw.json"), rel(out_dir / "completed.json"),
            rel(out_dir / "episodes"), SCRIPT_REL, rel(runner.PROTOCOL), rel(runner.GATE_PATH), rel(runner.BANK_PATH),
        ],
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
        "## 2026-09-28 vehicle safe-shortening v1 case9 seed2 counterfactual v1\n\n"
        "UTC: %s. Ran two deterministic one-variable counterfactual episodes on already-opened fresh devval shard13/case9 seed2: force H25 at the saved adaptive H10 step, and inject H10 at the same step into the matched-H25 path. Budget: %d episodes, %d control steps, 0 training/gradient steps; sealed test closed. Outcome: %s Artifacts: `%s`, `%s`, `%s`; backup request `%s`.\n"
        % (
            MARKER, raw["created_utc"], raw["budget_actual"]["new_rollout_episodes"], raw["budget_actual"]["new_control_steps"],
            raw["diagnostic_conclusion"], rel(out_dir / "summary.md"), rel(out_dir / "raw.json"), rel(out_dir / "completed.json"), rel(backup),
        )
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")
    decision = (
        "\n<!-- %s-decision -->\n"
        "### Decision: case9 seed2 v1 failure is a horizon-choice causal hazard, not dispatch/noise\n\n"
        "Before evidence: full devval64 rejected safe-shortening v1; metadata diagnostic showed adaptive and matched-H25 case9 trajectories were identical before step57, where adaptive selected H10 once, then adaptive failed while matched H25 succeeded.\n\n"
        "Diagnostic change: two one-variable deterministic replays on the already-opened case only. Force H25 at the adaptive H10 step, and inject H10 into the fixed-H25 path at the same step. No training, no controller/terminal/model/source mutation, no sealed test.\n\n"
        "Outcome: see `%s`. This supports revising the IMPROVED method toward state-level safety/value estimation rather than revalidating v1. Evidence remains development-only and contaminated by case9 diagnosis.\n"
        % (MARKER, rel(out_dir / "summary.md"))
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
    def put(names: Iterable[str], value: Any) -> None:
        for n in names:
            if n in lower:
                row[lower[n]] = str(value); return
    put(("timestamp", "created_utc", "utc", "time"), raw["created_utc"])
    put(("method",), raw["method"])
    put(("script", "source", "script_path"), SCRIPT_REL)
    put(("split",), raw["split"])
    put(("status", "result", "outcome"), "completed_development_counterfactual")
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
    out_dir = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_counterfactual_v1_%s" % now.strftime("%Y%m%dT%H%M%SZ"))
    if out_dir.exists():
        raise RuntimeError("Refusing to overwrite existing output: %s" % rel(out_dir))
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": now.isoformat(), "pid": os.getpid(), "method": "IMPROVED_vehicle_safe_shortening_v1_case9_counterfactual_v1",
        "split": "already_opened_fresh_devval_shard13_case9_development_diagnostic_only", "sealed_test_accessed": False,
    })
    runtime = runner.verify_runtime()
    if not runtime.get("passed"):
        raise RuntimeError(runtime.get("diagnosis") + ": " + runtime.get("exception", ""))
    runner.v1.latency_verify()
    gate = runner.freeze_gate_if_needed()
    bank = read_json(runner.BANK_PATH)
    case = bank["cases"][CASE_ID]
    arms = {a["arm_id"]: a for a in gate["schedule"]["arms"]}
    adaptive_arm = arms["safe_shortening_v1_vehicle_s2"]
    fixed_arm = arms["matched_terminal_fixed_H25_vehicle_s2"]
    terminal_source = adaptive_arm["terminal_source"]
    if fixed_arm["terminal_source"] != terminal_source:
        raise RuntimeError("Expected same terminal source for adaptive and fixed H25 case9 seed2")
    load_start = time.perf_counter()
    terminal, terminal_receipt = runner.load_terminal(terminal_source)
    terminal_load_s = time.perf_counter() - load_start

    existing_adaptive_dir, existing_adaptive_summary_path, existing_adaptive_trace, existing_adaptive_summary = find_existing_episode("shard13/episodes/exec*_case09_safe_shortening_v1_vehicle_s2")
    existing_fixed_dir, existing_fixed_summary_path, existing_fixed_trace, existing_fixed_summary = find_existing_episode("shard13/episodes/exec*_case09_matched_terminal_fixed_H25_vehicle_s2")
    existing_short_steps = [i for i, r in enumerate(existing_adaptive_trace) if int(r.get("horizon", BASE_H)) < BASE_H]
    if existing_short_steps != [H10_STEP]:
        raise RuntimeError("Expected single existing H10 step %d, got %s" % (H10_STEP, existing_short_steps))

    common_row = {"case_index": CASE_ID, "case_block_index": 13, "arm_index": -1, "local_arm_order": -1, "terminal_source": terminal_source}
    row_force = dict(common_row, execution_index=9100, arm_id=adaptive_arm["arm_id"], family=adaptive_arm["family"], seed=SEED, controller_h="adaptive_force_H25_at_step57")
    row_inject = dict(common_row, execution_index=9101, arm_id=fixed_arm["arm_id"], family=fixed_arm["family"], seed=SEED, controller_h="fixed_H25_inject_H10_at_step57")
    summary_force, trace_force = run_counterfactual(out_dir, "adaptive_force_H25_at_step57", row_force, adaptive_arm, case, terminal, terminal_load_s)
    summary_inject, trace_inject = run_counterfactual(out_dir, "fixed_H25_inject_H10_at_step57", row_inject, fixed_arm, case, terminal, terminal_load_s)
    created = dt.datetime.now(dt.timezone.utc)
    elapsed = (created - SERVICE_START).total_seconds()
    force_success = bool(summary_force.get("success"))
    inject_failure_like = (not bool(summary_inject.get("success"))) and int(summary_inject.get("steps", 0)) == int(existing_adaptive_summary.get("steps", 0))
    if force_success and inject_failure_like:
        conclusion = "The one-step horizon choice is causally implicated: forcing H25 at the saved H10 decision rescues the case, while injecting H10 into the matched-H25 path reproduces the adaptive failure pattern."
    elif force_success:
        conclusion = "Forcing H25 at the saved H10 decision rescues the case, but the H10 injection replay did not fully reproduce the adaptive failure pattern; the H10 decision is strongly implicated but additional warm-start/path effects should be checked."
    else:
        conclusion = "Forcing H25 at the saved H10 decision did not rescue the case; the single H10 decision alone is not sufficient as a causal explanation and broader trajectory/solver diagnostics are required."
    raw: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_case9_counterfactual_v1_development_only",
        "split": "already_opened_fresh_devval_shard13_case9_development_diagnostic_only",
        "formal_scientific_evidence": False,
        "access_control": {"sealed_test_accessed": False, "historical_validation64_bank_opened": False, "final_test_authorization_requested": False},
        "budget_actual": {
            "new_rollout_episodes": 2,
            "new_control_steps": int(summary_force.get("steps", 0)) + int(summary_inject.get("steps", 0)),
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_episodes": 0,
            "sealed_test_control_steps": 0,
            "already_opened_existing_reference_traces_loaded": 2,
        },
        "runtime_preflight": runtime,
        "service_elapsed_at_diagnostic": {"seconds_since_2026-09-26T10:55:29.419331Z": elapsed, "hours": elapsed / 3600.0, "research_sqlite_total_tokens": "unknown; repository tools cannot access supervisor sqlite"},
        "case": {"case_id": CASE_ID, "seed": SEED, "target_step": H10_STEP, "short_h": SHORT_H, "base_h": BASE_H, "bank": {"path": rel(runner.BANK_PATH), "sha256": sha256(runner.BANK_PATH)}},
        "protocol": {"path": rel(runner.PROTOCOL), "sha256": sha256(runner.PROTOCOL)},
        "gate": {"path": rel(runner.GATE_PATH), "sha256": sha256(runner.GATE_PATH)},
        "terminal_source": terminal_receipt,
        "source_hashes": {
            SCRIPT_REL: sha256(ROOT / SCRIPT_REL),
            "experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py": sha256(ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py"),
            "experiments/bohn2021_aws/vehicle_safe_shortening_v1_smoke.py": sha256(ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_smoke.py"),
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "existing_reference": {
            "adaptive_dir": rel(existing_adaptive_dir), "adaptive_summary_path": rel(existing_adaptive_summary_path), "adaptive_brief": episode_brief(existing_adaptive_summary),
            "fixed_H25_dir": rel(existing_fixed_dir), "fixed_H25_summary_path": rel(existing_fixed_summary_path), "fixed_H25_brief": episode_brief(existing_fixed_summary),
            "existing_adaptive_short_steps": existing_short_steps,
        },
        "counterfactuals": {
            "adaptive_force_H25_at_step57": {"summary_path": rel(out_dir / "episodes" / "exec9100_case09_safe_shortening_v1_vehicle_s2" / "summary.json"), "brief": episode_brief(summary_force), "interpretation": "rescue if success and close to matched H25"},
            "fixed_H25_inject_H10_at_step57": {"summary_path": rel(out_dir / "episodes" / "exec9101_case09_matched_terminal_fixed_H25_vehicle_s2" / "summary.json"), "brief": episode_brief(summary_inject), "interpretation": "failure reproduction if it matches the original adaptive failure"},
        },
        "trace_comparisons": {
            "adaptive_force_H25_vs_existing_fixed_H25": compare_trace(trace_force, existing_fixed_trace, "force_H25_vs_existing_fixed_H25"),
            "fixed_inject_H10_vs_existing_adaptive": compare_trace(trace_inject, existing_adaptive_trace, "inject_H10_vs_existing_adaptive"),
            "existing_adaptive_vs_existing_fixed_H25_common_prefix": compare_trace(existing_adaptive_trace[:H10_STEP], existing_fixed_trace[:H10_STEP], "existing_prefix_before_step57"),
        },
        "diagnostic_conclusion": conclusion,
        "next_action": "Use case9 plus case43 causal diagnostics to freeze an IMPROVED v2 amendment that avoids state-unsafe shortening and learns/selects horizons from state-level continuation-risk/value rather than strict static gates; do not revalidate unchanged v1 or open sealed test.",
    }
    write_json(out_dir / "raw.json", raw)
    write_summary(raw, out_dir)
    backup = backup_request(raw, out_dir)
    append_docs(raw, out_dir, backup)
    append_registry(raw, out_dir)
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"] + [backup, ROOT / SCRIPT_REL, runner.PROTOCOL, runner.GATE_PATH, runner.BANK_PATH]
    write_json(out_dir / "completed.json", {
        "passed": True,
        "method": raw["method"],
        "formal_scientific_evidence": False,
        "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"],
        "new_control_steps": raw["budget_actual"]["new_control_steps"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request": rel(backup),
        "headline": {
            "adaptive_force_H25_at_step57": episode_brief(summary_force),
            "fixed_H25_inject_H10_at_step57": episode_brief(summary_inject),
            "diagnostic_conclusion": conclusion,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(out_dir / "completed.json"), "summary": rel(out_dir / "summary.md"), "raw": rel(out_dir / "raw.json"),
        "new_rollout_episodes": raw["budget_actual"]["new_rollout_episodes"], "new_control_steps": raw["budget_actual"]["new_control_steps"],
        "sealed_test_accessed": False, "force_H25_brief": episode_brief(summary_force), "inject_H10_brief": episode_brief(summary_inject),
        "diagnostic_conclusion": conclusion,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        fail_dir = ROOT / "research_artifacts/aws_diagnostics" / ("vehicle_safe_shortening_v1_case9_counterfactual_v1_failure_%s" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        fail_dir.mkdir(parents=True, exist_ok=True)
        write_json(fail_dir / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "method": "IMPROVED_vehicle_safe_shortening_v1_case9_counterfactual_v1",
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_accessed": False,
            "historical_validation64_bank_opened": False,
            "next_recovery_hint": "Preserve this failed diagnostic; inspect schema/output conflict and rerun only a versioned fix. Do not open sealed test.",
        })
        raise
