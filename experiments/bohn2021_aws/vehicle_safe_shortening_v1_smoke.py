#!/usr/bin/env python3
"""Vehicle safe-shortening v1 engineering smoke.

This bounded diagnostic exercises the v1 risk-sensitive safe-shortening wrapper
specified in research_artifacts/aws_protocols/vehicle_safe_shortening_v1_protocol_20260927.md.
It reuses completed old gated-horizon vehicle policies, clamps/falls back to H25
when required, and compares against same-seed fixed H25 on the registered
vehicle smoke bank only.  It is not validation/model-selection evidence and it
never opens validation64 or sealed test content.
"""

from __future__ import annotations

import copy
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
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "experiments/bohn2021_reproduction"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from latency_tree_protocol import TASKS, bank_name, model_dir, verify as latency_verify  # noqa:E402
from gated_horizon_policy import BASE as GATED_BASE, decide as gated_decide  # noqa:E402
from conservative_canonical_reset import make_env  # noqa:E402
from branch_calibration_run import meter, observed_step  # noqa:E402
from branch_calibration_audit import audit_trace  # noqa:E402
from gated_horizon_timing import LoggingTimer  # noqa:E402
from gated_horizon_search import case_metrics  # noqa:E402
from relative_policy_features import context  # noqa:E402
from runtime import imports  # noqa:E402
from run import weights_hash  # noqa:E402
import conservative_solver_recovery as recovery_module  # noqa:E402

TASK = "vehicle"
SEEDS = (0, 1, 2)
CASES_EXPECTED = 2
REPEATS = 2
MAX_STEPS = 150
ORDER_SEED = 2609278100
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_safe_shortening_v1_protocol_20260927.md"
OLD_OUT = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25"
MARKER = "vehicle-safe-shortening-v1-smoke-20260927"


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(data.size),
        "sum": float(data.sum()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
    }


def clean_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(trace)
    for row in cleaned:
        row.pop("timing", None)
        for attempt in row.get("recovery", {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def verify_existing_completed() -> Dict[str, Any]:
    completed = read_json(OUT_DIR / "completed.json")
    assert completed.get("passed") is True
    for name, digest in completed.get("hashes", {}).items():
        assert sha256(ROOT / name) == digest, name
    return completed


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    if (OUT_DIR / "completed.json").exists():
        verify_existing_completed()
        raise SystemExit("vehicle safe-shortening v1 smoke already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    assert not leftovers, "Partial v1 smoke output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10])


def old_policy_path(seed: int) -> Path:
    return OLD_OUT / "train" / ("vehicle_s%d" % seed) / "policy.json"


def load_old_policy(seed: int) -> Dict[str, Any]:
    path = old_policy_path(seed)
    policy = read_json(path)
    assert policy["task"] == TASK and int(policy["training_seed"]) == seed
    assert int(policy["short_h"]) <= GATED_BASE[TASK], policy
    return policy


def load_terminal(seed: int, terminal_hashes: Dict[str, Any]) -> Tuple[Any, Any]:
    _, SAC, _ = imports()
    folder = model_dir(TASK, seed)
    done = read_json(folder / "completed.json")
    manifest = read_json(folder / "manifest.json")
    assert done["status"] == "complete" and done["steps"] == 15000
    assert manifest["task"] == TASK and manifest["seed"] == seed and manifest["fixed_horizon"] == 25
    model = SAC.load(str(folder / "model.zip"))
    try:
        assert weights_hash(model) == done["final_hash"]
        terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally:
        model.sess.close()
    terminal_hashes[str(seed)] = {
        "folder": rel(folder),
        "model_zip_sha256": sha256(folder / "model.zip"),
        "manifest_sha256": sha256(folder / "manifest.json"),
        "completed_sha256": sha256(folder / "completed.json"),
        "weights_hash": done["final_hash"],
    }
    return terminal


def build_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = []
    for seed in SEEDS:
        policy = load_old_policy(seed)
        arms.append({
            "arm_id": "safe_shortening_v1_vehicle_s%d" % seed,
            "seed": seed,
            "family": "IMPROVED_safe_shortening_v1_reused_gated_policy",
            "policy": policy,
            "policy_path": rel(old_policy_path(seed)),
            "policy_sha256": sha256(old_policy_path(seed)),
        })
        arms.append({
            "arm_id": "fixed_H25_vehicle_s%d" % seed,
            "seed": seed,
            "family": "fixed_H25_primary_same_seed",
            "policy": {"id": "fixed", "task": TASK},
            "policy_path": None,
            "policy_sha256": None,
        })
    assert len(arms) == 6
    return arms


def randomized_schedule(arms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    index = 0
    for repeat in range(REPEATS):
        for arm_index in rng.permutation(len(arms)).tolist():
            for case_id in rng.permutation(CASES_EXPECTED).tolist():
                schedule.append({
                    "episode_index": index,
                    "repeat": int(repeat),
                    "arm_id": arms[int(arm_index)]["arm_id"],
                    "seed": arms[int(arm_index)]["seed"],
                    "case": int(case_id),
                })
                index += 1
    assert len(schedule) == len(arms) * CASES_EXPECTED * REPEATS == 24
    return schedule


def safe_decide(arm: Dict[str, Any], ctx: Dict[str, Any], previous_initial_failure: bool, previous_final_failure: bool) -> Tuple[int, Dict[str, Any]]:
    base = GATED_BASE[TASK]
    if arm["family"].startswith("fixed"):
        return base, {"kind": "fixed", "reason": "same_seed_H25"}
    if previous_initial_failure or previous_final_failure:
        return base, {"kind": "safe_shortening_v1", "fallback": True, "reason": "previous_solver_failure", "raw_horizon": base}
    raw_h, gate = gated_decide(arm["policy"], ctx)
    if int(raw_h) not in range(5, 51, 5):
        raise RuntimeError("unsafe non-grid horizon requested: %r" % (raw_h,))
    selected = min(int(raw_h), base)
    return selected, {
        "kind": "safe_shortening_v1",
        "fallback": False,
        "raw_horizon": int(raw_h),
        "selected_horizon": int(selected),
        "clamped_to_base": int(raw_h) > base,
        "gate": gate,
    }


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float, terminal_load_s: float,
                      episode_wall_s: float, case_id: int, repeat: int, arm: Dict[str, Any]) -> Dict[str, Any]:
    metric = case_metrics(TASK, trace)
    decision = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller = [float(r["timing"]["controller_s"]) for r in trace]
    selection = [float(r["timing"]["selection_s"]) for r in trace]
    solver_times: List[float] = []
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    clamps = 0
    fallbacks = 0
    for row in trace:
        horizons[str(row["horizon"])] = horizons.get(str(row["horizon"]), 0) + 1
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
        "decision_timing_s": values_summary(decision),
        "decision_gross_timing_s": values_summary(decision_gross),
        "controller_timing_s_logging_deducted": values_summary(controller),
        "selection_timing_s": values_summary(selection),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision, dtype=float) > 0.1)),
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "clamped_steps": int(clamps),
        "solver_failure_fallback_steps": int(fallbacks),
    })
    return out


def run_episode(item: Dict[str, Any], arm: Dict[str, Any], case: Dict[str, Any], terminal: Tuple[Any, Any], terminal_load_s: float) -> Dict[str, Any]:
    ep_dir = OUT_DIR / "episodes" / ("ep%03d_r%d_%s_case%02d" % (item["episode_index"], item["repeat"], arm["arm_id"], item["case"]))
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = make_env(TASK, arm["seed"], aligned=True, scaled_obs=True)
    counts = meter(env, ep_dir)
    env.set_value_function_weights_and_biases(*terminal)
    construction_s = time.perf_counter() - construct_start
    controller = env.control_system.controller
    original = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    with LoggingTimer(ep_dir) as logging:
        recovery = recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=item["case"], step=-1)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(case))
        reset_gross_s = time.perf_counter() - reset_start
        assert obs is not None
        reset_measure = measured.pop() if measured else {"controller_gross_s": None, "logging_s": None, "controller_s": None}
        reset_record = {"reset_gross_s": float(reset_gross_s), **reset_measure}
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        previous_initial = False
        previous_final = False
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                recovery["step"] = t
                ctx = context(env, TASK)
                ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                select_start = time.perf_counter()
                selected_h, decision = safe_decide(arm, ctx, previous_initial, previous_final)
                selection_s = time.perf_counter() - select_start
                _, terminated, row = observed_step(env, TASK, selected_h, case, t)
                assert measured, "expected one timed controller call during observed_step"
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
        assert trace and trace[-1].get("termination"), "episode did not terminate within max steps"
        audit_trace(TASK, case, trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(trace, reset_record, construction_s, terminal_load_s, time.perf_counter() - episode_start, item["case"], item["repeat"], arm)
        summary.update({"episode_index": item["episode_index"], "path": rel(ep_dir), "steps_metered": counts["step_calls"], "resets_metered": counts["reset_calls"], "solver_counts": recovery["counts"], "logging_operations": logging.operations, "logging_total_s": float(logging.seconds)})
        write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


def aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "episodes": len(episodes),
        "steps": int(sum(e.get("steps", 0) for e in episodes)),
        "success_count": int(sum(1 for e in episodes if e.get("success"))),
        "episode_failure_count": int(sum(1 for e in episodes if e.get("episode_failure"))),
        "constraint_count": int(sum(1 for e in episodes if e.get("constraint"))),
        "initial_failed_steps": int(sum(e.get("initial_failed_steps", 0) for e in episodes)),
        "solver_failure_steps": int(sum(e.get("solver_failure_steps", 0) for e in episodes)),
        "retries": int(sum(e.get("retries", 0) for e in episodes)),
        "recovered_steps": int(sum(e.get("recovered_steps", 0) for e in episodes)),
        "deadline_exceed_steps": int(sum(e.get("deadline_exceed_steps", 0) for e in episodes)),
        "switches": int(sum(e.get("switches", 0) for e in episodes)),
        "clamped_steps": int(sum(e.get("clamped_steps", 0) for e in episodes)),
        "solver_failure_fallback_steps": int(sum(e.get("solver_failure_fallback_steps", 0) for e in episodes)),
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "decision_total_s": float(math.fsum(float(e["decision_timing_s"]["sum"]) for e in episodes)),
        "decision_gross_total_s": float(math.fsum(float(e["decision_gross_timing_s"]["sum"]) for e in episodes)),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
    }
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[h] = horizons.get(h, 0) + int(n)
        for h, n in (e.get("raw_horizon_counts_before_clamp") or {}).items():
            raw_horizons[h] = raw_horizons.get(h, 0) + int(n)
    out.update({
        "total_cost_mean_episode": out["total_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "physical_constraint_cost_mean_episode": out["physical_constraint_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "decision_mean_s_per_step": out["decision_total_s"] / out["steps"] if out["steps"] else None,
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "adapted_below_H25": any(int(h) < 25 for h in horizons),
        "unsafe_above_H25_dispatch": any(int(h) > 25 for h in horizons),
    })
    return out


def compare(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        a = aggregates["safe_shortening_v1_vehicle_s%d" % seed]
        b = aggregates["fixed_H25_vehicle_s%d" % seed]
        out[str(seed)] = {
            "adaptive_unique_horizons": a["unique_horizons"],
            "fixed_unique_horizons": b["unique_horizons"],
            "adapted_below_H25": a["adapted_below_H25"],
            "unsafe_above_H25_dispatch": a["unsafe_above_H25_dispatch"],
            "success_counts": {"adaptive": a["success_count"], "fixed": b["success_count"]},
            "episode_failure_counts": {"adaptive": a["episode_failure_count"], "fixed": b["episode_failure_count"]},
            "physical_constraint_delta_adaptive_minus_fixed": a["physical_constraint_cost_sum"] - b["physical_constraint_cost_sum"],
            "total_cost_delta_adaptive_minus_fixed": a["total_cost_sum"] - b["total_cost_sum"],
            "decision_time_ratio_adaptive_over_fixed": a["decision_total_s"] / b["decision_total_s"] if b["decision_total_s"] else None,
        }
    return out


def write_summary(raw: Dict[str, Any]) -> None:
    lines = [
        "# Vehicle safe-shortening v1 smoke",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Engineering smoke only: existing vehicle smoke bank, two cases, two deterministic repeats; no validation64 access and no sealed-test access.",
        "",
        f"Protocol: `{raw['protocol']['path']}` sha256 `{raw['protocol']['sha256']}`.",
        "",
        "## Budget",
        "",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- New gradient steps: `{raw['budget_actual']['new_gradient_steps']}`.",
        f"- Replay passed: `{raw['replay']['passed']}` over `{raw['replay']['pairs_checked']}` repeat pairs.",
        "",
        "## Arm aggregates",
        "",
        "| arm | seed | episodes | steps | success | failures | physical+constraint | decision mean s/step | horizons | clamps | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for arm_id in sorted(raw["aggregates"]):
        a = raw["aggregates"][arm_id]
        lines.append(f"| `{arm_id}` | {a['seed']} | {a['episodes']} | {a['steps']} | {a['success_count']} | {a['episode_failure_count']} | {a['physical_constraint_cost_sum']:.6g} | {a['decision_mean_s_per_step']:.6g} | {a['horizon_counts']} | {a['clamped_steps']} | {a['solver_failure_fallback_steps']} |")
    lines += ["", "## Same-seed smoke deltas", ""]
    for seed, c in raw["paired_comparisons"].items():
        lines.append(f"- seed {seed}: adapted_below_H25={c['adapted_below_H25']}; unsafe_above_H25_dispatch={c['unsafe_above_H25_dispatch']}; physical delta={c['physical_constraint_delta_adaptive_minus_fixed']:.6g}; total delta={c['total_cost_delta_adaptive_minus_fixed']:.6g}; decision ratio={c['decision_time_ratio_adaptive_over_fixed']:.6g}; horizons={c['adaptive_unique_horizons']}.")
    lines += ["", "## Interpretation", "", "This smoke tests only implementation readiness of the safe-shortening wrapper and AWS timing/accounting boundaries. It is not fresh validation/model-selection evidence and cannot support a reproduction or improved-method success claim. If passed and externally backed up, the next scientific step is a frozen fresh vehicle development-validation block against strong fixed-H baselines."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Dict[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-27 vehicle safe-shortening v1 smoke\n\n"
        f"UTC: {raw['created_utc']}. Engineering smoke for IMPROVED safe-shortening wrapper completed on vehicle smoke bank only: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, replay passed={raw['replay']['passed']}. "
        "Validation64 and sealed test remained closed. This is not model-selection/final evidence. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def main() -> int:
    assert TASK in TASKS
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "validation_accessed": False, "test_accessed": False, "method": "IMPROVED_safe_shortening_v1_smoke"})
    latency_verify()
    assert PROTOCOL.exists(), "v1 protocol file must be frozen before smoke"
    bank = bank_name(TASK, "smoke")
    banks_done = read_json(bank.parent / "completed.json")
    assert sha256(bank) == banks_done["hashes"][str(bank)]
    bank_data = read_json(bank)
    assert bank_data["task"] == TASK and bank_data["split"] == "smoke" and len(bank_data["cases"]) == CASES_EXPECTED
    arms = build_arms()
    schedule = randomized_schedule(arms)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "arms": arms})
    terminal_hashes: Dict[str, Any] = {}
    terminals: Dict[int, Tuple[Any, Any]] = {}
    terminal_load_s: Dict[int, float] = {}
    for seed in SEEDS:
        start = time.perf_counter()
        terminals[seed] = load_terminal(seed, terminal_hashes)
        terminal_load_s[seed] = time.perf_counter() - start
    write_json(OUT_DIR / "terminal_sources.json", terminal_hashes)
    arm_by_id = {a["arm_id"]: a for a in arms}
    episodes: List[Dict[str, Any]] = []
    trace_index: Dict[Tuple[str, int, int], Path] = {}
    replay = {"passed": True, "pairs_checked": 0, "mismatches": []}
    for item in schedule:
        arm = arm_by_id[item["arm_id"]]
        summary = run_episode(item, arm, bank_data["cases"][item["case"]], terminals[arm["seed"]], terminal_load_s[arm["seed"]])
        episodes.append(summary)
        trace_path = ROOT / summary["path"] / "trace.json"
        trace_index[(item["arm_id"], item["case"], item["repeat"])] = trace_path
        if item["repeat"] == 1:
            base = trace_index[(item["arm_id"], item["case"], 0)]
            replay["pairs_checked"] += 1
            if clean_trace(read_json(base)) != clean_trace(read_json(trace_path)):
                replay["passed"] = False
                replay["mismatches"].append({"arm_id": item["arm_id"], "case": item["case"], "repeat0": rel(base), "repeat1": rel(trace_path)})
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule), "control_steps_done": int(sum(e["steps"] for e in episodes)), "last_episode": {k: summary[k] for k in ("episode_index", "arm_id", "case", "repeat", "steps", "success", "termination")}}
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    assert replay["passed"], replay["mismatches"][:3]
    control_steps = int(sum(e["steps"] for e in episodes))
    assert len(episodes) == 24 and control_steps <= 3600
    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({"seed": arm["seed"], "family": arm["family"], "policy": arm["policy"], "policy_path": arm["policy_path"], "policy_sha256": arm["policy_sha256"]})
        aggregates[arm["arm_id"]] = agg
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_safe_shortening_v1_reused_gated_policies_not_original_SAC",
        "formal_scientific_evidence": False,
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "split": "vehicle_smoke_bank_only",
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "bank": {"path": rel(bank), "sha256": sha256(bank), "cases": CASES_EXPECTED},
        "source_hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()), rel(REPRO / "gated_horizon_policy.py"): sha256(REPRO / "gated_horizon_policy.py"), rel(REPRO / "latency_tree_protocol.py"): sha256(REPRO / "latency_tree_protocol.py"), rel(REPRO / "conservative_canonical_reset.py"): sha256(REPRO / "conservative_canonical_reset.py"), rel(REPRO / "conservative_solver_recovery.py"): sha256(REPRO / "conservative_solver_recovery.py")},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "budget_declared": {"episodes_exact": 24, "control_step_upper_bound": 3600, "new_gradient_steps": 0, "validation_episodes": 0, "test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "resets": int(sum(e["resets_metered"] for e in episodes)), "environment_constructions": len(episodes), "new_gradient_steps": 0, "validation_episodes": 0, "test_episodes": 0},
        "arms": arms,
        "randomized_schedule": schedule,
        "terminal_sources": terminal_hashes,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired_comparisons": compare(aggregates),
        "replay": replay,
        "interpretation_limits": ["engineering smoke only", "not validation/model selection", "reuses unequal historical adaptive search budget", "H distribution is not timing evidence"],
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    backup_request = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_SMOKE_20260927T000000Z.json"
    write_json(backup_request, {"requested_utc": raw["created_utc"], "reason": "backup v1 safe-shortening protocol/source/smoke artifacts before fresh validation", "artifacts": [rel(OUT_DIR), rel(PROTOCOL), rel(Path(__file__).resolve())], "validation_accessed": False, "test_accessed": False})
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [backup_request, PROTOCOL, Path(__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(set(files))}, "backup_request": rel(backup_request), "formal_scientific_evidence": False, "validation_accessed": False, "test_accessed": False})
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "replay_passed": replay["passed"], "validation_accessed": False, "test_accessed": False, "comparisons": raw["paired_comparisons"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation_accessed": False, "test_accessed": False})
        raise
