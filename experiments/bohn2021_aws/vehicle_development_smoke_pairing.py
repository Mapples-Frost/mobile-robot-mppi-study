#!/usr/bin/env python3
"""AWS-only no-validation vehicle paired timing/control smoke.

This is an engineering/development smoke runner for the post-amendment
latency-tree migration.  It intentionally uses only the registered
vehicle_smoke_bank, compares the three saved learned vehicle policies against
same-seed fixed H25, and writes a separate diagnostics bundle.  It is not a
formal validation gate, does not open validation64, and does not open the sealed
test bank.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "experiments/bohn2021_reproduction"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from latency_tree_protocol import ART, OUT as LAT_OUT, REG, TASKS, bank_name, model_dir, sha, verify  # noqa:E402
from latency_tree_policy import choose, constant, features, policy_key  # noqa:E402
from conservative_canonical_reset import make_env  # noqa:E402
from branch_calibration_run import meter, observed_step  # noqa:E402
from branch_calibration_audit import audit_trace  # noqa:E402
from gated_horizon_timing import LoggingTimer  # noqa:E402
from gated_horizon_search import case_metrics  # noqa:E402
from relative_policy_features import context  # noqa:E402
from runtime import imports  # noqa:E402
from run import weights_hash  # noqa:E402
import conservative_solver_recovery as recovery_module  # noqa:E402

OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing"
FREEZE_JSON = ROOT / "research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/vehicle_development_timing_freeze.json"
FREEZE_RAW = ROOT / "research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/raw.json"
TASK = "vehicle"
SEEDS = (0, 1, 2)
CASES_EXPECTED = 2
REPEATS = 2
ORDER_SEED = 2609268200
MAX_STEPS = 150
MARKER = "vehicle-development-smoke-pairing-20260926"


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
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def clean_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(trace)
    for row in cleaned:
        row.pop("timing", None)
        for attempt in row.get("recovery", {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def verify_existing_completed() -> Dict[str, Any]:
    completed = read_json(OUT_DIR / "completed.json")
    for name, digest in completed.get("hashes", {}).items():
        assert sha256(ROOT / name) == digest, name
    assert completed.get("passed") is True
    return completed


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    if (OUT_DIR / "completed.json").exists():
        verify_existing_completed()
        raise SystemExit("vehicle development smoke already completed and verified; refusing to rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    assert not leftovers, "Partial vehicle smoke output exists; inspect/audit before recovery: " + ", ".join(rel(p) for p in leftovers[:10])


def load_policy(seed: int, freeze: Dict[str, Any]) -> Dict[str, Any]:
    path = LAT_OUT / "train" / ("vehicle_s%d" % seed) / "policy.json"
    policy = read_json(path)
    arm_id = "learned_latency_tree_vehicle_s%d" % seed
    assert sha256(path) == freeze["candidate_policy_hashes"][arm_id], arm_id
    assert canonical_hash(policy) == freeze["candidate_policy_canonical_hashes"][arm_id], arm_id
    assert policy.get("task") == TASK
    return policy


def load_terminal(seed: int, terminal_hashes: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    _, SAC, _ = imports()
    folder = model_dir(TASK, seed)
    done = read_json(folder / "completed.json")
    manifest = read_json(folder / "manifest.json")
    assert done["status"] == "complete" and done["steps"] == 15000
    assert manifest["task"] == TASK and manifest["seed"] == seed and manifest["fixed_horizon"] == 25
    model_zip_hash = sha256(folder / "model.zip")
    completed_hash = sha256(folder / "completed.json")
    manifest_hash = sha256(folder / "manifest.json")
    model = SAC.load(str(folder / "model.zip"))
    try:
        assert weights_hash(model) == done["final_hash"]
        terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally:
        model.sess.close()
    terminal_hashes[str(seed)] = {
        "folder": rel(folder),
        "model_zip_sha256": model_zip_hash,
        "manifest_sha256": manifest_hash,
        "completed_sha256": completed_hash,
        "weights_hash": done["final_hash"],
    }
    return terminal


def build_arms(freeze: Dict[str, Any]) -> List[Dict[str, Any]]:
    frozen = freeze["minimal_development_block"]
    assert frozen["randomized_order_seed"] == ORDER_SEED
    assert frozen["cases"] == CASES_EXPECTED and frozen["repeats"] == REPEATS
    assert frozen["episode_budget_exact"] == 24
    arms = []
    for item in frozen["arms"]:
        arm_id = item["arm_id"]
        seed = int(item["seed"])
        if arm_id.startswith("learned_latency_tree"):
            policy = load_policy(seed, freeze)
            family = "learned_latency_tree_IMPROVED"
        else:
            policy = constant(TASK, 25)
            assert policy == item["policy"]
            family = "fixed_H25_primary"
        arms.append({"arm_id": arm_id, "seed": seed, "family": family, "policy": policy, "policy_key": policy_key(policy)})
    assert len(arms) == 6 and sorted(a["seed"] for a in arms) == [0, 0, 1, 1, 2, 2]
    return arms


def randomized_schedule(arms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    episode_index = 0
    for repeat in range(REPEATS):
        for arm_index in rng.permutation(len(arms)).tolist():
            case_order = rng.permutation(CASES_EXPECTED).tolist()
            for case_id in case_order:
                row = {
                    "episode_index": episode_index,
                    "repeat": repeat,
                    "arm_id": arms[int(arm_index)]["arm_id"],
                    "seed": arms[int(arm_index)]["seed"],
                    "case": int(case_id),
                }
                schedule.append(row)
                episode_index += 1
    assert len(schedule) == 24
    assert sorted((r["repeat"], r["arm_id"], r["case"]) for r in schedule) == sorted(
        (repeat, arm["arm_id"], case_id) for repeat in range(REPEATS) for arm in arms for case_id in range(CASES_EXPECTED)
    )
    return schedule


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


def summarize_episode(task: str, trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                      terminal_load_s: float, episode_wall_s: float, case_id: int, repeat: int,
                      arm: Dict[str, Any]) -> Dict[str, Any]:
    metric = case_metrics(task, trace)
    decision = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller = [float(r["timing"]["controller_s"]) for r in trace]
    selection = [float(r["timing"]["selection_s"]) for r in trace]
    logging = [float(r["timing"]["logging_s"]) for r in trace]
    solver_times = []
    for row in trace:
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    horizons: Dict[str, int] = {}
    for row in trace:
        h = str(row.get("horizon"))
        horizons[h] = horizons.get(h, 0) + 1
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
        "logging_timing_s": values_summary(logging),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision) > 0.1)),
        "horizon_counts": horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
    })
    return out


def run_episode(schedule_row: Dict[str, Any], arm: Dict[str, Any], case: Dict[str, Any], terminal: Tuple[Any, Any],
                terminal_load_s: float) -> Dict[str, Any]:
    ep_dir = OUT_DIR / "episodes" / ("ep%03d_r%d_%s_case%02d" % (
        schedule_row["episode_index"], schedule_row["repeat"], arm["arm_id"], schedule_row["case"]))
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
    reset_record: Dict[str, Any]
    trace: List[Dict[str, Any]] = []
    with LoggingTimer(ep_dir) as logging:
        recovery = recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            assert 0.0 <= logged < gross
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=schedule_row["case"], step=-1)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(case))
        reset_gross_s = time.perf_counter() - reset_start
        assert obs is not None
        assert len(measured) == 1, "expected one controller call during reset warmup"
        reset_record = {"reset_gross_s": float(reset_gross_s), **measured.pop()}
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        previous_initial = False
        previous_final = False
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                recovery["step"] = t
                select_start = time.perf_counter()
                if arm["policy"].get("kind") in ("tree", "combined"):
                    ctx = context(env, TASK)
                    ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                    selected_h, decision = choose(arm["policy"], ctx)
                else:
                    selected_h, decision = choose(arm["policy"], {"state": env.control_system.current_state})
                    ctx = None
                selection_s = time.perf_counter() - select_start
                if ctx is None:
                    ctx = context(env, TASK)
                    ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                _, terminated, row = observed_step(env, TASK, selected_h, case, t)
                assert len(measured) == 1
                timing = measured.pop()
                timing.update({
                    "selection_s": float(selection_s),
                    "decision_s": float(selection_s + timing["controller_s"]),
                    "decision_gross_s": float(selection_s + timing["controller_gross_s"]),
                })
                row.update({
                    "policy_context": ctx,
                    "tree_features": features(TASK, ctx),
                    "decision": decision,
                    "recovery": recovery["events"][-1],
                    "timing": timing,
                })
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                previous_initial = not bool(row["recovery"]["attempts"][0]["success"])
                previous_final = not bool(row["solver_success"])
                if terminated:
                    break
        assert trace and trace[-1].get("termination"), "episode did not terminate within max_steps"
        audit_trace(TASK, case, trace)
        assert logging.operations == 1 + 3 * recovery["counts"]["solve_completed"], (
            logging.operations, recovery["counts"])
        write_json(ep_dir / "trace.json", trace)
        episode_wall_s = time.perf_counter() - episode_start
        summary = summarize_episode(TASK, trace, reset_record, construction_s, terminal_load_s, episode_wall_s,
                                    schedule_row["case"], schedule_row["repeat"], arm)
        summary.update({
            "episode_index": schedule_row["episode_index"],
            "path": rel(ep_dir),
            "steps_metered": counts["step_calls"],
            "resets_metered": counts["reset_calls"],
            "solver_counts": recovery["counts"],
            "logging_operations": logging.operations,
            "logging_total_s": float(logging.seconds),
        })
        write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name not in ("completed.json",)]
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
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "performance_cost_sum": float(math.fsum(float(e.get("performance_cost", 0.0)) for e in episodes)),
        "constraint_cost_sum": float(math.fsum(float(e.get("constraint_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "h_penalty_sum": float(math.fsum(float(e.get("h_penalty", 0.0)) for e in episodes)),
        "decision_total_s": float(math.fsum(float(e["decision_timing_s"]["sum"]) for e in episodes)),
        "decision_gross_total_s": float(math.fsum(float(e["decision_gross_timing_s"]["sum"]) for e in episodes)),
        "logging_total_s": float(math.fsum(float(e["logging_timing_s"]["sum"]) for e in episodes)),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
    }
    horizons: Dict[str, int] = {}
    decisions: List[float] = []
    solvers: List[float] = []
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[h] = horizons.get(h, 0) + int(n)
        # Episode summaries keep distributions summarized; exact per-step timings
        # remain in trace files.  For aggregate p95/median use per-episode means
        # as a compact top-level overview, and raw traces as authoritative source.
        if e["decision_timing_s"]["mean"] is not None:
            decisions.append(float(e["decision_timing_s"]["mean"]))
        if e["solver_attempt_timing_s"]["mean"] is not None:
            solvers.append(float(e["solver_attempt_timing_s"]["mean"]))
    out.update({
        "total_cost_mean_episode": out["total_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "decision_mean_s_per_step": out["decision_total_s"] / out["steps"] if out["steps"] else None,
        "horizon_counts": horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "episode_mean_decision_s_distribution": values_summary(decisions),
        "episode_mean_solver_attempt_s_distribution": values_summary(solvers),
    })
    return out


def compare_to_fixed(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    for seed in SEEDS:
        learned_id = "learned_latency_tree_vehicle_s%d" % seed
        fixed_id = "fixed_H25_primary_vehicle_s%d" % seed
        learned = aggregates[learned_id]
        fixed = aggregates[fixed_id]
        comparisons[str(seed)] = {
            "learned_arm": learned_id,
            "fixed_arm": fixed_id,
            "learned_unique_horizons": learned.get("unique_horizons"),
            "fixed_unique_horizons": fixed.get("unique_horizons"),
            "learned_switches": learned.get("switches"),
            "fixed_switches": fixed.get("switches"),
            "total_cost_delta_learned_minus_fixed": learned["total_cost_sum"] - fixed["total_cost_sum"],
            "physical_constraint_delta_learned_minus_fixed": learned["physical_constraint_cost_sum"] - fixed["physical_constraint_cost_sum"],
            "decision_time_ratio_learned_over_fixed": learned["decision_total_s"] / fixed["decision_total_s"] if fixed["decision_total_s"] else None,
            "decision_mean_ratio_learned_over_fixed": learned["decision_mean_s_per_step"] / fixed["decision_mean_s_per_step"] if fixed["decision_mean_s_per_step"] else None,
            "success_counts": {"learned": learned["success_count"], "fixed": fixed["success_count"]},
            "constraint_counts": {"learned": learned["constraint_count"], "fixed": fixed["constraint_count"]},
            "initial_failed_steps": {"learned": learned["initial_failed_steps"], "fixed": fixed["initial_failed_steps"]},
            "solver_failure_steps": {"learned": learned["solver_failure_steps"], "fixed": fixed["solver_failure_steps"]},
            "smoke_interpretation_only": "development smoke; not validation evidence and not model selection",
        }
    return comparisons


def write_markdown(raw: Dict[str, Any]) -> None:
    lines = [
        "# Vehicle development smoke pairing",
        "",
        f"Created UTC: {raw['created_utc']}",
        "",
        "Scope: AWS-only engineering/development smoke. Uses only `vehicle_smoke_bank` (2 cases), two repeats, learned latency-tree policies s0/s1/s2 versus same-seed fixed H25. No validation64 content/outcome and no sealed test content/outcome were opened.",
        "",
        f"Formal scientific evidence: {raw['formal_scientific_evidence']}",
        f"Validation accessed: {raw['validation_accessed']}; test accessed: {raw['test_accessed']}",
        "",
        "## Replay and budget checks",
        "",
        f"- replay pairs checked: {raw['replay']['pairs_checked']}",
        f"- replay passed: {raw['replay']['passed']}",
        f"- episodes run: {raw['budget_actual']['episodes']} / declared {raw['budget_declared']['episodes_exact']}",
        f"- control steps: {raw['budget_actual']['control_steps']} / upper bound {raw['budget_declared']['control_step_upper_bound']}",
        "",
        "## Arm aggregates",
        "",
        "| arm | seed | episodes | steps | success | constraints | init-fail steps | final-fail steps | retries | deadlines | total cost | physical+constraint cost | decision mean s/step | decision total s | horizons | switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm_id in sorted(raw["aggregates"]):
        agg = raw["aggregates"][arm_id]
        lines.append(
            f"| `{arm_id}` | {agg['seed']} | {agg['episodes']} | {agg['steps']} | {agg['success_count']} | {agg['constraint_count']} | "
            f"{agg['initial_failed_steps']} | {agg['solver_failure_steps']} | {agg['retries']} | {agg['deadline_exceed_steps']} | "
            f"{agg['total_cost_sum']:.6g} | {agg['physical_constraint_cost_sum']:.6g} | {agg['decision_mean_s_per_step']:.6g} | "
            f"{agg['decision_total_s']:.6g} | {agg['horizon_counts']} | {agg['switches']} |"
        )
    lines.extend(["", "## Paired learned vs fixed-H25 smoke deltas", ""])
    for seed, cmp in raw["paired_comparisons"].items():
        lines.append(
            f"- seed {seed}: cost delta learned-fixed={cmp['total_cost_delta_learned_minus_fixed']:.6g}; "
            f"physical+constraint delta={cmp['physical_constraint_delta_learned_minus_fixed']:.6g}; "
            f"decision time ratio={cmp['decision_time_ratio_learned_over_fixed']:.6g}; "
            f"learned horizons={cmp['learned_unique_horizons']}; switches={cmp['learned_switches']}."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This smoke can only establish that the migrated runner, timing boundaries, solver accounting, and replay controls work on AWS. It cannot establish adaptive superiority, cannot select a model, and cannot support a Bøhn 2021 reproduction claim. Validation64 remains unopened and sealed test remains unauthorized.",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Dict[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-26 vehicle development smoke pairing\n\n"
        f"UTC: {raw['created_utc']}. Non-formal AWS-only vehicle smoke completed on `vehicle_smoke_bank` with "
        f"{raw['budget_actual']['episodes']} episodes and {raw['budget_actual']['control_steps']} control steps. "
        f"Replay passed={raw['replay']['passed']}; validation_accessed=false; test_accessed=false. "
        "Use only for engineering readiness/timing-boundary checks, not validation/model selection. "
        f"Artifacts: `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'summary.md')}`. New artifacts require backup before formal evidence.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def main() -> int:
    assert TASK in TASKS
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED latency-tree AWS vehicle development smoke; not original SAC; not formal evidence",
        "validation_accessed": False,
        "test_accessed": False,
    })
    verify()
    freeze = read_json(FREEZE_JSON)
    freeze_raw = read_json(FREEZE_RAW)
    assert freeze["status"] == "development_timing_smoke_freeze_only_not_formal_validation_gate"
    assert freeze.get("validation_accessed") is False and freeze.get("test_accessed") is False
    bank = bank_name(TASK, "smoke")
    bank_data = read_json(bank)
    assert bank_data["task"] == TASK and bank_data["split"] == "smoke" and len(bank_data["cases"]) == CASES_EXPECTED
    smoke_bank_hash = sha256(bank)
    expected_bank_hash = freeze_raw["split_records"]["development_smoke_bank"]["record"]["sha256"]
    assert smoke_bank_hash == expected_bank_hash
    arms = build_arms(freeze)
    schedule = randomized_schedule(arms)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "arms": arms})

    terminal_hashes: Dict[str, Any] = {}
    terminal_load_s: Dict[int, float] = {}
    terminals: Dict[int, Tuple[Any, Any]] = {}
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
        case = bank_data["cases"][item["case"]]
        summary = run_episode(item, arm, case, terminals[arm["seed"]], terminal_load_s[arm["seed"]])
        episodes.append(summary)
        trace_path = ROOT / summary["path"] / "trace.json"
        trace_index[(item["arm_id"], item["case"], item["repeat"])] = trace_path
        if item["repeat"] == 1:
            base = trace_index[(item["arm_id"], item["case"], 0)]
            current_clean = clean_trace(read_json(trace_path))
            base_clean = clean_trace(read_json(base))
            replay["pairs_checked"] += 1
            if current_clean != base_clean:
                replay["passed"] = False
                replay["mismatches"].append({"arm_id": item["arm_id"], "case": item["case"], "repeat0": rel(base), "repeat1": rel(trace_path)})
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(e["steps"] for e in episodes)),
            "last_episode": {k: summary[k] for k in ("episode_index", "arm_id", "case", "repeat", "steps", "success", "termination")},
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    assert replay["pairs_checked"] == len(arms) * CASES_EXPECTED
    assert replay["passed"], replay["mismatches"][:3]
    assert len(episodes) == freeze["minimal_development_block"]["episode_budget_exact"]
    control_steps = int(sum(e["steps"] for e in episodes))
    assert control_steps <= freeze["minimal_development_block"]["control_step_upper_bound"]

    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({"seed": arm["seed"], "family": arm["family"], "policy_key": arm["policy_key"]})
        aggregates[arm["arm_id"]] = agg
    comparisons = compare_to_fixed(aggregates)
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED latency-tree vehicle no-validation development timing/control smoke",
        "formal_scientific_evidence": False,
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "split": "vehicle_smoke_bank_only",
        "bank": {"path": rel(bank), "sha256": smoke_bank_hash, "cases": CASES_EXPECTED},
        "freeze": {"path": rel(FREEZE_JSON), "sha256": sha256(FREEZE_JSON), "raw_path": rel(FREEZE_RAW), "raw_sha256": sha256(FREEZE_RAW)},
        "source_hashes": {
            rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            rel(REPRO / "latency_tree_policy.py"): sha256(REPRO / "latency_tree_policy.py"),
            rel(REPRO / "conservative_canonical_reset.py"): sha256(REPRO / "conservative_canonical_reset.py"),
            rel(REPRO / "conservative_solver_recovery.py"): sha256(REPRO / "conservative_solver_recovery.py"),
            rel(REPRO / "branch_calibration_run.py"): sha256(REPRO / "branch_calibration_run.py"),
            rel(REPRO / "gated_horizon_timing.py"): sha256(REPRO / "gated_horizon_timing.py"),
            rel(REPRO / "relative_policy_features.py"): sha256(REPRO / "relative_policy_features.py"),
            rel(REPRO / "gated_horizon_search.py"): sha256(REPRO / "gated_horizon_search.py"),
            rel(REG): sha256(REG),
        },
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")},
        },
        "budget_declared": {
            "episodes_exact": freeze["minimal_development_block"]["episode_budget_exact"],
            "control_step_upper_bound": freeze["minimal_development_block"]["control_step_upper_bound"],
            "new_gradient_steps": 0,
            "validation_episodes": 0,
            "test_episodes": 0,
        },
        "budget_actual": {
            "episodes": len(episodes),
            "control_steps": control_steps,
            "resets": int(sum(e["resets_metered"] for e in episodes)),
            "environment_constructions": len(episodes),
            "new_gradient_steps": 0,
            "validation_episodes": 0,
            "test_episodes": 0,
        },
        "randomized_schedule": schedule,
        "terminal_sources": terminal_hashes,
        "arms": arms,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "replay": replay,
        "interpretation_limits": [
            "Non-formal engineering/development evidence only.",
            "Uses two smoke-bank cases, not validation64 and not sealed test128.",
            "No model selection, no success claim, no timing inference beyond instrumentation readiness.",
            "Seed0 and seed1 learned policies are structurally fixed-H25; seed2 is the only structurally switching candidate.",
        ],
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_markdown(raw)
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"]
    write_json(OUT_DIR / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "raw": rel(OUT_DIR / "raw.json"),
        "validation_accessed": False,
        "test_accessed": False,
        "formal_scientific_evidence": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "replay_passed": replay["passed"],
        "seed2_learned_horizons": aggregates["learned_latency_tree_vehicle_s2"].get("horizon_counts"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
