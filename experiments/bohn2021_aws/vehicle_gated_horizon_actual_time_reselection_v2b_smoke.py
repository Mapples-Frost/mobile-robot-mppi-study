#!/usr/bin/env python3
"""Vehicle gated-horizon actual-time-aware re-selection V2b smoke.

Engineering-only smoke for the V2b metadata repair result.  It executes the
V2b selected policy per seed (seed0 fixed H25 fallback; seeds1/2 adaptive
h15_p1_g5) with the same common safe wrapper used by prior gated-horizon
smokes, plus current stored gated-policy and same-seed fixed-H25 comparators.

No training, no validation64-bank access/reopen, and no sealed final-test
access.  This script is intended to run only after external backup of the V2b
repair/source/protocol artifacts has been verified.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

# Reuse the already-audited smoke machinery (environment construction, trace
# audit, deterministic replay, timing accounting) but do not reuse its risk-v1
# policy selection or reporting labels.
import vehicle_gated_horizon_risk_reselection_v1_smoke as base  # noqa:E402

v1 = base.v1
ROOT = base.ROOT
TASK = "vehicle"
SEEDS = (0, 1, 2)
BASE_H = 25
CASES_EXPECTED = 2
REPEATS = 2
MAX_STEPS = 150
BANK_RNG = 2609285100
ORDER_SEED = 2609285200
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_gated_horizon_actual_time_reselection_v2b_smoke_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_protocol_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_protocol_20260928.json"
V2B_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z/completed.json"
V2B_NOMINATED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z/nominated_policies.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-gated-horizon-actual-time-reselection-v2b-smoke-20260928"


def configure_base_globals() -> None:
    """Point reused base helpers at this smoke's paths/constants."""
    base.TASK = TASK
    base.SEEDS = SEEDS
    base.BASE_H = BASE_H
    base.CASES_EXPECTED = CASES_EXPECTED
    base.REPEATS = REPEATS
    base.MAX_STEPS = MAX_STEPS
    base.OUT_DIR = OUT_DIR
    base.BANK_DIR = BANK_DIR
    base.BANK_PATH = BANK_PATH
    base.BANK_COMPLETED = BANK_COMPLETED
    base.PROTOCOL = PROTOCOL
    base.PROTOCOL_JSON = PROTOCOL_JSON
    base.common_safe_decide = common_safe_decide  # global lookup used by base.run_episode


def rel(path: Path) -> str:
    return base.rel(path)


def read_json(path: Path) -> Any:
    return base.read_json(path)


def write_json(path: Path, value: Any) -> None:
    return base.write_json(path, value)


def sha256(path: Path) -> str:
    return base.sha256(path)


def canonical_sha(value: Any) -> str:
    return base.canonical_sha(value)


def verify_completed_marker(path: Path) -> Dict[str, Any]:
    return base.verify_completed_marker(path)


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    if (OUT_DIR / "completed.json").exists():
        verify_completed_marker(OUT_DIR / "completed.json")
        raise SystemExit("actual-time V2b smoke already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise RuntimeError("Partial actual-time V2b smoke output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10]))


def generate_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        verify_completed_marker(BANK_COMPLETED)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "actual_time_reselection_v2b_smoke" or len(bank.get("cases", [])) != CASES_EXPECTED:
            raise RuntimeError("Existing actual-time V2b smoke bank metadata mismatch")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": CASES_EXPECTED}
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, gen_dir)
    recovery = v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(BANK_RNG)
    np.random.seed(BANK_RNG)
    cases: List[Dict[str, Any]] = []
    for cid in range(CASES_EXPECTED):
        recovery.update(enabled=False, events=[], case=cid, step=-1)
        env.reset()
        cases.append(base.snapshot(env))
    if counts["step_calls"] != 0 or counts["reset_calls"] != CASES_EXPECTED:
        raise RuntimeError("Unexpected bank generation counts: %r" % counts)
    bank = {
        "task": TASK,
        "split": "actual_time_reselection_v2b_smoke",
        "rng": BANK_RNG,
        "cases": cases,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "fresh engineering smoke only for IMPROVED vehicle actual-time-aware gated-horizon V2b selection",
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "task": TASK,
        "split": "actual_time_reselection_v2b_smoke",
        "rng": BANK_RNG,
        "cases": CASES_EXPECTED,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "new_scored_control_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": CASES_EXPECTED, "reset_calls": counts["reset_calls"]}


def load_v2b_nominations() -> Dict[int, Dict[str, Any]]:
    completed = read_json(V2B_COMPLETED)
    if completed.get("passed") is not True or completed.get("acceptance_for_smoke_met") is not True:
        raise RuntimeError("V2b completed marker is not passed/accepted for smoke: %s" % rel(V2B_COMPLETED))
    data = read_json(V2B_NOMINATED)
    if data.get("acceptance_for_smoke_met") is not True:
        raise RuntimeError("V2b nomination file is not accepted for smoke")
    expected_ids = {0: "fixed", 1: "h15_p1_g5", 2: "h15_p1_g5"}
    out: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        seed_block = data["nominations"][str(seed)]
        row = seed_block["nominated"]
        cid = str(row["candidate_id"])
        if cid != expected_ids[seed]:
            raise RuntimeError("Unexpected V2b nominated id for seed %d: %s" % (seed, cid))
        if cid == "fixed" or bool(seed_block.get("fallback_to_fixed")):
            policy = {"id": "fixed", "task": TASK, "h": BASE_H, "training_seed": int(seed), "fallback_reason": "actual_time_v2b_no_eligible_adaptive_candidate"}
        else:
            if not bool(row.get("actual_time_primary_eligible")):
                raise RuntimeError("Adaptive V2b nomination is not primary eligible for seed %d" % seed)
            policy = {
                "id": cid,
                "task": TASK,
                "short_h": int(row["short_h"]),
                "profile": int(row["profile"]),
                "guard": int(row["guard"]),
                "training_seed": int(seed),
                "v2b_estimated_net_time_saving_fraction": row.get("estimated_net_time_saving_fraction"),
                "v2b_timing_ratio_median_vs_H25": row.get("timing_ratio_median_vs_H25"),
            }
        out[seed] = policy
    return out


def build_arms() -> List[Dict[str, Any]]:
    nominations = load_v2b_nominations()
    arms: List[Dict[str, Any]] = []
    for seed in SEEDS:
        selected_policy = nominations[seed]
        current_policy = v1.load_old_policy(seed)
        selected_role = "actual_time_v2b_selected_fixed_H25_fallback" if selected_policy.get("id") == "fixed" else "actual_time_v2b_selected_adaptive_candidate"
        arms.append({
            "arm_id": "actual_time_v2b_selected_vehicle_s%d" % seed,
            "role": selected_role,
            "family": "IMPROVED_actual_time_reselected_v2b_common_safe_wrapper",
            "seed": int(seed),
            "policy": selected_policy,
            "policy_path": rel(V2B_NOMINATED),
            "policy_sha256": canonical_sha(selected_policy),
        })
        arms.append({
            "arm_id": "current_gated_vehicle_s%d" % seed,
            "role": "current_stored_gated_policy_comparator",
            "family": "IMPROVED_current_historical_gated_policy_common_safe_wrapper",
            "seed": int(seed),
            "policy": current_policy,
            "policy_path": rel(v1.old_policy_path(seed)),
            "policy_sha256": sha256(v1.old_policy_path(seed)),
        })
        arms.append({
            "arm_id": "fixed_H25_vehicle_s%d" % seed,
            "role": "same_seed_fixed_H25_comparator",
            "family": "fixed_H25_primary_same_seed",
            "seed": int(seed),
            "policy": {"id": "fixed", "task": TASK, "h": BASE_H},
            "policy_path": None,
            "policy_sha256": canonical_sha({"id": "fixed", "task": TASK, "h": BASE_H}),
        })
    if len(arms) != 9:
        raise RuntimeError("Unexpected arm count")
    return arms


def common_safe_decide(arm: Mapping[str, Any], ctx: Dict[str, Any], previous_initial_failure: bool,
                       previous_final_failure: bool) -> Tuple[int, Dict[str, Any]]:
    policy = arm.get("policy") or {}
    if arm["role"] == "same_seed_fixed_H25_comparator" or policy.get("id") == "fixed":
        return BASE_H, {
            "kind": "fixed" if arm["role"] == "same_seed_fixed_H25_comparator" else "actual_time_v2b_fixed_fallback",
            "raw_horizon": BASE_H,
            "selected_horizon": BASE_H,
            "fallback": arm["role"] != "same_seed_fixed_H25_comparator",
            "reason": "same_seed_H25" if arm["role"] == "same_seed_fixed_H25_comparator" else "actual_time_v2b_seed_fallback_to_fixed_H25",
            "policy_role": arm["role"],
        }
    if previous_initial_failure or previous_final_failure:
        return BASE_H, {
            "kind": "common_safe_gated_wrapper",
            "fallback": True,
            "reason": "previous_solver_failure",
            "raw_horizon": BASE_H,
            "selected_horizon": BASE_H,
            "policy_role": arm["role"],
        }
    raw_h, gate = v1.gated_decide(policy, ctx)
    raw_h = int(raw_h)
    if raw_h not in range(5, 51, 5):
        raise RuntimeError("unsafe non-grid horizon requested: %r" % raw_h)
    selected = min(raw_h, BASE_H)
    return int(selected), {
        "kind": "common_safe_gated_wrapper",
        "fallback": False,
        "raw_horizon": raw_h,
        "selected_horizon": int(selected),
        "clamped_to_base": raw_h > BASE_H,
        "policy_role": arm["role"],
        "gate": gate,
    }


def compare(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        selected = aggregates["actual_time_v2b_selected_vehicle_s%d" % seed]
        current = aggregates["current_gated_vehicle_s%d" % seed]
        fixed = aggregates["fixed_H25_vehicle_s%d" % seed]
        out[str(seed)] = {
            "selected_unique_horizons": selected["unique_horizons"],
            "current_unique_horizons": current["unique_horizons"],
            "fixed_unique_horizons": fixed["unique_horizons"],
            "selected_adapted_below_H25": selected["adapted_below_H25"],
            "current_adapted_below_H25": current["adapted_below_H25"],
            "selected_unsafe_above_H25_dispatch": selected["unsafe_above_H25_dispatch"],
            "current_unsafe_above_H25_dispatch": current["unsafe_above_H25_dispatch"],
            "selected_minus_fixed": base.arm_delta(selected, fixed),
            "selected_minus_current": base.arm_delta(selected, current),
            "current_minus_fixed": base.arm_delta(current, fixed),
        }
    return out


def smoke_pass_checks(raw: Mapping[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["runtime_preflight_passed"] = bool(raw["runtime_preflight"].get("passed"))
    checks["episode_count_ok"] = raw["budget_actual"]["episodes"] == raw["budget_declared"]["episodes_exact"]
    checks["control_step_bound_ok"] = raw["budget_actual"]["control_steps"] <= raw["budget_declared"]["control_step_upper_bound"]
    checks["replay_passed"] = bool(raw["replay"].get("passed"))
    aggregates = raw["aggregates"]
    checks["no_selected_or_current_above_H25_dispatch"] = all(
        not a.get("unsafe_above_H25_dispatch") for a in aggregates.values() if a.get("role") != "same_seed_fixed_H25_comparator"
    )
    safety_ok = True
    safety_failures: List[Dict[str, Any]] = []
    for seed in SEEDS:
        selected = aggregates["actual_time_v2b_selected_vehicle_s%d" % seed]
        fixed = aggregates["fixed_H25_vehicle_s%d" % seed]
        comparisons = [
            ("success_count", selected["success_count"] < fixed["success_count"]),
            ("constraint_count", selected["constraint_count"] > fixed["constraint_count"]),
            ("initial_failed_steps", selected["initial_failed_steps"] > fixed["initial_failed_steps"]),
            ("solver_failure_steps", selected["solver_failure_steps"] > fixed["solver_failure_steps"]),
        ]
        bad = [name for name, failed in comparisons if failed]
        if bad:
            safety_ok = False
            safety_failures.append({"seed": seed, "failed_checks": bad})
    checks["selected_not_worse_than_fixed_smoke_safety_counts"] = safety_ok
    checks["selected_safety_failures"] = safety_failures
    checks["actual_time_v2b_selected_below_H25_seed_count"] = sum(
        1 for seed in SEEDS if aggregates["actual_time_v2b_selected_vehicle_s%d" % seed].get("adapted_below_H25")
    )
    checks["diagnostic_expectation_two_or_more_selected_seeds_adapted"] = checks["actual_time_v2b_selected_below_H25_seed_count"] >= 2
    checks["selected_vs_fixed_decision_ratios_recorded"] = all(
        raw["paired_comparisons"][str(seed)]["selected_minus_fixed"].get("decision_time_ratio") is not None for seed in SEEDS
    )
    hard_keys = [
        "runtime_preflight_passed", "episode_count_ok", "control_step_bound_ok", "replay_passed",
        "no_selected_or_current_above_H25_dispatch", "selected_not_worse_than_fixed_smoke_safety_counts",
        "selected_vs_fixed_decision_ratios_recorded",
    ]
    checks["hard_pass"] = all(bool(checks[k]) for k in hard_keys)
    return checks


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle gated-horizon actual-time-aware re-selection V2b smoke",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Engineering smoke only on a fresh actual-time V2b smoke bank. No validation64 access and no sealed-test access.",
        "",
        f"Protocol: `{raw['protocol']['path']}` sha256 `{raw['protocol']['sha256']}`.",
        f"V2b input: `{raw['actual_time_reselection']['completed_path']}` sha256 `{raw['actual_time_reselection']['completed_sha256']}`.",
        "",
        "## Budget/access",
        "",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- New training episodes / gradient steps: `{raw['budget_actual']['new_training_episodes']}` / `{raw['budget_actual']['new_gradient_steps']}`.",
        f"- Bank generation resets: `{raw['budget_actual']['bank_generation_resets']}`.",
        f"- Replay passed: `{raw['replay']['passed']}` over `{raw['replay']['pairs_checked']}` repeat pairs.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Hard smoke checks",
        "",
    ]
    for key, value in raw["smoke_pass_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Arm aggregates",
        "",
        "| arm | seed | role | episodes | steps | success | failures | constraints | init-fail | final-fail | physical+constraint | total | decision mean s/step | horizons | raw horizons | fallbacks |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for arm_id in sorted(raw["aggregates"]):
        a = raw["aggregates"][arm_id]
        lines.append(
            f"| `{arm_id}` | {a['seed']} | {a['role']} | {a['episodes']} | {a['steps']} | {a['success_count']} | "
            f"{a['episode_failure_count']} | {a['constraint_count']} | {a['initial_failed_steps']} | {a['solver_failure_steps']} | "
            f"{a['physical_constraint_cost_sum']:.6g} | {a['total_cost_sum']:.6g} | {a['decision_mean_s_per_step']:.6g} | "
            f"{a['horizon_counts']} | {a['raw_horizon_counts_before_clamp']} | {a['solver_failure_fallback_steps']} |"
        )
    lines += ["", "## Same-seed smoke deltas", ""]
    for seed, cmp in raw["paired_comparisons"].items():
        sf = cmp["selected_minus_fixed"]
        sc = cmp["selected_minus_current"]
        cf = cmp["current_minus_fixed"]
        lines.append(
            f"- seed {seed}: selected horizons={cmp['selected_unique_horizons']}, current horizons={cmp['current_unique_horizons']}; "
            f"selected-fixed physical Δ={sf['physical_constraint_delta']:.6g}, total Δ={sf['total_cost_delta']:.6g}, decision ratio={sf['decision_time_ratio']:.6g}; "
            f"selected-current physical Δ={sc['physical_constraint_delta']:.6g}, total Δ={sc['total_cost_delta']:.6g}, decision ratio={sc['decision_time_ratio']:.6g}; "
            f"current-fixed physical Δ={cf['physical_constraint_delta']:.6g}, total Δ={cf['total_cost_delta']:.6g}, decision ratio={cf['decision_time_ratio']:.6g}."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is an implementation smoke only. It can justify scheduling a fresh development-validation comparison after backup, but it is not independent validation/model-selection evidence and cannot support a reproduction or improved-method success claim. H distribution is reported as behavior, not as timing proof; measured decision/solver timings are retained in raw traces.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(raw: Dict[str, Any]) -> None:
    stamp = raw["created_utc"].replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_ACTUAL_TIME_RESELECTION_V2B_SMOKE_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup actual-time V2b smoke protocol/source/artifacts before any larger fresh development validation",
        "artifacts": [rel(OUT_DIR), rel(PROTOCOL), rel(PROTOCOL_JSON), rel(Path(__file__).resolve()), rel(Path(base.__file__).resolve()), rel(V2B_COMPLETED), rel(V2B_NOMINATED)],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    raw["backup_request"] = rel(path)


def append_docs(raw: Mapping[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle actual-time-aware gated-horizon V2b smoke\n\n"
        f"UTC: {raw['created_utc']}. IMPROVED actual-time-aware gated-horizon V2b smoke completed on a fresh engineering bank: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, hard_pass={raw['smoke_pass_checks']['hard_pass']}, "
        f"selected_below_H25_seed_count={raw['smoke_pass_checks']['actual_time_v2b_selected_below_H25_seed_count']}. "
        "No validation64 bank or sealed test was opened. This is not model-selection/final evidence and does not modify any frozen devval campaign. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(), Path(base.__file__).resolve(), Path(v1.__file__).resolve(), PROTOCOL, PROTOCOL_JSON, V2B_COMPLETED, V2B_NOMINATED,
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/relative_policy_features.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def main() -> int:
    configure_base_globals()
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED_vehicle_gated_horizon_actual_time_reselection_v2b_smoke",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    preflight = base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise RuntimeError(preflight["diagnosis"] + " " + preflight.get("exception", ""))
    v1.latency_verify()
    for p in (PROTOCOL, PROTOCOL_JSON, V2B_COMPLETED, V2B_NOMINATED):
        if not p.exists():
            raise RuntimeError("Required frozen input missing: %s" % rel(p))
    bank_info = generate_bank_if_needed()
    bank_data = read_json(BANK_PATH)
    arms = build_arms()
    schedule = base.randomized_schedule(arms)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "arms": arms})
    terminal_hashes: Dict[str, Any] = {}
    terminals: Dict[int, Tuple[Any, Any]] = {}
    terminal_load_s: Dict[int, float] = {}
    for seed in SEEDS:
        start = time.perf_counter()
        terminals[seed] = v1.load_terminal(seed, terminal_hashes)
        terminal_load_s[seed] = time.perf_counter() - start
    write_json(OUT_DIR / "terminal_sources.json", terminal_hashes)
    arm_by_id = {a["arm_id"]: a for a in arms}
    episodes: List[Dict[str, Any]] = []
    trace_index: Dict[Tuple[str, int, int], Path] = {}
    replay = {"passed": True, "pairs_checked": 0, "mismatches": []}
    for item in schedule:
        arm = arm_by_id[item["arm_id"]]
        summary = base.run_episode(item, arm, bank_data["cases"][item["case"]], terminals[int(arm["seed"])], terminal_load_s[int(arm["seed"])])
        episodes.append(summary)
        trace_path = ROOT / summary["path"] / "trace.json"
        trace_index[(item["arm_id"], item["case"], item["repeat"])] = trace_path
        if item["repeat"] == 1:
            base_path = trace_index[(item["arm_id"], item["case"], 0)]
            replay["pairs_checked"] += 1
            if base.clean_trace(read_json(base_path)) != base.clean_trace(read_json(trace_path)):
                replay["passed"] = False
                replay["mismatches"].append({"arm_id": item["arm_id"], "case": item["case"], "repeat0": rel(base_path), "repeat1": rel(trace_path)})
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(e["steps"] for e in episodes)),
            "last_episode": {k: summary[k] for k in ("episode_index", "arm_id", "case", "repeat", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    if not replay["passed"]:
        raise RuntimeError("Repeat replay mismatch: %r" % replay["mismatches"][:3])
    control_steps = int(sum(e["steps"] for e in episodes))
    if len(episodes) != len(arms) * CASES_EXPECTED * REPEATS or control_steps > len(arms) * CASES_EXPECTED * REPEATS * MAX_STEPS:
        raise RuntimeError("Smoke budget exceeded or incomplete")
    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = base.aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({
            "seed": int(arm["seed"]), "family": arm["family"], "role": arm["role"],
            "policy": arm["policy"], "policy_path": arm["policy_path"], "policy_sha256": arm["policy_sha256"],
        })
        aggregates[arm["arm_id"]] = agg
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_vehicle_gated_horizon_actual_time_reselection_v2b_smoke_not_original_SAC",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": "fresh_engineering_smoke_bank_only_actual_time_reselection_v2b",
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "actual_time_reselection": {"completed_path": rel(V2B_COMPLETED), "completed_sha256": sha256(V2B_COMPLETED), "nominated_path": rel(V2B_NOMINATED), "nominated_sha256": sha256(V2B_NOMINATED)},
        "bank": bank_info,
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "method_change": {
            "from": "risk-first and current stored gated policies selected without a measured-time primary objective",
            "to": "actual-time-aware metadata re-selected V2b policies: seed0 fixed H25 fallback, seed1/seed2 h15_p1_g5",
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "v2b_seed0_overhead_repair_applied": True,
        },
        "budget_declared": {"episodes_exact": len(arms) * CASES_EXPECTED * REPEATS, "control_step_upper_bound": len(arms) * CASES_EXPECTED * REPEATS * MAX_STEPS, "bank_generation_resets": CASES_EXPECTED, "new_training_episodes": 0, "new_gradient_steps": 0, "development_validation_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "resets": int(sum(e["resets_metered"] for e in episodes)), "environment_constructions": len(episodes), "bank_generation_resets": CASES_EXPECTED if bank_info.get("created_now") else 0, "new_training_episodes": 0, "new_gradient_steps": 0, "development_validation_episodes": 0, "sealed_test_episodes": 0},
        "arms": arms,
        "randomized_schedule": schedule,
        "terminal_sources": terminal_hashes,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired_comparisons": compare(aggregates),
        "replay": replay,
        "interpretation_limits": ["engineering smoke only", "not validation/model selection", "not ORIGINAL SAC", "finite candidate re-selection not neural training", "H distribution is not timing evidence", "sealed final test remains closed"],
    }
    raw["smoke_pass_checks"] = smoke_pass_checks(raw)
    write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], PROTOCOL, PROTOCOL_JSON, Path(__file__).resolve(), Path(base.__file__).resolve(), Path(v1.__file__).resolve(), V2B_COMPLETED, V2B_NOMINATED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": bool(raw["smoke_pass_checks"]["hard_pass"]),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
        "backup_request": raw["backup_request"],
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "headline": {"control_steps": control_steps, "smoke_pass_checks": raw["smoke_pass_checks"], "paired_comparisons": raw["paired_comparisons"]},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "hard_pass": raw["smoke_pass_checks"]["hard_pass"],
        "smoke_pass_checks": raw["smoke_pass_checks"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "paired_comparisons": raw["paired_comparisons"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        configure_base_globals()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "next_recovery_hint": "Inspect failure, preserve this directory, and create a versioned one-variable recovery if necessary. Use legacy interpreter for TF1 runtime.",
        })
        raise
