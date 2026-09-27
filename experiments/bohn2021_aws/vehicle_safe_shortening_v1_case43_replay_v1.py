#!/usr/bin/env python3
"""Development replay of IMPROVED safe-shortening v1 on vehicle validation case43.

Purpose
-------
The frozen latency-tree candidate failed validation case43 after a single H35
choice; prior deterministic ablations showed that forcing H25 at that step
rescued the case and injecting H35 into the H25 path reproduced the failure.
The safe-shortening v1 amendment is intended to preserve the adaptive-horizon
idea while disallowing H>25 and extracting only shorter-horizon opportunities.

This script performs a small *development diagnostic* on the already-opened
validation case43 only.  It compares each safe-shortening seed against its
same-seed fixed-H25 comparator on that same case.  It does not generate or open a
fresh validation split and never accesses the sealed test.

Scientific status: not formal model-selection evidence; validation case43 was
already used for diagnosis.  Use this only to decide the next engineering or
fresh-development-validation step.
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

import vehicle_safe_shortening_v1_smoke_v3 as smoke_v3  # noqa:E402

v1 = smoke_v3.v1
ROOT = v1.ROOT
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v1"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE43_REPLAY_V1_20260927T000000Z.json"
MARKER = "vehicle-safe-shortening-v1-case43-replay-v1-20260927"
CASE_ID = 43
REPEATS = 1

# Route all reused helpers into this diagnostic directory and keep the v3
# summarize_episode accounting repair.  No controller, terminal, seed or policy
# rule is changed here.
v1.OUT_DIR = OUT_DIR
v1.MARKER = MARKER
v1.summarize_episode = smoke_v3.fixed_summarize_episode


def _tensorflow_preflight() -> Dict[str, Any]:
    try:
        import tensorflow as tf  # noqa:F401
    except BaseException as exc:
        return {
            "passed": False,
            "exception": repr(exc),
            "diagnosis": "TensorFlow is unavailable in this interpreter; rerun with run_experiment interpreter='legacy'.",
            "python": sys.version,
            "executable": sys.executable,
        }
    return {
        "passed": True,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "python": sys.version,
        "executable": sys.executable,
    }


def _write_backup_request(raw: Dict[str, Any]) -> None:
    v1.write_json(BACKUP_REQUEST, {
        "requested_utc": raw["created_utc"],
        "reason": "backup small development replay of safe-shortening v1 on previously opened vehicle validation case43",
        "artifacts": [
            v1.rel(OUT_DIR),
            v1.rel(Path(__file__).resolve()),
            v1.rel(Path(smoke_v3.__file__).resolve()),
            v1.rel(Path(v1.__file__).resolve()),
            v1.rel(v1.PROTOCOL),
        ],
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "sealed_test_accessed": False,
    })


def _aggregate_pair(episodes: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    fixed_id = "fixed_H25_vehicle_s%d" % seed
    adaptive_id = "safe_shortening_v1_vehicle_s%d" % seed
    fixed_eps = [e for e in episodes if e["arm_id"] == fixed_id]
    adaptive_eps = [e for e in episodes if e["arm_id"] == adaptive_id]
    if len(fixed_eps) != REPEATS or len(adaptive_eps) != REPEATS:
        raise RuntimeError("missing pair for seed %d" % seed)
    fixed = v1.aggregate(fixed_eps)
    adaptive = v1.aggregate(adaptive_eps)
    return {
        "seed": seed,
        "fixed_arm": fixed_id,
        "adaptive_arm": adaptive_id,
        "fixed": fixed,
        "adaptive": adaptive,
        "adaptive_unique_horizons": adaptive["unique_horizons"],
        "fixed_unique_horizons": fixed["unique_horizons"],
        "adaptive_used_shorter_than_25": any(int(h) < 25 for h in adaptive["unique_horizons"]),
        "adaptive_used_above_25": any(int(h) > 25 for h in adaptive["unique_horizons"]),
        "success_delta_adaptive_minus_fixed": int(adaptive["success_count"] - fixed["success_count"]),
        "failure_delta_adaptive_minus_fixed": int(adaptive["episode_failure_count"] - fixed["episode_failure_count"]),
        "physical_constraint_delta_adaptive_minus_fixed": float(adaptive["physical_constraint_cost_sum"] - fixed["physical_constraint_cost_sum"]),
        "total_cost_delta_adaptive_minus_fixed": float(adaptive["total_cost_sum"] - fixed["total_cost_sum"]),
        "decision_time_ratio_adaptive_over_fixed": (
            float(adaptive["decision_timing_s"]["mean"] / fixed["decision_timing_s"]["mean"])
            if fixed["decision_timing_s"].get("mean") else None
        ),
    }


def _write_summary(raw: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Vehicle safe-shortening v1 case43 development replay")
    lines.append("")
    lines.append("Created UTC: `%s`." % raw["created_utc"])
    lines.append("")
    lines.append("Development diagnostic only: reopens previously diagnosed validation case43; no fresh validation generation and no sealed-test access.")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    lines.append("- Episodes: `%d` / declared `%d`." % (raw["budget_actual"]["episodes"], raw["budget_declared"]["episodes_exact"]))
    lines.append("- Control steps: `%d` / upper bound `%d`." % (raw["budget_actual"]["control_steps"], raw["budget_declared"]["control_step_upper_bound"]))
    lines.append("- New gradient steps: `0`.")
    lines.append("- Validation case43 reopened for development diagnostic: `True`.")
    lines.append("- Sealed test accessed: `False`.")
    lines.append("")
    lines.append("## Same-seed case43 pairs")
    lines.append("")
    lines.append("| seed | fixed success/steps/cost | adaptive success/steps/cost | adaptive horizons | phys delta | decision ratio | notes |")
    lines.append("|---:|---|---|---|---:|---:|---|")
    for seed in v1.SEEDS:
        p = raw["paired"][str(seed)]
        f = p["fixed"]
        a = p["adaptive"]
        notes = []
        if p["adaptive_used_above_25"]:
            notes.append("UNSAFE H>25")
        if p["adaptive_used_shorter_than_25"]:
            notes.append("uses H<25")
        if p["failure_delta_adaptive_minus_fixed"] > 0:
            notes.append("more failures")
        if not notes:
            notes.append("no short dispatch")
        lines.append(
            "| %d | %d/%d/%.6g | %d/%d/%.6g | %s | %.6g | %s | %s |" % (
                seed,
                f["success_count"], f["steps"], f["physical_constraint_cost_sum"],
                a["success_count"], a["steps"], a["physical_constraint_cost_sum"],
                a["horizon_counts"],
                p["physical_constraint_delta_adaptive_minus_fixed"],
                "%.6g" % p["decision_time_ratio_adaptive_over_fixed"] if p["decision_time_ratio_adaptive_over_fixed"] is not None else "NA",
                "; ".join(notes),
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This diagnostic tests whether the safe-shortening wrapper avoids the previously implicated H35 failure mode on the contaminated case43. Passing here is necessary engineering evidence, not sufficient validation evidence. A fresh dev-validation bank remains required after backup verification.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    v1.assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    v1.write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_case43_development_replay",
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "test_accessed": False,
        "case_id": CASE_ID,
    })
    runtime_preflight = _tensorflow_preflight()
    v1.write_json(OUT_DIR / "runtime_preflight.json", runtime_preflight)
    if not runtime_preflight.get("passed"):
        raise RuntimeError(runtime_preflight["diagnosis"] + " " + runtime_preflight.get("exception", ""))

    v1.latency_verify()
    assert v1.PROTOCOL.exists(), "safe-shortening protocol must exist"
    bank = v1.bank_name(v1.TASK, "validation")
    banks_done = v1.read_json(bank.parent / "completed.json")
    assert v1.sha256(bank) == banks_done["hashes"][str(bank)]
    bank_data = v1.read_json(bank)
    assert bank_data["task"] == v1.TASK and bank_data["split"] == "validation"
    assert CASE_ID < len(bank_data["cases"])

    arms_all = v1.build_arms()
    arms = [a for a in arms_all if a["family"] in ("fixed_H25", "safe_shortening_v1")]
    # Deterministic compact schedule: fixed/adaptive pairs per seed.  One repeat is
    # enough because the v3 smoke already established replay equality and this
    # diagnostic is causal/safety-oriented, not a timing-estimation run.
    schedule: List[Dict[str, Any]] = []
    idx = 0
    for seed in v1.SEEDS:
        for family in ("fixed_H25", "safe_shortening_v1"):
            arm_id = "%s_vehicle_s%d" % (family, seed)
            schedule.append({"episode_index": idx, "arm_id": arm_id, "case": CASE_ID, "repeat": 0})
            idx += 1
    v1.write_json(OUT_DIR / "schedule.json", {"episodes": schedule, "arms": arms, "case_id": CASE_ID, "repeats": REPEATS})

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
    for item in schedule:
        arm = arm_by_id[item["arm_id"]]
        summary = v1.run_episode(item, arm, bank_data["cases"][CASE_ID], terminals[arm["seed"]], terminal_load_s[arm["seed"]])
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(e["steps"] for e in episodes)),
            "last_episode": {k: summary[k] for k in ("episode_index", "arm_id", "case", "repeat", "steps", "success", "termination")},
            "validation_case43_reopened_for_development_diagnostic": True,
            "test_accessed": False,
        }
        v1.write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    control_steps = int(sum(e["steps"] for e in episodes))
    assert len(episodes) == len(schedule)
    assert control_steps <= 900
    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = v1.aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({"seed": arm["seed"], "family": arm["family"], "policy": arm["policy"], "policy_path": arm["policy_path"], "policy_sha256": arm["policy_sha256"]})
        aggregates[arm["arm_id"]] = agg
    paired = {str(seed): _aggregate_pair(episodes, seed) for seed in v1.SEEDS}
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_safe_shortening_v1_case43_development_replay_not_formal_validation",
        "formal_scientific_evidence": False,
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "validation64_full_bank_used_for_model_selection": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "case_id": CASE_ID,
        "split": "already_opened_validation_case43_development_diagnostic_only",
        "protocol": {"path": v1.rel(v1.PROTOCOL), "sha256": v1.sha256(v1.PROTOCOL)},
        "bank": {"path": v1.rel(bank), "sha256": v1.sha256(bank), "case_id": CASE_ID},
        "source_hashes": {
            v1.rel(Path(__file__).resolve()): v1.sha256(Path(__file__).resolve()),
            v1.rel(Path(smoke_v3.__file__).resolve()): v1.sha256(Path(smoke_v3.__file__).resolve()),
            v1.rel(Path(v1.__file__).resolve()): v1.sha256(Path(v1.__file__).resolve()),
            v1.rel(v1.REPRO / "gated_horizon_policy.py"): v1.sha256(v1.REPRO / "gated_horizon_policy.py"),
            v1.rel(v1.REPRO / "latency_tree_protocol.py"): v1.sha256(v1.REPRO / "latency_tree_protocol.py"),
            v1.rel(v1.REPRO / "conservative_canonical_reset.py"): v1.sha256(v1.REPRO / "conservative_canonical_reset.py"),
            v1.rel(v1.REPRO / "conservative_solver_recovery.py"): v1.sha256(v1.REPRO / "conservative_solver_recovery.py"),
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "budget_declared": {"episodes_exact": len(schedule), "control_step_upper_bound": 900, "new_gradient_steps": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "resets": int(sum(e["resets_metered"] for e in episodes)), "environment_constructions": len(episodes), "new_gradient_steps": 0, "sealed_test_episodes": 0},
        "arms": arms,
        "schedule": schedule,
        "terminal_sources": terminal_hashes,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired": paired,
        "interpretation_limits": [
            "case43 was already opened for development diagnosis and cannot be fresh independent validation",
            "one deterministic repeat per arm is not a timing-estimation campaign",
            "safe-shortening success here would not by itself support a final-test gate",
        ],
    }
    v1.write_json(OUT_DIR / "raw.json", raw)
    _write_summary(raw)
    _write_backup_request(raw)
    docs_snippet = """
<!-- vehicle-safe-shortening-v1-case43-replay-v1-20260927 -->
## 2026-09-27 vehicle safe-shortening v1 case43 development replay

UTC: {created}. Ran 6 development diagnostic episodes on already-opened vehicle validation case43: same-seed fixed-H25 and safe-shortening-v1 arms for seeds 0/1/2. Budget: {episodes} episodes, {steps} control steps, 0 gradient steps. Fresh validation generation: false. Sealed test: closed.

Headline pairs: {pairs}

Artifacts: `{summary}`, `{raw}`, completed marker `{completed}`. Backup requested at `{backup}`.
""".format(
        created=raw["created_utc"],
        episodes=raw["budget_actual"]["episodes"],
        steps=raw["budget_actual"]["control_steps"],
        pairs={k: {
            "fixed_success": v["fixed"]["success_count"],
            "adaptive_success": v["adaptive"]["success_count"],
            "adaptive_horizons": v["adaptive"]["horizon_counts"],
            "phys_delta": v["physical_constraint_delta_adaptive_minus_fixed"],
            "decision_ratio": v["decision_time_ratio_adaptive_over_fixed"],
            "used_shorter": v["adaptive_used_shorter_than_25"],
            "used_above_25": v["adaptive_used_above_25"],
        } for k, v in raw["paired"].items()},
        summary=v1.rel(OUT_DIR / "summary.md"),
        raw=v1.rel(OUT_DIR / "raw.json"),
        completed=v1.rel(OUT_DIR / "completed.json"),
        backup=v1.rel(BACKUP_REQUEST),
    )
    for doc in (ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "RESULTS_AUDIT.md"):
        with doc.open("a", encoding="utf-8") as f:
            f.write("\n" + docs_snippet + "\n")
    decision_snippet = """
<!-- vehicle-safe-shortening-v1-case43-replay-v1-decision-20260927 -->
### Decision: use case43 replay only as contaminated development evidence for safe-shortening v1

Before evidence: validation64 learned_s2 failed case43 after one H35 action; deterministic ablation indicated the H35 intervention was causal. The safe-shortening v1 smoke passed and showed no H>25 dispatch.

Change tested here: no algorithm or controller change beyond the already frozen safe-shortening v1 wrapper; this replay merely applies same-seed fixed-H25 and safe-shortening policies to the already-opened case43.

Outcome: see `{summary}`. This remains non-formal development evidence. If it exposes a failure, revise before fresh dev-validation; if it passes, backup verification is still required before generating the fresh devval64 bank.
""".format(summary=v1.rel(OUT_DIR / "summary.md"))
    with (ROOT / "DECISIONS.md").open("a", encoding="utf-8") as f:
        f.write("\n" + decision_snippet + "\n")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [BACKUP_REQUEST, v1.PROTOCOL, Path(__file__).resolve(), Path(smoke_v3.__file__).resolve(), Path(v1.__file__).resolve()]
    v1.write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hashes": {v1.rel(p): v1.sha256(p) for p in sorted(set(files))},
        "backup_request": v1.rel(BACKUP_REQUEST),
        "formal_scientific_evidence": False,
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "test_accessed": False,
        "headline_pairs": {k: {
            "fixed_success": v["fixed"]["success_count"],
            "adaptive_success": v["adaptive"]["success_count"],
            "adaptive_horizons": v["adaptive"]["horizon_counts"],
            "physical_constraint_delta_adaptive_minus_fixed": v["physical_constraint_delta_adaptive_minus_fixed"],
            "decision_time_ratio_adaptive_over_fixed": v["decision_time_ratio_adaptive_over_fixed"],
            "adaptive_used_shorter_than_25": v["adaptive_used_shorter_than_25"],
            "adaptive_used_above_25": v["adaptive_used_above_25"],
        } for k, v in raw["paired"].items()},
    })
    print(json.dumps({
        "completed": v1.rel(OUT_DIR / "completed.json"),
        "summary": v1.rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "case_id": CASE_ID,
        "validation_case43_reopened_for_development_diagnostic": True,
        "test_accessed": False,
        "paired": raw["paired"],
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
            "validation_case43_reopened_for_development_diagnostic": True,
            "fresh_validation_bank_generated": False,
            "test_accessed": False,
            "next_recovery_hint": "Inspect only this failure and minimal new artifacts; preserve failed output and version any fix.",
        })
        raise
