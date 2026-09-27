#!/usr/bin/env python3
"""Collate preserved v2 case43 replay episodes for safe-shortening v1.

Context
-------
`vehicle_safe_shortening_v1_case43_replay_v2.py` repaired the v1 arm-filter
bug and successfully ran all six intended already-opened validation case43
episodes (fixed-H25 and safe-shortening-v1 for seeds 0/1/2).  It then failed in
post-processing with `KeyError('decision_timing_s')` because `_aggregate_pair()`
expected the per-step timing-summary key on the aggregate object returned by the
shared smoke helper.  The shared aggregate schema instead exposes
`decision_total_s` and `decision_mean_s_per_step`.

This v3 source performs a collation-only recovery from the already-written v2
episode summaries and completed markers.  It writes a fresh v3 output directory
and does not rerun control simulations, load TensorFlow, open fresh validation
banks, or access sealed test data.

Scientific status
-----------------
This remains contaminated development evidence: validation case43 had already
been opened for diagnosis before safe-shortening v1.  Passing this collation is
an engineering precondition for backup-gated fresh development-validation, not
independent validation or final-test evidence.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
V2_OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v2"
V1_OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v1"
V2_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260927T081434_04d04181/registry.json"
V1_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260927T080923_12435ebb/registry.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v3_collate"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE43_REPLAY_V2_FAILURE_AND_V3_COLLATION_20260927T000000Z.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_safe_shortening_v1_protocol_20260927.md"
V1_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case43_replay_v1.py"
V2_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_case43_replay_v2.py"
SMOKE_V1_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_smoke.py"
SMOKE_V3_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_smoke_v3.py"
CASE_ID = 43
SEEDS = (0, 1, 2)
MARKER = "vehicle-safe-shortening-v1-case43-replay-v3-collate-20260927"

EXPECTED_EPISODES = [
    (0, "fixed_H25_vehicle_s0", "ep000_r0_fixed_H25_vehicle_s0_case43"),
    (0, "safe_shortening_v1_vehicle_s0", "ep001_r0_safe_shortening_v1_vehicle_s0_case43"),
    (1, "fixed_H25_vehicle_s1", "ep002_r0_fixed_H25_vehicle_s1_case43"),
    (1, "safe_shortening_v1_vehicle_s1", "ep003_r0_safe_shortening_v1_vehicle_s1_case43"),
    (2, "fixed_H25_vehicle_s2", "ep004_r0_fixed_H25_vehicle_s2_case43"),
    (2, "safe_shortening_v1_vehicle_s2", "ep005_r0_safe_shortening_v1_vehicle_s2_case43"),
]


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


def file_record(path: Path) -> Dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "sha256": sha256(path) if path.exists() and path.is_file() else None,
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        done = read_json(completed)
        if done.get("passed") is True:
            for name, digest in done.get("hashes", {}).items():
                path = ROOT / name
                assert path.exists(), name
                assert sha256(path) == digest, name
            raise SystemExit("v3 collation already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    assert not leftovers, "Partial v3 collation output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10])


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None}
    data_sorted = sorted(data)
    n = len(data_sorted)
    median = data_sorted[n // 2] if n % 2 else 0.5 * (data_sorted[n // 2 - 1] + data_sorted[n // 2])
    if n == 1:
        p95 = data_sorted[0]
    else:
        pos = 0.95 * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        p95 = data_sorted[lo] * (1.0 - frac) + data_sorted[hi] * frac
    return {"count": n, "sum": float(math.fsum(data_sorted)), "mean": float(math.fsum(data_sorted) / n), "median": float(median), "p95": float(p95), "max": float(data_sorted[-1])}


def verify_episode_completed(ep_dir: Path) -> Dict[str, str]:
    completed = read_json(ep_dir / "completed.json")
    assert completed.get("passed") is True, rel(ep_dir / "completed.json")
    verified: Dict[str, str] = {}
    for name, digest in completed.get("hashes", {}).items():
        path = ROOT / name
        assert path.exists(), name
        actual = sha256(path)
        assert actual == digest, "%s expected %s got %s" % (name, digest, actual)
        verified[name] = actual
    return verified


def load_v2_episode_summaries() -> List[Dict[str, Any]]:
    assert V2_OUT.exists(), "v2 output directory missing"
    failure = read_json(V2_OUT / "failure.json")
    assert "KeyError('decision_timing_s')" in failure.get("exception", ""), failure.get("exception")
    assert failure.get("fresh_validation_bank_generated") is False
    assert failure.get("test_accessed") is False
    progress = read_json(V2_OUT / "progress.json")
    assert progress.get("episodes_done") == 6 and progress.get("control_steps_done") == 576, progress
    assert progress.get("fresh_validation_bank_generated") is False
    assert progress.get("test_accessed") is False
    schedule = read_json(V2_OUT / "schedule.json")
    assert len(schedule.get("episodes", [])) == 6, schedule
    summaries: List[Dict[str, Any]] = []
    for seed, arm_id, dirname in EXPECTED_EPISODES:
        ep_dir = V2_OUT / "episodes" / dirname
        verify_episode_completed(ep_dir)
        summary = read_json(ep_dir / "summary.json")
        assert summary["seed"] == seed, (summary["seed"], seed)
        assert summary["arm_id"] == arm_id, (summary["arm_id"], arm_id)
        assert summary["case"] == CASE_ID, summary["case"]
        assert summary["repeat"] == 0, summary["repeat"]
        assert summary["steps_metered"] == summary["steps"], summary["arm_id"]
        assert (ep_dir / "trace.json").exists(), rel(ep_dir / "trace.json")
        summaries.append(summary)
    return summaries


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
    per_step_means: List[float] = []
    per_step_p95s: List[float] = []
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[str(h)] = horizons.get(str(h), 0) + int(n)
        for h, n in (e.get("raw_horizon_counts_before_clamp") or {}).items():
            raw_horizons[str(h)] = raw_horizons.get(str(h), 0) + int(n)
        if e.get("decision_timing_s", {}).get("mean") is not None:
            per_step_means.append(float(e["decision_timing_s"]["mean"]))
        if e.get("decision_timing_s", {}).get("p95") is not None:
            per_step_p95s.append(float(e["decision_timing_s"]["p95"]))
    out.update({
        "total_cost_mean_episode": out["total_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "physical_constraint_cost_mean_episode": out["physical_constraint_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "decision_mean_s_per_step": out["decision_total_s"] / out["steps"] if out["steps"] else None,
        "decision_gross_mean_s_per_step": out["decision_gross_total_s"] / out["steps"] if out["steps"] else None,
        "decision_episode_mean_summary_s": values_summary(per_step_means),
        "decision_episode_p95_summary_s": values_summary(per_step_p95s),
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "adapted_below_H25": any(int(h) < 25 for h in horizons),
        "unsafe_above_H25_dispatch": any(int(h) > 25 for h in horizons),
    })
    return out


def aggregate_by_arm(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _, arm_id, _ in EXPECTED_EPISODES:
        eps = [e for e in episodes if e["arm_id"] == arm_id]
        assert len(eps) == 1, arm_id
        agg = aggregate(eps)
        agg.update({"seed": eps[0]["seed"], "family": eps[0]["family"]})
        out[arm_id] = agg
    return out


def paired(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for seed in SEEDS:
        fixed = aggregates["fixed_H25_vehicle_s%d" % seed]
        adaptive = aggregates["safe_shortening_v1_vehicle_s%d" % seed]
        ratio = adaptive["decision_mean_s_per_step"] / fixed["decision_mean_s_per_step"] if fixed.get("decision_mean_s_per_step") else None
        out[str(seed)] = {
            "seed": seed,
            "fixed_arm": "fixed_H25_vehicle_s%d" % seed,
            "adaptive_arm": "safe_shortening_v1_vehicle_s%d" % seed,
            "fixed": fixed,
            "adaptive": adaptive,
            "adaptive_unique_horizons": adaptive["unique_horizons"],
            "fixed_unique_horizons": fixed["unique_horizons"],
            "adaptive_used_shorter_than_25": adaptive["adapted_below_H25"],
            "adaptive_used_above_25": adaptive["unsafe_above_H25_dispatch"],
            "success_delta_adaptive_minus_fixed": int(adaptive["success_count"] - fixed["success_count"]),
            "failure_delta_adaptive_minus_fixed": int(adaptive["episode_failure_count"] - fixed["episode_failure_count"]),
            "physical_constraint_delta_adaptive_minus_fixed": float(adaptive["physical_constraint_cost_sum"] - fixed["physical_constraint_cost_sum"]),
            "total_cost_delta_adaptive_minus_fixed": float(adaptive["total_cost_sum"] - fixed["total_cost_sum"]),
            "decision_time_ratio_adaptive_over_fixed": ratio,
        }
    return out


def provenance() -> Dict[str, Any]:
    return {
        "v1_failure": {
            "out_dir": rel(V1_OUT),
            "run_registry": file_record(V1_RUN_REGISTRY),
            "failure": file_record(V1_OUT / "failure.json"),
            "diagnosis": "v1 failed before any case43 simulation due exact-family arm-filter bug",
        },
        "v2_failure": {
            "out_dir": rel(V2_OUT),
            "run_registry": file_record(V2_RUN_REGISTRY),
            "failure": file_record(V2_OUT / "failure.json"),
            "progress": file_record(V2_OUT / "progress.json"),
            "diagnosis": "v2 ran all six case43 episodes and failed only in final pair aggregation due aggregate schema mismatch KeyError('decision_timing_s')",
        },
        "v3_change": "collate-only recovery from v2 episode summaries; use aggregate decision_mean_s_per_step / decision_total_s schema; no new simulations and no controller/policy/terminal/case/seed/horizon/solver change",
    }


def write_summary(raw: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Vehicle safe-shortening v1 case43 development replay v3 collation")
    lines.append("")
    lines.append("Created UTC: `%s`." % raw["created_utc"])
    lines.append("")
    lines.append("Development diagnostic only: uses preserved v2 episode outputs from already-opened validation case43; no new control simulation, no fresh validation bank generation, and no sealed-test access.")
    lines.append("")
    lines.append("## Recovery")
    lines.append("")
    lines.append("- v1 failure: arm-filter diagnostic bug before any case43 episode.")
    lines.append("- v2 failure: all 6 episodes completed, then post-processing failed with `KeyError('decision_timing_s')`.")
    lines.append("- v3 change: collation only; compute paired ratios from aggregate `decision_mean_s_per_step`/`decision_total_s` fields.")
    lines.append("- Controller/policy/terminal/case/seed/solver/horizon rule changed: `False`.")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    lines.append("- New v3 simulations/control steps/gradient steps: `0 / 0 / 0`.")
    lines.append("- Preserved v2 diagnostic episodes/control steps collated: `%d / %d`." % (raw["v2_preserved_budget"]["episodes"], raw["v2_preserved_budget"]["control_steps"]))
    lines.append("- Validation case43 reopened in v2 for development diagnostic: `True`.")
    lines.append("- Fresh validation bank generated: `False`.")
    lines.append("- Sealed test accessed: `False`.")
    lines.append("")
    lines.append("## Same-seed case43 pairs")
    lines.append("")
    lines.append("| seed | fixed success/steps/phys+constraint | adaptive success/steps/phys+constraint | adaptive horizons | phys delta | decision ratio | notes |")
    lines.append("|---:|---|---|---|---:|---:|---|")
    for seed in SEEDS:
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
        if p["success_delta_adaptive_minus_fixed"] < 0:
            notes.append("fewer successes")
        if not notes:
            notes.append("no short dispatch")
        ratio = p["decision_time_ratio_adaptive_over_fixed"]
        lines.append("| %d | %d/%d/%.6g | %d/%d/%.6g | %s | %.6g | %s | %s |" % (
            seed,
            f["success_count"], f["steps"], f["physical_constraint_cost_sum"],
            a["success_count"], a["steps"], a["physical_constraint_cost_sum"],
            a["horizon_counts"],
            p["physical_constraint_delta_adaptive_minus_fixed"],
            "%.6g" % ratio if ratio is not None else "NA",
            "; ".join(notes),
        ))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The preserved v2 episodes support the narrow engineering hypothesis that safe-shortening v1 avoids the specific H35 dispatch mode on contaminated case43: every adaptive case43 episode succeeded in 96 steps, none used H>25, and seeds 0/1/2 used shorter horizons on 31/35/1 steps respectively. This is not fresh validation and does not establish timing superiority; the case43 replay is one deterministic diagnostic case with noisy per-step timing.")
    lines.append("")
    lines.append("Next: verify backup covering v2 failure/v3 collation artifacts, then run the frozen fresh safe-shortening development-validation block against the strong fixed-H grid; sealed test remains closed.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(raw: Dict[str, Any]) -> None:
    write_json(BACKUP_REQUEST, {
        "requested_utc": raw["created_utc"],
        "reason": "backup v2 case43 replay failure with completed episode outputs plus v3 collation source/artifacts before fresh vehicle dev-validation",
        "artifacts": [
            rel(V1_OUT),
            rel(V2_OUT),
            rel(OUT_DIR),
            rel(V1_RUN_REGISTRY),
            rel(V2_RUN_REGISTRY),
            rel(Path(__file__).resolve()),
            rel(V1_SOURCE),
            rel(V2_SOURCE),
            rel(SMOKE_V1_SOURCE),
            rel(SMOKE_V3_SOURCE),
            rel(PROTOCOL),
        ],
        "v3_new_simulations": 0,
        "v2_preserved_simulations_collated": raw["v2_preserved_budget"],
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "sealed_test_accessed": False,
    })


def append_docs(raw: Dict[str, Any]) -> None:
    headline = {
        seed: {
            "fixed_success": raw["paired"][str(seed)]["fixed"]["success_count"],
            "adaptive_success": raw["paired"][str(seed)]["adaptive"]["success_count"],
            "adaptive_horizons": raw["paired"][str(seed)]["adaptive"]["horizon_counts"],
            "phys_delta": raw["paired"][str(seed)]["physical_constraint_delta_adaptive_minus_fixed"],
            "decision_ratio": raw["paired"][str(seed)]["decision_time_ratio_adaptive_over_fixed"],
            "used_above_25": raw["paired"][str(seed)]["adaptive_used_above_25"],
        }
        for seed in SEEDS
    }
    text = (
        "\n<!-- %s -->\n"
        "## 2026-09-27 vehicle safe-shortening v1 case43 replay v3 collation\n\n"
        "UTC: %s. Collated preserved v2 episode outputs after v2 post-processing failure. "
        "New v3 simulations/control steps/gradient steps: 0/0/0; preserved v2 diagnostic budget collated: %d episodes, %d control steps. "
        "All six same-seed case43 episodes succeeded; no adaptive H>25 dispatch; adaptive horizons by seed: %s. "
        "This reuses already-opened validation case43 and is contaminated development evidence only, not formal validation or final test. "
        "Fresh validation bank remains unopened/uncreated; sealed test remains closed. Artifacts: `%s`, `%s`, `%s`; backup request `%s`.\n"
        % (
            MARKER,
            raw["created_utc"],
            raw["v2_preserved_budget"]["episodes"],
            raw["v2_preserved_budget"]["control_steps"],
            headline,
            rel(OUT_DIR / "summary.md"),
            rel(OUT_DIR / "raw.json"),
            rel(OUT_DIR / "completed.json"),
            rel(BACKUP_REQUEST),
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
        "### Decision: v3 collation recovers v2 case43 replay without rerunning simulations\n\n"
        "Before evidence: v2 repaired the v1 arm-filter bug and ran all six already-opened case43 episodes, but failed after simulation because `_aggregate_pair()` read `decision_timing_s` from an aggregate that exposes `decision_total_s` and `decision_mean_s_per_step`.\n\n"
        "Change: v3 is collation-only from completed v2 episode summaries and uses the aggregate timing schema already produced by the smoke helper. Controller, policies, terminal models, case43, seeds, horizon rule, solver/recovery behavior and validation/test access policy are unchanged.\n\n"
        "Outcome: see `%s`. This confirms only an engineering precondition: safe-shortening v1 avoids H>25 on contaminated case43 in preserved v2 episodes. It does not authorize final test; backup and fresh development-validation remain required.\n"
        % (MARKER, rel(OUT_DIR / "summary.md"))
    )
    path = ROOT / "DECISIONS.md"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if MARKER + "-decision" not in old:
            path.write_text(old.rstrip() + "\n" + decision, encoding="utf-8")


def main() -> int:
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED_safe_shortening_v1_case43_replay_v3_collation_only",
        "new_simulations": 0,
        "validation_case43_reopened_in_prior_v2": True,
        "fresh_validation_bank_generated": False,
        "sealed_test_accessed": False,
        "source_amendment": provenance()["v3_change"],
    })
    episodes = load_v2_episode_summaries()
    aggregates = aggregate_by_arm(episodes)
    pair_stats = paired(aggregates)
    control_steps = int(sum(e["steps"] for e in episodes))
    all_adaptive_no_h_above_25 = all(not pair_stats[str(seed)]["adaptive_used_above_25"] for seed in SEEDS)
    all_success_no_extra_failure = all(pair_stats[str(seed)]["success_delta_adaptive_minus_fixed"] == 0 and pair_stats[str(seed)]["failure_delta_adaptive_minus_fixed"] == 0 for seed in SEEDS)
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_safe_shortening_v1_case43_development_replay_v3_collation_only_not_formal_validation",
        "formal_scientific_evidence": False,
        "source_amendment": provenance(),
        "new_v3_budget": {"simulations": 0, "control_steps": 0, "gradient_steps": 0, "sealed_test_episodes": 0},
        "v2_preserved_budget": {"episodes": len(episodes), "control_steps": control_steps, "gradient_steps": 0, "sealed_test_episodes": 0},
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "validation64_full_bank_used_for_model_selection": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "case_id": CASE_ID,
        "split": "already-opened validation case43 development diagnostic; v3 collation only",
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "protocol": file_record(PROTOCOL),
        "source_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), V1_SOURCE, V2_SOURCE, SMOKE_V1_SOURCE, SMOKE_V3_SOURCE] if p.exists()},
        "v2_artifact_hashes": {rel(p): sha256(p) for p in [V2_OUT / "failure.json", V2_OUT / "progress.json", V2_OUT / "run_started.json", V2_OUT / "runtime_preflight.json", V2_OUT / "schedule.json", V2_OUT / "terminal_sources.json"] if p.exists()},
        "episodes": episodes,
        "aggregates": aggregates,
        "paired": pair_stats,
        "engineering_checks": {
            "all_adaptive_no_h_above_25": all_adaptive_no_h_above_25,
            "all_pairs_equal_success_and_failure_counts": all_success_no_extra_failure,
            "all_v2_episode_completed_hashes_verified": True,
            "v2_failed_only_after_all_episodes_done": True,
        },
        "interpretation_limits": [
            "case43 was already opened for development diagnosis and cannot be fresh independent validation",
            "v3 creates no new simulations; it collates v2 partial output",
            "one deterministic case/repeat per arm is not a timing-estimation campaign",
            "safe-shortening success here would not by itself support a final-test gate",
        ],
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    write_backup_request(raw)
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [BACKUP_REQUEST, PROTOCOL, Path(__file__).resolve(), V1_SOURCE, V2_SOURCE, SMOKE_V1_SOURCE, SMOKE_V3_SOURCE, V1_RUN_REGISTRY, V2_RUN_REGISTRY]
    files.extend([V2_OUT / "failure.json", V2_OUT / "progress.json", V2_OUT / "schedule.json"])
    for _, _, dirname in EXPECTED_EPISODES:
        ep_dir = V2_OUT / "episodes" / dirname
        files.extend([ep_dir / "summary.json", ep_dir / "completed.json"])
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists() and p.is_file()},
        "backup_request": rel(BACKUP_REQUEST),
        "formal_scientific_evidence": False,
        "new_v3_simulations": 0,
        "v2_preserved_episodes_collated": len(episodes),
        "v2_preserved_control_steps_collated": control_steps,
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "test_accessed": False,
        "engineering_checks": raw["engineering_checks"],
        "headline_pairs": {k: {
            "fixed_success": val["fixed"]["success_count"],
            "adaptive_success": val["adaptive"]["success_count"],
            "adaptive_horizons": val["adaptive"]["horizon_counts"],
            "physical_constraint_delta_adaptive_minus_fixed": val["physical_constraint_delta_adaptive_minus_fixed"],
            "decision_time_ratio_adaptive_over_fixed": val["decision_time_ratio_adaptive_over_fixed"],
            "adaptive_used_shorter_than_25": val["adaptive_used_shorter_than_25"],
            "adaptive_used_above_25": val["adaptive_used_above_25"],
        } for k, val in pair_stats.items()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "new_v3_simulations": 0,
        "v2_preserved_episodes_collated": len(episodes),
        "v2_preserved_control_steps_collated": control_steps,
        "validation_case43_reopened_for_development_diagnostic": True,
        "fresh_validation_bank_generated": False,
        "test_accessed": False,
        "engineering_checks": raw["engineering_checks"],
        "paired": pair_stats,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "new_v3_simulations": 0,
            "validation_case43_reopened_for_development_diagnostic": True,
            "fresh_validation_bank_generated": False,
            "test_accessed": False,
            "source_amendment": provenance(),
            "next_recovery_hint": "Inspect this v3 collation failure only. If the episode summaries are complete and the error is reporting-only, version a smaller collation repair. Do not rerun simulations or generate fresh dev-validation until case43 collation and backup gates are satisfied.",
        })
        raise
