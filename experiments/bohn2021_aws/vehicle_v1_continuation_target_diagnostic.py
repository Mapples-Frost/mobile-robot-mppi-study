#!/usr/bin/env python3
"""Mine V1 fixed-H evidence for controlled continuation diagnostic targets.

Metadata/trace-analysis only.  Reads already-created development V1 fixed-H
rollouts and the V1 postdiagnostic.  It does not construct environments, run
simulations, train/refit, open historical validation64 banks, or open sealed
final tests.

The goal is concrete: identify the smallest set of representative case/prefix
states where a future one-variable continuation replay can discriminate (a)
real within-episode horizon opportunity, (b) terminal/value/transition mismatch,
and (c) fixed-H oracle hindsight that is not state-predictable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
V1_POST_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/completed.json"
V1_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/raw.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_continuation_target_diagnostic_20260928T1155Z.md"
PROTOCOL_PATH = ROOT / "research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.json"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_CONTINUATION_TARGET_DIAGNOSTIC_20260928T1155Z.json"
MARKER = "vehicle-v1-continuation-target-diagnostic-20260928T1155Z"
H15_REFERENCE = 15
H10_FAST = 10
H30_SOLVER_RISK = 30
MAX_TARGET_CASES = 4
MAX_BRANCH_STEPS_PER_CASE = 2
STATE_CLOSE_THRESHOLD = 0.75
MIN_PROXY_TAIL_GAIN = 1.0


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = read_json(V1_RAW)
    done = read_json(V1_COMPLETED)
    post = read_json(V1_POST_COMPLETED)
    post_raw = read_json(V1_POST_RAW)
    for label, obj in (("v1_completed", done), ("v1_post_completed", post)):
        if obj.get("passed") is not True or obj.get("hard_pass") is not True:
            raise RuntimeError(label + " did not pass")
        if obj.get("historical_validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise RuntimeError(label + " has invalid access flags")
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("V1 raw access flags invalid")
    if post_raw.get("historical_validation64_bank_opened") is not False or post_raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("V1 post raw access flags invalid")
    if post.get("headline", {}).get("physical_oracle_material") is not True or post.get("headline", {}).get("total_oracle_material") is not True:
        raise RuntimeError("V1 postdiagnostic did not find material oracle opportunity; continuation target mining is not the right next step")
    return raw, post_raw


def table_by_case_h(episodes: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, Mapping[str, Any]]]:
    out: Dict[int, Dict[int, Mapping[str, Any]]] = {}
    for e in episodes:
        out.setdefault(int(e["case"]), {})[int(e["horizon"])] = e
    return out


def success_no_constraint(row: Mapping[str, Any]) -> bool:
    return bool(row.get("success")) and not bool(row.get("constraint"))


def strict(row: Mapping[str, Any]) -> bool:
    return success_no_constraint(row) and int(row.get("initial_failed_steps", 0)) == 0 and int(row.get("solver_failure_steps", 0)) == 0


def decision_sum(row: Mapping[str, Any]) -> float:
    return safe_float((row.get("decision_timing_s") or {}).get("sum"))


def choose(rows: Sequence[Mapping[str, Any]], objective: str) -> Mapping[str, Any]:
    candidates = [r for r in rows if strict(r)] or [r for r in rows if success_no_constraint(r)] or list(rows)
    if objective == "physical":
        return min(candidates, key=lambda r: (safe_float(r.get("physical_constraint_cost")), decision_sum(r), int(r["horizon"])))
    if objective == "total":
        return min(candidates, key=lambda r: (safe_float(r.get("total_cost")), decision_sum(r), int(r["horizon"])))
    raise ValueError(objective)


def read_trace(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    path = ROOT / str(row["path"]) / "trace.json"
    if not path.exists():
        raise RuntimeError("missing trace: " + rel(path))
    return read_json(path)


def state_vec(step: Mapping[str, Any], key: str = "previous_state") -> Tuple[float, float, float]:
    state = step.get(key) or {}
    return (safe_float(state.get("x"), float("nan")), safe_float(state.get("y"), float("nan")), safe_float(state.get("theta"), float("nan")))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def step_physical(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("performance")) + safe_float(row.get("constraint"))


def step_total(row: Mapping[str, Any]) -> float:
    return step_physical(row) + safe_float(row.get("compute"))


def tail_sums(trace: Sequence[Mapping[str, Any]], which: str) -> List[float]:
    vals = [step_physical(r) if which == "physical" else step_total(r) for r in trace]
    tails = [0.0] * (len(vals) + 1)
    running = 0.0
    for i in range(len(vals) - 1, -1, -1):
        running += vals[i]
        tails[i] = float(running)
    return tails[:-1]


def trace_pair_proxy(case_id: int, ref_row: Mapping[str, Any], alt_row: Mapping[str, Any]) -> Dict[str, Any]:
    ref_trace = read_trace(ref_row)
    alt_trace = read_trace(alt_row)
    n = min(len(ref_trace), len(alt_trace))
    ref_phys_tail = tail_sums(ref_trace, "physical")
    alt_phys_tail = tail_sums(alt_trace, "physical")
    ref_total_tail = tail_sums(ref_trace, "total")
    alt_total_tail = tail_sums(alt_trace, "total")
    rows: List[Dict[str, Any]] = []
    first_divergence = None
    for k in range(n):
        d = state_distance(state_vec(ref_trace[k], "previous_state"), state_vec(alt_trace[k], "previous_state"))
        if first_divergence is None and d > STATE_CLOSE_THRESHOLD:
            first_divergence = k
        phys_gain = ref_phys_tail[k] - alt_phys_tail[k]
        total_gain = ref_total_tail[k] - alt_total_tail[k]
        # Prefer noninitial branch states that are still close under the already
        # observed fixed-H trajectories and show a large tail-cost contrast.
        noninitial_bonus = 1.0 if k >= 3 else 0.0
        close_bonus = max(0.0, STATE_CLOSE_THRESHOLD - d)
        score = phys_gain + 0.25 * total_gain + close_bonus + noninitial_bonus - 2.0 * max(0.0, d - STATE_CLOSE_THRESHOLD)
        rows.append({
            "step": k,
            "state_distance_proxy": d,
            "ref_tail_physical": ref_phys_tail[k],
            "alt_tail_physical": alt_phys_tail[k],
            "proxy_tail_physical_gain_ref_minus_alt": phys_gain,
            "ref_tail_total": ref_total_tail[k],
            "alt_tail_total": alt_total_tail[k],
            "proxy_tail_total_gain_ref_minus_alt": total_gain,
            "score": score,
        })
    close_positive = [r for r in rows if r["step"] >= 3 and r["state_distance_proxy"] <= STATE_CLOSE_THRESHOLD and r["proxy_tail_physical_gain_ref_minus_alt"] >= MIN_PROXY_TAIL_GAIN]
    if close_positive:
        best = max(close_positive, key=lambda r: (r["score"], r["proxy_tail_physical_gain_ref_minus_alt"], -r["step"]))
    else:
        noninitial = [r for r in rows if r["step"] >= 3]
        best = max(noninitial or rows, key=lambda r: (r["score"], r["proxy_tail_physical_gain_ref_minus_alt"], -r["state_distance_proxy"]))
    candidate_steps = {0, int(best["step"])}
    if first_divergence is not None and first_divergence > 3:
        candidate_steps.add(int(first_divergence - 1))
    # Avoid too-late targets with little remaining horizon; keep branch states
    # within the first 2/3 of the episode when possible.
    candidate_steps = {s for s in candidate_steps if s < max(1, min(n, 120))}
    return {
        "case": case_id,
        "alt_horizon": int(alt_row["horizon"]),
        "ref_horizon": int(ref_row["horizon"]),
        "ref_episode_path": str(ref_row["path"]),
        "alt_episode_path": str(alt_row["path"]),
        "n_common_steps": n,
        "first_state_distance_gt_threshold_step": first_divergence,
        "best_proxy_branch_step": best,
        "recommended_branch_steps_for_this_alt": sorted(candidate_steps),
        "episode_delta_physical_ref_minus_alt": safe_float(ref_row.get("physical_constraint_cost")) - safe_float(alt_row.get("physical_constraint_cost")),
        "episode_delta_total_ref_minus_alt": safe_float(ref_row.get("total_cost")) - safe_float(alt_row.get("total_cost")),
        "episode_delta_decision_s_ref_minus_alt": decision_sum(ref_row) - decision_sum(alt_row),
        "trace_hashes": {
            str(ref_row["path"] + "/trace.json"): sha256(ROOT / str(ref_row["path"]) / "trace.json"),
            str(alt_row["path"] + "/trace.json"): sha256(ROOT / str(alt_row["path"]) / "trace.json"),
        },
    }


def summarize_case(case_id: int, table: Mapping[int, Mapping[int, Mapping[str, Any]]], meta: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(table[case_id].values())
    ref = table[case_id][H15_REFERENCE]
    best_phys = choose(rows, "physical")
    best_total = choose(rows, "total")
    alts = sorted({int(best_phys["horizon"]), int(best_total["horizon"]), H10_FAST, H30_SOLVER_RISK} - {H15_REFERENCE})
    proxy = [trace_pair_proxy(case_id, ref, table[case_id][h]) for h in alts]
    gain_phys = safe_float(ref.get("physical_constraint_cost")) - safe_float(best_phys.get("physical_constraint_cost"))
    gain_total = safe_float(ref.get("total_cost")) - safe_float(best_total.get("total_cost"))
    score = max(gain_phys, 0.0) + max(gain_total, 0.0) + 0.2 * sum(max(0.0, p["best_proxy_branch_step"]["proxy_tail_physical_gain_ref_minus_alt"]) for p in proxy)
    return {
        "case": case_id,
        "source_candidate_index": int(meta.get("candidate_index", -1)),
        "stratum": meta.get("stratum"),
        "theta_r": safe_float(meta.get("theta_r")),
        "traj_steps": int(safe_float(meta.get("traj_steps"))),
        "min_reference_obstacle_clearance": safe_float(meta.get("min_reference_obstacle_clearance")),
        "h15_physical": safe_float(ref.get("physical_constraint_cost")),
        "h15_total": safe_float(ref.get("total_cost")),
        "h15_decision_s": decision_sum(ref),
        "h15_initial_failed_steps": int(ref.get("initial_failed_steps", 0)),
        "best_physical_horizon": int(best_phys["horizon"]),
        "best_physical_cost": safe_float(best_phys.get("physical_constraint_cost")),
        "best_total_horizon": int(best_total["horizon"]),
        "best_total_cost": safe_float(best_total.get("total_cost")),
        "physical_gain_vs_H15": gain_phys,
        "total_gain_vs_H15": gain_total,
        "alt_horizons_for_future_branch": alts,
        "selection_score": score,
        "trace_pair_proxies": proxy,
    }


def select_targets(case_summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    eligible = [c for c in case_summaries if c["physical_gain_vs_H15"] > 2.0 or c["total_gain_vs_H15"] > 2.0 or c["best_physical_horizon"] != H15_REFERENCE or c["best_total_horizon"] != H15_REFERENCE]
    ranked = sorted(eligible, key=lambda c: (c["selection_score"], c["physical_gain_vs_H15"], c["total_gain_vs_H15"]), reverse=True)
    chosen: List[Mapping[str, Any]] = []
    used_cases = set()
    # Ensure diversity: include strong physical, total-H10/30, and long-horizon cases when available.
    buckets = [
        lambda c: c["physical_gain_vs_H15"] > 5.0,
        lambda c: c["best_total_horizon"] == H10_FAST,
        lambda c: c["best_total_horizon"] == H30_SOLVER_RISK,
        lambda c: c["best_physical_horizon"] >= 35,
    ]
    for pred in buckets:
        cand = [c for c in ranked if int(c["case"]) not in used_cases and pred(c)]
        if cand and len(chosen) < MAX_TARGET_CASES:
            chosen.append(cand[0])
            used_cases.add(int(cand[0]["case"]))
    for c in ranked:
        if len(chosen) >= MAX_TARGET_CASES:
            break
        if int(c["case"]) not in used_cases:
            chosen.append(c)
            used_cases.add(int(c["case"]))
    targets: List[Dict[str, Any]] = []
    for c in chosen:
        branch_steps = {0}
        # Add best proxy steps from top alts, capped.
        proxies = sorted(c["trace_pair_proxies"], key=lambda p: (p["best_proxy_branch_step"]["score"], p["best_proxy_branch_step"]["proxy_tail_physical_gain_ref_minus_alt"]), reverse=True)
        for p in proxies:
            for s in p["recommended_branch_steps_for_this_alt"]:
                if s > 0:
                    branch_steps.add(int(s))
                if len(branch_steps) >= MAX_BRANCH_STEPS_PER_CASE + 1:  # includes 0; later remove if too many
                    break
            if len(branch_steps) >= MAX_BRANCH_STEPS_PER_CASE + 1:
                break
        nonzero = sorted([s for s in branch_steps if s > 0])[:MAX_BRANCH_STEPS_PER_CASE]
        selected_steps = sorted(set(([0] if not nonzero else nonzero)))
        branch_horizons = sorted(set([H15_REFERENCE, H10_FAST, H30_SOLVER_RISK, int(c["best_physical_horizon"]), int(c["best_total_horizon"])]))
        targets.append({
            "case": int(c["case"]),
            "source_candidate_index": int(c["source_candidate_index"]),
            "stratum": c["stratum"],
            "rationale": {
                "physical_gain_vs_H15": c["physical_gain_vs_H15"],
                "total_gain_vs_H15": c["total_gain_vs_H15"],
                "best_physical_horizon": c["best_physical_horizon"],
                "best_total_horizon": c["best_total_horizon"],
                "selection_score": c["selection_score"],
            },
            "prefix_policy": "fixed_H15_common_prefix_then_branch",
            "branch_steps": selected_steps,
            "branch_horizons": branch_horizons,
            "alt_proxy_summaries": [
                {
                    "alt_horizon": p["alt_horizon"],
                    "first_state_distance_gt_0p75_step": p["first_state_distance_gt_threshold_step"],
                    "best_proxy_step": p["best_proxy_branch_step"],
                    "episode_delta_physical_ref_minus_alt": p["episode_delta_physical_ref_minus_alt"],
                    "episode_delta_total_ref_minus_alt": p["episode_delta_total_ref_minus_alt"],
                }
                for p in proxies[:4]
            ],
        })
    return targets


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle V1 continuation target diagnostic",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata/trace analysis only: no simulations, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Why this diagnostic",
        "",
        "V1 showed material episode-level fixed-H oracle opportunity, but simple case-metadata LOOCV selectors were worse than strong fixed-H baselines. The next high-information experiment is therefore not broad retraining, but controlled continuation from identical replayed prefix states.",
        "",
        "## Selected future continuation targets",
        "",
        "| case | source candidate | stratum | branch steps | branch horizons | physical gain vs H15 | total gain vs H15 | best physical H | best total H |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for t in raw["selected_targets"]:
        r = t["rationale"]
        lines.append("| %d | %d | `%s` | `%s` | `%s` | %.6g | %.6g | %d | %d |" % (
            t["case"], t["source_candidate_index"], t["stratum"], t["branch_steps"], t["branch_horizons"],
            r["physical_gain_vs_H15"], r["total_gain_vs_H15"], r["best_physical_horizon"], r["best_total_horizon"],
        ))
    lines += [
        "",
        "## Frozen next diagnostic design",
        "",
        f"- Common prefix: replay the same saved V1 case with fixed H15 until each listed branch step; then branch to each listed horizon and continue to termination. This avoids comparing different fixed-H trajectories as if they shared states.",
        f"- Maximum future rollout episodes: `{raw['future_protocol']['max_episodes']}`; control-step upper bound: `{raw['future_protocol']['control_step_upper_bound']}`; no training/gradient updates; no validation64/test access.",
        "- Primary discriminators: continuation physical+constraint cost, total cost, success/safety, initial/final solver failures, actual decision and solver timing, and terminal-value-source sensitivity if included.",
        "- Interpretation: if identical-state continuation confirms material state-dependent branch gains, freeze one IMPROVED selector/value-refit smoke; if not, prioritize versioned scenario redesign or document weak canonical adaptive opportunity.",
        "",
        f"Backup request before any future simulation: `{raw['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 V1 continuation target diagnostic\n\n"
        f"UTC: {raw['created_utc']}. Metadata/trace-only diagnostic selected {len(raw['selected_targets'])} V1 cases for a future controlled continuation replay; "
        f"max future episodes={raw['future_protocol']['max_episodes']}, control-step bound={raw['future_protocol']['control_step_upper_bound']}. "
        "No simulations/training/validation64/test access occurred. "
        f"Next action after verified backup: {raw['next_action']}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(PROTOCOL_PATH)}`.\n"
    )
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
        return 0
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    raw, post_raw = verify_inputs()
    horizons = [int(h) for h in raw["protocol_full"]["rollout_design"]["horizons"]]
    table = table_by_case_h(raw["episodes"])
    metas = list(raw["bank_selection"]["selected_metadata"])
    case_summaries = [summarize_case(c, table, metas[c]) for c in sorted(table)]
    selected_targets = select_targets(case_summaries)
    if not selected_targets:
        raise RuntimeError("No continuation targets selected despite material V1 oracle opportunity")
    max_episodes = sum(len(t["branch_steps"]) * len(t["branch_horizons"]) for t in selected_targets)
    control_bound = int(max_episodes * 150)
    future_protocol = {
        "protocol_id": "vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928",
        "created_utc": created,
        "classification": "development_IMPROVED_diagnostic_controlled_continuation_not_model_selection_not_final_test",
        "motivating_evidence": [
            "V1 fixed-H oracle physical and total opportunity passed materiality thresholds.",
            "V1 simple scenario-metadata LOOCV selectors failed to beat strong fixed-H references.",
            "Episode-level oracle is not an upper bound on within-episode adaptive switching; identical-state continuation is needed.",
        ],
        "access_rules": {"historical_validation64_bank_opened": False, "sealed_test_accessed": False, "development_only": True, "requires_verified_backup_before_rollout": True},
        "source_bank": {"v1_raw": rel(V1_RAW), "v1_raw_sha256": sha256(V1_RAW), "v1_completed": rel(V1_COMPLETED), "v1_completed_sha256": sha256(V1_COMPLETED)},
        "design": {
            "common_prefix": "for each target, reset the saved V1 case and run fixed H15 until branch_step; repeat the identical prefix separately for every branch arm",
            "branch_horizons": "target-specific set including H15, H10 fastest-safe, H30 solver-risk reference, and the case's V1 best physical/total horizons",
            "max_episodes": max_episodes,
            "control_step_upper_bound": control_bound,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "metrics": ["physical_constraint_cost", "total_cost", "success", "constraint", "initial_failed_steps", "solver_failure_steps", "decision_timing_s", "solver_attempt_timing_s", "terminal_value_estimate_if_available"],
        },
        "targets": selected_targets,
        "acceptance_for_followup_selector_smoke": "At least two noninitial branch states must show a safe branch horizon improving continuation physical or total cost by >=3 absolute units versus H15 without solver/constraint regression; otherwise do not launch selector/refit solely from episode-level V1 oracle labels.",
    }
    write_json(PROTOCOL_JSON, future_protocol)
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(
        "# Vehicle V1 controlled continuation diagnostic v0 frozen protocol\n\n"
        f"Frozen UTC: `{created}`. Development diagnostic only; no validation64/test access. Requires verified backup before rollout.\n\n"
        "## Rationale\n\nV1 fixed-H oracle gains are material, but case-metadata LOOCV selectors failed. The next experiment must compare horizons from identical intermediate states by replaying a common H15 prefix before branching.\n\n"
        "## Budget\n\n"
        f"- Max episodes: `{max_episodes}`\n- Control-step upper bound: `{control_bound}`\n- Training episodes / gradient steps: `0 / 0`\n\n"
        "## Targets\n\n"
        + json.dumps(selected_targets, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    result = {
        "created_utc": created,
        "method": "vehicle_v1_continuation_target_diagnostic_metadata_trace_only",
        "classification": "metadata_trace_only_development_not_validation_not_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_hashes": {rel(p): sha256(p) for p in [V1_RAW, V1_COMPLETED, V1_POST_COMPLETED, V1_POST_RAW]},
        "v1_post_headline": read_json(V1_POST_COMPLETED).get("headline"),
        "case_summaries": case_summaries,
        "selected_targets": selected_targets,
        "future_protocol": future_protocol["design"],
        "protocol_md": rel(PROTOCOL_PATH),
        "protocol_json": rel(PROTOCOL_JSON),
        "next_action": "after verified backup, implement/run the frozen controlled-continuation diagnostic before selector/refit training",
        "backup_required_before_more_simulations": True,
        "backup_request": rel(BACKUP_PATH),
        "limitations": [
            "Trace-pair tail gains are only proxies because different fixed-H episodes are not identical-state continuations after divergence.",
            "The frozen follow-up protocol resolves this by replaying a common prefix for every branch arm.",
            "No validation64 or sealed test evidence is used or accessed.",
        ],
    }
    write_json(OUT_DIR / "raw.json", result)
    write_summary(result)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 continuation target diagnostic state\n\nUTC: {created}. Metadata/trace-only diagnostic selected {len(selected_targets)} cases and froze `{rel(PROTOCOL_PATH)}` for the next controlled-continuation rollout after backup. No simulation/training/validation64/test access occurred. Backup required before simulation.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup V1 continuation target diagnostic and frozen protocol before controlled-continuation simulations",
        "backup_required_before_more_simulations": True,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(PROTOCOL_PATH), rel(PROTOCOL_JSON), rel(BACKUP_PATH), rel(Path(__file__).resolve())],
    })
    append_docs(result)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, PROTOCOL_PATH, PROTOCOL_JSON, BACKUP_PATH, Path(__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_required_before_more_simulations": True,
        "backup_request": rel(BACKUP_PATH),
        "headline": {
            "selected_target_count": len(selected_targets),
            "selected_cases": [t["case"] for t in selected_targets],
            "future_max_episodes": max_episodes,
            "future_control_step_upper_bound": control_bound,
            "next_action": result["next_action"],
        },
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "protocol": rel(PROTOCOL_PATH),
        "selected_cases": [t["case"] for t in selected_targets],
        "future_max_episodes": max_episodes,
        "future_control_step_upper_bound": control_bound,
        "new_rollouts": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_PATH),
        "next_action": result["next_action"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
