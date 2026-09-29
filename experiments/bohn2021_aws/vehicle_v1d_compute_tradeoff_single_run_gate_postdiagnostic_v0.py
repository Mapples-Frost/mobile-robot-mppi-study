#!/usr/bin/env python3
"""No-simulation precheck of the v1d compute-tradeoff confirmation gate.

This diagnostic reuses the already-opened v1d development smoke raw rows and
applies the newly frozen repeated-timing confirmation rules to the original
single-run branch rollouts.  It is meant to decide whether the 186-episode
backup-gated confirmation is still informative, without consuming any new
simulation, validation64, sealed-test, training, or refit budget.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_runner as confirm  # noqa:E402

STAMP = "20260929T0415Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_single_run_gate_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_v1d_compute_tradeoff_single_run_gate_postdiagnostic_v0_{STAMP}.md"
REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1D_COMPUTE_TRADEOFF_SINGLE_RUN_GATE_POSTDIAGNOSTIC_V0_{STAMP}.json"
MARKER = f"vehicle-v1d-compute-tradeoff-single-run-gate-postdiagnostic-v0-{STAMP}"
FIRST_SUPERVISOR_EVENT = dt.datetime(2026, 9, 26, 10, 55, 29, 419331, tzinfo=dt.timezone.utc)


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def query_tokens() -> Dict[str, Any]:
    for db in (ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"):
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db)); cur = con.cursor(); total = 0; found = False; by_table = {}
            for (table,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val = int(cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0] or 0)
                    total += val; found = True; by_table[table] = val
            con.close()
            if found:
                return {"available": True, "path": rel(db), "total_tokens": total, "by_table": by_table}
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": repr(exc)}
    return {"available": False, "reason": "research.sqlite not found in repository-visible candidate paths"}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def normalize_episode(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, Mapping):
        return None
    if "state_id" not in row or "terminal_mode" not in row:
        return None
    h_raw = row.get("branch_horizon", row.get("horizon"))
    try:
        h = int(h_raw)
    except Exception:
        return None
    out = dict(row)
    out["branch_horizon"] = h
    if "decision_timing_s" not in out or not isinstance(out.get("decision_timing_s"), Mapping):
        for k in ("decision_sum_s", "decision_wall_time_sum_s", "whole_decision_wall_time_sum_s", "decision_time_sum_s"):
            if k in out:
                out["decision_timing_s"] = {"sum": fnum(out.get(k))}
                break
    if "solver_attempt_timing_s" not in out or not isinstance(out.get("solver_attempt_timing_s"), Mapping):
        for k in ("solver_attempt_sum_s", "solver_attempt_wall_time_sum_s", "solver_wall_time_sum_s", "solver_time_sum_s"):
            if k in out:
                out["solver_attempt_timing_s"] = {"sum": fnum(out.get(k))}
                break
    if "continuation_physical_constraint_cost_from_branch" not in out:
        for k in ("physical_constraint_cost_from_branch", "continuation_physical_cost_from_branch", "physical_cost_from_branch"):
            if k in out:
                out["continuation_physical_constraint_cost_from_branch"] = fnum(out.get(k))
                break
    if "continuation_total_cost_from_branch" not in out:
        for k in ("total_cost_from_branch", "continuation_cost_from_branch"):
            if k in out:
                out["continuation_total_cost_from_branch"] = fnum(out.get(k))
                break
    if "decision_timing_s" not in out or "solver_attempt_timing_s" not in out:
        out["_timing_schema_incomplete_after_normalization"] = True
    return out


def collect_candidate_branch_episodes(v1d_raw: Mapping[str, Any], schedule: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    state_ids = set(str(s) for s in schedule["state_ids"])
    needed_h: Dict[str, set] = {sid: {10, 15, 25} for sid in state_ids}
    for row in schedule["episodes"]:
        needed_h[str(row["state_id"])].add(int(row["horizon"]))
    terminal_modes = {"zero_terminal", "h15_common_terminal"}
    top_lists = []
    for key in ("branch_episodes", "episodes", "episode_summaries", "summaries"):
        val = v1d_raw.get(key)
        if isinstance(val, list):
            top_lists.append((key, val))
    rows: List[Dict[str, Any]] = []
    seen = set()
    source_counts: Dict[str, int] = {}
    for source_key, vals in top_lists:
        for row in vals:
            ep = normalize_episode(row)
            if ep is None:
                continue
            sid = str(ep.get("state_id"))
            mode = str(ep.get("terminal_mode"))
            h = int(ep.get("branch_horizon"))
            if sid not in state_ids or mode not in terminal_modes or h not in needed_h.get(sid, set()):
                continue
            # Require at least one outcome/timing field to avoid accidentally selecting frozen schedule rows.
            if not any(k in ep for k in ("success", "decision_timing_s", "solver_attempt_timing_s", "continuation_physical_constraint_cost_from_branch", "path")):
                continue
            key = (sid, mode, h, int(ep.get("branch_step", -1)), int(ep.get("case", -1)), str(ep.get("path", "")), int(ep.get("execution_index", -1)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(ep)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
    return rows, {"top_level_lists_seen": [k for k, _ in top_lists], "source_counts": source_counts, "deduped_rows": len(rows)}


def mode_short(row: Mapping[str, Any], mode: str) -> str:
    pm = row.get("per_mode", {}).get(mode, {}) if isinstance(row.get("per_mode"), Mapping) else {}
    if pm.get("missing_required_arm"):
        return "missing"
    a = pm.get("H10_vs_H15", {})
    b = pm.get("H10_vs_H25", {})
    return (
        f"strict={pm.get('strict_mode')} relaxed={pm.get('relaxed_mode')} "
        f"d15={fnum(a.get('decision_gain_s')):.4g}s s15={fnum(a.get('solver_gain_s')):.4g}s "
        f"phys15={fnum(a.get('physical_loss')):.4g} safe15={a.get('safe')} "
        f"d25={fnum(b.get('decision_gain_s')):.4g}s s25={fnum(b.get('solver_gain_s')):.4g}s "
        f"phys25={fnum(b.get('physical_loss')):.4g} safe25={b.get('safe')}"
    )


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    lines = [
        "# Vehicle v1d compute-tradeoff single-run gate postdiagnostic v0",
        "",
        f"UTC: `{raw['created_utc']}`. No new simulations/training/refit; no validation64 or sealed-test access.",
        "",
        "## Headline",
        "",
        f"- Existing v1d branch rows reused for planned states: `{raw['input_row_audit']['candidate_branch_rows']}`; missing required arms: `{raw['input_row_audit']['missing_required_arm_count']}`.",
        f"- Applying the repeated-timing confirmation gate to the original single-run rows gives strict H10-labelled states `{analysis['strict_confirmed_h10_labelled_count']}/6`, relaxed `{analysis['relaxed_confirmed_h10_labelled_count']}/6`, strict negative controls `{analysis['strict_confirmed_negative_control_count']}/4`.",
        f"- Single-run pass-to-selector-refit-design by the repeated confirmation rule: `{analysis['pass_to_compact_compute_safe_selector_refit_design']}`.",
        f"- Recommendation: `{raw['decision']['next_action']}`.",
        "",
        "## Per-state single-run gate compatibility",
        "",
        "| state | role | strict | relaxed | zero_terminal summary | h15_common_terminal summary |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in analysis["state_rows"]:
        lines.append(
            f"| `{row['state_id']}` | `{row['role']}` | `{row['strict_confirmed_H10_compute_state']}` | `{row['relaxed_confirmed_H10_compute_state']}` | {mode_short(row, 'zero_terminal')} | {mode_short(row, 'h15_common_terminal')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        raw["decision"]["interpretation"],
        "",
        f"Backup request after this no-simulation diagnostic: `{raw['backup_request_after_diagnostic']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if (OUT / "completed.json").exists():
        confirm.completed_ok(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    inputs = confirm.load_inputs()
    schedule = confirm.build_schedule(inputs)
    rows, collect_audit = collect_candidate_branch_episodes(inputs["v1d_raw"], schedule)
    if not rows:
        raise ContractError("no v1d branch rows found for the compute-tradeoff schedule states")
    expected_arms = set()
    for sid in schedule["state_ids"]:
        for mode in inputs["protocol"]["rollout_design"]["post_branch_terminal_modes"]:
            for h in (sorted({int(r["horizon"]) for r in schedule["episodes"] if str(r["state_id"]) == sid})):
                expected_arms.add((sid, str(mode), h))
    present_arms = {(str(r["state_id"]), str(r["terminal_mode"]), int(r["branch_horizon"])) for r in rows}
    missing_arms = sorted(expected_arms - present_arms)
    analysis = confirm.analyze_repeats(rows, inputs["protocol"], schedule)
    strict_labelled = int(analysis["strict_confirmed_h10_labelled_count"])
    strict_controls = int(analysis["strict_confirmed_negative_control_count"])
    if missing_arms:
        next_action = "repair schema/row extraction before any repeated timing rollout"
        interpretation = "The no-simulation precheck could not see every required original v1d arm, so it cannot judge whether the backup-gated confirmation is worthwhile. Preserve this as a schema diagnostic and repair extraction only."
    elif strict_labelled >= 4 and strict_controls <= 1:
        next_action = "after verified backup, run the blocked repeated timing confirmation"
        interpretation = "The original single-run rows satisfy the same strict gate on enough labelled states, so the repeated timing confirmation remains a high-value discriminator of timing repeatability versus noise before any selector/refit."
    elif 2 <= strict_labelled <= 3 and strict_controls <= 1:
        next_action = "after backup, consider a smaller or full repeated-timing check; do not selector/refit before confirming timing stability"
        interpretation = "The single-run evidence is marginal under the stricter repeated gate. Repeated timing may still be informative, but the expected outcome is now a timing-noise/objective-scale decision rather than a likely selector/refit trigger."
    else:
        next_action = "do not spend the 186-episode confirmation yet; freeze negative/absorbed compute-opportunity diagnosis or design a versioned scenario/value diagnostic"
        interpretation = "The original single-run rows do not satisfy the stricter repeated confirmation rule often enough. This weakens the case for immediate repeated timing and further argues that current-source vehicle scenarios/terminal objectives provide sparse adaptive-H opportunity."
    now = dt.datetime.now(dt.timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {
        "created_utc": now.isoformat(),
        "method": "vehicle_v1d_compute_tradeoff_single_run_gate_postdiagnostic_v0_no_simulation",
        "classification": "development_no_simulation_existing_v1d_rows_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "elapsed_since_first_supervisor_event_seconds": (now - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "server_api_total_tokens_best_effort": query_tokens(),
        "inputs": {
            "v1d_raw": rel(confirm.V1D_RAW), "v1d_raw_sha256": sha256(confirm.V1D_RAW),
            "solver_raw": rel(confirm.SOLVER_RAW), "solver_raw_sha256": sha256(confirm.SOLVER_RAW),
            "h10_absorption_raw": rel(confirm.H10_RAW), "h10_absorption_raw_sha256": sha256(confirm.H10_RAW),
            "dryrun_completed": rel(confirm.DRYRUN_DIR / "completed.json"), "dryrun_completed_exists": (confirm.DRYRUN_DIR / "completed.json").exists(),
            "frozen_protocol": rel(confirm.PROTOCOL_JSON), "frozen_protocol_sha256": sha256(confirm.PROTOCOL_JSON),
        },
        "input_row_audit": {
            **collect_audit,
            "candidate_branch_rows": len(rows),
            "expected_unique_arms": len(expected_arms),
            "present_unique_arms": len(present_arms),
            "missing_required_arm_count": len(missing_arms),
            "missing_required_arms": missing_arms,
            "rows_with_incomplete_timing_schema_after_normalization": sum(1 for r in rows if r.get("_timing_schema_incomplete_after_normalization")),
        },
        "analysis": analysis,
        "decision": {"train_or_refit_now": False, "run_confirmation_now": False, "next_action": next_action, "interpretation": interpretation},
        "backup_request_after_diagnostic": rel(REQ),
        "interpretation_limits": ["development-only postdiagnostic", "uses already inspected v1d rows", "single-run timing is not a speed claim", "not validation/model-selection/final test", "no selector/refit/training"],
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Vehicle v1d compute-tradeoff single-run gate postdiagnostic\n\nUTC: {now.isoformat()}. Existing v1d rows under repeated gate: strict_labelled={strict_labelled}/6, strict_controls={strict_controls}/4, missing_arms={len(missing_arms)}. Next: {next_action}. No new simulations/training/validation/test.\n",
        encoding="utf-8",
    )
    write_json(REQ, {
        "requested_utc": now.isoformat(),
        "reason": "backup no-simulation single-run gate diagnostic before deciding repeated timing or scenario/value pivot",
        "backup_required_before_more_simulations": True,
        "episodes": 0,
        "control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(REQ), rel(Path(__file__).resolve())],
    })
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle v1d compute-tradeoff single-run gate postdiagnostic

UTC: {now.isoformat()}. No-simulation diagnostic applied the frozen repeated-timing confirmation gate to existing v1d single-run branch rows for the 6 H10-labelled states plus 4 negative controls. Strict confirmed H10-labelled states={strict_labelled}/6, relaxed={analysis['relaxed_confirmed_h10_labelled_count']}/6, strict negative controls={strict_controls}/4, missing required arms={len(missing_arms)}. Decision: {next_action}. No validation64/sealed-test access and no training/refit. Backup requested at `{rel(REQ)}` before further simulation.
"""
    append_docs(block)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE, REQ, Path(__file__).resolve(), confirm.PROTOCOL_JSON, confirm.V1D_RAW, confirm.SOLVER_RAW, confirm.H10_RAW]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": now.isoformat(),
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "backup_request": rel(REQ),
        "headline": {
            "strict_confirmed_h10_labelled_count": strict_labelled,
            "relaxed_confirmed_h10_labelled_count": analysis["relaxed_confirmed_h10_labelled_count"],
            "strict_confirmed_negative_control_count": strict_controls,
            "missing_required_arm_count": len(missing_arms),
            "pass_to_compact_compute_safe_selector_refit_design_single_run": analysis["pass_to_compact_compute_safe_selector_refit_design"],
            "train_or_refit_now": False,
            "next_action": next_action,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "strict_confirmed_h10_labelled_count": strict_labelled, "relaxed_confirmed_h10_labelled_count": analysis["relaxed_confirmed_h10_labelled_count"], "strict_confirmed_negative_control_count": strict_controls, "missing_required_arm_count": len(missing_arms), "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "train_or_refit_now": False, "next_action": next_action}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve failure. Repair only no-simulation schema/row extraction before any repeated timing rollout."})
        raise
