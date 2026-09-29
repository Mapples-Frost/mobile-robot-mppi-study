#!/usr/bin/env python3
"""No-simulation upper-bound diagnostic for v1d compute-tradeoff labels.

The prior single-run gate found only 1/6 strict H10-labelled states, despite
6/6 relaxed labels.  This script asks a more discriminating question before any
new rollouts/refit: even if an oracle knew the existing development branch
outcomes, how much measured decision/solver time could a conditional short-H
policy save on these states without physical/safety loss relative to fixed H25?

Inputs are already-opened development diagnostics only.  No validation64 bank,
sealed test, simulation, training, or refit is used.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0355Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_oracle_upper_bound_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_v1d_compute_tradeoff_oracle_upper_bound_postdiagnostic_v0_{STAMP}.md"
REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1D_COMPUTE_TRADEOFF_ORACLE_UPPER_BOUND_POSTDIAGNOSTIC_V0_{STAMP}.json"
MARKER = f"vehicle-v1d-compute-tradeoff-oracle-upper-bound-postdiagnostic-v0-{STAMP}"
FIRST_SUPERVISOR_EVENT = dt.datetime(2026, 9, 26, 10, 55, 29, 419331, tzinfo=dt.timezone.utc)

SINGLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_single_run_gate_postdiagnostic_v0_20260929T0415Z/raw.json"
SINGLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_single_run_gate_postdiagnostic_v0_20260929T0415Z/completed.json"
H10_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_20260929T0350Z/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_frozen_20260929T0410Z.json"

MODES = ("zero_terminal", "h15_common_terminal")
BASELINE_H = 25
ALT_BASELINE_H = 15


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
            con = sqlite3.connect(str(db)); cur = con.cursor(); total = 0; by_table = {}; found = False
            for (table,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val = int(cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0] or 0)
                    total += val; by_table[table] = val; found = True
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


def require_done(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing input marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"input marker not passed: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("sealed_test_bank_opened") is True:
        raise ContractError(f"sealed-test access flag in {rel(path)}")
    return obj


def arm(row: Mapping[str, Any], mode: str, h: int) -> Optional[Mapping[str, Any]]:
    pm = (row.get("per_mode") or {}).get(mode) or {}
    arms = pm.get("arms") or {}
    return arms.get(str(h)) or arms.get(h)


def safe(arm_row: Optional[Mapping[str, Any]]) -> bool:
    if not arm_row:
        return False
    repeat = int(fnum(arm_row.get("repeat_count"), 1))
    return (
        int(fnum(arm_row.get("success_count"), 0)) >= repeat
        and int(fnum(arm_row.get("constraint_count"), 0)) == 0
        and int(fnum(arm_row.get("solver_failure_steps_sum"), 0)) == 0
        and int(fnum(arm_row.get("initial_failed_steps_sum"), 0)) == 0
        and int(fnum(arm_row.get("final_failed_steps_sum"), 0)) == 0
    )


def choose_best_time(row: Mapping[str, Any], mode: str, eps: float, baseline_h: int = BASELINE_H) -> int:
    base = arm(row, mode, baseline_h)
    if not safe(base):
        return baseline_h
    base_phys = fnum(base.get("physical_median"))
    candidates: List[Tuple[float, int]] = []
    pm = (row.get("per_mode") or {}).get(mode) or {}
    for h_raw, a in (pm.get("arms") or {}).items():
        try:
            h = int(h_raw)
        except Exception:
            continue
        if safe(a) and fnum(a.get("physical_median")) <= base_phys + eps:
            candidates.append((fnum(a.get("decision_sum_median_s")), h))
    if not candidates:
        return baseline_h
    return min(candidates)[1]


def policy_choice(policy: str, row: Mapping[str, Any], mode: str) -> int:
    role = str(row.get("role"))
    if policy == "fixed_H25":
        return 25
    if policy == "fixed_H15":
        return 15
    if policy == "strict_state_H10_else_H25":
        return 10 if role == "h10_labelled_compute" and row.get("strict_confirmed_H10_compute_state") is True else 25
    if policy == "relaxed_labelled_H10_else_H25":
        return 10 if role == "h10_labelled_compute" and row.get("relaxed_confirmed_H10_compute_state") is True else 25
    if policy == "relaxed_any_state_H10_else_H25":
        return 10 if row.get("relaxed_confirmed_H10_compute_state") is True else 25
    if policy == "mode_aware_strict_H10_else_H25_oracle":
        pm = (row.get("per_mode") or {}).get(mode) or {}
        return 10 if pm.get("strict_mode") is True else 25
    if policy == "best_time_safe_eps001_oracle":
        return choose_best_time(row, mode, 0.01)
    if policy == "best_time_safe_eps01_oracle":
        return choose_best_time(row, mode, 0.1)
    raise KeyError(policy)


def summarize_policy(name: str, rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "policy": name,
        "state_mode_count": 0,
        "state_count_with_H10": 0,
        "mode_count_H10": 0,
        "mode_count_H15": 0,
        "mode_count_H25": 0,
        "mode_count_H30_or_other": 0,
        "decision_sum_s": 0.0,
        "solver_sum_s": 0.0,
        "physical_sum": 0.0,
        "total_sum": 0.0,
        "steps_sum": 0.0,
        "safe_mode_count": 0,
        "unsafe_or_missing_modes": [],
        "choices": [],
    }
    h10_states = set()
    for row in rows:
        sid = str(row["state_id"])
        for mode in MODES:
            h = policy_choice(name, row, mode)
            a = arm(row, mode, h)
            if a is None:
                h = BASELINE_H
                a = arm(row, mode, h)
            if a is None:
                out["unsafe_or_missing_modes"].append({"state_id": sid, "mode": mode, "horizon": h, "reason": "missing arm"})
                continue
            out["state_mode_count"] += 1
            if h == 10:
                out["mode_count_H10"] += 1; h10_states.add(sid)
            elif h == 15:
                out["mode_count_H15"] += 1
            elif h == 25:
                out["mode_count_H25"] += 1
            else:
                out["mode_count_H30_or_other"] += 1
            out["decision_sum_s"] += fnum(a.get("decision_sum_median_s"))
            out["solver_sum_s"] += fnum(a.get("solver_attempt_sum_median_s"))
            out["physical_sum"] += fnum(a.get("physical_median"))
            out["total_sum"] += fnum(a.get("total_median"))
            out["steps_sum"] += fnum(a.get("steps_median"))
            if safe(a):
                out["safe_mode_count"] += 1
            else:
                out["unsafe_or_missing_modes"].append({"state_id": sid, "mode": mode, "horizon": h, "reason": "unsafe arm"})
            out["choices"].append({"state_id": sid, "mode": mode, "horizon": h, "role": row.get("role")})
    out["state_count_with_H10"] = len(h10_states)
    out["all_modes_safe"] = (out["safe_mode_count"] == out["state_mode_count"] and not out["unsafe_or_missing_modes"])
    return out


def add_comparison(policy: Dict[str, Any], base25: Mapping[str, Any], base15: Mapping[str, Any]) -> None:
    for label, base in (("vs_H25", base25), ("vs_H15", base15)):
        b_dec = fnum(base.get("decision_sum_s")); b_solver = fnum(base.get("solver_sum_s"))
        policy[f"decision_gain_s_{label}"] = b_dec - fnum(policy.get("decision_sum_s"))
        policy[f"decision_rel_gain_{label}"] = (b_dec - fnum(policy.get("decision_sum_s"))) / b_dec if b_dec > 0 else None
        policy[f"solver_gain_s_{label}"] = b_solver - fnum(policy.get("solver_sum_s"))
        policy[f"solver_rel_gain_{label}"] = (b_solver - fnum(policy.get("solver_sum_s"))) / b_solver if b_solver > 0 else None
        policy[f"physical_loss_{label}"] = fnum(policy.get("physical_sum")) - fnum(base.get("physical_sum"))
        policy[f"total_delta_{label}"] = fnum(policy.get("total_sum")) - fnum(base.get("total_sum"))
        policy[f"safe_mode_delta_{label}"] = int(policy.get("safe_mode_count", 0)) - int(base.get("safe_mode_count", 0))


def table_line(p: Mapping[str, Any]) -> str:
    return (
        f"| `{p['policy']}` | {p['mode_count_H10']} | {p['mode_count_H15']} | {p['mode_count_H25']} | {p['mode_count_H30_or_other']} | "
        f"{p['safe_mode_count']}/{p['state_mode_count']} | {p['decision_sum_s']:.3f} | {p['decision_gain_s_vs_H25']:.3f} | "
        f"{p['decision_rel_gain_vs_H25']*100.0 if p['decision_rel_gain_vs_H25'] is not None else float('nan'):.2f}% | {p['solver_gain_s_vs_H25']:.3f} | "
        f"{p['physical_loss_vs_H25']:.6g} | {p['total_delta_vs_H25']:.6g} |"
    )


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle v1d compute-tradeoff oracle upper-bound postdiagnostic v0",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations/training/refit; no validation64 or sealed-test access.",
        "",
        "## Headline",
        "",
        f"- Input single-run strict/relaxed gate: H10-labelled strict `{raw['input_gate']['strict_confirmed_h10_labelled_count']}/6`, relaxed `{raw['input_gate']['relaxed_confirmed_h10_labelled_count']}/6`, strict controls `{raw['input_gate']['strict_confirmed_negative_control_count']}/4`.",
        f"- Strict state-level H10 oracle chooses H10 in `{raw['policies_by_name']['strict_state_H10_else_H25']['state_count_with_H10']}` of `{raw['state_count']}` states and saves `{raw['policies_by_name']['strict_state_H10_else_H25']['decision_rel_gain_vs_H25']*100.0:.2f}%` decision time vs fixed H25 on this branch-state bank.",
        f"- Best safe measured-time oracle at physical epsilon 0.01 saves `{raw['policies_by_name']['best_time_safe_eps001_oracle']['decision_rel_gain_vs_H25']*100.0:.2f}%` vs H25 but is an upper bound using observed branch outcomes, not a learnable policy result.",
        f"- Decision: `{raw['decision']['classification']}`; next action: `{raw['decision']['next_action']}`.",
        "",
        "## Same-branch policy/oracle table (development-only)",
        "",
        "| policy | H10 modes | H15 modes | H25 modes | other modes | safe modes | decision sum s | decision gain vs H25 s | decision gain vs H25 | solver gain vs H25 s | physical loss vs H25 | total delta vs H25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in raw["policy_order"]:
        lines.append(table_line(raw["policies_by_name"][name]))
    lines += [
        "",
        "## Interpretation",
        "",
        raw["decision"]["interpretation"],
        "",
        "This is not a speed claim: it reuses single-run development branch timings. It only bounds whether a selector/refit is worth preparing before repeated randomized timing and fresh validation.",
        "",
        f"Backup request: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if (OUT / "completed.json").exists():
        done = require_done(OUT / "completed.json")
        tok = done.get("server_api_total_tokens_best_effort") or {}
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "elapsed_since_first_supervisor_event_seconds": done.get("elapsed_since_first_supervisor_event_seconds"), "server_api_total_tokens_best_effort": tok, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    for p in (SINGLE_RAW, SINGLE_DONE, H10_DONE, PROTOCOL):
        if not p.exists():
            raise ContractError(f"missing required input {rel(p)}")
    single_done = require_done(SINGLE_DONE)
    h10_done = require_done(H10_DONE)
    raw_in = read_json(SINGLE_RAW)
    if raw_in.get("validation64_bank_opened") is not False or raw_in.get("sealed_test_accessed") is not False:
        raise ContractError("single-run input has invalid access flags")
    rows = list((raw_in.get("analysis") or {}).get("state_rows") or [])
    if len(rows) != 10:
        raise ContractError(f"expected 10 state rows, got {len(rows)}")
    for row in rows:
        for mode in MODES:
            if arm(row, mode, 25) is None or arm(row, mode, 15) is None or arm(row, mode, 10) is None:
                raise ContractError(f"missing H10/H15/H25 arm for {row.get('state_id')} {mode}")
    policy_order = [
        "fixed_H25",
        "fixed_H15",
        "strict_state_H10_else_H25",
        "relaxed_labelled_H10_else_H25",
        "relaxed_any_state_H10_else_H25",
        "mode_aware_strict_H10_else_H25_oracle",
        "best_time_safe_eps001_oracle",
        "best_time_safe_eps01_oracle",
    ]
    policies = {name: summarize_policy(name, rows) for name in policy_order}
    for name in policy_order:
        add_comparison(policies[name], policies["fixed_H25"], policies["fixed_H15"])
    strict = policies["strict_state_H10_else_H25"]
    eps001 = policies["best_time_safe_eps001_oracle"]
    relaxed = policies["relaxed_labelled_H10_else_H25"]
    strict_rel = fnum(strict.get("decision_rel_gain_vs_H25"))
    eps001_rel = fnum(eps001.get("decision_rel_gain_vs_H25"))
    relaxed_rel = fnum(relaxed.get("decision_rel_gain_vs_H25"))
    if strict["state_count_with_H10"] < 2 and strict_rel < 0.02:
        classification = "strict_compute_selector_upper_bound_too_sparse"
        next_action = "do not run the 186-episode repeated timing confirmation or selector/refit now; freeze a negative current-scenario compute-opportunity diagnosis, then design a versioned source-supported scenario/value diagnostic"
        interpretation = (
            "The stricter terminal-stable H10 rule leaves only one state-level H10 choice. "
            "On the same branch-state bank, even an oracle using that strict rule has too little measured-time leverage versus fixed H25. "
            "Relaxed labels can save more time but rely on epsilon/terminal tolerance and already include at least one relaxed negative-control state, so they are not a stable training target."
        )
    elif eps001_rel >= 0.05 and eps001.get("all_modes_safe") and fnum(eps001.get("physical_loss_vs_H25")) <= 0.05:
        classification = "oracle_time_upper_bound_exists_but_requires_new_label_definition"
        next_action = "after backup, freeze a smaller repeated-timing diagnostic around the oracle-selected states before any selector/refit"
        interpretation = (
            "The perfect observed-outcome oracle still shows a nontrivial strict-epsilon timing bound, but this is not a deployable selector. "
            "A smaller repeated-timing confirmation of exactly the oracle-selected states would be justified before refit."
        )
    elif relaxed_rel >= 0.05:
        classification = "relaxed_compute_opportunity_epsilon_sensitive_not_training_ready"
        next_action = "audit timing noise/objective scale or scenario opportunity; do not train/refit on relaxed-only labels"
        interpretation = (
            "The relaxed rule can cross a timing threshold, but the strict rule does not. This points to objective/threshold sensitivity rather than a robust label set."
        )
    else:
        classification = "compute_opportunity_absorbed_or_too_small_for_refit"
        next_action = "freeze current-scenario negative compute-opportunity diagnosis and pivot to scenario/value design"
        interpretation = "Neither strict nor relaxed oracle summaries provide a robust enough decision-time/control tradeoff to justify selector/refit."
    now = dt.datetime.now(dt.timezone.utc)
    tokens = query_tokens()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {
        "created_utc": now.isoformat(),
        "method": "vehicle_v1d_compute_tradeoff_oracle_upper_bound_postdiagnostic_v0_no_simulation",
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
        "server_api_total_tokens_best_effort": tokens,
        "inputs": {
            "single_run_raw": rel(SINGLE_RAW), "single_run_raw_sha256": sha256(SINGLE_RAW),
            "single_run_completed": rel(SINGLE_DONE), "single_run_completed_sha256": sha256(SINGLE_DONE),
            "h10_absorption_completed": rel(H10_DONE), "h10_absorption_completed_sha256": sha256(H10_DONE),
            "frozen_repeated_timing_protocol": rel(PROTOCOL), "frozen_repeated_timing_protocol_sha256": sha256(PROTOCOL),
        },
        "input_gate": single_done.get("headline") or {},
        "h10_absorption_context": h10_done.get("headline") or {},
        "state_count": len(rows),
        "terminal_modes": list(MODES),
        "policy_order": policy_order,
        "policies_by_name": policies,
        "decision": {"classification": classification, "next_action": next_action, "interpretation": interpretation, "train_or_refit_now": False, "run_186_episode_confirmation_now": False},
        "limits": ["development-only", "single-run timing reused", "oracle policies use observed outcomes and are upper bounds", "not validation64", "not sealed test", "no selector/refit/training"],
        "backup_request": rel(REQ),
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Vehicle v1d compute-tradeoff oracle upper-bound postdiagnostic\n\nUTC: {now.isoformat()}. Strict state oracle H10 states={strict['state_count_with_H10']}/{len(rows)}, decision_gain_vs_H25={strict['decision_rel_gain_vs_H25']*100.0:.2f}%; eps0.01 best-time oracle decision_gain_vs_H25={eps001['decision_rel_gain_vs_H25']*100.0:.2f}%. Classification={classification}. Next={next_action}. No simulations/training/validation/test.\n",
        encoding="utf-8",
    )
    write_json(REQ, {
        "requested_utc": now.isoformat(),
        "reason": "backup v1d compute-tradeoff oracle upper-bound diagnostic before scenario/value pivot or any new simulation",
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
## 2026-09-29 vehicle v1d compute-tradeoff oracle upper-bound postdiagnostic

UTC: {now.isoformat()}. No-simulation upper-bound analysis over existing v1d branch rows: strict state-level H10 oracle selects H10 in {strict['state_count_with_H10']}/{len(rows)} states and saves {strict['decision_rel_gain_vs_H25']*100.0:.2f}% decision time vs fixed H25 on this branch bank; relaxed labelled oracle saves {relaxed['decision_rel_gain_vs_H25']*100.0:.2f}% but is epsilon/terminal sensitive. Classification: `{classification}`. Decision: {next_action}. No validation64/sealed-test access and no training/refit. Backup requested at `{rel(REQ)}` before further simulation.
"""
    append_docs(block)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE, REQ, Path(__file__).resolve(), SINGLE_RAW, SINGLE_DONE, H10_DONE, PROTOCOL]
    done = {
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
        "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
        "server_api_total_tokens_best_effort": tokens,
        "backup_request": rel(REQ),
        "headline": {
            "classification": classification,
            "strict_state_h10_states": strict["state_count_with_H10"],
            "strict_state_decision_rel_gain_vs_H25": strict["decision_rel_gain_vs_H25"],
            "relaxed_labelled_h10_states": relaxed["state_count_with_H10"],
            "relaxed_labelled_decision_rel_gain_vs_H25": relaxed["decision_rel_gain_vs_H25"],
            "best_time_safe_eps001_decision_rel_gain_vs_H25": eps001["decision_rel_gain_vs_H25"],
            "train_or_refit_now": False,
            "run_186_episode_confirmation_now": False,
            "next_action": next_action,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", done)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "classification": classification,
        "strict_state_h10_states": strict["state_count_with_H10"],
        "strict_state_decision_rel_gain_vs_H25": strict["decision_rel_gain_vs_H25"],
        "relaxed_labelled_decision_rel_gain_vs_H25": relaxed["decision_rel_gain_vs_H25"],
        "best_time_safe_eps001_decision_rel_gain_vs_H25": eps001["decision_rel_gain_vs_H25"],
        "next_action": next_action,
        "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
        "server_api_total_tokens_best_effort": tokens,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "next_recovery_hint": "Preserve failure; repair only no-simulation oracle extraction before new simulations.",
        })
        raise
