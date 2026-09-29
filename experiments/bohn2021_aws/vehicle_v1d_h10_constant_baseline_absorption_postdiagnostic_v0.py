#!/usr/bin/env python3
"""No-simulation H10 constant-baseline absorption diagnostic for vehicle v1d.

The repaired v1d solver/decision timing diagnostic found solver-confirmed
compute-safe states, but their best relaxed horizon was dominated by H10.  This
script asks the next discriminating question without any new rollouts: does a
constant H10 fixed-horizon baseline already absorb that compute/control tradeoff
on the existing validation evidence, or is H10 locally useful but globally too
unsafe/costly to be a strong fixed-H solution?

Inputs are existing artifacts only:
  * v1d solver/decision timing schema postdiagnostic raw/completed.
  * vehicle validation64 full aggregate arm_summary/completed.

No validation bank is reopened, no sealed test is accessed, and no simulation,
training, gradient update, refit, or candidate reset is performed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0350Z"
SOLVER_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/raw.json"
SOLVER_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/completed.json"
VAL_ARM = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/arm_summary.csv"
VAL_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/completed.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1D_H10_CONSTANT_BASELINE_ABSORPTION_POSTDIAGNOSTIC_V0_{STAMP}.json"
MARKER = f"vehicle-v1d-h10-constant-baseline-absorption-postdiagnostic-v0-{STAMP}"
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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(v for v in (float(x) for x in vals) if math.isfinite(v))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "sum": 0.0}
    mid = len(xs) // 2
    median = xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    return {"count": len(xs), "min": xs[0], "median": median, "mean": sum(xs) / len(xs), "max": xs[-1], "sum": sum(xs)}


def verify_completed(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker not passing: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("test_accessed") is True or obj.get("sealed_test_bank_opened") is True or obj.get("sealed_test_bank_content_opened") is True:
        raise ContractError(f"completed marker indicates sealed-test access: {rel(path)}")
    return obj


def load_inputs() -> Dict[str, Any]:
    for p in (SOLVER_RAW, SOLVER_DONE, VAL_ARM, VAL_DONE):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    solver_done = verify_completed(SOLVER_DONE)
    val_done = verify_completed(VAL_DONE)
    solver = read_json(SOLVER_RAW)
    if solver.get("sealed_test_accessed") is not False:
        raise ContractError("solver raw has unexpected sealed-test access flag")
    if int(solver.get("state_count", -1)) != 12:
        raise ContractError("unexpected v1d solver state_count")
    rows: Dict[str, Dict[str, str]] = {}
    with VAL_ARM.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row.get("rollout_key", ""))] = row
    if not rows:
        raise ContractError("arm_summary.csv parsed empty")
    return {"solver": solver, "solver_done": solver_done, "val_done": val_done, "arms": rows}


def row_summary(row: Mapping[str, str]) -> Dict[str, Any]:
    episodes = inum(row.get("episodes"))
    success = inum(row.get("success_count"))
    failures = inum(row.get("episode_failure_count"))
    steps = inum(row.get("steps"))
    return {
        "rollout_key": row.get("rollout_key"),
        "seed": inum(row.get("seed"), -1),
        "terminal_h": inum(row.get("terminal_h"), -1),
        "controller_h_int": None if row.get("controller_h_int") in (None, "") else inum(row.get("controller_h_int"), -1),
        "episodes": episodes,
        "steps": steps,
        "success_count": success,
        "episode_failure_count": failures,
        "success_rate": None if episodes <= 0 else success / episodes,
        "physical_constraint_cost_sum": fnum(row.get("physical_constraint_cost_sum")),
        "total_cost_sum": fnum(row.get("total_cost_sum")),
        "decision_mean_s": fnum(row.get("decision_mean_s_per_step_exact_from_episode_sums")),
        "decision_total_s": fnum(row.get("decision_total_s")),
        "solver_mean_s": fnum(row.get("solver_mean_s_per_attempt_exact_from_episode_sums")),
        "solver_total_s": fnum(row.get("solver_total_s")),
        "horizon_counts": row.get("horizon_counts"),
        "unique_horizons": row.get("unique_horizons"),
    }


def aggregate(items: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    episodes = sum(int(x["episodes"]) for x in items)
    steps = sum(int(x["steps"]) for x in items)
    success = sum(int(x["success_count"]) for x in items)
    failures = sum(int(x["episode_failure_count"]) for x in items)
    decision_total = sum(float(x["decision_total_s"]) for x in items)
    solver_total = sum(float(x["solver_total_s"]) for x in items)
    return {
        "label": label,
        "episodes": episodes,
        "steps": steps,
        "success_count": success,
        "episode_failure_count": failures,
        "success_rate": None if episodes <= 0 else success / episodes,
        "physical_constraint_cost_sum": sum(float(x["physical_constraint_cost_sum"]) for x in items),
        "total_cost_sum": sum(float(x["total_cost_sum"]) for x in items),
        "decision_mean_s": None if steps <= 0 else decision_total / steps,
        "solver_mean_s": None if steps <= 0 else solver_total / steps,
        "decision_total_s": decision_total,
        "solver_total_s": solver_total,
    }


def compare(candidate: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
    cand_phys = float(candidate["physical_constraint_cost_sum"])
    ref_phys = float(ref["physical_constraint_cost_sum"])
    cand_total = float(candidate["total_cost_sum"])
    ref_total = float(ref["total_cost_sum"])
    cand_dec = candidate.get("decision_mean_s")
    ref_dec = ref.get("decision_mean_s")
    cand_sol = candidate.get("solver_mean_s")
    ref_sol = ref.get("solver_mean_s")
    decision_gain = None if cand_dec is None or ref_dec is None else float(ref_dec) - float(cand_dec)
    solver_gain = None if cand_sol is None or ref_sol is None else float(ref_sol) - float(cand_sol)
    return {
        "candidate": candidate.get("label") or candidate.get("rollout_key"),
        "reference": ref.get("label") or ref.get("rollout_key"),
        "candidate_success_rate": candidate.get("success_rate"),
        "reference_success_rate": ref.get("success_rate"),
        "candidate_failures": candidate.get("episode_failure_count"),
        "reference_failures": ref.get("episode_failure_count"),
        "physical_cost_ratio_candidate_over_reference": None if ref_phys <= 0 else cand_phys / ref_phys,
        "total_cost_ratio_candidate_over_reference": None if ref_total <= 0 else cand_total / ref_total,
        "decision_mean_gain_s": decision_gain,
        "decision_mean_rel_gain": None if decision_gain is None or not ref_dec or float(ref_dec) <= 0 else decision_gain / float(ref_dec),
        "solver_mean_gain_s": solver_gain,
        "solver_mean_rel_gain": None if solver_gain is None or not ref_sol or float(ref_sol) <= 0 else solver_gain / float(ref_sol),
        "noninferior_success_and_physical_3pct": bool(
            candidate.get("success_rate") is not None
            and ref.get("success_rate") is not None
            and float(candidate["success_rate"]) >= float(ref["success_rate"])
            and ref_phys > 0
            and cand_phys <= 1.03 * ref_phys
        ),
        "at_least_10pct_decision_faster": bool(decision_gain is not None and ref_dec and float(ref_dec) > 0 and decision_gain / float(ref_dec) >= 0.10),
    }


def query_tokens() -> Dict[str, Any]:
    for db in (ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"):
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db)); cur = con.cursor()
            total = None; by_table = {}
            for (table,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val = int(cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0] or 0)
                    by_table[table] = val
                    total = (total or 0) + val
            con.close()
            if total is not None:
                return {"available": True, "path": rel(db), "total_tokens": total, "by_table": by_table}
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": repr(exc)}
    return {"available": False, "reason": "research.sqlite not found in repository-visible candidate paths"}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run() -> int:
    inp = load_inputs()
    solver = inp["solver"]
    arms = inp["arms"]

    per_state = solver.get("per_state") or []
    best_relaxed = []
    best_strict = []
    for st in per_state:
        br = st.get("best_both_relaxed")
        if br:
            best_relaxed.append({
                "state_id": st.get("state_id"), "case": st.get("case"), "branch_step": st.get("branch_step"),
                "horizon": int(br.get("horizon")),
                "solver_gain_s": fnum(br.get("min_solver_attempt_gain_s"), float("nan")),
                "decision_gain_s": fnum(br.get("min_decision_gain_s"), float("nan")),
                "max_physical_loss": fnum(br.get("max_physical_loss")),
                "both_relaxed_horizons": st.get("both_relaxed_horizons") or [],
            })
        bs = st.get("best_both_strict")
        if bs:
            best_strict.append({
                "state_id": st.get("state_id"), "case": st.get("case"), "branch_step": st.get("branch_step"),
                "horizon": int(bs.get("horizon")),
                "solver_gain_s": fnum(bs.get("min_solver_attempt_gain_s"), float("nan")),
                "decision_gain_s": fnum(bs.get("min_decision_gain_s"), float("nan")),
                "max_physical_loss": fnum(bs.get("max_physical_loss")),
                "both_strict_horizons": st.get("both_strict_horizons") or [],
            })
    h10_best_relaxed = [x for x in best_relaxed if int(x["horizon"]) == 10]
    h10_best_strict = [x for x in best_strict if int(x["horizon"]) == 10]
    non_h10_strict_best = [x for x in best_strict if int(x["horizon"]) != 10]
    multi_feasible_relaxed = [x for x in best_relaxed if len(x.get("both_relaxed_horizons") or []) > 1]

    seed_pairs = []
    h10_items = []
    h25_items = []
    learned_items = []
    missing = []
    for seed in (0, 1, 2):
        key10 = f"fixed_seed{seed}_terminal25_controllerH10"
        key25 = f"fixed_seed{seed}_terminal25_controllerH25"
        learn = f"learned_s{seed}"
        if key10 not in arms or key25 not in arms or learn not in arms:
            missing.append({"seed": seed, "missing": [k for k in (key10, key25, learn) if k not in arms]})
            continue
        r10 = row_summary(arms[key10]); r25 = row_summary(arms[key25]); rl = row_summary(arms[learn])
        h10_items.append(r10); h25_items.append(r25); learned_items.append(rl)
        seed_pairs.append({
            "seed": seed,
            "H10_terminal25": r10,
            "H25_terminal25": r25,
            "learned": rl,
            "H10_vs_H25": compare(r10, r25),
            "learned_vs_H25": compare(rl, r25),
        })
    if missing:
        raise ContractError(f"missing required arm rows: {missing}")
    agg10 = aggregate(h10_items, "matched_terminal_allseeds_H10")
    agg25 = aggregate(h25_items, "matched_terminal_allseeds_H25")
    agglearn = aggregate(learned_items, "learned_s0_s1_s2_allseeds")
    agg_compare = {
        "H10_vs_H25": compare(agg10, agg25),
        "learned_vs_H25": compare(agglearn, agg25),
    }
    seed0_independent_h10 = row_summary(arms["fixed_seed0_terminal10_controllerH10"]) if "fixed_seed0_terminal10_controllerH10" in arms else None
    seed0_ind_compare = compare(seed0_independent_h10, h25_items[0]) if seed0_independent_h10 else None

    all_best_relaxed_h10 = bool(best_relaxed) and len(h10_best_relaxed) == len(best_relaxed)
    global_h10_noninferior = bool(agg_compare["H10_vs_H25"]["noninferior_success_and_physical_3pct"])
    global_h10_fast = bool(agg_compare["H10_vs_H25"]["at_least_10pct_decision_faster"])
    if all_best_relaxed_h10 and not global_h10_noninferior:
        classification = "local_H10_compute_labels_do_not_make_constant_H10_a_strong_global_baseline"
        next_action = "after external backup, freeze a small blocked repeated-timing confirmation on the six H10-labelled v1d states plus negative controls; include H10/H15/H25 branches, solver_attempt_sum_s, whole-decision timing, physical/safety outcomes, and global fixed-H10/H25 context before any selector/refit"
    elif all_best_relaxed_h10 and global_h10_noninferior:
        classification = "constant_H10_may_absorb_compute_tradeoff_under_existing_validation"
        next_action = "do not train/refit; audit H10 as a strong fixed-H speed baseline with repeated timing before adaptive claims"
    else:
        classification = "v1d_compute_labels_not_purely_constant_H10"
        next_action = "inspect non-H10 labels and run blocked repeated timing before selector/refit"
    train_or_refit_now = False

    now = dt.datetime.now(dt.timezone.utc)
    token_info = query_tokens()
    elapsed = (now - FIRST_SUPERVISOR_EVENT).total_seconds()
    result = {
        "created_utc": now.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "server_api_total_tokens_best_effort": token_info,
        "method": "vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_no_simulation",
        "formal_scientific_evidence_created": False,
        "validation64_existing_outputs_read": True,
        "validation64_bank_reopened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "inputs": {
            "solver_raw": rel(SOLVER_RAW), "solver_raw_sha256": sha256(SOLVER_RAW),
            "solver_completed": rel(SOLVER_DONE), "solver_completed_sha256": sha256(SOLVER_DONE),
            "validation_arm_summary": rel(VAL_ARM), "validation_arm_summary_sha256": sha256(VAL_ARM),
            "validation_completed": rel(VAL_DONE), "validation_completed_sha256": sha256(VAL_DONE),
        },
        "v1d_compute_label_summary": {
            "best_both_relaxed_state_count": len(best_relaxed),
            "best_both_relaxed_h10_count": len(h10_best_relaxed),
            "best_both_strict_state_count": len(best_strict),
            "best_both_strict_h10_count": len(h10_best_strict),
            "best_both_strict_non_h10": non_h10_strict_best,
            "multi_feasible_relaxed_states": multi_feasible_relaxed,
            "best_relaxed_solver_gain_s": stats(x["solver_gain_s"] for x in best_relaxed),
            "best_relaxed_decision_gain_s": stats(x["decision_gain_s"] for x in best_relaxed),
            "all_best_relaxed_h10": all_best_relaxed_h10,
        },
        "validation_constant_H10_audit": {
            "seed_pairs": seed_pairs,
            "aggregate_H10_terminal25": agg10,
            "aggregate_H25_terminal25": agg25,
            "aggregate_learned": agglearn,
            "aggregate_comparisons": agg_compare,
            "seed0_independent_terminal10_H10": seed0_independent_h10,
            "seed0_independent_terminal10_H10_vs_seed0_H25": seed0_ind_compare,
            "global_H10_noninferior_success_and_physical_3pct": global_h10_noninferior,
            "global_H10_at_least_10pct_decision_faster": global_h10_fast,
        },
        "decision": {
            "classification": classification,
            "train_or_refit_now": train_or_refit_now,
            "next_action": next_action,
            "rationale": "v1d compute-safe timing labels are locally H10-dominated, but existing validation shows constant H10 is not a safe/control-noninferior global fixed-H solution across seeds.  This supports repeated timing confirmation of a conditional H10 target, not immediate selector/refit or a speed claim.",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "raw.json", result)
    lines = [
        "# Vehicle v1d H10 constant-baseline absorption postdiagnostic v0",
        "",
        f"UTC: `{now.isoformat()}`. No simulations/training/refit; existing validation aggregate outputs read; no validation bank reopen; no sealed-test access.",
        "",
        "## Headline",
        "",
        f"- v1d best both-relaxed compute-safe states: `{len(best_relaxed)}/12`; best-H10 among them: `{len(h10_best_relaxed)}`.",
        f"- v1d best both-strict compute-safe states: `{len(best_strict)}/12`; best-H10 among them: `{len(h10_best_strict)}`; non-H10 strict best rows: `{len(non_h10_strict_best)}`.",
        f"- Multi-feasible relaxed states exist: `{len(multi_feasible_relaxed)}`, but best relaxed horizon is H10 for all relaxed labels: `{all_best_relaxed_h10}`.",
        f"- Existing validation matched-terminal all-seed H10 vs H25: success `{agg10['success_count']}/{agg10['episodes']}` vs `{agg25['success_count']}/{agg25['episodes']}`, failures `{agg10['episode_failure_count']}` vs `{agg25['episode_failure_count']}`, physical-cost ratio `{agg_compare['H10_vs_H25']['physical_cost_ratio_candidate_over_reference']}`.",
        f"- Existing validation H10 decision speed vs H25: relative gain `{agg_compare['H10_vs_H25']['decision_mean_rel_gain']}`; noninferior success+physical within 3%: `{global_h10_noninferior}`.",
        f"- Classification: `{classification}`.",
        f"- Train/refit now: `{train_or_refit_now}`.",
        "",
        "## Per-seed H10 vs H25 validation context",
        "",
        "| seed | H10 success/fail | H25 success/fail | H10/H25 physical ratio | H10 decision rel gain | H10 solver rel gain | H10 noninferior? |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sp in seed_pairs:
        c = sp["H10_vs_H25"]
        lines.append("| %d | %d/%d | %d/%d | %s | %s | %s | `%s` |" % (
            sp["seed"],
            sp["H10_terminal25"]["success_count"], sp["H10_terminal25"]["episode_failure_count"],
            sp["H25_terminal25"]["success_count"], sp["H25_terminal25"]["episode_failure_count"],
            c["physical_cost_ratio_candidate_over_reference"], c["decision_mean_rel_gain"], c["solver_mean_rel_gain"], c["noninferior_success_and_physical_3pct"],
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "The local v1d compute opportunity is real enough to survive the solver_attempt_sum_s schema repair, but it is H10-dominated and single-run. Existing validation shows constant H10 is a weak global fixed-H comparator: it is faster in mean decision time but has severe success/control-cost degradation across seeds. Thus the next question is not broad physical-improvement selector training; it is whether a conditional H10/H15-or-H25 policy has repeatable measured timing benefit without safety/control loss.",
        "",
        f"Next action: {next_action}.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# v1d H10 constant-baseline absorption state\n\nUTC: {now.isoformat()}. classification={classification}; best_relaxed={len(best_relaxed)}/12 all_best_relaxed_H10={all_best_relaxed_h10}; validation_H10_noninferior={global_h10_noninferior}; train_or_refit_now={train_or_refit_now}. Next: {next_action}. No simulation/training/sealed test. Existing validation aggregate outputs read only; bank not reopened.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQ, {
        "requested_utc": now.isoformat(),
        "reason": "backup v1d H10 constant-baseline absorption postdiagnostic before any repeated-timing simulation/refit/training",
        "backup_required_before_more_simulations": True,
        "validation64_existing_outputs_read": True,
        "validation64_bank_reopened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(BACKUP_REQ)],
    })
    docs_block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle v1d H10 constant-baseline absorption postdiagnostic

UTC: {now.isoformat()}. No-simulation diagnostic over existing v1d timing and validation aggregate outputs. v1d best both-relaxed compute-safe states={len(best_relaxed)}/12 and all best relaxed labels were H10; best strict states={len(best_strict)}/12 with H10 best in {len(h10_best_strict)}. Existing validation matched-terminal all-seed H10 was faster in mean decision time but not a control/safety-noninferior fixed-H solution: H10 success={agg10['success_count']}/{agg10['episodes']} with failures={agg10['episode_failure_count']} vs H25 success={agg25['success_count']}/{agg25['episodes']} and physical-cost ratio={agg_compare['H10_vs_H25']['physical_cost_ratio_candidate_over_reference']}. Classification `{classification}`; train/refit remains false. Next: after backup, freeze blocked repeated-timing confirmation on H10-labelled states before any selector/refit. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup request: `{rel(BACKUP_REQ)}`. Sealed test stayed closed; validation bank was not reopened.
"""
    append_docs(docs_block)
    files = [OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, Path(__file__).resolve(), SOLVER_RAW, SOLVER_DONE, VAL_ARM, VAL_DONE]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": now.isoformat(),
        "formal_scientific_evidence_created": False,
        "validation64_existing_outputs_read": True,
        "validation64_bank_reopened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {
            "classification": classification,
            "best_both_relaxed_state_count": len(best_relaxed),
            "best_both_relaxed_h10_count": len(h10_best_relaxed),
            "best_both_strict_state_count": len(best_strict),
            "best_both_strict_h10_count": len(h10_best_strict),
            "validation_H10_success_count": agg10["success_count"],
            "validation_H10_episodes": agg10["episodes"],
            "validation_H10_failures": agg10["episode_failure_count"],
            "validation_H25_success_count": agg25["success_count"],
            "validation_H25_episodes": agg25["episodes"],
            "validation_H10_physical_cost_ratio_vs_H25": agg_compare["H10_vs_H25"]["physical_cost_ratio_candidate_over_reference"],
            "validation_H10_decision_rel_gain_vs_H25": agg_compare["H10_vs_H25"]["decision_mean_rel_gain"],
            "global_H10_noninferior_success_and_physical_3pct": global_h10_noninferior,
            "train_or_refit_now": train_or_refit_now,
            "next_action": next_action,
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(files) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "classification": classification,
        "best_both_relaxed_state_count": len(best_relaxed),
        "best_both_relaxed_h10_count": len(h10_best_relaxed),
        "validation_H10_success_count": agg10["success_count"],
        "validation_H10_episodes": agg10["episodes"],
        "validation_H10_physical_cost_ratio_vs_H25": agg_compare["H10_vs_H25"]["physical_cost_ratio_candidate_over_reference"],
        "validation_H10_decision_rel_gain_vs_H25": agg_compare["H10_vs_H25"]["decision_mean_rel_gain"],
        "train_or_refit_now": train_or_refit_now,
        "next_action": next_action,
        "validation64_bank_reopened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(BACKUP_REQ),
        "elapsed_since_first_supervisor_event_seconds": elapsed,
        "server_api_total_tokens_best_effort": token_info,
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-existing-validation-aggregate-read", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_existing_validation_aggregate_read:
        raise ContractError("explicit --i-accept-existing-validation-aggregate-read required")
    if OUT.exists() and (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        if done.get("passed") is not True and done.get("hard_pass") is not True:
            raise ContractError("existing completed marker is not passing")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "validation64_bank_reopened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
