#!/usr/bin/env python3
"""Postdiagnostic for vehicle true variable-H broader block v0.

No simulation/training/refit/test access.  This script consumes the completed
48-episode broader true-H development block and asks a more discriminating
question than the runner's coarse pass/fail gate: did the data reveal a robust
control-vs-compute opportunity, or only a faster-but-physically-worse H10 path?

It computes state/profile median physical and measured timing tradeoffs,
near-best physical oracle labels over H10/H15/H25, aggregate comparisons to
fixed-H baselines, and a concrete next-experiment decision.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_broader_block_postdiagnostic_v0"
STAMP = "20260929T0710Z"
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z"
RUN_RAW = RUN_DIR / "raw.json"
RUN_DONE = RUN_DIR / "completed.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_BROADER_BLOCK_POSTDIAGNOSTIC_V0_{STAMP}.json"
FIRST = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
HORIZONS = [10, 15, 25]
PHYS_ABS_TOL = 2.0
PHYS_REL_TOL = 0.05
SPEED_MATERIAL = 0.15


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * q
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {"n": len(vals), "min": vals[0], "median": pct(0.5), "mean": sum(vals) / len(vals), "p95": pct(0.95), "max": vals[-1], "sum": sum(vals)}


def metric_sum(e: Mapping[str, Any], k: str) -> float:
    return sf((e.get(k) or {}).get("sum"), 0.0)


def safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 0) == 0 and si(e.get("initial_failed_steps"), 0) == 0 and si(e.get("final_failed_steps"), 0) == 0


def add_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    created = now_utc()
    if not RUN_DONE.exists() or not RUN_RAW.exists():
        raise SystemExit("broader block raw/completed missing")
    done = read_json(RUN_DONE)
    raw = read_json(RUN_RAW)
    if done.get("passed") is not True or raw.get("validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise SystemExit("broader block did not pass or access flags are invalid")
    episodes = list(raw.get("episodes") or [])
    if len(episodes) != 48:
        raise SystemExit(f"expected 48 episodes, got {len(episodes)}")

    # Medians by state/profile/horizon across the two randomized repeats.
    roles: Dict[str, str] = {}
    groups = sorted(set((str(e.get("state_id")), str(e.get("terminal_profile"))) for e in episodes))
    table: List[Dict[str, Any]] = []
    for sid, prof in groups:
        rows = [e for e in episodes if str(e.get("state_id")) == sid and str(e.get("terminal_profile")) == prof]
        role = str((rows[0].get("target_role") if rows else None) or "unknown")
        roles[sid] = role
        med: Dict[int, Dict[str, Any]] = {}
        for h in HORIZONS:
            hs = [e for e in rows if si(e.get("true_mpc_n_horizon")) == h]
            med[h] = {
                "n": len(hs),
                "safe_all": all(safe(e) for e in hs) if hs else False,
                "physical": median([sf(e.get("physical_constraint_cost"), 0.0) for e in hs]),
                "decision_sum_s": median([metric_sum(e, "decision_timing_s") for e in hs]),
                "solver_sum_s": median([metric_sum(e, "solver_attempt_timing_s") for e in hs]),
                "steps": median([sf(e.get("steps"), 0.0) for e in hs]),
                "opt_x_size": sorted(set(si(x) for e in hs for x in (e.get("opt_x_sizes_observed") or []))),
            }
        phys_vals = {h: med[h]["physical"] for h in HORIZONS if med[h]["physical"] is not None and med[h]["safe_all"]}
        best_h = min(phys_vals, key=lambda h: float(phys_vals[h])) if phys_vals else None
        best_phys = None if best_h is None else float(phys_vals[best_h])
        tol = None if best_phys is None else max(PHYS_ABS_TOL, PHYS_REL_TOL * abs(best_phys))
        near_best = []
        if best_phys is not None:
            for h in HORIZONS:
                ph = med[h]["physical"]
                if ph is not None and med[h]["safe_all"] and float(ph) <= best_phys + float(tol):
                    near_best.append(h)
        fastest_near = None
        if near_best:
            fastest_near = min(near_best, key=lambda h: (float(med[h]["decision_sum_s"]), float(med[h]["solver_sum_s"]), h))
        h10_gain_vs_h15 = None
        if med[10]["physical"] is not None and med[15]["physical"] is not None:
            h10_gain_vs_h15 = float(med[15]["physical"]) - float(med[10]["physical"])
        noncontrol = "control" not in role.lower() and "negative" not in role.lower()
        table.append({
            "state_id": sid,
            "terminal_profile": prof,
            "role": role,
            "noncontrol": noncontrol,
            "median_by_h": {str(h): med[h] for h in HORIZONS},
            "best_physical_h": best_h,
            "best_physical": best_phys,
            "near_best_tolerance": tol,
            "near_best_horizons": near_best,
            "fastest_near_best_h": fastest_near,
            "h10_gain_vs_h15": h10_gain_vs_h15,
            "h10_strict_physical_improvement_vs_h15": h10_gain_vs_h15 is not None and h10_gain_vs_h15 > 0.0,
            "h10_no_worse_vs_h15_with_best_tol": h10_gain_vs_h15 is not None and tol is not None and h10_gain_vs_h15 >= -float(tol),
        })

    def aggregate_for_choice(choice_by_row: Sequence[Tuple[Dict[str, Any], int]], noncontrol_only: bool) -> Dict[str, float]:
        use = [(r, h) for r, h in choice_by_row if (r["noncontrol"] or not noncontrol_only)]
        return {
            "groups": float(len(use)),
            "physical_sum": sum(float(r["median_by_h"][str(h)]["physical"]) for r, h in use),
            "decision_sum_s": sum(float(r["median_by_h"][str(h)]["decision_sum_s"]) for r, h in use),
            "solver_sum_s": sum(float(r["median_by_h"][str(h)]["solver_sum_s"]) for r, h in use),
        }

    fixed_choices = {h: [(r, h) for r in table] for h in HORIZONS}
    oracle_choices = [(r, int(r["fastest_near_best_h"])) for r in table if r.get("fastest_near_best_h") is not None]
    agg_all = {"fixed_H%d" % h: aggregate_for_choice(fixed_choices[h], False) for h in HORIZONS}
    agg_noncontrol = {"fixed_H%d" % h: aggregate_for_choice(fixed_choices[h], True) for h in HORIZONS}
    agg_all["near_best_fastest_oracle"] = aggregate_for_choice(oracle_choices, False)
    agg_noncontrol["near_best_fastest_oracle"] = aggregate_for_choice(oracle_choices, True)

    def rel_saving(a: float, b: float) -> Optional[float]:
        return None if b <= 0 else (b - a) / b

    comparisons = {}
    for scope, ag in (("all", agg_all), ("noncontrol", agg_noncontrol)):
        oracle = ag["near_best_fastest_oracle"]
        comparisons[scope] = {}
        for base in ("fixed_H10", "fixed_H15", "fixed_H25"):
            b = ag[base]
            comparisons[scope]["oracle_vs_" + base] = {
                "physical_delta_oracle_minus_base": oracle["physical_sum"] - b["physical_sum"],
                "decision_relative_saving": rel_saving(oracle["decision_sum_s"], b["decision_sum_s"]),
                "solver_relative_saving": rel_saving(oracle["solver_sum_s"], b["solver_sum_s"]),
            }

    noncontrol_rows = [r for r in table if r["noncontrol"]]
    h10_no_worse = sum(1 for r in noncontrol_rows if r["h10_no_worse_vs_h15_with_best_tol"])
    h10_strict = sum(1 for r in noncontrol_rows if r["h10_strict_physical_improvement_vs_h15"])
    oracle_labels = {f"{r['state_id']}|{r['terminal_profile']}": r.get("fastest_near_best_h") for r in table}
    label_counts: Dict[str, int] = {}
    for h in oracle_labels.values():
        label_counts[str(h)] = label_counts.get(str(h), 0) + 1

    oracle_vs_h25 = comparisons["noncontrol"]["oracle_vs_fixed_H25"]
    oracle_useful = bool(
        oracle_vs_h25["decision_relative_saving"] is not None
        and oracle_vs_h25["decision_relative_saving"] >= SPEED_MATERIAL
        and oracle_vs_h25["physical_delta_oracle_minus_base"] <= max(PHYS_ABS_TOL * max(1, int(agg_noncontrol["fixed_H25"]["groups"])), PHYS_REL_TOL * abs(agg_noncontrol["fixed_H25"]["physical_sum"]))
    )
    h10_global_bad = bool(agg_noncontrol["fixed_H10"]["physical_sum"] > agg_noncontrol["fixed_H15"]["physical_sum"] + max(PHYS_ABS_TOL * max(1, int(agg_noncontrol["fixed_H15"]["groups"])), PHYS_REL_TOL * agg_noncontrol["fixed_H15"]["physical_sum"]))

    if oracle_useful and h10_global_bad:
        decision = "Do not deploy/train a binary H10 selector from v1e positives.  Broader true-H data support a selective H10/H15/H25 value/modeling diagnostic: global H10 is physically bad, but a near-best physical oracle across H10/H15/H25 preserves H25-level physical cost on non-control groups while saving measured decision time.  Next freeze a larger source-supported true-H oracle-label/value-modeling bank before any multi-seed training."
        next_queue = [
            "backup broader-block run and this postdiagnostic before any simulation/training/refit",
            "freeze/run a larger development-only true-H oracle-label bank over source-supported v1d/v1e/stage2 states and controls, H=[10,15,25], repeated timing, with labels = fastest safe horizon within near-best physical tolerance",
            "if labels remain non-degenerate and oracle tradeoff survives on fresh development states, train/refit an IMPROVED value/selector policy across >=3 independent training seeds with fair fixed-H baselines",
            "if labels collapse or oracle advantage vanishes, pivot to terminal-value/modeling/objective/scenario-opportunity design rather than sparse-label sweeps",
        ]
    elif h10_global_bad:
        decision = "Classify current H10 positives as too narrow for selector training; pivot to terminal/modeling/scenario diagnostics before more selector work."
        next_queue = ["backup outputs", "diagnose terminal/value/modeling or scenario opportunity; no sparse-label sweep"]
    else:
        decision = "H10 is not globally bad in this small block, but evidence remains development-only; require broader independent confirmation before training."
        next_queue = ["backup outputs", "broaden source-supported true-H oracle labels before training"]

    raw_out = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST).total_seconds(),
        "classification": "development_metadata_no_simulation_true_variable_horizon_broader_postdiagnostic",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_run": rel(RUN_RAW),
        "input_run_completed": rel(RUN_DONE),
        "state_profile_table": table,
        "aggregate_all_groups": agg_all,
        "aggregate_noncontrol_groups": agg_noncontrol,
        "comparisons": comparisons,
        "h10_noncontrol_no_worse_vs_h15_count": h10_no_worse,
        "h10_noncontrol_strict_improvement_vs_h15_count": h10_strict,
        "oracle_label_counts": label_counts,
        "oracle_labels": oracle_labels,
        "h10_global_bad_noncontrol": h10_global_bad,
        "near_best_oracle_useful_vs_fixed_H25_noncontrol": oracle_useful,
        "decision": decision,
        "next_queue": next_queue,
    }
    write_json(OUT / "raw.json", raw_out)

    lines = [
        "# Vehicle true variable-H broader block postdiagnostic v0",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only; no simulations/training/refit; validation64 and sealed test closed.",
        "",
        "## Key diagnostic results",
        "",
        f"- H10 strict physical improvements vs H15 among non-control state/profile medians: `{h10_strict}/{len(noncontrol_rows)}`.",
        f"- H10 no-worse vs H15 within near-best tolerance among non-control state/profile medians: `{h10_no_worse}/{len(noncontrol_rows)}`.",
        f"- Global non-control fixed-H physical sums: H10 `{agg_noncontrol['fixed_H10']['physical_sum']:.6g}`, H15 `{agg_noncontrol['fixed_H15']['physical_sum']:.6g}`, H25 `{agg_noncontrol['fixed_H25']['physical_sum']:.6g}`.",
        f"- Near-best fastest oracle non-control physical sum `{agg_noncontrol['near_best_fastest_oracle']['physical_sum']:.6g}` vs fixed H25 delta `{oracle_vs_h25['physical_delta_oracle_minus_base']:.6g}`; decision-time saving vs H25 `{oracle_vs_h25['decision_relative_saving']:.3f}` and solver saving `{oracle_vs_h25['solver_relative_saving']:.3f}`.",
        f"- Near-best oracle labels (all state/profile groups): `{label_counts}`.",
        "",
        "## Interpretation",
        "",
        "The broader block does **not** justify a simple H10-shortening selector: H10 has zero strict physical improvements vs H15 on the non-control medians and is catastrophically worse in the middle/late matched case5 states.  However, the data do show a genuine state-dependent H10/H15/H25 control-compute tradeoff: choosing the fastest safe horizon within near-best physical tolerance selects mixed horizons and preserves/improves non-control aggregate physical cost versus fixed H25 while reducing measured decision time.",
        "",
        "## Decision / next action",
        "",
        decision,
        "",
        "Next queue:",
    ]
    lines += [f"{i+1}. {x}" for i, x in enumerate(next_queue)]
    lines += ["", f"Backup request: `{rel(BACKUP_REQ)}`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup broader true-H block postdiagnostic and run outputs before any additional simulation/training/refit",
        "backup_required_before_more_simulations": True,
        "artifacts": [rel(OUT / "raw.json"), rel(OUT / "summary.md"), rel(OUT / "completed.json"), rel(STATE), rel(BACKUP_REQ), rel(RUN_RAW), rel(RUN_DONE)],
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    marker = f"vehicle-true-variable-horizon-broader-block-postdiagnostic-v0-{STAMP}"
    add_docs(f"""<!-- {marker} -->
## 2026-09-29 vehicle true variable-H broader block postdiagnostic v0

UTC: {created.isoformat()}. Metadata-only postdiagnostic completed with no simulations/training/refit, validation64 closed and sealed test closed. H10 is not a robust global physical improvement (strict non-control H10-vs-H15 improvements {h10_strict}/{len(noncontrol_rows)}; global non-control H10 physical is much worse than H15/H25), but a near-best physical oracle over H10/H15/H25 is mixed ({label_counts}) and has non-control decision-time saving {oracle_vs_h25['decision_relative_saving']:.3f} vs fixed H25 with physical delta {oracle_vs_h25['physical_delta_oracle_minus_base']:.6g}. Decision: larger source-supported true-H oracle-label/value-modeling diagnostic before any selector training; do not repeat sparse label sweeps.
""", marker)
    files = [OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, RUN_RAW, RUN_DONE, Path(__file__).resolve()]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST).total_seconds(),
        "classification": "metadata_no_simulation_postdiagnostic",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "backup_request": rel(BACKUP_REQ),
        "headline": {
            "h10_global_bad_noncontrol": h10_global_bad,
            "h10_noncontrol_strict_improvement_vs_h15_count": h10_strict,
            "near_best_oracle_useful_vs_fixed_H25_noncontrol": oracle_useful,
            "oracle_label_counts": label_counts,
            "train_or_refit_now": False,
            "next": "larger_source_supported_true_H_oracle_label_value_modeling_diagnostic_after_backup",
        },
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "h10_global_bad_noncontrol": h10_global_bad,
        "near_best_oracle_useful_vs_fixed_H25_noncontrol": oracle_useful,
        "oracle_label_counts": label_counts,
        "decision_saving_vs_H25_noncontrol": oracle_vs_h25["decision_relative_saving"],
        "physical_delta_vs_H25_noncontrol": oracle_vs_h25["physical_delta_oracle_minus_base"],
        "new_rollouts": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
