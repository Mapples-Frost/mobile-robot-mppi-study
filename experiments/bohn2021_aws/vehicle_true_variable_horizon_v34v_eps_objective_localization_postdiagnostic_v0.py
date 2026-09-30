#!/usr/bin/env python3
"""v34v zero-solve postdiagnostic for A13c-3 objective residuals.

This script performs no new optimization, no plant rollout, no validation-bank
access, no sealed-test access, and no training/refit. It reads the already
completed v34u/v34t A13c-3 artifact and localizes the observed hard objective
contract failure numerically across the recorded 16 v34k alias candidates.

Purpose: determine whether the large forced-nonconverged residual is explained
by the epsilon/slack contribution that was zero/tied at the previously converged
v34m/v34k point, while preserving the frozen failed G2 result as failed.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v34v_eps_objective_localization_postdiagnostic_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
INPUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z"
RAW = INPUT_DIR / "raw.json"
COMPLETED = INPUT_DIR / "completed.json"
CELL_CSV = INPUT_DIR / "cell_metrics.csv"
ALIAS_CSV = INPUT_DIR / "alias_16_candidate_metrics.csv"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34v_eps_objective_localization_postdiagnostic.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34V_EPS_OBJECTIVE_LOCALIZATION_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-v34v-eps-objective-localization-{STAMP}"
REQUEST_ID = f"v34v-eps-objective-localization-{STAMP}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(v: Any) -> Any:
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


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


def fnum(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        out = float(x)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def finite_min(rows: Iterable[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [fnum(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return None if not vals else float(min(vals))


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    missing = [rel(p) for p in [RAW, COMPLETED, CELL_CSV, ALIAS_CSV] if not p.exists()]
    if missing:
        raise RuntimeError("missing required v34u inputs: " + repr(missing))

    raw = read_json(RAW)
    completed = read_json(COMPLETED)
    cell_rows = read_csv(CELL_CSV)
    alias_rows = read_csv(ALIAS_CSV)
    aliases_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in alias_rows:
        aliases_by_arm.setdefault(str(r.get("arm_id")), []).append(r)

    per_arm: List[Dict[str, Any]] = []
    for c in cell_rows:
        arm = str(c.get("arm_id"))
        rows = aliases_by_arm.get(arm, [])
        eps_rows = [r for r in rows if "|eps=True|" in str(r.get("candidate_id"))]
        noeps_rows = [r for r in rows if "|eps=False|" in str(r.get("candidate_id"))]
        main_rows = [r for r in rows if r.get("is_main_candidate") in ("True", True)]
        best_eps_rel = finite_min(eps_rows, "relative_error")
        best_noeps_rel = finite_min(noeps_rows, "relative_error")
        best_eps_abs = finite_min(eps_rows, "absolute_error")
        best_noeps_abs = finite_min(noeps_rows, "absolute_error")
        eps_terms = [fnum(r.get("epsterm_total")) for r in eps_rows]
        eps_terms = [v for v in eps_terms if v is not None]
        epsterm_min = None if not eps_terms else float(min(eps_terms))
        epsterm_max = None if not eps_terms else float(max(eps_terms))
        abs_resid = fnum(c.get("abs_residual"))
        eps_delta = fnum(c.get("eps_delta_vs_main"))
        relerr = fnum(c.get("relative_error"))
        direct_f_rel = fnum(c.get("direct_f_rel_error_vs_solver"))
        direct_g = fnum(c.get("direct_g_linf_vs_saved"))
        per_arm.append({
            "arm_id": arm,
            "role": c.get("role"),
            "horizon": int(float(c.get("horizon"))) if c.get("horizon") not in (None, "") else None,
            "initialization": c.get("initialization"),
            "return_status": c.get("return_status"),
            "solve_calls": int(float(c.get("solve_calls"))) if c.get("solve_calls") not in (None, "") else 0,
            "forced_nonconverged_or_near_offoptimal": str(c.get("forced_nonconverged_or_near_offoptimal")).lower() == "true",
            "frozen_noeps_relative_error": relerr,
            "frozen_noeps_abs_residual": abs_resid,
            "best_noeps_relative_error": best_noeps_rel,
            "best_noeps_absolute_error": best_noeps_abs,
            "best_eps_relative_error": best_eps_rel,
            "best_eps_absolute_error": best_eps_abs,
            "eps_delta_vs_main": eps_delta,
            "epsterm_min_across_eps_aliases": epsterm_min,
            "epsterm_max_across_eps_aliases": epsterm_max,
            "abs_residual_minus_eps_delta_abs": None if abs_resid is None or eps_delta is None else float(abs(abs_resid - abs(eps_delta))),
            "eps_term_range_width": None if epsterm_min is None or epsterm_max is None else float(epsterm_max - epsterm_min),
            "direct_f_rel_error_vs_solver": direct_f_rel,
            "direct_g_linf_vs_saved": direct_g,
            "alias_rows": len(rows),
            "eps_alias_rows": len(eps_rows),
            "noeps_alias_rows": len(noeps_rows),
            "main_candidate_rows": len(main_rows),
            "localized_to_omitted_eps": bool(
                relerr is not None and relerr > 1e-4
                and best_eps_rel is not None and best_eps_rel <= 1e-12
                and abs_resid is not None and eps_delta is not None and abs(abs_resid - abs(eps_delta)) <= 1e-8
                and epsterm_min is not None and epsterm_max is not None and abs(epsterm_max - epsterm_min) <= 1e-8
                and direct_f_rel is not None and direct_f_rel <= 1e-12
                and direct_g is not None and direct_g <= 1e-12
            ),
        })

    headline_in = (completed.get("headline") or {})
    constructor_gate = bool(headline_in.get("T_B3_A_constructor_binding_gate_all_cells"))
    effective_capture_count = int(headline_in.get("T_B3_A_effective_max_iter_1_capture_count") or 0)
    direct_cells = int(headline_in.get("direct_nlp_eval_available_cells") or 0)
    forced_cells = int(headline_in.get("forced_nonconverged_or_near_offoptimal_cells") or 0)
    alias_rows_expected = 4 * 16
    now = dt.datetime.now(dt.timezone.utc)
    headline = {
        "diagnostic_pass": bool(
            constructor_gate
            and effective_capture_count == 4
            and direct_cells == 4
            and forced_cells == 4
            and len(alias_rows) == alias_rows_expected
            and len(per_arm) == 4
            and all(a["localized_to_omitted_eps"] for a in per_arm)
        ),
        "input_v34u_hard_pass": bool(completed.get("hard_pass")),
        "input_v34u_G2_pass": bool(headline_in.get("G2_pass")),
        "input_v34u_hard_objective_contract_defect": bool(headline_in.get("G2_hard_objective_contract_defect")),
        "constructor_binding_gate_all_cells": constructor_gate,
        "effective_max_iter_1_capture_count": effective_capture_count,
        "forced_nonconverged_or_near_offoptimal_cells": forced_cells,
        "direct_nlp_eval_available_cells": direct_cells,
        "alias_rows_total": len(alias_rows),
        "per_arm_localized_to_omitted_eps_count": int(sum(1 for a in per_arm if a["localized_to_omitted_eps"])),
        "frozen_noeps_relative_error_range": [min(a["frozen_noeps_relative_error"] for a in per_arm if a["frozen_noeps_relative_error"] is not None), max(a["frozen_noeps_relative_error"] for a in per_arm if a["frozen_noeps_relative_error"] is not None)],
        "best_eps_relative_error_max": max(a["best_eps_relative_error"] for a in per_arm if a["best_eps_relative_error"] is not None),
        "best_eps_absolute_error_max": max(a["best_eps_absolute_error"] for a in per_arm if a["best_eps_absolute_error"] is not None),
        "max_abs_residual_minus_eps_delta_abs": max(a["abs_residual_minus_eps_delta_abs"] for a in per_arm if a["abs_residual_minus_eps_delta_abs"] is not None),
        "interpretation": "The frozen no-epsilon objective formula failed under forced infeasible/nonconverged iterates; the recorded epsilon/slack term exactly accounts for the residual in all four cells. This does not retroactively pass v34u G2, change acceptance thresholds, or authorize Task-C without active-lead disposition.",
        "budget_actual": {
            "new_solver_calls": 0,
            "plant_steps": 0,
            "env_step_calls_after_construction": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
    }

    comparison_csv = RUN_DIR / "eps_alias_localization.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(per_arm[0].keys()) if per_arm else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_arm:
            w.writerow({k: clean(v) for k, v in row.items()})

    raw_out = {
        "created_utc": now.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (now - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "zero_solve_development_postdiagnostic_not_validation_not_test",
        "purpose": "Localize the v34u A13c-3 hard objective-contract defect using already recorded direct NLP and alias-candidate evidence.",
        "input_artifacts": {
            "completed": rel(COMPLETED),
            "raw": rel(RAW),
            "cell_metrics_csv": rel(CELL_CSV),
            "alias_16_candidate_metrics_csv": rel(ALIAS_CSV),
        },
        "input_hashes": {rel(p): sha256(p) for p in [COMPLETED, RAW, CELL_CSV, ALIAS_CSV, Path(__file__).resolve()]},
        "headline": headline,
        "per_arm": per_arm,
        "no_new_science_limits": [
            "zero new solver calls",
            "zero plant/env.step calls",
            "zero validation64 episodes",
            "zero sealed/final test access",
            "does not change v34u hard_pass=false",
            "does not authorize Task-C; active lead must decide whether to amend the objective accessor/formula and rerun a bounded gate",
        ],
    }
    write_json(RUN_DIR / "raw.json", raw_out)

    summary_lines = [
        "# v34v epsilon/slack objective-localization postdiagnostic",
        "",
        f"UTC: `{now.isoformat()}`. Zero-solve postdiagnostic of v34u A13c-3; not validation, not final-test evidence.",
        "",
        "## Budget",
        "- New solver calls: `0`; plant/env.step: `0`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Headline",
        f"- Diagnostic pass (localization only): `{headline['diagnostic_pass']}`.",
        f"- Input v34u G2/hard_pass remains: `G2_pass={headline['input_v34u_G2_pass']}`, `hard_pass={headline['input_v34u_hard_pass']}`, hard objective defect `{headline['input_v34u_hard_objective_contract_defect']}`.",
        f"- Constructor binding/effective max_iter evidence: gate_all_cells=`{constructor_gate}`, effective captures=`{effective_capture_count}`.",
        f"- Direct NLP f/g available cells: `{direct_cells}` / 4; forced nonconverged/off-optimal cells: `{forced_cells}` / 4.",
        f"- Frozen no-eps relative-error range: `{headline['frozen_noeps_relative_error_range']}`.",
        f"- Best eps-including relative-error max: `{headline['best_eps_relative_error_max']}`; best eps absolute-error max: `{headline['best_eps_absolute_error_max']}`.",
        f"- Max |abs_residual - eps_delta|: `{headline['max_abs_residual_minus_eps_delta_abs']}`.",
        "",
        "## Per-arm localization",
        "",
        "| role | H | status | no-eps rel err | best eps rel err | abs residual | eps delta | localized |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for a in per_arm:
        summary_lines.append("| `%s` | %s | `%s` | %s | %s | %s | %s | `%s` |" % (
            a["role"], a["horizon"], a["return_status"], a["frozen_noeps_relative_error"],
            a["best_eps_relative_error"], a["frozen_noeps_abs_residual"], a["eps_delta_vs_main"],
            a["localized_to_omitted_eps"],
        ))
    summary_lines += [
        "",
        "## Executor conclusion for lead review",
        "",
        "The direct NLP objective exactly matches the solver-reported objective in all four forced `Maximum_Iterations_Exceeded` cells, while the precommitted no-epsilon reconstruction misses the objective by the recorded epsilon/slack term. The eps-including aliases close the residual to numerical precision. This is a local objective-accessor/formula finding only: it does not retroactively pass v34u G2, does not touch validation64/sealed tests, and does not authorize Task-C until the active lead decides whether to amend and rerun the bounded gate.",
        "",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`; CSV: `{rel(comparison_csv)}`; completed: `{rel(RUN_DIR/'completed.json')}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34v_eps_objective_localization_postdiagnostic",
        "created_utc": now.isoformat(),
        "backup_required_before_more_unique_science": True,
        "reason": "new v34v postdiagnostic source/output/docs and v34u-derived lead handoff must be externally recoverable before additional solver-bearing diagnostics or Task-C",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "env_step_calls_after_construction": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "Wait for active-lead disposition on whether to amend objective formula to include epsilon/slack and rerun an objective-contract gate before any Task-C solver calls.",
    })

    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": now.isoformat(),
        "status": "analysis_requested",
        "trigger": "v34u A13c-3 hard objective defect localized by zero-solve v34v postdiagnostic",
        "experiment_id": NAME,
        "active_lead_expected": "claude-opus-5-5",
        "question": "Interpret v34u/v34v evidence and choose next action. The constructor/max_iter/direct-NLP gates passed, but the frozen no-epsilon formula failed in all four forced infeasible cells; eps-including aliases match direct NLP/solver to numerical precision. Decide whether to amend the objective-contract formula to include epsilon/slack and rerun a bounded gate or pursue a different lead-approved diagnostic. Do not open validation64/sealed test.",
        "evidence_paths": [rel(RUN_DIR/"summary.md"), rel(RUN_DIR/"raw.json"), rel(RUN_DIR/"eps_alias_localization.csv"), rel(RUN_DIR/"completed.json"), rel(COMPLETED), rel(RAW), rel(CELL_CSV), rel(ALIAS_CSV)],
        "headline": headline,
        "budget_actual": headline["budget_actual"],
        "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
    })

    block = f"""
<!-- {MARKER} -->
## v34v epsilon/slack objective-localization postdiagnostic

UTC: {now.isoformat()}. Ran a zero-solve postdiagnostic over v34u A13c-3 artifacts. Budget: new solver calls=0, plant/env.step=0, training/refit=0, validation64=0, sealed test=0. Diagnostic_pass={headline['diagnostic_pass']}; input v34u remains G2_pass={headline['input_v34u_G2_pass']} and hard_pass={headline['input_v34u_hard_pass']}. Constructor binding passed in all cells and effective max_iter=1 captures={effective_capture_count}; direct NLP f/g available cells={direct_cells}/4; forced nonconverged/off-optimal cells={forced_cells}/4. Frozen no-eps relative-error range={headline['frozen_noeps_relative_error_range']}; best eps-including relative-error max={headline['best_eps_relative_error_max']}; max |abs_residual - eps_delta|={headline['max_abs_residual_minus_eps_delta_abs']}. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'eps_alias_localization.csv')}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`. Next: active-lead disposition is required before amending formula or spending Task-C solver calls.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34v epsilon/slack objective-localization postdiagnostic\n\n"
        + f"UTC: {now.isoformat()}\n\n"
        + f"Headline: {json.dumps(clean(headline), sort_keys=True)}\n\n"
        + "Result: v34u remains failed (G2/hard_pass false) under the frozen no-epsilon formula. The zero-solve diagnostic localizes the residual to omitted epsilon/slack cost: eps-including aliases match direct NLP/solver to numerical precision in all four forced nonconverged cells.\n\n"
        + f"Artifacts: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'eps_alias_localization.csv')}`, `{rel(RUN_DIR/'completed.json')}`.\n\n"
        + f"Next: request active-lead review via `{rel(NEXT_REVIEW_REQUEST)}`. Do not spend Task-C solver calls or open validation64/sealed test until lead disposition and external backup for `{rel(BACKUP_REQUEST)}`.\n",
        encoding="utf-8",
    )

    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([now.isoformat(), NAME, raw_out["classification"], "zero_solve_v34u_eps_alias_localization", "opened_development_postdiagnostic_no_validation_no_test", len(per_arm), 0, 0, 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])

    files = [Path(__file__).resolve(), RUN_DIR / "raw.json", RUN_DIR / "summary.md", comparison_csv, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, STATE, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
    completed_out = {
        "status": "complete",
        "hard_pass": headline["diagnostic_pass"],
        "created_utc": now.isoformat(),
        "classification": raw_out["classification"],
        "headline": headline,
        "summary": rel(RUN_DIR / "summary.md"),
        "raw": rel(RUN_DIR / "raw.json"),
        "eps_alias_localization_csv": rel(comparison_csv),
        "input_v34u_completed": rel(COMPLETED),
        "backup_request": rel(BACKUP_REQUEST),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "budget_actual": headline["budget_actual"],
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    }
    write_json(RUN_DIR / "completed.json", completed_out)
    print(json.dumps(clean({"completed": rel(RUN_DIR/"completed.json"), "summary": rel(RUN_DIR/"summary.md"), "headline": headline, "backup_request": rel(BACKUP_REQUEST)}), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
