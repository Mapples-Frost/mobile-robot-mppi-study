#!/usr/bin/env python3
"""Offline terminal-profile/fixed-short absorption postdiagnostic for risk-anchor acquisition v0.

No simulation, no training, no refit, no validation64 and no sealed-test access.

This script reads the completed development-only true-variable-H risk-anchor
acquisition run and quantifies whether labels are driven by terminal-profile
choice rather than deployable state-dependent horizon opportunity.  It also
records the concrete next intervention decision before any more rollout/training:
post-run backup first, then freeze a targeted H10/H15 risk-aware objective/value
repair diagnostic rather than scaling the failed H10/H15/H25 selector.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_terminal_profile_effect_postdiagnostic_v0"
STAMP = "20260929T0835Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z"
RAW_PATH = RUN_DIR / "raw.json"
DONE_PATH = RUN_DIR / "completed.json"
SUMMARY_PATH = RUN_DIR / "summary.md"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-terminal-profile-effect-postdiagnostic-v0-{STAMP}"
HORIZONS = [10, 15, 25]
MATERIAL_GAIN = 3.0
MIN_DECISION_SAVING = 0.15


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * q
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "median": pct(0.5), "mean": float(math.fsum(xs) / len(xs)), "p95": pct(0.95), "max": xs[-1], "sum": float(math.fsum(xs))}


def is_risk(role: Any) -> bool:
    text = str(role or "")
    return "risk_anchor" in text and "control" not in text


def ensure_inputs() -> Dict[str, Any]:
    for path in (RAW_PATH, DONE_PATH, SUMMARY_PATH):
        if not path.exists():
            raise ContractError("missing required input: " + rel(path))
    done = read_json(DONE_PATH)
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("risk-anchor acquisition completed marker did not pass")
    for key in ("validation64_bank_opened", "sealed_test_accessed"):
        if done.get(key) is not False:
            raise ContractError(f"{key} must be false in acquisition marker")
    raw = read_json(RAW_PATH)
    if raw.get("validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("acquisition raw access flags are not closed")
    if (raw.get("analysis") or {}).get("gates", {}).get("train_or_refit_now") is not False:
        raise ContractError("unexpected upstream permission to train/refit")
    return raw


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    raw = ensure_inputs()
    analysis = raw["analysis"]
    rows = analysis["state_profile_rows"]
    by_state: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    state_role: Dict[str, str] = {}
    state_case: Dict[str, int] = {}
    for r in rows:
        sid = str(r["state_id"])
        profile = str(r["terminal_profile"])
        by_state.setdefault(sid, {})[profile] = r
        state_role[sid] = str(r.get("target_role"))
        state_case[sid] = int(r.get("case", -1))

    state_effects: List[Dict[str, Any]] = []
    per_h_abs_phys_delta: Dict[str, List[float]] = {str(h): [] for h in HORIZONS}
    per_h_abs_dec_delta: Dict[str, List[float]] = {str(h): [] for h in HORIZONS}
    material_nonh15_terminal_effect_states: List[str] = []
    for sid, profs in sorted(by_state.items()):
        matched = profs.get("matched_terminal")
        shared = profs.get("shared_h15_terminal")
        if not matched or not shared:
            continue
        effects: Dict[str, Any] = {}
        max_nonh15_abs = 0.0
        for h in HORIZONS:
            m = matched["median_by_h"][str(h)]
            s = shared["median_by_h"][str(h)]
            phys_delta = sf(s.get("physical")) - sf(m.get("physical"))
            dec_delta = sf(s.get("decision_sum_s")) - sf(m.get("decision_sum_s"))
            effects[str(h)] = {
                "shared_minus_matched_physical": phys_delta,
                "shared_minus_matched_decision_s": dec_delta,
                "matched_physical": sf(m.get("physical")),
                "shared_h15_physical": sf(s.get("physical")),
            }
            per_h_abs_phys_delta[str(h)].append(abs(phys_delta))
            per_h_abs_dec_delta[str(h)].append(abs(dec_delta))
            if h != 15:
                max_nonh15_abs = max(max_nonh15_abs, abs(phys_delta))
        if max_nonh15_abs >= MATERIAL_GAIN:
            material_nonh15_terminal_effect_states.append(sid)
        state_effects.append({
            "state_id": sid,
            "case": state_case[sid],
            "target_role": state_role[sid],
            "is_risk_anchor_primary": is_risk(state_role[sid]),
            "matched_label": matched.get("oracle_label"),
            "shared_h15_label": shared.get("oracle_label"),
            "label_flipped": matched.get("oracle_label") != shared.get("oracle_label"),
            "per_horizon_terminal_effects": effects,
            "max_abs_nonH15_physical_terminal_effect": max_nonh15_abs,
        })

    flip_states = [s for s in state_effects if s["label_flipped"]]
    risk_states = [s for s in state_effects if s["is_risk_anchor_primary"]]
    risk_flips = [s for s in risk_states if s["label_flipped"]]
    # H15 is intentionally identical under matched and shared-H15 terminals; if
    # non-H15 deltas dominate while H15 delta is zero/nearly zero, the label map
    # is not a robust horizon-control effect under current terminal modelling.
    terminal_effect_summaries = {
        str(h): {
            "abs_physical_delta": summary(per_h_abs_phys_delta[str(h)]),
            "abs_decision_delta_s": summary(per_h_abs_dec_delta[str(h)]),
        }
        for h in HORIZONS
    }
    fixed_abs = analysis["fixed_short_absorption_risk_groups"]
    h15_absorbs = bool(fixed_abs.get("fixed_H15_absorbs_vs_H25"))
    h10_absorbs = bool(fixed_abs.get("fixed_H10_absorbs_vs_H25"))
    terminal_dependence_strong = bool(
        len(flip_states) / float(len(state_effects) or 1) > 0.5
        and len(risk_flips) / float(len(risk_states) or 1) > 0.5
        and terminal_effect_summaries["15"]["abs_physical_delta"]["max"] <= 1e-9
        and len(material_nonh15_terminal_effect_states) >= 2
    )
    fixed_short_absorption_strong = bool(h15_absorbs or h10_absorbs)
    evidence_table = {
        "scenario_opportunity": {
            "verified": "Fresh source-independent risk states do show a measured H25-vs-short compute tradeoff, but stable H15/H25 labels disappear across terminal profiles and fixed H15 absorbs the risk-group tradeoff vs H25.",
            "competing_hypotheses": [
                "There is useful H10/H15 risk-aware opportunity, but current terminal models make labels unstable.",
                "The stress-v1 states mostly require a profile-specific fixed H15 rather than state-adaptive H10/H15/H25.",
                "Scenario opportunity exists only in narrow case-specific transients and will not generalize without redesigned state sampling."
            ],
            "missing_evidence": "A terminal-consistent H10/H15 risk classifier or value repair has not been evaluated on fresh development states; no validation/test access.",
            "discriminating_experiment": "After verified backup, freeze an offline H10-vs-H15 safety/regret classifier/value-calibration diagnostic using acquisition+oracle labels under one predeclared terminal convention; only if it beats fixed H15 in leave-one-case/state-out CV should any rollout follow."
        },
        "reward_terminal_objective": {
            "verified": "Terminal-profile choice flips labels in %d/%d states (%d/%d risk states). H15 physical cost is invariant by construction, while non-H15 terminal effects are material in %d states." % (len(flip_states), len(state_effects), len(risk_flips), len(risk_states), len(material_nonh15_terminal_effect_states)),
            "competing_hypotheses": [
                "Per-H terminal value estimates are mismatched/calibrated on incompatible objectives, causing spurious H10/H25 preferences.",
                "Shared-H15 terminal is too biased against H10/H25, while matched-terminal profiles are too permissive for H10.",
                "The physical-cost label rule tolerances are too coarse for small real control differences."
            ],
            "missing_evidence": "Direct terminal-value residuals against realised continuation and a small bounded value-refit/calibration ablation.",
            "discriminating_experiment": "No-sim terminal residual audit from logged continuations, then a bounded value-calibration/refit if residuals are horizon/profile biased."
        },
        "training_selection": {
            "verified": "Current selector/refit remains blocked: pass_to_offline_deployable_selector_cv=false, stable risk H15/H25 cases=[], and train_or_refit_now=false.",
            "competing_hypotheses": [
                "A lower-ambition H10/H15 risk-aware classifier may be learnable even though H25 labels are not stable.",
                "Current feature set lacks observability for terminal-induced safety/regret, so retraining without terminal repair would overfit.",
                "Gradient retraining is premature until terminal targets are made consistent."
            ],
            "missing_evidence": "Cross-validated H10/H15 risk/regret predictability with case-held-out splits and a fixed-H15 comparator.",
            "discriminating_experiment": "Freeze and run an offline H10/H15 risk-aware selector CV only after backup; if it fails, pivot to terminal-value calibration or scenario redesign before any rollout."
        },
        "comparisons_timing": {
            "verified": "Measured risk oracle vs H25 decision saving is %.3f and solver saving %.3f, but fixed H15 vs H25 also saves %.3f decision time with lower physical cost than H25." % (
                sf(analysis["comparisons"]["risk_anchor_primary"]["oracle_vs_fixed_H25"].get("decision_relative_saving")),
                sf(analysis["comparisons"]["risk_anchor_primary"]["oracle_vs_fixed_H25"].get("solver_relative_saving")),
                sf(fixed_abs.get("fixed_H15_decision_saving_vs_H25")),
            ),
            "competing_hypotheses": [
                "Fixed H15 is the appropriate strong baseline for these stress states.",
                "An H10/H15 adaptive rule could still improve compute over fixed H15 without H10 failures, but the present label source is unstable.",
                "Runtime noise could affect small margins, but the H-size timing ordering is large and blocked."
            ],
            "missing_evidence": "Deployable selector timing with overhead and paired comparison against fixed H15 on fresh development/validation states.",
            "discriminating_experiment": "Any future selector rollout must be paired/block-randomized against fixed true H10/H15/H25 and report whole-decision plus solver timing."
        },
    }
    proceed_to_selector_or_value_refit_now = False
    if terminal_dependence_strong:
        next_action = "await external backup, then freeze a bounded terminal-value residual/calibration audit plus H10/H15 risk-aware offline CV; do not run closed-loop selector or gradient/value refit yet"
    elif fixed_short_absorption_strong:
        next_action = "await backup, then test whether H10/H15 adaptive risk classifier can beat fixed H15 offline; otherwise make fixed H15 the stress-v1 baseline conclusion"
    else:
        next_action = "await backup, then freeze deployable selector CV before any closed-loop rollout"

    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_TERMINAL_PROFILE_EFFECT_POSTDIAGNOSTIC_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw_out = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_terminal_profile_effect_postdiagnostic_not_validation_not_final_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_artifacts": {"risk_anchor_raw": rel(RAW_PATH), "risk_anchor_completed": rel(DONE_PATH), "risk_anchor_summary": rel(SUMMARY_PATH)},
        "state_effects": state_effects,
        "terminal_effect_summaries": terminal_effect_summaries,
        "label_flip_counts": {
            "states": len(state_effects),
            "flipped_states": len(flip_states),
            "risk_states": len(risk_states),
            "risk_flipped_states": len(risk_flips),
            "material_nonH15_terminal_effect_states": len(material_nonh15_terminal_effect_states),
        },
        "terminal_dependence_strong": terminal_dependence_strong,
        "fixed_short_absorption_strong": fixed_short_absorption_strong,
        "upstream_gates": analysis["gates"],
        "upstream_fixed_short_absorption": fixed_abs,
        "evidence_table": evidence_table,
        "proceed_to_selector_or_value_refit_now": proceed_to_selector_or_value_refit_now,
        "next_action": next_action,
        "backup_request_after_postdiagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw_out)

    lines = [
        "# Vehicle true-variable-H terminal-profile effect postdiagnostic v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic using the completed risk-anchor acquisition only; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Label flips across terminal profiles: `{len(flip_states)}/{len(state_effects)}` states; risk-only flips: `{len(risk_flips)}/{len(risk_states)}`.",
        f"- Material non-H15 terminal physical effects (|shared-H15 minus matched| >= {MATERIAL_GAIN}): `{len(material_nonh15_terminal_effect_states)}` states.",
        f"- H15 terminal-effect max physical delta: `{terminal_effect_summaries['15']['abs_physical_delta']['max']}` (expected ~0 because H15 matched == shared-H15).",
        f"- Fixed-short absorption remains active: `{fixed_abs}`.",
        f"- Terminal-dependence strong: `{terminal_dependence_strong}`; proceed to selector/value refit now: `{proceed_to_selector_or_value_refit_now}`.",
        "",
        "## State-level terminal effects",
        "",
        "| state | case | risk? | matched label | shared-H15 label | flip | max |Δphys| non-H15 | H10 Δphys | H15 Δphys | H25 Δphys |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in state_effects:
        eff = s["per_horizon_terminal_effects"]
        lines.append("| `%s` | %d | `%s` | `%s` | `%s` | `%s` | %.6g | %.6g | %.6g | %.6g |" % (
            s["state_id"], int(s["case"]), str(s["is_risk_anchor_primary"]), str(s["matched_label"]), str(s["shared_h15_label"]), str(s["label_flipped"]),
            float(s["max_abs_nonH15_physical_terminal_effect"]), float(eff["10"]["shared_minus_matched_physical"]), float(eff["15"]["shared_minus_matched_physical"]), float(eff["25"]["shared_minus_matched_physical"]),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "The acquisition confirms measured compute savings from shorter true MPC horizons, but it does not justify scaling the H10/H15/H25 supervised selector.  Conservative labels are not stable under terminal-profile choice, and fixed true H15 already absorbs the H25 control/compute tradeoff on the risk-anchor subset.  This separates lack of physical-cost improvement from the control-vs-compute tradeoff: there is a tradeoff, but the deployable adaptive rule is not yet established beyond a strong fixed H15 baseline.",
        "",
        "## Next concrete intervention",
        "",
        next_action + ".",
        "",
        "No sealed test or validation64 bank was accessed. This is IMPROVED development evidence only, not ORIGINAL reproduction.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text((OUT_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup risk-anchor terminal-profile postdiagnostic before any selector CV, terminal-value audit/refit, simulation or training",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "backup_required_before_offline_selector_cv": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(req)],
    })
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H terminal-profile effect postdiagnostic v0

UTC: {created.isoformat()}. No simulations/training/refit; validation64 and sealed test stayed closed. Parsed `{rel(RAW_PATH)}` after the source-independent risk-anchor acquisition. Label flips across terminal profiles were {len(flip_states)}/{len(state_effects)} states ({len(risk_flips)}/{len(risk_states)} risk states), material non-H15 terminal physical effects occurred in {len(material_nonh15_terminal_effect_states)} states, and fixed H15 absorbed the risk-group H25 tradeoff (`fixed_H15_absorbs_vs_H25={h15_absorbs}`). Selector/refit remains blocked. Next action after backup: {next_action}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`.
"""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE_PATH, req, RAW_PATH, DONE_PATH]
    done = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw_out["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {
            "terminal_dependence_strong": terminal_dependence_strong,
            "fixed_short_absorption_strong": fixed_short_absorption_strong,
            "label_flipped_states": len(flip_states),
            "state_count": len(state_effects),
            "risk_label_flipped_states": len(risk_flips),
            "risk_state_count": len(risk_states),
            "proceed_to_selector_or_value_refit_now": proceed_to_selector_or_value_refit_now,
        },
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT_DIR / "completed.json", done)
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "headline": done["headline"],
        "backup_request": rel(req),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
