#!/usr/bin/env python3
"""No-simulation terminal/objective alignment diagnostic for stress-v1b full ablation.

This bounded development diagnostic reads the completed v0c terminal/reward-source
ablation and the immediately preceding full postdiagnostic.  It performs no
rollouts, no training/refit, opens no validation64 bank and never accesses the
sealed test.  The purpose is to decide whether the next intervention should be a
terminal/objective refit or a fresh terminal-stable label-density probe before
any selector training.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0025Z"
FULL_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair"
FULL_RAW = FULL_DIR / "raw.json"
FULL_DONE = FULL_DIR / "completed.json"
POST_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_full_postdiagnostic_v0_20260929T0018Z"
POST_RAW = POST_DIR / "raw.json"
POST_DONE = POST_DIR / "completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_terminal_objective_alignment_postdiagnostic.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-stress-v1b-terminal-objective-alignment-postdiagnostic-v0-{STAMP}"
MATERIAL_GAIN = 3.0


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, Path):
        return rel(v)
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


def append_once(path: Path, block: str, marker: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def fnum(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def gain_phys(r: Mapping[str, Any]) -> float:
    for k in ("gain_vs_per_h_H15_physical", "gain_vs_H15_physical", "physical_gain", "gain_physical"):
        if k in r:
            return fnum(r[k])
    return float("nan")


def gain_total(r: Mapping[str, Any]) -> float:
    for k in ("gain_vs_per_h_H15_total", "gain_vs_H15_total", "total_gain", "gain_total"):
        if k in r:
            return fnum(r[k])
    return float("nan")


def material(r: Mapping[str, Any]) -> bool:
    if "material_any" in r:
        return bool(r.get("material_any"))
    if "material_positive_vs_per_h_H15" in r:
        return bool(r.get("material_positive_vs_per_h_H15"))
    gp, gt = gain_phys(r), gain_total(r)
    return bool((math.isfinite(gp) and gp >= MATERIAL_GAIN) or (math.isfinite(gt) and gt >= MATERIAL_GAIN))


def success(r: Mapping[str, Any]) -> bool:
    return bool(r.get("success", True))


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return {"count": 0, "mean": None, "median": None, "max": None}
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"count": n, "mean": math.fsum(vals) / n, "median": med, "max": vals[-1]}


def objective_margin(sorted_rows: List[Mapping[str, Any]], sign: int) -> Optional[float]:
    vals = [sign * fnum(r.get("objective")) for r in sorted_rows if math.isfinite(fnum(r.get("objective")))]
    if len(vals) < 2:
        return None
    vals = sorted(vals)
    return vals[1] - vals[0]


def choose_by_objective(rows: List[Mapping[str, Any]], direction: str) -> Tuple[Optional[Mapping[str, Any]], Optional[float], bool]:
    sign = 1 if direction == "min" else -1
    obj_rows = [r for r in rows if math.isfinite(fnum(r.get("objective")))]
    if not obj_rows:
        return None, None, False
    chosen = min(obj_rows, key=lambda r: (sign * fnum(r.get("objective")), int(r.get("horizon", 999))))
    margin = objective_margin(obj_rows, sign)
    if margin is None:
        ambiguous = False
    else:
        scale = max(1.0, abs(fnum(chosen.get("objective"))))
        ambiguous = bool(margin <= max(1e-6, 1e-3 * scale))
    return chosen, margin, ambiguous


def horizon(r: Optional[Mapping[str, Any]]) -> Optional[int]:
    if r is None:
        return None
    try:
        return int(r.get("horizon"))
    except Exception:
        return None


def main() -> int:
    for p in (FULL_RAW, FULL_DONE, POST_RAW, POST_DONE):
        if not p.exists():
            raise RuntimeError("missing input artifact: %s" % rel(p))
    full_done = read_json(FULL_DONE)
    post_done = read_json(POST_DONE)
    full = read_json(FULL_RAW)
    post = read_json(POST_RAW)
    for label, obj in (("full_done", full_done), ("post_done", post_done), ("full_raw", full), ("post_raw", post)):
        if obj.get("sealed_test_accessed") is not False or obj.get("sealed_test_bank_opened") not in (False, None):
            raise RuntimeError(label + " indicates sealed-test access")
        if obj.get("historical_validation64_bank_opened") not in (False, None) or obj.get("validation64_bank_opened") not in (False, None):
            raise RuntimeError(label + " indicates validation64 access")
    if full_done.get("passed") is not True or post_done.get("passed") is not True:
        raise RuntimeError("input completion marker did not pass")
    analysis = full.get("analysis") or {}
    rows = list(analysis.get("comparison_rows") or [])
    if not rows:
        raise RuntimeError("full ablation raw lacks comparison_rows")

    state_ids = sorted({str(r.get("state_id")) for r in rows})
    modes = sorted({str(r.get("terminal_mode")) for r in rows})
    state_mode_rows: List[Dict[str, Any]] = []
    mode_aggs: Dict[str, Dict[str, Any]] = {}
    for sid in state_ids:
        srows = [r for r in rows if str(r.get("state_id")) == sid]
        role = str(srows[0].get("state_role", srows[0].get("role", "unknown")))
        case = int(srows[0].get("case", -1))
        branch_step = int(srows[0].get("branch_step", -1))
        for mode in modes:
            mrows = [r for r in srows if str(r.get("terminal_mode")) == mode]
            if not mrows:
                continue
            best_phys = max(mrows, key=lambda r: (gain_phys(r) if math.isfinite(gain_phys(r)) else -1e300, -int(r.get("horizon", 999))))
            best_total = max(mrows, key=lambda r: (gain_total(r) if math.isfinite(gain_total(r)) else -1e300, -int(r.get("horizon", 999))))
            min_obj, min_margin, min_amb = choose_by_objective(mrows, "min")
            max_obj, max_margin, max_amb = choose_by_objective(mrows, "max")
            mat_h = sorted(int(r.get("horizon")) for r in mrows if material(r))
            min_loss_phys = None if min_obj is None else max(0.0, gain_phys(best_phys) - gain_phys(min_obj))
            min_loss_total = None if min_obj is None else max(0.0, gain_total(best_total) - gain_total(min_obj))
            strict_min_failure = bool(mat_h and min_obj is not None and not material(min_obj) and not min_amb)
            row = {
                "state_id": sid,
                "case": case,
                "branch_step": branch_step,
                "role": role,
                "terminal_mode": mode,
                "material_horizons": mat_h,
                "best_physical_horizon": horizon(best_phys),
                "best_physical_gain": gain_phys(best_phys),
                "best_total_horizon": horizon(best_total),
                "best_total_gain": gain_total(best_total),
                "min_objective_horizon": horizon(min_obj),
                "min_objective_value": None if min_obj is None else fnum(min_obj.get("objective")),
                "min_objective_margin": min_margin,
                "min_objective_ambiguous_tie": min_amb,
                "min_objective_material": None if min_obj is None else material(min_obj),
                "min_objective_success": None if min_obj is None else success(min_obj),
                "min_objective_physical_gain": None if min_obj is None else gain_phys(min_obj),
                "min_objective_total_gain": None if min_obj is None else gain_total(min_obj),
                "min_objective_physical_loss_vs_best": min_loss_phys,
                "min_objective_total_loss_vs_best": min_loss_total,
                "strict_min_objective_alignment_failure": strict_min_failure,
                "max_objective_horizon": horizon(max_obj),
                "max_objective_material": None if max_obj is None else material(max_obj),
                "max_objective_ambiguous_tie": max_amb,
                "non_success_horizons": sorted(int(r.get("horizon")) for r in mrows if not success(r)),
            }
            state_mode_rows.append(row)
    for mode in modes:
        rs = [r for r in state_mode_rows if r["terminal_mode"] == mode]
        with_mat = [r for r in rs if r["material_horizons"]]
        mode_aggs[mode] = {
            "state_rows": len(rs),
            "states_with_material": len(with_mat),
            "strict_min_objective_alignment_failures": int(sum(1 for r in rs if r["strict_min_objective_alignment_failure"])),
            "min_objective_ambiguous_ties": int(sum(1 for r in rs if r["min_objective_ambiguous_tie"])),
            "min_objective_selects_material_when_material_exists": int(sum(1 for r in with_mat if r["min_objective_material"] is True)),
            "min_objective_selects_non_success": int(sum(1 for r in rs if r["min_objective_success"] is False)),
            "min_objective_physical_loss_vs_best": finite_summary([r["min_objective_physical_loss_vs_best"] for r in rs if r["min_objective_physical_loss_vs_best"] is not None]),
            "min_objective_total_loss_vs_best": finite_summary([r["min_objective_total_loss_vs_best"] for r in rs if r["min_objective_total_loss_vs_best"] is not None]),
        }

    spread_rows = []
    for sid in state_ids:
        hs = sorted({int(r.get("horizon")) for r in rows if str(r.get("state_id")) == sid})
        for h in hs:
            xrows = [r for r in rows if str(r.get("state_id")) == sid and int(r.get("horizon")) == h]
            if len(xrows) < 2:
                continue
            vals = [fnum(r.get("value_fn")) for r in xrows if math.isfinite(fnum(r.get("value_fn")))]
            objs = [fnum(r.get("objective")) for r in xrows if math.isfinite(fnum(r.get("objective")))]
            gps = [gain_phys(r) for r in xrows if math.isfinite(gain_phys(r))]
            mats = {str(r.get("terminal_mode")): material(r) for r in xrows}
            spread_rows.append({
                "state_id": sid,
                "horizon": h,
                "terminal_modes": sorted(str(r.get("terminal_mode")) for r in xrows),
                "value_fn_range": (max(vals) - min(vals)) if vals else None,
                "objective_range": (max(objs) - min(objs)) if objs else None,
                "physical_gain_range": (max(gps) - min(gps)) if gps else None,
                "material_by_mode": mats,
                "material_flip_across_modes": len(set(mats.values())) > 1,
                "non_success_modes": sorted(str(r.get("terminal_mode")) for r in xrows if not success(r)),
            })
    top_spreads = sorted(spread_rows, key=lambda r: (r["value_fn_range"] if r["value_fn_range"] is not None else -1), reverse=True)[:12]

    terminal_flips = len(analysis.get("terminal_material_label_flips") or [])
    synthetic_only = len(analysis.get("synthetic_only_material_rows") or [])
    robust_same_h = (((post.get("label_density") or {}).get("positive_with_same_h_per_h_and_zero_material")))
    controls_material = (((post.get("label_density") or {}).get("control_states_with_any_material_label")))
    strict_failures_total = sum(v["strict_min_objective_alignment_failures"] for v in mode_aggs.values())
    non_success_objective = sum(v["min_objective_selects_non_success"] for v in mode_aggs.values())
    if terminal_flips >= 10 or strict_failures_total > 0 or non_success_objective > 0:
        next_action = "freeze fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics before any selector training; include terminal-value/refit arm only if stable labels are dense enough"
    else:
        next_action = "freeze compact selector/refit smoke on artifact-resistant labels, then confirm on fresh development states"

    created = dt.datetime.now(dt.timezone.utc).isoformat()
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_OBJECTIVE_ALIGNMENT_POSTDIAGNOSTIC_V0_{STAMP}.json"
    four_axis = {
        "SCENARIOS": {"verified": f"Existing full ablation remains mined: robust same-H zero-terminal positives={robust_same_h}, controls with material={controls_material}; this diagnostic adds no fresh states.", "hypothesis": "Local opportunity is real in mined high-transient states but must be tested on fresh non-mined states under stable terminal handling.", "missing": "Fresh terminal-stable label density and state-feature coverage.", "next_discriminator": "Run the frozen fresh terminal-stable label-density probe after backup rather than unchanged validation."},
        "REWARD": {"verified": f"Terminal material-label flips={terminal_flips}, synthetic-only rows={synthetic_only}; objective alignment failures={strict_failures_total}, objective-selected non-success rows={non_success_objective}.", "hypothesis": "Per-H terminal/source objectives are not a safe training target; labels should be realized continuation physical/total/timing with terminal source controlled.", "missing": "Whether terminal-value refit improves objective ranking on fresh states.", "next_discriminator": "Record min-objective alignment under zero/common/per-H terminal in the next fresh probe and only then refit terminal values if labels are dense."},
        "TRAINING": {"verified": "Zero training/refit/gradient steps here. Existing historical selectors should not be trained/reselected on per-H terminal objective labels.", "hypothesis": "Near-constant policies may be an appropriate response to sparse robust labels or a consequence of objective/terminal misalignment.", "missing": "Stable label set large enough for a compact selector or terminal-value refit.", "next_discriminator": "After fresh stable labels, run a small IMPROVED selector/value-refit smoke with fixed criteria if density gate passes."},
        "COMPARISONS": {"verified": "No validation64/sealed-test access; no superiority claim. Synthetic total cost remains separate from measured runtime.", "hypothesis": "Fixed-H baselines may absorb gains after artifacts are removed.", "missing": "Fair fixed-H and timing comparisons under a revised protocol.", "next_discriminator": "Retune strong fixed-H only after method/protocol freeze and fresh confirmation."},
    }
    result = {"created_utc": created, "method": "vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0", "classification": "development_no_simulation_objective_alignment_not_validation_not_final_test", "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "inputs": {rel(FULL_RAW): sha256(FULL_RAW), rel(FULL_DONE): sha256(FULL_DONE), rel(POST_RAW): sha256(POST_RAW), rel(POST_DONE): sha256(POST_DONE)}, "terminal_flips": terminal_flips, "synthetic_only_material_rows": synthetic_only, "robust_same_h_zero_positive_states": robust_same_h, "control_states_with_material": controls_material, "mode_alignment_summary": mode_aggs, "state_mode_alignment_rows": state_mode_rows, "top_terminal_value_spreads": top_spreads, "four_axis_evidence": four_axis, "decision": {"train_or_refit_now": False, "reason": "The ablation remains mined/development-only and terminal/objective alignment is not yet safe enough for selector training; first collect fresh stable labels or demonstrate objective-ranking repair.", "next_action": next_action, "backup_required_before_more_simulation_or_training": True}, "backup_request": rel(backup_request)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "raw.json", result)
    lines = ["# Vehicle stress-v1b terminal/objective alignment postdiagnostic v0", "", f"UTC: `{created}`. No simulation/training; validation64 and sealed test stayed closed.", "", "## Headline", "", f"- Terminal flips: `{terminal_flips}`; synthetic-only material rows: `{synthetic_only}`.", f"- Robust same-H per-H & zero-terminal positives from prior postdiagnostic: `{robust_same_h}`; controls with material: `{controls_material}`.", f"- Strict min-objective alignment failures across terminal modes: `{strict_failures_total}`; min-objective selected non-success rows: `{non_success_objective}`.", f"- Train/refit now: `False`; next: {next_action}.", "", "## Mode-level objective alignment (assuming lower reported objective is preferred)", "", "| terminal mode | states | states with material | min-obj selects material | strict failures | ambiguous ties | min-obj non-success | mean phys loss | max phys loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in modes:
        m = mode_aggs[mode]
        phys = m["min_objective_physical_loss_vs_best"]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %d | %s | %s |" % (mode, m["state_rows"], m["states_with_material"], m["min_objective_selects_material_when_material_exists"], m["strict_min_objective_alignment_failures"], m["min_objective_ambiguous_ties"], m["min_objective_selects_non_success"], "" if phys["mean"] is None else ("%.6g" % phys["mean"]), "" if phys["max"] is None else ("%.6g" % phys["max"])))
    lines += ["", "## Material states and objective-selected horizons", "", "| state | role | mode | material H | best phys H/gain | min-obj H/gain | ambiguous | strict failure | non-success H |", "|---|---|---|---|---|---|---|---|---|"]
    for r in state_mode_rows:
        if r["material_horizons"] or r["non_success_horizons"] or r["strict_min_objective_alignment_failure"]:
            lines.append("| `%s` | `%s` | `%s` | `%s` | H%s/%.6g | H%s/%s | `%s` | `%s` | `%s` |" % (r["state_id"], r["role"], r["terminal_mode"], r["material_horizons"], r["best_physical_horizon"], r["best_physical_gain"], r["min_objective_horizon"], "" if r["min_objective_physical_gain"] is None else ("%.6g" % r["min_objective_physical_gain"]), r["min_objective_ambiguous_tie"], r["strict_min_objective_alignment_failure"], r["non_success_horizons"]))
    lines += ["", "## Largest terminal value spreads", "", "| state | H | value range | objective range | physical-gain range | material by mode | non-success modes |", "|---|---:|---:|---:|---:|---|---|"]
    for r in top_spreads[:8]:
        lines.append("| `%s` | %d | %s | %s | %s | `%s` | `%s` |" % (r["state_id"], r["horizon"], "" if r["value_fn_range"] is None else ("%.6g" % r["value_fn_range"]), "" if r["objective_range"] is None else ("%.6g" % r["objective_range"]), "" if r["physical_gain_range"] is None else ("%.6g" % r["physical_gain_range"]), r["material_by_mode"], r["non_success_modes"]))
    lines += ["", "## Four-axis evidence", ""]
    for axis, body in four_axis.items():
        lines += [f"### {axis}", ""]
        for key in ("verified", "hypothesis", "missing", "next_discriminator"):
            lines.append(f"- **{key}:** {body[key]}")
        lines.append("")
    lines += ["## Decision", "", result["decision"]["reason"], "", f"Backup request: `{rel(backup_request)}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(backup_request, {"requested_utc": created, "reason": "backup terminal/objective alignment postdiagnostic before fresh terminal-stable simulations or refit", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(backup_request)]})
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text("# Continue state after terminal/objective alignment postdiagnostic (%s)\n\nUTC: %s. No validation64/test/training/simulation. Terminal flips=%d, synthetic-only rows=%d, robust same-H zero positives=%s, controls material=%s, strict objective-alignment failures=%d, objective-selected non-success rows=%d. Decision: do not train/refit now; next after backup is to freeze/run a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics, then consider terminal-value/refit only if stable labels are dense. Artifacts: `%s`, `%s`.\n" % (STAMP, created, terminal_flips, synthetic_only, robust_same_h, controls_material, strict_failures_total, non_success_objective, rel(OUT_DIR / "summary.md"), rel(OUT_DIR / "raw.json")), encoding="utf-8")
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1b terminal/objective alignment postdiagnostic

UTC: {created}. No-simulation postdiagnostic of the completed full terminal/reward-source ablation. No validation64/sealed-test access and no training/refit. Terminal flips={terminal_flips}, synthetic-only rows={synthetic_only}, robust same-H zero positives={robust_same_h}, control material states={controls_material}, strict min-objective alignment failures={strict_failures_total}, min-objective selected non-success rows={non_success_objective}. Decision: do not train/refit now; next after backup is a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, state `{rel(STATE_PATH)}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_once(ROOT / doc, block, MARKER)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, backup_request, Path(__file__).resolve(), FULL_RAW, FULL_DONE, POST_RAW, POST_DONE]
    write_json(OUT_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "headline": {"terminal_flips": terminal_flips, "synthetic_only_material_rows": synthetic_only, "strict_min_objective_alignment_failures": strict_failures_total, "min_objective_selected_non_success_rows": non_success_objective, "train_or_refit_now": False, "next_action": next_action}, "backup_request": rel(backup_request), "hashes": {rel(p): sha256(p) for p in files if p.exists()}})
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "state": rel(STATE_PATH), "terminal_flips": terminal_flips, "synthetic_only_material_rows": synthetic_only, "strict_min_objective_alignment_failures": strict_failures_total, "min_objective_selected_non_success_rows": non_success_objective, "train_or_refit_now": False, "next_action": next_action, "backup_request": rel(backup_request), "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_training_episodes": 0, "new_gradient_steps": 0}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
