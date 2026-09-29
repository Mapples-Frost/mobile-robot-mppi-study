#!/usr/bin/env python3
"""Schema-repair terminal/objective alignment diagnostic for stress-v1b.

The immediately preceding v0 no-simulation diagnostic looked for an ``objective``
field, but the completed terminal/reward ablation stores MPC objective values as
``branch_objective_opt_f_num`` and value estimates as ``branch_mpc_value_fn``.
Consequently v0's zero strict objective-alignment failures were non-informative
(all objective-selected horizons were None).  This v0b script performs the same
bounded analysis with explicit schema aliases and preserves v0 as superseded
negative/bug evidence.

No rollouts, no training/refit, no validation64 bank access and no sealed-test
access are performed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0035Z"
FULL_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair"
FULL_RAW = FULL_DIR / "raw.json"
FULL_DONE = FULL_DIR / "completed.json"
POST_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_full_postdiagnostic_v0_20260929T0018Z"
POST_RAW = POST_DIR / "raw.json"
POST_DONE = POST_DIR / "completed.json"
V0_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0_20260929T0025Z"
V0_RAW = V0_DIR / "raw.json"
V0_DONE = V0_DIR / "completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_terminal_objective_alignment_v0b_schema_repair.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-stress-v1b-terminal-objective-alignment-v0b-schema-repair-{STAMP}"
MATERIAL_GAIN = 3.0


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    return value


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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def first_finite(row: Mapping[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        if key in row:
            val = fnum(row.get(key))
            if math.isfinite(val):
                return val
    return float("nan")


def objective(row: Mapping[str, Any]) -> float:
    return first_finite(row, ("objective", "branch_objective_opt_f_num", "opt_f_num", "mpc_objective"))


def value_fn(row: Mapping[str, Any]) -> float:
    return first_finite(row, ("value_fn", "branch_mpc_value_fn", "mpc_value_fn", "terminal_value"))


def gain_phys(row: Mapping[str, Any]) -> float:
    return first_finite(row, ("gain_vs_per_h_H15_physical", "gain_vs_H15_physical", "physical_gain", "gain_physical"))


def gain_total(row: Mapping[str, Any]) -> float:
    return first_finite(row, ("gain_vs_per_h_H15_total", "gain_vs_H15_total", "total_gain", "gain_total"))


def material(row: Mapping[str, Any]) -> bool:
    if "material_any" in row:
        return bool(row.get("material_any"))
    gp, gt = gain_phys(row), gain_total(row)
    return bool((math.isfinite(gp) and gp >= MATERIAL_GAIN) or (math.isfinite(gt) and gt >= MATERIAL_GAIN))


def success(row: Mapping[str, Any]) -> bool:
    return bool(row.get("success", True))


def horizon(row: Optional[Mapping[str, Any]]) -> Optional[int]:
    if row is None:
        return None
    try:
        return int(row.get("horizon"))
    except Exception:
        return None


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return {"count": 0, "mean": None, "median": None, "max": None}
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"count": n, "mean": math.fsum(vals) / n, "median": med, "max": vals[-1]}


def choose_by_objective(rows: Sequence[Mapping[str, Any]], direction: str = "min") -> Tuple[Optional[Mapping[str, Any]], Optional[float], bool, int]:
    sign = 1 if direction == "min" else -1
    obj_rows = [r for r in rows if math.isfinite(objective(r))]
    if not obj_rows:
        return None, None, False, 0
    chosen = min(obj_rows, key=lambda r: (sign * objective(r), int(r.get("horizon", 999))))
    vals = sorted(sign * objective(r) for r in obj_rows)
    if len(vals) < 2:
        return chosen, None, False, len(obj_rows)
    margin = vals[1] - vals[0]
    scale = max(1.0, abs(objective(chosen)))
    ambiguous = bool(margin <= max(1e-6, 1e-3 * scale))
    return chosen, margin, ambiguous, len(obj_rows)


def main() -> int:
    for p in (FULL_RAW, FULL_DONE, POST_RAW, POST_DONE, V0_RAW, V0_DONE):
        if not p.exists():
            raise RuntimeError("missing input artifact: %s" % rel(p))
    full = read_json(FULL_RAW)
    full_done = read_json(FULL_DONE)
    post = read_json(POST_RAW)
    post_done = read_json(POST_DONE)
    v0 = read_json(V0_RAW)
    v0_done = read_json(V0_DONE)
    for label, obj in (("full", full), ("full_done", full_done), ("post", post), ("post_done", post_done), ("v0", v0), ("v0_done", v0_done)):
        if obj.get("sealed_test_accessed") not in (False, None) or obj.get("sealed_test_bank_opened") not in (False, None):
            raise RuntimeError(label + " indicates sealed-test access")
        if obj.get("historical_validation64_bank_opened") not in (False, None) or obj.get("validation64_bank_opened") not in (False, None):
            raise RuntimeError(label + " indicates validation64 access")
    if full_done.get("passed") is not True or post_done.get("passed") is not True or v0_done.get("passed") is not True:
        raise RuntimeError("an input completed marker did not pass")
    rows = list((full.get("analysis") or {}).get("comparison_rows") or [])
    if not rows:
        raise RuntimeError("full ablation raw lacks comparison_rows")

    objective_key_coverage = {
        "rows": len(rows),
        "objective_field": int(sum(1 for r in rows if math.isfinite(fnum(r.get("objective"))))),
        "branch_objective_opt_f_num": int(sum(1 for r in rows if math.isfinite(fnum(r.get("branch_objective_opt_f_num"))))),
        "value_fn_field": int(sum(1 for r in rows if math.isfinite(fnum(r.get("value_fn"))))),
        "branch_mpc_value_fn": int(sum(1 for r in rows if math.isfinite(fnum(r.get("branch_mpc_value_fn"))))),
    }
    v0_rows = list(v0.get("state_mode_alignment_rows") or [])
    v0_missing_objective_selected = int(sum(1 for r in v0_rows if r.get("min_objective_horizon") is None))

    state_ids = sorted({str(r.get("state_id")) for r in rows})
    modes = sorted({str(r.get("terminal_mode")) for r in rows})
    state_mode_rows: List[Dict[str, Any]] = []
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
            min_obj, min_margin, min_amb, n_obj = choose_by_objective(mrows, "min")
            max_obj, max_margin, max_amb, _ = choose_by_objective(mrows, "max")
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
                "objective_rows_available": n_obj,
                "material_horizons": mat_h,
                "best_physical_horizon": horizon(best_phys),
                "best_physical_gain": gain_phys(best_phys),
                "best_total_horizon": horizon(best_total),
                "best_total_gain": gain_total(best_total),
                "min_objective_horizon": horizon(min_obj),
                "min_objective_value": None if min_obj is None else objective(min_obj),
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

    mode_aggs: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        rs = [r for r in state_mode_rows if r["terminal_mode"] == mode]
        with_mat = [r for r in rs if r["material_horizons"]]
        mode_aggs[mode] = {
            "state_rows": len(rs),
            "states_with_material": len(with_mat),
            "objective_state_rows_available": int(sum(1 for r in rs if r["objective_rows_available"] > 0)),
            "strict_min_objective_alignment_failures": int(sum(1 for r in rs if r["strict_min_objective_alignment_failure"])),
            "min_objective_ambiguous_ties": int(sum(1 for r in rs if r["min_objective_ambiguous_tie"])),
            "min_objective_selects_material_when_material_exists": int(sum(1 for r in with_mat if r["min_objective_material"] is True)),
            "min_objective_selects_non_success": int(sum(1 for r in rs if r["min_objective_success"] is False)),
            "min_objective_physical_loss_vs_best": finite_summary([r["min_objective_physical_loss_vs_best"] for r in rs if r["min_objective_physical_loss_vs_best"] is not None]),
            "min_objective_total_loss_vs_best": finite_summary([r["min_objective_total_loss_vs_best"] for r in rs if r["min_objective_total_loss_vs_best"] is not None]),
        }

    spread_rows: List[Dict[str, Any]] = []
    for sid in state_ids:
        for h in sorted({int(r.get("horizon")) for r in rows if str(r.get("state_id")) == sid}):
            xrows = [r for r in rows if str(r.get("state_id")) == sid and int(r.get("horizon")) == h]
            vals = [value_fn(r) for r in xrows if math.isfinite(value_fn(r))]
            objs = [objective(r) for r in xrows if math.isfinite(objective(r))]
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
    top_spreads = sorted(spread_rows, key=lambda r: (r["value_fn_range"] if r["value_fn_range"] is not None else -1.0), reverse=True)[:12]

    terminal_flips = len((full.get("analysis") or {}).get("terminal_material_label_flips") or [])
    synthetic_only = len((full.get("analysis") or {}).get("synthetic_only_material_rows") or [])
    robust_same_h = ((post.get("label_density") or {}).get("positive_with_same_h_per_h_and_zero_material"))
    controls_material = ((post.get("label_density") or {}).get("control_states_with_any_material_label"))
    strict_failures_total = sum(v["strict_min_objective_alignment_failures"] for v in mode_aggs.values())
    non_success_objective = sum(v["min_objective_selects_non_success"] for v in mode_aggs.values())
    material_state_mode_rows = sum(v["states_with_material"] for v in mode_aggs.values())
    material_rows_selected = sum(v["min_objective_selects_material_when_material_exists"] for v in mode_aggs.values())
    if non_success_objective > 0:
        next_action = "freeze a fresh terminal-stable label-density probe that excludes unsafe terminal-source choices from training labels; keep h25/common-terminal only as a diagnostic arm, not a selector target"
    elif strict_failures_total > 0:
        next_action = "run a bounded terminal-value/objective refit diagnostic before selector training because objective ranking misses material horizons"
    else:
        next_action = "freeze a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics before selector training"

    created = dt.datetime.now(dt.timezone.utc).isoformat()
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_OBJECTIVE_ALIGNMENT_V0B_SCHEMA_REPAIR_{STAMP}.json"
    four_axis = {
        "SCENARIOS": {
            "verified": f"No fresh states here; mined stress-v1b robust same-H zero positives={robust_same_h}, controls with material={controls_material}.",
            "hypothesis": "Opportunity exists in a narrow high-transient subfamily; generalization remains unproven and must be tested with non-mined development states.",
            "missing": "Fresh terminal-stable label density and state-feature coverage.",
            "next_discriminator": "Freeze and back up a fresh terminal-stable label-density protocol; do not rerun unchanged validation.",
        },
        "REWARD": {
            "verified": f"v0 objective-alignment metrics were schema-incomplete ({v0_missing_objective_selected}/{len(v0_rows)} rows had no selected objective); v0b found objective key coverage {objective_key_coverage}. Terminal flips={terminal_flips}, synthetic-only rows={synthetic_only}, strict min-objective failures={strict_failures_total}, objective-selected non-success rows={non_success_objective}.",
            "hypothesis": "Per-H terminal/source choices materially change realized labels; h25-terminal can produce catastrophic non-success from identical states. Training labels should use realized continuation under zero/common terminal and preserve physical/total/timing separately.",
            "missing": "Whether this terminal-stable label rule yields enough non-mined positives and whether terminal-value refit can repair unsafe source-specific behavior.",
            "next_discriminator": next_action,
        },
        "TRAINING": {
            "verified": "Zero training/refit/gradient steps. Existing historical selectors remain near-constant/reselected policies; no selector should be trained on the per-H terminal labels yet.",
            "hypothesis": "Training failure may be downstream of sparse labels and terminal-source bias, not only policy class capacity.",
            "missing": "A stable label set large enough for a compact IMPROVED selector/value-refit smoke and later 3 independent seeds.",
            "next_discriminator": "After the fresh stable-label density gate, either train/refit a compact selector if dense or revise scenario/terminal modeling if sparse.",
        },
        "COMPARISONS": {
            "verified": "No validation64/sealed-test access and no superiority claim. Synthetic h_penalty remains separate from measured decision/solver timing.",
            "hypothesis": "Strong fixed-H/Pareto baselines may absorb any adaptive gains when terminal artifacts are removed.",
            "missing": "Fair fixed-H retuning and randomized measured timing under a frozen revised protocol.",
            "next_discriminator": "Retune baselines only after method/scenario/terminal protocol is frozen and confirmed on fresh development data.",
        },
    }
    result = {
        "created_utc": created,
        "method": "vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair",
        "classification": "development_no_simulation_objective_alignment_schema_repair_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {rel(p): sha256(p) for p in (FULL_RAW, FULL_DONE, POST_RAW, POST_DONE, V0_RAW, V0_DONE)},
        "schema_repair": {
            "v0_missing_objective_selected_rows": v0_missing_objective_selected,
            "v0_state_mode_rows": len(v0_rows),
            "objective_key_coverage": objective_key_coverage,
            "supersedes_v0_objective_alignment_counts": True,
        },
        "terminal_flips": terminal_flips,
        "synthetic_only_material_rows": synthetic_only,
        "robust_same_h_zero_positive_states": robust_same_h,
        "control_states_with_material": controls_material,
        "material_state_mode_rows": material_state_mode_rows,
        "material_state_mode_rows_min_objective_selects_material": material_rows_selected,
        "strict_min_objective_alignment_failures": strict_failures_total,
        "min_objective_selected_non_success_rows": non_success_objective,
        "mode_alignment_summary": mode_aggs,
        "state_mode_alignment_rows": state_mode_rows,
        "top_terminal_value_spreads": top_spreads,
        "four_axis_evidence": four_axis,
        "decision": {
            "train_or_refit_now": False,
            "reason": "Corrected objective extraction shows objective metrics are now evaluable, but the evidence remains mined/development-only with strong terminal-source flips and unsafe h25-terminal non-success cases; use fresh terminal-stable labels before selector training.",
            "next_action": next_action,
            "backup_required_before_more_simulation_or_training": True,
        },
        "backup_request": rel(backup_request),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "raw.json", result)
    lines = [
        "# Vehicle stress-v1b terminal/objective alignment postdiagnostic v0b schema repair",
        "",
        f"UTC: `{created}`. No simulation/training; validation64 and sealed test stayed closed.",
        "",
        "## Headline",
        "",
        f"- v0 schema issue: `{v0_missing_objective_selected}/{len(v0_rows)}` v0 state-mode rows had no selected objective because the raw key is `branch_objective_opt_f_num`, not `objective`.",
        f"- Objective key coverage in full rows: `{objective_key_coverage}`.",
        f"- Terminal flips: `{terminal_flips}`; synthetic-only material rows: `{synthetic_only}`.",
        f"- Material state-mode rows: `{material_state_mode_rows}`; min-objective selected material in `{material_rows_selected}` of them.",
        f"- Strict min-objective alignment failures: `{strict_failures_total}`; min-objective selected non-success rows: `{non_success_objective}`.",
        f"- Train/refit now: `False`; next: {next_action}.",
        "",
        "## Mode-level objective alignment (lower `branch_objective_opt_f_num` assumed preferred)",
        "",
        "| terminal mode | states | obj rows | states with material | min-obj selects material | strict failures | ambiguous ties | min-obj non-success | mean phys loss | max phys loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        m = mode_aggs[mode]
        phys = m["min_objective_physical_loss_vs_best"]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %d | %d | %s | %s |" % (
            mode, m["state_rows"], m["objective_state_rows_available"], m["states_with_material"],
            m["min_objective_selects_material_when_material_exists"], m["strict_min_objective_alignment_failures"],
            m["min_objective_ambiguous_ties"], m["min_objective_selects_non_success"],
            "" if phys["mean"] is None else "%.6g" % phys["mean"],
            "" if phys["max"] is None else "%.6g" % phys["max"],
        ))
    lines += [
        "",
        "## Material or unsafe state-mode rows",
        "",
        "| state | role | mode | material H | best phys H/gain | min-obj H/gain | min-obj value | ambiguous | strict failure | non-success H |",
        "|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for r in state_mode_rows:
        if r["material_horizons"] or r["non_success_horizons"] or r["strict_min_objective_alignment_failure"]:
            lines.append("| `%s` | `%s` | `%s` | `%s` | H%s/%.6g | H%s/%s | %s | `%s` | `%s` | `%s` |" % (
                r["state_id"], r["role"], r["terminal_mode"], r["material_horizons"],
                r["best_physical_horizon"], r["best_physical_gain"],
                r["min_objective_horizon"], "" if r["min_objective_physical_gain"] is None else "%.6g" % r["min_objective_physical_gain"],
                "" if r["min_objective_value"] is None else "%.6g" % r["min_objective_value"],
                r["min_objective_ambiguous_tie"], r["strict_min_objective_alignment_failure"], r["non_success_horizons"],
            ))
    lines += [
        "",
        "## Largest terminal value/objective spreads",
        "",
        "| state | H | value range | objective range | physical-gain range | material by mode | non-success modes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in top_spreads[:8]:
        lines.append("| `%s` | %d | %s | %s | %s | `%s` | `%s` |" % (
            r["state_id"], r["horizon"],
            "" if r["value_fn_range"] is None else "%.6g" % r["value_fn_range"],
            "" if r["objective_range"] is None else "%.6g" % r["objective_range"],
            "" if r["physical_gain_range"] is None else "%.6g" % r["physical_gain_range"],
            r["material_by_mode"], r["non_success_modes"],
        ))
    lines += ["", "## Four-axis evidence", ""]
    for axis, body in four_axis.items():
        lines += [f"### {axis}", ""]
        for key in ("verified", "hypothesis", "missing", "next_discriminator"):
            lines.append(f"- **{key}:** {body[key]}")
        lines.append("")
    lines += ["## Decision", "", result["decision"]["reason"], "", f"Backup request: `{rel(backup_request)}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(backup_request, {
        "requested_utc": created,
        "reason": "backup corrected terminal/objective alignment diagnostic before fresh terminal-stable simulations or refit",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(backup_request)],
    })
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        "# Continue state after terminal/objective alignment v0b schema repair (%s)\n\n"
        "UTC: %s. No validation64/test/training/simulation. Corrected v0 objective schema: v0 missing selected objective rows %d/%d; objective coverage %s. Terminal flips=%d, synthetic-only rows=%d, robust same-H zero positives=%s, controls material=%s, strict objective failures=%d, objective-selected non-success=%d. Decision: do not train/refit now; next after backup is to freeze/run a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics. Artifacts: `%s`, `%s`.\n"
        % (STAMP, created, v0_missing_objective_selected, len(v0_rows), objective_key_coverage, terminal_flips, synthetic_only, robust_same_h, controls_material, strict_failures_total, non_success_objective, rel(OUT_DIR / "summary.md"), rel(OUT_DIR / "raw.json")),
        encoding="utf-8",
    )
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1b terminal/objective alignment v0b schema repair

UTC: {created}. No-simulation schema-repair postdiagnostic. v0 objective-alignment counts are superseded because v0 looked for `objective` while full rows use `branch_objective_opt_f_num`; v0 missing selected objectives {v0_missing_objective_selected}/{len(v0_rows)}. Corrected coverage {objective_key_coverage}. Terminal flips={terminal_flips}, synthetic-only rows={synthetic_only}, material state-mode rows={material_state_mode_rows}, min-objective selected material in {material_rows_selected}, strict min-objective failures={strict_failures_total}, min-objective selected non-success rows={non_success_objective}. Decision: do not train/refit now; after backup freeze/run a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, state `{rel(STATE_PATH)}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_once(ROOT / doc, block, MARKER)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, backup_request, Path(__file__).resolve(), FULL_RAW, FULL_DONE, POST_RAW, POST_DONE, V0_RAW, V0_DONE]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "headline": {
            "v0_missing_objective_selected_rows": v0_missing_objective_selected,
            "objective_rows": objective_key_coverage,
            "terminal_flips": terminal_flips,
            "synthetic_only_material_rows": synthetic_only,
            "strict_min_objective_alignment_failures": strict_failures_total,
            "min_objective_selected_non_success_rows": non_success_objective,
            "train_or_refit_now": False,
            "next_action": next_action,
        },
        "backup_request": rel(backup_request),
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "summary": rel(OUT_DIR / "summary.md"),
        "completed": rel(OUT_DIR / "completed.json"),
        "state": rel(STATE_PATH),
        "backup_request": rel(backup_request),
        "v0_missing_objective_selected_rows": v0_missing_objective_selected,
        "objective_key_coverage": objective_key_coverage,
        "terminal_flips": terminal_flips,
        "synthetic_only_material_rows": synthetic_only,
        "strict_min_objective_alignment_failures": strict_failures_total,
        "min_objective_selected_non_success_rows": non_success_objective,
        "train_or_refit_now": False,
        "next_action": next_action,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
