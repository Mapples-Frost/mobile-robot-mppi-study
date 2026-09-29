#!/usr/bin/env python3
"""Postdiagnostic for the full v0c vehicle stress-v1b terminal/reward ablation.

No simulation, no validation64/sealed-test access, no training/refit.  This
script reads only the completed development full-ablation artifacts and converts
them into a state-level diagnosis: terminal/source robustness, corrected-label
density, failure/safety artifacts, and the next discriminating experiment.  It
exists because the full run completed after the previous state handoff; the next
bounded action should be evidence extraction rather than another rollout.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0018Z"
FULL_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair"
FULL_RAW = FULL_DIR / "raw.json"
FULL_DONE = FULL_DIR / "completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_full_postdiagnostic_v0_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_full_terminal_reward_ablation_postdiagnostic.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-stress-v1b-terminal-reward-ablation-full-postdiagnostic-v0-{STAMP}"
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


def finite(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"count": n, "min": vals[0], "median": med, "mean": math.fsum(vals) / n, "max": vals[-1]}


def row_key(r: Mapping[str, Any]) -> Tuple[str, int, str]:
    return str(r["state_id"]), int(r["horizon"]), str(r["terminal_mode"])


def main() -> int:
    if not FULL_RAW.exists() or not FULL_DONE.exists():
        raise RuntimeError("full ablation artifacts missing")
    done = read_json(FULL_DONE)
    raw = read_json(FULL_RAW)
    if done.get("passed") is not True or raw.get("sealed_test_accessed") is not False or raw.get("historical_validation64_bank_opened") is not False:
        raise RuntimeError("full ablation marker/access flags invalid")
    if raw.get("budget_actual", {}).get("new_training_episodes", 0) != 0 or raw.get("budget_actual", {}).get("new_gradient_steps", 0) != 0:
        raise RuntimeError("unexpected training budget in full ablation")

    analysis = raw["analysis"]
    rows = list(analysis["comparison_rows"])
    sens = list(analysis["terminal_sensitivity_rows"])
    by_key = {row_key(r): r for r in rows}
    state_ids = sorted({str(r["state_id"]) for r in rows})

    state_reports: List[Dict[str, Any]] = []
    positive_reports: List[Dict[str, Any]] = []
    control_reports: List[Dict[str, Any]] = []
    robust_same_h_zero_positive = []
    matched_h15_terminal_positive = []
    per_h_positive = []
    any_control_material = []
    failure_rows = []
    extreme_terminal_rows = []

    for sid in state_ids:
        srows = [r for r in rows if r["state_id"] == sid]
        role = str(srows[0].get("state_role"))
        horizons = sorted({int(r["horizon"]) for r in srows if int(r["horizon"]) != 15})
        per_h_mat = sorted([int(r["horizon"]) for r in srows if r["terminal_mode"] == "per_h" and r.get("material_any")])
        zero_mat = sorted([int(r["horizon"]) for r in srows if r["terminal_mode"] == "zero_terminal" and r.get("material_any")])
        h15term_mat = sorted([int(r["horizon"]) for r in srows if r["terminal_mode"] == "h15_terminal" and r.get("material_any")])
        h25term_mat = sorted([int(r["horizon"]) for r in srows if r["terminal_mode"] == "h25_terminal" and r.get("material_any")])
        same_h_zero = sorted(set(per_h_mat).intersection(zero_mat))
        same_h_h15term = sorted(set(per_h_mat).intersection(h15term_mat))
        state_sens = [x for x in sens if x["state_id"] == sid]
        flips = [x for x in state_sens if x.get("material_label_flip")]
        major = [x for x in state_sens if abs(float(x.get("physical_delta_vs_per_h", 0.0))) >= MATERIAL_GAIN or abs(float(x.get("total_delta_vs_per_h", 0.0))) >= MATERIAL_GAIN]
        non_success = [r for r in srows if not bool(r.get("success"))]
        physical_gain_by_mode = {
            mode: finite([float(r["gain_vs_per_h_H15_physical"]) for r in srows if r["terminal_mode"] == mode and int(r["horizon"]) != 15])
            for mode in sorted({str(r["terminal_mode"]) for r in srows})
        }
        report = {
            "state_id": sid,
            "role": role,
            "target_index": int(srows[0].get("target_index", -1)),
            "case": int(srows[0].get("case", -1)),
            "branch_step": int(srows[0].get("branch_step", -1)),
            "candidate_horizons": horizons,
            "per_h_material_horizons": per_h_mat,
            "zero_terminal_material_horizons": zero_mat,
            "h15_terminal_material_horizons": h15term_mat,
            "h25_terminal_material_horizons": h25term_mat,
            "same_h_per_h_and_zero_material_horizons": same_h_zero,
            "same_h_per_h_and_h15_terminal_material_horizons": same_h_h15term,
            "terminal_label_flips": len(flips),
            "major_terminal_cost_deltas": len(major),
            "non_success_rows": [row_key(r) for r in non_success],
            "max_abs_terminal_physical_delta": max([abs(float(x.get("physical_delta_vs_per_h", 0.0))) for x in state_sens] or [0.0]),
            "physical_gain_summary_by_terminal_mode": physical_gain_by_mode,
        }
        state_reports.append(report)
        if role == "corrected_positive":
            positive_reports.append(report)
            if per_h_mat:
                per_h_positive.append(sid)
            if same_h_zero:
                robust_same_h_zero_positive.append({"state_id": sid, "horizons": same_h_zero})
            if same_h_h15term:
                matched_h15_terminal_positive.append({"state_id": sid, "horizons": same_h_h15term})
        else:
            control_reports.append(report)
            if per_h_mat or zero_mat or h15term_mat:
                any_control_material.append(report)
        failure_rows.extend([r for r in srows if not bool(r.get("success"))])

    for x in sens:
        if abs(float(x.get("physical_delta_vs_per_h", 0.0))) >= 50.0 or bool(x.get("material_label_flip")):
            extreme_terminal_rows.append(x)

    label_density = {
        "corrected_positive_states": len(positive_reports),
        "positive_with_per_h_material": len(per_h_positive),
        "positive_with_same_h_per_h_and_zero_material": len(robust_same_h_zero_positive),
        "positive_with_same_h_per_h_and_h15_terminal_material": len(matched_h15_terminal_positive),
        "control_states": len(control_reports),
        "control_states_with_any_material_label": len(any_control_material),
        "full_episodes": int(raw["budget_actual"]["episodes"]),
        "full_control_steps": int(raw["budget_actual"]["control_steps"]),
    }

    # Conservative decision: terminal sensitivity is strong and robust labels are
    # mined/sparse.  The next experiment should not train a selector from per-H
    # labels; it should freeze a terminal-source/matched-terminal or terminal-
    # value-refit protocol on fresh development states.
    terminal_flips = int(len(analysis.get("terminal_material_label_flips") or []))
    synthetic_only = int(len(analysis.get("synthetic_only_material_rows") or []))
    robust_same_h_count = label_density["positive_with_same_h_per_h_and_zero_material"]
    train_now = False
    if terminal_flips == 0 and robust_same_h_count >= 3 and label_density["control_states_with_any_material_label"] == 0:
        next_action = "freeze compact selector/refit smoke on fresh development states"
    else:
        next_action = "freeze matched/zero-terminal label protocol or terminal-value-refit diagnostic before selector training"

    four_axis = {
        "SCENARIOS": {
            "verified": f"Full mined-state ablation covered {len(positive_reports)} corrected positives and {len(control_reports)} fixed controls; branch-state identity summary {analysis.get('prefix_branch_state_distance_summary')}; robust same-H zero-terminal positives={robust_same_h_count}/{len(positive_reports)}.",
            "hypothesis": "State-dependent opportunity exists locally but remains sparse/mined and may be too terminal-source-dependent for robust generalization.",
            "missing": "Fresh non-mined development states and revised scenario distribution label density under a terminal-source-stable protocol.",
            "next_discriminator": "Freeze a fresh matched/zero-terminal label-density probe (or terminal-value refit probe) before selector training or validation claims.",
        },
        "REWARD": {
            "verified": f"Full ablation terminal material-label flips={terminal_flips}; major terminal deltas={len(analysis.get('major_terminal_cost_deltas_abs_ge_material_threshold') or [])}; synthetic-only material rows={synthetic_only}.",
            "hypothesis": "Per-H terminal values/source labels are a material confound; synthetic horizon penalty is not the positive-label driver but total cost must stay separate from measured runtime.",
            "missing": "Whether zero-terminal or common-terminal labels are stable enough across fresh states, and whether terminal-value refit removes catastrophic/source-specific failures.",
            "next_discriminator": "Use identical branch states with common terminal-value source/zero terminal and actual timing, or run a bounded terminal-value refit ablation if label density warrants.",
        },
        "TRAINING": {
            "verified": "This diagnostic performed zero training/refit/gradient steps; current policies remain historical finite-search/reselection. Per-H label training remains unsafe from this evidence.",
            "hypothesis": "Near-constant learned policies may reflect sparse robust labels and terminal-value bias rather than only policy-capacity failure.",
            "missing": "Training/selector coverage on labels generated by artifact-resistant terminal protocol; value-terminal accuracy if refit is attempted.",
            "next_discriminator": "Do not retrain on per-H labels now; after a frozen stable-label protocol, run a compact selector/value-refit smoke with explicit label-density and safety criteria.",
        },
        "COMPARISONS": {
            "verified": "No validation64/sealed-test access and no fixed-H reproduction claim. Full ablation is development-only and cannot establish superiority.",
            "hypothesis": "Strong fixed-H/Pareto baselines may absorb gains once terminal-source artifacts are removed; timing must be randomized/measured rather than synthetic.",
            "missing": "Fair fixed-H retuning under any revised terminal/scenario protocol; matched-terminal and independent-terminal baseline comparisons.",
            "next_discriminator": "Only after method/protocol revision is frozen, retune fixed-H baselines and evaluate on fresh paired validation; keep final test sealed.",
        },
    }

    created = dt.datetime.now(dt.timezone.utc).isoformat()
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_FULL_POSTDIAGNOSTIC_V0_{STAMP}.json"
    result = {
        "created_utc": created,
        "method": "vehicle_stress_v1b_terminal_reward_ablation_full_postdiagnostic_v0_no_simulation",
        "classification": "development_postdiagnostic_no_simulation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_full_raw": rel(FULL_RAW),
        "input_full_completed": rel(FULL_DONE),
        "input_hashes": {rel(FULL_RAW): sha256(FULL_RAW), rel(FULL_DONE): sha256(FULL_DONE)},
        "label_density": label_density,
        "state_reports": state_reports,
        "failure_rows": failure_rows,
        "extreme_or_flipped_terminal_sensitivity_rows": extreme_terminal_rows,
        "four_axis_evidence": four_axis,
        "decision": {
            "train_or_refit_now": train_now,
            "reason": "Full ablation still shows strong terminal-source sensitivity and mined/sparse robust labels; selector/refit should wait for artifact-resistant labels or terminal-value refit evidence.",
            "next_action": next_action,
            "backup_required_before_more_simulation_or_training": True,
        },
        "backup_request": rel(backup_request),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "raw.json", result)
    lines = [
        "# Vehicle stress-v1b terminal/reward ablation full postdiagnostic v0",
        "",
        f"UTC: `{created}`. No simulation/training; validation64 and sealed test stayed closed.",
        "",
        "## Headline",
        "",
        f"- Full ablation input: `{label_density['full_episodes']}` episodes / `{label_density['full_control_steps']}` control steps.",
        f"- Terminal material-label flips: `{terminal_flips}`; synthetic-only material rows: `{synthetic_only}`.",
        f"- Corrected positives with same-H per-H and zero-terminal material label: `{robust_same_h_count}/{len(positive_reports)}`.",
        f"- Controls with any material label: `{label_density['control_states_with_any_material_label']}/{len(control_reports)}`.",
        f"- Train/refit now: `{train_now}`; next action: {next_action}.",
        "",
        "## State-level labels",
        "",
        "| state | role | per-H material H | zero-terminal material H | same-H robust zero H | H15-terminal material H | flips | major deltas | failures |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]
    for r in state_reports:
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %d | %d | `%s` |" % (
            r["state_id"], r["role"], r["per_h_material_horizons"], r["zero_terminal_material_horizons"], r["same_h_per_h_and_zero_material_horizons"], r["h15_terminal_material_horizons"], int(r["terminal_label_flips"]), int(r["major_terminal_cost_deltas"]), r["non_success_rows"]
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
        "reason": "backup full terminal/reward-source ablation outputs and postdiagnostic before further simulation/training/refit",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(FULL_DIR), rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(backup_request)],
    })

    state_text = "\n".join([
        f"# Continue state after full terminal/reward ablation postdiagnostic ({STAMP})",
        "",
        f"UTC: {created}. Full v0c development ablation completed and postdiagnosed. No validation64/sealed-test/training access in the postdiagnostic.",
        "",
        f"Key evidence: full episodes={label_density['full_episodes']}, control steps={label_density['full_control_steps']}, terminal flips={terminal_flips}, synthetic-only material rows={synthetic_only}, robust same-H zero-terminal positives={robust_same_h_count}/{len(positive_reports)}, controls with any material={label_density['control_states_with_any_material_label']}/{len(control_reports)}.",
        "",
        "Decision: do not train/refit a selector from current per-H terminal labels. Next informative action after backup is to freeze an artifact-resistant matched/zero-terminal label-density probe or a bounded terminal-value-refit diagnostic, then only train a compact IMPROVED selector/value refit if dense stable labels are found.",
        "",
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`. Backup request: `{rel(backup_request)}`.",
    ])
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state_text + "\n", encoding="utf-8")

    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1b terminal/reward-source ablation full postdiagnostic

UTC: {created}. No-simulation postdiagnostic of the completed full v0c ablation. Full input budget was {label_density['full_episodes']} episodes / {label_density['full_control_steps']} control steps; validation64 and sealed test stayed closed; no training/refit. Terminal flips={terminal_flips}, synthetic-only material rows={synthetic_only}, robust same-H zero-terminal positives={robust_same_h_count}/{len(positive_reports)}, controls with material labels={label_density['control_states_with_any_material_label']}/{len(control_reports)}. Decision: no selector/refit from current per-H labels; next after backup is an artifact-resistant matched/zero-terminal label-density probe or terminal-value-refit diagnostic. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, state `{rel(STATE_PATH)}`. Backup request: `{rel(backup_request)}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_once(ROOT / doc, block, MARKER)

    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, backup_request, Path(__file__).resolve(), FULL_RAW, FULL_DONE]
    completed = {
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
        "headline": {"terminal_flips": terminal_flips, "synthetic_only_material_rows": synthetic_only, "robust_same_h_zero_positive_states": robust_same_h_count, "train_or_refit_now": train_now, "next_action": next_action},
        "backup_request": rel(backup_request),
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    }
    write_json(OUT_DIR / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "state": rel(STATE_PATH),
        "terminal_flips": terminal_flips,
        "synthetic_only_material_rows": synthetic_only,
        "robust_same_h_zero_positive_states": robust_same_h_count,
        "control_states_with_material": label_density["control_states_with_any_material_label"],
        "train_or_refit_now": train_now,
        "next_action": next_action,
        "backup_request": rel(backup_request),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
