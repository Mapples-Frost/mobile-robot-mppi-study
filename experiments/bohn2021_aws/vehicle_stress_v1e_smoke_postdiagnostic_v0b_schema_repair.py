#!/usr/bin/env python3
"""Schema/parse repair for vehicle stress-v1e smoke postdiagnostic v0.

v0 was a no-simulation postdiagnostic, but failed after writing a partial raw.json
because the report lacked ``backup_request`` before summary rendering. Its partial
raw also chose the Markdown table over raw analysis rows on a tie and therefore
misparsed table columns. This v0b source preserves that failure and uses the
parent smoke's authoritative raw ``analysis.state_rows`` instead.

No simulations, no training/refit, no validation64 bank, no sealed-test access.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair"
STAMP = "20260929T0600Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z"
SMOKE_RAW = SMOKE_DIR / "raw.json"
SMOKE_DONE = SMOKE_DIR / "completed.json"
SMOKE_SUMMARY = SMOKE_DIR / "summary.md"
DRYRUN_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_dryrun_20260929T0545Z/completed.json"
PREPARE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z.json"
AMENDMENT = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_amendment_20260929T0545Z.json"
FAILED_V0_PARTIAL = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0_20260929T0550Z/raw.json"
FAILED_V0_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T054700_c0fd8706/registry.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_SMOKE_POSTDIAGNOSTIC_V0B_SCHEMA_REPAIR_{STAMP}.json"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-stress-v1e-smoke-postdiagnostic-v0b-schema-repair-{STAMP}"


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
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
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


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def completed_ok(path: Path) -> Mapping[str, Any]:
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise RuntimeError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise RuntimeError(f"sealed-test flag not false: {rel(path)}")
    if obj.get("validation64_bank_opened") is not False:
        raise RuntimeError(f"validation64 flag not false: {rel(path)}")
    return obj


def sqlite_quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def best_effort_token_total() -> Dict[str, Any]:
    candidates = [
        ROOT / "research.sqlite",
        ROOT / "research_artifacts/research.sqlite",
        ROOT / "scripts/research_service/research.sqlite",
        ROOT.parent / "research.sqlite",
        Path("/data/openai-agent/research.sqlite"),
        Path("/data/openai-agent/state/research.sqlite"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            total = 0
            sources = []
            for table in tables:
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({sqlite_quote(table)})").fetchall()]
                if "total_tokens" not in cols:
                    continue
                value = cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {sqlite_quote(table)}").fetchone()[0]
                total += int(value or 0)
                sources.append({"table": table, "sum_total_tokens": int(value or 0)})
            conn.close()
            return {"path": str(path), "total_tokens": total, "sources": sources, "available": bool(sources)}
        except Exception as exc:
            return {"path": str(path), "available": False, "error": repr(exc), "total_tokens": None}
    return {"available": False, "path": None, "total_tokens": None, "note": "research.sqlite not found from known service paths"}


def finite_summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * p
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "mean": sum(xs) / len(xs), "median": pct(0.5), "p95": pct(0.95), "min": xs[0], "max": xs[-1]}


def extract_rows(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    analysis = raw.get("analysis") or {}
    rows = analysis.get("state_rows") or []
    if not isinstance(rows, list) or len(rows) != int(analysis.get("state_count", 0)):
        raise RuntimeError("parent smoke raw analysis.state_rows missing or inconsistent")
    out: List[Dict[str, Any]] = []
    for r in rows:
        best = r.get("best_by_physical_any_mode") or {}
        mode_results = r.get("mode_results") or {}
        terminal_summaries = {}
        for mode, m in mode_results.items():
            ref = (m or {}).get("reference_H15") or {}
            comps = (m or {}).get("comparisons") or []
            by_h = {int(c.get("horizon")): c for c in comps if c.get("horizon") is not None}
            h10 = by_h.get(10)
            terminal_summaries[str(mode)] = {
                "material_horizons": list((m or {}).get("material_horizons") or []),
                "reference_H15": {
                    "success": ref.get("success"),
                    "constraint": ref.get("constraint"),
                    "steps": ref.get("steps"),
                    "physical": as_float(ref.get("continuation_physical")),
                    "total": as_float(ref.get("continuation_total")),
                    "path": ref.get("path"),
                },
                "H10": None if h10 is None else {
                    "success": h10.get("success"),
                    "constraint": h10.get("constraint"),
                    "steps": h10.get("steps"),
                    "physical": as_float(h10.get("continuation_physical")),
                    "total": as_float(h10.get("continuation_total")),
                    "gain_vs_H15_physical": as_float(h10.get("gain_vs_H15_physical")),
                    "gain_vs_H15_total": as_float(h10.get("gain_vs_H15_total")),
                    "decision_sum_s": as_float(h10.get("decision_sum_s")),
                    "solver_attempt_sum_s": as_float(h10.get("solver_attempt_sum_s")),
                    "solver_failure_steps": h10.get("solver_failure_steps"),
                    "path": h10.get("path"),
                },
            }
        out.append({
            "state_id": str(r.get("state_id")),
            "target_index": int(r.get("target_index")),
            "case": int(r.get("case")),
            "case_role": r.get("case_role"),
            "selection_group": r.get("selection_group"),
            "branch_step": int(r.get("branch_step")),
            "source_candidate_index": int(r.get("source_candidate_index", -1)),
            "is_control_state": bool(r.get("is_control_state")),
            "robust_positive_state": bool(r.get("robust_positive_state")),
            "label": r.get("label"),
            "robust_positive_horizons": list(r.get("robust_positive_horizons") or []),
            "best_by_physical_any_mode": {
                "terminal_mode": best.get("terminal_mode"),
                "horizon": best.get("horizon"),
                "physical": as_float(best.get("continuation_physical")),
                "total": as_float(best.get("continuation_total")),
                "decision_sum_s": as_float(best.get("decision_sum_s")),
                "solver_attempt_sum_s": as_float(best.get("solver_attempt_sum_s")),
                "success": best.get("success"),
                "constraint": best.get("constraint"),
                "path": best.get("path"),
            },
            "terminal_mode_summaries": terminal_summaries,
        })
    return out


def counter(items: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        key = str(item)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(report: Mapping[str, Any]) -> None:
    h = report["headline"]
    lines = [
        "# Vehicle stress-v1e smoke postdiagnostic v0b schema repair",
        "",
        f"UTC: `{report['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Repair scope",
        "",
        "- Parent v1e v0b smoke is unchanged and remains the data source.",
        "- Failed postdiagnostic v0 is preserved as an implementation/reporting failure: it omitted `backup_request` before summary rendering and its partial raw parsed the Markdown table with shifted columns.",
        "- v0b uses authoritative `raw.json -> analysis.state_rows` from the completed smoke, not the Markdown table, so control FPR and labels match the parent runner.",
        "",
        "## Run/access verification",
        "",
        f"- Parent smoke completed: `{report['smoke_completed']}`; episodes `{h['episodes']}`, control steps `{h['control_steps']}`.",
        f"- Access flags: validation64_bank_opened=`{h['validation64_bank_opened']}`, sealed_test_accessed=`{h['sealed_test_accessed']}`.",
        f"- New budgets in this postdiagnostic: rollouts `{report['new_rollouts']}`, control steps `{report['new_control_steps']}`, training episodes `{report['new_training_episodes']}`, gradient steps `{report['new_gradient_steps']}`, refit steps `{report['new_refit_steps']}`.",
        "",
        "## Scientific headline",
        "",
        f"- Robust-positive states: `{h['robust_positive_state_count']}` / `{h['target_count']}` across cases `{h['robust_positive_cases']}`.",
        f"- Negative/neutral states: `{h['negative_or_neutral_state_count']}`.",
        f"- Control positives: `{h['control_positive_state_count']}` / `{h['control_state_count']}` (FPR `{h['control_false_positive_rate']}`).",
        f"- Blocking artifacts: `{h['blocking_artifact_count']}`.",
        f"- Frozen pass-to-selector/value-refit gate: `{h['smoke_pass_to_selector_or_value_refit_design']}`.",
        "",
        "The result partially supports state-selection coverage failure because targeted missed case 5 produced terminal-stable positives. It does **not** justify selector/value refit: positives are confined to one case, so the frozen >=2-case gate fails. This sparse-label conclusion applies to this supervised selector/refit path, not to every possible value/modeling intervention.",
        "",
        "## Gate components",
        "",
        "| component | required | observed | pass |",
        "|---|---:|---:|---|",
    ]
    for key, row in report["gate_components"].items():
        lines.append(f"| `{key}` | `{row['required']}` | `{row['observed']}` | `{row['pass']}` |")
    lines += ["", "## Positive states (development only)", ""]
    positives = report["positive_state_rows"]
    if positives:
        lines += ["| state | case | role | branch | robust H | best physical mode/H/cost | H10 h15-common physical gain |", "|---|---:|---|---:|---|---|---:|"]
        for row in positives:
            best = row.get("best_by_physical_any_mode") or {}
            h10_common = ((row.get("terminal_mode_summaries") or {}).get("h15_common_terminal") or {}).get("H10") or {}
            lines.append("| `%s` | %s | `%s` | %s | `%s` | `%s`/H%s/%.6g | %s |" % (
                row.get("state_id"), row.get("case"), row.get("case_role"), row.get("branch_step"), row.get("robust_positive_horizons"), best.get("terminal_mode"), best.get("horizon"), float(best.get("physical") or 0.0), h10_common.get("gain_vs_H15_physical")))
    else:
        lines.append("No robust-positive state rows.")
    lines += [
        "",
        "## Four-axis decision update",
        "",
        "- **Scenarios:** same stress-v1 target distribution produced opportunity only in case 5; case 4 did not become robust-positive even after targeted H15-prefix coverage.",
        "- **Reward/terminal:** labels are realised terminal-stable physical improvements; previous raw-objective and simple terminal/ridge repairs remain locked out.",
        "- **Training:** no new training occurred and no selector/refit is justified from this case-concentrated label bank. A later value/modeling experiment remains possible if it has a separate falsifiable hypothesis.",
        "- **Comparisons/timing:** this is not validation and not a speed claim. The next useful simulation is a repeated paired positive-state timing/stability check for case 5 H10 vs H15, not another unchanged label-density sweep.",
        "",
        "## Decision and next bounded action",
        "",
        f"- Train/refit now: `{report['decision']['train_or_refit_now']}`.",
        f"- Next action after backup: `{report['decision']['next_action']}`.",
        f"- Backup request: `{report['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for p in (SMOKE_RAW, SMOKE_DONE, SMOKE_SUMMARY, DRYRUN_DONE, PREPARE_PROTOCOL, AMENDMENT):
        if not p.exists():
            raise RuntimeError(f"required artifact missing: {rel(p)}")
    OUT.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc)
    done = completed_ok(SMOKE_DONE)
    completed_ok(DRYRUN_DONE)
    raw = read_json(SMOKE_RAW)
    analysis = raw.get("analysis") or {}
    rows = extract_rows(raw)
    positives = [r for r in rows if r["robust_positive_state"]]
    controls = [r for r in rows if r["is_control_state"]]
    control_pos = [r for r in controls if r["robust_positive_state"]]
    robust_cases = sorted({int(r["case"]) for r in positives})
    robust_count = int(analysis.get("robust_positive_state_count", len(positives)))
    negative_count = int(analysis.get("negative_or_neutral_state_count", len(rows) - len(positives)))
    control_total = int(analysis.get("control_state_count", len(controls)))
    control_positive = int(analysis.get("control_positive_state_count", len(control_pos)))
    control_fpr = float(analysis.get("control_false_positive_rate", (control_positive / control_total if control_total else 0.0)))
    blocking = int(analysis.get("blocking_artifact_count", 0))
    gate = bool(analysis.get("smoke_pass_to_selector_or_value_refit_design"))
    if robust_count != len(positives) or sorted(analysis.get("robust_positive_cases") or []) != robust_cases:
        raise RuntimeError("parent smoke analysis aggregate does not match extracted raw state rows")
    gate_components = {
        "robust_positive_states_ge_2": {"required": ">=2", "observed": robust_count, "pass": robust_count >= 2},
        "robust_positive_cases_ge_2": {"required": ">=2", "observed": len(robust_cases), "pass": len(robust_cases) >= 2},
        "retained_negative_or_control_states_ge_4": {"required": ">=4", "observed": negative_count, "pass": negative_count >= 4},
        "control_false_positive_rate_le_0p25": {"required": "<=0.25", "observed": round(control_fpr, 6), "pass": control_fpr <= 0.25},
        "blocking_artifacts_eq_0": {"required": "0", "observed": blocking, "pass": blocking == 0},
    }
    computed_gate = all(bool(v["pass"]) for v in gate_components.values())
    if computed_gate != gate:
        raise RuntimeError(f"computed gate {computed_gate} disagrees with parent analysis gate {gate}")
    if gate:
        next_action = "freeze compact IMPROVED selector/value-refit smoke with strong fixed-H/Pareto baselines; still no validation64/sealed-test use"
    elif robust_count > 0:
        next_action = "freeze and, after verified backup, run a small repeated positive-state stability/timing diagnostic for the two case-5 robust-positive H15-prefix states using paired H10 versus H15 under both terminal modes"
    else:
        next_action = "do not refit from this bank; pivot to terminal/modeling or sparse-opportunity scenario diagnosis rather than another unchanged label-density sweep"
    token_info = best_effort_token_total()
    report: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "server_api_total_tokens_best_effort": token_info,
        "classification": "development_no_simulation_v1e_smoke_postdiagnostic_v0b_schema_repair_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "smoke_completed": rel(SMOKE_DONE),
        "smoke_summary": rel(SMOKE_SUMMARY),
        "smoke_raw": rel(SMOKE_RAW),
        "failed_v0_postdiagnostic_partial_raw": rel(FAILED_V0_PARTIAL) if FAILED_V0_PARTIAL.exists() else None,
        "failed_v0_registry": rel(FAILED_V0_REGISTRY) if FAILED_V0_REGISTRY.exists() else None,
        "backup_request": rel(BACKUP_REQUEST),
        "headline": {
            "episodes": int(done.get("episodes", (raw.get("budget_actual") or {}).get("episodes", 0))),
            "control_steps": int(done.get("control_steps", (raw.get("budget_actual") or {}).get("control_steps", 0))),
            "target_count": len(rows),
            "robust_positive_state_count": robust_count,
            "robust_positive_cases": robust_cases,
            "negative_or_neutral_state_count": negative_count,
            "control_state_count": control_total,
            "control_positive_state_count": control_positive,
            "control_false_positive_rate": round(control_fpr, 6),
            "blocking_artifact_count": blocking,
            "smoke_pass_to_selector_or_value_refit_design": gate,
            "schema_repair": "postdiagnostic uses parent raw analysis.state_rows; failed v0 omitted backup_request and misparsed md table",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "gate_components": gate_components,
        "state_rows": rows,
        "positive_state_rows": positives,
        "positive_cases_by_role": counter(r.get("case_role") for r in positives),
        "timing_decision_sum_by_horizon_mode_s_from_parent_single_run": analysis.get("timing_decision_sum_by_horizon_mode_s"),
        "decision": {"train_or_refit_now": False, "do_not_train_from_v1e_labels_because_gate_failed": not gate, "next_action": next_action},
        "source_hashes": {
            rel(SOURCE): sha256(SOURCE),
            rel(SMOKE_RAW): sha256(SMOKE_RAW),
            rel(SMOKE_DONE): sha256(SMOKE_DONE),
            rel(SMOKE_SUMMARY): sha256(SMOKE_SUMMARY),
            rel(DRYRUN_DONE): sha256(DRYRUN_DONE),
            rel(PREPARE_PROTOCOL): sha256(PREPARE_PROTOCOL),
            rel(AMENDMENT): sha256(AMENDMENT),
        },
        "interpretation_limits": [
            "development diagnostic only",
            "not validation/model selection",
            "not final test",
            "not a speed claim from horizon length or single-run branch timings",
            "selector/value-refit gate failed because positives did not span enough cases",
            "failed postdiagnostic v0 partial raw is superseded by this schema/parse repair",
        ],
    }
    write_json(OUT / "raw.json", report)
    write_summary(report)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(BACKUP_REQUEST, {
        "requested_utc": created.isoformat(),
        "reason": "backup repaired v1e smoke postdiagnostic and parent v0b smoke outputs before any repeated positive-state timing/stability diagnostic or training/refit decision",
        "backup_required_before_more_simulations": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(SOURCE), rel(SMOKE_DIR), rel(BACKUP_REQUEST), rel(FAILED_V0_PARTIAL) if FAILED_V0_PARTIAL.exists() else "failed_v0_partial_raw_absent"],
    })
    docs_block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1e smoke postdiagnostic v0b schema repair

UTC: {created.isoformat()}. No-simulation repaired postdiagnostic of the completed v1e v0b schema-repair smoke. Failed postdiagnostic v0 is preserved as a reporting/parser failure (missing `backup_request`; Markdown table columns misparsed) and is superseded. Authoritative parent raw analysis shows {robust_count} robust-positive states across cases {robust_cases}, {negative_count} negative/neutral states, control positives {control_positive}/{control_total}, blocking artifacts {blocking}, and gate={gate}. Positives are concentrated in case 5, so do not train/refit a selector from this bank. Next bounded action after backup: {next_action}. No validation64 bank or sealed-test access; no training/refit. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    append_docs(docs_block)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE, BACKUP_REQUEST, SMOKE_RAW, SMOKE_DONE, SMOKE_SUMMARY, DRYRUN_DONE, PREPARE_PROTOCOL, AMENDMENT]
    if FAILED_V0_PARTIAL.exists():
        files.append(FAILED_V0_PARTIAL)
    if FAILED_V0_REGISTRY.exists():
        files.append(FAILED_V0_REGISTRY)
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "server_api_total_tokens_best_effort": token_info,
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQUEST),
        "headline": report["headline"],
        "decision": report["decision"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "elapsed_since_first_supervisor_event_seconds": round((created - FIRST_SUPERVISOR_EVENT).total_seconds(), 3),
        "server_api_total_tokens_best_effort": token_info,
        "robust_positive_state_count": robust_count,
        "robust_positive_cases": robust_cases,
        "negative_or_neutral_state_count": negative_count,
        "control_positive_state_count": control_positive,
        "control_state_count": control_total,
        "smoke_pass_to_selector_or_value_refit_design": gate,
        "train_or_refit_now": False,
        "next_action": next_action,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQUEST),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
