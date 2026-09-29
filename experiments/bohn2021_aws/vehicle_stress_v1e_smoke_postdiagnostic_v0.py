#!/usr/bin/env python3
"""Postdiagnostic for vehicle stress-v1e targeted common-prefix smoke v0b.

No simulation, no validation64/sealed-test access, no training/refit.  This
summarises the just-completed v1e v0b schema-repair smoke, checks the frozen
selector/value-refit gate components, and preserves a next-action handoff.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_stress_v1e_smoke_postdiagnostic_v0"
STAMP = "20260929T0550Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z"
SMOKE_RAW = SMOKE_DIR / "raw.json"
SMOKE_DONE = SMOKE_DIR / "completed.json"
SMOKE_SUMMARY = SMOKE_DIR / "summary.md"
DRYRUN_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_dryrun_20260929T0545Z/completed.json"
PREPARE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z.json"
AMENDMENT = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_amendment_20260929T0545Z.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_SMOKE_POSTDIAGNOSTIC_V0_{STAMP}.json"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-stress-v1e-smoke-postdiagnostic-v0-{STAMP}"


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def remove_ticks(cell: str) -> str:
    return cell.replace("`", "").strip()


def completed_ok(path: Path) -> Mapping[str, Any]:
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise RuntimeError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise RuntimeError(f"sealed-test flag not false: {rel(path)}")
    if obj.get("validation64_bank_opened") is not False:
        raise RuntimeError(f"validation64 flag not false: {rel(path)}")
    return obj


def best_effort_token_total() -> Dict[str, Any]:
    candidates = [
        ROOT / "research.sqlite",
        ROOT / "research_artifacts/research.sqlite",
        ROOT / "scripts/research_service/research.sqlite",
        ROOT.parent / "research.sqlite",
        Path("/data/openai-agent/research.sqlite"),
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
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" not in cols:
                    continue
                value = cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0]
                total += int(value or 0)
                sources.append({"table": table, "sum_total_tokens": int(value or 0)})
            conn.close()
            return {"path": rel(path), "total_tokens": total, "sources": sources, "available": bool(sources)}
        except Exception as exc:
            return {"path": rel(path), "available": False, "error": repr(exc)}
    return {"available": False, "path": None, "total_tokens": None, "note": "research.sqlite not found from postdiagnostic script search paths"}


def collect_state_lists(obj: Any, path: str = "$", depth: int = 0, out: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if out is None:
        out = []
    if depth > 7:
        return out
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        n = len(obj)
        stateish = sum(1 for x in obj if any(k in x for k in ("state_id", "target_state_id", "target_id", "state")))
        labelish = sum(1 for x in obj if any(k in x for k in ("label", "classification", "is_robust_positive", "robust_positive", "robust_horizons", "robust_positive_horizons")))
        targetish = sum(1 for x in obj if any(k in x for k in ("target_index", "case", "case_role", "role")))
        if n >= 2 and stateish >= max(1, n // 2) and (labelish >= 1 or (n <= 20 and targetish >= max(1, n // 2))):
            out.append({"path": path, "length": n, "stateish": stateish, "labelish": labelish, "targetish": targetish, "rows": obj})
        for i, x in enumerate(obj[:5]):
            collect_state_lists(x, f"{path}[{i}]", depth + 1, out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"branch_episodes", "schedule"}:
                # These are large episode-level/schedule lists; state-label rows are usually in analysis.
                continue
            collect_state_lists(v, f"{path}.{k}", depth + 1, out)
    return out


def parse_ints(text: Any) -> List[int]:
    if isinstance(text, (list, tuple)):
        vals = []
        for x in text:
            y = as_int(x)
            if y is not None:
                vals.append(y)
        return sorted(set(vals))
    return sorted(set(int(x) for x in re.findall(r"(?<!\d)(?:H)?(10|15|20|25|30|35|45|50)(?!\d)", str(text))))


def normalize_row(row: Mapping[str, Any], target_by_index: Mapping[int, Mapping[str, Any]], target_by_state: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    state_id = str(row.get("state_id") or row.get("target_state_id") or row.get("target_id") or row.get("state") or "")
    ti = as_int(row.get("target_index"))
    target = target_by_index.get(ti) if ti is not None else None
    if target is None and state_id:
        target = target_by_state.get(state_id)
    target = target or {}
    label = str(row.get("label") or row.get("classification") or row.get("state_label") or "")
    robust = row.get("robust_horizons") or row.get("robust_positive_horizons") or row.get("robust_H") or row.get("robust_positive_H") or []
    best_phys = row.get("best_phys") or row.get("best_physical") or row.get("best_physical_mode_h_cost") or row.get("best_phys_mode_h_cost")
    out = {
        "state_id": state_id or str(target.get("state_id") or ""),
        "target_index": ti if ti is not None else as_int(target.get("target_index")),
        "case": as_int(row.get("case"), as_int(target.get("case"))),
        "case_role": str(row.get("case_role") or row.get("role") or target.get("case_role") or target.get("role") or ""),
        "selection_group": str(row.get("selection_group") or target.get("selection_group") or ""),
        "branch_step": as_int(row.get("branch_step"), as_int(target.get("branch_step"))),
        "label": label,
        "robust_horizons": parse_ints(robust),
        "best_physical": best_phys,
        "raw_keys": sorted(str(k) for k in row.keys())[:80],
    }
    return out


def row_is_positive(row: Mapping[str, Any]) -> bool:
    label = str(row.get("label") or "").lower()
    if "negative" in label or "neutral" in label:
        return False
    if "positive" in label:
        return True
    if row.get("is_robust_positive") is True or row.get("robust_positive") is True:
        return True
    return bool(row.get("robust_horizons"))


def parse_summary_rows(text: str, target_by_state: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("## Per-state labels"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.strip().startswith("|"):
            continue
        if "---" in line or "state" in line.lower() and "label" in line.lower():
            continue
        parts = [remove_ticks(p) for p in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        state_id = parts[0]
        if not state_id or state_id.lower() == "state":
            continue
        case = as_int(parts[1])
        branch = as_int(parts[4]) if len(parts) > 4 else None
        label = parts[5] if len(parts) > 5 else ""
        robust = parse_ints(parts[6] if len(parts) > 6 else "")
        best = parts[7] if len(parts) > 7 else ""
        target = target_by_state.get(state_id, {})
        rows.append({
            "state_id": state_id,
            "target_index": as_int(target.get("target_index")),
            "case": case if case is not None else as_int(target.get("case")),
            "case_role": str(target.get("case_role") or target.get("role") or parts[3] if len(parts) > 3 else ""),
            "selection_group": parts[2] if len(parts) > 2 else str(target.get("selection_group") or ""),
            "branch_step": branch,
            "label": label,
            "robust_horizons": robust,
            "best_physical": best,
            "source": "summary_md_per_state_table",
        })
    return rows


def choose_state_rows(raw: Mapping[str, Any], summary_text: str, targets: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_by_index = {int(t.get("target_index", i)): t for i, t in enumerate(targets)}
    target_by_state = {str(t.get("state_id")): t for t in targets if t.get("state_id")}
    candidates = collect_state_lists(raw.get("analysis") or {})
    normalized_candidates: List[Dict[str, Any]] = []
    expected = len(targets) or 12
    for cand in candidates:
        norm = [normalize_row(r, target_by_index, target_by_state) for r in cand["rows"]]
        nonempty_labels = sum(1 for r in norm if r.get("label") or r.get("robust_horizons"))
        score = 5 * int(len(norm) == expected) + 2 * min(nonempty_labels, expected) - abs(len(norm) - expected)
        normalized_candidates.append({"path": cand["path"], "length": cand["length"], "score": score, "positive_count": sum(1 for r in norm if row_is_positive(r)), "rows": norm})
    md_rows = parse_summary_rows(summary_text, target_by_state)
    if md_rows:
        normalized_candidates.append({"path": "summary_md.Per-state labels", "length": len(md_rows), "score": 5 * int(len(md_rows) == expected) + 2 * sum(1 for r in md_rows if r.get("label")), "positive_count": sum(1 for r in md_rows if row_is_positive(r)), "rows": md_rows})
    if not normalized_candidates:
        # Last resort: target metadata only, enough to preserve case/role concentration but not labels.
        target_rows = [normalize_row(dict(t), target_by_index, target_by_state) for t in targets]
        return target_rows, []
    normalized_candidates.sort(key=lambda x: (x["score"], x["positive_count"], -abs(x["length"] - expected)), reverse=True)
    return normalized_candidates[0]["rows"], [{k: v for k, v in c.items() if k != "rows"} for c in normalized_candidates[:5]]


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
    gate = report["gate_components"]
    positives = report["positive_state_rows"]
    lines = [
        "# Vehicle stress-v1e targeted common-prefix smoke postdiagnostic v0",
        "",
        f"UTC: `{report['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Run/access verification",
        "",
        f"- Parent v1e v0b smoke completed: `{report['smoke_completed']}`; episodes `{h['episodes']}`, control steps `{h['control_steps']}`.",
        f"- Access flags: validation64_bank_opened=`{h['validation64_bank_opened']}`, sealed_test_accessed=`{h['sealed_test_accessed']}`.",
        f"- Schema repair was implementation-only: `{h['schema_repair']}`.",
        "",
        "## Scientific headline",
        "",
        f"- Robust-positive states: `{h['robust_positive_state_count']}` / `{h['target_count']}` across cases `{h['robust_positive_cases']}`.",
        f"- Negative/neutral states: `{h['negative_or_neutral_state_count']}`.",
        f"- Control positives: `{h['control_positive_state_count']}` / `{h['control_state_count']}` (false-positive rate `{h['control_false_positive_rate']}`).",
        f"- Blocking artifacts: `{h['blocking_artifact_count']}`.",
        f"- Frozen pass-to-selector/value-refit gate: `{h['smoke_pass_to_selector_or_value_refit_design']}`.",
        "",
        "The outcome partially supports the coverage-failure hypothesis because targeted missed stress-v1 case 5 produced terminal-stable positives, but it does not support a general selector/value-refit yet: positives are concentrated in one case and the frozen gate requires positives across at least two cases.",
        "",
        "## Gate components",
        "",
        "| component | required | observed | pass |",
        "|---|---:|---:|---|",
    ]
    for key, row in gate.items():
        lines.append(f"| `{key}` | `{row['required']}` | `{row['observed']}` | `{row['pass']}` |")
    lines += ["", "## Positive states (development only)", ""]
    if positives:
        lines += ["| state | case | role | branch | robust H | best physical row |", "|---|---:|---|---:|---|---|"]
        for row in positives:
            lines.append(f"| `{row.get('state_id')}` | {row.get('case')} | `{row.get('case_role')}` | {row.get('branch_step')} | `{row.get('robust_horizons')}` | `{row.get('best_physical')}` |")
    else:
        lines.append("No robust-positive state rows were recovered.")
    lines += [
        "",
        "## Four-axis decision update",
        "",
        "- **Scenarios:** targeted H15-prefix states exposed some opportunity, but it is case-concentrated (case 5 only). Case 4, despite being a missed physical>=3 source case, did not yield a robust positive under the strict terminal-stable rule.",
        "- **Reward/terminal:** positives required terminal-stable realised physical improvement; raw objective-min and simple terminal/ridge repairs remain locked out. Do not reinterpret this as proof of terminal-value correctness.",
        "- **Training:** no new training occurred. Because the frozen gate failed, do not train/refit a selector from this bank. Sparse labels here block this supervised selector only, not all future value/modeling interventions.",
        "- **Comparisons/timing:** this smoke is not validation and not a speed claim. Single-run branch timing, if present in raw traces, is descriptive only; a later repeated paired timing/stability block is needed before claiming a control/compute tradeoff.",
        "",
        "## Decision and next bounded action",
        "",
        f"- Train/refit now: `{report['decision']['train_or_refit_now']}`.",
        f"- Next action: `{report['decision']['next_action']}`.",
        "- Rationale: repeating unchanged label-density sweeps is lower value than checking whether the two case-5 positives are stable, timed, and non-artifactual; if they are not, pivot to terminal/modeling or scenario-opportunity diagnosis before training.",
        "",
        f"Backup request: `{report['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for p in (SMOKE_RAW, SMOKE_DONE, SMOKE_SUMMARY, DRYRUN_DONE, PREPARE_PROTOCOL, AMENDMENT):
        if not p.exists():
            raise RuntimeError(f"required artifact missing: {rel(p)}")
    OUT.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc)
    done = completed_ok(SMOKE_DONE)
    dry = completed_ok(DRYRUN_DONE)
    raw = read_json(SMOKE_RAW)
    summary_text = SMOKE_SUMMARY.read_text(encoding="utf-8")
    analysis = raw.get("analysis") or {}
    targets = raw.get("targets") or []
    state_rows, row_sources = choose_state_rows(raw, summary_text, targets)
    positive_rows = [r for r in state_rows if row_is_positive(r)]
    positive_cases = sorted(set(int(r["case"]) for r in positive_rows if r.get("case") is not None))
    target_roles = [str((t.get("case_role") or t.get("role") or "")) for t in targets]
    control_rows = [r for r in state_rows if "control" in str(r.get("case_role", "")).lower()]
    control_positive_rows = [r for r in control_rows if row_is_positive(r)]
    headline_done = done.get("headline") or {}
    robust_count = as_int(headline_done.get("robust_positive_state_count"), len(positive_rows))
    robust_cases = headline_done.get("robust_positive_cases") if isinstance(headline_done.get("robust_positive_cases"), list) else positive_cases
    negative_count = as_int(headline_done.get("negative_or_neutral_state_count"), max(0, len(state_rows) - int(robust_count or 0)))
    blocking_count = as_int(headline_done.get("blocking_artifact_count"), as_int(analysis.get("blocking_artifact_count"), 0))
    pass_gate = bool(headline_done.get("smoke_pass_to_selector_or_value_refit_design"))
    control_total = len(control_rows)
    control_pos = len(control_positive_rows)
    control_fpr = None if control_total == 0 else control_pos / float(control_total)
    target_count = len(targets) or len(state_rows)
    gate_components = {
        "robust_positive_states_ge_2": {"required": ">=2", "observed": int(robust_count or 0), "pass": int(robust_count or 0) >= 2},
        "robust_positive_cases_ge_2": {"required": ">=2", "observed": len(robust_cases or []), "pass": len(robust_cases or []) >= 2},
        "retained_negative_or_control_states_ge_4": {"required": ">=4", "observed": int(negative_count or 0), "pass": int(negative_count or 0) >= 4},
        "control_false_positive_rate_le_0p25": {"required": "<=0.25", "observed": None if control_fpr is None else round(control_fpr, 6), "pass": (control_fpr is not None and control_fpr <= 0.25)},
        "blocking_artifacts_eq_0": {"required": "0", "observed": int(blocking_count or 0), "pass": int(blocking_count or 0) == 0},
    }
    # If the completed marker explicitly declares the gate, keep it authoritative.
    computed_gate = all(bool(v["pass"]) for v in gate_components.values())
    if pass_gate != computed_gate:
        gate_components["completed_marker_gate_authoritative"] = {"required": "match frozen runner", "observed": pass_gate, "pass": pass_gate}
    if pass_gate:
        next_action = "after backup, freeze a compact IMPROVED selector/value-refit smoke with strong fixed-H/Pareto baselines; still no validation64/sealed-test use"
        train_or_refit_now = False
    elif int(robust_count or 0) > 0:
        next_action = "after backup, freeze a small repeated positive-state stability/timing diagnostic for the two case-5 robust-positive H15-prefix states versus H15 reference; no selector/refit unless stability and measured timing/control tradeoff survive"
        train_or_refit_now = False
    else:
        next_action = "do not refit from this bank; pivot to terminal/modeling or sparse-opportunity scenario diagnosis rather than another unchanged label-density sweep"
        train_or_refit_now = False
    token_info = best_effort_token_total()
    episode_count = as_int(done.get("episodes"), as_int((raw.get("budget_actual") or {}).get("episodes"), len(raw.get("branch_episodes") or [])))
    control_steps = as_int(done.get("control_steps"), as_int((raw.get("budget_actual") or {}).get("control_steps"), 0))
    report = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "server_api_total_tokens_best_effort": token_info,
        "classification": "development_no_simulation_v1e_smoke_postdiagnostic_not_validation_not_final_test",
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
        "dryrun_completed": rel(DRYRUN_DONE),
        "headline": {
            "episodes": episode_count,
            "control_steps": control_steps,
            "target_count": target_count,
            "robust_positive_state_count": int(robust_count or 0),
            "robust_positive_cases": robust_cases,
            "negative_or_neutral_state_count": int(negative_count or 0),
            "control_state_count": control_total,
            "control_positive_state_count": control_pos,
            "control_false_positive_rate": None if control_fpr is None else round(control_fpr, 6),
            "blocking_artifact_count": int(blocking_count or 0),
            "smoke_pass_to_selector_or_value_refit_design": pass_gate,
            "schema_repair": raw.get("schema_repair") or done.get("schema_repair"),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "gate_components": gate_components,
        "state_row_sources_considered": row_sources,
        "state_rows": state_rows,
        "positive_state_rows": positive_rows,
        "positive_cases_by_role": counter(r.get("case_role") for r in positive_rows),
        "target_roles": counter(target_roles),
        "analysis_top_level_keys": sorted(str(k) for k in analysis.keys()),
        "completed_headline": headline_done,
        "decision": {"train_or_refit_now": train_or_refit_now, "next_action": next_action, "do_not_train_from_v1e_labels_because_gate_failed": not pass_gate},
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
            "v0 parent failure remains implementation/schema failure only",
            "selector/value-refit gate failed because positives did not span enough cases",
        ],
    }
    write_json(OUT / "raw.json", report)
    write_summary(report)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(BACKUP_REQUEST, {
        "requested_utc": created.isoformat(),
        "reason": "backup v1e smoke postdiagnostic, v0b smoke outputs, docs/state before any repeated positive-state timing/stability diagnostic or further training/refit decision",
        "backup_required_before_more_simulations": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(SOURCE), rel(SMOKE_DIR), rel(BACKUP_REQUEST)],
    })
    docs_block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1e smoke postdiagnostic v0

UTC: {created.isoformat()}. No-simulation postdiagnostic of the completed v1e v0b schema-repair smoke. The smoke produced {int(robust_count or 0)} robust-positive states across cases {robust_cases}, with {int(negative_count or 0)} negative/neutral states and gate={pass_gate}; positives are concentrated in case 5, so do not train/refit a selector from this bank. Next bounded action after backup: {next_action}. No validation64 bank or sealed test access; no training/refit. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    append_docs(docs_block)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE, BACKUP_REQUEST, SMOKE_RAW, SMOKE_DONE, SMOKE_SUMMARY, DRYRUN_DONE, PREPARE_PROTOCOL, AMENDMENT]
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
        "robust_positive_state_count": int(robust_count or 0),
        "robust_positive_cases": robust_cases,
        "negative_or_neutral_state_count": int(negative_count or 0),
        "control_positive_state_count": control_pos,
        "control_state_count": control_total,
        "smoke_pass_to_selector_or_value_refit_design": pass_gate,
        "train_or_refit_now": train_or_refit_now,
        "next_action": next_action,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQUEST),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
