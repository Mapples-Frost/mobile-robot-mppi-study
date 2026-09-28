#!/usr/bin/env python3
"""Vehicle V1 transient-state shadow selection v0b unique-case repair.

No-simulation/no-training repair of
experiments/bohn2021_aws/vehicle_v1_transient_state_shadow_selection_v0.py.

The first v0 no-simulation target selector correctly avoided non-H15 branch
outcomes, but its H15 reference iterator treated the two H15 arms per fresh case
(two nominal branch-step metadata entries) as independent cases.  The resulting
12-target schedule duplicated the same case/step pairs and would waste rollout
budget, conflicting with the frozen protocol's "max one high and one low state
per case where feasible" rule.

This v0b repair makes exactly the target-selection fix needed before any
simulation: collapse H15 references to one deterministic H15 trace per case
(lowest nominal fresh-probe branch step), then select at most one high-transient
and at most one low-transient control state per case using H15-only trace fields.
It still performs no rollouts, no training/refit, opens no validation64 bank, and
never accesses sealed final test outcomes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_v1_transient_state_shadow_selection_v0 as v0  # noqa:E402

OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
V0_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z/completed.json"
MARKER = "vehicle-v1-transient-state-shadow-selection-v0b-unique-repair-20260928T1520Z"
PREFIX_H = v0.PREFIX_H
BRANCH_HORIZONS = list(v0.BRANCH_HORIZONS)
MAX_SELECTED_STATES = v0.MAX_SELECTED_STATES
MAX_HIGH = v0.MAX_HIGH
MAX_LOW = v0.MAX_LOW
DEDUP_WINDOW = v0.DEDUP_WINDOW


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def unique_h15_reference_paths(fresh: Mapping[str, Any]) -> Dict[int, Tuple[int, Path]]:
    grouped: Dict[int, List[Tuple[int, Path]]] = {}
    for row in fresh.get("episodes", []):
        if int(row.get("branch_horizon", -1)) != PREFIX_H:
            continue
        case_id = int(row["case"])
        branch_step = int(row["branch_step"])
        trace_path = ROOT / str(row["path"]) / "trace.json"
        if not trace_path.exists():
            raise ContractError("missing H15 trace: " + rel(trace_path))
        grouped.setdefault(case_id, []).append((branch_step, trace_path))
    if not grouped:
        raise ContractError("no H15 reference traces found")
    out: Dict[int, Tuple[int, Path]] = {}
    for case_id, rows in sorted(grouped.items()):
        # Deterministic H15-only collapse: choose the lowest nominal H15 branch
        # step, not any non-H15 outcome.  A full duplicate audit is recorded below.
        rows_sorted = sorted(rows, key=lambda x: (int(x[0]), rel(x[1])))
        out[int(case_id)] = (int(rows_sorted[0][0]), rows_sorted[0][1])
    return out


def duplicate_audit() -> Dict[str, Any]:
    audit: Dict[str, Any] = {"v0_completed_present": V0_COMPLETED.exists()}
    if not V0_COMPLETED.exists():
        return audit
    done = read_json(V0_COMPLETED)
    targets = done.get("selected_targets", [])
    keys = [(int(t.get("case", -1)), str(t.get("kind")), int(t.get("step", -1))) for t in targets]
    counts: Dict[str, int] = {}
    for case_id, kind, step in keys:
        k = f"case{case_id:02d}_{kind}_step{step:03d}"
        counts[k] = counts.get(k, 0) + 1
    audit.update({
        "v0_selected_target_count": len(targets),
        "v0_unique_case_kind_step_count": len(counts),
        "v0_duplicate_case_kind_steps": {k: c for k, c in sorted(counts.items()) if c > 1},
        "v0_completed_sha256": sha256(V0_COMPLETED),
    })
    return audit


def dedup_pick(candidates: Sequence[Dict[str, Any]], used_steps: Sequence[int], descending: bool) -> Optional[Dict[str, Any]]:
    ordered = sorted(candidates, key=lambda r: (float(r["transient_score"]), -int(r["step"])), reverse=descending)
    for r in ordered:
        if all(abs(int(r["step"]) - int(u)) > DEDUP_WINDOW for u in used_steps):
            return r
    return None


def per_case_candidates(fresh: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    refs = unique_h15_reference_paths(fresh)
    per_case: List[Dict[str, Any]] = []
    high_candidates: List[Dict[str, Any]] = []
    low_candidates: List[Dict[str, Any]] = []
    for case_id, (nominal_step, trace_path) in sorted(refs.items()):
        trace = read_json(trace_path)
        n = len(trace)
        rows = [v0.row_features(trace, i) for i in range(5, max(5, n - 5))]
        if not rows:
            rows = [v0.row_features(trace, i) for i in range(0, n)]
        v0.add_scores(rows)
        high = dedup_pick(rows, [], descending=True)
        low = dedup_pick(rows, [int(high["step"])] if high else [], descending=False)
        entry = {
            "case": int(case_id),
            "chosen_nominal_h15_branch_step": int(nominal_step),
            "h15_trace": rel(trace_path),
            "h15_trace_sha256": sha256(trace_path),
            "trace_steps": int(n),
            "high_candidate": high,
            "low_candidate": low,
            "score_summary": {
                "min": min(float(r["transient_score"]) for r in rows),
                "median": v0.quantile([float(r["transient_score"]) for r in rows], 0.50),
                "max": max(float(r["transient_score"]) for r in rows),
            },
        }
        per_case.append(entry)
        if high is not None:
            h = dict(high)
            h.update({
                "case": int(case_id),
                "kind": "high_transient",
                "h15_trace": rel(trace_path),
                "chosen_nominal_h15_branch_step": int(nominal_step),
            })
            high_candidates.append(h)
        if low is not None:
            l = dict(low)
            l.update({
                "case": int(case_id),
                "kind": "low_transient_control",
                "h15_trace": rel(trace_path),
                "chosen_nominal_h15_branch_step": int(nominal_step),
            })
            low_candidates.append(l)
    highs = sorted(high_candidates, key=lambda r: (float(r["transient_score"]), -int(r["case"])), reverse=True)[:MAX_HIGH]
    high_cases = {int(r["case"]) for r in highs}
    lows_non_high = sorted([r for r in low_candidates if int(r["case"]) not in high_cases], key=lambda r: (float(r["transient_score"]), int(r["case"])))
    lows_high = sorted([r for r in low_candidates if int(r["case"]) in high_cases], key=lambda r: (float(r["transient_score"]), int(r["case"])))
    lows = (lows_non_high + lows_high)[:MAX_LOW]
    targets = highs + lows
    if len(targets) > MAX_SELECTED_STATES:
        raise ContractError("selected too many targets")
    unique_case_kind = {(int(t["case"]), str(t["kind"])) for t in targets}
    if len(unique_case_kind) != len(targets):
        raise ContractError("v0b target selection still duplicates case/kind")
    unique_case_kind_step = {(int(t["case"]), str(t["kind"]), int(t["step"])) for t in targets}
    if len(unique_case_kind_step) != len(targets):
        raise ContractError("v0b target selection still duplicates case/kind/step")
    targets = sorted(targets, key=lambda r: (0 if r["kind"] == "high_transient" else 1, int(r["case"]), int(r["step"])))
    for i, t in enumerate(targets):
        t["target_index"] = int(i)
        t["branch_horizons"] = list(BRANCH_HORIZONS)
        t["prefix_horizon"] = PREFIX_H
        t["rollout_episodes"] = len(BRANCH_HORIZONS)
    return targets, {
        "per_case_candidates": per_case,
        "unique_h15_reference_cases": len(refs),
        "h15_reference_collapse_rule": "one H15 trace per case, chosen by lowest nominal fresh-probe H15 branch_step; non-H15 outcomes ignored",
        "available_high_candidates": len(high_candidates),
        "available_low_candidates": len(low_candidates),
        "selected_high_count": sum(1 for t in targets if t["kind"] == "high_transient"),
        "selected_low_count": sum(1 for t in targets if t["kind"] == "low_transient_control"),
    }


def write_backup_request(raw: Mapping[str, Any]) -> str:
    path = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_SHADOW_SELECTION_V0B_UNIQUE_REPAIR_20260928T1520Z.json"
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup no-simulation unique-case transient target repair and source before writing/running any continuation rollout runner",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(v0.PROTOCOL_JSON), rel(v0.PROTOCOL_MD), rel(V0_COMPLETED)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle V1 transient-state shadow selection v0b unique-case repair",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "No-simulation/no-training repair of the v0 target-selection schedule. v0 duplicated case/step targets because it treated two H15 arms per case as separate H15 references. v0b collapses to one H15 reference trace per case before selecting high/low transient states.",
        "",
        "## Access and budget",
        "",
        "- New rollouts/control steps/training episodes/gradient steps: `0 / 0 / 0 / 0`.",
        "- historical_validation64_bank_opened: `False`; sealed_test_accessed: `False`.",
        f"- Unique H15 reference cases: `{raw['selection_diagnostics']['unique_h15_reference_cases']}`.",
        f"- Selected targets: `{len(raw['selected_targets'])}`; future rollout episodes if executed: `{raw['future_rollout_budget']['episodes']}` / control-step upper bound `{raw['future_rollout_budget']['control_step_upper_bound']}`.",
        "",
        "## v0 duplicate audit",
        "",
        f"- v0 selected targets: `{raw['duplicate_audit'].get('v0_selected_target_count')}`; unique case/kind/step: `{raw['duplicate_audit'].get('v0_unique_case_kind_step_count')}`.",
        f"- duplicated case/kind/steps: `{raw['duplicate_audit'].get('v0_duplicate_case_kind_steps')}`.",
        "",
        "## v0b selected targets",
        "",
        "| target | kind | case | step | score | tracking | heading | perf step | clearance proxy | chosen H15 nominal branch | branch horizons |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in raw["selected_targets"]:
        lines.append("| %d | `%s` | %d | %d | %.6g | %.6g | %.6g | %.6g | %s | %d | `%s` |" % (
            int(t["target_index"]), str(t["kind"]), int(t["case"]), int(t["step"]), float(t["transient_score"]),
            float(t["tracking_error_norm"]), float(t["heading_error_abs"]), float(t["performance_step_cost"]),
            "NA" if t.get("obstacle_clearance_proxy") is None else "%.6g" % float(t["obstacle_clearance_proxy"]),
            int(t["chosen_nominal_h15_branch_step"]), t["branch_horizons"],
        ))
    lines += [
        "",
        "## Decision",
        "",
        "The duplicate-schedule implementation defect is repaired before any rollout budget was spent. This remains only a target-selection artifact; adaptive opportunity is still unknown until the bounded continuation rollout is executed after verified backup.",
        "",
        f"Backup request before simulations/source-dependent rollout work: `{raw['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 transient-state shadow selection v0b unique repair\n\n"
        f"UTC: {raw['created_utc']}. No-simulation repair fixed the v0 duplicated transient target schedule by collapsing to one H15 reference trace per case before target selection. "
        f"Selected {len(raw['selected_targets'])} unique case/kind targets; future rollout budget {raw['future_rollout_budget']['episodes']} episodes / {raw['future_rollout_budget']['control_step_upper_bound']} steps. "
        "No validation64/test/training access. Verified backup is required before any continuation rollout. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json"), "selected_target_count": done.get("selected_target_count")}, sort_keys=True))
        return 0
    fresh, bank, protocol = v0.verify_inputs()
    targets, diagnostics = per_case_candidates(fresh)
    if len(targets) < 2:
        raise ContractError("too few v0b targets selected")
    future_episodes = sum(len(t["branch_horizons"]) for t in targets)
    if future_episodes != 48:
        raise ContractError("unexpected future rollout episode count after unique repair: %d" % future_episodes)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "vehicle_v1_transient_state_shadow_selection_v0b_unique_case_repair_no_simulation_H15_traces_only",
        "classification": "development_no_simulation_target_selection_repair_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "protocol": {"json": rel(v0.PROTOCOL_JSON), "json_sha256": sha256(v0.PROTOCOL_JSON), "md": rel(v0.PROTOCOL_MD), "md_sha256": sha256(v0.PROTOCOL_MD), "full": protocol},
        "fresh_inputs": {"raw": rel(v0.FRESH_RAW), "raw_sha256": sha256(v0.FRESH_RAW), "completed": rel(v0.FRESH_COMPLETED), "completed_sha256": sha256(v0.FRESH_COMPLETED), "bank": rel(v0.BANK_PATH), "bank_sha256": sha256(v0.BANK_PATH), "postdiagnostic_completed": rel(v0.POSTDIAG_COMPLETED), "postdiagnostic_completed_sha256": sha256(v0.POSTDIAG_COMPLETED)},
        "repair_lineage": {"v0_source": rel(Path(v0.__file__).resolve()), "v0_source_sha256": sha256(Path(v0.__file__).resolve()), "v0_completed": rel(V0_COMPLETED) if V0_COMPLETED.exists() else None, "one_variable_change": "collapse H15 references to one trace per case before high/low target selection"},
        "duplicate_audit": duplicate_audit(),
        "state_selection_access_rule": "H15 reference traces only; one H15 trace per case chosen by lowest nominal H15 branch step; non-H15 branch outcomes ignored for target selection",
        "selected_targets": targets,
        "selection_diagnostics": diagnostics,
        "future_rollout_budget": {"episodes": future_episodes, "control_step_upper_bound": int(future_episodes * 150), "branch_horizons": BRANCH_HORIZONS, "prefix_horizon": PREFIX_H, "new_training_episodes": 0, "new_gradient_steps": 0},
        "next_action": "after verified backup covering v0/v0b no-simulation target outputs and this source, write/freeze the continuation rollout runner using these unique targets and execute the bounded probe; if fewer than two material positives, proceed to versioned stress-scenario opportunity design rather than retraining on sparse canonical labels",
        "source_hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()), rel(Path(v0.__file__).resolve()): sha256(Path(v0.__file__).resolve())},
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 transient-state shadow selection v0b unique repair state ({raw['created_utc']})\n\n"
        f"No-simulation repair selected {len(targets)} unique case/kind H15-only branch targets; future rollout budget {future_episodes} episodes / {future_episodes * 150} steps. "
        "No validation64/test/training access. Backup required before writing/running rollout runner.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, ROOT / raw["backup_request"], Path(__file__).resolve(), Path(v0.__file__).resolve(), v0.PROTOCOL_JSON, v0.PROTOCOL_MD]
    completed = {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "selected_target_count": len(targets),
        "selected_unique_case_kind_count": len({(int(t["case"]), str(t["kind"])) for t in targets}),
        "selected_targets": [{"target_index": int(t["target_index"]), "kind": t["kind"], "case": int(t["case"]), "step": int(t["step"]), "transient_score": float(t["transient_score"])} for t in targets],
        "future_rollout_budget": raw["future_rollout_budget"],
        "backup_request": raw["backup_request"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT_DIR / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "selected_target_count": len(targets),
        "selected_unique_case_kind_count": completed["selected_unique_case_kind_count"],
        "future_episodes": future_episodes,
        "future_control_step_upper_bound": future_episodes * 150,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": raw["backup_request"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
