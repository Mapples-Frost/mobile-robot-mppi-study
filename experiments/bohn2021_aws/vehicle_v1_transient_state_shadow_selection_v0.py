#!/usr/bin/env python3
"""Vehicle V1 transient-state shadow selection v0.

No-simulation/no-training diagnostic that implements the state-selection half of
research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.*.

It reads only the already-saved H15 reference traces and fresh-probe bank
metadata.  It intentionally does not read non-H15 branch outcomes when selecting
states, does not construct environments, performs no rollouts, no training/refit,
opens no historical validation64 bank, and never accesses sealed final test.

The output is a frozen target schedule for the next rollout runner: up to eight
high-transient H15-prefix states plus four low-transient controls, max one high
and one low state per fresh case where feasible, with branch horizons
[10, 15, 30, 35].
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
FRESH_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928"
FRESH_RAW = FRESH_DIR / "raw.json"
FRESH_COMPLETED = FRESH_DIR / "completed.json"
BANK_PATH = FRESH_DIR / "bank/vehicle_v1_fresh_continuation_label_probe_v0_bank.json"
POSTDIAG_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_20260928T1510Z/completed.json"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-v1-transient-state-shadow-selection-v0-20260928T1515Z"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 30, 35]
MAX_SELECTED_STATES = 12
MAX_HIGH = 8
MAX_LOW = 4
DEDUP_WINDOW = 3


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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


def verify_completed(path: Path) -> Mapping[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("completed marker did not pass: " + rel(path))
    if done.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError("historical validation64 flag invalid in " + rel(path))
    if done.get("sealed_test_accessed") not in (False, None):
        raise ContractError("sealed test flag invalid in " + rel(path))
    return done


def verify_inputs() -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    for p in (FRESH_RAW, FRESH_COMPLETED, BANK_PATH, POSTDIAG_COMPLETED, PROTOCOL_JSON, PROTOCOL_MD):
        if not p.exists():
            raise ContractError("missing required input: " + rel(p))
    fresh_done = verify_completed(FRESH_COMPLETED)
    post_done = verify_completed(POSTDIAG_COMPLETED)
    if post_done.get("headline", {}).get("fresh_positive_states") != 0:
        raise ContractError("unexpected postdiagnostic headline; fresh-positive count changed")
    fresh = read_json(FRESH_RAW)
    bank = read_json(BANK_PATH)
    protocol = read_json(PROTOCOL_JSON)
    if fresh.get("historical_validation64_bank_opened") is not False or fresh.get("sealed_test_accessed") is not False:
        raise ContractError("fresh raw access flags invalid")
    if protocol.get("protocol_id") != "vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928":
        raise ContractError("unexpected transient protocol id")
    design = protocol.get("rollout_design_after_backup", {})
    if int(design.get("prefix_horizon", -1)) != PREFIX_H or [int(x) for x in design.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("transient protocol horizon design mismatch")
    if int(design.get("max_selected_states", -1)) != MAX_SELECTED_STATES:
        raise ContractError("transient protocol selected-state budget mismatch")
    return fresh, bank, protocol


def h15_summary_paths(fresh: Mapping[str, Any]) -> Dict[Tuple[int, int], Path]:
    out: Dict[Tuple[int, int], Path] = {}
    for row in fresh.get("episodes", []):
        if int(row.get("branch_horizon", -1)) == PREFIX_H:
            key = (int(row["case"]), int(row["branch_step"]))
            out[key] = ROOT / str(row["path"]) / "trace.json"
    return out


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def finite_values(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs if math.isfinite(float(x))]


def quantile(xs: Sequence[float], q: float, default: float = 1.0) -> float:
    vals = sorted(finite_values(xs))
    if not vals:
        return default
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def obs_ref_xy(row: Mapping[str, Any]) -> List[Tuple[float, float]]:
    obs = row.get("observation") or []
    pairs: List[Tuple[float, float]] = []
    # In this vehicle trace layout, indices after the first three ego/reference-error
    # features appear as x/y/r obstacle triples.  Use all finite pairs with a
    # following positive radius as a robust proximity proxy; if layout changes,
    # the score gracefully falls back to tracking/heading error.
    for i in range(3, max(3, len(obs) - 2), 3):
        x = safe_float(obs[i], float("nan"))
        y = safe_float(obs[i + 1], float("nan"))
        r = safe_float(obs[i + 2], float("nan"))
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(r) and r > 0:
            pairs.append((x, y))
    return pairs


def obstacle_clearance_proxy(row: Mapping[str, Any]) -> float:
    prev = row.get("previous_state") or {}
    px = safe_float(prev.get("x"), float("nan"))
    py = safe_float(prev.get("y"), float("nan"))
    if not (math.isfinite(px) and math.isfinite(py)):
        return float("nan")
    obs = row.get("observation") or []
    vals: List[float] = []
    # Prefer object summaries if present in case metadata is impossible here; use
    # observation triples as a local proxy.  Radius is subtracted when available.
    for i in range(3, max(3, len(obs) - 2), 3):
        ox = safe_float(obs[i], float("nan"))
        oy = safe_float(obs[i + 1], float("nan"))
        rr = safe_float(obs[i + 2], 0.0)
        if math.isfinite(ox) and math.isfinite(oy):
            vals.append(math.hypot(px - ox, py - oy) - max(0.0, rr))
    return min(vals) if vals else float("nan")


def row_features(trace: Sequence[Mapping[str, Any]], idx: int) -> Dict[str, Any]:
    row = trace[idx]
    obs = row.get("observation") or []
    tracking = math.hypot(safe_float(obs[0], 0.0), safe_float(obs[1], 0.0)) if len(obs) >= 2 else 0.0
    heading = abs(safe_float(obs[2], 0.0)) if len(obs) >= 3 else 0.0
    perf = safe_float(row.get("performance"), 0.0)
    inp = row.get("input") or {}
    us = inp.get("u_s") or [0.0]
    uo = inp.get("u_omega") or [0.0]
    speed_cmd = abs(safe_float(us[0], 0.0)) if us else 0.0
    omega_cmd = abs(safe_float(uo[0], 0.0)) if uo else 0.0
    clearance = obstacle_clearance_proxy(row)
    if idx > 0:
        prev_obs = trace[idx - 1].get("observation") or []
        dtrack = tracking - (math.hypot(safe_float(prev_obs[0], 0.0), safe_float(prev_obs[1], 0.0)) if len(prev_obs) >= 2 else 0.0)
        dheading = heading - (abs(safe_float(prev_obs[2], 0.0)) if len(prev_obs) >= 3 else 0.0)
    else:
        dtrack = 0.0
        dheading = 0.0
    prev_state = row.get("previous_state") or {}
    return {
        "step": int(idx),
        "tracking_error_norm": float(tracking),
        "heading_error_abs": float(heading),
        "performance_step_cost": float(perf),
        "speed_cmd_abs": float(speed_cmd),
        "omega_cmd_abs": float(omega_cmd),
        "delta_tracking_error": float(dtrack),
        "delta_heading_error": float(dheading),
        "obstacle_clearance_proxy": float(clearance) if math.isfinite(clearance) else None,
        "previous_state": {
            "x": safe_float(prev_state.get("x"), float("nan")),
            "y": safe_float(prev_state.get("y"), float("nan")),
            "theta": safe_float(prev_state.get("theta"), float("nan")),
        },
    }


def add_scores(rows: List[Dict[str, Any]]) -> None:
    tr_scale = max(quantile([r["tracking_error_norm"] for r in rows], 0.90, 1.0), 1e-9)
    hd_scale = max(quantile([r["heading_error_abs"] for r in rows], 0.90, 1.0), 1e-9)
    perf_scale = max(quantile([r["performance_step_cost"] for r in rows], 0.90, 1.0), 1e-9)
    omg_scale = max(quantile([r["omega_cmd_abs"] for r in rows], 0.90, 1.0), 1e-9)
    close_vals = [r["obstacle_clearance_proxy"] for r in rows if r.get("obstacle_clearance_proxy") is not None]
    close_ref = quantile(close_vals, 0.25, 1.0)
    close_scale = max(abs(quantile(close_vals, 0.75, close_ref) - close_ref), 1e-6)
    for r in rows:
        clearance = r.get("obstacle_clearance_proxy")
        proximity = 0.0 if clearance is None else max(0.0, (close_ref - float(clearance)) / close_scale)
        transient = (
            1.00 * r["tracking_error_norm"] / tr_scale
            + 0.75 * r["heading_error_abs"] / hd_scale
            + 0.50 * r["performance_step_cost"] / perf_scale
            + 0.25 * r["omega_cmd_abs"] / omg_scale
            + 0.35 * max(0.0, r["delta_tracking_error"] / tr_scale)
            + 0.20 * max(0.0, r["delta_heading_error"] / hd_scale)
            + 0.50 * proximity
        )
        # Avoid branch states so close to termination that continuations are uninformative.
        r["transient_score"] = float(transient)
        r["score_components"] = {
            "tracking_scaled": r["tracking_error_norm"] / tr_scale,
            "heading_scaled": r["heading_error_abs"] / hd_scale,
            "performance_scaled": r["performance_step_cost"] / perf_scale,
            "omega_scaled": r["omega_cmd_abs"] / omg_scale,
            "proximity_scaled": proximity,
        }


def dedup_pick(candidates: Sequence[Dict[str, Any]], used_steps: Sequence[int], descending: bool) -> Optional[Dict[str, Any]]:
    ordered = sorted(candidates, key=lambda r: (r["transient_score"], -r["step"]), reverse=descending)
    for r in ordered:
        if all(abs(int(r["step"]) - int(u)) > DEDUP_WINDOW for u in used_steps):
            return r
    return None


def build_candidates(fresh: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = h15_summary_paths(fresh)
    selected: List[Dict[str, Any]] = []
    per_case: List[Dict[str, Any]] = []
    for (case_id, nominal_step), trace_path in sorted(paths.items()):
        trace = read_json(trace_path)
        # Candidate interval: after early warmup and before the final five steps.
        n = len(trace)
        rows = [row_features(trace, i) for i in range(5, max(5, n - 5))]
        if not rows:
            rows = [row_features(trace, i) for i in range(0, n)]
        add_scores(rows)
        high = dedup_pick(rows, [], descending=True)
        low = dedup_pick(rows, [high["step"]] if high else [], descending=False)
        entry = {
            "case": int(case_id),
            "nominal_fresh_probe_branch_step": int(nominal_step),
            "h15_trace": rel(trace_path),
            "h15_trace_sha256": sha256(trace_path),
            "trace_steps": int(n),
            "high_candidate": high,
            "low_candidate": low,
            "score_summary": {
                "min": min(r["transient_score"] for r in rows),
                "median": quantile([r["transient_score"] for r in rows], 0.50),
                "max": max(r["transient_score"] for r in rows),
            },
        }
        per_case.append(entry)
        if high:
            x = dict(high)
            x.update({"case": int(case_id), "kind": "high_transient", "h15_trace": rel(trace_path), "nominal_fresh_probe_branch_step": int(nominal_step)})
            selected.append(x)
        if low:
            x = dict(low)
            x.update({"case": int(case_id), "kind": "low_transient_control", "h15_trace": rel(trace_path), "nominal_fresh_probe_branch_step": int(nominal_step)})
            selected.append(x)
    highs = sorted([s for s in selected if s["kind"] == "high_transient"], key=lambda r: (r["transient_score"], -r["case"]), reverse=True)[:MAX_HIGH]
    high_cases = {int(r["case"]) for r in highs}
    # Prefer controls from cases not already represented among highs when possible.
    lows_all = [s for s in selected if s["kind"] == "low_transient_control"]
    lows_ranked = sorted(lows_all, key=lambda r: ((int(r["case"]) in high_cases), r["transient_score"], r["case"]))
    lows = lows_ranked[:MAX_LOW]
    targets = highs + lows
    targets = sorted(targets, key=lambda r: (0 if r["kind"] == "high_transient" else 1, r["case"], r["step"]))
    for i, t in enumerate(targets):
        t["target_index"] = int(i)
        t["branch_horizons"] = list(BRANCH_HORIZONS)
        t["prefix_horizon"] = PREFIX_H
        t["rollout_episodes"] = len(BRANCH_HORIZONS)
    return targets, {"per_case_candidates": per_case, "available_h15_references": len(paths)}


def write_backup_request(raw: Mapping[str, Any]) -> str:
    path = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_SHADOW_SELECTION_V0_20260928T1515Z.json"
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup no-simulation transient-state target selection and source before writing/running any rollout runner",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle V1 transient-state shadow selection v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "No-simulation/no-training target-selection diagnostic for the frozen transient-state continuation probe. It used H15 reference traces only and ignored non-H15 branch outcomes for state selection.",
        "",
        "## Access and budget",
        "",
        "- New rollouts/control steps/training episodes/gradient steps: `0 / 0 / 0 / 0`.",
        "- historical_validation64_bank_opened: `False`; sealed_test_accessed: `False`.",
        f"- H15 reference trace entries available: `{raw['selection_diagnostics']['available_h15_references']}`.",
        f"- Selected targets: `{len(raw['selected_targets'])}`; future rollout episodes if executed: `{raw['future_rollout_budget']['episodes']}` / control-step upper bound `{raw['future_rollout_budget']['control_step_upper_bound']}`.",
        "",
        "## Selected targets",
        "",
        "| target | kind | case | step | score | tracking | heading | perf step | clearance proxy | branch horizons |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in raw["selected_targets"]:
        lines.append("| %d | `%s` | %d | %d | %.6g | %.6g | %.6g | %.6g | %s | `%s` |" % (
            int(t["target_index"]), t["kind"], int(t["case"]), int(t["step"]), float(t["transient_score"]),
            float(t["tracking_error_norm"]), float(t["heading_error_abs"]), float(t["performance_step_cost"]),
            "NA" if t.get("obstacle_clearance_proxy") is None else "%.6g" % float(t["obstacle_clearance_proxy"]),
            t["branch_horizons"],
        ))
    lines += [
        "",
        "## Decision",
        "",
        "This no-simulation selection confirms that the frozen transient-state probe has concrete H15-only branch targets without inspecting non-H15 outcomes. It does not establish adaptive opportunity; the next evidence-producing action remains the bounded continuation rollout after verified backup and rollout-runner source freeze.",
        "",
        f"Backup request before simulations/source-dependent rollout work: `{raw['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 transient-state shadow selection v0\n\n"
        f"UTC: {raw['created_utc']}. No-simulation target-selection diagnostic selected {len(raw['selected_targets'])} H15-only transient/control branch states "
        f"for the frozen continuation probe; future rollout budget {raw['future_rollout_budget']['episodes']} episodes / {raw['future_rollout_budget']['control_step_upper_bound']} steps. "
        "No validation64/test/training access. Backup remains required before simulations. "
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
        print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json"), "targets": done.get("selected_targets")}, sort_keys=True))
        return 0
    fresh, bank, protocol = verify_inputs()
    targets, diagnostics = build_candidates(fresh)
    if len(targets) < 2:
        raise ContractError("too few transient/control targets selected")
    future_episodes = sum(len(t["branch_horizons"]) for t in targets)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "vehicle_v1_transient_state_shadow_selection_v0_no_simulation_H15_traces_only",
        "classification": "development_no_simulation_target_selection_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD), "full": protocol},
        "fresh_inputs": {"raw": rel(FRESH_RAW), "raw_sha256": sha256(FRESH_RAW), "completed": rel(FRESH_COMPLETED), "completed_sha256": sha256(FRESH_COMPLETED), "bank": rel(BANK_PATH), "bank_sha256": sha256(BANK_PATH), "postdiagnostic_completed": rel(POSTDIAG_COMPLETED), "postdiagnostic_completed_sha256": sha256(POSTDIAG_COMPLETED)},
        "state_selection_access_rule": "H15 reference traces and H15 success/safety metadata only; non-H15 branch outcomes ignored for target selection",
        "selected_targets": targets,
        "selection_diagnostics": diagnostics,
        "future_rollout_budget": {"episodes": future_episodes, "control_step_upper_bound": int(future_episodes * 150), "branch_horizons": BRANCH_HORIZONS, "prefix_horizon": PREFIX_H, "new_training_episodes": 0, "new_gradient_steps": 0},
        "next_action": "after verified backup covering this source/output and v0c repair, freeze/write the rollout runner using these H15-only targets and execute the bounded continuation probe; if fewer than two material positives, proceed to versioned stress-scenario opportunity design rather than retraining on sparse canonical labels",
        "source_hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 transient-state shadow selection v0 state ({raw['created_utc']})\n\n"
        f"No-simulation diagnostic selected {len(targets)} H15-only branch targets; future rollout budget {future_episodes} episodes / {future_episodes * 150} steps. "
        "No validation64/test/training access. Backup required before writing/running rollout runner.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, ROOT / raw["backup_request"], Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD]
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
        "future_episodes": future_episodes,
        "future_control_step_upper_bound": future_episodes * 150,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": raw["backup_request"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
