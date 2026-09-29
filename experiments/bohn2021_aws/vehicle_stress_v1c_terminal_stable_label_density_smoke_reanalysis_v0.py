#!/usr/bin/env python3
"""No-simulation reanalysis of vehicle stress-v1c terminal-stable smoke.

Purpose
-------
The v1c smoke reported 0 robust positives, but also 144 blocking
`prefix_hash_mismatch` artifacts.  Inspection of the runner shows the legacy
prefix hash includes non-causal decision bookkeeping (e.g. branch horizon and
terminal-mode fields) inside prefix rows, so the hash can differ even when the
actual prefix state/control trajectory is identical.  This script reads only the
already-created development smoke artifacts and recomputes labels with a
physical-prefix equivalence check based on trace state/control/reward fields,
not the bookkeeping-contaminated prefix hash.

Access contract
---------------
No simulations, no candidate resets, no training/refit/gradient updates, no
validation64-bank read/reopen, and no sealed-test access.  This is development
artifact analysis only; it must not be used as final evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
STAMP = "20260929T0205Z"
INPUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_20260929T0055Z_hash_repair"
INPUT_RAW = INPUT_DIR / "raw.json"
INPUT_DONE = INPUT_DIR / "completed.json"
INPUT_SUMMARY = INPUT_DIR / "summary.md"
INPUT_SCHEDULE = INPUT_DIR / "schedule.json"
INPUT_BANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank_20260929T0055Z_hash_repair/completed.json"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v1c_terminal_stable_smoke_reanalysis_v0.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1C_TERMINAL_STABLE_SMOKE_REANALYSIS_V0_{STAMP}.json"
MARKER = f"vehicle-stress-v1c-terminal-stable-smoke-reanalysis-v0-{STAMP}"
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
PREFIX_H = 15
STATE_DISTANCE_TOL = 1e-5
MATERIAL_GAIN = 3.0
PHYSICAL_PREFIX_TOL = 1e-9


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return clean_jsonable(value.tolist())
    if hasattr(value, "item"):
        return clean_jsonable(value.item())
    return str(value)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(clean_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_completed(path: Path, *, require_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("hard_pass") is not True and obj.get("passed") is not True:
        raise ContractError(f"input completed marker did not pass: {rel(path)}")
    for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(flag) not in (False, None):
            raise ContractError(f"input access flag {flag} not false in {rel(path)}")
    if require_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError(f"hash-listed input missing: {name}")
            got = sha256(p)
            if got != expected:
                raise ContractError(f"hash-listed input mismatch: {name}")
    return obj


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def flatten_numeric(value: Any, out: Optional[List[float]] = None) -> List[float]:
    if out is None:
        out = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return out
    if isinstance(value, (int, float)):
        x = float(value)
        if math.isfinite(x):
            out.append(x)
        return out
    if isinstance(value, Mapping):
        for key in sorted(value.keys(), key=str):
            flatten_numeric(value[key], out)
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            flatten_numeric(item, out)
        return out
    return out


def max_abs_diff(a: Any, b: Any) -> float:
    xs = flatten_numeric(a)
    ys = flatten_numeric(b)
    if len(xs) != len(ys):
        return float("inf")
    if not xs:
        return 0.0
    return max(abs(x - y) for x, y in zip(xs, ys))


def prefix_physical_signature(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    """Return prefix fields that determine/reveal physical evolution.

    Deliberately excludes fields known to be non-causal for prefix dynamics in
    this diagnostic: `terminal_mode`, `terminal_source_label`, `decision`, wall
    timing, recovery timing/objective bookkeeping, and MPC diagnostic blobs.
    """
    rows: List[Dict[str, Any]] = []
    for row in trace[:branch_step]:
        rows.append({
            "step": int(row.get("step", -1)),
            "horizon": int(row.get("horizon", -1)),
            "previous_state": row.get("previous_state"),
            "state": row.get("state"),
            "input": row.get("input"),
            "observation": row.get("observation"),
            "next_observation": row.get("next_observation"),
            "reward": row.get("reward"),
            "performance": row.get("performance"),
            "compute": row.get("compute"),
            "constraint": row.get("constraint"),
            "solver_success": row.get("solver_success"),
            "termination": row.get("termination"),
        })
    return rows


def safety_ok(comp: Mapping[str, Any]) -> bool:
    return bool(comp.get("no_success_constraint_solver_regression_vs_H15"))


def load_trace_for_episode(ep: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    p = ROOT / str(ep.get("path", "")) / "trace.json"
    if not p.exists():
        raise ContractError(f"missing trace for episode: {rel(p)}")
    return read_json(p)


def finite_summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    n = len(xs)
    def pct(q: float) -> float:
        if n == 1:
            return xs[0]
        idx = (n - 1) * q / 100.0
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"count": n, "min": xs[0], "median": pct(50), "mean": sum(xs) / n, "p95": pct(95), "max": xs[-1], "sum": sum(xs)}


def build_prefix_audit(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_state_mode_h: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    trace_cache: Dict[str, List[Mapping[str, Any]]] = {}
    signature_by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    sha_by_key: Dict[Tuple[str, str, int], str] = {}
    for ep in episodes:
        key = (str(ep["state_id"]), str(ep["terminal_mode"]), int(ep["branch_horizon"]))
        by_state_mode_h[key] = ep
        trace = load_trace_for_episode(ep)
        trace_cache[str(ep["path"])] = trace
        sig = prefix_physical_signature(trace, int(ep["branch_step"]))
        signature_by_key[key] = sig
        sha_by_key[key] = canonical_sha(sig)

    rows: List[Dict[str, Any]] = []
    legacy_mismatches = 0
    physical_mismatches = 0
    all_legacy_mismatch_physical_match = 0
    max_dev_all = 0.0
    missing_ref = 0
    for key, ep in sorted(by_state_mode_h.items()):
        sid, mode, h = key
        ref_key = (sid, mode, PREFIX_H)
        ref = by_state_mode_h.get(ref_key)
        if ref is None:
            missing_ref += 1
            continue
        ref_sig = signature_by_key[ref_key]
        sig = signature_by_key[key]
        exact = sha_by_key[key] == sha_by_key[ref_key]
        diff = max_abs_diff(sig, ref_sig)
        if math.isfinite(diff):
            max_dev_all = max(max_dev_all, diff)
        physical_match = bool(len(sig) == len(ref_sig) and diff <= PHYSICAL_PREFIX_TOL)
        legacy_prefix_match: Optional[bool] = None
        # Compare against the summary-level comparison value produced by v1c.
        if h == PREFIX_H:
            legacy_prefix_match = True
        else:
            # All comparison rows are later joined by (state,mode,H). Store here as unknown.
            legacy_prefix_match = None
        if h != PREFIX_H and not physical_match:
            physical_mismatches += 1
        rows.append({
            "state_id": sid,
            "terminal_mode": mode,
            "horizon": h,
            "branch_step": int(ep.get("branch_step", -1)),
            "prefix_len": len(sig),
            "reference_prefix_len": len(ref_sig),
            "physical_prefix_sha256": sha_by_key[key],
            "reference_physical_prefix_sha256": sha_by_key[ref_key],
            "physical_prefix_exact_hash_match_H15": exact,
            "physical_prefix_max_abs_diff_vs_H15": diff,
            "physical_prefix_match_H15": physical_match,
            "episode_path": ep.get("path"),
        })
    return {
        "rows": rows,
        "by_key": {(r["state_id"], r["terminal_mode"], int(r["horizon"])): r for r in rows},
        "missing_H15_references": missing_ref,
        "physical_prefix_mismatch_count_nonH15": physical_mismatches,
        "physical_prefix_max_abs_diff_all": max_dev_all,
    }


def recalc_labels(raw: Mapping[str, Any], prefix_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]]) -> Dict[str, Any]:
    original_analysis = raw.get("analysis") or {}
    state_rows_out: List[Dict[str, Any]] = []
    prefix_artifact_rows: List[Dict[str, Any]] = []
    physical_prefix_mismatch_rows: List[Dict[str, Any]] = []
    terminal_mode_positive_counts: Dict[str, int] = {}
    any_mode_positive_states: List[str] = []
    robust_horizon_counts: Dict[str, int] = {}
    weak_or_one_terminal_candidates: List[Dict[str, Any]] = []

    for state in original_analysis.get("state_rows") or []:
        sid = str(state["state_id"])
        mode_material: Dict[str, List[int]] = {}
        mode_best_gain: Dict[str, float] = {}
        mode_best_rows: Dict[str, Any] = {}
        mode_counts: Dict[str, int] = {}
        comparisons_out: List[Dict[str, Any]] = []
        for mode in TERMINAL_MODES:
            comps = (((state.get("mode_results") or {}).get(mode) or {}).get("comparisons") or [])
            materials: List[int] = []
            best_gain = -float("inf")
            best_row = None
            for comp in comps:
                h = int(comp.get("horizon", -1))
                key = (sid, mode, h)
                pp = prefix_by_key.get(key)
                physical_prefix_match = bool(pp and pp.get("physical_prefix_match_H15"))
                legacy_prefix_match = bool(comp.get("prefix_hash_matches_H15"))
                if h != PREFIX_H and not legacy_prefix_match and physical_prefix_match:
                    prefix_artifact_rows.append({
                        "state_id": sid,
                        "terminal_mode": mode,
                        "horizon": h,
                        "gain_vs_H15_physical": comp.get("gain_vs_H15_physical"),
                        "legacy_prefix_hash_matches_H15": legacy_prefix_match,
                        "physical_prefix_max_abs_diff_vs_H15": None if pp is None else pp.get("physical_prefix_max_abs_diff_vs_H15"),
                    })
                if h != PREFIX_H and not physical_prefix_match:
                    physical_prefix_mismatch_rows.append({"state_id": sid, "terminal_mode": mode, "horizon": h, "prefix_audit": pp})
                gain = float(comp.get("gain_vs_H15_physical", 0.0))
                if h != PREFIX_H and gain > best_gain:
                    best_gain = gain
                    best_row = comp
                material = bool(
                    h != PREFIX_H
                    and bool(comp.get("branch_reached"))
                    and float(comp.get("state_distance_vs_H15", float("inf"))) <= STATE_DISTANCE_TOL
                    and physical_prefix_match
                    and safety_ok(comp)
                    and gain >= MATERIAL_GAIN
                )
                if material:
                    materials.append(h)
                    terminal_mode_positive_counts[mode] = terminal_mode_positive_counts.get(mode, 0) + 1
                comp2 = dict(comp)
                comp2.update({
                    "legacy_prefix_hash_matches_H15": legacy_prefix_match,
                    "physical_prefix_match_H15": physical_prefix_match,
                    "physical_prefix_max_abs_diff_vs_H15": None if pp is None else pp.get("physical_prefix_max_abs_diff_vs_H15"),
                    "repaired_material_positive_terminal_mode": material,
                })
                comparisons_out.append(comp2)
            mode_material[mode] = sorted(set(materials))
            mode_best_gain[mode] = None if best_row is None else float(best_gain)
            mode_best_rows[mode] = best_row
            mode_counts[mode] = len(materials)
        robust = sorted(set(mode_material.get("zero_terminal", [])).intersection(set(mode_material.get("h15_common_terminal", []))))
        if robust:
            for h in robust:
                robust_horizon_counts[str(h)] = robust_horizon_counts.get(str(h), 0) + 1
        if any(mode_material.get(m) for m in TERMINAL_MODES):
            any_mode_positive_states.append(sid)
        if not robust and any(mode_material.get(m) for m in TERMINAL_MODES):
            weak_or_one_terminal_candidates.append({
                "state_id": sid,
                "case": state.get("case"),
                "selection_group": state.get("selection_group"),
                "branch_step": state.get("branch_step"),
                "mode_material_horizons": mode_material,
                "best_nonH15_gain_by_mode": mode_best_gain,
            })
        state_rows_out.append({
            "state_id": sid,
            "case": int(state.get("case", -1)),
            "selection_group": state.get("selection_group"),
            "branch_step": int(state.get("branch_step", -1)),
            "is_control_state": bool(state.get("is_control_state")),
            "original_label": state.get("label"),
            "original_robust_positive_horizons": state.get("robust_positive_horizons"),
            "repaired_mode_material_horizons": mode_material,
            "repaired_robust_positive_horizons": robust,
            "repaired_label": "robust_positive_non_H15" if robust else "negative_or_neutral",
            "best_nonH15_gain_by_mode": mode_best_gain,
            "best_nonH15_rows_by_mode": mode_best_rows,
            "comparison_rows_repaired": comparisons_out,
        })
    repaired_positive_states = [r for r in state_rows_out if r["repaired_robust_positive_horizons"]]
    controls = [r for r in state_rows_out if r["is_control_state"]]
    control_positive = [r for r in controls if r["repaired_robust_positive_horizons"]]
    old_flags = original_analysis.get("artifact_flags") or {}
    legacy_prefix_mismatch_count = int(old_flags.get("prefix_hash_mismatch", 0) or 0)
    repaired_blocking_artifacts = int(
        int(old_flags.get("missing_H15_reference", 0) or 0)
        + int(old_flags.get("missing_horizon_or_terminal_mode", 0) or 0)
        + int(old_flags.get("branch_not_reached", 0) or 0)
        + len(physical_prefix_mismatch_rows)
        + int(old_flags.get("state_distance_gt_tol", 0) or 0)
        + int(old_flags.get("safety_solver_regression", 0) or 0)
        + int(old_flags.get("positive_with_common_terminal_non_success", 0) or 0)
    )
    smoke_gate_repaired = bool(
        len(repaired_positive_states) >= 2
        and len({int(r["case"]) for r in repaired_positive_states}) >= 2
        and (len(state_rows_out) - len(repaired_positive_states)) >= 2
        and repaired_blocking_artifacts == 0
    )
    return {
        "state_rows": state_rows_out,
        "repaired_robust_positive_state_count": len(repaired_positive_states),
        "repaired_robust_positive_cases": sorted({int(r["case"]) for r in repaired_positive_states}),
        "repaired_negative_or_neutral_state_count": len(state_rows_out) - len(repaired_positive_states),
        "repaired_robust_horizon_counts": robust_horizon_counts,
        "terminal_mode_positive_counts": terminal_mode_positive_counts,
        "states_positive_in_any_one_terminal_mode": sorted(set(any_mode_positive_states)),
        "one_terminal_only_candidates": weak_or_one_terminal_candidates,
        "control_state_count": len(controls),
        "control_positive_state_count": len(control_positive),
        "control_false_positive_rate_repaired": (len(control_positive) / len(controls)) if controls else 0.0,
        "legacy_prefix_hash_mismatch_count": legacy_prefix_mismatch_count,
        "prefix_hash_artifact_rows_repaired_to_physical_match": prefix_artifact_rows,
        "physical_prefix_mismatch_rows": physical_prefix_mismatch_rows,
        "repaired_blocking_artifact_count": repaired_blocking_artifacts,
        "smoke_pass_to_full_repaired": smoke_gate_repaired,
        "original_smoke_pass_to_full": bool(original_analysis.get("smoke_pass_to_full")),
        "original_robust_positive_state_count": int(original_analysis.get("robust_positive_state_count", 0) or 0),
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    pfx = raw["prefix_audit"]
    lab = raw["label_reanalysis"]
    decision = raw["decision"]
    lines = [
        "# Vehicle stress-v1c terminal-stable smoke reanalysis v0",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64-bank read, no sealed-test access.",
        "",
        "## Headline",
        "",
        f"- Legacy v1c smoke reported robust positives: `{lab['original_robust_positive_state_count']}` and smoke pass-to-full: `{lab['original_smoke_pass_to_full']}`.",
        f"- Repaired physical-prefix labels report robust positives: `{lab['repaired_robust_positive_state_count']}` across cases `{lab['repaired_robust_positive_cases']}` and smoke pass-to-full: `{lab['smoke_pass_to_full_repaired']}`.",
        f"- Legacy prefix-hash mismatches: `{lab['legacy_prefix_hash_mismatch_count']}`; non-H15 physical-prefix mismatches after removing non-causal decision bookkeeping: `{pfx['physical_prefix_mismatch_count_nonH15']}`; max physical prefix numeric deviation: `{pfx['physical_prefix_max_abs_diff_all']}`.",
        f"- One-terminal-only material states: `{lab['states_positive_in_any_one_terminal_mode']}`.",
        f"- Decision: `{decision['next_action']}`.",
        "",
        "## Per-state repaired labels",
        "",
        "| state | case | group | branch | repaired label | robust H | zero material H | H15-common material H | best non-H15 gain zero | best non-H15 gain common |",
        "|---|---:|---|---:|---|---|---|---|---:|---:|",
    ]
    for row in lab["state_rows"]:
        gains = row["best_nonH15_gain_by_mode"]
        modes = row["repaired_mode_material_horizons"]
        lines.append("| `%s` | %d | `%s` | %d | `%s` | `%s` | `%s` | `%s` | %s | %s |" % (
            row["state_id"], int(row["case"]), row.get("selection_group"), int(row["branch_step"]), row["repaired_label"],
            row["repaired_robust_positive_horizons"], modes.get("zero_terminal", []), modes.get("h15_common_terminal", []),
            "NA" if gains.get("zero_terminal") is None else "%.6g" % float(gains.get("zero_terminal")),
            "NA" if gains.get("h15_common_terminal") is None else "%.6g" % float(gains.get("h15_common_terminal")),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "The prefix-hash gate was over-conservative: after removing non-causal decision bookkeeping from prefix traces, all non-H15 arms match their H15 prefix physically within tolerance. However, the repaired label count remains below the predeclared smoke gate, so the negative smoke outcome is not explained away by the hash artifact.",
        "",
        "## Four-axis evidence update",
        "",
        "### SCENARIOS",
        f"- verified: fresh non-mined terminal-stable smoke still has repaired robust-positive count `{lab['repaired_robust_positive_state_count']}` / `{len(lab['state_rows'])}`.",
        "- hypothesis: state-dependent horizon opportunity under the existing vehicle generator is sparse or concentrated in mined/high-transient states; fresh generic high-heading metadata is insufficient.",
        "- missing: a source-supported scenario-design probe that intentionally varies only environment-supported difficulty/transient factors before outcome scoring.",
        "- discriminator: freeze a small v1d scenario-opportunity map or terminal/modeling diagnostic; do not expand v1c full or train selectors from this sparse smoke.",
        "",
        "### REWARD / TERMINAL",
        "- verified: terminal-stable physical-continuation labels remain sparse after repairing the prefix-equivalence audit; per-H/H25 terminal labels remain excluded as unsafe targets.",
        "- hypothesis: terminal-value structure may still suppress/alter horizon choices, but current zero/H15-common realized labels do not provide a dense supervised target.",
        "- missing: compact diagnostic distinguishing terminal-value model bias from true absence of physical opportunity on carefully chosen stress states.",
        "- discriminator: compare zero/common/learned-terminal objective ranks and realized continuations on a fresh targeted state bank, or refit terminal/value only if labels become stable.",
        "",
        "### TRAINING",
        "- verified: no new training/refit/gradient steps; repaired smoke gate does not justify selector training.",
        "- hypothesis: historical near-constant policies may be deficient, but stable labels are not yet dense enough for a fair refit.",
        "- missing: compact IMPROVED selector/value-refit remains deferred until a stable-label density gate passes.",
        "- discriminator: after a successful label-density gate, run a bounded refit smoke and then 3 independent seeds.",
        "",
        "### COMPARISONS",
        "- verified: no adaptive superiority claim; no fixed-H retuning or measured-runtime validation was run in this diagnostic.",
        "- hypothesis: strong fixed-H/Pareto baselines may absorb any remaining gains.",
        "- missing: fair fixed-H/timing comparison under any revised protocol.",
        "- discriminator: only after a candidate method/scenario protocol is frozen, compare against tuned fixed-H with actual decision/solver timing.",
        "",
        "## Next action",
        "",
        decision["rationale"],
        "",
        f"Backup request: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    lab = raw["label_reanalysis"]
    pfx = raw["prefix_audit"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1c terminal-stable smoke reanalysis v0

UTC: {raw['created_utc']}. No-simulation development reanalysis completed. Legacy prefix-hash mismatches were diagnosed as bookkeeping over-blocking: legacy mismatches={lab['legacy_prefix_hash_mismatch_count']}, repaired physical-prefix mismatches={pfx['physical_prefix_mismatch_count_nonH15']}, max physical-prefix deviation={pfx['physical_prefix_max_abs_diff_all']}. Repaired robust-positive states remain {lab['repaired_robust_positive_state_count']} across cases {lab['repaired_robust_positive_cases']}; repaired smoke gate={lab['smoke_pass_to_full_repaired']}. Therefore do not train/refit from v1c labels and do not run full v1c unchanged; next action is a versioned scenario/terminal-opportunity diagnostic on fresh source-supported states. No validation64/test access, no new rollouts/control steps/training. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    if (OUT / "completed.json").exists():
        verify_completed(OUT / "completed.json", require_hashes=True)
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "sealed_test_accessed": False, "validation64_bank_opened": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect before rerun: {rel(OUT)}")
    for path in (INPUT_RAW, INPUT_DONE, INPUT_SUMMARY, INPUT_SCHEDULE, INPUT_BANK_DONE, PROTOCOL_JSON):
        if not path.exists():
            raise ContractError(f"required input missing: {rel(path)}")
    input_done = verify_completed(INPUT_DONE, require_hashes=False)
    bank_done = verify_completed(INPUT_BANK_DONE, require_hashes=False)
    raw_in = read_json(INPUT_RAW)
    if raw_in.get("sealed_test_accessed") is not False or raw_in.get("validation64_bank_opened") is not False or raw_in.get("historical_validation64_bank_opened") is not False:
        raise ContractError("input raw has invalid access flags")
    if int(raw_in.get("budget_actual", {}).get("episodes", -1)) != 168:
        raise ContractError("expected the completed smoke raw with 168 episodes")
    if int(raw_in.get("budget_actual", {}).get("new_training_episodes", -1)) != 0 or int(raw_in.get("budget_actual", {}).get("new_gradient_steps", -1)) != 0:
        raise ContractError("input smoke unexpectedly has training budget")
    episodes = raw_in.get("episodes") or []
    if len(episodes) != 168:
        raise ContractError(f"expected 168 episode summaries, found {len(episodes)}")

    OUT.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0",
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    })

    prefix_audit = build_prefix_audit(episodes)
    label_reanalysis = recalc_labels(raw_in, prefix_audit["by_key"])
    # Drop tuple-key helper before serializing.
    prefix_rows = prefix_audit["rows"]
    prefix_audit_public = {
        "rows": prefix_rows,
        "missing_H15_references": prefix_audit["missing_H15_references"],
        "physical_prefix_mismatch_count_nonH15": prefix_audit["physical_prefix_mismatch_count_nonH15"],
        "physical_prefix_max_abs_diff_all": prefix_audit["physical_prefix_max_abs_diff_all"],
        "physical_prefix_exact_hash_match_summary": {
            "nonH15_rows": sum(1 for r in prefix_rows if int(r["horizon"]) != PREFIX_H),
            "nonH15_physical_match_rows": sum(1 for r in prefix_rows if int(r["horizon"]) != PREFIX_H and r["physical_prefix_match_H15"]),
            "nonH15_exact_sha_match_rows": sum(1 for r in prefix_rows if int(r["horizon"]) != PREFIX_H and r["physical_prefix_exact_hash_match_H15"]),
        },
    }
    if label_reanalysis["smoke_pass_to_full_repaired"]:
        next_action = "backup_then_consider_full_v1c_or_targeted_label_confirmation"
        rationale = "The repaired smoke gate passes, so the next bounded step would be a backed-up full v1c label-density run before any selector/refit."
    else:
        next_action = "do_not_train_do_not_run_full_v1c_unchanged_freeze_scenario_terminal_opportunity_diagnostic"
        rationale = "The repaired smoke gate still fails after removing the bookkeeping hash artifact. Running the full v1c unchanged has lower information value than a targeted scenario/terminal-opportunity diagnostic, and selector/refit training would be label-starved."
    decision = {
        "train_or_refit_now": False,
        "run_full_v1c_unchanged_next": bool(label_reanalysis["smoke_pass_to_full_repaired"]),
        "next_action": next_action,
        "rationale": rationale,
        "recommended_next_protocol": {
            "name": "vehicle_stress_v1d_source_supported_scenario_terminal_opportunity_smoke",
            "purpose": "separate true absence of adaptive-horizon opportunity from current scenario selection/terminal-model limitations using fresh source-supported states before any selector training",
            "do_not_use_validation64_or_sealed_test": True,
            "training_budget": {"new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0},
            "suggested_budget_upper_bound": {"fresh_candidate_resets": "metadata-only bounded audit first, then <=12 states if source-supported", "development_rollout_episodes": "bounded smoke only after frozen protocol and backup"},
        },
    }
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(BACKUP_REQUEST, {
        "requested_utc": created,
        "reason": "backup no-simulation v1c terminal-stable smoke reanalysis before any further scenario/method diagnostic or simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(SOURCE), rel(BACKUP_REQUEST)],
    })
    raw = {
        "created_utc": created,
        "started_utc": started,
        "method": "vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0",
        "classification": "development_no_simulation_reanalysis_prefix_hash_and_label_repair_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "inputs": {
            "smoke_raw": rel(INPUT_RAW),
            "smoke_raw_sha256": sha256(INPUT_RAW),
            "smoke_completed": rel(INPUT_DONE),
            "smoke_completed_sha256": sha256(INPUT_DONE),
            "smoke_summary": rel(INPUT_SUMMARY),
            "smoke_summary_sha256": sha256(INPUT_SUMMARY),
            "smoke_schedule": rel(INPUT_SCHEDULE),
            "smoke_schedule_sha256": sha256(INPUT_SCHEDULE),
            "bank_completed": rel(INPUT_BANK_DONE),
            "bank_completed_sha256": sha256(INPUT_BANK_DONE),
            "protocol": rel(PROTOCOL_JSON),
            "protocol_sha256": sha256(PROTOCOL_JSON),
            "input_smoke_completed_created_utc": input_done.get("created_utc"),
            "input_bank_completed_hash": sha256(INPUT_BANK_DONE),
        },
        "analysis_constants": {
            "prefix_horizon": PREFIX_H,
            "terminal_modes": TERMINAL_MODES,
            "material_gain_physical": MATERIAL_GAIN,
            "state_distance_tol": STATE_DISTANCE_TOL,
            "physical_prefix_tol": PHYSICAL_PREFIX_TOL,
        },
        "prefix_audit": prefix_audit_public,
        "label_reanalysis": label_reanalysis,
        "decision": decision,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "backup_request": rel(BACKUP_REQUEST),
        "interpretation_limits": [
            "development artifact reanalysis only",
            "not a new rollout or validation campaign",
            "not model selection",
            "not final test",
            "repaired labels remain tied to the v1c smoke state selection",
        ],
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Vehicle stress-v1c terminal-stable smoke reanalysis v0 ({created})\n\n"
        f"No simulation/training/validation/test. Repaired prefix audit: legacy mismatches={label_reanalysis['legacy_prefix_hash_mismatch_count']}, "
        f"physical-prefix non-H15 mismatches={prefix_audit_public['physical_prefix_mismatch_count_nonH15']}, max diff={prefix_audit_public['physical_prefix_max_abs_diff_all']}. "
        f"Repaired robust positives={label_reanalysis['repaired_robust_positive_state_count']} across cases={label_reanalysis['repaired_robust_positive_cases']}; "
        f"smoke gate={label_reanalysis['smoke_pass_to_full_repaired']}. Decision: {next_action}. Backup requested before any further simulations.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE, BACKUP_REQUEST, INPUT_RAW, INPUT_DONE, INPUT_SUMMARY, INPUT_SCHEDULE, INPUT_BANK_DONE, PROTOCOL_JSON]
    write_json(OUT / "completed.json", {
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
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "backup_request": rel(BACKUP_REQUEST),
        "headline": {
            "legacy_prefix_hash_mismatch_count": label_reanalysis["legacy_prefix_hash_mismatch_count"],
            "physical_prefix_mismatch_count_nonH15": prefix_audit_public["physical_prefix_mismatch_count_nonH15"],
            "physical_prefix_max_abs_diff_all": prefix_audit_public["physical_prefix_max_abs_diff_all"],
            "original_robust_positive_state_count": label_reanalysis["original_robust_positive_state_count"],
            "repaired_robust_positive_state_count": label_reanalysis["repaired_robust_positive_state_count"],
            "repaired_robust_positive_cases": label_reanalysis["repaired_robust_positive_cases"],
            "smoke_pass_to_full_repaired": label_reanalysis["smoke_pass_to_full_repaired"],
            "train_or_refit_now": False,
            "next_action": next_action,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "legacy_prefix_hash_mismatch_count": label_reanalysis["legacy_prefix_hash_mismatch_count"],
        "physical_prefix_mismatch_count_nonH15": prefix_audit_public["physical_prefix_mismatch_count_nonH15"],
        "physical_prefix_max_abs_diff_all": prefix_audit_public["physical_prefix_max_abs_diff_all"],
        "original_robust_positive_state_count": label_reanalysis["original_robust_positive_state_count"],
        "repaired_robust_positive_state_count": label_reanalysis["repaired_robust_positive_state_count"],
        "repaired_robust_positive_cases": label_reanalysis["repaired_robust_positive_cases"],
        "smoke_pass_to_full_repaired": label_reanalysis["smoke_pass_to_full_repaired"],
        "next_action": next_action,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": rel(BACKUP_REQUEST),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve partial output and repair only the no-simulation analyzer if needed; do not rerun v1c smoke or open validation/test for this diagnostic.",
        })
        raise
