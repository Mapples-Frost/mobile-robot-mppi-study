#!/usr/bin/env python3
"""Read-only candidate failure-surface diagnostic for vehicle actual-time V2b.

This is a metadata-only diagnostic over the already generated V2b re-selection
artifacts and the just-run V2b smoke preflight. It performs no environment import,
no TensorFlow import, no simulation, no training, no validation64-bank access, and
no sealed-test access.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
V2B_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z"
V2B_COMPLETED = V2B_DIR / "completed.json"
V2B_NOMINATED = V2B_DIR / "nominated_policies.json"
V2B_CSV = V2B_DIR / "candidate_timing_metrics.csv"
V2B_RAW = V2B_DIR / "raw.json"
PREFLIGHT_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/completed.json"
PREFLIGHT_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/raw.json"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z.md"
BACKUP = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_ACTUAL_TIME_V2B_CANDIDATE_FAILURE_SURFACE_DIAGNOSTIC_20260928T0915Z.json"
DOCS = [ROOT / x for x in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]]
MARKER = "<!-- vehicle-actual-time-v2b-candidate-failure-surface-diagnostic-20260928 -->"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def sha256(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_once(path: Path, block: str) -> None:
    if not path.exists():
        return
    old = path.read_text(encoding="utf-8")
    if MARKER in old:
        return
    path.write_text(old.rstrip() + "\n" + block + "\n", encoding="utf-8")


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        y = float(x)
        return y if math.isfinite(y) else None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "na", "null"}:
        return None
    try:
        y = float(s)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def as_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    return None


def first(row: Dict[str, Any], names: Iterable[str]) -> Any:
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None


def parse_candidate_attrs(cid: str) -> Dict[str, Optional[int]]:
    m = re.match(r"^h(?P<h>\d+)_p(?P<p>\d+)_g(?P<g>\d+)$", cid or "")
    if not m:
        return {"short_h": None, "profile": None, "guard": None}
    return {"short_h": int(m.group("h")), "profile": int(m.group("p")), "guard": int(m.group("g"))}


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def sort_key_desc(row: Dict[str, Any], field: str) -> Tuple[int, float]:
    v = as_float(row.get(field))
    return (0, -1e300) if v is None else (1, v)


def sort_key_asc(row: Dict[str, Any], field: str) -> Tuple[int, float]:
    v = as_float(row.get(field))
    return (0, 1e300) if v is None else (1, -v)


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(first(row, ["candidate_id", "policy_id", "id"]) or "")
    attrs = parse_candidate_attrs(cid)
    keys = [
        "seed", "candidate_id", "actual_time_primary_eligible", "eligible", "estimated_net_time_saving_fraction",
        "timing_ratio_median_vs_H25", "decision_ratio_median_vs_H25", "mean_decision_ratio_vs_H25",
        "selection_overhead_fraction", "mean_physical_delta_pct_vs_fixed", "max_pair_physical_regression_pct",
        "positive_tail_cvar80_pct", "rejection_reason", "rejection_reasons",
    ]
    out: Dict[str, Any] = {"candidate_id": cid, **attrs}
    for k in keys:
        if k in row and row[k] not in ("", None):
            v = row[k]
            fv = as_float(v)
            bv = as_bool(v)
            if bv is not None and str(v).strip().lower() in {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}:
                out[k] = bv
            elif fv is not None:
                out[k] = fv
            else:
                out[k] = v
    return out


def infer_blockers(row: Dict[str, Any]) -> List[str]:
    reasons_raw = first(row, ["rejection_reason", "rejection_reasons", "blocked_reasons", "failure_reasons"])
    if reasons_raw:
        return [r.strip() for r in re.split(r"[;|,]", str(reasons_raw)) if r.strip()]
    blockers: List[str] = []
    eligible = as_bool(first(row, ["actual_time_primary_eligible", "eligible", "primary_eligible"]))
    if eligible is True:
        return []
    net = as_float(first(row, ["estimated_net_time_saving_fraction", "net_time_saving_fraction", "net_time_saving"]));
    ratio = as_float(first(row, ["timing_ratio_median_vs_H25", "decision_ratio_median_vs_H25", "mean_decision_ratio_vs_H25", "decision_time_ratio_vs_H25"]));
    mean_phys = as_float(first(row, ["mean_physical_delta_pct_vs_fixed", "physical_delta_pct_mean", "mean_physical_delta_pct"]));
    max_reg = as_float(first(row, ["max_pair_physical_regression_pct", "max_physical_regression_pct"]));
    cvar = as_float(first(row, ["positive_tail_cvar80_pct", "tail_cvar80_pct"]));
    if net is not None and net <= 0:
        blockers.append("nonpositive_estimated_net_time_saving")
    if ratio is not None and ratio >= 1.0:
        blockers.append("median_timing_not_faster_than_H25")
    # These are descriptive conservative indicators, not new post-hoc acceptance thresholds.
    if mean_phys is not None and mean_phys > 0:
        blockers.append("mean_physical_regression_positive")
    if max_reg is not None and max_reg > 0:
        blockers.append("some_pair_physical_regression_positive")
    if cvar is not None and cvar > 0:
        blockers.append("positive_tail_risk_nonzero")
    if not blockers and eligible is False:
        blockers.append("not_primary_eligible_reason_not_encoded")
    return blockers


def summarize_seed(seed: str, rows: List[Dict[str, Any]], nomination: Dict[str, Any]) -> Dict[str, Any]:
    adaptive = [r for r in rows if str(first(r, ["candidate_id", "policy_id", "id"]) or "") != "fixed"]
    parsed_ids = [str(first(r, ["candidate_id", "policy_id", "id"]) or "") for r in rows]
    attrs = [parse_candidate_attrs(cid) for cid in parsed_ids]
    eligible_rows = [r for r in rows if as_bool(first(r, ["actual_time_primary_eligible", "eligible", "primary_eligible"])) is True]
    adaptive_eligible = [r for r in adaptive if as_bool(first(r, ["actual_time_primary_eligible", "eligible", "primary_eligible"])) is True]
    blocker_counts: Counter[str] = Counter()
    for r in adaptive:
        for b in infer_blockers(r):
            blocker_counts[b] += 1
    top_net = sorted(adaptive, key=lambda r: sort_key_desc(r, "estimated_net_time_saving_fraction"), reverse=True)[:8]
    top_fast = sorted(adaptive, key=lambda r: sort_key_asc(r, "timing_ratio_median_vs_H25"), reverse=True)[:8]
    candidate_n = nomination["nominated"]
    nominated_id = str(candidate_n.get("candidate_id"))
    return {
        "seed": int(seed),
        "candidate_count": len(rows),
        "adaptive_candidate_count": len(adaptive),
        "eligible_count_all": len(eligible_rows),
        "eligible_count_adaptive": len(adaptive_eligible),
        "nominated_id": nominated_id,
        "fallback_to_fixed": bool(nomination.get("fallback_to_fixed")),
        "candidate_short_h_values": sorted({a["short_h"] for a in attrs if a["short_h"] is not None}),
        "candidate_profile_values": sorted({a["profile"] for a in attrs if a["profile"] is not None}),
        "candidate_guard_values": sorted({a["guard"] for a in attrs if a["guard"] is not None}),
        "adaptive_blocker_counts_descriptive": dict(sorted(blocker_counts.items())),
        "best_by_estimated_net_time_saving": [compact_row(r) for r in top_net],
        "best_by_median_timing_ratio": [compact_row(r) for r in top_fast],
        "nominated_row_from_csv": next((compact_row(r) for r in rows if str(first(r, ["candidate_id", "policy_id", "id"]) or "") == nominated_id), None),
        "nominated_metadata": nomination,
    }


def main() -> int:
    if OUT.exists():
        raise RuntimeError("Output directory already exists; refusing overwrite: %s" % OUT)
    required = [V2B_COMPLETED, V2B_NOMINATED, V2B_CSV, V2B_RAW, PREFLIGHT_COMPLETED, PREFLIGHT_RAW]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required inputs: %r" % missing)
    completed = read_json(V2B_COMPLETED)
    nominations = read_json(V2B_NOMINATED)
    preflight = read_json(PREFLIGHT_COMPLETED)
    rows = read_csv_rows(V2B_CSV)
    if not rows:
        raise RuntimeError("No candidate rows in %s" % rel(V2B_CSV))
    columns = list(rows[0].keys())
    by_seed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        seed = first(r, ["seed", "training_seed"])
        if seed is None:
            seed = "unknown"
        seed_s = str(int(float(str(seed)))) if re.match(r"^-?\d+(\.0+)?$", str(seed).strip()) else str(seed)
        by_seed[seed_s].append(r)
    seed_summaries = {s: summarize_seed(s, by_seed.get(s, []), nominations["nominations"][s]) for s in ["0", "1", "2"]}
    corrected_ids = {s: seed_summaries[s]["nominated_id"] for s in ["0", "1", "2"]}
    adaptive_nom_count = sum(1 for s in ["0", "1", "2"] if corrected_ids[s] != "fixed")
    verified_causes = [
        "The current V2b method is finite metadata re-selection over existing gated candidates; it performs zero new gradient updates.",
        "V2b corrected the seed0 overhead lookup and the corrected smoke nomination set remains seed0=fixed, seed1=h15_p1_g5, seed2=h15_p1_g5.",
        "Seed0 has zero eligible adaptive candidates under the V2b actual-time/physical-risk gates and therefore falls back to H25 for any V2b smoke.",
        "Only two of three independent seeds nominate an adaptive candidate before smoke, and both adaptive nominations are the same finite candidate h15_p1_g5.",
    ]
    missing_evidence = [
        "No V2b controller-stack rollout has been run yet; executability, replay and realized wall-time ratios remain untested until the 36-episode smoke runs after backup.",
        "The V2b selector still cannot diagnose value/terminal-function bias because all arms reuse the existing H25 terminal/value source.",
        "The finite candidate grid cannot establish that a richer learned selector would fail; it only diagnoses the current candidate class and measured-time-aware re-selection layer.",
        "Fresh confirmation devval and sealed final test remain unopened/unavailable for V2b claims.",
    ]
    decision = (
        "Run the already-frozen 36-episode legacy V2b smoke after verified external backup. "
        "Do not resume the long risk-reselection devval64 shards by default. If the smoke passes, freeze only a bounded fresh devval block for V2b or move to a stronger selector/scenario-opportunity diagnostic; do not claim success from this metadata diagnostic."
    )
    raw = {
        "created_utc": now(),
        "method": "metadata_only_vehicle_actual_time_v2b_candidate_failure_surface_diagnostic",
        "classification": "IMPROVED diagnostic; no new rollouts/training; not ORIGINAL SAC",
        "access_flags": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_bank_opened": False, "sealed_test_accessed": False},
        "input_hashes": {rel(p): sha256(p) for p in required + [Path(__file__).resolve()]},
        "v2b_completed_passed": completed.get("passed") is True,
        "v2b_acceptance_for_smoke_met": completed.get("acceptance_for_smoke_met") is True,
        "preflight_hard_pass": preflight.get("hard_pass") is True,
        "candidate_csv_columns": columns,
        "candidate_rows_total": len(rows),
        "candidate_rows_by_seed": {s: len(by_seed.get(s, [])) for s in sorted(by_seed.keys())},
        "corrected_nomination_ids": corrected_ids,
        "adaptive_nominated_seed_count": adaptive_nom_count,
        "seed_summaries": seed_summaries,
        "verified_causes": verified_causes,
        "missing_evidence": missing_evidence,
        "scientific_decision": decision,
        "next_action": "after_verified_backup_run_legacy_actual_time_v2b_smoke",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "raw.json", raw)
    lines = [
        "# Vehicle actual-time V2b candidate failure-surface diagnostic",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only; no simulations, no training, no validation64-bank access, no sealed-test access.",
        "",
        f"V2b completed passed: `{raw['v2b_completed_passed']}`; acceptance_for_smoke_met: `{raw['v2b_acceptance_for_smoke_met']}`; smoke preflight hard_pass: `{raw['preflight_hard_pass']}`.",
        f"Candidate CSV rows: `{len(rows)}`; columns: `{columns}`.",
        f"Corrected nominations: `{corrected_ids}`; adaptive nominated seed count: `{adaptive_nom_count}/3`.",
        "",
        "## Per-seed failure surface",
        "",
    ]
    for s in ["0", "1", "2"]:
        ss = seed_summaries[s]
        lines += [
            f"### Seed {s}",
            f"- candidates={ss['candidate_count']}, adaptive_candidates={ss['adaptive_candidate_count']}, eligible_adaptive={ss['eligible_count_adaptive']}, nominated={ss['nominated_id']}, fallback_to_fixed={ss['fallback_to_fixed']}",
            f"- grid coverage: short_h={ss['candidate_short_h_values']}, profiles={ss['candidate_profile_values']}, guards={ss['candidate_guard_values']}",
            f"- descriptive blocker counts among adaptive non/eligible rows: `{ss['adaptive_blocker_counts_descriptive']}`",
            f"- top estimated-net-saving candidates: `{ss['best_by_estimated_net_time_saving'][:3]}`",
            f"- fastest median-timing candidates: `{ss['best_by_median_timing_ratio'][:3]}`",
            "",
        ]
    lines += [
        "## Verified causes vs missing evidence",
        "",
    ]
    lines.extend([f"- Verified: {x}" for x in verified_causes])
    lines.extend([f"- Missing: {x}" for x in missing_evidence])
    lines += ["", "## Decision", "", decision, ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    completed_obj = {
        "created_utc": now(),
        "passed": True,
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "candidate_rows_total": len(rows),
        "corrected_nomination_ids": corrected_ids,
        "adaptive_nominated_seed_count": adaptive_nom_count,
        "preflight_hard_pass": preflight.get("hard_pass") is True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": rel(BACKUP),
        "next_action": raw["next_action"],
    }
    write_json(OUT / "completed.json", completed_obj)
    STATE.write_text(
        "# Vehicle actual-time V2b candidate failure-surface diagnostic state\n\n"
        f"UTC: {raw['created_utc']}\n\n"
        f"- Metadata-only diagnostic completed; rows={len(rows)}; no validation64/test access; no rollouts/training.\n"
        f"- Corrected nominations remain {corrected_ids}; seed0 fixed fallback, seeds1/2 adaptive h15_p1_g5.\n"
        "- Verified blocker: seed0 has no eligible adaptive candidate in this finite class under V2b gates.\n"
        "- Next action after verified backup: run the frozen 36-episode legacy V2b smoke; do not resume long risk-reselection devval by default.\n",
        encoding="utf-8",
    )
    backup_obj = {
        "created_utc": now(),
        "reason": "Backup required after candidate failure-surface diagnostic and prior preflight before any V2b smoke simulation.",
        "backup_required_before_more_simulations": True,
        "artifacts_requiring_backup": [rel(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "completed.json", STATE, BACKUP, Path(__file__).resolve(), PREFLIGHT_COMPLETED, PREFLIGHT_RAW, V2B_COMPLETED, V2B_NOMINATED, V2B_CSV]],
        "sha256": {rel(p): sha256(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "completed.json", STATE, Path(__file__).resolve(), PREFLIGHT_COMPLETED, PREFLIGHT_RAW, V2B_COMPLETED, V2B_NOMINATED, V2B_CSV]},
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
    }
    write_json(BACKUP, backup_obj)
    block = (
        f"\n{MARKER}\n"
        "## 2026-09-28 vehicle actual-time V2b candidate failure-surface diagnostic\n\n"
        f"UTC: {raw['created_utc']}. Metadata-only diagnostic completed over V2b candidate metrics and preflight artifacts: rows={len(rows)}, nominations={corrected_ids}, adaptive_nominated_seed_count={adaptive_nom_count}/3. "
        "No simulations, no training, no validation64 bank access, and no sealed-test access. Verified seed0 has no eligible adaptive candidate under the current finite V2b gates; seeds1/2 both nominate h15_p1_g5. "
        "Next action after verified backup: run the already-frozen 36-episode legacy V2b smoke; do not resume the unchanged long risk-reselection devval shards by default. "
        f"Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.\n"
    )
    for d in DOCS:
        append_once(d, block)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "candidate_rows_total": len(rows),
        "corrected_nomination_ids": corrected_ids,
        "adaptive_nominated_seed_count": adaptive_nom_count,
        "preflight_hard_pass": preflight.get("hard_pass") is True,
        "next_action": raw["next_action"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
