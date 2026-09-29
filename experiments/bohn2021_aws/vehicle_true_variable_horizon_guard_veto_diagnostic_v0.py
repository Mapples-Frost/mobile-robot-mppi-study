#!/usr/bin/env python3
"""Development-only guard/representation veto diagnostic for true-variable-H vehicle branch.

This script does NOT run simulations.  It consumes the already completed fresh-source
H10/H15 branch bank and the earlier oracle/risk source banks.  The goal is to
separate two explanations for the failed fresh-source confirmation:

  H1: the source-trained guard failed mainly because it discarded terminal-profile
      disagreement states as unusable, leaving almost no negative support; treating
      disagreement states as abstention/veto anchors could repair false positives.
  H2: the online-observable representation/source labels are not adequate; even
      simple conservative veto/representation variants cannot recover a useful
      no-false-positive compute tradeoff on the fresh bank.

Any candidate that looks good here is development-tuned evidence only and requires
another frozen fresh-source confirmation before validation/test use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_guard_veto_diagnostic_v0"
STAMP = "20260929T1035Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

FRESH_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z"
FRESH_RAW = FRESH_DIR / "raw.json"
FRESH_DONE = FRESH_DIR / "completed.json"
FRESH_MANIFEST = FRESH_DIR / "selected_state_manifest.json"
ORACLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"
RISK_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_FILE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1035_after_guard_veto_diagnostic.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"

PRIMARY_PROFILE = "shared_h15_terminal"
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
TRUE_HORIZONS = [10, 15]
MARKER = f"vehicle-true-variable-H-guard-veto-diagnostic-v0-{STAMP}"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, Mapping):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
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


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def feature_from_observation_state(obs: Any, state: Any) -> List[float]:
    vals: List[float] = []
    if isinstance(obs, list) and obs:
        vals.extend(sf(x, 0.0) for x in obs[:14])
    if not vals and isinstance(state, Mapping):
        vals.extend([sf(state.get("x"), 0.0) / 30.0, sf(state.get("y"), 0.0) / 30.0, sf(state.get("theta"), 0.0) / math.pi])
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def feature_from_episode_summary(e: Mapping[str, Any]) -> List[float]:
    br = e.get("branch_reset") or {}
    return feature_from_observation_state(br.get("initial_observation_at_branch"), br.get("branch_state_target"))


def transform(x: Sequence[float], mode: str) -> List[float]:
    raw = [sf(v, 0.0) for v in x[:14]] + [0.0] * max(0, 14 - len(x))
    raw = raw[:14]
    if mode == "raw":
        return raw
    if mode == "raw_abs_l2":
        z = raw + [abs(v) for v in raw]
        norm = math.sqrt(sum(v * v for v in z))
        return [v / norm for v in z] if norm > 1e-12 else z
    if mode == "pose_goal_obs_l2":
        # First five dimensions are normalized pose/near-goal trajectory features in
        # the existing vehicle observation; append absolute pose for sign-robustness.
        z = raw[:5] + [abs(v) for v in raw[:5]]
        norm = math.sqrt(sum(v * v for v in z))
        return [v / norm for v in z] if norm > 1e-12 else z
    if mode == "obstacle_slice_l2":
        # Object/constraint-proximity portion of the observation used only as an
        # exploratory representation diagnostic, not a deployment claim.
        z = raw[5:14] + [abs(v) for v in raw[5:14]]
        norm = math.sqrt(sum(v * v for v in z))
        return [v / norm for v in z] if norm > 1e-12 else z
    raise ValueError("unknown mode " + mode)


def completed_guard(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise RuntimeError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("sealed_test_accessed") is not False or obj.get("validation64_bank_opened") is not False:
        raise RuntimeError("prerequisite opened validation/test: " + rel(path))
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise RuntimeError("prerequisite did not complete cleanly: " + rel(path))
    return obj


def load_source_examples(raw_path: Path, source_name: str) -> List[Dict[str, Any]]:
    raw = read_json(raw_path)
    rows = (((raw.get("analysis") or {}).get("state_profile_rows")) or [])
    episodes = raw.get("episodes") or []
    labels: Dict[str, Dict[str, Optional[int]]] = {}
    for r in rows:
        sid = str(r.get("state_id"))
        prof = str(r.get("terminal_profile"))
        lab = r.get("oracle_label")
        labels.setdefault(sid, {})[prof] = None if lab is None else int(lab)
    ep_by_sid_prof: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    any_h15_by_sid: Dict[str, Mapping[str, Any]] = {}
    for e in episodes:
        sid = str(e.get("state_id"))
        prof = str(e.get("terminal_profile") or e.get("terminal_mode"))
        if si(e.get("true_mpc_n_horizon"), -1) == 15:
            ep_by_sid_prof.setdefault((sid, prof), e)
            any_h15_by_sid.setdefault(sid, e)
    out: List[Dict[str, Any]] = []
    for sid, profs in sorted(labels.items()):
        labs = {p: profs.get(p) for p in TERMINAL_PROFILES if profs.get(p) is not None}
        ep = ep_by_sid_prof.get((sid, PRIMARY_PROFILE)) or ep_by_sid_prof.get((sid, "matched_terminal")) or any_h15_by_sid.get(sid)
        if ep is None:
            continue
        vals = [labs[p] for p in TERMINAL_PROFILES if p in labs]
        agreement = len(vals) == len(TERMINAL_PROFILES) and len(set(vals)) == 1
        if agreement and vals[0] == 10:
            category = "agreement_positive_h10"
            binary = 1
        elif agreement:
            category = "agreement_non_h10"
            binary = 0
        else:
            category = "terminal_disagreement_or_missing"
            binary = 0
        out.append({
            "source": source_name,
            "state_id": sid,
            "labels_by_profile": labs,
            "category": category,
            "binary_h10": binary,
            "feature": feature_from_episode_summary(ep),
            "episode_path": ep.get("path"),
        })
    return out


def build_fresh_rows(fresh_raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    manifest = fresh_raw.get("selected_state_manifest")
    if isinstance(manifest, Mapping) and "selected_states" in manifest:
        states = manifest.get("selected_states") or []
    elif isinstance(manifest, list):
        states = manifest
    else:
        states = read_json(FRESH_MANIFEST).get("selected_states") or []
    by_base = {str(s.get("base_state_id")): s for s in states}
    rows: List[Dict[str, Any]] = []
    for r in ((fresh_raw.get("analysis") or {}).get("state_profile_rows") or []):
        base = str(r.get("base_state_id"))
        man = by_base.get(base, {})
        feat = feature_from_observation_state(man.get("initial_observation_from_h15_trace"), man.get("branch_previous_state"))
        med = r.get("median_by_h") or {}
        h10 = med.get("10") or med.get(10) or {}
        h15 = med.get("15") or med.get(15) or {}
        h15_phys = sf(h15.get("physical"), 0.0)
        h10_phys = sf(h10.get("physical"), 0.0)
        rows.append({
            "base_state_id": base,
            "terminal_profile": str(r.get("terminal_profile")),
            "fresh_confirmation_group": r.get("fresh_confirmation_group"),
            "branch_state_slot": si(r.get("branch_state_slot"), -1),
            "feature": feat,
            "stage_a_trace_risk_score": sf(man.get("stage_a_trace_risk_score"), 0.0),
            "h10_beneficial_vs_h15": bool(r.get("h10_beneficial_vs_h15")),
            "oracle_label": r.get("oracle_label"),
            "h10_physical": h10_phys,
            "h15_physical": h15_phys,
            "row_physical_tolerance": max(2.0, 0.05 * abs(h15_phys)),
            "h10_decision_sum_s": sf(h10.get("decision_sum_s"), 0.0),
            "h15_decision_sum_s": sf(h15.get("decision_sum_s"), 0.0),
            "h10_solver_sum_s": sf(h10.get("solver_sum_s"), 0.0),
            "h15_solver_sum_s": sf(h15.get("solver_sum_s"), 0.0),
            "h10_safe_all": bool(h10.get("safe_all")),
            "h15_safe_all": bool(h15.get("safe_all")),
        })
    return rows


def fit_predictor(examples: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "raw"))
    pos = [transform(e["feature"], mode) for e in examples if e.get("category") == "agreement_positive_h10"]
    if spec.get("negative_pool") == "agreement_only":
        neg_examples = [e for e in examples if e.get("category") == "agreement_non_h10"]
    elif spec.get("negative_pool") == "all_nonpositive_including_disagreement":
        neg_examples = [e for e in examples if e.get("category") != "agreement_positive_h10"]
    else:
        neg_examples = []
    if spec.get("veto_pool") == "disagreement_only":
        veto_examples = [e for e in examples if e.get("category") == "terminal_disagreement_or_missing"]
    elif spec.get("veto_pool") == "all_nonpositive_including_disagreement":
        veto_examples = [e for e in examples if e.get("category") != "agreement_positive_h10"]
    else:
        veto_examples = []
    neg = [transform(e["feature"], mode) for e in neg_examples]
    veto = [transform(e["feature"], mode) for e in veto_examples]
    nn: List[float] = []
    for i, p in enumerate(pos):
        ds = [dist(p, q) for j, q in enumerate(pos) if i != j]
        if ds:
            nn.append(min(ds))
    radius = max(quantile(nn, sf(spec.get("positive_radius_quantile"), 0.5)) * sf(spec.get("positive_radius_scale"), 1.25), 1e-6) if nn else 0.0
    return {"spec": dict(spec), "mode": mode, "pos": pos, "neg": neg, "veto": veto, "radius": radius, "counts": {"positive": len(pos), "negative": len(neg), "veto": len(veto)}}


def predict(model: Mapping[str, Any], feature: Sequence[float]) -> Tuple[int, Dict[str, Any]]:
    spec = model["spec"]
    x = transform(feature, str(model["mode"]))
    pos: List[List[float]] = model["pos"]
    neg: List[List[float]] = model["neg"]
    veto: List[List[float]] = model["veto"]
    min_support = si(spec.get("min_positive_support"), 1)
    if len(pos) < min_support:
        return 15, {"reason": "insufficient_positive_training", "counts": model["counts"]}
    dpos = sorted(dist(x, p) for p in pos)
    dneg = sorted(dist(x, n) for n in neg) if neg else [float("inf")]
    dveto = sorted(dist(x, v) for v in veto) if veto else [float("inf")]
    nearest_pos = dpos[0]
    nearest_neg = dneg[0]
    nearest_veto = dveto[0]
    support = sum(1 for d in dpos if d <= sf(model.get("radius"), 0.0))
    neg_ok = nearest_neg >= sf(spec.get("negative_margin"), 1.25) * max(nearest_pos, 1e-12)
    support_ok = support >= min_support
    veto_margin = sf(spec.get("veto_margin"), 1.0)
    veto_ok = nearest_veto >= veto_margin * max(nearest_pos, 1e-12)
    choose = bool(nearest_pos < nearest_neg and neg_ok and support_ok and veto_ok)
    return (10 if choose else 15), {
        "nearest_pos": nearest_pos,
        "nearest_neg": nearest_neg,
        "nearest_veto": nearest_veto,
        "support": support,
        "radius": model.get("radius"),
        "support_ok": support_ok,
        "neg_ok": neg_ok,
        "veto_ok": veto_ok,
        "reason": "h10" if choose else "abstain_h15",
    }


def evaluate(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], profile: str) -> Dict[str, Any]:
    prof_rows = [r for r in rows if r.get("terminal_profile") == profile]
    fixed_phys = sum(sf(r.get("h15_physical"), 0.0) for r in prof_rows)
    fixed_dec = sum(sf(r.get("h15_decision_sum_s"), 0.0) for r in prof_rows)
    fixed_sol = sum(sf(r.get("h15_solver_sum_s"), 0.0) for r in prof_rows)
    tol = sum(sf(r.get("row_physical_tolerance"), 2.0) for r in prof_rows)
    policy_phys = 0.0; policy_dec = 0.0; policy_sol = 0.0
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    chosen_counts = {"10": 0, "15": 0}
    details: List[Dict[str, Any]] = []
    false_positive_rows: List[Dict[str, Any]] = []
    catastrophic_rows: List[Dict[str, Any]] = []
    unsafe_rows: List[Dict[str, Any]] = []
    for r in prof_rows:
        pred, diag = predict(model, r["feature"])
        chosen_counts[str(pred)] += 1
        positive = bool(r.get("h10_beneficial_vs_h15"))
        if pred == 10 and positive:
            counts["TP"] += 1
        elif pred == 10 and not positive:
            counts["FP"] += 1
        elif pred != 10 and positive:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
        if pred == 10:
            policy_phys += sf(r.get("h10_physical"), 0.0)
            policy_dec += sf(r.get("h10_decision_sum_s"), 0.0)
            policy_sol += sf(r.get("h10_solver_sum_s"), 0.0)
            if not bool(r.get("h10_safe_all")):
                unsafe_rows.append({"base_state_id": r.get("base_state_id"), "reason": "predicted H10 not safe_all"})
        else:
            policy_phys += sf(r.get("h15_physical"), 0.0)
            policy_dec += sf(r.get("h15_decision_sum_s"), 0.0)
            policy_sol += sf(r.get("h15_solver_sum_s"), 0.0)
        phys_delta_row = sf(r.get("h10_physical"), 0.0) - sf(r.get("h15_physical"), 0.0)
        if pred == 10 and not positive:
            fp = {"base_state_id": r.get("base_state_id"), "group": r.get("fresh_confirmation_group"), "phys_delta_h10_minus_h15": phys_delta_row, "decision_delta_h10_minus_h15": sf(r.get("h10_decision_sum_s"), 0.0) - sf(r.get("h15_decision_sum_s"), 0.0), "row_tolerance": sf(r.get("row_physical_tolerance"), 2.0)}
            false_positive_rows.append(fp)
            if phys_delta_row > sf(r.get("row_physical_tolerance"), 2.0):
                catastrophic_rows.append(fp)
        details.append({"base_state_id": r.get("base_state_id"), "group": r.get("fresh_confirmation_group"), "pred": pred, "h10_beneficial": positive, "phys_delta_h10_minus_h15": phys_delta_row, "decision_delta_h10_minus_h15": sf(r.get("h10_decision_sum_s"), 0.0) - sf(r.get("h15_decision_sum_s"), 0.0), "risk_score": r.get("stage_a_trace_risk_score"), "diag": diag})
    phys_delta = policy_phys - fixed_phys
    decision_saving = 1.0 - policy_dec / fixed_dec if fixed_dec > 0 else 0.0
    solver_saving = 1.0 - policy_sol / fixed_sol if fixed_sol > 0 else 0.0
    return {
        "profile": profile,
        "groups": len(prof_rows),
        "chosen_counts": chosen_counts,
        "confusion": counts,
        "fixed_H15": {"physical_sum": fixed_phys, "decision_sum_s": fixed_dec, "solver_sum_s": fixed_sol},
        "policy": {"physical_sum": policy_phys, "decision_sum_s": policy_dec, "solver_sum_s": policy_sol},
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol,
        "physical_gate": phys_delta <= tol,
        "decision_relative_saving_vs_fixed_H15": decision_saving,
        "solver_relative_saving_vs_fixed_H15": solver_saving,
        "decision_saving_gate_5pct": decision_saving >= 0.05,
        "decision_saving_gate_10pct": decision_saving >= 0.10,
        "false_positive_rows": false_positive_rows,
        "catastrophic_false_positive_rows": catastrophic_rows,
        "unsafe_rows": unsafe_rows,
        "no_catastrophic_fp": not catastrophic_rows and not unsafe_rows,
        "pass_5pct_no_cat_fp": bool((not catastrophic_rows) and (not unsafe_rows) and phys_delta <= tol and decision_saving >= 0.05),
        "pass_10pct_no_cat_fp": bool((not catastrophic_rows) and (not unsafe_rows) and phys_delta <= tol and decision_saving >= 0.10),
        "details": details,
    }


def nearest_source_table(model: Mapping[str, Any], examples: Sequence[Mapping[str, Any]], fresh_row: Mapping[str, Any], n: int = 8) -> List[Dict[str, Any]]:
    mode = str(model["mode"])
    x = transform(fresh_row["feature"], mode)
    rows = []
    for e in examples:
        rows.append({"distance": dist(x, transform(e["feature"], mode)), "source": e.get("source"), "state_id": e.get("state_id"), "category": e.get("category"), "labels_by_profile": e.get("labels_by_profile")})
    return sorted(rows, key=lambda z: z["distance"])[:n]


def append_docs(text: str) -> None:
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"]:
        path = ROOT / name
        prior = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(prior.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists() and MARKER not in reg.read_text(encoding="utf-8", errors="ignore"):
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"\n{STAMP},{NAME},development_guard_veto_diagnostic,0,0,0,0,{rel(OUT_DIR / 'summary.md')},{MARKER}\n")


def write_summary(raw: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append(f"# Vehicle true-variable-H guard/veto diagnostic v0")
    lines.append("")
    lines.append(f"UTC: `{raw['created_utc']}`. No new simulations, no validation64, no sealed test, no training/refit.")
    lines.append("")
    lines.append("## Source-label anatomy")
    src = raw["source_summary"]
    lines.append(f"- Source examples with H15 features: `{src['total']}`; agreement H10 positives `{src['agreement_positive_h10']}`, agreement non-H10 `{src['agreement_non_h10']}`, terminal-disagreement/missing veto candidates `{src['terminal_disagreement_or_missing']}`.")
    lines.append("- This directly tests whether the previous terminal-agreement-only rule was too positive-heavy, not whether all learning prerequisites are absent.")
    lines.append("")
    lines.append("## Candidate guard variants on fresh shared-H15 terminal rows")
    lines.append("")
    lines.append("| variant | H10 preds | TP/FP/FN/TN | phys Δ / tol | decision saving | solver saving | no catastrophic FP | pass >=5% | pass >=10% |")
    lines.append("|---|---:|---|---:|---:|---:|---|---|---|")
    for row in raw["variant_table_shared_h15"]:
        lines.append(f"| `{row['variant_id']}` | {row['chosen_counts']['10']} | {row['confusion']} | {row['physical_delta_vs_fixed_H15']:.4g} / {row['physical_tolerance_sum']:.4g} | {row['decision_relative_saving_vs_fixed_H15']:.3f} | {row['solver_relative_saving_vs_fixed_H15']:.3f} | `{row['no_catastrophic_fp']}` | `{row['pass_5pct_no_cat_fp']}` | `{row['pass_10pct_no_cat_fp']}` |")
    lines.append("")
    if raw["passing_variants_shared_h15_5pct"]:
        lines.append("Passing development candidates (>=5%, no catastrophic FP) exist, but are outcome-informed and require a new frozen confirmation block before validation:")
        for p in raw["passing_variants_shared_h15_5pct"][:5]:
            lines.append(f"- `{p['variant_id']}`: H10 `{p['chosen_counts']['10']}`, decision saving `{p['decision_relative_saving_vs_fixed_H15']:.3f}`, phys Δ `{p['physical_delta_vs_fixed_H15']:.4g}`.")
    else:
        lines.append("No tested source-label veto/representation variant achieved the >=5% no-catastrophic-FP fresh development gate; this favors representation/value/objective or scenario-design intervention over another selector threshold sweep.")
    lines.append("")
    lines.append("## Baseline false-positive anatomy")
    for fp in raw["baseline_false_positive_anatomy"]:
        lines.append(f"- `{fp['base_state_id']}` ({fp['terminal_profile']}): H10-H15 physical Δ `{fp['phys_delta_h10_minus_h15']:.4g}`, decision Δ `{fp['decision_delta_h10_minus_h15']:.4g}`, risk score `{fp['risk_score']:.3f}`.")
        lines.append("  Nearest source examples under baseline raw representation:")
        for n in fp["nearest_source_examples"][:5]:
            lines.append(f"  - d={n['distance']:.3f} `{n['source']}:{n['state_id']}` {n['category']} labels={n['labels_by_profile']}")
    lines.append("")
    lines.append("## Decision")
    lines.append(raw["decision"])
    lines.append("")
    lines.append(f"Artifacts: `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--i-accept-development-guard-veto-diagnostic-v0", action="store_true", required=True)
    args = ap.parse_args()
    if not args.run or not args.i_accept_development_guard_veto_diagnostic_v0:
        raise RuntimeError("explicit development diagnostic acknowledgement required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed_guard(FRESH_DONE)
    fresh_raw = read_json(FRESH_RAW)
    if fresh_raw.get("validation64_bank_opened") is not False or fresh_raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("fresh raw access flags invalid")
    source_examples = load_source_examples(ORACLE_RAW, "oracle_bank") + load_source_examples(RISK_RAW, "risk_anchor")
    fresh_rows = build_fresh_rows(fresh_raw)
    specs: List[Dict[str, Any]] = []
    for mode in ["raw", "raw_abs_l2", "pose_goal_obs_l2", "obstacle_slice_l2"]:
        specs.append({"variant_id": f"baseline_current_{mode}_s2_agreementNeg_m1.25", "mode": mode, "min_positive_support": 2, "positive_radius_quantile": 0.5, "positive_radius_scale": 1.25, "negative_margin": 1.25, "negative_pool": "agreement_only", "veto_pool": "none"})
        for margin in [1.0, 1.25, 1.5]:
            specs.append({"variant_id": f"recall_s1_{mode}_agreementNeg_m{margin}", "mode": mode, "min_positive_support": 1, "positive_radius_quantile": 0.5, "positive_radius_scale": 1.25, "negative_margin": margin, "negative_pool": "agreement_only", "veto_pool": "none"})
            specs.append({"variant_id": f"disagreementAsNegative_s1_{mode}_m{margin}", "mode": mode, "min_positive_support": 1, "positive_radius_quantile": 0.5, "positive_radius_scale": 1.25, "negative_margin": margin, "negative_pool": "all_nonpositive_including_disagreement", "veto_pool": "none"})
            specs.append({"variant_id": f"disagreementVeto_s1_{mode}_m{margin}_v1.0", "mode": mode, "min_positive_support": 1, "positive_radius_quantile": 0.5, "positive_radius_scale": 1.25, "negative_margin": margin, "negative_pool": "agreement_only", "veto_pool": "disagreement_only", "veto_margin": 1.0})
            specs.append({"variant_id": f"allNonposVeto_s1_{mode}_m{margin}_v1.0", "mode": mode, "min_positive_support": 1, "positive_radius_quantile": 0.5, "positive_radius_scale": 1.25, "negative_margin": margin, "negative_pool": "agreement_only", "veto_pool": "all_nonpositive_including_disagreement", "veto_margin": 1.0})
    variant_results: List[Dict[str, Any]] = []
    baseline_model: Optional[Mapping[str, Any]] = None
    for spec in specs:
        model = fit_predictor(source_examples, spec)
        if spec["variant_id"] == "baseline_current_raw_s2_agreementNeg_m1.25":
            baseline_model = model
        shared = evaluate(model, fresh_rows, PRIMARY_PROFILE)
        matched = evaluate(model, fresh_rows, "matched_terminal")
        shared_public = {k: v for k, v in shared.items() if k != "details"}
        matched_public = {k: v for k, v in matched.items() if k != "details"}
        variant_results.append({"variant_id": spec["variant_id"], "spec": spec, "model_counts": model["counts"], "radius": model["radius"], "shared_h15": shared_public, "matched_terminal": matched_public, "shared_details": shared["details"], "matched_details": matched["details"]})
    variant_results.sort(key=lambda r: (not r["shared_h15"]["pass_10pct_no_cat_fp"], not r["shared_h15"]["pass_5pct_no_cat_fp"], -sf(r["shared_h15"]["decision_relative_saving_vs_fixed_H15"]), r["shared_h15"]["chosen_counts"]["10"], r["variant_id"]))
    passing5 = [dict(v["shared_h15"], variant_id=v["variant_id"], spec=v["spec"]) for v in variant_results if v["shared_h15"]["pass_5pct_no_cat_fp"]]
    passing10 = [dict(v["shared_h15"], variant_id=v["variant_id"], spec=v["spec"]) for v in variant_results if v["shared_h15"]["pass_10pct_no_cat_fp"]]
    table = [dict(v["shared_h15"], variant_id=v["variant_id"]) for v in variant_results[:18]]
    # Ensure the original baseline appears even if not in the top table.
    baseline_res = next(v for v in variant_results if v["variant_id"] == "baseline_current_raw_s2_agreementNeg_m1.25")
    if all(t["variant_id"] != baseline_res["variant_id"] for t in table):
        table.append(dict(baseline_res["shared_h15"], variant_id=baseline_res["variant_id"]))
    if baseline_model is None:
        baseline_model = fit_predictor(source_examples, {"variant_id": "baseline_current_raw_s2_agreementNeg_m1.25", "mode": "raw", "min_positive_support": 2, "positive_radius_quantile": 0.5, "positive_radius_scale": 1.25, "negative_margin": 1.25, "negative_pool": "agreement_only", "veto_pool": "none"})
    baseline_eval = evaluate(baseline_model, fresh_rows, PRIMARY_PROFILE)
    baseline_fp_anatomy = []
    for d in baseline_eval["details"]:
        if d["pred"] == 10 and not d["h10_beneficial"]:
            fr = next(r for r in fresh_rows if r["base_state_id"] == d["base_state_id"] and r["terminal_profile"] == PRIMARY_PROFILE)
            baseline_fp_anatomy.append({
                "base_state_id": d["base_state_id"],
                "terminal_profile": PRIMARY_PROFILE,
                "phys_delta_h10_minus_h15": d["phys_delta_h10_minus_h15"],
                "decision_delta_h10_minus_h15": d["decision_delta_h10_minus_h15"],
                "risk_score": d["risk_score"],
                "baseline_diag": d["diag"],
                "nearest_source_examples": nearest_source_table(baseline_model, source_examples, fr, n=8),
            })
    source_counts: Dict[str, int] = {"agreement_positive_h10": 0, "agreement_non_h10": 0, "terminal_disagreement_or_missing": 0}
    for e in source_examples:
        source_counts[e["category"]] = source_counts.get(e["category"], 0) + 1
    if passing10:
        decision = "Exploratory source-label veto/representation variants can exceed the strong 10% fresh development gate without catastrophic H10 false positives. Because this is outcome-informed, freeze a new fresh-source confirmation block for the top simple variant before any validation64/test access."
    elif passing5:
        decision = "A source-label veto/representation repair can meet the weak 5% fresh development gate without catastrophic H10 false positives, but not strong 10%. Freeze a small independent fresh-source confirmation for the simplest passing variant; if it fails, move to value/objective/representation refit rather than another label-density sweep."
    else:
        decision = "No tested source-label veto/representation repair meets the weak fresh development gate. The bottleneck is not just terminal-disagreement exclusion; prioritize a bounded terminal-value/objective/representation calibration or value-refit/training ablation before further selector rollout."
    created = now_utc()
    backup_req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_GUARD_VETO_DIAGNOSTIC_V0_{created.strftime('%Y%m%dT%H%M%S.%f%z')}.json"
    write_json(backup_req, {"request": "backup_after_no_simulation_guard_veto_diagnostic", "created_utc": created.isoformat(), "paths": [rel(Path(__file__)), rel(OUT_DIR)], "required_before_next_simulation_training_or_refit": True, "sealed_test_accessed": False, "validation64_bank_opened": False})
    raw_out: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_no_simulation_guard_representation_veto_diagnostic_not_validation_not_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "inputs": {"fresh_raw": rel(FRESH_RAW), "fresh_completed": rel(FRESH_DONE), "oracle_raw": rel(ORACLE_RAW), "risk_raw": rel(RISK_RAW)},
        "source_summary": {"total": len(source_examples), **source_counts},
        "variant_results": variant_results,
        "variant_table_shared_h15": table,
        "passing_variants_shared_h15_5pct": passing5,
        "passing_variants_shared_h15_10pct": passing10,
        "baseline_false_positive_anatomy": baseline_fp_anatomy,
        "decision": decision,
        "backup_request": rel(backup_req),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_env": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
    }
    write_json(OUT_DIR / "raw.json", raw_out)
    write_summary(raw_out)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text((OUT_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(f"""# Continue state after guard/veto diagnostic v0

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.

Completed no-simulation diagnostic `{NAME}` using fresh-source branch evidence and source oracle/risk banks. No validation64, no sealed test, no new simulations, no training/refit.

Decision: {decision}

Passing weak variants: {[p['variant_id'] for p in passing5[:5]]}
Passing strong variants: {[p['variant_id'] for p in passing10[:5]]}

Next action: backup this new diagnostic source/artifacts before any new simulation/training/refit. If a passing variant exists, freeze a new independent fresh-source confirmation for the simplest passing variant before validation; otherwise freeze a bounded value/objective/representation refit/calibration diagnostic with explicit budget and baseline.

Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
Backup request: `{rel(backup_req)}`.
""", encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H guard/veto diagnostic v0

UTC: {created.isoformat()}. No-simulation development diagnostic after fresh-source confirmation failure. Source labels: {source_counts}; total {len(source_examples)} examples. Passing weak fresh shared-H15 variants: {[p['variant_id'] for p in passing5[:3]]}; passing strong variants: {[p['variant_id'] for p in passing10[:3]]}. Decision: {decision}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
""")
    files = [Path(__file__), FRESH_DONE, FRESH_RAW, ORACLE_RAW, RISK_RAW, OUT_DIR / "summary.md", OUT_DIR / "raw.json", STATE_FILE, CONTINUE_STATE, backup_req]
    write_json(OUT_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "classification": raw_out["classification"], "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "decision": decision, "passing_variant_count_5pct": len(passing5), "passing_variant_count_10pct": len(passing10), "backup_request": rel(backup_req), "hashes": {rel(p): sha256(p) for p in files if p.exists()}})
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "passing_5pct": len(passing5), "passing_10pct": len(passing10), "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
