#!/usr/bin/env python3
"""Development-only failure-aware selector refit over opened fresh-source banks v0/v1/v2.

Purpose
-------
The v2 source-only nearest-neighbour/veto selector achieved measured compute
savings on an unused fresh-source block, but failed the physical gate through two
large H10 false positives.  This script is the next bounded diagnostic requested
by the research contract: use ONLY already opened development banks (fresh_v0,
fresh_v1, fresh_v2) to test whether pre-decision features/objectives can separate
catastrophic H10 failures while retaining measured whole-decision savings.

It performs no simulations, no MPC rollouts, no gradient training/refit, no
validation64 access and no sealed-test access.  It is development-contaminated by
v2 outcomes; any positive selector from this script requires a separately frozen
unused fresh-source confirmation before validation64/test use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_failure_aware_selector_refit_v3_lobo"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
PRIMARY_PROFILE = "shared_h15_terminal"
MATCHED_PROFILE = "matched_terminal"
TERMINAL_PROFILES = [MATCHED_PROFILE, PRIMARY_PROFILE]
MARKER = f"vehicle-true-variable-H-failure-aware-selector-refit-v3-lobo-{STAMP}"

BANK_DIRS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z",
}
GROUPS = [
    "fresh_high_heading_long_or_medium",
    "fresh_high_heading_short",
    "fresh_low_heading_low_clearance_control",
    "fresh_lower_stress_control",
]
WINDOWS = ["early_mid", "mid_late"]
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_FILE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_failure_aware_selector_refit_v3_lobo.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def l2(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def normalize_l2(vec: Sequence[float]) -> List[float]:
    vals = [sf(v, 0.0) for v in vec]
    norm = math.sqrt(sum(v * v for v in vals))
    return [v / norm for v in vals] if norm > 1e-12 else vals


def obs14_from_manifest(st: Mapping[str, Any]) -> List[float]:
    obs = st.get("initial_observation_from_h15_trace")
    vals: List[float] = []
    if isinstance(obs, list):
        vals = [sf(x, 0.0) for x in obs[:14]]
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def median_summary(medians: Mapping[str, Any], horizon: int) -> Mapping[str, Any]:
    return medians.get(str(horizon)) or medians.get(horizon) or {}


def assert_dev_only(raw: Mapping[str, Any], completed: Mapping[str, Any], bank_id: str) -> None:
    for obj_name, obj in [("raw", raw), ("completed", completed)]:
        if obj.get("sealed_test_accessed") is True or obj.get("sealed_test_bank_opened") is True:
            raise ContractError(f"{bank_id} {obj_name} explicitly accessed sealed test")
        if obj.get("validation64_bank_opened") is True:
            raise ContractError(f"{bank_id} {obj_name} explicitly opened validation64")
    if completed.get("passed") is not True and completed.get("hard_pass") is not True:
        raise ContractError(f"{bank_id} did not complete cleanly")


def load_manifest(raw: Mapping[str, Any], manifest_path: Path) -> List[Mapping[str, Any]]:
    man = raw.get("selected_state_manifest")
    if isinstance(man, Mapping) and isinstance(man.get("selected_states"), list):
        return list(man.get("selected_states") or [])
    if isinstance(man, list):
        return list(man)
    if manifest_path.exists():
        fb = read_json(manifest_path)
        if isinstance(fb, Mapping):
            return list(fb.get("selected_states") or [])
        if isinstance(fb, list):
            return list(fb)
    raise ContractError("cannot locate selected_state_manifest for " + rel(manifest_path))


def load_bank(bank_id: str, bank_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    raw_path = bank_dir / "raw.json"
    done_path = bank_dir / "completed.json"
    man_path = bank_dir / "selected_state_manifest.json"
    if not raw_path.exists() or not done_path.exists():
        raise ContractError(f"missing bank artifacts for {bank_id}: {rel(bank_dir)}")
    raw = read_json(raw_path)
    done = read_json(done_path)
    assert_dev_only(raw, done, bank_id)
    states = load_manifest(raw, man_path)
    by_base = {str(s.get("base_state_id")): s for s in states}
    rows: List[Dict[str, Any]] = []
    for r in ((raw.get("analysis") or {}).get("state_profile_rows") or []):
        base = str(r.get("base_state_id") or r.get("state_id"))
        profile = str(r.get("terminal_profile"))
        if profile not in TERMINAL_PROFILES:
            continue
        st = by_base.get(base, {})
        med = r.get("median_by_h") or {}
        h10 = median_summary(med, 10)
        h15 = median_summary(med, 15)
        h10_phys = sf(h10.get("physical"), 0.0)
        h15_phys = sf(h15.get("physical"), 0.0)
        h10_dec = sf(h10.get("decision_sum_s"), 0.0)
        h15_dec = sf(h15.get("decision_sum_s"), 0.0)
        h10_sol = sf(h10.get("solver_sum_s"), 0.0)
        h15_sol = sf(h15.get("solver_sum_s"), 0.0)
        row_tol = sf(r.get("row_physical_tolerance_vs_H15", r.get("row_physical_tolerance")), max(2.0, 0.05 * abs(h15_phys)))
        row_tol = max(2.0, row_tol)
        h10_safe = bool(h10.get("safe_all"))
        h15_safe = bool(h15.get("safe_all"))
        if "h10_beneficial_vs_h15" in r:
            beneficial = bool(r.get("h10_beneficial_vs_h15"))
        else:
            beneficial = bool(h15_safe and h10_safe and (h10_phys - h15_phys <= row_tol) and (h10_dec < h15_dec))
        catastrophic = bool((not h10_safe) or (h15_safe and (h10_phys - h15_phys > row_tol)))
        obs = obs14_from_manifest(st)
        prev = st.get("branch_previous_state") if isinstance(st.get("branch_previous_state"), Mapping) else {}
        window = str(st.get("window") or ("early_mid" if si(r.get("branch_state_slot", st.get("branch_state_slot")), 0) == 0 else "mid_late"))
        group = str(r.get("fresh_confirmation_group") or st.get("fresh_confirmation_group") or "unknown")
        rows.append({
            "bank_id": bank_id,
            "base_state_id": base,
            "terminal_profile": profile,
            "group": group,
            "window": window,
            "source_candidate_index": st.get("source_candidate_index"),
            "branch_state_slot": si(r.get("branch_state_slot", st.get("branch_state_slot")), -1),
            "branch_step": si(st.get("branch_step"), -1),
            "stage_a_trace_risk_score": sf(st.get("stage_a_trace_risk_score"), 0.0),
            "obs14": obs,
            "prev_x": sf(prev.get("x"), 0.0),
            "prev_y": sf(prev.get("y"), 0.0),
            "prev_theta": sf(prev.get("theta"), 0.0),
            "h10_physical": h10_phys,
            "h15_physical": h15_phys,
            "row_physical_tolerance": row_tol,
            "h10_decision_sum_s": h10_dec,
            "h15_decision_sum_s": h15_dec,
            "h10_solver_sum_s": h10_sol,
            "h15_solver_sum_s": h15_sol,
            "h10_safe_all": h10_safe,
            "h15_safe_all": h15_safe,
            "h10_beneficial_vs_h15": beneficial,
            "h10_catastrophic_vs_h15": catastrophic,
            "phys_delta_h10_minus_h15": h10_phys - h15_phys,
            "decision_delta_h10_minus_h15": h10_dec - h15_dec,
            "solver_delta_h10_minus_h15": h10_sol - h15_sol,
            "oracle_label": r.get("oracle_label"),
            "prior_policy_prediction": r.get("primary_policy_prediction"),
        })
    hashes = {rel(raw_path): sha256(raw_path), rel(done_path): sha256(done_path)}
    if man_path.exists():
        hashes[rel(man_path)] = sha256(man_path)
    return rows, hashes


def grouped_by_base(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]]:
    out: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        out[(str(r.get("bank_id")), str(r.get("base_state_id")))][str(r.get("terminal_profile"))] = r
    return out


def label_for_pair(primary: Mapping[str, Any], matched: Optional[Mapping[str, Any]], mode: str) -> str:
    p_pos = bool(primary.get("h10_beneficial_vs_h15"))
    p_cat = bool(primary.get("h10_catastrophic_vs_h15"))
    m_pos = bool(matched.get("h10_beneficial_vs_h15")) if matched is not None else p_pos
    m_cat = bool(matched.get("h10_catastrophic_vs_h15")) if matched is not None else False
    disagreement = (matched is not None) and (p_pos != m_pos)
    if mode == "primary_only":
        return "positive" if p_pos else ("catastrophic_veto" if p_cat else "negative")
    if mode == "primary_plus_matched_cat_veto":
        if p_cat or m_cat:
            return "catastrophic_veto"
        return "positive" if p_pos else "negative"
    if mode == "terminal_consensus_positive":
        if p_cat or m_cat or disagreement:
            return "catastrophic_veto"
        return "positive" if (p_pos and m_pos) else "negative"
    if mode == "low_regret_positive_primary":
        phys_delta = sf(primary.get("phys_delta_h10_minus_h15"), 0.0)
        # Low-regret positives are those with a strict physical margin, not merely
        # within the broad per-row tolerance.  This intentionally sacrifices recall
        # to test whether objective/label risk is the bottleneck.
        low_regret = p_pos and (phys_delta <= min(0.5 * sf(primary.get("row_physical_tolerance"), 2.0), 0.75))
        if p_cat or m_cat:
            return "catastrophic_veto"
        return "positive" if low_regret else "negative"
    raise ContractError("unknown label mode: " + mode)


def one_hot(value: str, choices: Sequence[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def raw_feature(row: Mapping[str, Any], representation: str) -> List[float]:
    obs = [sf(v, 0.0) for v in (row.get("obs14") or [])[:14]]
    while len(obs) < 14:
        obs.append(0.0)
    state = [sf(row.get("prev_x"), 0.0) / 30.0, sf(row.get("prev_y"), 0.0) / 30.0, sf(row.get("prev_theta"), 0.0) / math.pi]
    step = max(0.0, sf(row.get("branch_step"), 0.0)) / 150.0
    slot = max(0.0, sf(row.get("branch_state_slot"), 0.0))
    risk = math.log1p(max(0.0, sf(row.get("stage_a_trace_risk_score"), 0.0))) / math.log(31.0)
    if representation == "raw_abs_l2":
        return normalize_l2(obs + [abs(v) for v in obs])
    if representation == "pose_goal_obs_l2":
        return normalize_l2(obs[:5] + [abs(v) for v in obs[:5]])
    if representation == "obstacle_slice_l2":
        return normalize_l2(obs[5:14] + [abs(v) for v in obs[5:14]])
    if representation == "deploy_obs_state_step_risk_std":
        return obs + [abs(v) for v in obs] + state + [slot, step, risk]
    if representation == "deploy_pose_state_step_risk_std":
        return obs[:5] + [abs(v) for v in obs[:5]] + state + [slot, step, risk]
    if representation == "deploy_obs_state_noabs_std":
        return obs + state + [slot, step, risk]
    if representation == "diagnostic_group_window_std":
        return obs + [abs(v) for v in obs] + state + [slot, step, risk] + one_hot(str(row.get("group")), GROUPS) + one_hot(str(row.get("window")), WINDOWS)
    raise ContractError("unknown representation: " + representation)


def fit_standardizer(vectors: Sequence[Sequence[float]], representation: str) -> Tuple[List[float], List[float]]:
    if not representation.endswith("_std"):
        return [], []
    if not vectors:
        return [], []
    n = max(len(v) for v in vectors)
    mat = [[sf(v[i], 0.0) if i < len(v) else 0.0 for i in range(n)] for v in vectors]
    means: List[float] = []
    stds: List[float] = []
    for j in range(n):
        col = [row[j] for row in mat]
        mu = sum(col) / len(col)
        var = sum((x - mu) ** 2 for x in col) / max(1, len(col) - 1)
        sd = math.sqrt(var)
        means.append(mu)
        stds.append(sd if sd > 1e-9 else 1.0)
    return means, stds


def apply_standardizer(vec: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    if not means:
        return [sf(v, 0.0) for v in vec]
    n = len(means)
    vals = [sf(vec[i], 0.0) if i < len(vec) else 0.0 for i in range(n)]
    return [(vals[i] - means[i]) / stds[i] for i in range(n)]


def build_examples(rows: Sequence[Mapping[str, Any]], train_banks: Sequence[str], label_mode: str) -> List[Dict[str, Any]]:
    by = grouped_by_base([r for r in rows if str(r.get("bank_id")) in set(train_banks)])
    examples: List[Dict[str, Any]] = []
    for (bank, base), profs in sorted(by.items()):
        primary = profs.get(PRIMARY_PROFILE)
        if primary is None:
            continue
        matched = profs.get(MATCHED_PROFILE)
        cat = label_for_pair(primary, matched, label_mode)
        ex = dict(primary)
        ex["training_category"] = cat
        ex["label_mode"] = label_mode
        ex["matched_catastrophic"] = bool(matched.get("h10_catastrophic_vs_h15")) if matched is not None else False
        ex["matched_beneficial"] = bool(matched.get("h10_beneficial_vs_h15")) if matched is not None else None
        examples.append(ex)
    return examples


def candidate_specs() -> List[Dict[str, Any]]:
    reps = [
        ("raw_abs_l2", True),
        ("pose_goal_obs_l2", True),
        ("obstacle_slice_l2", True),
        ("deploy_obs_state_step_risk_std", True),
        ("deploy_pose_state_step_risk_std", True),
        ("deploy_obs_state_noabs_std", True),
        ("diagnostic_group_window_std", False),
    ]
    label_modes = [
        "primary_only",
        "primary_plus_matched_cat_veto",
        "terminal_consensus_positive",
        "low_regret_positive_primary",
    ]
    pool_pairs = [
        ("all_nonpositive", "catastrophic_only"),
        ("all_nonpositive", "all_nonpositive"),
        ("catastrophic_only", "catastrophic_only"),
    ]
    specs: List[Dict[str, Any]] = []
    for rep, deployable in reps:
        for label_mode in label_modes:
            for negative_pool, veto_pool in pool_pairs:
                for support in [1, 2]:
                    for radius_q in [0.5, 0.75, 1.0]:
                        for radius_scale in [0.75, 1.0, 1.25, 1.5, 2.0]:
                            for neg_margin in [1.0, 1.25, 1.5, 2.0]:
                                for veto_margin in [0.75, 1.0, 1.25, 1.5, 2.0]:
                                    specs.append({
                                        "representation": rep,
                                        "deployable_features_only": deployable,
                                        "label_mode": label_mode,
                                        "negative_pool": negative_pool,
                                        "veto_pool": veto_pool,
                                        "min_positive_support": support,
                                        "positive_radius_quantile": radius_q,
                                        "positive_radius_scale": radius_scale,
                                        "negative_margin": neg_margin,
                                        "veto_margin": veto_margin,
                                        "variant_id": f"v3_{rep}_{label_mode}_neg{negative_pool}_veto{veto_pool}_s{support}_q{radius_q:g}_rs{radius_scale:g}_nm{neg_margin:g}_vm{veto_margin:g}",
                                    })
    return specs


def select_pool(examples: Sequence[Mapping[str, Any]], pool: str) -> List[Mapping[str, Any]]:
    if pool == "all_nonpositive":
        return [e for e in examples if e.get("training_category") != "positive"]
    if pool == "catastrophic_only":
        return [e for e in examples if e.get("training_category") == "catastrophic_veto"]
    if pool == "none":
        return []
    raise ContractError("unknown pool: " + pool)


def fit_model(examples: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> Dict[str, Any]:
    rep = str(spec["representation"])
    raw_vectors = [raw_feature(e, rep) for e in examples]
    means, stds = fit_standardizer(raw_vectors, rep)

    def embed(row: Mapping[str, Any]) -> List[float]:
        return apply_standardizer(raw_feature(row, rep), means, stds)

    pos_examples = [e for e in examples if e.get("training_category") == "positive"]
    neg_examples = select_pool(examples, str(spec.get("negative_pool")))
    veto_examples = select_pool(examples, str(spec.get("veto_pool")))
    pos = [embed(e) for e in pos_examples]
    neg = [embed(e) for e in neg_examples]
    veto = [embed(e) for e in veto_examples]
    nn: List[float] = []
    for i, p in enumerate(pos):
        ds = [l2(p, q) for j, q in enumerate(pos) if i != j]
        if ds:
            nn.append(min(ds))
    if nn:
        radius = max(1e-9, quantile(nn, sf(spec.get("positive_radius_quantile"), 0.75)) * sf(spec.get("positive_radius_scale"), 1.0))
    elif pos:
        # Single-positive edge case: allow support=1 but make the radius finite from
        # positive-to-negative scale if possible.  This is rarely used with these banks.
        pn = [l2(pos[0], n) for n in neg]
        radius = max(1e-9, (min(pn) if pn else 1.0) * 0.5)
    else:
        radius = 0.0
    return {
        "spec": dict(spec),
        "means": means,
        "stds": stds,
        "pos": pos,
        "neg": neg,
        "veto": veto,
        "radius": radius,
        "counts": {"positive": len(pos), "negative": len(neg), "veto": len(veto), "examples": len(examples)},
    }


def predict(model: Mapping[str, Any], row: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
    spec = model["spec"]
    rep = str(spec["representation"])
    x = apply_standardizer(raw_feature(row, rep), model.get("means") or [], model.get("stds") or [])
    pos: List[List[float]] = list(model.get("pos") or [])
    neg: List[List[float]] = list(model.get("neg") or [])
    veto: List[List[float]] = list(model.get("veto") or [])
    min_support = si(spec.get("min_positive_support"), 1)
    if len(pos) < max(1, min_support):
        return 15, {"reason": "insufficient_positive_training", "counts": model.get("counts")}
    dpos = sorted(l2(x, p) for p in pos)
    dneg = sorted(l2(x, n) for n in neg) if neg else [float("inf")]
    dveto = sorted(l2(x, v) for v in veto) if veto else [float("inf")]
    nearest_pos = dpos[0]
    nearest_neg = dneg[0]
    nearest_veto = dveto[0]
    radius = sf(model.get("radius"), 0.0)
    support = sum(1 for d in dpos if d <= radius)
    support_ok = support >= min_support
    neg_ok = nearest_neg >= sf(spec.get("negative_margin"), 1.0) * max(nearest_pos, 1e-12)
    veto_ok = nearest_veto >= sf(spec.get("veto_margin"), 1.0) * max(nearest_pos, 1e-12)
    radius_ok = nearest_pos <= radius
    choose = bool(radius_ok and support_ok and neg_ok and veto_ok)
    return (10 if choose else 15), {
        "reason": "h10" if choose else "abstain_h15",
        "nearest_pos": nearest_pos,
        "nearest_neg": nearest_neg,
        "nearest_veto": nearest_veto,
        "support": support,
        "radius": radius,
        "radius_ok": radius_ok,
        "support_ok": support_ok,
        "neg_ok": neg_ok,
        "veto_ok": veto_ok,
        "counts": model.get("counts"),
    }


def evaluate_model(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], bank_id: str, profile: str) -> Dict[str, Any]:
    eval_rows = [r for r in rows if str(r.get("bank_id")) == bank_id and str(r.get("terminal_profile")) == profile]
    fixed_phys = sum(sf(r.get("h15_physical"), 0.0) for r in eval_rows)
    fixed_dec = sum(sf(r.get("h15_decision_sum_s"), 0.0) for r in eval_rows)
    fixed_sol = sum(sf(r.get("h15_solver_sum_s"), 0.0) for r in eval_rows)
    tol_sum = sum(sf(r.get("row_physical_tolerance"), 2.0) for r in eval_rows)
    policy_phys = 0.0
    policy_dec = 0.0
    policy_sol = 0.0
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    chosen_counts = {"10": 0, "15": 0}
    false_positive_rows: List[Dict[str, Any]] = []
    catastrophic_rows: List[Dict[str, Any]] = []
    unsafe_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for r in eval_rows:
        pred, diag = predict(model, r)
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
        phys_delta_row = sf(r.get("phys_delta_h10_minus_h15"), 0.0)
        if pred == 10 and not positive:
            fp = {
                "bank_id": bank_id,
                "base_state_id": r.get("base_state_id"),
                "group": r.get("group"),
                "window": r.get("window"),
                "phys_delta_h10_minus_h15": phys_delta_row,
                "decision_delta_h10_minus_h15": sf(r.get("decision_delta_h10_minus_h15"), 0.0),
                "row_tolerance": sf(r.get("row_physical_tolerance"), 2.0),
                "diagnostics": diag,
            }
            false_positive_rows.append(fp)
            if phys_delta_row > sf(r.get("row_physical_tolerance"), 2.0):
                catastrophic_rows.append(fp)
        details.append({
            "bank_id": bank_id,
            "base_state_id": r.get("base_state_id"),
            "group": r.get("group"),
            "window": r.get("window"),
            "pred": pred,
            "h10_beneficial": positive,
            "h10_catastrophic": bool(r.get("h10_catastrophic_vs_h15")),
            "phys_delta_h10_minus_h15": phys_delta_row,
            "decision_delta_h10_minus_h15": sf(r.get("decision_delta_h10_minus_h15"), 0.0),
            "diagnostics": diag,
        })
    phys_delta = policy_phys - fixed_phys
    decision_save = 1.0 - policy_dec / fixed_dec if fixed_dec > 0 else 0.0
    solver_save = 1.0 - policy_sol / fixed_sol if fixed_sol > 0 else 0.0
    no_bad = (not catastrophic_rows) and (not unsafe_rows)
    physical_gate = phys_delta <= tol_sum
    return {
        "bank_id": bank_id,
        "profile": profile,
        "groups": len(eval_rows),
        "chosen_counts": chosen_counts,
        "confusion": counts,
        "fixed_H15": {"physical_sum": fixed_phys, "decision_sum_s": fixed_dec, "solver_sum_s": fixed_sol},
        "policy": {"physical_sum": policy_phys, "decision_sum_s": policy_dec, "solver_sum_s": policy_sol},
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "physical_gate": physical_gate,
        "decision_relative_saving_vs_fixed_H15": decision_save,
        "solver_relative_saving_vs_fixed_H15": solver_save,
        "false_positive_rows": false_positive_rows,
        "catastrophic_false_positive_rows": catastrophic_rows,
        "unsafe_rows": unsafe_rows,
        "no_catastrophic_fp": no_bad,
        "pass_5pct_no_cat_fp": bool(no_bad and physical_gate and decision_save >= 0.05),
        "pass_10pct_no_cat_fp": bool(no_bad and physical_gate and decision_save >= 0.10),
        "details": details,
    }


def oracle_eval(rows: Sequence[Mapping[str, Any]], bank_id: str, profile: str) -> Dict[str, Any]:
    eval_rows = [r for r in rows if str(r.get("bank_id")) == bank_id and str(r.get("terminal_profile")) == profile]
    fixed_phys = sum(sf(r.get("h15_physical"), 0.0) for r in eval_rows)
    fixed_dec = sum(sf(r.get("h15_decision_sum_s"), 0.0) for r in eval_rows)
    fixed_sol = sum(sf(r.get("h15_solver_sum_s"), 0.0) for r in eval_rows)
    pol_phys = 0.0
    pol_dec = 0.0
    pol_sol = 0.0
    counts = {"10": 0, "15": 0}
    for r in eval_rows:
        if bool(r.get("h10_beneficial_vs_h15")):
            counts["10"] += 1
            pol_phys += sf(r.get("h10_physical"), 0.0)
            pol_dec += sf(r.get("h10_decision_sum_s"), 0.0)
            pol_sol += sf(r.get("h10_solver_sum_s"), 0.0)
        else:
            counts["15"] += 1
            pol_phys += sf(r.get("h15_physical"), 0.0)
            pol_dec += sf(r.get("h15_decision_sum_s"), 0.0)
            pol_sol += sf(r.get("h15_solver_sum_s"), 0.0)
    return {
        "bank_id": bank_id,
        "profile": profile,
        "chosen_counts": counts,
        "physical_delta_vs_fixed_H15": pol_phys - fixed_phys,
        "decision_relative_saving_vs_fixed_H15": (1.0 - pol_dec / fixed_dec) if fixed_dec > 0 else 0.0,
        "solver_relative_saving_vs_fixed_H15": (1.0 - pol_sol / fixed_sol) if fixed_sol > 0 else 0.0,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"banks": {}}
    for bank in sorted({str(r.get("bank_id")) for r in rows}):
        out["banks"][bank] = {}
        for prof in TERMINAL_PROFILES:
            rr = [r for r in rows if str(r.get("bank_id")) == bank and str(r.get("terminal_profile")) == prof]
            h15_dec = sum(sf(r.get("h15_decision_sum_s"), 0.0) for r in rr)
            out["banks"][bank][prof] = {
                "groups": len(rr),
                "h10_beneficial": sum(1 for r in rr if r.get("h10_beneficial_vs_h15")),
                "h10_catastrophic": sum(1 for r in rr if r.get("h10_catastrophic_vs_h15")),
                "all_h10_physical_delta": sum(sf(r.get("phys_delta_h10_minus_h15"), 0.0) for r in rr),
                "all_h10_decision_saving": (sum(sf(r.get("h15_decision_sum_s"), 0.0) - sf(r.get("h10_decision_sum_s"), 0.0) for r in rr) / h15_dec) if h15_dec > 0 else None,
                "oracle": oracle_eval(rows, bank, prof),
            }
    by_group_window: Dict[str, Dict[str, Any]] = {}
    for r in [x for x in rows if str(x.get("terminal_profile")) == PRIMARY_PROFILE]:
        key = f"{r.get('group')}::{r.get('window')}"
        rec = by_group_window.setdefault(key, {"n": 0, "beneficial": 0, "catastrophic": 0, "phys_deltas": [], "decision_deltas": [], "banks": defaultdict(int)})
        rec["n"] += 1
        rec["beneficial"] += 1 if r.get("h10_beneficial_vs_h15") else 0
        rec["catastrophic"] += 1 if r.get("h10_catastrophic_vs_h15") else 0
        rec["phys_deltas"].append(sf(r.get("phys_delta_h10_minus_h15"), 0.0))
        rec["decision_deltas"].append(sf(r.get("decision_delta_h10_minus_h15"), 0.0))
        rec["banks"][str(r.get("bank_id"))] += 1
    out["primary_group_window_summary"] = {}
    for key, rec in sorted(by_group_window.items()):
        out["primary_group_window_summary"][key] = {
            "n": rec["n"],
            "beneficial": rec["beneficial"],
            "catastrophic": rec["catastrophic"],
            "median_physical_delta_h10_minus_h15": median(rec["phys_deltas"]),
            "median_decision_delta_h10_minus_h15": median(rec["decision_deltas"]),
            "banks": dict(rec["banks"]),
        }
    return out


def feature_scalar(row: Mapping[str, Any], name: str) -> float:
    obs = [sf(v, 0.0) for v in (row.get("obs14") or [])]
    if name.startswith("obs_abs_"):
        idx = int(name.split("_")[-1])
        return abs(obs[idx]) if idx < len(obs) else 0.0
    if name.startswith("obs_"):
        idx = int(name.split("_")[-1])
        return obs[idx] if idx < len(obs) else 0.0
    if name == "risk_score":
        return sf(row.get("stage_a_trace_risk_score"), 0.0)
    if name == "branch_step":
        return sf(row.get("branch_step"), 0.0)
    if name == "branch_state_slot":
        return sf(row.get("branch_state_slot"), 0.0)
    if name == "abs_theta":
        return abs(sf(row.get("prev_theta"), 0.0))
    if name == "prev_x":
        return sf(row.get("prev_x"), 0.0)
    if name == "abs_prev_y":
        return abs(sf(row.get("prev_y"), 0.0))
    return 0.0


def univariate_cat_audit(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    primary = [r for r in rows if str(r.get("terminal_profile")) == PRIMARY_PROFILE]
    cats = [r for r in primary if r.get("h10_catastrophic_vs_h15")]
    positives = [r for r in primary if r.get("h10_beneficial_vs_h15")]
    names = ["risk_score", "branch_step", "branch_state_slot", "abs_theta", "prev_x", "abs_prev_y"] + [f"obs_{i}" for i in range(14)] + [f"obs_abs_{i}" for i in range(14)]
    audits: List[Dict[str, Any]] = []
    for name in names:
        vals = sorted({feature_scalar(r, name) for r in primary})
        if len(vals) <= 1:
            continue
        thresholds: List[float] = []
        for a, b in zip(vals[:-1], vals[1:]):
            thresholds.append(0.5 * (a + b))
        thresholds.extend([vals[0], vals[-1]])
        for direction in ["<=", ">="]:
            best: Optional[Dict[str, Any]] = None
            for thr in thresholds:
                def hit(row: Mapping[str, Any]) -> bool:
                    v = feature_scalar(row, name)
                    return v <= thr if direction == "<=" else v >= thr
                cat_hits = [r for r in cats if hit(r)]
                pos_hits = [r for r in positives if hit(r)]
                all_hits = [r for r in primary if hit(r)]
                rec = {
                    "feature": name,
                    "direction": direction,
                    "threshold": thr,
                    "catastrophic_covered": len(cat_hits),
                    "catastrophic_total": len(cats),
                    "positive_vetoed": len(pos_hits),
                    "positive_total": len(positives),
                    "all_rows_hit": len(all_hits),
                    "hit_rows": [f"{r.get('bank_id')}:{r.get('base_state_id')}" for r in all_hits[:12]],
                }
                score = (len(cat_hits), -len(pos_hits), -len(all_hits))
                if best is None or score > best["_score"]:
                    rec["_score"] = score
                    best = rec
            if best is not None:
                best.pop("_score", None)
                audits.append(best)
    # Prefer rules that cover many catastrophic rows while vetoing few positives.
    audits.sort(key=lambda r: (r["catastrophic_covered"], -r["positive_vetoed"], -r["all_rows_hit"]), reverse=True)
    return audits[:30]


def aggregate_candidate_result(spec: Mapping[str, Any], holdout_evals: Mapping[str, Any]) -> Dict[str, Any]:
    primary = [holdout_evals[b][PRIMARY_PROFILE] for b in sorted(holdout_evals)]
    matched = [holdout_evals[b][MATCHED_PROFILE] for b in sorted(holdout_evals)]
    primary_bad = sum(len(g.get("catastrophic_false_positive_rows") or []) + len(g.get("unsafe_rows") or []) for g in primary)
    matched_bad = sum(len(g.get("catastrophic_false_positive_rows") or []) + len(g.get("unsafe_rows") or []) for g in matched)
    primary_phys_fail = sum(0 if g.get("physical_gate") else 1 for g in primary)
    min_dec = min(sf(g.get("decision_relative_saving_vs_fixed_H15"), -999.0) for g in primary)
    avg_dec = sum(sf(g.get("decision_relative_saving_vs_fixed_H15"), 0.0) for g in primary) / max(1, len(primary))
    total_h10 = sum(si((g.get("chosen_counts") or {}).get("10"), 0) for g in primary)
    max_phys_delta = max(sf(g.get("physical_delta_vs_fixed_H15"), 0.0) for g in primary)
    all_primary_strong = all(bool(g.get("pass_10pct_no_cat_fp")) for g in primary)
    all_primary_weak = all(bool(g.get("pass_5pct_no_cat_fp")) for g in primary)
    all_matched_no_bad = matched_bad == 0
    return {
        "variant_id": spec["variant_id"],
        "spec": dict(spec),
        "deployable_features_only": bool(spec.get("deployable_features_only")),
        "primary_lobo_strong_pass": all_primary_strong,
        "primary_lobo_weak_pass": all_primary_weak,
        "primary_bad_count_total": primary_bad,
        "primary_physical_gate_fail_count": primary_phys_fail,
        "matched_bad_count_total": matched_bad,
        "matched_all_no_bad": all_matched_no_bad,
        "min_primary_decision_saving": min_dec,
        "avg_primary_decision_saving": avg_dec,
        "primary_h10_predictions_total": total_h10,
        "max_primary_physical_delta": max_phys_delta,
        "holdout_evaluations": holdout_evals,
        "sort_score": (all_primary_strong, all_primary_weak, -primary_bad, -primary_phys_fail, min_dec, avg_dec, -matched_bad, total_h10, -max_phys_delta, bool(spec.get("deployable_features_only"))),
    }


def run_refit(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    banks = sorted({str(r.get("bank_id")) for r in rows})
    specs = candidate_specs()
    ranked: List[Dict[str, Any]] = []
    for spec in specs:
        holdout_evals: Dict[str, Any] = {}
        for holdout in banks:
            train_banks = [b for b in banks if b != holdout]
            examples = build_examples(rows, train_banks, str(spec["label_mode"]))
            model = fit_model(examples, spec)
            holdout_evals[holdout] = {
                PRIMARY_PROFILE: evaluate_model(model, rows, holdout, PRIMARY_PROFILE),
                MATCHED_PROFILE: evaluate_model(model, rows, holdout, MATCHED_PROFILE),
                "train_counts": model.get("counts"),
            }
        ranked.append(aggregate_candidate_result(spec, holdout_evals))
    ranked.sort(key=lambda r: r["sort_score"], reverse=True)
    # sort_score contains booleans/tuples for internal ranking only; keep concise output.
    for r in ranked:
        r.pop("sort_score", None)
    return ranked


def minimal_eval_for_table(rec: Mapping[str, Any]) -> Dict[str, Any]:
    holds = rec.get("holdout_evaluations") or {}
    out: Dict[str, Any] = {}
    for bank, ev in holds.items():
        p = ev[PRIMARY_PROFILE]
        out[bank] = {
            "primary_counts": p.get("chosen_counts"),
            "primary_confusion": p.get("confusion"),
            "primary_decision_saving": p.get("decision_relative_saving_vs_fixed_H15"),
            "primary_physical_delta": p.get("physical_delta_vs_fixed_H15"),
            "primary_bad": len(p.get("catastrophic_false_positive_rows") or []) + len(p.get("unsafe_rows") or []),
        }
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    ranked = raw.get("ranked_candidates") or []
    top = ranked[0] if ranked else None
    top_deploy = next((r for r in ranked if r.get("deployable_features_only")), None)
    strong = [r for r in ranked if r.get("primary_lobo_strong_pass")]
    strong_deploy = [r for r in strong if r.get("deployable_features_only")]
    lines: List[str] = [
        "# Vehicle true-variable-H failure-aware selector refit v3 LOBO",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only opened-bank leave-one-bank-out refit/diagnostic; no simulations, no validation64, no sealed test, no gradient training.",
        "",
        "## Why this was run",
        "v2 fresh-source confirmation met measured timing thresholds but failed the physical gate through catastrophic H10 false positives. This diagnostic tests whether pre-decision features and a cost-aware/veto label objective can eliminate such false positives across opened fresh_v0/fresh_v1/fresh_v2 banks before deciding between another fresh-source confirmation and value/objective retraining.",
        "",
        "## Bank/opportunity summary",
        "",
    ]
    for bank, summary in raw["row_summary"]["banks"].items():
        p = summary[PRIMARY_PROFILE]
        lines.append(f"- `{bank}` primary shared-H15: groups `{p['groups']}`, H10-beneficial `{p['h10_beneficial']}`, H10-catastrophic `{p['h10_catastrophic']}`, all-H10 physical Δ `{p['all_h10_physical_delta']:.4g}`, all-H10 decision saving `{p['all_h10_decision_saving']:.3f}`, oracle decision saving `{p['oracle']['decision_relative_saving_vs_fixed_H15']:.3f}`.")
    lines += ["", "Primary group/window anatomy:"]
    for key, rec in raw["row_summary"]["primary_group_window_summary"].items():
        lines.append(f"- `{key}`: n `{rec['n']}`, beneficial `{rec['beneficial']}`, catastrophic `{rec['catastrophic']}`, median phys Δ `{rec['median_physical_delta_h10_minus_h15']:.4g}`, median decision Δ `{rec['median_decision_delta_h10_minus_h15']:.4g}`.")
    lines += ["", "## Top LOBO selector candidates", "", "| rank | deployable | variant | strong | weak | bad | phys-gate-fail | matched bad | min dec save | avg dec save | H10 total | max phys Δ |", "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(ranked[:20], 1):
        lines.append("| %d | `%s` | `%s` | `%s` | `%s` | %d | %d | %d | %.3f | %.3f | %d | %.4g |" % (
            i,
            row.get("deployable_features_only"),
            row.get("variant_id"),
            row.get("primary_lobo_strong_pass"),
            row.get("primary_lobo_weak_pass"),
            int(row.get("primary_bad_count_total", 0)),
            int(row.get("primary_physical_gate_fail_count", 0)),
            int(row.get("matched_bad_count_total", 0)),
            sf(row.get("min_primary_decision_saving"), 0.0),
            sf(row.get("avg_primary_decision_saving"), 0.0),
            int(row.get("primary_h10_predictions_total", 0)),
            sf(row.get("max_primary_physical_delta"), 0.0),
        ))
    lines += ["", f"Strong LOBO candidates: `{len(strong)}` total, `{len(strong_deploy)}` deployable-feature-only."]
    if top:
        lines += ["", "## Best overall candidate", "", f"Variant: `{top['variant_id']}`", f"Spec: `{top['spec']}`", f"Compact holdout table: `{minimal_eval_for_table(top)}`"]
    if top_deploy and top_deploy is not top:
        lines += ["", "## Best deployable-feature-only candidate", "", f"Variant: `{top_deploy['variant_id']}`", f"Spec: `{top_deploy['spec']}`", f"Compact holdout table: `{minimal_eval_for_table(top_deploy)}`"]
    lines += ["", "## Univariate catastrophic-veto audit (diagnostic only)", "", "| feature | rule | cat covered | positives vetoed | rows hit | examples |", "|---|---|---:|---:|---:|---|"]
    for rec in raw.get("univariate_cat_audit", [])[:12]:
        examples = ", ".join(rec.get("hit_rows") or [])
        lines.append(f"| `{rec['feature']}` | `{rec['direction']} {rec['threshold']:.4g}` | {rec['catastrophic_covered']}/{rec['catastrophic_total']} | {rec['positive_vetoed']}/{rec['positive_total']} | {rec['all_rows_hit']} | {examples} |")
    lines += ["", "## Decision", "", raw["decision"], "", "Interpretation limits: this is opened-development evidence. A candidate that passes here is not validation; it was selected using v2 outcomes and must be confirmed on a future unused fresh-source block before validation64. Timing values are measured whole-decision/solver sums from existing branch rollouts; no speed is inferred from nominal H alone.", "", f"Raw: `{rel(OUT_DIR / 'raw.json')}`; completed: `{rel(OUT_DIR / 'completed.json')}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    headline = raw.get("headline") or {}
    block = f"""
## {MARKER}

Development-only failure-aware selector refit/LOBO diagnostic over opened fresh_v0/v1/v2 banks. No simulations, no validation64, no sealed test, no gradient training.

Headline: best deployable strong pass = `{headline.get('best_deployable_strong_pass')}`, strong deployable candidates = `{headline.get('strong_deployable_count')}`, best variant = `{headline.get('best_variant')}`, decision = {raw.get('decision')}.

Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
""".strip()
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    line = f"{now_utc().isoformat()},{NAME},development_failure_aware_selector_refit_lobo,no_simulation_no_validation_no_test,0,0,0,0,0,False,{rel(OUT_DIR / 'completed.json')},{MARKER}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old[-100000:]:
        reg.write_text(old.rstrip() + "\n" + line, encoding="utf-8")


def write_state(raw: Mapping[str, Any]) -> None:
    headline = raw.get("headline") or {}
    text = f"""# Continue state after failure-aware selector refit v3 LOBO

UTC: {raw['created_utc']}

Access/safety: validation64 unopened, sealed final test unopened, no simulations, no training/refit/gradient steps. Current branch remains IMPROVED development-only true-variable-H H10/H15; not ORIGINAL SAC reproduction.

Result: best deployable strong pass `{headline.get('best_deployable_strong_pass')}`, strong deployable candidates `{headline.get('strong_deployable_count')}`, all strong candidates `{headline.get('strong_count')}`, best variant `{headline.get('best_variant')}`.

Decision: {raw.get('decision')}

Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.

Next action: {raw.get('next_action')}
"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--backup-verified-commit", required=True)
    parser.add_argument("--i-accept-development-failure-aware-selector-refit-v3-lobo", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not args.run or not args.i_accept_development_failure_aware_selector_refit_v3_lobo:
        raise ContractError("requires --run and explicit v3 LOBO development diagnostic acknowledgement")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    hashes: Dict[str, str] = {rel(Path(__file__)): sha256(Path(__file__))}
    for bank, directory in BANK_DIRS.items():
        bank_rows, bank_hashes = load_bank(bank, directory)
        rows.extend(bank_rows)
        hashes.update(bank_hashes)
    if len(rows) < 90:
        raise ContractError(f"unexpectedly few rows loaded: {len(rows)}")
    ranked = run_refit(rows)
    strong = [r for r in ranked if r.get("primary_lobo_strong_pass")]
    strong_deploy = [r for r in strong if r.get("deployable_features_only")]
    top = ranked[0] if ranked else None
    best_deploy = next((r for r in ranked if r.get("deployable_features_only")), None)
    best_deploy_strong = strong_deploy[0] if strong_deploy else None
    if best_deploy_strong is not None:
        decision = "A deployable-feature failure-aware selector class can pass opened-bank LOBO strong gates; freeze an unused fresh-source confirmation v3 before validation64, because this result used v2 outcomes."
        next_action = "After external backup, freeze and run a new unused fresh-source confirmation for the best deployable LOBO candidate; do not open validation64 yet."
    else:
        decision = "No deployable-feature selector class passed opened-bank LOBO strong gates; prioritize bounded value/objective/representation retraining or terminal-value refit rather than another nearest-neighbour threshold sweep."
        next_action = "After backup, design the smallest value/objective/representation refit or retraining ablation with fresh development confirmation; validation64 remains blocked."
    created = now_utc()
    backup_req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_FAILURE_AWARE_SELECTOR_REFIT_V3_LOBO_{created.strftime('%Y%m%dT%H%M%S.%f%z')}.json"
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": "development_IMPROVED_opened_bank_failure_aware_selector_refit_lobo_no_simulation_no_validation_no_test",
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_selector_refits_enumerated": len(candidate_specs()),
        "row_summary": summarize_rows(rows),
        "univariate_cat_audit": univariate_cat_audit(rows),
        "ranked_candidates": ranked[:100],
        "headline": {
            "best_variant": top.get("variant_id") if top else None,
            "best_deployable_variant": best_deploy.get("variant_id") if best_deploy else None,
            "best_deployable_strong_variant": best_deploy_strong.get("variant_id") if best_deploy_strong else None,
            "best_deployable_strong_pass": best_deploy_strong is not None,
            "strong_count": len(strong),
            "strong_deployable_count": len(strong_deploy),
            "best_min_primary_decision_saving": top.get("min_primary_decision_saving") if top else None,
            "best_primary_bad_count_total": top.get("primary_bad_count_total") if top else None,
            "best_matched_bad_count_total": top.get("matched_bad_count_total") if top else None,
        },
        "decision": decision,
        "next_action": next_action,
        "hashes": hashes,
        "backup_request": rel(backup_req),
        "system": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    hashes[rel(OUT_DIR / "raw.json")] = sha256(OUT_DIR / "raw.json")
    hashes[rel(OUT_DIR / "summary.md")] = sha256(OUT_DIR / "summary.md")
    raw["hashes"] = hashes
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    write_state(raw)
    append_docs(raw)
    write_json(backup_req, {
        "created_utc": created.isoformat(),
        "reason": "backup after development-only failure-aware selector refit v3 LOBO diagnostic",
        "artifacts": [rel(OUT_DIR), rel(STATE_FILE), rel(Path(__file__))],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    completed = {
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "passed": True,
        "hard_pass": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_selector_refits_enumerated": len(candidate_specs()),
        "headline": raw["headline"],
        "decision": decision,
        "backup_request": rel(backup_req),
        "hashes": {**hashes, rel(STATE_FILE): sha256(STATE_FILE), rel(backup_req): sha256(backup_req)},
    }
    write_json(OUT_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "headline": raw["headline"], "decision": decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
