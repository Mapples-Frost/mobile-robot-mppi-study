#!/usr/bin/env python3
"""History/static-feature representation diagnostic for true-variable-H H10/H15.

Development-only IMPROVED diagnostic.  It consumes only already-opened
fresh-source banks v0/v1/v2 and asks whether the catastrophic H10 false
positives that defeated v3b/v4/v6 become predictable when the selector is given
richer *pre-decision* information: recent H15-roll-in observation/action/cost
history and static trajectory/clearance metadata from the frozen case protocol.

The main comparison is leave-one-fresh-bank-out (LOBO) over the three opened
banks.  No validation64 bank, sealed test, MPC simulation, gradient training, or
checkpoint writing is performed.  A separate non-deployable diagnostic family is
included that uses same-step H15 solver information; if only that family works,
the result points to terminal/value or online lookahead requirements rather than
a cheap selector.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_tree_representation_v6 as base  # noqa:E402

NAME = "vehicle_true_variable_horizon_history_representation_v7_lobo"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_history_representation_v7_lobo.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-history-representation-v7-{STAMP}"
PROTOCOLS = {
    "fresh_v0": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json",
    "fresh_v1": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json",
    "fresh_v2": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json",
}
PROFILE = "shared_h15_terminal"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
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


def row_key(r: Mapping[str, Any]) -> str:
    return str(r["bank_id"]) + "/" + str(r["base_state_id"])


def scalar0(x: Any, default: float = 0.0) -> float:
    if isinstance(x, (list, tuple)) and x:
        return scalar0(x[0], default)
    return sf(x, default)


def attempt(row: Mapping[str, Any]) -> Mapping[str, Any]:
    arr = ((row.get("recovery") or {}).get("attempts") or [])
    if arr and isinstance(arr[0], Mapping):
        return arr[0]
    return {}


def solver_iter(row: Mapping[str, Any]) -> float:
    return sf(attempt(row).get("iterations"), 0.0)


def solver_obj(row: Mapping[str, Any]) -> float:
    return sf(attempt(row).get("objective_opt_f_num"), 0.0)


def solver_s(row: Mapping[str, Any]) -> float:
    return sf(attempt(row).get("solver_s"), 0.0)


def input_abs_omega(row: Mapping[str, Any]) -> float:
    inp = row.get("input") or {}
    return abs(scalar0(inp.get("u_omega") if isinstance(inp, Mapping) else None, 0.0))


def input_speed(row: Mapping[str, Any]) -> float:
    inp = row.get("input") or {}
    return scalar0(inp.get("u_s") if isinstance(inp, Mapping) else None, 0.0)


def state_field(row: Mapping[str, Any], key: str) -> float:
    st = row.get("previous_state") or row.get("state") or {}
    return scalar0(st.get(key) if isinstance(st, Mapping) else None, 0.0)


def load_trace(trace_dir_rel: str) -> List[Mapping[str, Any]]:
    d = ROOT / trace_dir_rel
    jl = d / "trace.jsonl"
    js = d / "trace.json"
    if jl.exists():
        out: List[Mapping[str, Any]] = []
        with jl.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, Mapping):
                        out.append(obj)
        return out
    if js.exists():
        obj = read_json(js)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, Mapping)]
    raise RuntimeError("missing trace for " + trace_dir_rel)


def load_manifest_meta() -> Tuple[Dict[str, Mapping[str, Any]], Dict[Tuple[str, int], Mapping[str, Any]], Dict[str, str]]:
    manifests: Dict[str, Mapping[str, Any]] = {}
    case_meta: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    hashes: Dict[str, str] = {rel(Path(__file__)): sha256(Path(__file__))}
    for bank, d in base.BANK_DIRS.items():
        raw = base.read_json(d / "raw.json")
        man_path = d / "selected_state_manifest.json"
        states = base.load_manifest(raw, man_path)
        hashes[rel(man_path)] = sha256(man_path)
        for st in states:
            manifests[bank + "/" + str(st.get("base_state_id"))] = st
        prot_path = PROTOCOLS[bank]
        prot = read_json(prot_path)
        hashes[rel(prot_path)] = sha256(prot_path)
        for c in ((prot.get("case_source") or {}).get("selected_fresh_cases") or []):
            case_meta[(bank, si(c.get("fresh_case_index"), -1))] = c
    return manifests, case_meta, hashes


def add_window_features(fd: Dict[str, float], prefix: str, seq: Sequence[Mapping[str, Any]]) -> None:
    n = max(1, len(seq))
    perfs = [sf(r.get("performance"), 0.0) for r in seq]
    cons = [sf(r.get("constraint"), 0.0) for r in seq]
    iters = [solver_iter(r) for r in seq]
    objs = [solver_obj(r) for r in seq]
    turns = [input_abs_omega(r) for r in seq]
    speeds = [input_speed(r) for r in seq]
    solvers = [solver_s(r) for r in seq]
    ys = [state_field(r, "y") for r in seq]
    ths = [state_field(r, "theta") for r in seq]
    for name, vals in (("perf", perfs), ("constraint", cons), ("iters", iters), ("obj", objs), ("turn", turns), ("speed", speeds), ("solver_s", solvers), ("abs_y", [abs(v) for v in ys]), ("abs_theta", [abs(v) for v in ths])):
        if vals:
            fd[f"{prefix}_{name}_mean"] = sum(vals) / max(1, len(vals))
            fd[f"{prefix}_{name}_max"] = max(vals)
            fd[f"{prefix}_{name}_sum"] = sum(vals)
        else:
            fd[f"{prefix}_{name}_mean"] = 0.0; fd[f"{prefix}_{name}_max"] = 0.0; fd[f"{prefix}_{name}_sum"] = 0.0
    fd[f"{prefix}_n"] = float(len(seq)) / 25.0


def augment_features(rows: Sequence[Mapping[str, Any]], all_groups: Sequence[str], manifests: Mapping[str, Mapping[str, Any]], case_meta: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]], Dict[str, Any]]:
    fdicts: Dict[str, Dict[str, float]] = {}
    trace_cache: Dict[str, List[Mapping[str, Any]]] = {}
    trace_hashes: Dict[str, str] = {}
    for r in rows:
        k = row_key(r)
        st = manifests.get(k)
        if not st:
            raise RuntimeError("missing manifest state for " + k)
        fd = base.feature_dict(r, all_groups)
        trace_dir = str(st.get("h15_trace_episode_path"))
        if trace_dir not in trace_cache:
            trace_cache[trace_dir] = load_trace(trace_dir)
            p = ROOT / trace_dir / ("trace.jsonl" if (ROOT / trace_dir / "trace.jsonl").exists() else "trace.json")
            trace_hashes[rel(p)] = sha256(p)
        trace = trace_cache[trace_dir]
        b = min(max(0, si(st.get("branch_step"), si(r.get("branch_step"), 0))), max(0, len(trace) - 1))
        current = trace[b] if trace else {}
        prior = trace[:b]
        for win in (3, 5, 10, 25):
            add_window_features(fd, f"hist{win}", prior[max(0, len(prior)-win):])
        add_window_features(fd, "hist_all", prior)
        obs_now = [sf(v) for v in (st.get("initial_observation_from_h15_trace") or [])[:14]]
        while len(obs_now) < 14:
            obs_now.append(0.0)
        for lag in (1, 5, 10):
            if b - lag >= 0 and b - lag < len(trace):
                obs_prev = [sf(v) for v in (trace[b-lag].get("observation") or [])[:14]]
            else:
                obs_prev = [0.0] * 14
            while len(obs_prev) < 14:
                obs_prev.append(0.0)
            for i in range(14):
                fd[f"dobs_lag{lag}_{i}"] = obs_now[i] - obs_prev[i]
        meta = case_meta.get((str(r["bank_id"]), si(st.get("fresh_case_index"), -1)), {})
        fd["static_abs_theta_r"] = abs(sf(meta.get("theta_r"), 0.0)) / math.pi
        fd["static_traj_steps"] = sf(meta.get("traj_steps"), 0.0) / 150.0
        fd["static_clearance"] = sf(meta.get("min_reference_obstacle_clearance"), 0.0) / 10.0
        fd["static_stress_v1_score"] = sf(meta.get("stress_v1_score"), 0.0) / 20.0
        # Same-step H15 solve features are a diagnostic upper-bound family only:
        # they require solving H15 before selecting H and are not a cheap selector.
        fd["same_h15_perf"] = sf(current.get("performance"), 0.0)
        fd["same_h15_constraint"] = sf(current.get("constraint"), 0.0)
        fd["same_h15_iters"] = solver_iter(current) / 100.0
        fd["same_h15_obj"] = solver_obj(current) / 100.0
        fd["same_h15_solver_s"] = solver_s(current)
        fd["same_h15_turn"] = input_abs_omega(current) / 5.0
        fd["same_h15_speed"] = input_speed(current) / 5.0
        fdicts[k] = fd
    keys = sorted({kk for fd in fdicts.values() for kk in fd})
    hist_keys = [kk for kk in keys if kk.startswith("hist") or kk.startswith("dobs_lag")]
    static_keys = ["static_abs_theta_r", "static_traj_steps", "static_clearance", "static_stress_v1_score"]
    same_keys = [kk for kk in keys if kk.startswith("same_h15_")]
    obs_pose = base.feature_names("deploy_obs_pose_step_no_risk", all_groups)
    obs_pose_risk = base.feature_names("deploy_obs_pose_step_risk", all_groups)
    families = {
        "online_obs_pose_baseline_recomputed": obs_pose,
        "online_obs_pose_history": obs_pose + hist_keys,
        "online_obs_pose_history_static": obs_pose + hist_keys + static_keys,
        "online_obs_pose_history_risk_static": obs_pose_risk + hist_keys + static_keys,
        "diagnostic_same_step_h15_solve_history_static": obs_pose_risk + hist_keys + static_keys + same_keys,
    }
    diagnostics = {"trace_files_hashed": trace_hashes, "feature_counts": {k: len(v) for k, v in families.items()}, "history_feature_count": len(hist_keys), "static_feature_count": len(static_keys), "same_step_h15_feature_count": len(same_keys)}
    return fdicts, families, diagnostics


def evaluate_family(rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Dict[str, float]], banks: Sequence[str], by_bank: Mapping[str, List[Mapping[str, Any]]], fam: str, names: Sequence[str]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for depth in (1, 2, 3):
        for min_leaf in (2, 3, 4):
            for min_pos in (1, 2, 3):
                for max_cat in (0, 1):
                    for min_gain in (0.0, 0.25, 0.5, 1.0):
                        hold = {}; trees = {}; strong = True; weak = True; bad = 0; phys_fail = 0; total_h10 = 0; min_save = 1e9; avg_save = 0.0
                        for hb in banks:
                            train = [r for r in rows if r["bank_id"] != hb]
                            tree = base.build_tree(train, fdicts, names, depth, min_leaf, min_pos, max_cat, min_gain)
                            choices = {row_key(r): base.predict(tree, fdicts[row_key(r)]) for r in by_bank[hb]}
                            ev = base.eval_choices(by_bank[hb], choices)
                            hold[hb] = ev; trees[hb] = base.compact_tree(tree)
                            strong = strong and bool(ev["pass_10pct_no_cat_fp"])
                            weak = weak and bool(ev["pass_5pct_no_cat_fp"])
                            bad += len(ev["catastrophic_false_positive_rows"])
                            phys_fail += 0 if ev["physical_gate"] else 1
                            total_h10 += int(ev["chosen_counts"].get("10", 0))
                            min_save = min(min_save, sf(ev["decision_relative_saving_vs_fixed_H15"]))
                            avg_save += sf(ev["decision_relative_saving_vs_fixed_H15"])
                        avg_save /= max(1, len(banks))
                        variants.append({"variant_id": f"v7_{fam}_d{depth}_ml{min_leaf}_mp{min_pos}_mc{max_cat}_mg{min_gain:g}", "family": fam, "spec": {"max_depth": depth, "min_leaf": min_leaf, "min_pos_leaf": min_pos, "max_train_cat_leaf": max_cat, "min_leaf_gain_s": min_gain}, "holdout_evaluations": hold, "trees_by_holdout": trees, "all_holdouts_strong": strong, "all_holdouts_weak": weak, "total_catastrophic_fp": bad, "physical_gate_fail_count": phys_fail, "total_h10": total_h10, "min_decision_saving": min_save, "avg_decision_saving": avg_save})
    return variants


def rank_key(v: Mapping[str, Any]) -> Tuple[Any, ...]:
    same = str(v.get("family", "")).startswith("diagnostic_same_step")
    return (not bool(v["all_holdouts_strong"]), not bool(v["all_holdouts_weak"]), int(v["total_catastrophic_fp"]), int(v["physical_gate_fail_count"]), same, -sf(v["min_decision_saving"]), -sf(v["avg_decision_saving"]), -int(v["total_h10"]))


def pct(x: Any) -> str:
    return f"{100.0*sf(x):.2f}%"


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H history/static representation v7 LOBO",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only opened-bank representation diagnostic; no simulations, validation64, sealed test, or gradient training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}`; feature families `{h['families']}`; tree variants `{h['tree_variants']}`; LOBO tree fits `{h['tree_fits']}`.",
        f"- Online-history strong candidates `{h['online_history_strong_candidates']}`, weak `{h['online_history_weak_candidates']}`; same-step-H15 diagnostic strong `{h['same_step_strong_candidates']}`.",
        f"- Best online-history `{h['best_online_history_variant']}`: min save `{pct(h['best_online_history_min_save'])}`, bad `{h['best_online_history_bad']}`.",
        f"- Best same-step diagnostic `{h['best_same_step_variant']}`: min save `{pct(h['best_same_step_min_save'])}`, bad `{h['best_same_step_bad']}`.",
        "",
        "## Top variants",
        "",
        "| rank | family | strong | weak | bad | phys fails | min save | avg save | H10 total | variant |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, v in enumerate(raw["top_variants"][:15], 1):
        lines.append(f"| {i} | `{v['family']}` | `{v['all_holdouts_strong']}` | `{v['all_holdouts_weak']}` | {v['total_catastrophic_fp']} | {v['physical_gate_fail_count']} | {pct(v['min_decision_saving'])} | {pct(v['avg_decision_saving'])} | {v['total_h10']} | `{v['variant_id']}` |")
    lines += ["", "## Best online-history held-out banks", "", "| bank | H counts | confusion | bad | phys Δ/tol | decision save | solver save | pass10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b, ev in raw["best_online_history_variant_full"]["holdout_evaluations"].items():
        lines.append(f"| `{b}` | `{ev['chosen_counts']}` | `{ev['confusion']}` | {len(ev['catastrophic_false_positive_rows'])} | {sf(ev['physical_delta_vs_fixed_H15']):.4g}/{sf(ev['physical_tolerance_sum']):.4g} | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} | `{ev['pass_10pct_no_cat_fp']}` |")
    lines += ["", "## Decision", "", str(raw["decision"]), "", "Interpretation: opened-development evidence only. Online-history features use past H15-roll-in trace quantities and static protocol metadata; they are a representation diagnostic, not validation. Same-step H15 features are explicitly non-deployable for a cheap selector and only diagnose whether an H15 lookahead/value signal would be needed."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if MARKER not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},development_opened_fresh_bank_history_static_representation_LOBO_no_simulation,development_no_validation_no_test,0,0,0,0,0,False,{rel(OUT / 'completed.json')}\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, input_hashes, all_groups = base.load_rows()
    manifests, case_meta, more_hashes = load_manifest_meta()
    fdicts, families, feat_diag = augment_features(rows, all_groups, manifests, case_meta)
    banks = sorted(base.BANK_DIRS)
    by_bank = {b: [r for r in rows if r["bank_id"] == b] for b in banks}
    variants: List[Dict[str, Any]] = []
    for fam, names in families.items():
        variants.extend(evaluate_family(rows, fdicts, banks, by_bank, fam, names))
    ranked = sorted(variants, key=rank_key)
    online = [v for v in variants if not str(v["family"]).startswith("diagnostic_same_step")]
    same = [v for v in variants if str(v["family"]).startswith("diagnostic_same_step")]
    best = ranked[0]
    best_online = sorted(online, key=rank_key)[0]
    best_same = sorted(same, key=rank_key)[0]
    online_strong = sum(1 for v in online if v["all_holdouts_strong"])
    online_weak = sum(1 for v in online if v["all_holdouts_weak"])
    same_strong = sum(1 for v in same if v["all_holdouts_strong"])
    if online_strong:
        decision = "Richer online history/static representation can meet opened-bank strong gates; next freeze an unused fresh-source confirmation with this feature family and actual selector overhead before validation64."
    elif online_weak:
        decision = "Richer online history/static representation reaches only weak opened-bank gates; next choose between a small unused fresh-source overhead confirmation and targeted risk-data acquisition, but do not claim strong compute tradeoff."
    elif same_strong:
        decision = "Only same-step H15-solve diagnostic features meet strong gates; a cheap pre-MPC selector is likely under-informed. Next intervention should be value-function/lookahead training or explicit risk-state acquisition, not more threshold/NN sweeps."
    else:
        decision = "History/static deployable features still fail opened-bank weak/strong gates. Prioritize targeted source-independent risk acquisition or value-function training/refit with richer risk/clearance representation rather than another selector-only sweep."
    created = now_utc()
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_HISTORY_REPRESENTATION_V7_{STAMP}.json"
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup history/static representation v7 opened-bank diagnostic", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_training_episodes": 0, "gradient_steps": 0, "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__)), rel(req)]})
    headline = {"rows": len(rows), "families": len(families), "tree_variants": len(variants), "tree_fits": len(variants) * len(banks), "online_history_strong_candidates": online_strong, "online_history_weak_candidates": online_weak, "same_step_strong_candidates": same_strong, "best_variant": best["variant_id"], "best_online_history_variant": best_online["variant_id"], "best_online_history_min_save": best_online["min_decision_saving"], "best_online_history_bad": best_online["total_catastrophic_fp"], "best_same_step_variant": best_same["variant_id"], "best_same_step_min_save": best_same["min_decision_saving"], "best_same_step_bad": best_same["total_catastrophic_fp"]}
    raw = {"created_utc": created.isoformat(), "classification": "development_IMPROVED_opened_fresh_bank_history_static_representation_no_simulation_no_validation_no_test", "hypothesis": "If catastrophic H10 risk is predictable from deployable recent history/static scenario features, LOBO history trees should pass opened-bank gates; if only same-step H15 solve features work, a value/lookahead method is needed; if neither works, gather targeted risk data.", "budgets": {"new_simulations": 0, "new_control_steps": 0, "new_training_episodes": 0, "gradient_steps": 0, "closed_form_tree_fits": len(variants) * len(banks), "validation64_episodes": 0, "sealed_test_episodes": 0}, "feature_diagnostics": feat_diag, "headline": headline, "top_variants": ranked[:20], "best_online_history_variant_full": best_online, "best_same_step_variant_full": best_same, "decision": decision, "input_hashes": {**input_hashes, **more_hashes, **feat_diag.get("trace_files_hashed", {})}, "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False}, "backup_request": rel(req)}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    completed = {"passed": True, "status": "complete", "marker": MARKER, "headline": headline, "decision": decision, "new_simulations": 0, "gradient_steps": 0, "closed_form_tree_fits": len(variants) * len(banks), "validation64_bank_opened": False, "sealed_test_accessed": False, "raw": rel(OUT / "raw.json"), "summary": rel(OUT / "summary.md"), "backup_request": rel(req), "hashes": {rel(Path(__file__)): sha256(Path(__file__)), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md")}}
    write_json(OUT / "completed.json", completed)
    block = f"""
<!-- {MARKER} -->
## 2026-09-29 true-variable-H history/static representation v7

UTC: {created.isoformat()}. Development-only opened-fresh-bank diagnostic, no simulations/control steps, no validation64, no sealed test, no gradient training. Headline: `{json.dumps(headline, sort_keys=True)}`. Decision: {decision}

Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup request: `{rel(req)}`.
"""
    append_docs(block)
    STATE.write_text(f"# Continue state after {NAME}\n\nUTC: {created.isoformat()}\n\nHeadline: {json.dumps(headline, sort_keys=True)}\n\nDecision: {decision}\n\nBudgets: no simulations/control steps, no validation64, no sealed test, no gradient training; closed_form_tree_fits={len(variants) * len(banks)}.\n\nNext: if online history strong, freeze unused fresh-source overhead confirmation; otherwise run targeted source-independent risk acquisition/value-lookahead training smoke with richer clearance/history representation.\n\nArtifacts: {rel(OUT / 'summary.md')}, {rel(OUT / 'raw.json')}, {rel(OUT / 'completed.json')}.\n", encoding="utf-8")
    print("SUMMARY", rel(OUT / "summary.md"))
    print(json.dumps({"headline": headline, "decision": decision, "best_online_holdouts": {b: {"save": ev["decision_relative_saving_vs_fixed_H15"], "bad": len(ev["catastrophic_false_positive_rows"]), "h_counts": ev["chosen_counts"], "pass10": ev["pass_10pct_no_cat_fp"]} for b, ev in best_online["holdout_evaluations"].items()}, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "completed.json", {"passed": False, "status": "failed", "marker": MARKER, "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
