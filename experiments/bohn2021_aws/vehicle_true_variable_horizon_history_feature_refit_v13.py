#!/usr/bin/env python3
"""v13 deployable history-feature risk/value refit for true-variable H10/H15.

Development-only IMPROVED diagnostic after v12.  v12 verified an objective/label
mismatch and showed oracle H10/H15 branch opportunity, but pure deployable static
shallow trees still failed nested opened-bank gates.  This script tests a concrete
next hypothesis: H10 catastrophic modes may require short H15-prefix/history
features (recent observation and pose trends), not only instantaneous branch
features.

It performs no MPC simulation, no validation64 access, no sealed-test access, and
no gradient training.  It refits bounded conservative support selectors over the
already opened development branch rows (fresh_v0/v1/v2/v8c/v11) using v12's
outcome-aligned labels.  Scenario/group/window identifiers are never used by the
primary deployable candidates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_tree_representation_v6 as v6  # noqa:E402
import vehicle_true_variable_horizon_outcome_aligned_refit_v12 as v12  # noqa:E402

NAME = "vehicle_true_variable_horizon_history_feature_refit_v13"
STAMP = "20260929T2055Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T2055_after_history_feature_refit_v13.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-history-feature-refit-v13-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

BANK_DIRS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z",
    "fresh_v8c": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z",
    "fresh_v11": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z",
}

class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


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
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {label}")


def row_key(r: Mapping[str, Any]) -> str:
    return str(r["bank_id"]) + "/" + str(r["base_state_id"])


def obs(row: Mapping[str, Any]) -> List[float]:
    vals = [sf(v) for v in (row.get("obs14") or [])[:14]]
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    rows0, hashes, _ = v12.load_all_rows()
    rows, label_audit = v12.relabel_rows(rows0)
    banks = sorted(set(str(r.get("bank_id")) for r in rows))
    if banks != ["fresh_v0", "fresh_v1", "fresh_v11", "fresh_v2", "fresh_v8c"] or len(rows) != 68:
        raise ContractError(f"unexpected opened rows/banks: {len(rows)} {banks}")
    for r in rows:
        bid = str(r.get("bank_id", "")).lower()
        if "validation" in bid or "test" in bid:
            raise ContractError("forbidden bank id: " + str(r.get("bank_id")))
    hashes["v12_label_audit_flip_count"] = str(label_audit.get("flip_count"))
    return [dict(r) for r in rows], hashes


def manifest_entries_for_bank(bank: str) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, str]]:
    d = BANK_DIRS[bank]
    hashes: Dict[str, str] = {}
    raw_path = d / "raw.json"
    done_path = d / "completed.json"
    man_path = d / "selected_state_manifest.json"
    if not raw_path.exists() or not done_path.exists():
        raise ContractError("missing opened bank raw/done for " + bank)
    raw = read_json(raw_path); done = read_json(done_path)
    assert_dev_only(raw, bank + " raw"); assert_dev_only(done, bank + " completed")
    hashes[rel(raw_path)] = sha256(raw_path); hashes[rel(done_path)] = sha256(done_path)
    if man_path.exists():
        man = read_json(man_path); hashes[rel(man_path)] = sha256(man_path)
        entries = man.get("selected_states") if isinstance(man, Mapping) else man
    else:
        sm = raw.get("selected_state_manifest") if isinstance(raw, Mapping) else None
        entries = sm.get("selected_states") if isinstance(sm, Mapping) else sm
    if not isinstance(entries, list):
        raise ContractError("missing selected states for " + bank)
    out: Dict[str, Mapping[str, Any]] = {}
    for e in entries:
        if not isinstance(e, Mapping):
            continue
        bid = str(e.get("base_state_id") or "")
        if bid:
            out[bid] = e
    return out, hashes


def state_from_trace_row(tr: Mapping[str, Any]) -> Mapping[str, Any]:
    st = tr.get("previous_state") if isinstance(tr.get("previous_state"), Mapping) else None
    if st is None:
        st = tr.get("state") if isinstance(tr.get("state"), Mapping) else None
    return st or {}


def obs_from_trace_row(tr: Mapping[str, Any]) -> List[float]:
    raw = tr.get("observation") or []
    vals = [sf(v) for v in raw[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def enrich_history(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, str]]:
    manifests: Dict[str, Mapping[str, Any]] = {}
    hashes: Dict[str, str] = {}
    for bank in BANK_DIRS:
        m, h = manifest_entries_for_bank(bank)
        hashes.update(h)
        for sid, e in m.items():
            manifests[bank + "/" + sid] = e
    features: Dict[str, Dict[str, float]] = {}
    trace_loaded = 0; trace_missing = 0; trace_hashes = 0
    missing_keys: List[str] = []
    for row in rows:
        key = row_key(row)
        o = obs(row)
        th = sf(row.get("prev_theta"))
        fd: Dict[str, float] = {
            "prev_x_30": sf(row.get("prev_x")) / 30.0,
            "prev_y_30": sf(row.get("prev_y")) / 30.0,
            "theta_sin": math.sin(th),
            "theta_cos": math.cos(th),
            "abs_theta_pi": abs(th) / math.pi,
            "branch_step_150": max(0.0, sf(row.get("branch_step"))) / 150.0,
            "slot": sf(row.get("branch_state_slot")),
            "trace_risk_20": sf(row.get("stage_a_trace_risk_score")) / 20.0,
            "obs01_norm": math.hypot(o[0], o[1]),
            "obs56_norm": math.hypot(o[5], o[6]),
            "obs89_norm": math.hypot(o[8], o[9]),
            "obs1112_norm": math.hypot(o[11], o[12]),
        }
        for i in range(14):
            fd[f"obs_{i}"] = o[i]
            fd[f"abs_obs_{i}"] = abs(o[i])
        for a, b in [(0, 1), (0, 2), (1, 2), (5, 6), (8, 9), (11, 12), (7, 10), (10, 13)]:
            fd[f"obs_{a}_x_obs_{b}"] = o[a] * o[b]
        man = manifests.get(key)
        if not man:
            trace_missing += 1; missing_keys.append(key); features[key] = fd; continue
        trace_rel = man.get("h15_trace_episode_path")
        trace_path = ROOT / str(trace_rel) / "trace.json" if trace_rel else None
        branch_step = si(man.get("branch_step", row.get("branch_step")), -1)
        if not trace_path or not trace_path.exists():
            trace_missing += 1; missing_keys.append(key); features[key] = fd; continue
        trace = read_json(trace_path)
        if not isinstance(trace, list) or not trace:
            trace_missing += 1; missing_keys.append(key); features[key] = fd; continue
        if trace_hashes < 12:
            hashes[rel(trace_path)] = sha256(trace_path); trace_hashes += 1
        prefix = [tr for tr in trace if si(tr.get("step"), -1) <= branch_step]
        if not prefix:
            prefix = trace[: max(1, min(len(trace), branch_step + 1 if branch_step >= 0 else len(trace)))]
        trace_loaded += 1
        for lag in [1, 3, 5, 10]:
            if len(prefix) >= lag + 1:
                old = prefix[-lag-1]; cur = prefix[-1]
                so = state_from_trace_row(old); sc = state_from_trace_row(cur)
                fd[f"hist_dx_lag{lag}"] = (sf(sc.get("x")) - sf(so.get("x"))) / max(1.0, float(lag))
                fd[f"hist_dy_lag{lag}"] = (sf(sc.get("y")) - sf(so.get("y"))) / max(1.0, float(lag))
                fd[f"hist_dtheta_lag{lag}"] = (sf(sc.get("theta")) - sf(so.get("theta"))) / max(1.0, float(lag))
                oo = obs_from_trace_row(old); oc = obs_from_trace_row(cur)
                for i in (0, 1, 2, 5, 6, 8, 9, 11, 12):
                    fd[f"hist_dobs{i}_lag{lag}"] = oc[i] - oo[i]
        win = prefix[-min(12, len(prefix)):]
        if win:
            obs_win = [obs_from_trace_row(x) for x in win]
            for i in (0, 1, 2, 5, 6, 8, 9, 11, 12):
                xs = [u[i] for u in obs_win]
                m = sum(xs) / len(xs)
                fd[f"hist_obs{i}_mean12"] = m
                fd[f"hist_obs{i}_std12"] = math.sqrt(sum((x-m)*(x-m) for x in xs) / max(1, len(xs)-1))
                fd[f"hist_obs{i}_min12"] = min(xs)
                fd[f"hist_obs{i}_max12"] = max(xs)
            pair_norms = []
            for u in obs_win:
                pair_norms.extend([math.hypot(u[5], u[6]), math.hypot(u[8], u[9]), math.hypot(u[11], u[12])])
            if pair_norms:
                fd["hist_pair_norm_min12"] = min(pair_norms)
                fd["hist_pair_norm_mean12"] = sum(pair_norms) / len(pair_norms)
                fd["hist_pair_norm_last"] = min(math.hypot(o[5], o[6]), math.hypot(o[8], o[9]), math.hypot(o[11], o[12]))
            states = [state_from_trace_row(x) for x in win]
            ths = [sf(s.get("theta")) for s in states]
            if ths:
                mt = sum(ths) / len(ths)
                fd["hist_theta_mean12"] = mt
                fd["hist_theta_std12"] = math.sqrt(sum((x-mt)*(x-mt) for x in ths) / max(1, len(ths)-1))
        features[key] = fd
    diag = {"rows": len(rows), "manifest_entries": len(manifests), "trace_loaded_rows": trace_loaded, "trace_missing_rows": trace_missing, "missing_keys_sample": missing_keys[:10], "history_feature_hashes_limited": trace_hashes}
    return features, diag, hashes


def make_family_features(all_features: Mapping[str, Mapping[str, float]], family: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, fd0 in all_features.items():
        fd = dict(fd0)
        if family == "base_no_history":
            fd = {n: v for n, v in fd.items() if not n.startswith("hist_")}
        elif family == "history_no_risk":
            fd = {n: v for n, v in fd.items() if n != "trace_risk_20"}
        elif family == "history_with_risk":
            pass
        else:
            raise ContractError("unknown family " + family)
        out[k] = fd
    return out


def train_scaler(train_rows: Sequence[Mapping[str, Any]], fdict: Mapping[str, Mapping[str, float]]) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    names = sorted(set().union(*(fdict[row_key(r)].keys() for r in train_rows)))
    means: Dict[str, float] = {}; stds: Dict[str, float] = {}
    for nm in names:
        xs = [sf(fdict[row_key(r)].get(nm)) for r in train_rows]
        m = sum(xs) / len(xs)
        var = sum((x-m)*(x-m) for x in xs) / max(1, len(xs)-1)
        means[nm] = m; stds[nm] = math.sqrt(var) if var > 1e-18 else 1.0
    return names, means, stds


def vec(row: Mapping[str, Any], fdict: Mapping[str, Mapping[str, float]], names: Sequence[str], means: Mapping[str, float], stds: Mapping[str, float]) -> List[float]:
    fd = fdict[row_key(row)]
    return [(sf(fd.get(nm)) - sf(means.get(nm))) / max(sf(stds.get(nm), 1.0), 1e-12) for nm in names]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x-y)*(x-y) for x, y in zip(a, b)) / max(1, len(a)))


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def eval_support(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], fdict: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    names, means, stds = train_scaler(train_rows, fdict)
    train_vecs = {row_key(r): vec(r, fdict, names, means, stds) for r in train_rows}
    pos_train = [r for r in train_rows if r["h10_beneficial_vs_h15"]]
    cat_train = [r for r in train_rows if r["h10_catastrophic_vs_h15"]]
    # Data-derived absolute-radius guard from train positive-positive distances.
    pp: List[float] = []
    for i, r in enumerate(pos_train):
        ds = [dist(train_vecs[row_key(r)], train_vecs[row_key(q)]) for j, q in enumerate(pos_train) if j != i]
        if ds:
            pp.append(min(ds))
    pp = sorted(pp)
    q = sf(cfg.get("max_pos_quantile"), 1.0)
    if pp:
        idx = max(0, min(len(pp)-1, int(round(q * (len(pp)-1)))))
        max_pos_dist = pp[idx] * sf(cfg.get("max_pos_mult"), 1.0)
    else:
        max_pos_dist = 1e9
    choices: Dict[str, int] = {}
    pred_info: Dict[str, Any] = {}
    for er in eval_rows:
        ex = vec(er, fdict, names, means, stds)
        pos_neigh = sorted([(dist(ex, train_vecs[row_key(r)]), r) for r in pos_train], key=lambda z: (z[0], row_key(z[1])))
        cat_neigh = sorted([(dist(ex, train_vecs[row_key(r)]), r) for r in cat_train], key=lambda z: (z[0], row_key(z[1])))
        if not pos_neigh:
            choices[row_key(er)] = 15; continue
        k = min(si(cfg.get("k_pos"), 1), len(pos_neigh))
        near = pos_neigh[:k]
        dpos = near[-1][0]
        dcat = cat_neigh[0][0] if cat_neigh else 1e9
        pred_gain = mean([sf(r["decision_gain_h10_vs_h15_s"]) for _, r in near])
        pred_phys = mean([sf(r["phys_delta_h10_minus_h15"]) for _, r in near])
        cat_guard = dcat > sf(cfg.get("cat_guard_ratio"), 1.25) * max(dpos, 1e-9) + sf(cfg.get("cat_margin"), 0.0)
        choose = bool(dpos <= max_pos_dist and cat_guard and pred_gain >= sf(cfg.get("min_gain_s"), 0.0) and pred_phys <= sf(cfg.get("pred_phys_max"), 2.0))
        choices[row_key(er)] = 10 if choose else 15
        pred_info[row_key(er)] = {"dpos": dpos, "dcat": dcat, "max_pos_dist": max_pos_dist, "pred_gain_s": pred_gain, "pred_phys_delta": pred_phys, "cat_guard": cat_guard, "feature_count": len(names)}
    ev = v6.eval_choices(eval_rows, choices)
    ev["prediction_scores"] = pred_info
    ev["feature_count"] = len(names)
    return ev


def grid() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for family in ["base_no_history", "history_no_risk", "history_with_risk"]:
        for k_pos in [1, 2, 3, 5]:
            for cat_guard_ratio in [1.05, 1.25, 1.5, 2.0, 3.0]:
                for max_pos_quantile in [0.25, 0.50, 0.75, 1.0]:
                    for max_pos_mult in [0.75, 1.0, 1.25]:
                        for min_gain_s in [0.0, 0.2, 0.5]:
                            for pred_phys_max in [0.0, 2.0, 20.0]:
                                cfgs.append({"family": family, "k_pos": k_pos, "cat_guard_ratio": cat_guard_ratio, "max_pos_quantile": max_pos_quantile, "max_pos_mult": max_pos_mult, "min_gain_s": min_gain_s, "pred_phys_max": pred_phys_max, "cat_margin": 0.0})
    return cfgs


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v13_%s_k%s_cg%s_q%s_m%s_g%s_p%s" % (c["family"], c["k_pos"], c["cat_guard_ratio"], c["max_pos_quantile"], c["max_pos_mult"], c["min_gain_s"], c["pred_phys_max"])


def summarize(hold: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    bad = sum(len(e.get("catastrophic_false_positive_rows") or []) for e in hold.values())
    phys_fail = sum(0 if e.get("physical_gate") else 1 for e in hold.values())
    saves = [sf(e.get("decision_relative_saving_vs_fixed_H15"), -9.0) for e in hold.values()]
    h10 = sum(int((e.get("chosen_counts") or {}).get("10", 0)) for e in hold.values())
    return {"all_pass10": all(bool(e.get("pass_10pct_no_cat_fp")) for e in hold.values()), "all_pass5": all(bool(e.get("pass_5pct_no_cat_fp")) for e in hold.values()), "total_bad": bad, "phys_fail": phys_fail, "min_save": min(saves) if saves else 0.0, "avg_save": mean(saves), "total_h10": h10}


def rank(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (int(s["total_bad"]), int(s["phys_fail"]), not bool(s["all_pass10"]), not bool(s["all_pass5"]), -sf(s["min_save"]), -sf(s["avg_save"]), -int(s["total_h10"]))


def eval_lobo(rows: Sequence[Mapping[str, Any]], banks: Sequence[str], all_fdict: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any], blocked: Sequence[str] = ()) -> Dict[str, Any]:
    blocked_set = set(blocked)
    fdict = make_family_features(all_fdict, str(cfg["family"]))
    hold: Dict[str, Any] = {}
    for hb in banks:
        if hb in blocked_set:
            continue
        tr = [r for r in rows if str(r["bank_id"]) not in blocked_set and str(r["bank_id"]) != hb]
        ev = [r for r in rows if str(r["bank_id"]) == hb]
        hold[hb] = eval_support(tr, ev, fdict, cfg)
    return hold


def ambiguity(rows: Sequence[Mapping[str, Any]], fdict: Mapping[str, Mapping[str, float]], family: str) -> Dict[str, Any]:
    names, means, stds = train_scaler(rows, fdict)
    vecs = {row_key(r): vec(r, fdict, names, means, stds) for r in rows}
    items = []
    for r in rows:
        others = [(dist(vecs[row_key(r)], vecs[row_key(q)]), q) for q in rows if row_key(q) != row_key(r)]
        cats = sorted([(d, q) for d, q in others if q["h10_catastrophic_vs_h15"]], key=lambda z: z[0])
        poss = sorted([(d, q) for d, q in others if q["h10_beneficial_vs_h15"]], key=lambda z: z[0])
        nc = cats[0] if cats else (1e9, None); np = poss[0] if poss else (1e9, None)
        items.append({"id": row_key(r), "beneficial": bool(r["h10_beneficial_vs_h15"]), "catastrophic": bool(r["h10_catastrophic_vs_h15"]), "nearest_cat_dist": nc[0], "nearest_cat_id": row_key(nc[1]) if nc[1] else None, "nearest_pos_dist": np[0], "nearest_pos_id": row_key(np[1]) if np[1] else None, "phys_delta": sf(r["phys_delta_h10_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"]), "group": r.get("group"), "window": r.get("window")})
    return {"family": family, "feature_count": len(names), "positive_with_cat_closer_than_pos": sum(1 for x in items if x["beneficial"] and x["nearest_cat_dist"] <= x["nearest_pos_dist"]), "cat_with_pos_closer_than_cat": sum(1 for x in items if x["catastrophic"] and x["nearest_pos_dist"] <= x["nearest_cat_dist"]), "closest_positive_to_cat": sorted([x for x in items if x["beneficial"]], key=lambda x: x["nearest_cat_dist"])[:8], "closest_cat_to_positive": sorted([x for x in items if x["catastrophic"]], key=lambda x: x["nearest_pos_dist"])[:8]}


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v13 history-feature risk/value refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED deployable history-feature refit over opened banks; no MPC simulation, validation64, sealed test, or gradient training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` banks `{h['banks']}`; positives `{h['positive_rows']}`; catastrophic H10 rows `{h['catastrophic_rows']}`.",
        f"- H15-prefix trace history loaded for `{h['trace_loaded_rows']}/{h['rows']}` rows; feature counts: `{h['feature_counts_by_family']}`.",
        f"- Configs `{h['config_count']}`; global pass10 `{h['global_pass10_count']}`, pass5 `{h['global_pass5_count']}`; history-family pass5 `{h['history_global_pass5_count']}`.",
        f"- Best global config `{h['best_global_config']}` min save `{pct(h['best_global_min_save'])}`, avg save `{pct(h['best_global_avg_save'])}`, bad `{h['best_global_bad']}`, H10 `{h['best_global_h10']}`.",
        f"- Nested aggregate save `{pct(h['nested_save'])}`, bad `{h['nested_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 `{h['nested_h10']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.",
        f"- Nested selected history configs `{h['nested_selected_history_configs']}/{len(h['banks'])}` outer banks.",
        f"- Decision: {raw['decision']}",
        "",
        "## Nested outer-bank selection",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer save |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b, x in raw["nested_outer"].items():
        ev = x["outer_eval"]
        lines.append(f"| `{b}` | `{x['selected_config_id']}` | `{x['inner_summary']['all_pass5']}` | {x['inner_summary']['total_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Top global LOBO configs", "", "| rank | config | pass10 | pass5 | bad | phys fails | min save | avg save | H10 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_lobo"][:12], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['all_pass10']}` | `{s['all_pass5']}` | {s['total_bad']} | {s['phys_fail']} | {pct(s['min_save'])} | {pct(s['avg_save'])} | {s['total_h10']} |")
    lines += ["", "## Feature-space ambiguity", ""]
    for fam, a in raw["ambiguity"].items():
        lines.append(f"- `{fam}`: features `{a['feature_count']}`, positive-with-closer-cat `{a['positive_with_cat_closer_than_pos']}`, catastrophic-with-closer-positive `{a['cat_with_pos_closer_than_cat']}`.")
    lines += ["", "## Interpretation", "", "This is a bounded representation/refit diagnostic. Passing would justify only a small unused-source overhead-aware confirmation. Failure means H15-prefix history and engineered static features are still insufficient for safe deployment on opened data, strengthening the case for explicit risk/terminal-value learning or additional targeted boundary data before validation64."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v13 history-feature risk/value refit

UTC: {raw['created_utc']}. Development-only IMPROVED H15-prefix/history feature conservative support refit; no MPC simulation, no validation64, no sealed test, no gradient training. rows={h['rows']}; traces={h['trace_loaded_rows']}/{h['rows']}; global_pass5={h['global_pass5_count']}; nested_save={h['nested_save']:.4f}; nested_bad={h['nested_bad']}; nested_pass5={h['nested_pass5']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old_tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in old_tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_history_feature_support_refit,opened_banks_lobo_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows, input_hashes = load_rows()
    all_features, hist_diag, hist_hashes = enrich_history(rows)
    input_hashes.update(hist_hashes)
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    cfgs = grid()
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_history_feature_support_refit_no_simulation_no_validation_no_test",
        "hypothesis": "If H10 catastrophic states are distinguishable from safe/beneficial H10 states by short online-observable H15-prefix history features, a conservative support selector using v12 outcome-aligned labels should reduce catastrophic false positives while retaining >=5% decision-time saving under nested opened-bank evaluation.",
        "before_evidence": [
            "v12 aligned oracle over opened banks saves 30.8% decision time with physical gate, but nested pure-deploy static trees save 6.74% with 5 catastrophic H10 false positives.",
            "v10b deployable kNN/poly static refit had 8.43% nested saving but 2 catastrophic false positives; conservative global zero-bad configs collapsed below 5% saving.",
        ],
        "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11; validation64 and sealed test forbidden",
        "feature_policy": "primary candidates use instantaneous deployable observation/pose/progress plus optional H15-prefix observation/pose history; no scenario group/window IDs",
        "config_count": len(cfgs),
        "decision_gate": "advance only if nested aggregate has zero catastrophic H10 false positives, physical gate, and >=5% measured branch decision saving; strong if >=10%",
        "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations_upper_bound": len(cfgs) * (len(banks) + len(banks) * (len(banks)-1)) + len(banks), "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "input_hashes": input_hashes,
        "history_availability_preoutcome": hist_diag,
    }
    write_json(PROTOCOL, protocol)

    selector_evals = 0
    global_rows: List[Dict[str, Any]] = []
    for cfg in cfgs:
        hold = eval_lobo(rows, banks, all_features, cfg)
        selector_evals += len(hold)
        global_rows.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summarize(hold), "holdout_evaluations": hold})
    ranked = sorted(global_rows, key=lambda x: rank(x["summary"]))

    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    for outer in banks:
        candidates: List[Dict[str, Any]] = []
        for cfg in cfgs:
            hold = eval_lobo(rows, banks, all_features, cfg, blocked=[outer])
            selector_evals += len(hold)
            candidates.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summarize(hold)})
        selected = sorted(candidates, key=lambda x: rank(x["summary"]))[0]
        fdict = make_family_features(all_features, str(selected["config"]["family"]))
        tr = [r for r in rows if str(r["bank_id"]) != outer]
        ev = [r for r in rows if str(r["bank_id"]) == outer]
        outer_eval = eval_support(tr, ev, fdict, selected["config"])
        selector_evals += 1
        nested_outer[outer] = {"selected_config_id": selected["config_id"], "selected_config": selected["config"], "inner_summary": selected["summary"], "outer_eval": outer_eval}
        for d in outer_eval["details"]:
            nested_choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = int(d["selected_h"])
    nested_agg = v6.eval_choices(rows, nested_choices)

    fam_counts: Dict[str, int] = {}
    for fam in ["base_no_history", "history_no_risk", "history_with_risk"]:
        fam_counts[fam] = len(train_scaler(rows, make_family_features(all_features, fam))[0])
    ambiguity_report = {fam: ambiguity(rows, make_family_features(all_features, fam), fam) for fam in ["base_no_history", "history_no_risk", "history_with_risk"]}
    best = ranked[0]; bs = best["summary"]
    history_global_pass5 = sum(1 for x in global_rows if x["summary"]["all_pass5"] and str(x["config"]["family"]).startswith("history"))
    headline = {
        "rows": len(rows), "banks": banks,
        "positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]),
        "catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]),
        "trace_loaded_rows": hist_diag["trace_loaded_rows"],
        "feature_counts_by_family": fam_counts,
        "config_count": len(cfgs),
        "global_pass10_count": sum(1 for x in global_rows if x["summary"]["all_pass10"]),
        "global_pass5_count": sum(1 for x in global_rows if x["summary"]["all_pass5"]),
        "history_global_pass5_count": history_global_pass5,
        "best_global_config": best["config_id"],
        "best_global_min_save": bs["min_save"],
        "best_global_avg_save": bs["avg_save"],
        "best_global_bad": bs["total_bad"],
        "best_global_h10": bs["total_h10"],
        "nested_save": nested_agg["decision_relative_saving_vs_fixed_H15"],
        "nested_bad": len(nested_agg["catastrophic_false_positive_rows"]),
        "nested_physical_gate": bool(nested_agg["physical_gate"]),
        "nested_h10": int(nested_agg["chosen_counts"].get("10", 0)),
        "nested_pass5": bool(nested_agg["pass_5pct_no_cat_fp"]),
        "nested_pass10": bool(nested_agg["pass_10pct_no_cat_fp"]),
        "nested_selected_history_configs": sum(1 for x in nested_outer.values() if str(x["selected_config"]["family"]).startswith("history")),
    }
    if headline["nested_pass10"]:
        decision = "History-feature support selector passes strong opened-bank gate; next freeze a tiny unused-source overhead-aware confirmation with actual selector overhead, still no validation64/sealed test."
    elif headline["nested_pass5"]:
        decision = "History-feature support selector passes weak opened-bank gate; next freeze a small unused-source overhead-aware confirmation requiring zero catastrophic H10 before any validation64."
    elif history_global_pass5 > 0 and headline["nested_bad"] > 0:
        decision = "History features can fit some opened splits but nested model selection still leaks catastrophic H10; next acquire targeted boundary labels or train calibrated risk/terminal value with stronger uncertainty, not validation rollout."
    elif headline["trace_loaded_rows"] < headline["rows"]:
        decision = "History-feature input is incomplete, so this refit is not decisive; next repair/extract missing deployable history features or acquire targeted risk data before rollout."
    else:
        decision = "H15-prefix history/static engineered features still do not yield a safe deployable selector; next prioritize explicit risk/terminal-value learning/refit with uncertainty or more boundary acquisition, not another static-feature sweep or validation rollout."

    raw = {
        "created_utc": now().isoformat(),
        "classification": protocol["classification"],
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": selector_evals, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "history_availability": hist_diag,
        "headline": headline,
        "top_global_lobo": ranked[:25],
        "nested_outer": nested_outer,
        "nested_aggregate": nested_agg,
        "ambiguity": ambiguity_report,
        "input_hashes": {**input_hashes, rel(PROTOCOL): sha256(PROTOCOL), rel(SOURCE): sha256(SOURCE)},
        "decision": decision,
        "platform": {"python": sys.version, "platform": platform.platform()},
    }
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-history-refit-v13", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_history_refit_v13:
        raise ContractError("requires --run and explicit v13 development acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_HISTORY_FEATURE_REFIT_V13_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v13 history-feature refit before any further science", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "validation64_bank_opened": False, "sealed_test_accessed": False})
    done = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req), "headline": raw["headline"], "decision": raw["decision"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": raw["budget_actual"]["selector_refit_evaluations"], "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(req): sha256(req)}}
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v13 history-feature refit\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req)}, "current_backup_status": "not verified after v13; run backup before further scientific simulation/refit", "next_action": "After backup, follow v13 decision: if passed, freeze unused-source overhead-aware confirmation; otherwise move to explicit calibrated risk/terminal-value learning or targeted boundary acquisition. No validation64/sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
